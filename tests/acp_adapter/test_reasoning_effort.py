"""BB/ACP effort selections must affect requests, not just be acknowledged."""

import json
from types import SimpleNamespace

import acp
import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from agent.transports import get_transport
from hermes_constants import parse_reasoning_effort
from hermes_state import SessionDB


@pytest.fixture
def server(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "sessions.db")
    manager = SessionManager(
        agent_factory=lambda: SimpleNamespace(
            model="gpt-6-sol", provider="custom", base_url="https://example.invalid/v1",
            api_mode="codex_responses", reasoning_config={"enabled": True, "effort": "xhigh"},
        ),
        db=db,
    )
    monkeypatch.setattr(manager, "_schedule_git_metadata", lambda *args: None)
    agent = HermesACPAgent(manager)
    monkeypatch.setattr(agent, "_build_model_state", lambda state: None)
    monkeypatch.setattr(agent, "_schedule_mcp_late_refresh", lambda state: None)
    yield agent
    db.close()


def reasoning_option(response):
    return next(option for option in response.config_options if option.id == "reasoning_effort")


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
async def test_setter_changes_live_agent_and_outgoing_request(server, effort):
    state = server.session_manager.create_session()
    response = await server.set_config_option("reasoning_effort", state.session_id, effort)
    expected = parse_reasoning_effort(effort)
    assert state.agent.reasoning_config == expected
    assert reasoning_option(response).current_value == effort

    transport = get_transport("codex_responses")
    request = transport.build_kwargs(
        model=state.model, provider=state.agent.provider, base_url=state.agent.base_url,
        messages=[{"role": "user", "content": "Hello"}],
        reasoning_config=state.agent.reasoning_config,
    )
    assert request["reasoning"]["effort"] == effort


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["", "typo", "show"])
async def test_invalid_effort_does_not_mutate_session(server, effort):
    state = server.session_manager.create_session()
    with pytest.raises(acp.RequestError) as error:
        await server.set_config_option("reasoning_effort", state.session_id, effort)
    assert error.value.code == -32602
    assert state.agent.reasoning_config == {"enabled": True, "effort": "xhigh"}
    assert state.reasoning_effort_override is None


@pytest.mark.asyncio
async def test_missing_session_is_a_protocol_error(server):
    with pytest.raises(acp.RequestError) as error:
        await server.set_config_option("reasoning_effort", "missing", "low")
    assert error.value.code == -32002


@pytest.mark.asyncio
async def test_effort_is_session_scoped_and_between_turns(server):
    first = server.session_manager.create_session()
    second = server.session_manager.create_session()
    first.is_running = True
    response = await server.set_config_option("reasoning_effort", first.session_id, "low")
    assert reasoning_option(response).current_value == "low"
    assert first.agent.reasoning_config["effort"] == "xhigh"
    assert second.agent.reasoning_config["effort"] == "xhigh"

    first.is_running = False
    assert server._claim_turn_or_queue(first, first.session_id, "Hello", "Hello", True) is None
    assert first.agent.reasoning_config == {"enabled": True, "effort": "low"}
    assert second.agent.reasoning_config["effort"] == "xhigh"


@pytest.mark.asyncio
async def test_override_survives_persistence_resume_and_fork(server, tmp_path):
    manager = server.session_manager
    state = manager.create_session(cwd=str(tmp_path))
    state.history = [{"role": "user", "content": "Keep this history"}]
    await server.set_config_option("reasoning_effort", state.session_id, "none")
    row = manager._get_db().get_session(state.session_id)
    assert json.loads(row["model_config"])["acp_reasoning_effort"] == "none"

    manager._sessions.clear()
    restored = manager.get_session(state.session_id)
    assert restored.reasoning_effort_override == "none"
    assert restored.agent.reasoning_config == {"enabled": False}
    assert [(message["role"], message["content"]) for message in restored.history] == [
        ("user", "Keep this history"),
    ]
    assert restored.history[0]["message_uid"] == state.history[0]["message_uid"]
    forked = manager.fork_session(state.session_id, cwd=str(tmp_path))
    assert forked.reasoning_effort_override == "none"
    assert forked.agent.reasoning_config == {"enabled": False}

    for response in (
        await server.load_session(str(tmp_path), restored.session_id),
        await server.resume_session(str(tmp_path), restored.session_id),
        await server.fork_session(str(tmp_path), restored.session_id),
    ):
        assert reasoning_option(response).current_value == "none"


@pytest.mark.asyncio
async def test_override_survives_model_rebuild(server, monkeypatch):
    state = server.session_manager.create_session()
    await server.set_config_option("reasoning_effort", state.session_id, "low")
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kwargs: SimpleNamespace(success=True, target_provider="custom", new_model="other"),
    )
    server._switch_model(state, "other")
    assert state.reasoning_effort_override == "low"
    assert state.agent.reasoning_config == {"enabled": True, "effort": "low"}


@pytest.mark.asyncio
async def test_new_session_advertises_resolved_default_without_pinning_it(server):
    response = await server.new_session(cwd=".")
    state = server.session_manager.get_session(response.session_id)
    assert reasoning_option(response).current_value == "xhigh"
    assert state.reasoning_effort_override is None
