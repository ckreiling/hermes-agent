"""Offline Responses -> real ACP callback/worker-loop delivery regressions."""
import asyncio
import threading
from types import SimpleNamespace
from typing import Any, Callable, cast
from unittest.mock import AsyncMock

import pytest
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionState
from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX
from agent.codex_runtime import _consume_codex_event_stream, make_codex_app_server_event_bridge, run_codex_stream
from agent.stream_delivery import StreamDeliveryMixin

PROGRESS = "Approve [this request](https://approval.example.test/request/fixture)."
FINAL = "The requested check completed."


class DeliveryAgent(StreamDeliveryMixin):
    """Only provider infrastructure is fake; delivery and Responses assembly are real."""
    tool_progress_callback: Callable[..., Any]
    stream_delta_callback: Any
    interim_assistant_callback: Any

    def __init__(self):
        self.session_id = "fixture"
        self.model = "fixture-model"
        self.provider = "fixture"
        self.platform = "acp"
        self.show_commentary = True
        self.interim_assistant_callback = None
        self.stream_delta_callback = self._stream_callback = self.reasoning_callback = None
        self._interrupt_requested = False
        self._stream_reasoning_hooks_enabled = False

    @staticmethod
    def _strip_think_blocks(text):
        return text

    def _touch_activity(self, *args):
        pass

    def _client_log_context(self):
        return "offline fixture"

    def _enqueue_stream_hook(self, *args, **kwargs):
        pass


def message(text, phase="commentary", ident="progress"):
    return {"id": ident, "type": "message", "role": "assistant", "phase": phase,
            "content": [{"type": "output_text", "text": text}]}


def stream_events(items):
    for item in items:
        yield {"type": "response.output_item.added", "item": item}
        if item["type"] == "message":
            text = item["content"][0]["text"]
            for delta in [text[:12], text[12:]]:
                yield {"type": "response.output_text.delta", "delta": delta}
        yield {"type": "response.output_item.done", "item": item}
    yield {"type": "response.completed", "response": {"status": "completed"}}


class RecordingClient:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update):
        self.updates.append(update.model_dump(by_alias=True, exclude_none=True))

    async def request_permission(self, **kwargs):
        raise AssertionError("no real approval requested")


def visible(client):
    return [u for u in client.updates if u["sessionUpdate"] == "agent_message_chunk"]


async def wired():
    agent = DeliveryAgent()
    state = SessionState(session_id="fixture", agent=agent)
    # Explicit legacy-direct unit coverage; default-route proof lives in test_gateway_fork_wire.
    server = object.__new__(HermesACPAgent)
    server.session_manager = cast(Any, SimpleNamespace(save_session=lambda sid: None))
    server._send_usage_update = AsyncMock()
    client = RecordingClient()
    cbs = server._wire_turn_callbacks(state, "fixture", client, asyncio.get_running_loop())
    return agent, state, server, client, cbs


async def finish(server, state, client, cbs, text=FINAL, **extra):
    await server._finish_turn(
        state, "fixture", client, {"final_response": text, **extra}, "fixture", cbs.streamed, cbs.interim_text,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("show_commentary", [True, False])
@pytest.mark.parametrize("live", [True, False])
async def test_responses_commentary_reaches_acp_before_blocking_tool(show_commentary, live):
    agent, state, server, client, cbs = await wired()
    agent.show_commentary = show_commentary
    items = [message("Private scratchpad", "analysis", "analysis"), message(PROGRESS),
             message("Second progress update.", ident="progress-2")]
    sdk = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: iter(stream_events(items))))
    expected = ([PROGRESS, "Second progress update."] if live else
                [PROGRESS + "\n\nSecond progress update."]) if show_commentary else []

    def worker():
        response = (run_codex_stream(agent, {"model": agent.model}, client=sdk) if live else
                    _consume_codex_event_stream(stream_events(items), model=agent.model))
        assert response is not None
        # Post-response assembly/re-entry must not replay live-delivered commentary.
        envelope = {"codex_message_items": response.output}
        agent._emit_interim_assistant_message(envelope)
        agent._emit_interim_assistant_message(envelope)
        agent.stream_delta_callback(None)
        agent.tool_progress_callback("tool.started", "terminal", None, {"command": "fixture-only"})
        # This is the point a real tool may block: updates must already be on the client.
        assert [u["content"]["text"] for u in visible(client)] == expected, client.updates
        agent.tool_progress_callback("tool.completed", "terminal", result="ok")

    await asyncio.to_thread(worker)
    await finish(server, state, client, cbs)
    chunks = visible(client)
    assert [u["content"]["text"] for u in chunks] == expected + [FINAL]
    assert len({u["messageId"] for u in chunks}) == len(chunks)
    thoughts = [u["content"]["text"] for u in client.updates if u["sessionUpdate"] == "agent_thought_chunk"]
    if live:
        assert "Private scratchpad" == "".join(thoughts[:2])
    assert not any("Private scratchpad" in u["content"]["text"] for u in chunks)
    if show_commentary:
        assert not any("approval.example.test" in text for text in thoughts)
        assert client.updates.index(chunks[0]) < next(
            i for i, u in enumerate(client.updates) if u["sessionUpdate"] == "tool_call")


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
async def test_non_codex_interim_dedup_and_unstreamed_final(streamed):
    agent, state, server, client, cbs = await wired()

    def worker():
        if streamed:
            agent._fire_stream_delta(PROGRESS)
        envelope = {"role": "assistant", "content": PROGRESS}
        agent._emit_interim_assistant_message(envelope)
        agent._emit_interim_assistant_message(envelope)
        agent.stream_delta_callback(None)
        agent.tool_progress_callback("tool.started", "terminal", None, {"command": "fixture-only"})
        assert [u["content"]["text"] for u in visible(client)] == [PROGRESS]
        agent.tool_progress_callback("tool.completed", "terminal", result="ok")

    await asyncio.to_thread(worker)
    await finish(server, state, client, cbs)
    chunks = visible(client)
    assert [u["content"]["text"] for u in chunks] == [PROGRESS, FINAL]
    assert chunks[0]["messageId"] != chunks[1]["messageId"]


@pytest.mark.asyncio
async def test_streamed_final_not_repeated_by_finish():
    agent, state, server, client, cbs = await wired()
    await asyncio.to_thread(agent._fire_stream_delta, FINAL)
    await finish(server, state, client, cbs)
    assert [u["content"]["text"] for u in visible(client)] == [FINAL]


@pytest.mark.asyncio
async def test_app_server_completed_final_and_finish_do_not_duplicate():
    agent, state, server, client, cbs = await wired()
    bridge = make_codex_app_server_event_bridge(agent)

    def worker():
        for ident, text in [("progress", PROGRESS), ("final", FINAL)]:
            bridge({"method": "item/agentMessage/delta", "params": {"itemId": ident, "delta": text}})
            bridge({"method": "item/completed", "params": {
                "item": {"id": ident, "type": "agentMessage", "text": text}}})

    await asyncio.to_thread(worker)
    await finish(server, state, client, cbs)
    chunks = visible(client)
    assert [u["content"]["text"] for u in chunks] == [PROGRESS, FINAL]
    assert chunks[0]["messageId"] != chunks[1]["messageId"]


@pytest.mark.asyncio
async def test_interrupt_wait_status_is_not_sent_as_final_prose():
    _, state, server, client, cbs = await wired()
    await finish(server, state, client, cbs, text=INTERRUPT_WAITING_FOR_MODEL_PREFIX + "fixture", interrupted=True)
    assert visible(client) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("interim_final", [False, True])
async def test_prompt_delivers_progress_while_the_tool_is_blocked(interim_final):
    agent, state, server, client, _ = await wired()
    server._conn = cast(Any, client)
    server.session_manager = cast(Any, SimpleNamespace(get_session=lambda sid: state))
    started = asyncio.Event()
    release = threading.Event()
    original_update = client.session_update

    async def record(session_id, update):
        await original_update(session_id, update)
        if update.session_update == "tool_call":
            started.set()

    client.session_update = record

    def conversation(**kwargs):
        agent._fire_streamed_codex_commentary(PROGRESS)
        agent.tool_progress_callback("tool.started", "terminal", None, {"command": "fixture-only"})
        assert release.wait(timeout=5), "test did not unblock fixture tool"
        agent.tool_progress_callback("tool.completed", "terminal", result="ok")
        if interim_final:
            agent._emit_interim_assistant_message({"content": FINAL})
        return {"final_response": FINAL}

    cast(Any, agent).run_conversation = conversation
    task = asyncio.create_task(server.prompt(
        prompt=[TextContentBlock(type="text", text="Run the fixture check")], session_id="fixture",
    ))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        assert not task.done()
        assert [u["content"]["text"] for u in visible(client)] == [PROGRESS]
    finally:
        release.set()
    result = await asyncio.wait_for(task, timeout=5)
    assert result.stop_reason == "end_turn"
    chunks = visible(client)
    assert [u["content"]["text"] for u in chunks] == [PROGRESS, FINAL]
    assert chunks[0]["messageId"] != chunks[1]["messageId"]
    assert state.is_running is False


@pytest.mark.asyncio
async def test_transformed_streamed_final_reuses_its_message_identity():
    agent, state, server, client, cbs = await wired()
    await asyncio.to_thread(agent._fire_stream_delta, FINAL)
    await finish(server, state, client, cbs, text="Rewritten final.", response_transformed=True)
    chunks = visible(client)
    assert [u["content"]["text"] for u in chunks] == [FINAL, "Rewritten final."]
    assert chunks[0]["messageId"] == chunks[1]["messageId"]
