"""Title-only input must not change history, eligibility, or fallback behavior."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.turn_context import _maybe_title_session_at_turn_start


@pytest.mark.parametrize("override, content, expected", [
    (None, "Ordinary request", "Ordinary request"),
    ("Actual question", "Injected model context", "Actual question"),
    ("  Actual question  ", "Injected model context", "Actual question"),
    ("", "Injected model context", None),
    ("   ", "Injected model context", None),
    (None, [{"type": "text", "text": "Image caption"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}], "Image caption"),
    (None, [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}], None),
])
def test_title_input_override_is_optional_and_history_is_unchanged(monkeypatch, override, content, expected):
    agent = SimpleNamespace(_session_db=Mock(), session_id="test", _session_db_created=True)
    messages = [{"role": "user", "content": content}]
    original = deepcopy(messages)
    title = Mock()
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", title)

    _maybe_title_session_at_turn_start(agent, messages, override)

    assert messages == original
    if expected is None:
        title.assert_not_called()
    else:
        assert title.call_args.args[2] == expected
        assert title.call_args.kwargs["conversation_history"] is messages


@pytest.mark.parametrize("override, expected", [(None, "attachment"), ("Original question", "Original question")])
def test_title_override_preserves_upstream_paste_preview(monkeypatch, override, expected):
    agent = SimpleNamespace(_session_db=Mock(), session_id="test", _session_db_created=True)
    messages = [{"role": "user", "content": "attachment",
                 "display_metadata": {"title_preview": "pasted topic"}}]
    original = deepcopy(messages)
    title = Mock()
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", title)
    _maybe_title_session_at_turn_start(agent, messages, override)
    assert title.call_args.args[2] == expected
    assert title.call_args.kwargs["title_preview"] == "pasted topic"
    assert title.call_args.kwargs["conversation_history"] is messages
    assert messages == original


@pytest.mark.parametrize("text, titleable", [("/help", True), ("[CONTEXT COMPACTION] synthetic handoff", False)])
def test_title_override_preserves_existing_control_message_eligibility(monkeypatch, text, titleable):
    agent = SimpleNamespace(_session_db=Mock(), session_id="test", _session_db_created=True)
    start = Mock()
    monkeypatch.setattr("agent.memory_provider.spawn_context_thread", start)

    _maybe_title_session_at_turn_start(agent, [{"role": "user", "content": "injected context"}], text)

    assert start.called is titleable


@pytest.mark.parametrize("platform", ["cron", "subagent", "CRON"])
def test_title_override_cannot_enable_titling_for_machine_runs(monkeypatch, platform):
    agent = SimpleNamespace(_session_db=Mock(), session_id="test", platform=platform)
    title = Mock()
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", title)

    _maybe_title_session_at_turn_start(agent, [], "Actual question")

    title.assert_not_called()


@pytest.mark.parametrize("platform", ["acp", "ACP"])
@pytest.mark.parametrize("db_created", [False, True])
def test_bb_owns_acp_thread_titles_without_native_title_work(monkeypatch, platform, db_created):
    monkeypatch.setenv("BB_THREAD_ID", "thr_title_regression")
    agent = SimpleNamespace(
        platform=platform, _session_db=Mock(), session_id="test",
        _session_db_created=db_created, _ensure_db_session=Mock(), _on_session_title=Mock(),
    )
    title = Mock()
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", title)

    _maybe_title_session_at_turn_start(
        agent, [{"role": "user", "content": "<system_instructions>\nInjected context"}],
        "Actual question",
    )

    title.assert_not_called()
    agent._ensure_db_session.assert_not_called()
    assert not agent._session_db.mock_calls
    agent._on_session_title.assert_not_called()
    assert not hasattr(agent, "_deferred_title_upgrade")


@pytest.mark.parametrize("platform, bb_thread_id", [
    ("acp", None), ("acp", ""), ("acp", "   "),
    ("cli", "thr_parent"), ("desktop", "thr_parent"), ("telegram", "thr_parent"),
])
def test_bb_title_ownership_does_not_disable_other_surfaces(monkeypatch, platform, bb_thread_id):
    if bb_thread_id is None:
        monkeypatch.delenv("BB_THREAD_ID", raising=False)
    else:
        monkeypatch.setenv("BB_THREAD_ID", bb_thread_id)
    agent = SimpleNamespace(
        platform=platform, _session_db=Mock(), session_id="test", _session_db_created=True,
    )
    title = Mock()
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", title)

    _maybe_title_session_at_turn_start(agent, [{"role": "user", "content": "Actual question"}])

    title.assert_called_once()
    assert title.call_args.args[2] == "Actual question"
