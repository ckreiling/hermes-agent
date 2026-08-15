"""Regression coverage for interrupted Hermes ACP tool turns."""

from unittest.mock import AsyncMock, MagicMock

import acp
import pytest
from acp.schema import AgentMessageChunk, TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


@pytest.mark.asyncio
async def test_interrupted_tool_turn_with_null_response_releases_session():
    """A null interrupt response must not wedge future prompts in the queue."""
    runtime = MagicMock(name="MockAIAgent")
    runtime.model = "test-model"
    runtime.provider = "openrouter"
    runtime.run_conversation.side_effect = [
        {
            "final_response": None,
            "messages": [
                {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
                {"role": "tool", "content": "[Command interrupted]"},
                {"role": "assistant", "content": "Operation interrupted."},
            ],
            "interrupted": True,
        },
        {
            "final_response": "continued",
            "messages": [
                {"role": "user", "content": "Continue"},
                {"role": "assistant", "content": "continued"},
            ],
            "interrupted": False,
        },
    ]

    manager = SessionManager(agent_factory=lambda: runtime)
    manager._persist = lambda state: None
    state = manager.create_session(cwd=".")
    agent = HermesACPAgent(session_manager=manager)

    mock_conn = MagicMock(spec=acp.Client)
    mock_conn.session_update = AsyncMock()
    agent._conn = mock_conn

    first = await agent.prompt(
        prompt=[TextContentBlock(type="text", text="first")],
        session_id=state.session_id,
    )

    assert first.stop_reason == "end_turn"
    assert state.is_running is False
    assert state.queued_prompts == []

    second = await agent.prompt(
        prompt=[TextContentBlock(type="text", text="Continue")],
        session_id=state.session_id,
    )

    assert second.stop_reason == "end_turn"
    assert state.is_running is False
    assert state.queued_prompts == []
    assert runtime.run_conversation.call_count == 2

    emitted_text = []
    for call in mock_conn.session_update.await_args_list:
        update = call.kwargs.get("update")
        if update is None and len(call.args) > 1:
            update = call.args[1]
        if isinstance(update, AgentMessageChunk):
            emitted_text.append(update.content.text)
    assert emitted_text == ["continued"]
