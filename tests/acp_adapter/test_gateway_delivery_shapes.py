"""Every delivery shape the owner publishes reaches the editor exactly once, through one live pump."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent

COMMENTARY, FINAL = "Let me check the file. ", "FINAL-ANSWER-OMEGA"
SHAPES = {
    # The owner could not stream: the answer arrives only as message.interim, the completion is empty.
    "interim_only": [("message.interim", {"text": FINAL, "already_streamed": False}),
                     ("message.complete", {"outcome": "completed", "text": ""})],
    # A suppressed second delivery: the completion's text is present and null.
    "streamed_null_completion": [("message.delta", {"text": FINAL}),
                                 ("message.complete", {"outcome": "completed", "text": None})],
    # Commentary deltas close at the tool call; the final repeats only its own stream.
    "commentary_tool_final": [("message.delta", {"text": COMMENTARY}),
                              ("tool.start", {"tool_call_id": "t1", "tool_name": "terminal", "args": {}}),
                              ("tool.complete", {"tool_call_id": "t1", "tool_name": "terminal", "result": "ok"}),
                              ("message.delta", {"text": FINAL}),
                              ("message.complete", {"outcome": "completed", "text": FINAL})],
    # The owner reuses the reply streamed before a housekeeping tool as the final (``response_reused``).
    "reused_after_tool": [("message.delta", {"text": FINAL}),
                          ("tool.start", {"tool_call_id": "t1", "tool_name": "memory", "args": {}}),
                          ("tool.complete", {"tool_call_id": "t1", "tool_name": "memory", "result": "ok"}),
                          ("message.complete", {"outcome": "completed", "text": FINAL, "response_reused": True})],
}


class Gateway:
    def __init__(self):
        self.events = asyncio.Queue()
        self.rpc = AsyncMock()
        self.seq = 0

    def publish(self, admission_id, kind, payload):
        self.seq += 1
        self.events.put_nowait({"params": {"session_id": "s", "type": kind, "admission_id": admission_id,
                                           "replay_epoch": "e", "seq": self.seq, "payload": payload}})


def editor_text(conn):
    return "".join(call.kwargs["update"].content.text for call in conn.session_update.await_args_list
                   if getattr(getattr(call.kwargs["update"], "content", None), "text", None) is not None)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", sorted(SHAPES))
async def test_each_delivery_shape_reaches_the_editor_once_and_the_pump_survives(shape):
    agent = GatewayACPAgent()
    agent._gateway = gateway = Gateway()
    agent._conn = AsyncMock()
    agent._snapshots["s"] = {"replay_epoch": "e", "last_sequence": 0}
    agent._event_task = asyncio.create_task(agent._events())
    try:
        for admission_id, frames, prompt in (("a1", SHAPES[shape], "first"),
                                             ("a2", [("message.complete", {"outcome": "completed",
                                                                           "text": "SECOND"})], "second")):
            agent._conn.session_update.reset_mock()
            gateway.rpc.return_value = {"admission_id": admission_id}
            for kind, payload in frames:
                gateway.publish(admission_id, kind, payload)
            response = await asyncio.wait_for(agent.prompt([acp.text_block(prompt)], "s"), 5)
            assert response.stop_reason == "end_turn"
            assert agent._failure is None
            text = editor_text(agent._conn)
            if admission_id == "a1":
                assert text.count(FINAL) == 1, text
                assert text.count(COMMENTARY) == (shape == "commentary_tool_final"), text
            else:
                assert text == "SECOND"
    finally:
        agent._event_task.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["streamed", "interim", "unstreamed"])
async def test_acp_gateway_settlement_keeps_thoughts_separate_and_final_once(monkeypatch, delivery):
    """Use the owner's real reasoning formatter and completion payload, not a hand-cleaned final."""
    from gateway import run, session_setting_commands
    from gateway.config import Platform
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin
    from gateway.session_settlement_recovery import completion_payload

    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"display": {"show_reasoning": True}})
    monkeypatch.setattr(session_setting_commands, "local_policy", lambda *_: SimpleNamespace(source="acp"))
    owner = GatewayTurnPersistenceMixin()
    source = SimpleNamespace(platform=Platform.LOCAL)
    reasoning = "**Comparing metadata records**"
    agent = GatewayACPAgent()
    agent._gateway = gateway = Gateway()
    agent._conn = AsyncMock()
    agent._snapshots["s"] = {"replay_epoch": "e", "last_sequence": 0}
    agent._event_task = asyncio.create_task(agent._events())
    try:
        # Equal final text in distinct admissions is legitimate, not a duplicate to discard.
        for aid in ("a1", "a2"):
            agent._conn.session_update.reset_mock()
            gateway.rpc.return_value = {"admission_id": aid}
            gateway.publish(aid, "reasoning.delta", {"text": reasoning})
            gateway.publish(aid, "message.delta", {"text": COMMENTARY})
            gateway.publish(aid, "tool.start", {"tool_call_id": aid, "tool_name": "terminal", "args": {}})
            gateway.publish(aid, "tool.complete", {"tool_call_id": aid, "tool_name": "terminal", "result": "ok"})
            if delivery == "streamed":
                gateway.publish(aid, "message.delta", {"text": FINAL})
            elif delivery == "interim":
                gateway.publish(aid, "message.interim", {"text": FINAL, "already_streamed": False})
            response = owner._hmwa_prepend_reasoning({"last_reasoning": reasoning}, FINAL, source, False)
            complete = completion_payload({"admission_id": aid, "request_id": aid},
                                          {"outcome": "completed"}, response,
                                          {"result": {"last_reasoning": reasoning, "local_viewer": True}})
            gateway.publish(aid, "message.complete", complete)
            result = await asyncio.wait_for(agent.prompt([acp.text_block("check")], "s"), 5)
            assert result.stop_reason == "end_turn"
            updates = [call.kwargs["update"] for call in agent._conn.session_update.await_args_list]
            reply = "".join(u.content.text for u in updates if u.session_update == "agent_message_chunk")
            thoughts = "".join(u.content.text for u in updates if u.session_update == "agent_thought_chunk")
            assert reply.count(FINAL) == 1, reply
            assert reply.count(COMMENTARY) == 1, reply
            assert reasoning not in reply
            assert thoughts == reasoning
            assert complete["reasoning"] == reasoning
            assert agent._failure is None
    finally:
        agent._event_task.cancel()
        await asyncio.gather(agent._event_task, return_exceptions=True)
