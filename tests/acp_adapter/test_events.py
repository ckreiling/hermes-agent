"""Tests for acp_adapter.events — callback factories for ACP notifications."""

import asyncio
import gc
import json
import uuid
import warnings
from collections import deque
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp.schema import AgentPlanUpdate, ContentToolCallContent, FileEditToolCallContent

from acp_adapter.events import (
    _build_plan_update_from_todo_result,
    _send_update,
    make_message_cb,
    make_step_cb,
    make_thinking_cb,
    make_tool_progress_cb,
)


@pytest.fixture()
def mock_conn():
    """Mock ACP Client connection."""
    conn = MagicMock(spec=acp.Client)
    conn.session_update = AsyncMock()
    return conn


@pytest.fixture()
def event_loop_fixture():
    """Create a real event loop for testing threadsafe coroutine submission."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Tool progress callback
# ---------------------------------------------------------------------------


class TestToolProgressCallback:
    def test_emits_tool_call_start(self, mock_conn, event_loop_fixture):
        """Tool progress should emit a ToolCallStart update."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        # Run callback in the event loop context
        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.started", "terminal", "$ ls -la", {"command": "ls -la"})

        # Should have tracked the tool call ID
        assert "terminal" in tool_call_ids

        # Should have called run_coroutine_threadsafe
        mock_rcts.assert_called_once()
        coro = mock_rcts.call_args[0][0]
        # The coroutine should be conn.session_update
        assert mock_conn.session_update.called or coro is not None



    def test_duplicate_same_name_tool_calls_use_fifo_ids(self, mock_conn, event_loop_fixture):
        """Multiple same-name tool calls should be tracked independently in order."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        progress_cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
        step_cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            progress_cb("tool.started", "terminal", "$ ls", {"command": "ls"})
            progress_cb("tool.started", "terminal", "$ pwd", {"command": "pwd"})
            assert len(tool_call_ids["terminal"]) == 2

            step_cb(1, [{"name": "terminal", "result": "ok-1"}])
            assert len(tool_call_ids["terminal"]) == 1

            step_cb(2, [{"name": "terminal", "result": "ok-2"}])
            assert "terminal" not in tool_call_ids


# ---------------------------------------------------------------------------
# Thinking callback
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Step callback
# ---------------------------------------------------------------------------


class TestStepCallback:
    def test_completes_tracked_tool_calls(self, mock_conn, event_loop_fixture):
        """Step callback should mark tracked tools as completed."""
        tool_call_ids = {"terminal": "tc-abc123"}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(1, [{"name": "terminal", "result": "success"}])

        # Tool should have been removed from tracking
        assert "terminal" not in tool_call_ids
        mock_rcts.assert_called_once()



    @pytest.mark.parametrize("raw, expected", [("", ""), (0, "0"), (False, "False")])
    def test_falsey_result_reaches_client_unchanged(self, mock_conn, event_loop_fixture, raw, expected):
        """A present-but-falsey ``result`` is the tool's real output, not a missing key (#10845)."""
        from collections import deque

        cb = make_step_cb(mock_conn, "session-1", event_loop_fixture, {"terminal": deque(["tc-f"])}, {})
        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            mock_rcts.return_value = MagicMock(spec=Future)
            cb(1, [{"name": "terminal", "result": raw}])
        mock_btc.assert_called_once_with("tc-f", "terminal", result=expected, function_args=None, snapshot=None)

    def test_result_passed_to_build_tool_complete(self, mock_conn, event_loop_fixture):
        """Tool result from prev_tools dict is forwarded to build_tool_complete."""
        from collections import deque

        tool_call_ids = {"terminal": deque(["tc-xyz789"])}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            # Provide a result string in the tool info dict
            cb(1, [{"name": "terminal", "result": '{"output": "hello"}'}])

        mock_btc.assert_called_once_with(
            "tc-xyz789", "terminal", result='{"output": "hello"}', function_args=None, snapshot=None
        )



    @pytest.mark.parametrize("completion_args", ["not-json", '["not", "an", "object"]'])
    def test_completion_argument_fallback_uses_captured_start_metadata(
        self, mock_conn, event_loop_fixture, completion_args
    ):
        tool_call_ids = {"write_file": deque(["tc-fallback"])}
        captured_args = {"path": "captured.txt", "content": "after\n"}
        snapshot = object()
        tool_call_meta = {
            "tc-fallback": {"args": captured_args, "snapshot": snapshot}
        }
        cb = make_step_cb(
            mock_conn,
            "session-1",
            event_loop_fixture,
            tool_call_ids,
            tool_call_meta,
        )

        with patch("acp_adapter.events._send_update"), \
             patch("acp_adapter.events.build_tool_complete") as mock_complete:
            cb(1, [{
                "name": "write_file",
                "result": '{"bytes_written": 6}',
                "arguments": completion_args,
            }])

        mock_complete.assert_called_once_with(
            "tc-fallback",
            "write_file",
            result='{"bytes_written": 6}',
            function_args=captured_args,
            snapshot=snapshot,
        )

    def test_tool_progress_captures_snapshot_metadata(self, mock_conn, event_loop_fixture):
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        with patch("acp_adapter.events.make_tool_call_id", return_value="tc-meta"), \
             patch("acp_adapter.events._send_update") as mock_send, \
             patch("agent.display.capture_local_edit_snapshot", return_value="snapshot") as mock_capture:
            cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
            cb("tool.started", "write_file", None, {"path": "diff-test.txt", "content": "hello"})

        assert list(tool_call_ids["write_file"]) == ["tc-meta"]
        assert tool_call_meta["tc-meta"] == {
            "args": {"path": "diff-test.txt", "content": "hello"},
            "snapshot": "snapshot",
        }
        mock_capture.assert_called_once_with(
            "write_file",
            {"path": "diff-test.txt", "content": "hello"},
            task_id="session-1",
        )
        mock_send.assert_called_once()

    def test_todo_completion_emits_native_plan_update_after_tool_completion(self, mock_conn, event_loop_fixture):
        from collections import deque

        tool_call_ids = {"todo": deque(["tc-todo"])}
        loop = event_loop_fixture
        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})
        todo_result = (
            '{"todos":['
            '{"id":"inspect","content":"Inspect ACP","status":"completed"},'
            '{"id":"patch","content":"Patch renderer","status":"in_progress"},'
            '{"id":"old","content":"Drop stale task","status":"cancelled"}'
            '],"summary":{"total":3}}'
        )

        with patch("acp_adapter.events._send_update") as mock_send:
            cb(1, [{"name": "todo", "result": todo_result}])

        updates = [call.args[3] for call in mock_send.call_args_list]
        assert [getattr(update, "session_update", None) for update in updates] == [
            "tool_call_update",
            "plan",
        ]
        plan = updates[1]
        assert isinstance(plan, AgentPlanUpdate)
        assert [entry.content for entry in plan.entries] == [
            "Inspect ACP",
            "Patch renderer",
            "[cancelled] Drop stale task",
        ]
        assert [entry.status for entry in plan.entries] == ["completed", "in_progress", "completed"]
        assert [entry.priority for entry in plan.entries] == ["medium", "medium", "medium"]

    def test_actual_write_file_callback_emits_empty_new_file_diff(
        self, mock_conn, event_loop_fixture, tmp_path
    ):
        from model_tools import handle_function_call

        session_id = "acp-write-integration"
        target = tmp_path / "empty.txt"
        args = {"path": str(target), "content": ""}
        tool_call_ids = {}
        tool_call_meta = {}
        sent = []
        progress_cb = make_tool_progress_cb(
            mock_conn, session_id, event_loop_fixture, tool_call_ids, tool_call_meta
        )
        step_cb = make_step_cb(
            mock_conn, session_id, event_loop_fixture, tool_call_ids, tool_call_meta
        )

        with patch("acp_adapter.events._send_update", side_effect=lambda *call: sent.append(call[3])):
            progress_cb("tool.started", "write_file", None, args)
            raw_result = handle_function_call(
                "write_file", args, task_id=session_id
            )
            step_cb(1, [{
                "name": "write_file",
                "result": raw_result,
                "arguments": json.dumps(args),
            }])

        assert target.exists()
        assert target.read_text(encoding="utf-8") == ""
        completion = next(update for update in sent if update.session_update == "tool_call_update")
        diffs = [item for item in completion.content if isinstance(item, FileEditToolCallContent)]
        summaries = [item for item in completion.content if isinstance(item, ContentToolCallContent)]
        assert completion.status == "completed"
        assert len(diffs) == 1
        assert diffs[0].path == str(target)
        assert diffs[0].old_text is None
        assert diffs[0].new_text == ""
        assert any("write_file completed" in item.content.text for item in summaries)

    def test_actual_patch_replace_callback_emits_complete_file_text(
        self, mock_conn, event_loop_fixture, tmp_path
    ):
        from model_tools import handle_function_call

        session_id = "acp-replace-integration"
        target = tmp_path / "replace.txt"
        original = "first\nkeep\nold\nlast\n"
        expected = "first\nkeep\nnew\nlast\n"
        target.write_text(original, encoding="utf-8")
        args = {
            "mode": "replace",
            "path": str(target),
            "old_string": "old",
            "new_string": "new",
        }
        tool_call_ids = {}
        tool_call_meta = {}
        sent = []
        progress_cb = make_tool_progress_cb(
            mock_conn, session_id, event_loop_fixture, tool_call_ids, tool_call_meta
        )
        step_cb = make_step_cb(
            mock_conn, session_id, event_loop_fixture, tool_call_ids, tool_call_meta
        )

        with patch("acp_adapter.events._send_update", side_effect=lambda *call: sent.append(call[3])):
            progress_cb("tool.started", "patch", None, args)
            raw_result = handle_function_call("patch", args, task_id=session_id)
            step_cb(1, [{
                "name": "patch",
                "result": raw_result,
                "arguments": json.dumps(args),
            }])

        assert json.loads(raw_result)["success"] is True
        assert target.read_text(encoding="utf-8") == expected
        completion = next(update for update in sent if update.session_update == "tool_call_update")
        diff = next(item for item in completion.content if isinstance(item, FileEditToolCallContent))
        assert diff.path == str(target)
        assert diff.old_text == original
        assert diff.new_text == expected

    def test_actual_v4a_callback_emits_full_multifile_and_deletion_states(
        self, mock_conn, event_loop_fixture, tmp_path
    ):
        from model_tools import handle_function_call

        session_id = "acp-v4a-integration"
        updated = tmp_path / "updated.txt"
        created = tmp_path / "created.txt"
        deleted = tmp_path / "deleted.txt"
        updated.write_text("alpha\nkeep\n", encoding="utf-8")
        deleted.write_text("remove me\n", encoding="utf-8")
        patch_body = (
            "*** Begin Patch\n"
            f"*** Update File: {updated}\n"
            "@@\n"
            "-alpha\n"
            "+beta\n"
            f"*** Add File: {created}\n"
            "+created content\n"
            f"*** Delete File: {deleted}\n"
            "*** End Patch"
        )
        args = {"mode": "patch", "patch": patch_body}
        tool_call_ids = {}
        tool_call_meta = {}
        sent = []
        progress_cb = make_tool_progress_cb(
            mock_conn, session_id, event_loop_fixture, tool_call_ids, tool_call_meta
        )
        step_cb = make_step_cb(
            mock_conn, session_id, event_loop_fixture, tool_call_ids, tool_call_meta
        )

        with patch("acp_adapter.events._send_update", side_effect=lambda *call: sent.append(call[3])):
            progress_cb("tool.started", "patch", None, args)
            raw_result = handle_function_call("patch", args, task_id=session_id)
            step_cb(1, [{
                "name": "patch",
                "result": raw_result,
                "arguments": json.dumps(args),
            }])

        assert json.loads(raw_result)["success"] is True
        completion = next(update for update in sent if update.session_update == "tool_call_update")
        diffs = {
            item.path: item
            for item in completion.content
            if isinstance(item, FileEditToolCallContent)
        }
        assert diffs[str(updated)].old_text == "alpha\nkeep\n"
        assert diffs[str(updated)].new_text == "beta\nkeep\n"
        assert diffs[str(created)].old_text is None
        assert diffs[str(created)].new_text == "created content"
        assert diffs[str(deleted)].old_text == "remove me\n"
        assert diffs[str(deleted)].new_text == ""
        assert any(
            isinstance(item, ContentToolCallContent)
            and "protocol has no deletion flag" in item.content.text
            for item in completion.content
        )

    def test_actual_failed_patch_keeps_file_and_emits_no_diff(
        self, mock_conn, event_loop_fixture, tmp_path
    ):
        from model_tools import handle_function_call

        session_id = "acp-failure-integration"
        target = tmp_path / "failure.txt"
        target.write_text("original\n", encoding="utf-8")
        args = {
            "mode": "replace",
            "path": str(target),
            "old_string": "missing",
            "new_string": "replacement",
        }
        tool_call_ids = {}
        tool_call_meta = {}
        sent = []
        progress_cb = make_tool_progress_cb(
            mock_conn, session_id, event_loop_fixture, tool_call_ids, tool_call_meta
        )
        step_cb = make_step_cb(
            mock_conn, session_id, event_loop_fixture, tool_call_ids, tool_call_meta
        )

        with patch("acp_adapter.events._send_update", side_effect=lambda *call: sent.append(call[3])):
            progress_cb("tool.started", "patch", None, args)
            raw_result = handle_function_call("patch", args, task_id=session_id)
            step_cb(1, [{
                "name": "patch",
                "result": raw_result,
                "arguments": json.dumps(args),
            }])

        assert json.loads(raw_result)["error"]
        assert target.read_text(encoding="utf-8") == "original\n"
        completion = next(update for update in sent if update.session_update == "tool_call_update")
        assert completion.status == "failed"
        assert not any(isinstance(item, FileEditToolCallContent) for item in completion.content)

    def test_parallel_same_name_callbacks_keep_snapshots_with_fifo_ids(
        self, mock_conn, event_loop_fixture, tmp_path
    ):
        from model_tools import handle_function_call

        session_id = "acp-parallel-integration"
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text("first before\n", encoding="utf-8")
        second.write_text("second before\n", encoding="utf-8")
        first_args = {"path": str(first), "content": "first after\n"}
        second_args = {"path": str(second), "content": "second after\n"}
        ids = iter(("tc-first", "tc-second"))
        tool_call_ids = {}
        tool_call_meta = {}
        sent = []
        progress_cb = make_tool_progress_cb(
            mock_conn, session_id, event_loop_fixture, tool_call_ids, tool_call_meta
        )
        step_cb = make_step_cb(
            mock_conn, session_id, event_loop_fixture, tool_call_ids, tool_call_meta
        )

        with patch("acp_adapter.events.make_tool_call_id", side_effect=lambda: next(ids)), \
             patch("acp_adapter.events._send_update", side_effect=lambda *call: sent.append(call[3])):
            progress_cb("tool.started", "write_file", None, first_args)
            progress_cb("tool.started", "write_file", None, second_args)
            first_result = handle_function_call("write_file", first_args, task_id=session_id)
            second_result = handle_function_call("write_file", second_args, task_id=session_id)
            step_cb(1, [
                {
                    "name": "write_file",
                    "result": first_result,
                    "arguments": json.dumps(first_args),
                },
                {
                    "name": "write_file",
                    "result": second_result,
                    "arguments": json.dumps(second_args),
                },
            ])

        completions = {
            update.tool_call_id: update
            for update in sent
            if update.session_update == "tool_call_update"
        }
        first_diff = next(
            item for item in completions["tc-first"].content
            if isinstance(item, FileEditToolCallContent)
        )
        second_diff = next(
            item for item in completions["tc-second"].content
            if isinstance(item, FileEditToolCallContent)
        )
        assert (first_diff.old_text, first_diff.new_text) == ("first before\n", "first after\n")
        assert (second_diff.old_text, second_diff.new_text) == ("second before\n", "second after\n")




# ---------------------------------------------------------------------------
# Message callback
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Scheduler-failure regression
# ---------------------------------------------------------------------------

class TestSendUpdate:
    def test_scheduler_failure_closes_update_coroutine(self, event_loop_fixture):
        """If run_coroutine_threadsafe raises, _send_update must close the coro."""
        created = {"coro": None}

        async def _session_update(session_id, update):
            return None

        conn = MagicMock()

        def _capture_update(session_id, update):
            created["coro"] = _session_update(session_id, update)
            return created["coro"]

        conn.session_update = _capture_update

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch(
                "agent.async_utils.asyncio.run_coroutine_threadsafe",
                side_effect=RuntimeError("scheduler down"),
            ):
                _send_update(conn, "session-1", event_loop_fixture, {"type": "noop"})
            gc.collect()

        assert created["coro"] is not None
        assert created["coro"].cr_frame is None
        # Only count warnings about THIS test's coroutine; other tests
        #  may emit unrelated
        # "coroutine was never awaited" warnings that bleed through.
        runtime_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "was never awaited" in str(w.message)
            and "_session_update" in str(w.message)
        ]
        assert runtime_warnings == []


class TestAssistantMessageIds:
    """Streamed chunks carry a per-message ACP messageId; the None flush sentinel starts a new one."""

    def test_deltas_share_one_uuid_until_flush(self, mock_conn, event_loop_fixture):
        from acp_adapter.events import AssistantMessageIdAllocator

        ids = AssistantMessageIdAllocator()
        cb = make_message_cb(mock_conn, "s", event_loop_fixture, ids)
        sent = []
        with patch("acp_adapter.events._send_update",
                   side_effect=lambda c, s, l, u: sent.append(u)):
            cb("Hello ")
            cb("")  # empty delta is ignored, not a flush
            cb("world")
            cb(None)  # flush sentinel — closes the message
            cb("next turn")
        assert sent[0].message_id == sent[1].message_id
        assert sent[2].message_id != sent[0].message_id
        # ACP requires UUID-format message ids.
        assert uuid.UUID(sent[0].message_id) and uuid.UUID(sent[2].message_id)

    def test_thought_chunks_carry_id(self, mock_conn, event_loop_fixture):
        from acp_adapter.events import AssistantMessageIdAllocator

        ids = AssistantMessageIdAllocator()
        think = make_thinking_cb(mock_conn, "s", event_loop_fixture, ids)
        msg = make_message_cb(mock_conn, "s", event_loop_fixture, ids)
        sent = []
        with patch("acp_adapter.events._send_update",
                   side_effect=lambda c, s, l, u: sent.append(u)):
            think("pondering")
            msg("answer")
        # Reasoning and answer of the same reply share one message id.
        assert sent[0].message_id == sent[1].message_id

    def test_no_allocator_keeps_legacy_shape(self, mock_conn, event_loop_fixture):
        cb = make_message_cb(mock_conn, "s", event_loop_fixture)
        sent = []
        with patch("acp_adapter.events._send_update",
                   side_effect=lambda c, s, l, u: sent.append(u)):
            cb("text")
        assert sent[0].message_id is None
