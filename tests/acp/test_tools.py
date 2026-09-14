"""Tests for acp_adapter.tools — tool kind mapping and ACP content building."""


from acp_adapter.edit_approval import EditProposal
from agent.display import capture_local_edit_snapshot
from acp_adapter.tools import (
    TOOL_KIND_MAP,
    build_tool_complete,
    build_tool_start,
    build_tool_title,
    extract_locations,
    get_tool_kind,
    make_tool_call_id,
)
from acp.schema import (
    FileEditToolCallContent,
    ContentToolCallContent,
    ToolCallLocation,
    ToolCallStart,
    ToolCallProgress,
)


# ---------------------------------------------------------------------------
# TOOL_KIND_MAP coverage
# ---------------------------------------------------------------------------


COMMON_HERMES_TOOLS = ["read_file", "search_files", "terminal", "patch", "write_file", "process"]


class TestToolKindMap:
    def test_all_hermes_tools_have_kind(self):
        """Every common hermes tool should appear in TOOL_KIND_MAP."""
        for tool in COMMON_HERMES_TOOLS:
            assert tool in TOOL_KIND_MAP, f"{tool} missing from TOOL_KIND_MAP"

    def test_tool_kind_read_file(self):
        assert get_tool_kind("read_file") == "read"

    def test_tool_kind_terminal(self):
        assert get_tool_kind("terminal") == "execute"








    def test_unknown_tool_returns_other_kind(self):
        assert get_tool_kind("nonexistent_tool_xyz") == "other"


# ---------------------------------------------------------------------------
# make_tool_call_id
# ---------------------------------------------------------------------------


class TestMakeToolCallId:
    def test_returns_string(self):
        tc_id = make_tool_call_id()
        assert isinstance(tc_id, str)

    def test_starts_with_tc_prefix(self):
        tc_id = make_tool_call_id()
        assert tc_id.startswith("tc-")

    def test_ids_are_unique(self):
        ids = {make_tool_call_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# build_tool_title
# ---------------------------------------------------------------------------


class TestBuildToolTitle:
    def test_terminal_title_includes_command(self):
        title = build_tool_title("terminal", {"command": "ls -la /tmp"})
        assert "ls -la /tmp" in title

    def test_terminal_title_truncates_long_command(self):
        long_cmd = "x" * 200
        title = build_tool_title("terminal", {"command": long_cmd})
        assert len(title) < 120
        assert "..." in title

    def test_read_file_title(self):
        title = build_tool_title("read_file", {"path": "/etc/hosts"})
        assert "/etc/hosts" in title


    def test_search_title(self):
        title = build_tool_title("search_files", {"pattern": "TODO"})
        assert "TODO" in title




    def test_skill_view_title_includes_skill_name(self):
        title = build_tool_title("skill_view", {"name": "github-pitfalls"})
        assert title == "skill view (github-pitfalls)"


    def test_execute_code_title_includes_first_code_line(self):
        title = build_tool_title("execute_code", {"code": "\nfrom hermes_tools import terminal\nprint('done')"})
        assert title == "python: from hermes_tools import terminal"


    def test_unknown_tool_uses_name(self):
        title = build_tool_title("some_new_tool", {"foo": "bar"})
        assert title == "some_new_tool"


# ---------------------------------------------------------------------------
# build_tool_start
# ---------------------------------------------------------------------------


class TestBuildToolStart:
    def test_build_tool_start_for_patch(self):
        """patch start should not duplicate the edit-approval diff."""
        args = {
            "path": "src/main.py",
            "old_string": "print('hello')",
            "new_string": "print('world')",
        }
        result = build_tool_start("tc-1", "patch", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "edit"
        assert len(result.content) >= 1
        item = result.content[0]
        assert isinstance(item, ContentToolCallContent)
        assert "Approval prompt shows the diff" in item.content.text
        assert "src/main.py" in item.content.text


    def test_auto_approved_edit_start_shows_diff_content(self):
        """Auto-approved edit starts need the diff because no approval card exists."""
        args = {"path": "/tmp/acp.txt", "old_string": "old", "new_string": "new"}
        result = build_tool_start(
            "tc-auto-edit",
            "patch",
            args,
            edit_diff=EditProposal("patch", "/tmp/acp.txt", "old\n", "new\n", args),
        )

        assert isinstance(result, ToolCallStart)
        assert result.kind == "edit"
        assert len(result.content) == 1
        item = result.content[0]
        assert isinstance(item, FileEditToolCallContent)
        assert item.path == "/tmp/acp.txt"
        assert item.old_text == "old\n"
        assert item.new_text == "new\n"







    def test_build_tool_start_for_browser_navigate(self):
        """browser_navigate should emit a polished start event."""
        args = {"url": "https://x.com"}
        result = build_tool_start("tc-browser-start", "browser_navigate", args)
        assert isinstance(result, ToolCallStart)
        assert result.title == "navigate: https://x.com"
        assert result.kind == "fetch"
        assert result.content[0].content.text == '{\n  "url": "https://x.com"\n}'
        assert result.raw_input is None








# ---------------------------------------------------------------------------
# build_tool_complete
# ---------------------------------------------------------------------------


class TestBuildToolComplete:
    def test_build_tool_complete_for_terminal(self):
        """Completed terminal call should include output text."""
        result = build_tool_complete("tc-2", "terminal", "total 42\ndrwxr-xr-x 2 root root 4096 ...")
        assert isinstance(result, ToolCallProgress)
        assert result.status == "completed"
        assert len(result.content) >= 1
        content_item = result.content[0]
        assert isinstance(content_item, ContentToolCallContent)
        assert "total 42" in content_item.content.text
        assert result.raw_output is None








    def test_build_tool_complete_marks_returncode_nonzero_as_failed(self):
        result = build_tool_complete("tc-fail", "execute_code", '{"output": "bad", "returncode": 2}')
        assert result.status == "failed"








    def test_build_tool_complete_for_search_files_formats_matches(self):
        result = build_tool_complete(
            "tc-search",
            "search_files",
            '{"total_count":2,"matches":[{"path":"README.md","line":3,"content":"TODO: fix this"},{"path":"src/app.py","line":9,"content":"needle"}],"truncated":true}\n\n[Hint: Results truncated. Use offset=12 to see more.]',
        )
        text = result.content[0].content.text
        assert "Search results" in text
        assert "Found 2 matches" in text
        assert "README.md:3" in text
        assert "TODO: fix this" in text
        assert "Results truncated" in text
        assert result.raw_output is None







    def test_build_tool_complete_generically_formats_unknown_json_dict_without_raw_output(self):
        result = build_tool_complete(
            "tc-recall-search",
            "memory_archive_search",
            '{"results":[{"id":"obs-1","status":"active","content":"Recall should render as a readable summary."}],"trust":"lower-trust archive evidence"}',
        )
        text = result.content[0].content.text
        assert "memory_archive_search result" in text
        assert "lower-trust archive evidence" in text
        assert "Recall should render as a readable summary" in text
        assert "{\"results\"" not in text
        assert result.raw_output is None

    def test_write_file_completion_contains_summary_and_full_file_diff(self, tmp_path):
        target = tmp_path / "note.txt"
        original = "heading\nkeep this line\nold value\n"
        updated = "heading\nkeep this line\nnew value\ntrailer\n"
        target.write_text(original, encoding="utf-8")
        snapshot = capture_local_edit_snapshot(
            "write_file", {"path": str(target)}, task_id="acp-test"
        )
        target.write_text(updated, encoding="utf-8")

        result = build_tool_complete(
            "tc-write",
            "write_file",
            '{"bytes_written": 41, "files_modified": ["note.txt"]}\n\n[Hint: context loaded]',
            function_args={"path": str(target), "content": updated},
            snapshot=snapshot,
        )

        summaries = [item for item in result.content if isinstance(item, ContentToolCallContent)]
        diffs = [item for item in result.content if isinstance(item, FileEditToolCallContent)]
        assert result.status == "completed"
        assert any("write_file completed" in item.content.text for item in summaries)
        assert len(diffs) == 1
        assert diffs[0].path == str(target)
        assert diffs[0].old_text == original
        assert diffs[0].new_text == updated

    def test_completion_without_live_snapshot_does_not_reconstruct_from_result_diff(self, tmp_path):
        target = tmp_path / "replayed.txt"
        target.write_text("current filesystem state\n", encoding="utf-8")
        result = build_tool_complete(
            "tc-replay",
            "patch",
            '{"success": true, "diff": "--- a/replayed.txt\\n+++ b/replayed.txt\\n@@ -1 +1 @@\\n-old\\n+new\\n"}',
            function_args={"path": str(target), "old_string": "old", "new_string": "new"},
        )

        assert result.status == "completed"
        assert not any(isinstance(item, FileEditToolCallContent) for item in result.content)
        assert any(isinstance(item, ContentToolCallContent) for item in result.content)

    def test_failed_and_noop_edits_do_not_emit_completion_diffs(self, tmp_path):
        target = tmp_path / "unchanged.txt"
        target.write_text("same\n", encoding="utf-8")
        snapshot = capture_local_edit_snapshot(
            "patch", {"path": str(target)}, task_id="acp-test"
        )

        failed = build_tool_complete(
            "tc-failed-edit",
            "patch",
            '{"success": false, "error": "old_string not found"}',
            function_args={"path": str(target)},
            snapshot=snapshot,
        )
        noop = build_tool_complete(
            "tc-noop-edit",
            "patch",
            '{"success": true, "no_change": true, "note": "already applied"}',
            function_args={"path": str(target)},
            snapshot=snapshot,
        )

        assert failed.status == "failed"
        assert not any(isinstance(item, FileEditToolCallContent) for item in failed.content)
        assert noop.status == "completed"
        assert not any(isinstance(item, FileEditToolCallContent) for item in noop.content)
        assert "already applied" in noop.content[0].content.text

    def test_missing_edit_result_is_not_reported_as_success(self, tmp_path):
        target = tmp_path / "unknown.txt"
        target.write_text("before\n", encoding="utf-8")
        snapshot = capture_local_edit_snapshot(
            "write_file", {"path": str(target)}, task_id="acp-test"
        )
        target.write_text("after\n", encoding="utf-8")

        result = build_tool_complete(
            "tc-missing-result",
            "write_file",
            None,
            function_args={"path": str(target)},
            snapshot=snapshot,
        )

        assert result.status == "failed"
        assert not any(isinstance(item, FileEditToolCallContent) for item in result.content)
        assert "unavailable" in result.content[0].content.text.lower()

    def test_large_and_binary_snapshots_are_omitted_instead_of_truncated(self, tmp_path):
        for name, original in (
            ("large.txt", "x" * 1_100_000),
            ("binary.dat", b"prefix\x00suffix"),
        ):
            target = tmp_path / name
            if isinstance(original, bytes):
                target.write_bytes(original)
            else:
                target.write_text(original, encoding="utf-8")
            snapshot = capture_local_edit_snapshot(
                "write_file", {"path": str(target)}, task_id="acp-test"
            )
            target.write_text("safe replacement\n", encoding="utf-8")

            result = build_tool_complete(
                f"tc-{name}",
                "write_file",
                '{"bytes_written": 17}',
                function_args={"path": str(target)},
                snapshot=snapshot,
            )

            assert not any(isinstance(item, FileEditToolCallContent) for item in result.content)
            assert any(
                isinstance(item, ContentToolCallContent)
                and "could not be captured safely" in item.content.text
                for item in result.content
            )









# ---------------------------------------------------------------------------
# extract_locations
# ---------------------------------------------------------------------------


class TestExtractLocations:
    def test_extract_locations_with_path(self):
        args = {"path": "src/app.py", "offset": 42}
        locs = extract_locations(args)
        assert len(locs) == 1
        assert isinstance(locs[0], ToolCallLocation)
        assert locs[0].path == "src/app.py"
        assert locs[0].line == 42

    def test_extract_locations_without_path(self):
        args = {"command": "echo hi"}
        locs = extract_locations(args)
        assert locs == []
