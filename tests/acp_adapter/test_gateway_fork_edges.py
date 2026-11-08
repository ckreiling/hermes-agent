"""Edge acceptance on gateway execution/edit events and session policy boundaries."""
import json

import pytest

from tests.acp_adapter.test_gateway_fork_wire import initialize, model_peer, new, prompt, updates
from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.fakes.fake_llm_provider import Text, ToolCall


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_default_wire_delete_rename_noop_failure_and_oversize(daemon, tmp_path, model_peer):
    (tmp_path / 'remove.txt').write_text('remove\n')
    (tmp_path / 'rename.txt').write_text('rename\n')
    (tmp_path / 'same.txt').write_text('same\n')
    patch = ('*** Begin Patch\n*** Delete File: remove.txt\n'
             '*** Move File: rename.txt -> moved.txt\n*** End Patch')
    model_peer.provider.push(
        ToolCall('patch', {'mode': 'patch', 'patch': patch}),
        ToolCall('read_file', {'path': 'same.txt'}),
        ToolCall('write_file', {'path': 'same.txt', 'content': 'same\n'}),
        ToolCall('patch', {'mode': 'replace', 'path': 'same.txt', 'old_string': 'absent', 'new_string': 'bad'}),
        ToolCall('write_file', {'path': 'large.txt', 'content': 'x' * 70000}), Text('EDGES_DONE'))
    async with editor(daemon, tmp_path) as acp:
        await initialize(acp)
        sid = await new(acp, tmp_path)
        assert 'result' in await acp.rpc('session/set_mode', sessionId=sid, modeId='accept_edits')
        await prompt(acp, sid)
        sent = updates(acp)
        completions = [u for u in sent if u['sessionUpdate'] == 'tool_call_update']
        diff_rows = [(u, c) for u in completions for c in u.get('content', []) if c['type'] == 'diff']
        actual = {c['path']: (c.get('oldText'), c['newText']) for _, c in diff_rows}
        assert actual == {str(tmp_path / 'remove.txt'): ('remove\n', ''),
                          str(tmp_path / 'rename.txt'): ('rename\n', ''),
                          str(tmp_path / 'moved.txt'): (None, 'rename\n')}, completions
        assert not (tmp_path / 'remove.txt').exists() and not (tmp_path / 'rename.txt').exists()
        assert (tmp_path / 'moved.txt').read_text() == 'rename\n'
        assert (tmp_path / 'same.txt').read_text() == 'same\n'
        assert (tmp_path / 'large.txt').stat().st_size == 70000
        failed = [u for u in completions if u['status'] == 'failed']
        assert len(failed) == 1
        assert not any(c['type'] == 'diff' for c in failed[0].get('content', []))
        assert 'Structured diff unavailable for 1 files' in json.dumps(completions)
        assert all('x' * 1000 not in json.dumps(u) for u in completions)
        print('FORK_EDGE_RECEIPT=' + json.dumps({'diffs': actual, 'failures': len(failed)}))


def test_binary_and_nonlocal_snapshots_never_invent_full_file_diffs(tmp_path, monkeypatch):
    from agent.display import capture_local_edit_snapshot
    from gateway.local_viewer_events import structured_edit_fields
    path = tmp_path / 'binary'
    path.write_bytes(b'\xff\x00binary')
    snapshot = capture_local_edit_snapshot('write_file', {'path': str(path)})
    path.write_text('now text')
    fields = structured_edit_fields('write_file', '{"bytes_written":8}', snapshot)
    assert fields == {'edit_changes': [], 'edit_skipped': 1}
    monkeypatch.setattr('tools.file_tools_paths._terminal_env_type_for_task', lambda task: 'ssh')
    assert capture_local_edit_snapshot('write_file', {'path': str(path)}, task_id='remote') is None
    assert structured_edit_fields('write_file', '{"bytes_written":8}', None)['edit_changes'] == []


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_reasoning_and_workspace_are_session_scoped_and_fork_inherits_explicit_effort(daemon, tmp_path, model_peer):
    model_peer.provider.push(Text('PARENT_HIGH'), Text('CHILD_NONE'))
    async with viewer(daemon) as ws, editor(daemon, tmp_path) as acp:
        await initialize(acp)
        parent = await new(acp, tmp_path)
        assert 'result' in await acp.rpc('session/set_config_option', sessionId=parent,
                                       configId='reasoning_effort', value='high')
        fork_cwd = tmp_path / 'forked'
        fork_cwd.mkdir()
        reply = await acp.rpc('session/fork', sessionId=parent, cwd=str(fork_cwd), mcpServers=[])
        assert 'result' in reply, reply
        child = reply['result']['sessionId']
        info = await ws.rpc('session.info', session_id=child)
        assert info['reasoning_effort'] == 'high' and info['cwd'] == str(fork_cwd)
        assert 'result' in await acp.rpc('session/set_config_option', sessionId=child,
                                       configId='reasoning_effort', value='none')
        await prompt(acp, parent)
        await prompt(acp, child)
        requests = model_peer.provider.main_requests()
        assert requests[0].get('reasoning_effort') == 'high'
        assert requests[1].get('reasoning_effort') in ('none', None)
        assert (await ws.rpc('session.info', session_id=parent))['reasoning_effort'] == 'high'
        assert (await ws.rpc('session.info', session_id=parent))['cwd'] == str(tmp_path)
