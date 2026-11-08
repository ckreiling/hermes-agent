"""ACP permission identity and full invocation payload regressions."""
import asyncio
import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from typing import cast

import pytest

import acp
from acp.schema import AllowedOutcome
from acp_adapter.edit_approval import EditProposal, make_acp_edit_approval_requester
from acp_adapter.events import flush_open_tool_calls, make_step_cb, make_tool_progress_cb
from acp_adapter.operations import ToolCallOperations
from acp_adapter.permissions import make_approval_callback
from acp_adapter.tools import build_tool_start


class Client:
    def __init__(self, outcomes=None):
        self.requests = []
        self.updates = []
        self.outcomes = iter(outcomes or ['allow_once'] * 10)

    async def session_update(self, session_id, update):
        self.updates.append(update)

    async def request_permission(self, **kwargs):
        self.requests.append(kwargs['tool_call'])
        outcome = next(self.outcomes)
        if outcome == 'error':
            raise RuntimeError('transport failed')
        if outcome == 'timeout':
            await asyncio.sleep(5)
        return SimpleNamespace(outcome=AllowedOutcome(outcome='selected', option_id=outcome))


def run_worker(worker):
    async def run():
        return await asyncio.to_thread(worker, asyncio.get_running_loop())
    return asyncio.run(run())


def proposal():
    return EditProposal('patch', '/fixture.md', 'old', 'new', {'path': '/fixture.md', 'old_string': 'old', 'new_string': 'new'})


def setup_progress(monkeypatch, loop, sent):
    monkeypatch.setattr('acp_adapter.events._send_update', lambda conn, session, loop, update: sent.append(update))
    operations = ToolCallOperations(sent.append)
    ids, meta, state = {}, {}, {}
    progress = make_tool_progress_cb(cast(acp.Client, None), 's1', loop, ids, meta, turn_state=state, operations=operations)
    return operations, progress, ids, meta, state


def test_nested_edit_permissions_share_operation_without_completing_it(monkeypatch):
    sent = []
    client = Client(['deny', 'allow_once', 'allow_once', 'allow_once', 'allow_once'])
    def worker(loop):
        ops, progress, ids, meta, _ = setup_progress(monkeypatch, loop, sent)
        args = {'code': 'for edit in edits: patch(**edit)'}
        progress('tool.started', 'execute_code', None, args)
        start = sent[0]
        requester = make_acp_edit_approval_requester(client.request_permission, loop, 's1', operations=ops)
        assert [requester(proposal()) for _ in range(5)] == [False, True, True, True, True]
        assert len(sent) == 1  # not five synthetic completions, nor a prematurely closed script
        assert all(r.tool_call_id == start.tool_call_id and r.title == start.title for r in client.requests)
        assert all(r.content[0].new_text == 'new' for r in client.requests)
        assert all(r.raw_input['arguments']['old_string'] == 'old' for r in client.requests)
        assert start.raw_input['code'] == args['code']  # nested proposals never overwrite the operation
        progress('tool.completed', 'execute_code', result='caught denial; continued')
        assert len(sent) == 2 and sent[-1].status == 'completed'
        assert sent[-1].tool_call_id == start.tool_call_id
        assert not ids and not meta
        ops.close()
    run_worker(worker)


def test_command_permission_preserves_exact_command_and_live_identity(monkeypatch):
    sent = []
    client = Client()
    def worker(loop):
        ops, progress, *_ = setup_progress(monkeypatch, loop, sent)
        command = 'printf first; printf second; printf third'
        progress('tool.started', 'terminal', None, {'command': command})
        callback = make_approval_callback(client.request_permission, loop, 's1', operations=ops)
        assert callback(command, 'Approve exact command') == 'once'
        assert len(sent) == 1
        assert client.requests[0].tool_call_id == sent[0].tool_call_id
        assert client.requests[0].raw_input['command'] == command
        assert command in client.requests[0].content[0].content.text
        ops.close()
    run_worker(worker)


@pytest.mark.parametrize('outcome,status', [('allow_once', 'completed'), ('deny', 'failed'), ('error', 'failed'), ('timeout', 'failed')])
def test_standalone_permission_always_has_named_start_and_one_terminal(outcome, status):
    sent = []
    client = Client([outcome])
    def worker(loop):
        ops = ToolCallOperations(sent.append)
        requester = make_acp_edit_approval_requester(client.request_permission, loop, 's1', timeout=.01, operations=ops)
        assert requester(proposal()) == (outcome == 'allow_once')
        assert len(sent) == 2
        assert sent[0].session_update == 'tool_call' and sent[0].title and sent[0].kind == 'edit'
        assert sent[0].content[0].new_text == 'new'
        assert sent[0].tool_call_id == sent[1].tool_call_id
        assert sent[1].status == status
        ops.close()
        assert len(sent) == 2
    run_worker(worker)


def test_same_name_parallel_workers_keep_approval_identity(monkeypatch):
    sent = []
    client = Client()
    async def run():
        loop = asyncio.get_running_loop()
        ops, progress, *_ = setup_progress(monkeypatch, loop, sent)
        barrier = Barrier(2)
        def worker(code):
            progress('tool.started', 'execute_code', None, {'code': code})
            own_id = next(s.tool_call_id for s in sent if s.raw_input['code'] == code)
            barrier.wait(timeout=5)
            callback = make_approval_callback(client.request_permission, loop, 's1', operations=ops)
            assert callback(code, 'nested command') == 'once'
            return own_id, code
        with ThreadPoolExecutor(2) as pool:
            results = await asyncio.gather(*[loop.run_in_executor(pool, worker, code) for code in ['first()', 'second()']])
        assert len(sent) == 2
        assert {(r.tool_call_id, r.raw_input['command']) for r in client.requests} == set(results)
        ops.close()
    asyncio.run(run())


@pytest.mark.parametrize('retirement', ['completion', 'step', 'flush', 'turn'])
def test_copied_context_cannot_reuse_retired_operation(monkeypatch, retirement):
    sent = []
    client = Client()
    def worker(loop):
        ops, progress, ids, meta, state = setup_progress(monkeypatch, loop, sent)
        progress('tool.started', 'execute_code', None, {'code': 'pass'})
        old_id = sent[0].tool_call_id
        copied = contextvars.copy_context()
        if retirement == 'completion':
            progress('tool.completed', 'execute_code', result='ok')
        elif retirement == 'step':
            make_step_cb(cast(acp.Client, None), 's1', loop, ids, meta, state, operations=ops)(1, [{'name': 'execute_code', 'result': 'ok'}])
        elif retirement == 'flush':
            flush_open_tool_calls(cast(acp.Client, None), 's1', loop, ids, meta, operations=ops)
        else:
            ops.close()
        callback = make_approval_callback(client.request_permission, loop, 's1', operations=ops)
        result = copied.run(callback, 'printf late', 'late request')
        if retirement == 'turn':
            assert result == 'deny' and not client.requests
        else:
            assert result == 'once'
            assert client.requests[-1].tool_call_id != old_id
            assert sent[-2].session_update == 'tool_call' and sent[-2].title
        ops.close()
    run_worker(worker)


def test_session_identity_does_not_leak_into_another_turn_or_session():
    sent = []
    client = Client()
    def worker(loop):
        first = ToolCallOperations(sent.append)
        first.bind(build_tool_start('original', 'execute_code', {'code': 'pass'}))
        other = ToolCallOperations(sent.append)
        callback = make_approval_callback(client.request_permission, loop, 's2', operations=other)
        assert callback('printf other', 'different session') == 'once'
        assert client.requests[0].tool_call_id != 'original'
        assert sent[0].session_update == 'tool_call'
        first.close()
        other.close()
    run_worker(worker)


def test_persistent_kernel_cell_authority_carries_current_operation(monkeypatch):
    from tools.code_kernel import CellAuthority
    sent = []
    client = Client()
    def worker(loop):
        ops, progress, *_ = setup_progress(monkeypatch, loop, sent)
        progress('tool.started', 'execute_code', None, {'code': 'patch(**edit)'})
        requester = make_acp_edit_approval_requester(client.request_permission, loop, 's1', operations=ops)
        monkeypatch.setattr('model_tools.handle_function_call', lambda *args, **kw: str(requester(proposal())))
        authority = CellAuthority('s1')
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(authority.dispatch, 'patch', {}).result(timeout=5) == 'True'
        assert client.requests[0].tool_call_id == sent[0].tool_call_id
        assert len(sent) == 1
        authority.retire()
        ops.close()
    run_worker(worker)


def test_teardown_closes_pending_standalone_once_even_if_reply_arrives_late(monkeypatch):
    sent = []
    ops = ToolCallOperations(sent.append)
    def reply_after_teardown(*args, **kwargs):
        ops.close()
        return SimpleNamespace(outcome=AllowedOutcome(outcome='selected', option_id='allow_once')), False
    monkeypatch.setattr('acp_adapter.permissions.await_permission', reply_after_teardown)
    callback = make_approval_callback(Client().request_permission, cast(asyncio.AbstractEventLoop, None), 's1', operations=ops)
    assert callback('printf pending', 'pending approval') == 'once'
    assert [(u.session_update, u.status) for u in sent] == [('tool_call', 'pending'), ('tool_call_update', 'failed')]
    assert sent[0].tool_call_id == sent[1].tool_call_id


def test_server_wires_shared_operation_identity_and_retires_it():
    from acp_adapter.server import HermesACPAgent
    from acp_adapter.session import SessionState
    client = Client()
    sent = client.updates
    # Explicit legacy-direct unit coverage; default-route proof lives in test_gateway_fork_wire.
    server = object.__new__(HermesACPAgent)
    state = SessionState(session_id='s1', agent=SimpleNamespace(), cwd='/project')
    def worker(loop):
        callbacks = server._wire_turn_callbacks(state, 's1', client, loop)
        state.agent.tool_progress_callback('tool.started', 'execute_code', None, {'code': 'patch(**edit)'})
        assert callbacks.edit_approval_requester(proposal())
        assert client.requests[-1].tool_call_id == sent[0].tool_call_id
        assert len(sent) == 1
        state.agent.tool_progress_callback('tool.completed', 'execute_code', result='ok')
        server._flush_turn_tool_calls(callbacks, 's1', client, loop)
        assert len(sent) == 2 and sent[-1].status == 'completed'
        assert callbacks.approval_cb('printf late', 'late callback') == 'deny'
        assert len(client.requests) == 1
    run_worker(worker)


@pytest.mark.parametrize('name,args', [
    ('terminal', {'command': 'printf first;\n' + 'printf more;\n' * 30, 'timeout': 45}),
    ('execute_code', {'code': 'print(1)\n' * 40}),
    ('search_files', {'pattern': 'needle', 'path': '/project', 'limit': 3}),
    ('web_search', {'query': 'exact query', 'limit': 4}),
    ('read_file', {'path': '/project/file.py', 'offset': 5, 'limit': 9}),
    ('write_file', {'path': '/project/new.py', 'content': 'print(1)\n' * 30}),
])
def test_structured_arguments_survive_compact_presentation(name, args):
    start = build_tool_start('full', name, args)
    wire = start.model_dump(by_alias=True, exclude_none=True)
    assert all(wire['rawInput'][key] == value for key, value in args.items())
    if name == 'execute_code':
        assert wire['rawInput']['command'] == args['code']
    assert json.loads(json.dumps(wire))['rawInput'] == wire['rawInput']


def test_full_invocation_payload_preserves_redaction_without_mutating_arguments():
    secret = 'sk-proj-' + 'aB7xQ9zW2kR8nT4u' * 4
    args = {'command': f'curl -H "Authorization: Bearer {secret}" https://example.com'}
    start = build_tool_start('safe', 'terminal', args)
    assert secret not in start.model_dump_json()
    assert secret in args['command']
    typed = build_tool_start('browser', 'browser_type', {'ref': '@e1', 'text': secret})
    assert secret not in typed.model_dump_json()
