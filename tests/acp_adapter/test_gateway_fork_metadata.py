"""Gateway-owned fork presentation guarantees (no live provider inference)."""
import asyncio
import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from acp.schema import AllowedOutcome, DeniedOutcome

from acp_adapter.gateway_server import GatewayACPAgent
from gateway.tool_operations import ToolOperations


def test_default_constructor_is_gateway_backed():
    from acp_adapter.server import HermesACPAgent
    assert isinstance(HermesACPAgent(), GatewayACPAgent)


def test_operation_identity_is_worker_scoped_and_retired_copies_cannot_reuse_it():
    operations = ToolOperations()
    barrier = Barrier(2)
    def run(name):
        operations.bind(name, 'execute_code', {'code': name})
        context = contextvars.copy_context()
        barrier.wait(timeout=5)
        current = operations.presentation()
        operations.retire(name)
        assert context.run(operations.presentation) is None
        return current['tool_call_id']
    with ThreadPoolExecutor(2) as pool:
        assert set(pool.map(run, ['a', 'b'])) == {'a', 'b'}
    operations.bind('duplicate', 'terminal', {'command': 'first'})
    old_context = contextvars.copy_context()
    operations.bind('duplicate', 'terminal', {'command': 'replacement'})
    assert old_context.run(operations.presentation) is None
    operations.retire('duplicate')
    operations.bind('c', 'terminal', {'command': 'true'})
    context = contextvars.copy_context()
    operations.close()
    assert context.run(operations.presentation) is None
    assert ToolOperations().presentation() is None


def test_structured_edit_metadata_is_bounded_secret_free_and_observed(tmp_path):
    from agent.display import capture_local_edit_snapshot
    from gateway.local_viewer_events import structured_edit_fields
    path = tmp_path / 'small'
    path.write_text('old\n')
    snapshot = capture_local_edit_snapshot('write_file', {'path': str(path)})
    path.write_text('new\n')
    fields = structured_edit_fields('write_file', '{"bytes_written":4}', snapshot)
    assert fields['edit_changes'] == [{'path': str(path), 'old_text': 'old\n', 'new_text': 'new\n', 'deleted': False}]
    assert structured_edit_fields('write_file', '{"error":"denied"}', snapshot)['edit_changes'] == []
    assert structured_edit_fields('write_file', '{"bytes_written":4}', snapshot, is_error=True)['edit_changes'] == []
    for value in ('x' * 70000, 'sk-proj-' + 'aB7xQ9zW2kR8nT4u' * 4):
        path.write_text(value)
        fields = structured_edit_fields('write_file', '{"bytes_written":4}', snapshot)
        assert fields['edit_changes'] == [] and fields['edit_skipped'] == 1
        assert value not in json.dumps(fields)


@pytest.mark.asyncio
@pytest.mark.parametrize('choice', ['allow_once', 'deny', None])
async def test_correlated_permission_never_completes_outer_operation(choice):
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    agent._gateway = AsyncMock()
    agent._conn.request_permission.return_value = SimpleNamespace(outcome=(
        AllowedOutcome(outcome='selected', option_id=choice) if choice else DeniedOutcome(outcome='cancelled')))
    event = {'session_id': 's', 'admission_id': 'a', 'type': 'tool.start',
             'payload': {'tool_call_id': 'outer', 'name': 'execute_code', 'args': {'code': 'patch(**edit)'}}}
    await agent._project(event)
    start = agent._conn.session_update.await_args.kwargs['update']
    await agent._answer_permission('s', {'kind': 'approval', 'prompt_id': 'p', 'execution_generation': 1,
        'command': 'exact nested command', 'choices': ['once', 'deny'],
        'operation': {'tool_call_id': 'outer', 'name': 'execute_code', 'args': {}, 'admission_id': 'a'}})
    assert agent._conn.session_update.await_count == 1
    request = agent._conn.request_permission.await_args.kwargs['tool_call']
    assert (request.tool_call_id, request.title) == ('outer', start.title)
    assert request.raw_input['command'] == 'exact nested command'
    await agent._project({**event, 'type': 'tool.complete', 'payload': {
        **event['payload'], 'result': '{"success":true}', 'is_error': False}})
    assert agent._conn.session_update.await_count == 2
    assert agent._conn.session_update.await_args.kwargs['update'].tool_call_id == 'outer'


@pytest.mark.asyncio
async def test_gateway_nested_tool_call_invocation_and_result_are_structured():
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    args = {'calls': [{'name': 'fixture__search', 'arguments': {'query': 'exact needle', 'limit': 3}},
                      {'name': 'fixture__read', 'arguments': {'path': '/fixture.txt', 'offset': 7}}]}
    payload = {'tool_call_id': 'nested', 'name': 'tool_call', 'args': args}
    await agent._project({'session_id': 's', 'admission_id': 'a', 'type': 'tool.start', 'payload': payload})
    start = agent._conn.session_update.await_args.kwargs['update']
    assert start.raw_input == args
    await agent._project({'session_id': 's', 'admission_id': 'a', 'type': 'tool.complete', 'payload': {
        **payload, 'result': json.dumps({'results': [{'name': 'fixture__search', 'result': {'matches': ['needle']}},
                                                     {'name': 'fixture__read', 'result': {'content': 'file body'}}]}),
        'is_error': False}})
    completion = agent._conn.session_update.await_args.kwargs['update']
    assert completion.tool_call_id == start.tool_call_id and completion.status == 'completed'
    assert 'needle' in completion.model_dump_json() and 'file body' in completion.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize('shape', ['same_interim', 'transformed'])
async def test_gateway_final_identity_and_exactly_once_completion_text(shape):
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    async def project(kind, payload):
        await agent._project({'session_id': 's', 'admission_id': 'a', 'type': kind, 'payload': payload})
    if shape == 'same_interim':
        await project('message.interim', {'text': 'FINAL', 'already_streamed': False})
        await project('message.complete', {'text': 'FINAL', 'outcome': 'completed'})
        chunks = [c.kwargs['update'] for c in agent._conn.session_update.await_args_list]
        assert ''.join(c.content.text for c in chunks).count('FINAL') == 1
    else:
        await project('message.delta', {'text': 'ORIGINAL'})
        await project('message.complete', {'text': 'TRANSFORMED', 'outcome': 'completed'})
        chunks = [c.kwargs['update'] for c in agent._conn.session_update.await_args_list]
        assert ''.join(c.content.text for c in chunks) == 'ORIGINAL\n\nTRANSFORMED'
        assert chunks[0].message_id == chunks[1].message_id
        # ACP has no replace-message primitive: upstream appends the correction
        # in the same message, once; this does not claim erasure of streamed text.


@pytest.mark.asyncio
async def test_standalone_permission_has_named_start_and_single_terminal_on_cancel():
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    agent._gateway = AsyncMock()
    entered = asyncio.Event()
    async def wait(**kwargs):
        entered.set()
        await asyncio.Event().wait()
    agent._conn.request_permission.side_effect = wait
    task = asyncio.create_task(agent._answer_permission('s', {'kind': 'approval', 'prompt_id': 'p',
        'execution_generation': 1, 'command': 'standalone', 'choices': ['once', 'deny']}))
    await entered.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    updates = [c.kwargs['update'] for c in agent._conn.session_update.await_args_list]
    assert len(updates) == 2
    assert updates[0].title and updates[0].session_update == 'tool_call'
    assert updates[0].tool_call_id == updates[1].tool_call_id
    assert updates[1].status == 'failed'
    agent._gateway.rpc.assert_not_awaited()
