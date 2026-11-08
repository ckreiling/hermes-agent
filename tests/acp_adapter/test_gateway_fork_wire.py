"""Real ACP stdio + isolated ordinary gateway + deterministic HTTP/tool peers.

These prove the default entrypoint, not direct ACP or real provider inference.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall
from tests.waiting import async_wait_until


@pytest.fixture
def model_peer():
    with FakeLLMServer() as provider:
        yield SimpleNamespace(server_port=provider.port, provider=provider)


def updates(acp):
    return [f['params']['update'] for f in acp.frames if f.get('method') == 'session/update']


def permissions(acp):
    return [f for f in acp.frames if f.get('method') == 'session/request_permission']


async def initialize(acp):
    assert 'result' in await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})


async def new(acp, cwd, servers=None):
    reply = await acp.rpc('session/new', cwd=str(cwd), mcpServers=servers or [])
    assert 'result' in reply, reply
    return reply['result']['sessionId']


async def prompt(acp, sid):
    reply = await acp.rpc('session/prompt', sessionId=sid, prompt=[{'type': 'text', 'text': 'Run deterministic fixture'}])
    assert reply.get('result', {}).get('stopReason') == 'end_turn', reply
    return reply


async def answer(acp, permission, choice):
    outcome = {'outcome': 'selected', 'optionId': choice} if choice else {'outcome': 'cancelled'}
    acp.process.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': permission['id'],
        'result': {'outcome': outcome}}).encode() + b'\n')
    await acp.process.stdin.drain()


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_default_wire_full_file_diffs_and_interim_final_once(daemon, tmp_path, model_peer):
    (tmp_path / 'two.txt').write_text('old two\n')
    patch = ('*** Begin Patch\n*** Update File: new.txt\n@@\n-second\n+third\n'
             '*** Update File: two.txt\n@@\n-old two\n+new two\n*** End Patch')
    model_peer.provider.push(
        ToolCall('write_file', {'path': 'new.txt', 'content': ''}, text='INTERIM_PROGRESS'),
        ToolCall('write_file', {'path': 'new.txt', 'content': 'first\n'}),
        ToolCall('patch', {'mode': 'replace', 'path': 'new.txt', 'old_string': 'first', 'new_string': 'second'}),
        ToolCall('patch', {'mode': 'patch', 'patch': patch}),
        Text('FINAL_ONCE'))
    async with viewer(daemon) as ws, editor(daemon, tmp_path) as acp:
        await initialize(acp)
        sid = await new(acp, tmp_path)
        assert 'result' in await acp.rpc('session/set_mode', sessionId=sid, modeId='accept_edits')
        await prompt(acp, sid)
        sent = updates(acp)
        completed = [u for u in sent if u['sessionUpdate'] == 'tool_call_update' and u.get('content')]
        diffs = [c for u in completed for c in u['content'] if c['type'] == 'diff']
        assert [(d.get('oldText'), d['newText']) for d in diffs] == [
            (None, ''), ('', 'first\n'), ('first\n', 'second\n'),
            ('second\n', 'third\n'), ('old two\n', 'new two\n')], completed
        assert all(d['path'].startswith(str(tmp_path)) for d in diffs)
        assert (tmp_path / 'new.txt').read_text() == 'third\n'
        assert (tmp_path / 'two.txt').read_text() == 'new two\n'
        chunks = [u for u in sent if u['sessionUpdate'] == 'agent_message_chunk']
        text = ''.join(u['content']['text'] for u in chunks)
        assert text.count('INTERIM_PROGRESS') == text.count('FINAL_ONCE') == 1, chunks
        progress = next(i for i, u in enumerate(sent) if 'INTERIM_PROGRESS' in str(u))
        first_completion = next(i for i, u in enumerate(sent) if u['sessionUpdate'] == 'tool_call_update')
        assert progress < first_completion
        assert chunks[0]['messageId'] != chunks[-1]['messageId']
        snapshot = await ws.rpc('session.resume', session_id=sid)
        assert 'FINAL_ONCE' in json.dumps(snapshot['messages'])
        print('FORK_DIFF_RECEIPT=' + json.dumps({'session_id': sid, 'diffs': diffs, 'text': text}))


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('first_choice', ['allow_once', 'deny', None])
async def test_default_wire_nested_edits_share_outer_card_until_real_completion(
        daemon, tmp_path, model_peer, first_choice):
    code = ('from hermes_tools import write_file\n'
            'print(write_file(path="one.txt", content="one"))\n'
            'print(write_file(path="two.txt", content="two"))\n')
    model_peer.provider.push(ToolCall('execute_code', {'code': code}), Text('SCRIPT_FINISHED'))
    async with viewer(daemon) as ws, editor(daemon, tmp_path) as acp:
        await initialize(acp)
        sid = await new(acp, tmp_path)
        task = asyncio.create_task(prompt(acp, sid))
        handled, edits, outer_id = set(), [], None
        try:
            while not task.done():
                await async_wait_until(lambda: task.done() or any(p['id'] not in handled for p in permissions(acp)),
                                       timeout=25, what='nested approval or completed fixture')
                for permission in [p for p in permissions(acp) if p['id'] not in handled]:
                    handled.add(permission['id'])
                    call = permission['params']['toolCall']
                    starts = [u for u in updates(acp) if u['sessionUpdate'] == 'tool_call']
                    outer = next(u for u in starts if u.get('rawInput', {}).get('code') == code)
                    outer_id = outer['toolCallId']
                    assert (call['toolCallId'], call['title']) == (outer_id, outer['title']), permission
                    assert not [u for u in updates(acp) if u['sessionUpdate'] == 'tool_call_update'
                                and u['toolCallId'] == outer_id], updates(acp)
                    is_edit = call.get('kind') == 'edit'
                    if is_edit:
                        edits.append(call)
                    choice = first_choice if is_edit and len(edits) == 1 else 'allow_once'
                    await answer(acp, permission, choice)
                    if choice is None:
                        # Editor dismissal must not decide the canonical waiter or close the operation.
                        pending = (await ws.rpc('session.resume', session_id=sid))['prompts']
                        assert len(pending) == 1
                        assert pending[0]['operation']['tool_call_id'] == outer_id
                        assert not (tmp_path / 'one.txt').exists()
                        await ws.rpc('approval.respond', session_id=sid, prompt_id=pending[0]['prompt_id'],
                                     execution_generation=pending[0]['execution_generation'], choice='deny')
            await task
            assert len(edits) == 2, permissions(acp)
            assert (tmp_path / 'one.txt').exists() == (first_choice == 'allow_once')
            assert (tmp_path / 'two.txt').read_text() == 'two'
            calls = [u for u in updates(acp) if u.get('toolCallId') == outer_id]
            assert [u['sessionUpdate'] for u in calls] == ['tool_call', 'tool_call_update'], calls
            assert calls[-1]['status'] == 'completed'
            print('FORK_NESTED_RECEIPT=' + json.dumps({'choice': first_choice, 'operation': outer_id,
                  'permission_count': len(handled), 'cards': calls}))
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_busy_reasoning_change_applies_only_from_next_turn(daemon, tmp_path, model_peer):
    import shlex
    target = tmp_path / 'owned-dir'
    target.mkdir()
    (target / 'owned').write_text('disposable')
    model_peer.provider.push(ToolCall('terminal', {'command': 'rm -r -- ' + shlex.quote(str(target))}),
                             Text('CURRENT_HIGH'), Text('NEXT_NONE'))
    async with viewer(daemon) as ws, editor(daemon, tmp_path) as acp:
        await initialize(acp)
        sid = await new(acp, tmp_path)
        assert 'result' in await acp.rpc('session/set_config_option', sessionId=sid,
                                       configId='reasoning_effort', value='high')
        task = asyncio.create_task(prompt(acp, sid))
        try:
            await async_wait_until(lambda: bool(permissions(acp)), timeout=20, what='held terminal approval')
            assert model_peer.provider.main_requests()[-1].get('reasoning_effort') == 'high'
            reply = await acp.rpc('session/set_config_option', sessionId=sid,
                                  configId='reasoning_effort', value='none')
            assert 'result' in reply, reply
            await answer(acp, permissions(acp)[0], 'allow_once')
            await task
            assert not target.exists()
            assert model_peer.provider.main_requests()[-1].get('reasoning_effort') == 'high'
            await prompt(acp, sid)
            assert model_peer.provider.main_requests()[-1].get('reasoning_effort') in ('none', None)
            assert (await ws.rpc('session.info', session_id=sid))['reasoning_effort'] == 'none'
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_reasoning_provider_and_cwd_survive_new_acp_process(daemon, tmp_path, model_peer):
    model_peer.provider.push(Text('HIGH'), Text('NONE'), ToolCall('write_file', {'path': 'moved.txt', 'content': 'moved'}), Text('MOVED'))
    async with viewer(daemon) as ws:
        async with editor(daemon, tmp_path) as acp:
            await initialize(acp)
            sid = await new(acp, tmp_path)
            assert 'result' in await acp.rpc('session/set_config_option', sessionId=sid, configId='reasoning_effort', value='high')
            await prompt(acp, sid)
            assert model_peer.provider.main_requests()[-1].get('reasoning_effort') == 'high'
            bad = await acp.rpc('session/set_config_option', sessionId=sid, configId='reasoning_effort', value='invalid')
            assert 'error' in bad
            assert (await ws.rpc('session.info', session_id=sid))['reasoning_effort'] == 'high'
            assert 'result' in await acp.rpc('session/set_config_option', sessionId=sid, configId='reasoning_effort', value='none')
        moved = tmp_path / 'moved'
        moved.mkdir()
        async with editor(daemon, tmp_path) as reloaded:
            await initialize(reloaded)
            loaded = await reloaded.rpc('session/load', sessionId=sid, cwd=str(moved), mcpServers=[])
            assert 'result' in loaded, loaded
            info = await ws.rpc('session.info', session_id=sid)
            assert info['reasoning_effort'] == 'none'
            assert info['provider'] == 'custom' and info['cwd'] == str(moved)
            await prompt(reloaded, sid)
            assert model_peer.provider.main_requests()[-1].get('reasoning_effort') in ('none', None)
            assert 'result' in await reloaded.rpc('session/set_mode', sessionId=sid, modeId='accept_edits')
            await prompt(reloaded, sid)
            assert (moved / 'moved.txt').read_text() == 'moved'
            assert not (tmp_path / 'moved.txt').exists()
            print('FORK_RESTORE_RECEIPT=' + json.dumps({'session_id': sid, 'provider': info['provider'],
                  'cwd': info['cwd'], 'reasoning': info['reasoning_effort']}))
