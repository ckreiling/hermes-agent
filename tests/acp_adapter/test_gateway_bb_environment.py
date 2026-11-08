"""Actual default ACP/gateway/tool transports with deterministic BB-like CLI/MCP.

These are local fixtures, not the real BB CLI, BB MCP or provider inference.
"""
import asyncio
import json
import os
import shlex
import sqlite3
import sys

import pytest

from tests.acp_adapter.test_gateway_fork_wire import (answer, initialize, model_peer, new, permissions, updates)
from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.waiting import async_wait_until


@pytest.fixture
def bb_fixture(tmp_path):
    path = tmp_path / 'bb-bin'
    path.mkdir()
    cli = path / 'bb'
    cli.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
        'assert sys.argv[1:]==["status","--json"]\n'
        'keys=("BB_THREAD_ID","BB_PROJECT_ID","BB_ENVIRONMENT_ID","BB_CLI","BB_THREAD_STORAGE")\n'
        'print(json.dumps({k:os.environ.get(k) for k in keys}|{"cwd":os.getcwd()}))\n')
    cli.chmod(0o700)
    def context(label):
        return {'PATH': str(path) + os.pathsep + '/usr/bin' + os.pathsep + '/bin', 'BB_CLI': str(cli),
                'BB_THREAD_ID': 'thr_' + label, 'BB_PROJECT_ID': 'proj_' + label,
                'BB_ENVIRONMENT_ID': 'env_' + label, 'BB_HOST_DAEMON_PORT': '9',
                'BB_SERVER_URL': 'http://127.0.0.1:9', 'BB_THREAD_STORAGE': str(tmp_path / ('storage-' + label)),
                'BB_AUTH_TOKEN': 'PRIVATE_MUST_NOT_TRAVEL'}
    return context


def editor_context(daemon, context):
    return daemon[0], daemon[1], {**daemon[2], **context}, daemon[3]


def policy(daemon, sid):
    with sqlite3.connect(f'file:{daemon[0]}/state.db?mode=ro', uri=True) as conn:
        data = conn.execute('SELECT value FROM state_meta WHERE key=?', ('gateway.local_policy.v1:' + sid,)).fetchone()
    return json.loads(data[0])['policy']


async def fixture_turn(acp, sid):
    offset = len(acp.frames)
    task = asyncio.create_task(acp.rpc('session/prompt', sessionId=sid,
        prompt=[{'type': 'text', 'text': 'Run the deterministic BB context fixture'}]))
    answered = set()
    try:
        while not task.done():
            await async_wait_until(lambda: task.done() or any(p['id'] not in answered for p in permissions(acp)),
                                   timeout=30, what='fixture result or scoped approval')
            for p in permissions(acp):
                if p['id'] not in answered:
                    answered.add(p['id'])
                    await answer(acp, p, 'allow_once')
        reply = await task
        assert reply.get('result', {}).get('stopReason') == 'end_turn', reply
        return [f['params']['update'] for f in acp.frames[offset:] if f.get('method') == 'session/update'
                and f['params']['update'].get('sessionUpdate') == 'tool_call_update']
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def native_calls(provider):
    provider.push(ToolCall('terminal', {'command': 'bb status --json'}),
        ToolCall('execute_code', {'code': 'import subprocess\nprint(subprocess.check_output(["bb","status","--json"],text=True))'}),
        Text('BB_FIXTURE_DONE'))


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_stdio_bb_native_kernel_load_model_and_fork_context_are_session_owned(daemon, tmp_path, model_peer, bb_fixture):
    config = json.loads((daemon[0] / 'config.yaml').read_text())
    config['providers'] = {'second': {'name': 'second', 'base_url': model_peer.provider.base_url,
                                     'api_key': 'loopback-only', 'default_model': 'fixture-model-two'}}
    (daemon[0] / 'config.yaml').write_text(json.dumps(config))
    context_a, context_b = bb_fixture('alpha'), bb_fixture('beta')
    native_calls(model_peer.provider)
    async with viewer(daemon) as ws:
        async with editor(editor_context(daemon, context_a), tmp_path) as acp:
            await initialize(acp)
            sid = await new(acp, tmp_path)
            result = await fixture_turn(acp, sid)
            assert len(result) == 2
            assert all('thr_alpha' in json.dumps(r) and 'proj_alpha' in json.dumps(r) for r in result), result
            assert 'PRIVATE_MUST_NOT_TRAVEL' not in json.dumps(policy(daemon, sid))
            assert 'BB_THREAD_ID' not in json.dumps(await ws.rpc('session.info', session_id=sid))
        # A new BB child attaches to the same logical session: the private env
        # mutation must drop its cached shell/kernel rather than retain alpha.
        async with editor(editor_context(daemon, context_b), tmp_path) as acp:
            await initialize(acp)
            loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
            assert 'result' in loaded, loaded
            picked = await acp.rpc('session/set_model', sessionId=sid, modelId='custom:second:fixture-model-two')
            assert 'result' in picked, picked
            native_calls(model_peer.provider)
            result = await fixture_turn(acp, sid)
            print('BB_REBOUND_OUTPUT=' + json.dumps(result))
            assert all('thr_beta' in json.dumps(r) and 'thr_alpha' not in json.dumps(r) for r in result), result
            fork = await acp.rpc('session/fork', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
            assert 'result' in fork, fork
            child = fork['result']['sessionId']
        async with editor(editor_context(daemon, bb_fixture('child')), tmp_path) as acp:
            await initialize(acp)
            assert 'result' in await acp.rpc('session/load', sessionId=child, cwd=str(tmp_path), mcpServers=[])
            native_calls(model_peer.provider)
            result = await fixture_turn(acp, child)
            assert all('thr_child' in json.dumps(r) for r in result), result
        assert json.loads(policy(daemon, sid)['client_env_json'])['BB_THREAD_ID'] == 'thr_beta'
        assert json.loads(policy(daemon, child)['client_env_json'])['BB_THREAD_ID'] == 'thr_child'
        # Ordinary session resume/history never exposes the private client map.
        info = await ws.rpc('session.info', session_id=child)
        assert 'client_env' not in info.get('launch_request', {})
        print('BB_STDIO_CONTEXT=' + json.dumps({'parent': sid, 'child': child, 'native_and_kernel': True}))


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_null_map_load_and_explicit_bb_like_mcp_metadata_survive_replacement(daemon, tmp_path, model_peer, bb_fixture):
    server = tmp_path / 'bb_mcp.py'
    server.write_text('''import json,subprocess,sys
for line in sys.stdin:
 r=json.loads(line); ident=r.get('id'); method=r.get('method')
 if ident is None: continue
 if method=='initialize': result={'protocolVersion':r['params']['protocolVersion'],'capabilities':{'tools':{}},'serverInfo':{'name':'bb-like-fixture','version':'1'}}
 elif method=='tools/list': result={'tools':[{'name':'status','description':'owned BB-like status','inputSchema':{'type':'object','properties':{}}}]}
 elif method=='tools/call': result={'content':[{'type':'text','text':subprocess.check_output(['bb','status','--json'],text=True)}]}
 else: result={}
 print(json.dumps({'jsonrpc':'2.0','id':ident,'result':result}),flush=True)
''')
    def spec(context):
        return {'name': 'bb-context-fixture', 'command': sys.executable, 'args': [str(server)],
                'env': [{'name': k, 'value': v} for k, v in context.items() if k != 'BB_AUTH_TOKEN']}
    async with viewer(daemon) as ws:
        created = await ws.rpc('session.create', request_id='old-null-env', source='acp', cwd=str(tmp_path))
        sid = created['session_id']
        assert policy(daemon, sid)['client_env_json'] is None
        for label in ('first', 'replacement'):
            context = bb_fixture(label)
            async with editor(editor_context(daemon, context), tmp_path) as acp:
                await initialize(acp)
                loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[spec(context)])
                assert 'result' in loaded, loaded
                p = policy(daemon, sid)
                assert json.loads(p['client_env_json'])['BB_THREAD_ID'] == 'thr_' + label
                target = next(iter(json.loads(p['editor_mcp_json'])['schemas']))
                model_peer.provider.push(ToolCall('tool_describe', {'names': [target]}),
                    ToolCall('tool_call', {'name': target, 'arguments': {}}), Text('MCP_CONTEXT_DONE'))
                result = await fixture_turn(acp, sid)
                assert 'thr_' + label in json.dumps(result), result
                snapshot = await ws.rpc('session.resume', session_id=sid)
                assert 'MCP_CONTEXT_DONE' in json.dumps(snapshot['messages'])
        print('BB_LIKE_MCP_CONTEXT=' + json.dumps({'session': sid, 'explicit_server_env': True, 'real_bb_mcp': False}))


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_busy_context_change_refuses_but_identical_reconnect_is_noop(daemon, tmp_path, model_peer, bb_fixture):
    context = bb_fixture('busy')
    target = tmp_path / 'protected-fixture'
    target.mkdir()
    model_peer.provider.push(ToolCall('terminal', {'command': 'rm -r -- ' + shlex.quote(str(target))}), Text('BUSY_DONE'))
    async with viewer(daemon) as ws, editor(editor_context(daemon, context), tmp_path) as acp:
        await initialize(acp)
        sid = await new(acp, tmp_path)
        task = asyncio.create_task(acp.rpc('session/prompt', sessionId=sid,
            prompt=[{'type': 'text', 'text': 'Run the held fixture'}]))
        try:
            await async_wait_until(lambda: bool(permissions(acp)), timeout=25, what='held fixture approval')
            snapshot = await ws.rpc('session.resume', session_id=sid)
            fields = {'session_id': sid, 'operation': 'client_env', 'expected_revision': snapshot['revision'],
                      'expected_generation': snapshot['execution_generation']}
            same = await ws.rpc('session.mutate', request_id='busy-noop', payload={'client_env': context}, **fields)
            assert same['status'] == 'noop'
            from hermes_cli.gateway_client import GatewayRPCError
            with pytest.raises(GatewayRPCError, match='session_busy'):
                await ws.rpc('session.mutate', request_id='busy-change', payload={'client_env': bb_fixture('intruder')}, **fields)
            assert json.loads(policy(daemon, sid)['client_env_json'])['BB_THREAD_ID'] == 'thr_busy'
            await answer(acp, permissions(acp)[0], 'deny')
            assert (await task).get('result', {}).get('stopReason') == 'end_turn'
            assert target.exists()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
