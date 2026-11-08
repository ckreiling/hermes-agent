"""Real nested service consent and injected MCP continuity over default ACP/gateway.

Only the deterministic HTTP/MCP peers are fixtures; no provider inference is used.
"""
import asyncio
import json
import sys
import pytest

from tests.acp_adapter.test_gateway_editor_mcp import MCPModelPeer
from tests.acp_adapter.test_gateway_fork_wire import answer, initialize, new, permissions, updates
from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.waiting import async_wait_until


@pytest.fixture
def consent_server(tmp_path, model_peer):
    model_peer.RequestHandlerClass = MCPModelPeer
    script = tmp_path / 'consent_peer.py'
    script.write_text('''import json, sys, os
from pathlib import Path
for line in sys.stdin:
 r=json.loads(line); method=r.get('method'); ident=r.get('id')
 if ident is None: continue
 if method=='initialize': result={'protocolVersion':r['params']['protocolVersion'],'capabilities':{'tools':{}},'serverInfo':{'name':'consent-peer','version':'1'}}
 elif method=='tools/list': result={'tools':[{'name':'echo','description':'owned echo','inputSchema':{'type':'object','properties':{}}}]}
 elif method=='tools/call':
  print(json.dumps({'jsonrpc':'2.0','id':'consent','method':'elicitation/create','params':{'message':'SERVICE_CONSENT_EXACT','requestedSchema':{'type':'object','properties':{}}}}),flush=True)
  while True:
   response=json.loads(sys.stdin.readline())
   if response.get('id')=='consent': break
   if response.get('id') is not None: print(json.dumps({'jsonrpc':'2.0','id':response['id'],'result':{}}),flush=True)
  action=response.get('result',{}).get('action','missing')
  result={'content':[{'type':'text','text':'OWNED_RESULT_'+action+' CWD='+os.getcwd()}]}
 else: result={}
 print(json.dumps({'jsonrpc':'2.0','id':ident,'result':result}),flush=True)
''')
    return {'name': 'consent-fixture', 'command': sys.executable, 'args': [str(script)], 'env': []}


async def run_consent(acp, ws, sid, choice='allow_once'):
    offset = len(acp.frames)
    task = asyncio.create_task(acp.rpc('session/prompt', sessionId=sid,
                                     prompt=[{'type': 'text', 'text': 'Use owned echo'}]))
    try:
        await async_wait_until(lambda: any(p for p in acp.frames[offset:]
                              if p.get('method') == 'session/request_permission'), timeout=25,
                              what='native MCP service consent')
        permission = next(p for p in acp.frames[offset:] if p.get('method') == 'session/request_permission')
        call = permission['params']['toolCall']
        starts = [u for u in updates(acp) if u['sessionUpdate'] == 'tool_call'
                  and (u.get('title', '').startswith('mcp__')
                       or u.get('rawInput', {}).get('name', '').startswith('mcp__'))]
        outer = starts[-1]
        assert (call['toolCallId'], call['title']) == (outer['toolCallId'], outer['title']), permission
        assert 'SERVICE_CONSENT_EXACT' in json.dumps(call)
        assert not [u for u in updates(acp) if u['sessionUpdate'] == 'tool_call_update'
                    and u['toolCallId'] == outer['toolCallId']]
        await answer(acp, permission, choice)
        if choice is None:
            pending = (await ws.rpc('session.resume', session_id=sid))['prompts']
            assert len(pending) == 1
            assert pending[0]['operation']['tool_call_id'] == outer['toolCallId']
            await ws.rpc('approval.respond', session_id=sid, prompt_id=pending[0]['prompt_id'],
                         execution_generation=pending[0]['execution_generation'], choice='deny')
        result = await task
        assert result.get('result', {}).get('stopReason') == 'end_turn', result
        own = [u for u in updates(acp) if u.get('toolCallId') == outer['toolCallId']]
        assert [u['sessionUpdate'] for u in own] == ['tool_call', 'tool_call_update'], own
        expected = 'accept' if choice == 'allow_once' else 'decline'
        assert 'OWNED_RESULT_' + expected in json.dumps(acp.frames[offset:]), acp.frames[offset:]
        return outer['toolCallId']
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('choice', ['allow_once', 'deny', None])
async def test_nested_tool_call_service_consent_reuses_live_operation(daemon, consent_server, tmp_path, choice):
    async with viewer(daemon) as ws, editor(daemon, tmp_path) as acp:
        await initialize(acp)
        sid = await new(acp, tmp_path, [consent_server])
        call_id = await run_consent(acp, ws, sid, choice)
        print('FORK_SERVICE_RECEIPT=' + json.dumps({'session_id': sid, 'operation': call_id, 'choice': choice}))


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('daemon', [{'providers': {'second': {'name': 'second',
    'base_url': 'http://127.0.0.1:9/v1', 'api_key': 'loopback-only', 'default_model': 'second-model'}}}], indirect=True)
async def test_injected_mcp_survives_provider_model_switch_and_empty_load_cwd_move(
        daemon, consent_server, tmp_path, model_peer):
    home = daemon[0]
    config = json.loads((home / 'config.yaml').read_text())
    config['providers']['second']['base_url'] = f'http://127.0.0.1:{model_peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps(config))
    async with viewer(daemon) as ws:
        async with editor(daemon, tmp_path) as acp:
            await initialize(acp)
            sid = await new(acp, tmp_path, [consent_server])
            await run_consent(acp, ws, sid)
            picked = await acp.rpc('session/set_model', sessionId=sid, modelId='custom:second:local-wire-stub')
            assert 'result' in picked, picked
            await run_consent(acp, ws, sid)
            info = await ws.rpc('session.info', session_id=sid)
            assert (info['provider'], info['model']) == ('custom:second', 'local-wire-stub')
            # Same model ID on another provider stays a distinct route; then a
            # same-provider model rebuild must retain the borrowed MCP manifest.
            picked = await acp.rpc('session/set_model', sessionId=sid, modelId='custom:second:second-model')
            assert 'result' in picked, picked
            await run_consent(acp, ws, sid)
            info = await ws.rpc('session.info', session_id=sid)
            assert (info['provider'], info['model']) == ('custom:second', 'second-model')
            assert model_peer.requests[-1]['model'] == 'second-model'
        moved = tmp_path / 'other-cwd'
        moved.mkdir()
        async with editor(daemon, tmp_path) as reloaded:
            await initialize(reloaded)
            loaded = await reloaded.rpc('session/load', sessionId=sid, cwd=str(moved), mcpServers=[])
            assert 'result' in loaded, loaded
            await run_consent(reloaded, ws, sid)
            assert 'CWD=' + str(moved) in json.dumps(reloaded.frames)
            info = await ws.rpc('session.info', session_id=sid)
            assert (info['provider'], info['model'], info['cwd']) == ('custom:second', 'second-model', str(moved))
            print('FORK_MCP_CONTINUITY=' + json.dumps({'session_id': sid, 'provider': info['provider'],
                  'model': info['model'], 'cwd': info['cwd']}))
