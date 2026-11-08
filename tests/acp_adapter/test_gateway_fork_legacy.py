"""Legacy maintained direct-ACP rows keep their explicit effort on gateway adoption."""
import json

import pytest

from tests.acp_adapter.test_gateway_fork_mcp import consent_server, run_consent
from tests.acp_adapter.test_gateway_fork_wire import initialize
from tests.acp_adapter.test_gateway_sessions import editor, viewer


@pytest.mark.parametrize('source,explicit,generic,expected', [
    ('acp', 'none', {'enabled': True, 'effort': 'high'}, 'none'),
    ('acp', 'high', {'enabled': False}, 'high'),
    ('acp', 'invalid', {'enabled': True, 'effort': 'low'}, 'low'),
    ('acp', 42, 'medium', 'medium'),
    ('acp', None, {'enabled': False}, 'none'),
    ('cli', 'none', {'enabled': True, 'effort': 'high'}, 'high'),
])
def test_legacy_acp_effort_precedence_and_invalid_non_acp_fallback(source, explicit, generic, expected, tmp_path):
    from gateway.session_local_migration import legacy_launch_params
    params, _ = legacy_launch_params({'source': source, 'model': 'fixture', 'cwd': str(tmp_path),
        'model_config': json.dumps({'acp_reasoning_effort': explicit, 'reasoning_config': generic})})
    assert params['reasoning'] == expected


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('effort', ['none', 'high'])
@pytest.mark.parametrize('daemon', [{'providers': {'second': {'name': 'second',
    'base_url': 'http://127.0.0.1:9/v1', 'api_key': 'loopback-only', 'default_model': 'legacy-model'}}}], indirect=True)
async def test_default_load_adopts_old_direct_row_preserving_effort_provider_cwd_and_mcp(
        daemon, tmp_path, model_peer, consent_server, effort):
    from hermes_state import SessionDB
    home = daemon[0]
    base_url = f'http://127.0.0.1:{model_peer.server_port}/v1'
    config = json.loads((home / 'config.yaml').read_text())
    config['providers']['second']['base_url'] = base_url
    (home / 'config.yaml').write_text(json.dumps(config))
    sid = 'legacy-maintained-acp'
    # The old maintained SessionManager format, not a new session policy or
    # copied live state. Generic/effective effort deliberately disagrees.
    with SessionDB(db_path=home / 'state.db') as db:
        db.create_session(session_id=sid, source='acp', model='legacy-model', model_config={
            'cwd': str(tmp_path), 'provider': 'custom:second', 'base_url': base_url,
            'api_mode': 'chat_completions', 'acp_reasoning_effort': effort,
            'reasoning_config': {'enabled': True, 'effort': 'low'}})
        db.append_message(sid, 'user', 'OLD_DIRECT_HISTORY')
        db.append_message(sid, 'assistant', 'OLD_DIRECT_REPLY')
    async with viewer(daemon) as ws, editor(daemon, tmp_path) as acp:
        await initialize(acp)
        loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[consent_server])
        assert 'result' in loaded, loaded
        info = await ws.rpc('session.info', session_id=sid)
        assert (info['provider'], info['model'], info['cwd'], info['reasoning_effort']) == (
            'custom:second', 'legacy-model', str(tmp_path), effort), info
        assert 'OLD_DIRECT_HISTORY' in json.dumps(acp.frames)
        await run_consent(acp, ws, sid)
        assert model_peer.requests[-1]['model'] == 'legacy-model'
        if effort == 'high':
            assert model_peer.requests[-1].get('reasoning_effort') == 'high'
        else:
            assert model_peer.requests[-1].get('reasoning_effort') in ('none', None)
        print('FORK_LEGACY_ADOPTION=' + json.dumps({'session_id': sid, 'provider': info['provider'],
              'effort': info['reasoning_effort'], 'cwd': info['cwd'], 'mcp_invoked': True}))
