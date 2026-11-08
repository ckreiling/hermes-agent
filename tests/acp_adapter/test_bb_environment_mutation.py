"""Authenticated client_env receipt/fencing tests against the actual session authority."""
import json
from dataclasses import replace

import pytest

from gateway.session_contract import SessionRef
from gateway.session_local import create_local_session
from hermes_state_local import local_receipt
from hermes_state_runtime import RuntimeStoreError
from tests.tools.test_session_tool_env import _owner


async def make_owner(tmp_path, monkeypatch, source='acp'):
    home = tmp_path / 'owned'
    home.mkdir()
    (home / 'config.yaml').write_text(json.dumps({'platform_toolsets': {'acp': [], 'cli': []},
        'model': {'provider': 'custom', 'default': 'fixture', 'base_url': 'http://127.0.0.1:9/v1'}}))
    owner, connection = await _owner(home, monkeypatch)
    # Unit fixture models the verified native-ticket bit; real ticket redemption
    # is covered separately by default-stdio/socket integration.
    connection.native_owner = True
    ref = create_local_session(owner, connection.actor, {'request_id': 'old-without-env', 'source': source,
        'cwd': str(home), 'toolsets': []})
    return owner, connection, ref


def request(owner, ref, values, ident='env-update'):
    row = owner.db.get_session(ref.session_id)
    return {'session_id': ref.session_id, 'request_id': ident, 'operation': 'client_env',
            'expected_revision': row['runtime_revision'], 'expected_generation': row['runtime_generation'],
            'payload': {'client_env': values}}


@pytest.mark.asyncio
async def test_private_receipt_exact_retry_noop_and_cache_invalidation(tmp_path, monkeypatch):
    owner, connection, ref = await make_owner(tmp_path, monkeypatch)
    dropped = []
    monkeypatch.setattr('gateway.session_policy_refresh.drop_session_terminal', lambda *args: dropped.append(args[1]))
    values = {'BB_THREAD_ID': 'thr_owned', 'BB_PROJECT_ID': 'proj_owned', 'PATH': '/fixture/bin',
              'BB_AUTH_TOKEN': 'PRIVATE_DO_NOT_STORE', 'OPENAI_API_KEY': 'PRIVATE_DO_NOT_STORE', 'UNLISTED': 'DROP'}
    params = request(owner, ref, values)
    try:
        result = await connection.mutate(ref, params)
        assert result['execution_generation'] == params['expected_generation'] + 1
        assert dropped == [ref]
        policy = local_receipt(owner.db, ref.session_id)['policy']
        assert json.loads(policy['client_env_json']) == {k: values[k] for k in ('BB_THREAD_ID', 'BB_PROJECT_ID', 'PATH')}
        assert 'client_env' not in json.loads(policy['request_json'])
        assert 'PRIVATE_DO_NOT_STORE' not in json.dumps(policy)
        with owner.db._read_ctx() as conn:
            receipts = [row[0] for row in conn.execute("SELECT value FROM state_meta WHERE key GLOB 'gateway.mutation.v1.*'")]
        assert len(receipts) == 1
        assert all('thr_owned' not in text and '/fixture/bin' not in text and 'PRIVATE_DO_NOT_STORE' not in text for text in receipts)
        assert 'thr_owned' not in json.dumps(result)
        assert await connection.mutate(ref, params) == result  # lost reply: exact receipt, not a new noop
        assert len(dropped) == 2  # exact retry repairs publication/cache invalidation idempotently
        noop = await connection.mutate(ref, request(owner, ref, values, 'same-map'))
        assert noop['status'] == 'noop'
        assert noop['revision'] == result['revision']
        changed = {**params, 'payload': {'client_env': {**values, 'BB_THREAD_ID': 'other'}}}
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await connection.mutate(ref, changed)
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['publication', 'terminal-teardown'])
async def test_exact_receipt_retry_repairs_interrupted_policy_publication(tmp_path, monkeypatch, boundary):
    import gateway.session_local as local
    import gateway.session_policy_refresh as refresh
    owner, connection, ref = await make_owner(tmp_path, monkeypatch)
    target, name = (local, 'publish_local_policy') if boundary == 'publication' else (refresh, 'drop_session_terminal')
    original = getattr(target, name)
    attempts = []
    def interrupted(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError('fixture interrupted after durable commit')
        return original(*args, **kwargs)
    monkeypatch.setattr(target, name, interrupted)
    params = request(owner, ref, {'BB_THREAD_ID': 'repair-thread'})
    try:
        with pytest.raises(RuntimeError, match='fixture interrupted'):
            await connection.mutate(ref, params)
        committed = owner.db.get_session(ref.session_id)['runtime_revision']
        result = await connection.mutate(ref, params)
        assert len(attempts) == 2
        assert result['revision'] == committed
        assert json.loads(local_receipt(owner.db, ref.session_id)['policy']['client_env_json']) == {'BB_THREAD_ID': 'repair-thread'}
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_pre_bb_frozen_path_rules_gain_only_the_exact_native_bb_names(tmp_path, monkeypatch):
    from agent.session_client_env import BB_ENV
    from gateway.session_policy import build_policy, profile_config
    owner, connection, _ = await make_owner(tmp_path, monkeypatch)
    data = {'source': 'acp', 'model': 'fixture', 'cwd': str(tmp_path), 'toolsets': [], 'client_env': {'PATH': '/old/path'}}
    old = build_policy(data, profile_config(owner), client_local=True)
    rules = json.loads(old.client_env_rules_json)
    rules['names'] = [name for name in rules['names'] if name not in BB_ENV]
    old = replace(old, client_env_rules_json=json.dumps(rules))
    ref = create_local_session(owner, connection.actor, {'request_id': 'pre-bb', **data}, trusted_policy=old)
    try:
        await connection.mutate(ref, request(owner, ref, {'PATH': '/old/path', 'BB_THREAD_ID': 'new-local',
            'BB_UNLISTED': 'DROP', 'BB_AUTH_TOKEN': 'DROP'}))
        values = json.loads(local_receipt(owner.db, ref.session_id)['policy']['client_env_json'])
        assert values == {'PATH': '/old/path', 'BB_THREAD_ID': 'new-local'}
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('case,reason', [('remote', 'permission_denied'), ('readonly', 'permission_denied'),
    ('foreign', 'profile_mismatch'), ('principal', 'permission_denied'), ('revision', 'revision_conflict'), ('generation', 'stale_generation'),
    ('missing-generation', 'invalid_params'), ('wrong-source', 'invalid_params')])
async def test_client_env_refusals_are_state_preserving(tmp_path, monkeypatch, case, reason):
    owner, connection, ref = await make_owner(tmp_path, monkeypatch, source='cli' if case == 'wrong-source' else 'acp')
    before = local_receipt(owner.db, ref.session_id)
    params = request(owner, ref, {'BB_THREAD_ID': 'must-not-land'})
    if case == 'remote':
        connection.native_owner = False
    elif case == 'readonly':
        connection.actor = replace(connection.actor, capabilities=frozenset({'session:read'}))
    elif case == 'principal':
        connection.actor = replace(connection.actor, subject='different-principal')
    elif case == 'foreign':
        ref = SessionRef('foreign-profile', ref.session_id)
    elif case == 'revision':
        params['expected_revision'] += 1
    elif case == 'generation':
        params['expected_generation'] += 1
    elif case == 'missing-generation':
        del params['expected_generation']
    try:
        with pytest.raises(RuntimeStoreError, match=reason):
            await connection.mutate(ref, params)
        assert local_receipt(owner.db, ref.session_id) == before
    finally:
        # Restore profile only so teardown uses its real binding.
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [None, [], {'PATH': 5}, {'PATH': 'PRIVATE' * 50000},
                                   {str(i): 'x' for i in range(257)}])
async def test_hostile_client_environment_cannot_enter_policy_or_receipts(tmp_path, monkeypatch, value):
    owner, connection, ref = await make_owner(tmp_path, monkeypatch)
    before = local_receipt(owner.db, ref.session_id)
    try:
        with pytest.raises(RuntimeStoreError, match='invalid_params') as exc:
            await connection.mutate(ref, request(owner, ref, value))
        assert 'PRIVATE' not in str(exc.value)
        assert local_receipt(owner.db, ref.session_id) == before
        with owner.db._read_ctx() as conn:
            assert conn.execute("SELECT count(*) FROM state_meta WHERE key GLOB 'gateway.mutation.v1.*'").fetchone()[0] == 0
    finally:
        await connection.close()
        owner.db.close()
