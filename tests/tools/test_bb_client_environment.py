"""BB control metadata is exact-name, session-local and never a daemon export."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.session_client_env import (BB_ENV, accept_client_env, client_env_scope, collect_client_env,
                                      session_env, toolchain_environ)


def test_exact_bb_names_and_configured_toolchain_keep_secret_rejection():
    raw = {k: 'fixture-' + k for k in BB_ENV} | {'PATH': '/fixture/bin', 'BB_AUTH_TOKEN': 'PRIVATE_DROP',
        'BB_UNKNOWN': 'DROP', 'NODE_AUTH_TOKEN': 'DROP', 'OPENAI_API_KEY': 'DROP', 'MY_CHAIN': 'keep'}
    config = {'terminal': {'client_env_passthrough': ['MY_CHAIN', 'BB_AUTH_TOKEN', 'NODE_AUTH_TOKEN']}}
    expected = set(BB_ENV) | {'PATH', 'MY_CHAIN'}
    assert set(collect_client_env(raw, config)) == expected
    assert set(accept_client_env(raw, config)) == expected
    assert not set(BB_ENV) & set(accept_client_env(raw, config, local=False))


@pytest.mark.parametrize('values,local', [(None, True), ({}, True), ({k: 'spoof' for k in BB_ENV}, False)])
def test_no_map_empty_or_remote_gateway_scope_masks_daemon_bb_only(monkeypatch, values, local):
    for name in BB_ENV:
        monkeypatch.setenv(name, 'daemon-' + name)
    base = dict(os.environ)
    with client_env_scope(values, local=local):
        assert all(session_env(k) == '' for k in BB_ENV)
        env = toolchain_environ()
        assert not set(BB_ENV) & set(env)
        assert env['PATH'] == base['PATH']
    assert dict(os.environ) == base
    assert session_env('BB_THREAD_ID') == 'daemon-BB_THREAD_ID'  # direct fallback restored


@pytest.mark.parametrize('value', [None, [], {'PATH': 42}, {'PATH': 'x' * (256 * 1024 + 1)},
                                   {str(i): 'x' for i in range(257)}])
def test_malformed_or_oversized_environment_refuses_without_values(value):
    with pytest.raises(ValueError, match='client_env must be a map') as exc:
        accept_client_env(value, {})
    assert 'PRIVATE' not in str(exc.value)


def test_two_overlapping_contexts_reach_native_background_and_kernel_envs_without_global_writes(monkeypatch, tmp_path):
    from tools.environments.local import _make_run_env
    from tools.process_registry import ProcessRegistry
    from tools.code_execution_env import _build_child_env
    for name in BB_ENV:
        monkeypatch.setenv(name, 'daemon-' + name)
    baseline = dict(os.environ)
    barrier = Barrier(2)
    command = [sys.executable, '-c', 'import os,json; print(json.dumps({k:v for k,v in os.environ.items() if k.startswith("BB_")}))']

    def run(label):
        values = {k: label + '-' + k for k in BB_ENV}
        with client_env_scope(values):
            barrier.wait(timeout=10)
            envs = [_make_run_env({}), ProcessRegistry._spawn_env({}),
                    _build_child_env(rpc_endpoint='fixture', rpc_token='fixture', tmpdir=str(tmp_path), child_python=sys.executable)]
            for env in envs:
                got = json.loads(subprocess.check_output(command, env=env, text=True, timeout=10))
                assert got == values
            assert session_env('BB_THREAD_ID') == values['BB_THREAD_ID']
        return label

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run, ['alpha', 'beta'])) == ['alpha', 'beta']
    assert dict(os.environ) == baseline


@pytest.mark.asyncio
async def test_overlapping_cached_native_terminals_never_exchange_bb_identity(tmp_path, monkeypatch):
    import asyncio
    from gateway.session_local import create_local_session
    from gateway.session_policy import policy_scope, restore_policy
    from gateway.session_context import set_session_vars, clear_session_vars
    from hermes_state_local import local_receipt
    from tests.tools.test_session_tool_env import _owner
    from tools.terminal_tool import terminal_tool
    home = tmp_path / 'owned-home'
    home.mkdir()
    (home / 'config.yaml').write_text(json.dumps({'platform_toolsets': {'acp': []},
        'model': {'provider': 'custom', 'default': 'fixture', 'base_url': 'http://127.0.0.1:9/v1'}}))
    bins = tmp_path / 'bin'
    bins.mkdir()
    cli = bins / 'bb'
    cli.write_text('#!/bin/sh\nprintf "%s\\n" "$BB_THREAD_ID"\n')
    cli.chmod(0o700)
    monkeypatch.setenv('BB_THREAD_ID', 'daemon-other')
    owner, connection = await _owner(home, monkeypatch)
    try:
        policies = {}
        for label in ('A', 'B'):
            ref = create_local_session(owner, connection.actor, {'source': 'acp', 'request_id': label,
                'cwd': str(home), 'toolsets': [], 'client_env': {'BB_THREAD_ID': label, 'BB_CLI': str(cli),
                'PATH': str(bins) + ':/usr/bin:/bin'}}, client_local=True)
            policies[label] = restore_policy(local_receipt(owner.db, ref.session_id)['policy'])
        barrier = Barrier(2)
        def run(label):
            tokens = set_session_vars(platform='local', session_key='fixture-' + label, session_id=label)
            try:
                with policy_scope(policies[label], authority=owner):
                    first = json.loads(terminal_tool('bb status --json', task_id='same-task'))['output']
                    barrier.wait(timeout=20)
                    second = json.loads(terminal_tool('bb status --json', task_id='same-task'))['output']
                    assert first.strip().endswith(label) and second.strip().endswith(label), (first, second)
            finally:
                clear_session_vars(tokens)
        def overlap():
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(run, ['A', 'B']))
        await asyncio.to_thread(overlap)
        assert os.environ['BB_THREAD_ID'] == 'daemon-other'
    finally:
        from tools.terminal_tool_lifecycle import cleanup_all_environments
        cleanup_all_environments()
        await connection.close()
        owner.db.close()


@pytest.mark.parametrize('env,ambient,suppressed', [({'BB_THREAD_ID': 'thread-A'}, '', True), ({}, 'daemon-other', False),
                                                   (None, 'daemon-other', False)])
def test_acp_title_guard_reads_its_session_not_daemon(monkeypatch, env, ambient, suppressed):
    from agent.turn_context import _maybe_title_session_at_turn_start
    monkeypatch.setenv('BB_THREAD_ID', ambient)
    agent = SimpleNamespace(platform='acp', _session_db=Mock(), session_id='fixture', _session_db_created=True)
    title = Mock()
    monkeypatch.setattr('agent.title_generator.maybe_auto_title', title)
    with client_env_scope(env):
        _maybe_title_session_at_turn_start(agent, [{'role': 'user', 'content': 'Real user question'}])
    assert title.called is not suppressed
