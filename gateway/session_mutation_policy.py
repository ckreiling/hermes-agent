"""Prepare the policy-only session edits: session-scope reasoning and the workspace (Move to project).

Both rewrite the frozen LocalSessionPolicy the next turn runs under, re-bound like a model switch
(policy identity covers config/cwd, so its config-secret reference is re-fingerprinted).
"""
import asyncio
import json
from dataclasses import asdict, replace

from hermes_state_runtime import RuntimeStoreError


def _rebind(authority, prepared, old, policy, **extra):
    from gateway.session_policy import bind_launch_key, launch_key
    from gateway.session_policy_credentials import recover_config_secrets
    secrets = recover_config_secrets(authority, old) if old.config_secret_ref else {}
    policy = bind_launch_key(authority, prepared['snapshot']['receipt']['session_id'], policy,
                             launch_key(authority, old), config_secrets=secrets)
    return dict(prepared, policy=asdict(policy), **extra)


async def prepare_reasoning(authority, live, payload, prepared):
    """``/reasoning <level>`` for this session: the level ``build_policy`` freezes at creation. ``None``
    (``/reasoning reset``) gives the level back to the profile."""
    from gateway.session_policy import pin_reasoning_effort, profile_config, restore_policy, unpin_reasoning
    old = restore_policy(prepared['snapshot']['receipt']['policy'])
    frozen = old.config()
    if payload['effort'] is not None:
        pin_reasoning_effort(frozen, payload['effort'], session=True)
    else:
        # A bypass launch never reads the profile: its reset is the built-in default.
        profile = {} if old.ignore_user_config else await asyncio.to_thread(profile_config, authority)
        unpin_reasoning(frozen, profile)
    return _rebind(authority, prepared, old, replace(old, config_json=json.dumps(frozen)))


async def prepare_workspace(authority, live, payload, prepared):
    """Move to project: execution cwd and TERMINAL_CWD move together (restore_policy's invariant)."""
    import os
    from gateway.session_policy import restore_policy
    from hermes_constants import translate_cwd_for_wsl_backend
    from tui_gateway import git_probe
    old = restore_policy(prepared['snapshot']['receipt']['policy'])

    def probe():
        # Off-loop: a stat and up to three `git` spawns.
        cwd = os.path.abspath(os.path.expanduser(translate_cwd_for_wsl_backend(payload['cwd'])))
        if not os.path.isdir(cwd):
            raise RuntimeStoreError('invalid_params')
        return cwd, git_probe.branch(cwd), git_probe.common_repo_root(cwd)
    cwd, branch, root = await asyncio.to_thread(probe)
    terminal = json.loads(old.terminal_json)
    terminal['TERMINAL_CWD'] = cwd
    policy = replace(old, cwd=cwd, terminal_json=json.dumps(terminal))
    return _rebind(authority, prepared, old, policy, git={'branch': branch, 'root': root})


async def prepare_edit_policy(authority, live, payload, prepared):
    """Main's ACP modes (Default / Accept Edits / Don't Ask) as the session's frozen edit approval
    policy. Only an ACP session's edits ask an editor; selecting the current policy writes nothing."""
    from gateway.session_policy import restore_policy
    old = restore_policy(prepared['snapshot']['receipt']['policy'])
    if old.source != 'acp':
        raise RuntimeStoreError('invalid_params', field='payload.policy',
                                detail='only an ACP session has an editor edit approval policy')
    if old.edit_approval_policy == payload['policy']:
        return {'status': 'noop', 'edit_approval_policy': old.edit_approval_policy}
    return _rebind(authority, prepared, old, replace(old, edit_approval_policy=payload['policy']))


def private_client_environment(authority, ref, payload):
    """Normalize before receipts: only its fingerprint is a public mutation payload."""
    import hashlib
    from agent.session_client_env import BB_ENV, client_env_rules, normalise_client_env
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    if not isinstance(payload, dict) or set(payload) != {'client_env'}:
        raise RuntimeStoreError('invalid_params')
    saved = local_receipt(authority.db, ref.session_id)
    if not saved or saved['policy']['source'] != 'acp':
        raise RuntimeStoreError('invalid_params')
    old = restore_policy(saved['policy'])
    rules = (json.loads(old.client_env_rules_json) if old.client_env_rules_json and not old.client_env_remote
             else client_env_rules(old.config(authority, secrets=False), local=True))
    # Pre-BB ACP policies may already have a PATH map with frozen older rules.
    # Only the exact new native names are added; profile-specific grants and
    # secret rejection remain frozen, including for exact receipt retries.
    rules = {**rules, 'names': sorted(set(rules['names']) | set(BB_ENV))}
    try:
        environment = normalise_client_env(payload['client_env'], rules)
    except ValueError as exc:
        raise RuntimeStoreError('invalid_params', field='payload.client_env') from exc
    stamp = hashlib.sha256(json.dumps(environment, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'client_env': {'fingerprint': stamp}}, (environment, rules)


def client_environment_noop(authority, actor, ref, params, environment):
    """A same-map reconnect can read/reoffer an approval while busy; it writes nothing.

    Exact retries replay their receipt first, with the original revision/generation,
    never replace a successful receipt with a new no-op verdict.
    """
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_runtime import _epoch, _text, mutation_receipt_key, replay_mutation_receipt
    for value in (actor.subject, ref.session_id, params['request_id']):
        _text(value)
    if any(type(params.get(k)) is not int or params[k] < 0 for k in ('expected_revision', 'expected_generation')):
        raise RuntimeStoreError('invalid_params')
    key, digest = mutation_receipt_key(actor.subject, ref.session_id, params['request_id'],
        operation='client_env', payload=params['payload'], expected_revision=params['expected_revision'],
        expected_generation=params['expected_generation'])
    with authority.db._read_ctx() as conn:
        _epoch(conn, authority.epoch)
        replay = replay_mutation_receipt(conn, key, digest)
        if replay is not None:
            # Re-enter the canonical writer: its exact-receipt path repairs a
            # commit whose terminal/cache/policy publication was interrupted.
            return None
        row = conn.execute('SELECT runtime_revision,runtime_generation FROM sessions WHERE id=?',
                           (ref.session_id,)).fetchone()
        saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + ref.session_id,)).fetchone()
        if row is None or saved is None:
            raise RuntimeStoreError('not_found')
        policy = json.loads(saved[0])['policy']
        if policy.get('source') != 'acp':
            raise RuntimeStoreError('invalid_params')
        if (policy.get('client_env_json') is not None and not policy.get('client_env_remote')
                and json.loads(policy['client_env_json']) == environment[0]):
            if row['runtime_revision'] != params['expected_revision']:
                raise RuntimeStoreError('revision_conflict')
            if row['runtime_generation'] != params['expected_generation']:
                raise RuntimeStoreError('stale_generation')
            return {'session_id': ref.session_id, 'operation': 'client_env', 'status': 'noop',
                    'revision': row['runtime_revision'], 'execution_generation': row['runtime_generation']}
    return None


async def prepare_client_env(authority, live, payload, prepared, *, environment=None):
    from gateway.session_policy import restore_policy
    if environment is None:
        raise RuntimeStoreError('runtime_coordination_required')
    old = restore_policy(prepared['snapshot']['receipt']['policy'])
    if old.source != 'acp':
        raise RuntimeStoreError('invalid_params')
    values, rules = environment
    policy = replace(old, client_env_json=json.dumps(values, sort_keys=True), client_env_remote=False,
                     client_env_rules_json=json.dumps(rules, sort_keys=True))
    return _rebind(authority, prepared, old, policy)


PREPARE = {'reasoning': prepare_reasoning, 'workspace': prepare_workspace, 'edit_policy': prepare_edit_policy,
           'client_env': prepare_client_env}
