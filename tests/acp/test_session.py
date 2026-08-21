"""Tests for acp_adapter.session — SessionManager and SessionState."""

import contextlib
import io
import json
import time
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, patch

from acp_adapter import session as acp_session
from acp_adapter.session import SessionManager, SessionState
from hermes_state import SessionDB


def _mock_agent():
    return MagicMock(name="MockAIAgent")


@pytest.fixture()
def manager():
    """SessionManager with a mock agent factory (avoids needing API keys)."""
    return SessionManager(agent_factory=_mock_agent)


# ---------------------------------------------------------------------------
# create / get
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_create_session_returns_state(self, manager):
        state = manager.create_session(cwd="/tmp/work")
        assert isinstance(state, SessionState)
        assert state.cwd == "/tmp/work"
        assert state.session_id
        assert state.history == []
        assert state.agent is not None



    def test_register_task_cwd_translates_windows_drive_for_wsl_tools(self, monkeypatch):
        captured = {}

        def fake_register_task_env_overrides(task_id, overrides):
            captured["task_id"] = task_id
            captured["overrides"] = overrides

        monkeypatch.setattr("hermes_constants._wsl_detected", True)
        monkeypatch.setattr(
            "tools.terminal_tool.register_task_env_overrides",
            fake_register_task_env_overrides,
        )

        acp_session._register_task_cwd("session-1", r"E:\Projects\AI\paperclip")

        assert captured == {
            "task_id": "session-1",
            "overrides": {"cwd": "/mnt/e/Projects/AI/paperclip"},
        }


    def test_get_session(self, manager):
        state = manager.create_session()
        fetched = manager.get_session(state.session_id)
        assert fetched is state


    def test_make_agent_stamps_session_cwd_for_codex_runtime(self, monkeypatch):
        class FakeAgent:
            model = "fake-model"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(
            "acp_adapter.session.load_config",
            lambda: {
                "model": {
                    "default": "fake-model",
                    "provider": "fake-provider",
                },
                "mcp_servers": {},
            },
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "model": {
                    "default": "fake-model",
                    "provider": "fake-provider",
                },
                "mcp_servers": {},
            },
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None: {
                "provider": requested,
                "api_mode": "codex_app_server",
                "base_url": "https://example.invalid",
                "api_key": "test-key",
            },
        )
        monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)

        state = SessionManager(db=None).create_session(cwd="/tmp/project")

        assert state.agent.session_cwd == "/tmp/project"




# ---------------------------------------------------------------------------
# WSL cwd translation
# ---------------------------------------------------------------------------


class TestWslCwdTranslation:
    def test_translate_acp_cwd_converts_windows_drive_path_when_wsl(self, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)

        assert acp_session._translate_acp_cwd(r"E:\Projects\AI\paperclip") == "/mnt/e/Projects/AI/paperclip"





    def test_fork_session_stores_translated_cwd_on_wsl(self, manager, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)
        original = manager.create_session(cwd="/tmp/base")

        forked = manager.fork_session(original.session_id, cwd=r"D:\work\project")

        assert forked is not None
        assert forked.cwd == "/mnt/d/work/project"

    def test_update_cwd_stores_translated_cwd_on_wsl(self, manager, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)
        state = manager.create_session(cwd="/tmp/old")

        updated = manager.update_cwd(state.session_id, cwd=r"C:\Users\foo\project")

        assert updated is not None
        assert updated.cwd == "/mnt/c/Users/foo/project"

# ---------------------------------------------------------------------------
# fork
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# list / cleanup / remove
# ---------------------------------------------------------------------------


class TestSymlinkAliasNormalization:
    """Ported from PrimeIntellect-ai/prime-agent#628 — symlink aliases of the
    same directory (macOS ``/var`` vs ``/private/var``, ``/tmp`` vs
    ``/private/tmp``) must compare equal, or ACP history filters silently drop
    a workspace's own sessions."""

    def test_symlink_alias_compares_equal(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(real)
        assert acp_session._normalize_cwd_for_compare(
            str(alias)
        ) == acp_session._normalize_cwd_for_compare(str(real))

    def test_distinct_dirs_still_compare_different(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        assert acp_session._normalize_cwd_for_compare(
            str(a)
        ) != acp_session._normalize_cwd_for_compare(str(b))

    def test_missing_path_keeps_lexical_normalization(self):
        # realpath(strict=False) is lexical for nonexistent paths, so cwds
        # that don't exist on this host (e.g. WSL-translated drives) behave
        # exactly as the old normpath comparison did.
        assert acp_session._normalize_cwd_for_compare(
            "/nonexistent-hermes-test/x/../y"
        ) == "/nonexistent-hermes-test/y"

    def test_list_sessions_matches_symlink_alias_cwd(self, manager, tmp_path):
        real = tmp_path / "proj"
        real.mkdir()
        alias = tmp_path / "link"
        alias.symlink_to(real)
        state = manager.create_session(cwd=str(real))
        state.history.append({"role": "user", "content": "hello"})
        listed = manager.list_sessions(cwd=str(alias))
        assert [s["session_id"] for s in listed] == [state.session_id]


# ---------------------------------------------------------------------------
# list / cleanup
# ---------------------------------------------------------------------------


class TestListAndCleanup:
    def test_list_sessions_empty(self, manager):
        assert manager.list_sessions() == []



    def test_save_session_preserves_existing_messages_on_encode_failure(self, manager):
        """Regression for #13675: a bad message in state.history must not
        clobber the previously-persisted transcript.  replace_messages()
        wraps DELETE + INSERT in a single rolled-back-on-exception txn.
        """
        state = manager.create_session()
        state.history.append({"role": "user", "content": "original"})
        manager.save_session(state.session_id)

        # Now swap history with a message whose tool_calls is non-JSON-serializable.
        # _execute_write rolls back; the previously persisted "original" stays.
        state.history = [
            {"role": "user", "content": "replacement"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"bad": object()}],
            },
        ]
        manager.save_session(state.session_id)

        db = manager._get_db()
        messages = db.get_messages_as_conversation(state.session_id)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "original"
        assert isinstance(messages[0].get("timestamp"), (int, float))




    def test_cleanup_clears_all(self, manager):
        s1 = manager.create_session()
        s2 = manager.create_session()
        s1.history.append({"role": "user", "content": "one"})
        s2.history.append({"role": "user", "content": "two"})
        assert len(manager.list_sessions()) == 2
        manager.cleanup()
        assert manager.list_sessions() == []

    def test_remove_session(self, manager):
        state = manager.create_session()
        assert manager.remove_session(state.session_id) is True
        assert manager.get_session(state.session_id) is None
        # Removing again returns False
        assert manager.remove_session(state.session_id) is False


# ---------------------------------------------------------------------------
# persistence — sessions survive process restarts (via SessionDB)
# ---------------------------------------------------------------------------


class TestPersistence:
    """Verify that sessions are persisted to SessionDB and can be restored."""














    def test_only_restores_acp_sessions(self, manager):
        """get_session should not restore non-ACP sessions from DB."""
        db = manager._get_db()
        # Manually create a CLI session in the DB.
        db.create_session(session_id="cli-session-123", source="cli", model="test")
        # Should not be found via ACP SessionManager.
        assert manager.get_session("cli-session-123") is None

    def test_sessions_searchable_via_fts(self, manager):
        """ACP sessions stored in SessionDB are searchable via FTS5."""
        state = manager.create_session()
        state.history.append({"role": "user", "content": "how do I configure nginx"})
        state.history.append({"role": "assistant", "content": "Here is the nginx config..."})
        manager.save_session(state.session_id)

        db = manager._get_db()
        results = db.search_messages("nginx")
        assert len(results) > 0
        session_ids = {r["session_id"] for r in results}
        assert state.session_id in session_ids


    def test_assistant_reasoning_fields_persisted(self, manager):
        """ACP session restore should preserve assistant reasoning context."""
        state = manager.create_session()
        state.history.append({
            "role": "assistant",
            "content": "hello",
            "reasoning": "step-by-step",
            "reasoning_details": [
                {"type": "thinking", "thinking": "first thought"},
            ],
            "codex_reasoning_items": [
                {"type": "reasoning", "id": "rs_123", "encrypted_content": "enc_blob"},
            ],
        })
        manager.save_session(state.session_id)

        with manager._lock:
            del manager._sessions[state.session_id]

        restored = manager.get_session(state.session_id)
        assert restored is not None
        msg = restored.history[0]
        assert isinstance(msg.pop("timestamp", None), (int, float))
        # Load-time durability stamp (#92231): rows materialized from the DB
        # are marked persisted so a later flush can't re-append them.
        assert msg.pop("_db_persisted", None) is True
        assert restored.history == [{
            "role": "assistant",
            "content": "hello",
            "reasoning": "step-by-step",
            "reasoning_details": [
                {"type": "thinking", "thinking": "first thought"},
            ],
            "codex_reasoning_items": [
                {"type": "reasoning", "id": "rs_123", "encrypted_content": "enc_blob"},
            ],
        }]


    def test_acp_agents_route_human_output_to_stderr(self, tmp_path, monkeypatch):
        """ACP agents must keep stdout clean for JSON-RPC stdio transport."""

        def fake_resolve_runtime_provider(requested=None, **kwargs):
            return {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.example/v1",
                "api_key": "test-key",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            return SimpleNamespace(model=kwargs.get("model"), _print_fn=None)

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": "openrouter", "default": "test-model"}
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        db = SessionDB(tmp_path / "state.db")

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            manager = SessionManager(db=db)
            state = manager.create_session(cwd="/work")

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            state.agent._print_fn("ACP noise")

        assert stdout_buf.getvalue() == ""
        assert stderr_buf.getvalue() == "ACP noise\n"


# ---------------------------------------------------------------------------
# persist/restore provider round-trip — regression for the un-restorable
# persisted-session bug: _persist stored the *normalized* provider ("custom")
# instead of the named provider ("custom:exe-llm"), so _restore could never
# resolve the named custom provider's API key again and every restore raised
# "No LLM provider configured".
# ---------------------------------------------------------------------------


NAMED_PROVIDER = "custom:exe-llm"


def _fake_resolve_runtime_provider(requested=None, **kwargs):
    """Only the *named* provider resolves credentials, mirroring production."""
    if requested == NAMED_PROVIDER:
        return {
            "provider": "custom",
            "requested_provider": NAMED_PROVIDER,
            "api_mode": "codex_responses",
            "base_url": "https://llm.example/v1",
            "api_key": "named-key",
            "command": None,
            "args": [],
        }
    # Bare "custom" / unknown providers fall through to the generic
    # env/config path: no API key.
    return {
        "provider": "custom",
        "requested_provider": requested or "custom",
        "api_mode": "chat_completions",
        "base_url": "https://openrouter.example/v1",
        "api_key": None,
        "command": None,
        "args": [],
    }


class _FakeAIAgent:
    """Mimics AIAgent's provider attributes + its no-credential failure."""

    def __init__(self, **kwargs):
        if not kwargs.get("api_key"):
            raise RuntimeError(
                "No LLM provider configured. Run `hermes model` ..."
            )
        self.kwargs = kwargs
        self.model = kwargs.get("model") or ""
        self.provider = kwargs.get("provider") or ""
        self.requested_provider = kwargs.get("requested_provider") or self.provider
        self.base_url = kwargs.get("base_url") or ""
        self.api_mode = kwargs.get("api_mode") or ""
        self._print_fn = None


@pytest.fixture()
def provider_env(tmp_path, monkeypatch):
    """Real _make_agent path (no agent_factory) with a named custom provider."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {"default": "test-model", "provider": NAMED_PROVIDER},
            "mcp_servers": {},
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _fake_resolve_runtime_provider,
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **kwargs: None,
    )
    db = SessionDB(tmp_path / "state.db")
    with patch("run_agent.AIAgent", _FakeAIAgent):
        yield SessionManager(db=db)


class TestProviderRoundTrip:
    def test_real_config_resolution_round_trips_named_provider(self, tmp_path, monkeypatch):
        """Exercise config.yaml -> resolver -> ACP persistence -> resolver."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            """\
model:
  default: test-model
  provider: custom:exe-llm
custom_providers:
  - name: exe-llm
    base_url: https://llm.example/v1
    api_key: named-key
    api_mode: codex_responses
""",
            encoding="utf-8",
        )
        from hermes_cli.config import load_config

        # An earlier test imports runtime_provider while config.load_config is
        # monkeypatched; restore the real loader that the resolver imported.
        monkeypatch.setattr("hermes_cli.runtime_provider.load_config", load_config)
        monkeypatch.setattr(
            "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
            lambda **kwargs: None,
        )
        manager = SessionManager(db=SessionDB(tmp_path / "state.db"))

        with patch("run_agent.AIAgent", _FakeAIAgent):
            state = manager.create_session(cwd="/work")
            state.history.append({"role": "user", "content": "hello"})
            manager.save_session(state.session_id)
            with manager._lock:
                manager._sessions.clear()
            restored = manager.get_session(state.session_id)

        assert restored is not None
        assert restored.agent.kwargs["api_key"] == "named-key"
        assert restored.agent.requested_provider == NAMED_PROVIDER
        assert [message["content"] for message in restored.history] == ["hello"]

    def test_persist_stores_named_provider_not_normalized(self, provider_env):
        manager = provider_env
        state = manager.create_session(cwd="/work")

        row = manager._get_db().get_session(state.session_id)
        meta = json.loads(row["model_config"])
        assert meta["provider"] == NAMED_PROVIDER

    def test_restored_session_resolves_named_provider_credentials(self, provider_env):
        manager = provider_env
        state = manager.create_session(cwd="/work")
        state.history.append({"role": "user", "content": "hello"})
        manager.save_session(state.session_id)
        session_id = state.session_id

        # Simulate a process restart: drop the in-memory session.
        with manager._lock:
            manager._sessions.clear()

        restored = manager.get_session(session_id)
        assert restored is not None
        assert restored.agent.kwargs["api_key"] == "named-key"
        assert restored.agent.requested_provider == NAMED_PROVIDER
        assert [m["content"] for m in restored.history] == ["hello"]

    def test_restore_falls_back_to_config_defaults_for_legacy_rows(self, provider_env):
        """Rows persisted before the fix carry provider='custom' (normalized).

        The first _make_agent attempt with that metadata fails (no credential
        resolvable); _restore must retry with current config defaults instead
        of returning None, because the conversation history is the valuable
        artifact.
        """
        manager = provider_env
        db = manager._get_db()
        db.create_session(
            session_id="legacy-acp-session",
            source="acp",
            model="test-model",
            model_config={
                "cwd": "/work",
                "provider": "custom",  # normalized — cannot resolve a key
                "base_url": "https://openrouter.example/v1",
                "api_mode": "chat_completions",
            },
        )
        db.replace_messages(
            "legacy-acp-session", [{"role": "user", "content": "old history"}]
        )

        restored = manager.get_session("legacy-acp-session")
        assert restored is not None
        # Fallback path: config defaults (named provider) resolved the key.
        assert restored.agent.kwargs["api_key"] == "named-key"
        assert [m["content"] for m in restored.history] == ["old history"]

    def test_restore_returns_none_when_fallback_also_fails(self, provider_env, monkeypatch):
        manager = provider_env
        db = manager._get_db()
        db.create_session(
            session_id="doomed-acp-session",
            source="acp",
            model="test-model",
            model_config={"cwd": "/work", "provider": "custom"},
        )
        # Now even the config default cannot resolve credentials.
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None, **kwargs: {
                "provider": "custom",
                "api_key": None,
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.example/v1",
                "command": None,
                "args": [],
            },
        )
        assert manager.get_session("doomed-acp-session") is None
