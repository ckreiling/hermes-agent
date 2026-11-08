"""Editor overlays inherit only their current profile's MCP connections and policies."""
import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from hermes_constants import (
    hermes_home_key, pin_process_hermes_home, reset_hermes_home_override, set_hermes_home_override,
)
from tools import mcp_tool as core, mcp_tool_discovery as disc, mcp_tool_handlers as handlers
from tools import mcp_tool_registration as registration
from tools.mcp_tool_scope import _KEYED_LEDGERS, _KEYED_SETS, _resolve_server_key, _server_key
from tools.registry import registry, session_tool_scope

CFG = {"url": "https://fixture.invalid/mcp"}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from agent import secret_scope
    from hermes_constants import get_routing_process_hermes_home, process_hermes_home_is_pinned
    was_pinned = process_hermes_home_is_pinned()
    previous_home = get_routing_process_hermes_home()
    home = tmp_path / "launch"
    home.mkdir()
    pin_process_hermes_home(home)
    token = set_hermes_home_override(home)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    for attr in (*_KEYED_LEDGERS, *_KEYED_SETS, "_mcp_tool_server_names"):
        monkeypatch.setattr(core, attr, type(getattr(core, attr))())
    monkeypatch.setattr(registry, "_tools", {})
    monkeypatch.setattr(registry, "_scoped_tools", {})
    monkeypatch.setattr(disc._loop, "_ensure_mcp_loop", lambda: None)
    monkeypatch.setattr(disc._loop, "_run_on_mcp_loop", lambda call, **kw: asyncio.run(call()))
    monkeypatch.setattr(handlers, "_trust_granted", lambda *a: False)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)
        pin_process_hermes_home(previous_home if was_pinned else None)


def install(name="profile", marker="PROFILE"):
    tool = SimpleNamespace(name="echo", description="fixture", inputSchema={"type": "object"}, annotations=None)
    server = SimpleNamespace(
        name=name, session=SimpleNamespace(call_tool=AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text=marker)], isError=False))),
        _config=CFG, _tools=[tool], tool_timeout=5, initialize_result=None,
        _registered_tool_names=[], _sampling=None, _park_reason=None, _auth_park=None,
        _rpc_lock=asyncio.Lock(),
    )
    disc._adopt_server(name, server)
    server._registered_tool_names = registration._register_server_tools(name, cast(core.MCPServerTask, server), CFG)
    return server


@pytest.mark.parametrize("profile_mode", ["launch", "served", "multiplex"])
@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_profile_tool_dispatch_status_and_lazy_owner(isolated, monkeypatch, populated, lazy, profile_mode, request):
    from agent import secret_scope
    token = None
    if profile_mode != "launch":
        profile_home = isolated.parent / profile_mode
        profile_home.mkdir()
        token = set_hermes_home_override(profile_home)
    if profile_mode == "multiplex":
        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    if token is not None:
        request.addfinalizer(lambda: reset_hermes_home_override(token))
    profile = install()
    profile_key = _server_key("profile")
    profile_scope = core._mcp_registry_scope()
    connected = []
    if lazy:
        core._servers.pop(profile_key)
        core._lazy_server_configs[profile_key] = dict(CFG)
        core._lazy_server_tool_names[profile_key] = profile._registered_tool_names

        async def connect(name, config):
            connected.append(core._mcp_registry_scope())
            disc._adopt_server(name, profile)
            return profile._registered_tool_names
        monkeypatch.setattr(disc, "_discover_and_register_server", connect)
    with session_tool_scope("editor-session:a"):
        if populated:
            install("editor", "EDITOR")
        assert registry.get_entry("mcp__profile__echo") is not None
        assert handlers._make_check_fn("profile")()
        assert disc.get_mcp_status({"profile": CFG})[0]["status"] == ("lazy" if lazy else "connected")
        result = registry.dispatch("mcp__profile__echo", {})
        assert "PROFILE" in result and "error" not in result, result
        assert disc.get_mcp_status({"profile": CFG})[0]["status"] == "connected"
        assert disc.get_mcp_status({"profile": CFG}, include_runtime=False)[0]["status"] == "configured"
        if lazy:
            assert connected == [profile_scope]
            assert ("editor-session:a", "profile") not in core._servers
        assert core._servers[profile_key] is profile


def test_editor_own_and_explicit_adoption_win_and_siblings_stay_private(isolated):
    profile = install()
    with session_tool_scope("editor-session:a"):
        own = install(marker="OWN")
        install("private", "PRIVATE")
        assert _resolve_server_key("profile") == ("editor-session:a", "profile")
        assert "OWN" in registry.dispatch("mcp__profile__echo", {})
    with session_tool_scope("editor-session:b"):
        assert _resolve_server_key("profile") == "profile"
        assert disc._get_connected_server_for_call("profile") is profile
        assert registry.get_entry("mcp__private__echo") is None
        assert disc._get_connected_server_for_call("private") is None
        assert _resolve_server_key("missing") == ("editor-session:b", "missing")
        core._server_tool_scopes[("editor-session:a", "profile")].add("editor-session:b")
        assert disc._get_connected_server_for_call("profile") is own
    # An explicit maintenance lookup must not inherit the ambient editor's profile.
    with session_tool_scope("editor-session:a"):
        assert _resolve_server_key("profile", "editor-session:absent", current=False) == (
            "editor-session:absent", "profile")


def test_connecting_editor_server_does_not_borrow_profile(isolated):
    install()
    with session_tool_scope("editor-session:a"):
        key = _server_key("profile")
        core._server_connecting.add(key)
        assert _resolve_server_key("profile") == key
        assert disc._get_connected_server_for_call("profile") is None


@pytest.mark.parametrize("multiplex", [False, True])
def test_profile_overlay_never_borrows_launch_credentials(isolated, monkeypatch, multiplex):
    from agent import secret_scope
    install(marker="LAUNCH")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", multiplex)
    foreign = isolated.parent / "foreign"
    foreign.mkdir()
    token = set_hermes_home_override(foreign)
    try:
        with session_tool_scope("editor-session:a"):
            assert disc._get_connected_server_for_call("profile") is None
            assert disc.get_mcp_status({"profile": CFG})[0]["status"] == "configured"
        foreign_server = install(marker="FOREIGN")
        with session_tool_scope("editor-session:a"):
            assert _resolve_server_key("profile") == (hermes_home_key(foreign), "profile")
            assert disc._get_connected_server_for_call("profile") is foreign_server
            assert "FOREIGN" in registry.dispatch("mcp__profile__echo", {})
    finally:
        reset_hermes_home_override(token)


def test_inherited_trust_and_cost_policy_remain_the_consuming_profiles(isolated, monkeypatch):
    install()
    core._server_trust_levels["profile"] = core._TRUST_UNTRUSTED
    asked = []
    monkeypatch.setattr(handlers, "_ask_for_call", lambda *a, **kw: asked.append(a) or "DENIED")
    with session_tool_scope("editor-session:a"):
        assert handlers._trust_gate_check("profile", "echo") == "DENIED"
        assert asked
        # An explicit editor connection keeps its own full-trust policy.
        install(marker="EDITOR")
        assert handlers._trust_gate_check("profile", "echo") is None
    # A profile adopting another profile's connection retains its own policy, not its owner's.
    foreign = isolated.parent / "adopter"
    foreign.mkdir()
    token = set_hermes_home_override(foreign)
    try:
        profile_scope = hermes_home_key(foreign)
        core._server_tool_scopes.setdefault("profile", set()).add(profile_scope)
        core._server_trust_levels[(profile_scope, "profile")] = core._TRUST_UNTRUSTED
        with session_tool_scope("editor-session:b"):
            assert handlers._trust_gate_check("profile", "echo") == "DENIED"
    finally:
        reset_hermes_home_override(token)


def test_inherited_cost_gate_and_rekeyed_launch_connection(isolated, monkeypatch):
    from agent import secret_scope
    from tools import mcp_tool_costs as costs
    from tools.mcp_tool_scope import rekey_unscoped_connections
    profile = install()
    scope = hermes_home_key(isolated)
    rekey_unscoped_connections(scope)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    key = (scope, "profile")
    core._server_cost_policies[key] = object()
    charged = []
    charge = SimpleNamespace(reserve_within_limit=lambda: (True, 0))
    monkeypatch.setattr(costs, "pending_charge", lambda policy_key, *a: charged.append(policy_key) or charge)
    with session_tool_scope("editor-session:a"):
        assert disc._get_connected_server_for_call("profile") is profile
        assert _resolve_server_key("profile") == key
        booked = []
        assert handlers._trust_gate_check("profile", "echo", booked) is None
        assert charged == [key] and booked == [charge]
