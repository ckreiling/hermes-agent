"""Native selectors only; availability is synthetic, never a credential operation."""
import copy
import json
import socket

import pytest

from toolsets import _HERMES_CORE_TOOLS, TOOLSETS, resolve_toolset

CONTROL = {"browser_exec", "browser_cdp", "browser_dialog"}
VAULT = {"browser_vault"}


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No network is allowed in native vault selector unit tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_additive_partition_keeps_legacy_browser_and_platform_bundles():
    assert set(resolve_toolset("browser_control")) == CONTROL
    assert set(resolve_toolset("browser_vault")) == VAULT
    assert set(TOOLSETS["browser"]["tools"]) == CONTROL | VAULT
    assert {n for n in _HERMES_CORE_TOOLS if n.startswith("browser_")} == CONTROL | VAULT
    for bundle in ("hermes-cli", "hermes-acp"):
        assert CONTROL | VAULT <= set(resolve_toolset(bundle))


@pytest.mark.parametrize("enabled", [["browser"], ["hermes-acp"], ["browser_vault"]])
def test_legacy_disabled_browser_still_wins_over_explicit_vault(enabled):
    from model_tools import _select_tool_names

    names = _select_tool_names(enabled + ["browser_vault"], ["browser"], True)
    assert not names & (CONTROL | VAULT)


def test_partition_uses_ordinary_subtraction_without_changing_other_core_tools():
    from model_tools import _select_tool_names

    original = _select_tool_names(["hermes-acp"], [], True)
    narrowed = _select_tool_names(["hermes-acp", "browser_vault"], ["browser_control"], True)
    assert narrowed == original - CONTROL
    assert VAULT <= narrowed
    assert _select_tool_names(["browser"], ["browser_vault"], True) == CONTROL


@pytest.fixture
def native_surface(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No network, credential operation or native backend probe is allowed")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    import model_tools
    from tools import registry as registry_module
    from tools import browser_vault_tool as native
    from tools.registry import invalidate_check_fn_cache, registry

    entry = registry.get_entry("browser_vault")
    assert entry is not None
    assert entry.handler is native._handle_vault
    assert entry.check_fn is native._check_vault_available
    assert entry.schema == native.BROWSER_VAULT_SCHEMA
    assert entry.toolset == "browser"
    assert registry._plugin_owner_of(entry.handler) is None
    original_schema = copy.deepcopy(entry.schema)
    state = {"available": True}
    # Mock only the unit-test verdict boundary; do not replace the native entry,
    # handler, schema, check function or registry owner.
    monkeypatch.setattr(registry_module, "_memo_check",
                        lambda fn, memo: state["available"] if fn is entry.check_fn else False)
    for name in ("browser_vault_list", "browser_vault_unlock", "browser_vault_fill",
                 "browser_vault_save_login", "browser_vault_enter_code"):
        monkeypatch.setattr(native, name, forbidden)
    model_tools._clear_tool_defs_cache()
    invalidate_check_fn_cache()

    def definitions(disabled=("browser_control",)):
        return model_tools.get_tool_definitions(
            enabled_toolsets=["file", "browser_vault"], disabled_toolsets=list(disabled),
            quiet_mode=True, skip_tool_search_assembly=True)

    yield state, definitions, entry, native, invalidate_check_fn_cache
    assert registry.get_entry("browser_vault") is entry
    assert entry.handler is native._handle_vault
    assert entry.check_fn is native._check_vault_available
    assert entry.schema == original_schema
    assert entry.toolset == "browser"
    assert registry._plugin_owner_of(entry.handler) is None
    model_tools._clear_tool_defs_cache()
    invalidate_check_fn_cache()


def test_native_identity_shape_description_and_selection_cache(native_surface):
    state, definitions, _entry, native, invalidate = native_surface
    first = definitions()
    vault = next(d["function"] for d in first if d["function"]["name"] == "browser_vault")
    assert vault["parameters"] == native.BROWSER_VAULT_SCHEMA["parameters"]
    assert "browser_exec" not in json.dumps(first)
    assert all(d["function"]["name"] not in CONTROL for d in first)
    assert sum(d["function"]["name"] == "browser_vault" for d in first) == 1
    assert "browser_vault" not in {d["function"]["name"] for d in definitions(("browser",))}
    assert definitions() == first
    state["available"] = False
    invalidate()
    assert "browser_vault" not in {d["function"]["name"] for d in definitions()}
    state["available"] = True
    invalidate()
    assert definitions() == first


def test_progressive_discovery_cannot_reintroduce_disabled_controls(native_surface, monkeypatch):
    from tools import tool_search

    _state, definitions, _entry, native, _invalidate = native_surface
    raw = definitions()
    cfg = tool_search.ToolSearchConfig.from_raw({"enabled": "on", "defer": ["browser_vault"]})
    monkeypatch.setattr(tool_search, "load_config_readonly", lambda: cfg)
    assembled = tool_search.assemble_tool_defs(raw, context_length=32768, config=cfg)
    assert assembled.activated
    assert assembled.deferred_count == 1
    eager = {d["function"]["name"] for d in assembled.tool_defs}
    assert "browser_vault" not in eager
    assert {"tool_search", "tool_describe", "tool_call"} <= eager
    assert not eager & CONTROL
    description = json.loads(tool_search.dispatch_tool_describe(
        {"names": ["browser_vault", "browser_exec"]}, current_tool_defs=raw, config=cfg))
    assert description["tools"]["browser_vault"]["parameters"] == native.BROWSER_VAULT_SCHEMA["parameters"]
    assert "browser_exec" not in description["tools"]
    assert "browser_exec" not in json.dumps(assembled.tool_defs)
    assert "browser_exec" not in json.dumps(description["tools"])
    absent = definitions(("browser",))
    unavailable = json.loads(tool_search.dispatch_tool_describe(
        {"names": ["browser_vault"]}, current_tool_defs=absent, config=cfg))
    assert "browser_vault" not in unavailable["tools"]
