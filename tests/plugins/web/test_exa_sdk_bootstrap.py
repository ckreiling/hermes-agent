"""Exa gateway/keyed routes must use native SDK extra admission, not a stale alias."""

import builtins
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from plugins.web.exa import provider as exa
from pm import extras
from pm.package import InstallError
from tools import web_tools
from tools.web_result_cache import search_memo


@pytest.fixture(params=["gateway", "gateway-keyed", "vendor"])
def sdk_route(request, monkeypatch):
    values = {
        "EXA_BASE_URL": "https://exa.example.test/api/" if request.param != "vendor" else "",
        "EXA_API_KEY": "test-key" if request.param != "gateway" else "",
    }
    monkeypatch.setattr("agent.web_search_provider.get_provider_env", lambda name: values.get(name, ""))
    monkeypatch.setattr(web_tools, "_exa_client", None)
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_backend": "exa"})
    monkeypatch.setattr("agent.web_search_registry._keyless_tier_enabled", lambda: True)
    monkeypatch.setattr(exa, "use_keyless", lambda *args: False)
    search_memo.clear()
    yield values
    search_memo.clear()


@pytest.fixture
def sdk(monkeypatch):
    clients = []
    hit = SimpleNamespace(url="https://example.test/result", title="Exa result", highlights=["SDK snippet"], text="SDK page")

    class FakeExa:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.headers = {}
            self.search = Mock(return_value=SimpleNamespace(results=[hit]))
            self.get_contents = Mock(return_value=SimpleNamespace(results=[hit]))
            clients.append(self)

    monkeypatch.setitem(sys.modules, "exa_py", SimpleNamespace(Exa=FakeExa))
    # Retain real pm.ensure_import / available / anchor resolution. Any attempt
    # to sync with an installed SDK is a bug, not something this fixture hides.
    sync = Mock(side_effect=InstallError("venv", "unexpected dependency install"))
    monkeypatch.setattr("pm.client.sync_venv", sync)
    return SimpleNamespace(clients=clients, sync=sync)


def test_installed_sdk_uses_declared_extra(sdk_route, sdk):
    project = Path(exa.__file__).resolve().parents[3] / "pyproject.toml"
    declared = tomllib.loads(project.read_text())["project"]["optional-dependencies"]
    assert "exa" in declared
    assert extras.ANCHORS["exa"] == "exa_py"
    client = exa._get_exa_client()
    assert client.headers["x-exa-integration"] == "hermes-agent"
    expected = {"api_key": sdk_route["EXA_API_KEY"] or "implicit"}
    if sdk_route["EXA_BASE_URL"]:
        expected["base_url"] = sdk_route["EXA_BASE_URL"].rstrip("/")
    assert client.options == expected
    sdk.sync.assert_not_called()
    provider = exa.ExaWebSearchProvider()
    assert provider.search("SDK bootstrap")["data"]["web"][0]["description"] == "SDK snippet"
    assert provider.extract(["https://example.test/result"])[0]["content"] == "SDK page"
    if not sdk_route["EXA_BASE_URL"]:
        assert exa._get_exa_client() is client
        sdk_route["EXA_API_KEY"] = "rotated-test-key"
        assert exa._get_exa_client() is not client
        assert sdk.clients[-1].options == {"api_key": "rotated-test-key"}


def test_missing_sdk_preserves_pm_admission_failure(sdk_route, sdk, monkeypatch):
    original = extras._importable
    monkeypatch.setattr(extras, "_importable", lambda anchor: False if anchor == "exa_py" else original(anchor))
    sdk.sync.side_effect = InstallError("venv", "dependency admission refused")
    result = exa.ExaWebSearchProvider().search("missing SDK")
    assert not result["success"]
    assert "dependency admission refused" in result["error"]
    assert "Exa SDK not installed" in result["error"]
    sdk.sync.assert_called_once_with(["exa"])
    assert sdk.clients == []
    assert web_tools._exa_client is None


def test_import_failure_does_not_poison_keyed_cache(sdk_route, sdk, monkeypatch):
    original = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "exa_py":
            raise ImportError("broken SDK dependency")
        return original(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", broken_import)
        result = exa.ExaWebSearchProvider().search("broken SDK")
    assert not result["success"]
    assert "broken SDK dependency" in result["error"]
    assert web_tools._exa_client is None
    sdk.sync.assert_not_called()
    assert exa.ExaWebSearchProvider().search("recovered SDK")["success"]


def _dispatch(monkeypatch):
    provider = exa.ExaWebSearchProvider()
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr("agent.web_search_registry.get_provider", lambda name: provider if name == "exa" else None)
    return json.loads(web_tools.web_search_tool("native Exa bootstrap regression", limit=2))


def test_native_search_success_has_no_rescue(sdk_route, sdk, monkeypatch):
    ring = Mock(side_effect=AssertionError("successful Exa call must not use rescue"))
    monkeypatch.setattr("plugins.web.keyless_mcp.search_with_failover", ring)
    result = _dispatch(monkeypatch)
    assert result["success"]
    assert result["data"]["web"][0]["title"] == "Exa result"
    assert "rescued_from" not in result["data"]
    assert "backend_error" not in result["data"]
    ring.assert_not_called()
    sdk.sync.assert_not_called()


def test_genuine_sdk_failure_still_rescues_and_retries(sdk_route, sdk, monkeypatch):
    # A keyed route is eligible for the existing rescue policy. Gateway-only
    # eligibility is deliberately outside this bootstrap fix's scope.
    if not sdk_route["EXA_API_KEY"]:
        sdk_route["EXA_API_KEY"] = "test-key"
    monkeypatch.setattr("plugins.web.keyless_mcp.use_keyless", lambda *args: False)
    sdk_client = sys.modules["exa_py"].Exa

    def failing_client(**kwargs):
        client = sdk_client(**kwargs)
        client.search.side_effect = RuntimeError("upstream unavailable")
        return client

    monkeypatch.setattr(sys.modules["exa_py"], "Exa", failing_client)
    ring = Mock(return_value={"success": True, "data": {"web": [{"url": "https://rescue.example.test"}]}})
    monkeypatch.setattr("plugins.web.keyless_mcp.search_with_failover", ring)
    result = _dispatch(monkeypatch)
    assert result["success"]
    assert result["data"]["rescued_from"] == "exa"
    assert "upstream unavailable" in result["data"]["backend_error"]
    assert "unexpected dependency install" not in result["data"]["backend_error"]
    # A rescued response cannot become a cached SDK success on the next call.
    again = _dispatch(monkeypatch)
    assert again["data"]["rescued_from"] == "exa"
    assert ring.call_count == 2
    assert sum(client.search.call_count for client in sdk.clients) == 2
    sdk.sync.assert_not_called()
