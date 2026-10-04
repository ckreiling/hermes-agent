"""Custom Exa REST gateways must not route through the public keyless MCP."""
from types import SimpleNamespace
import sys

import pytest

from plugins.web.exa import provider as exa


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    values = {}
    monkeypatch.setattr(exa, "provider_env", lambda name: values.get(name, ""))
    monkeypatch.setattr(exa, "lazy_ensure", lambda feature: None)
    return values


@pytest.mark.parametrize("url", ["https://exa.int.exe.xyz/", "http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000"])
def test_valid_endpoints(isolated_env, url):
    isolated_env["EXA_BASE_URL"] = url
    assert exa._custom_base_url() == url.rstrip("/")
    assert exa.ExaWebSearchProvider().is_available()


@pytest.mark.parametrize("url", ["http://exa.int.exe.xyz", "file:///tmp/exa", "https://user:secret@example.com", "https://example.com?key=secret", "https://example.com#fragment", "https://"])
def test_invalid_endpoints_fail_closed(isolated_env, url):
    isolated_env["EXA_BASE_URL"] = url
    result = exa.ExaWebSearchProvider().search("test")
    assert not result["success"]
    assert "EXA_BASE_URL" in result["error"]
    assert "secret" not in result["error"]


@pytest.mark.parametrize("key", ["", "vendor-key"])
def test_custom_sdk_and_both_operations(monkeypatch, isolated_env, key):
    isolated_env.update(EXA_BASE_URL="https://exa.int.exe.xyz/", EXA_API_KEY=key)
    clients = []
    hit = SimpleNamespace(url="https://example.com", title="Example", highlights=["snippet"], text="page text")

    class FakeExa:
        def __init__(self, **kwargs):
            self.headers = {}
            clients.append(kwargs)

        def search(self, query, **kwargs):
            assert kwargs == {"num_results": 5, "contents": {"highlights": True}}
            return SimpleNamespace(results=[hit])

        def get_contents(self, urls, **kwargs):
            assert kwargs == {"text": True}
            return SimpleNamespace(results=[hit])

    monkeypatch.setitem(sys.modules, "exa_py", SimpleNamespace(Exa=FakeExa))
    monkeypatch.setattr(exa, "use_keyless", lambda *args: pytest.fail("custom endpoint used keyless routing"))
    provider = exa.ExaWebSearchProvider()
    assert provider.search("test")["data"]["web"][0]["description"] == "snippet"
    assert provider.extract([hit.url])[0]["content"] == "page text"
    assert clients == [{"api_key": key or "implicit", "base_url": "https://exa.int.exe.xyz"}] * 2


def test_custom_endpoint_tool_availability(monkeypatch):
    from tools import web_tools
    monkeypatch.setattr(web_tools, "_has_env", lambda name: name == "EXA_BASE_URL")
    assert web_tools._BUILTIN_AVAILABILITY["exa"]()


def test_default_keyless_unchanged(monkeypatch):
    monkeypatch.setattr(exa, "use_keyless", lambda *args: True)
    monkeypatch.setattr(exa, "keyless_search", lambda *args: {"success": True, "data": {"web": []}})
    assert exa.ExaWebSearchProvider().search("test")["success"]
