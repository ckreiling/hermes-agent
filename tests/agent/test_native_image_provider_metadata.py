"""Dynamic provider metadata must reach the main-agent image guard and Responses wire."""

from copy import deepcopy
import base64
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest

import providers
from agent import models_dev
from agent.image_routing import decide_image_input_mode
from agent.transports.codex import ResponsesApiTransport
from providers.base import ProviderProfile
from run_agent import AIAgent
from tests.agent.test_models_dev_provider_capabilities import _isolated_registry


class CatalogProfile(ProviderProfile):
    """A mixed catalog relay: only advertised model identities carry a verdict."""

    entries: dict[str, dict]
    lookups: list[bool]
    warm: bool

    def get_model_capabilities(self, model, *, allow_network=False):
        self.lookups.append(allow_network)
        if allow_network:
            self.warm = True
        if not self.warm:
            return {}
        return self.entries.get(model, {})


@pytest.fixture
def relay(monkeypatch):
    _isolated_registry(monkeypatch)
    monkeypatch.setattr(models_dev, "_load_model_overrides", dict)
    monkeypatch.setattr("hermes_cli.config.load_config", dict)
    profile = CatalogProfile(
        name="fixture-relay", aliases=("fixture-named",), supports_vision=True,
        api_mode="codex_responses", base_url="https://relay.example/v1",
    )
    profile.entries = {
        "gpt-6.1-sol": {"supports_vision": True},
        "openai/gpt-6.1-sol": {"supports_vision": True},
        "text-only": {"supports_vision": False},
    }
    profile.warm = False
    profile.lookups = []
    providers.register_provider(profile)
    return profile


def make_agent(provider, model):
    # No constructor/config/client bootstrap: exercise AIAgent's actual inherited guards.
    agent = cast(Any, object.__new__(AIAgent))
    agent.provider = provider
    agent.model = model
    agent.api_mode = "codex_responses"
    agent._no_list_tool_content_models = set()
    agent._describe_image_for_anthropic_fallback = Mock(side_effect=AssertionError("auxiliary called"))
    return agent


def image_result():
    # One tiny deterministic PNG, also usable for live native-path acceptance.
    image = Path(__file__).with_name("fixtures") / "native-image.png"
    url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode("ascii")
    return {"_multimodal": True, "text_summary": "attached", "content": [
        {"type": "text", "text": "attached"},
        {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
    ]}


@pytest.mark.parametrize("provider,model", [
    ("fixture-relay", "gpt-6.1-sol"),
    ("fixture-relay", "openai/gpt-6.1-sol"),
    ("fixture-named", "gpt-6.1-sol"),
])
def test_registered_metadata_survives_main_guard_and_responses_envelope(relay, provider, model):
    agent = make_agent(provider, model)
    result = image_result()
    content = agent._tool_result_content_for_active_model("vision_analyze", result)
    assert content is result["content"]
    messages = [
        {"role": "user", "content": result["content"]},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_image", "type": "function",
            "function": {"name": "vision_analyze", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call_image", "content": content},
    ]
    original = deepcopy(messages)
    assert agent._prepare_messages_for_non_vision_model(messages) is messages
    transport = ResponsesApiTransport()
    kwargs = transport.preflight_kwargs(transport.build_kwargs(
        model, messages, provider=provider, base_url=relay.base_url,
        reasoning_config={"enabled": True, "effort": "high"},
    ))
    fake_client = Mock()
    fake_client.responses.create(**kwargs)
    sent = fake_client.responses.create.call_args.kwargs
    assert sent["model"] == model
    assert sent["store"] is False
    assert sent["reasoning"]["effort"] == "high"
    assert "service_tier" not in sent
    assert "messages" not in sent
    user = next(item for item in sent["input"] if item.get("role") == "user")
    output = next(item for item in sent["input"] if item["type"] == "function_call_output")
    for parts in (user["content"], output["output"]):
        image = next(part for part in parts if part["type"] == "input_image")
        assert image == {"type": "input_image", "image_url": result["content"][1]["image_url"]["url"], "detail": "high"}
    assert output["call_id"] == "call_image"
    assert messages == original
    agent._describe_image_for_anthropic_fallback.assert_not_called()


@pytest.mark.parametrize("model", ["text-only", "unadvertised-gpt", "unknown"])
def test_text_and_unknown_models_keep_fallback_and_computer_use_refusal(relay, model):
    agent = make_agent("fixture-relay", model)
    capability = models_dev.get_model_capabilities(relay.name, model, allow_network=True)
    if model == "text-only":
        assert capability is not None and capability.supports_vision is False
    else:
        assert capability is None
    assert agent._tool_result_content_for_active_model("vision_analyze", image_result()) == "attached"
    refusal = agent._tool_result_content_for_active_model("computer_use", image_result())
    assert "does not support image input" in refusal
    agent._describe_image_for_anthropic_fallback = Mock(return_value="auxiliary description")
    messages = [{"role": "user", "content": image_result()["content"]}]
    assert isinstance(agent._prepare_messages_for_non_vision_model(messages)[0]["content"], str)
    agent._describe_image_for_anthropic_fallback.assert_called_once()


def test_dynamic_metadata_respects_network_policy_overrides_and_route_mode(relay, monkeypatch):
    assert models_dev.get_model_capabilities(relay.name, "gpt-6.1-sol") is None
    assert relay.lookups == [False]
    fresh = models_dev.get_model_capabilities(relay.name, "gpt-6.1-sol", allow_network=True)
    assert fresh is not None and fresh.supports_vision is True
    cached = models_dev.get_model_capabilities(relay.name, "gpt-6.1-sol")
    assert cached is not None and cached.supports_vision is True
    assert relay.lookups == [False, True, False]
    explicit_aux = {"auxiliary": {"vision": {"provider": "other", "model": "chosen"}}}
    assert decide_image_input_mode(relay.name, "gpt-6.1-sol", explicit_aux) == "text"
    assert decide_image_input_mode(relay.name, "gpt-6.1-sol", {**explicit_aux, "agent": {"image_input_mode": "native"}}) == "native"
    monkeypatch.setattr(models_dev, "_load_model_overrides", lambda: {
        relay.name: {"gpt-6.1-sol": {"supports_vision": False}},
    })
    assert make_agent(relay.name, "gpt-6.1-sol")._model_supports_vision() is False


def test_native_metadata_does_not_bypass_tool_wire_veto_or_reactive_cache(relay):
    agent = make_agent(relay.name, "gpt-6.1-sol")
    assert agent._model_supports_vision() is True
    relay.supports_vision_tool_messages = False
    assert agent._tool_result_content_for_active_model("vision_analyze", image_result()) == "attached"
    relay.supports_vision_tool_messages = True
    agent._no_list_tool_content_models.add((relay.name, agent.model))
    assert agent._tool_result_content_for_active_model("vision_analyze", image_result()) == "attached"


def test_imports_resolve_to_tested_checkout():
    import agent.vision_message_prep as prep
    from providers import base
    import run_agent

    root = Path(__file__).resolve().parents[2]
    for module in (models_dev, prep, base, run_agent):
        assert module.__file__ is not None
        assert Path(module.__file__).resolve().is_relative_to(root)
