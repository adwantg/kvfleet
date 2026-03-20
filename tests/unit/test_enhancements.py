"""Tests for kvfleet enhancements (E-1 through E-10)."""

from __future__ import annotations

from kvfleet.adapters.base import ChatMessage, ChatRequest
from kvfleet.config.schema import (
    GatewayConfig,
    ModelCapabilities,
    ModelConfig,
    ProviderType,
)
from kvfleet.router.multimodal import filter_json_mode_capable, filter_tool_capable

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _model(
    name: str,
    supports_tools: bool = False,
    supports_json_mode: bool = False,
    supports_vision: bool = False,
    model_type: str = "chat",
) -> ModelConfig:
    return ModelConfig(
        name=name,
        endpoint=f"http://{name}:8000",
        provider=ProviderType.OPENAI_COMPAT,
        capabilities=ModelCapabilities(
            supports_tools=supports_tools,
            supports_json_mode=supports_json_mode,
            supports_vision=supports_vision,
        ),
    )


def _request(
    tools: list | None = None,
    response_format: dict | None = None,
    metadata: dict | None = None,
) -> ChatRequest:
    return ChatRequest(
        messages=[ChatMessage(role="user", content="hello")],
        tools=tools,
        response_format=response_format,
        metadata=metadata or {},
    )


# ===================================================================
# E-1: Header pass-through
# ===================================================================


class TestE1HeaderPassthrough:
    """E-1: passthrough_headers config and metadata convention."""

    def test_gateway_config_defaults_empty(self):
        gw = GatewayConfig()
        assert gw.passthrough_headers == []

    def test_gateway_config_accepts_headers(self):
        gw = GatewayConfig(passthrough_headers=["X-Access-Token", "X-Request-ID"])
        assert len(gw.passthrough_headers) == 2
        assert "X-Access-Token" in gw.passthrough_headers

    def test_passthrough_metadata_survives_copy(self):
        """Verify metadata is preserved when ChatRequest is copied (fallback path)."""
        req = _request(metadata={"_passthrough_headers": {"X-Foo": "bar"}})
        copy = ChatRequest(
            messages=req.messages,
            model=req.model,
            temperature=req.temperature,
            metadata=req.metadata,
        )
        assert copy.metadata["_passthrough_headers"]["X-Foo"] == "bar"

    def test_passthrough_metadata_absent_by_default(self):
        req = _request()
        assert req.metadata.get("_passthrough_headers") is None


# ===================================================================
# E-2: Tool-use capability filter
# ===================================================================


class TestE2ToolFilter:
    """E-2: filter_tool_capable correctly gates on supports_tools."""

    def test_no_tools_returns_all(self):
        models = [_model("a"), _model("b")]
        result = filter_tool_capable(models, _request())
        assert len(result) == 2

    def test_tools_filters_incapable(self):
        models = [
            _model("capable", supports_tools=True),
            _model("incapable", supports_tools=False),
        ]
        req = _request(tools=[{"type": "function", "function": {"name": "f"}}])
        result = filter_tool_capable(models, req)
        assert len(result) == 1
        assert result[0].name == "capable"

    def test_tools_no_capable_model_returns_all(self):
        models = [_model("a"), _model("b")]
        req = _request(tools=[{"type": "function", "function": {"name": "f"}}])
        result = filter_tool_capable(models, req)
        assert len(result) == 2  # graceful fallback

    def test_tools_empty_list_is_noop(self):
        models = [_model("a")]
        req = _request(tools=[])
        result = filter_tool_capable(models, req)
        assert len(result) == 1


# ===================================================================
# E-8: Response format capability filter
# ===================================================================


class TestE8JsonModeFilter:
    """E-8: filter_json_mode_capable correctly gates on supports_json_mode."""

    def test_no_format_returns_all(self):
        models = [_model("a"), _model("b")]
        result = filter_json_mode_capable(models, _request())
        assert len(result) == 2

    def test_json_object_filters_incapable(self):
        models = [
            _model("json_ok", supports_json_mode=True),
            _model("no_json", supports_json_mode=False),
        ]
        req = _request(response_format={"type": "json_object"})
        result = filter_json_mode_capable(models, req)
        assert len(result) == 1
        assert result[0].name == "json_ok"

    def test_json_object_no_capable_returns_all(self):
        models = [_model("a"), _model("b")]
        req = _request(response_format={"type": "json_object"})
        result = filter_json_mode_capable(models, req)
        assert len(result) == 2  # graceful fallback

    def test_text_format_is_noop(self):
        models = [_model("a"), _model("b")]
        req = _request(response_format={"type": "text"})
        result = filter_json_mode_capable(models, req)
        assert len(result) == 2

    def test_none_format_is_noop(self):
        models = [_model("a")]
        result = filter_json_mode_capable(models, _request(response_format=None))
        assert len(result) == 1


# ===================================================================
# E-3: Per-request strategy override
# ===================================================================


class TestE3StrategyOverride:
    """E-3: GatewayConfig.strategy_header and RouteExplanation.strategy_overridden."""

    def test_strategy_header_default(self):
        gw = GatewayConfig()
        assert gw.strategy_header == "X-KVFleet-Strategy"

    def test_strategy_header_disable(self):
        gw = GatewayConfig(strategy_header="")
        assert gw.strategy_header == ""

    def test_explanation_strategy_overridden_default_false(self):
        from kvfleet.router.explain import RouteExplanation

        exp = RouteExplanation()
        assert exp.strategy_overridden is False

    def test_valid_route_strategy_enum(self):
        from kvfleet.config.schema import RouteStrategy

        s = RouteStrategy("cheap_cascade")
        assert s == RouteStrategy.CHEAP_CASCADE

    def test_invalid_route_strategy_raises(self):
        import pytest

        from kvfleet.config.schema import RouteStrategy

        with pytest.raises(ValueError):
            RouteStrategy("nonexistent_strategy")


# ===================================================================
# E-4: Per-request model allowlist
# ===================================================================


class TestE4ModelAllowlist:
    """E-4: GatewayConfig.model_allowlist_header."""

    def test_model_allowlist_header_default(self):
        gw = GatewayConfig()
        assert gw.model_allowlist_header == "X-KVFleet-Models"

    def test_allowlist_parsing(self):
        val = "claude-4.5, opus-4.6 , claude-4"
        result = [m.strip() for m in val.split(",") if m.strip()]
        assert result == ["claude-4.5", "opus-4.6", "claude-4"]

    def test_allowlist_empty_string_no_filter(self):
        val = ""
        result = [m.strip() for m in val.split(",") if m.strip()]
        assert result == []


# ===================================================================
# E-5: Tenant ID from header
# ===================================================================


class TestE5TenantHeader:
    """E-5: GatewayConfig.tenant_header."""

    def test_tenant_header_default_empty(self):
        gw = GatewayConfig()
        assert gw.tenant_header == ""

    def test_tenant_header_configurable(self):
        gw = GatewayConfig(tenant_header="X-Tenant-ID")
        assert gw.tenant_header == "X-Tenant-ID"


# ===================================================================
# E-9: Per-request timeout override
# ===================================================================


class TestE9TimeoutOverride:
    """E-9: timeout from metadata used by fallback chain."""

    def test_timeout_header_default(self):
        gw = GatewayConfig()
        assert gw.timeout_header == "X-KVFleet-Timeout"

    def test_timeout_metadata_stored(self):
        req = _request(metadata={"_timeout_ms": 5000})
        assert req.metadata["_timeout_ms"] == 5000

    def test_timeout_metadata_absent_by_default(self):
        req = _request()
        assert "_timeout_ms" not in req.metadata


# ===================================================================
# E-10: Request ID propagation
# ===================================================================


class TestE10RequestIDPropagation:
    """E-10: X-Request-ID forwarded or generated."""

    def test_request_id_in_explanation(self):
        from kvfleet.router.explain import RouteExplanation

        exp = RouteExplanation(request_id="test-123")
        assert exp.request_id == "test-123"
        d = exp.to_dict()
        assert d["request_id"] == "test-123"


# ===================================================================
# E-6: Model type classification
# ===================================================================


class TestE6ModelType:
    """E-6: model_type field on ModelCapabilities and registry filter."""

    def test_default_model_type_is_chat(self):
        caps = ModelCapabilities()
        assert caps.model_type == "chat"

    def test_embedding_model_type(self):
        caps = ModelCapabilities(model_type="embedding")
        assert caps.model_type == "embedding"

    def test_registry_filters_by_model_type(self):
        from kvfleet.registry.models import ModelRegistry

        reg = ModelRegistry()
        reg.register(_model("gpt-4"))
        embed = _model("text-embed")
        embed.capabilities.model_type = "embedding"
        reg.register(embed)

        chat_models = reg.list_models(model_type="chat")
        assert len(chat_models) == 1
        assert chat_models[0].name == "gpt-4"

        embed_models = reg.list_models(model_type="embedding")
        assert len(embed_models) == 1
        assert embed_models[0].name == "text-embed"

    def test_registry_no_filter_returns_all(self):
        from kvfleet.registry.models import ModelRegistry

        reg = ModelRegistry()
        reg.register(_model("a"))
        embed = _model("b")
        embed.capabilities.model_type = "embedding"
        reg.register(embed)
        assert len(reg.list_models()) == 2


# ===================================================================
# E-7: Shared endpoint optimization
# ===================================================================


class TestE7SharedEndpoint:
    """E-7: shared connection pool and health dedup."""

    def test_shared_pool_reuses_client(self):
        from kvfleet.adapters.openai_compat import OpenAICompatAdapter

        # Clear pool to isolate test
        OpenAICompatAdapter._shared_pool.clear()

        a1 = OpenAICompatAdapter(endpoint="http://shared:8000", model_id="m1", api_key="key1")
        a2 = OpenAICompatAdapter(endpoint="http://shared:8000", model_id="m2", api_key="key1")

        client1 = a1._get_client()
        client2 = a2._get_client()
        assert client1 is client2  # Same object from pool

        # Clean up pool
        OpenAICompatAdapter._shared_pool.clear()

    def test_different_keys_get_different_clients(self):
        from kvfleet.adapters.openai_compat import OpenAICompatAdapter

        OpenAICompatAdapter._shared_pool.clear()

        a1 = OpenAICompatAdapter(endpoint="http://shared:8000", model_id="m1", api_key="key1")
        a2 = OpenAICompatAdapter(endpoint="http://shared:8000", model_id="m2", api_key="key2")

        client1 = a1._get_client()
        client2 = a2._get_client()
        assert client1 is not client2

        OpenAICompatAdapter._shared_pool.clear()


# ===================================================================
# BUG FIX: api_key pass-through from ModelConfig to adapters
# ===================================================================


class TestApiKeyPassthrough:
    """Verify api_key is configurable on ModelConfig and passed to adapters."""

    def test_model_config_api_key_defaults_empty(self):
        model = ModelConfig(name="m1", endpoint="http://local:8000")
        assert model.api_key == ""

    def test_model_config_api_key_set(self):
        model = ModelConfig(name="m1", endpoint="http://local:8000", api_key="sk-test-123")
        assert model.api_key == "sk-test-123"

    def test_adapter_receives_api_key(self):
        from kvfleet.adapters.openai_compat import OpenAICompatAdapter

        OpenAICompatAdapter._shared_pool.clear()

        adapter = OpenAICompatAdapter(endpoint="http://api:8000", model_id="m1", api_key="sk-key")
        client = adapter._get_client()
        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == "Bearer sk-key"

        OpenAICompatAdapter._shared_pool.clear()

    def test_adapter_no_auth_header_without_key(self):
        from kvfleet.adapters.openai_compat import OpenAICompatAdapter

        OpenAICompatAdapter._shared_pool.clear()

        adapter = OpenAICompatAdapter(endpoint="http://api:8000", model_id="m1", api_key="")
        client = adapter._get_client()
        assert "Authorization" not in client.headers

        OpenAICompatAdapter._shared_pool.clear()

    def test_init_adapters_passes_api_key(self):
        """End-to-end: Router._init_adapters should pass api_key from config."""
        from kvfleet.adapters.openai_compat import OpenAICompatAdapter
        from kvfleet.config.schema import FleetConfig

        OpenAICompatAdapter._shared_pool.clear()

        config = FleetConfig(
            fleet_name="test",
            models=[
                ModelConfig(
                    name="openai-model",
                    endpoint="http://api:8000",
                    provider="openai_compat",
                    api_key="sk-from-config",
                ),
            ],
        )
        from kvfleet.router.engine import Router

        router = Router(config)
        adapter = router._adapters["openai-model"]
        assert isinstance(adapter, OpenAICompatAdapter)
        assert adapter.api_key == "sk-from-config"

        client = adapter._get_client()
        assert client.headers["Authorization"] == "Bearer sk-from-config"

        OpenAICompatAdapter._shared_pool.clear()


# ===================================================================
# BUG-3: api_key for ALL adapters (TGI, Triton, Ollama, CustomHTTP)
# ===================================================================


class TestApiKeyAllAdapters:
    """Verify api_key is wired through all adapter constructors."""

    def test_tgi_adapter_auth_header(self):
        from kvfleet.adapters.tgi import TGIAdapter

        adapter = TGIAdapter(endpoint="http://tgi:8000", api_key="sk-tgi-test")
        client = adapter._get_client()
        assert client.headers["Authorization"] == "Bearer sk-tgi-test"
        assert adapter.api_key == "sk-tgi-test"

    def test_tgi_adapter_no_auth_without_key(self):
        from kvfleet.adapters.tgi import TGIAdapter

        adapter = TGIAdapter(endpoint="http://tgi:8000")
        client = adapter._get_client()
        assert "Authorization" not in client.headers

    def test_triton_adapter_auth_header(self):
        from kvfleet.adapters.triton import TritonAdapter

        adapter = TritonAdapter(endpoint="http://triton:8000", api_key="sk-triton")
        client = adapter._get_client()
        assert client.headers["Authorization"] == "Bearer sk-triton"

    def test_ollama_adapter_auth_header(self):
        from kvfleet.adapters.ollama import OllamaAdapter

        adapter = OllamaAdapter(endpoint="http://ollama:11434", api_key="sk-proxy")
        client = adapter._get_client()
        assert client.headers["Authorization"] == "Bearer sk-proxy"

    def test_custom_http_adapter_auth_header(self):
        from kvfleet.adapters.custom_http import CustomHTTPAdapter

        adapter = CustomHTTPAdapter(endpoint="http://custom:8000", api_key="sk-custom")
        client = adapter._get_client()
        assert client.headers["Authorization"] == "Bearer sk-custom"

    def test_custom_http_explicit_auth_header_not_overridden(self):
        """If custom_headers already has Authorization, api_key should not override it."""
        from kvfleet.adapters.custom_http import CustomHTTPAdapter

        adapter = CustomHTTPAdapter(
            endpoint="http://custom:8000",
            api_key="sk-fallback",
            headers={"Authorization": "Token my-explicit-token"},
        )
        client = adapter._get_client()
        assert client.headers["Authorization"] == "Token my-explicit-token"

    def test_base_adapter_stores_api_key(self):
        # Can't instantiate abstract class, but verify via TGI subclass
        from kvfleet.adapters.tgi import TGIAdapter

        adapter = TGIAdapter(endpoint="http://test:8000", api_key="sk-base-test")
        assert adapter.api_key == "sk-base-test"


# ===================================================================
# BUG-1: CustomHTTP config fields wired from ModelConfig
# ===================================================================


class TestCustomHTTPConfigWiring:
    """Verify custom_http adapter gets its config from ModelConfig."""

    def test_model_config_custom_fields_default(self):
        model = ModelConfig(name="m1", endpoint="http://local:8000")
        assert model.custom_headers == {}
        assert model.custom_chat_path == "/generate"
        assert model.custom_health_path == "/health"
        assert model.custom_request_template == {}

    def test_model_config_custom_fields_set(self):
        model = ModelConfig(
            name="m1",
            endpoint="http://local:8000",
            custom_headers={"X-Api-Key": "secret"},
            custom_chat_path="/v1/generate",
            custom_health_path="/v1/health",
            custom_request_template={"extra_param": True},
        )
        assert model.custom_headers == {"X-Api-Key": "secret"}
        assert model.custom_chat_path == "/v1/generate"

    def test_init_adapters_passes_custom_http_config(self):
        from kvfleet.adapters.custom_http import CustomHTTPAdapter
        from kvfleet.config.schema import FleetConfig

        config = FleetConfig(
            fleet_name="test",
            models=[
                ModelConfig(
                    name="custom-model",
                    endpoint="http://custom:8000",
                    provider="custom_http",
                    api_key="sk-custom-key",
                    custom_headers={"X-Api-Key": "header-key"},
                    custom_chat_path="/v1/generate",
                    custom_health_path="/ping",
                ),
            ],
        )
        from kvfleet.router.engine import Router

        router = Router(config)
        adapter = router._adapters["custom-model"]
        assert isinstance(adapter, CustomHTTPAdapter)
        assert adapter.chat_path == "/v1/generate"
        assert adapter.health_path == "/ping"
        assert adapter.api_key == "sk-custom-key"


# ===================================================================
# BUG-2: Gateway includes tool_calls in response
# ===================================================================


class TestGatewayToolCallsResponse:
    """Verify ChatResponse.tool_calls are preserved in response serialization."""

    def test_chat_response_with_tool_calls(self):
        from kvfleet.adapters.base import ChatResponse

        resp = ChatResponse(
            content="",
            tool_calls=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1

    def test_chat_response_without_tool_calls(self):
        from kvfleet.adapters.base import ChatResponse

        resp = ChatResponse(content="Hello!")
        assert resp.tool_calls is None


# ===================================================================
# BUG-4: stop sequences in ChatRequest
# ===================================================================


class TestStopSequences:
    """Verify stop sequences are included in ChatRequest."""

    def test_chat_request_with_stop(self):
        from kvfleet.adapters.base import ChatMessage, ChatRequest

        req = ChatRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            stop=["STOP", "END"],
        )
        d = req.to_openai_dict()
        assert d["stop"] == ["STOP", "END"]

    def test_chat_request_without_stop(self):
        from kvfleet.adapters.base import ChatMessage, ChatRequest

        req = ChatRequest(messages=[ChatMessage(role="user", content="Hi")])
        d = req.to_openai_dict()
        assert "stop" not in d


# ===================================================================
# BUG-5: api_key excluded from save_config
# ===================================================================


class TestApiKeyNotInSavedConfig:
    """Verify api_key is never written to YAML."""

    def test_save_config_excludes_api_key(self, tmp_path):
        from kvfleet.config.loader import save_config
        from kvfleet.config.schema import FleetConfig

        config = FleetConfig(
            fleet_name="test",
            models=[
                ModelConfig(
                    name="m1",
                    endpoint="http://api:8000",
                    api_key="sk-secret-key",
                ),
            ],
        )
        out = tmp_path / "fleet.yaml"
        save_config(config, out)
        content = out.read_text()
        assert "sk-secret-key" not in content
        assert "api_key" not in content


# ===================================================================
# BUG-6: Gateway preserves full message fields
# ===================================================================


class TestMessageFieldPreservation:
    """Verify name, tool_call_id, and tool_calls are preserved in ChatMessage."""

    def test_chat_message_with_all_fields(self):
        from kvfleet.adapters.base import ChatMessage

        msg = ChatMessage(
            role="tool",
            content="75°F",
            name="get_weather",
            tool_call_id="call_abc123",
        )
        assert msg.name == "get_weather"
        assert msg.tool_call_id == "call_abc123"
        assert msg.role == "tool"

    def test_chat_message_with_tool_calls(self):
        from kvfleet.adapters.base import ChatMessage

        msg = ChatMessage(
            role="assistant",
            content="",
            tool_calls=[{"type": "function", "function": {"name": "search", "arguments": "{}"}}],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1


# ===================================================================
# BUG-7: to_openai_dict strictly sends ONLY temperature OR top_p
# ===================================================================


class TestGenerationParametersConflict:
    """Verify temperature and top_p do not conflict."""

    def test_temperature_prioritized_over_top_p(self):
        from kvfleet.adapters.base import ChatRequest

        req = ChatRequest(
            messages=[],
            temperature=0.7,
            top_p=0.9,
        )
        d = req.to_openai_dict()
        assert "temperature" in d
        assert d["temperature"] == 0.7
        assert "top_p" not in d

    def test_top_p_sent_when_temperature_default(self):
        from kvfleet.adapters.base import ChatRequest

        req = ChatRequest(
            messages=[],
            temperature=1.0,  # Default
            top_p=0.9,
        )
        d = req.to_openai_dict()
        assert "top_p" in d
        assert d["top_p"] == 0.9
        assert "temperature" not in d

    def test_neither_sent_when_both_default(self):
        from kvfleet.adapters.base import ChatRequest

        req = ChatRequest(
            messages=[],
            temperature=1.0,
            top_p=1.0,
        )
        d = req.to_openai_dict()
        assert "temperature" not in d
        assert "top_p" not in d
