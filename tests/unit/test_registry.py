"""Tests for model registry."""

import pytest

from kvfleet.config.schema import ModelConfig, ProviderType
from kvfleet.registry.models import ModelRegistry


@pytest.fixture
def sample_models():
    return [
        ModelConfig(
            name="llama-8b",
            endpoint="http://a:8000",
            provider=ProviderType.VLLM,
            quality_score=0.7,
            cost_per_1k_input_tokens=0.0,
            tags={"domain": "general"},
        ),
        ModelConfig(
            name="llama-70b",
            endpoint="http://b:8000",
            provider=ProviderType.VLLM,
            quality_score=0.9,
            cost_per_1k_input_tokens=0.0,
            tags={"domain": "coding"},
        ),
        ModelConfig(
            name="gpt-4o",
            endpoint="https://api.openai.com",
            provider=ProviderType.OPENAI_COMPAT,
            quality_score=0.95,
            cost_per_1k_input_tokens=0.005,
            allowed_data_classes=["public"],
        ),
    ]


@pytest.fixture
def registry(sample_models):
    return ModelRegistry.from_configs(sample_models)


class TestModelRegistry:
    def test_register_and_get(self, registry):
        model = registry.get("llama-8b")
        assert model.name == "llama-8b"
        assert model.provider == ProviderType.VLLM

    def test_count(self, registry):
        assert registry.count == 3
        assert registry.enabled_count == 3

    def test_duplicate_register(self, registry):
        with pytest.raises(ValueError, match="already registered"):
            registry.register(ModelConfig(name="llama-8b", endpoint="http://x:8000"))

    def test_unregister(self, registry):
        registry.unregister("gpt-4o")
        assert registry.count == 2
        with pytest.raises(KeyError):
            registry.get("gpt-4o")

    def test_unregister_missing(self, registry):
        with pytest.raises(KeyError):
            registry.unregister("nonexistent")

    def test_list_all(self, registry):
        models = registry.list_models()
        assert len(models) == 3

    def test_filter_by_provider(self, registry):
        models = registry.list_models(provider=ProviderType.VLLM)
        assert len(models) == 2
        assert all(m.provider == ProviderType.VLLM for m in models)

    def test_filter_by_tags(self, registry):
        models = registry.list_models(tags={"domain": "coding"})
        assert len(models) == 1
        assert models[0].name == "llama-70b"

    def test_filter_by_data_class(self, registry):
        models = registry.list_models(data_class="confidential")
        assert len(models) == 2  # two vllm models allow confidential
        assert all(m.provider == ProviderType.VLLM for m in models)

    def test_filter_by_quality(self, registry):
        models = registry.list_models(min_quality=0.9)
        assert len(models) == 2

    def test_filter_by_cost(self, registry):
        models = registry.list_models(max_cost=0.001)
        assert len(models) == 2  # free models only

    def test_enabled_filter(self, registry):
        registry.disable("llama-8b")
        models = registry.list_models(enabled_only=True)
        assert len(models) == 2
        assert "llama-8b" not in [m.name for m in models]

    def test_enable_disable(self, registry):
        registry.disable("gpt-4o")
        assert not registry.get("gpt-4o").enabled
        registry.enable("gpt-4o")
        assert registry.get("gpt-4o").enabled

    def test_update(self, registry):
        updated = registry.update("llama-8b", quality_score=0.8)
        assert updated.quality_score == 0.8
        assert registry.get("llama-8b").quality_score == 0.8

    def test_summary(self, registry):
        summary = registry.summary()
        assert summary["total"] == 3
        assert summary["enabled"] == 3
        assert len(summary["providers"]) >= 1

    def test_from_configs(self, sample_models):
        reg = ModelRegistry.from_configs(sample_models)
        assert reg.count == 3
