"""Tests for kvfleet config schema and loader."""


import pytest
import yaml

from kvfleet.config.loader import _coerce_value, _deep_merge, load_config, save_config
from kvfleet.config.schema import (
    FleetConfig,
    ModelCapabilities,
    ModelConfig,
    ProviderType,
    RouteRuleConfig,
    RouteStrategy,
    ScoringWeights,
)

# ───────────────────────── Schema Tests ─────────────────────────


class TestModelConfig:
    def test_defaults(self):
        m = ModelConfig(name="test", endpoint="http://localhost:8000")
        assert m.name == "test"
        assert m.provider == ProviderType.OPENAI_COMPAT
        assert m.enabled is True
        assert m.quality_score == 0.5
        assert m.weight == 1.0

    def test_get_model_id_default(self):
        m = ModelConfig(name="llama", endpoint="http://localhost:8000")
        assert m.get_model_id() == "llama"

    def test_get_model_id_explicit(self):
        m = ModelConfig(
            name="llama", endpoint="http://localhost:8000", model_id="meta-llama/Llama-3"
        )
        assert m.get_model_id() == "meta-llama/Llama-3"

    def test_all_endpoints(self):
        m = ModelConfig(
            name="llama",
            endpoint="http://host1:8000",
            replicas=["http://host2:8000", "http://host3:8000"],
        )
        assert m.all_endpoints() == ["http://host1:8000", "http://host2:8000", "http://host3:8000"]

    def test_quality_bounds(self):
        with pytest.raises(ValueError):
            ModelConfig(name="x", endpoint="http://x", quality_score=1.5)

    def test_capabilities(self):
        m = ModelConfig(
            name="test",
            endpoint="http://x",
            capabilities=ModelCapabilities(supports_tools=True, max_context_window=128000),
        )
        assert m.capabilities.supports_tools is True
        assert m.capabilities.max_context_window == 128000


class TestFleetConfig:
    def test_defaults(self):
        cfg = FleetConfig()
        assert cfg.fleet_name == "default"
        assert cfg.strategy == RouteStrategy.HYBRID_SCORE
        assert cfg.fallback.enabled is True
        assert cfg.cache_affinity.enabled is True

    def test_with_models(self):
        cfg = FleetConfig(
            models=[
                ModelConfig(name="m1", endpoint="http://a:8000"),
                ModelConfig(name="m2", endpoint="http://b:8000"),
            ]
        )
        assert len(cfg.models) == 2

    def test_scoring_weights_defaults(self):
        w = ScoringWeights()
        assert w.cost == 0.3
        assert w.latency == 0.3
        assert w.quality == 0.3
        assert w.cache_affinity == 0.1


class TestRouteRuleConfig:
    def test_basic_rule(self):
        rule = RouteRuleConfig(
            name="code-to-deepseek",
            condition={"tags.domain": "coding"},
            target_model="deepseek-coder",
        )
        assert rule.priority == 100
        assert rule.name == "code-to-deepseek"


# ───────────────────────── Loader Tests ─────────────────────────


class TestConfigLoader:
    def test_load_from_yaml(self, tmp_path):
        cfg = {
            "fleet_name": "test-fleet",
            "strategy": "cost_first",
            "models": [
                {"name": "m1", "endpoint": "http://a:8000", "provider": "vllm"},
            ],
        }
        config_path = tmp_path / "fleet.yaml"
        config_path.write_text(yaml.dump(cfg))

        result = load_config(config_path)
        assert result.fleet_name == "test-fleet"
        assert result.strategy == RouteStrategy.COST_FIRST
        assert len(result.models) == 1
        assert result.models[0].provider == ProviderType.VLLM

    def test_load_with_overrides(self, tmp_path):
        cfg = {"fleet_name": "base", "strategy": "static"}
        config_path = tmp_path / "fleet.yaml"
        config_path.write_text(yaml.dump(cfg))

        result = load_config(config_path, overrides={"fleet_name": "overridden"})
        assert result.fleet_name == "overridden"

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_save_and_load_roundtrip(self, tmp_path):
        cfg = FleetConfig(
            fleet_name="roundtrip",
            models=[ModelConfig(name="m1", endpoint="http://a:8000")],
        )
        path = tmp_path / "out.yaml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.fleet_name == "roundtrip"
        assert len(loaded.models) == 1

    def test_env_var_override(self, tmp_path, monkeypatch):
        cfg = {"fleet_name": "base", "strategy": "static"}
        config_path = tmp_path / "fleet.yaml"
        config_path.write_text(yaml.dump(cfg))

        monkeypatch.setenv("KVFLEET__STRATEGY", "cost_first")
        result = load_config(config_path)
        assert result.strategy == RouteStrategy.COST_FIRST

    def test_load_empty_yaml(self, tmp_path):
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")
        result = load_config(config_path)
        assert isinstance(result, FleetConfig)


class TestDeepMerge:
    def test_flat_merge(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_nested_merge(self):
        result = _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3, "z": 4}})
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_override_non_dict(self):
        result = _deep_merge({"a": {"x": 1}}, {"a": "replaced"})
        assert result == {"a": "replaced"}


class TestCoerceValue:
    def test_bool_true(self):
        assert _coerce_value("true") is True
        assert _coerce_value("yes") is True

    def test_bool_false(self):
        assert _coerce_value("false") is False
        assert _coerce_value("no") is False

    def test_int(self):
        assert _coerce_value("42") == 42

    def test_float(self):
        assert _coerce_value("3.14") == 3.14

    def test_string(self):
        assert _coerce_value("hello") == "hello"
