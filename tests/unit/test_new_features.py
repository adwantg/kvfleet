"""Tests for rate limit awareness, multimodal routing, cost sync, and dashboard."""

import time
import pytest

from kvfleet.telemetry.rate_limits import RateLimitTracker, RateLimitState
from kvfleet.telemetry.cost_sync import CostSyncManager, KNOWN_MODEL_COSTS
from kvfleet.router.multimodal import (
    detect_modality, filter_vision_capable, estimate_multimodal_cost,
    ModalityDetection,
)
from kvfleet.gateway.dashboard import DashboardState
from kvfleet.config.schema import ModelConfig, ModelCapabilities
from kvfleet.adapters.base import ChatMessage


# ───────────────────── Rate Limit Tracker ─────────────────────


class TestRateLimitTracker:
    def test_record_request(self):
        tracker = RateLimitTracker(default_rpm=100)
        tracker.record_request("http://api:8000", "llama-8b")
        state = tracker.get_state("http://api:8000", "llama-8b")
        assert state is not None
        assert state.requests_this_minute == 1

    def test_throttle_detection(self):
        tracker = RateLimitTracker(default_rpm=10, throttle_threshold=0.8)
        for _ in range(9):
            tracker.record_request("http://api:8000", "model-a")
        assert tracker.should_throttle("http://api:8000", "model-a")

    def test_no_throttle_below_threshold(self):
        tracker = RateLimitTracker(default_rpm=100, throttle_threshold=0.85)
        for _ in range(5):
            tracker.record_request("http://api:8000", "model-a")
        assert not tracker.should_throttle("http://api:8000", "model-a")

    def test_429_cooldown(self):
        tracker = RateLimitTracker(cooldown_seconds=2.0)
        tracker.record_429("http://api:8000", "model-a", retry_after=2.0)
        assert tracker.should_throttle("http://api:8000", "model-a")
        state = tracker.get_state("http://api:8000", "model-a")
        assert state.is_throttled

    def test_capacity_score(self):
        tracker = RateLimitTracker(default_rpm=100)
        # Unknown endpoint = full capacity
        assert tracker.get_capacity_score("http://nowhere:8000") == 1.0
        # After some requests
        for _ in range(50):
            tracker.record_request("http://api:8000", "model-a")
        score = tracker.get_capacity_score("http://api:8000", "model-a")
        assert 0.4 < score < 0.6

    def test_rate_limit_headers(self):
        tracker = RateLimitTracker()
        tracker.record_rate_limit_headers("http://api:8000", "model-a", headers={
            "x-ratelimit-limit-requests": "60",
            "x-ratelimit-remaining-requests": "45",
        })
        state = tracker.get_state("http://api:8000", "model-a")
        assert state.requests_per_minute == 60
        assert state.remaining_requests == 45

    def test_summary(self):
        tracker = RateLimitTracker(default_rpm=100)
        tracker.record_request("http://api:8000", "m1")
        summary = tracker.summary()
        assert len(summary) == 1

    def test_all_states(self):
        tracker = RateLimitTracker()
        tracker.record_request("http://a:8000", "m1")
        tracker.record_request("http://b:8000", "m2")
        states = tracker.get_all_states()
        assert len(states) == 2


# ───────────────────── Cost Sync ─────────────────────


class TestCostSync:
    def test_builtin_costs(self):
        manager = CostSyncManager()
        assert manager.model_count > 20

    def test_get_known_model(self):
        manager = CostSyncManager()
        cost = manager.get_cost("gpt-4o")
        assert cost is not None
        assert cost.input_cost_per_1k == 0.0025
        assert cost.output_cost_per_1k == 0.01

    def test_get_unknown_model(self):
        manager = CostSyncManager()
        cost = manager.get_cost("totally-unknown-model-xyz")
        assert cost is None

    def test_partial_match(self):
        manager = CostSyncManager()
        cost = manager.get_cost("meta-llama/llama-3-8b-instruct")
        assert cost is not None  # Partial match to "llama-3-8b"

    def test_set_custom_cost(self):
        manager = CostSyncManager()
        manager.set_cost("my-model", 0.001, 0.002, source="manual")
        cost = manager.get_cost("my-model")
        assert cost is not None
        assert cost.input_cost_per_1k == 0.001
        assert cost.source == "manual"

    def test_estimate_request_cost(self):
        manager = CostSyncManager()
        cost = manager.estimate_request_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        expected = 0.0025 + (0.5 * 0.01)
        assert abs(cost - expected) < 0.001

    def test_cheapest_model(self):
        manager = CostSyncManager()
        cheapest = manager.get_cheapest_model(["gpt-4o", "gpt-4o-mini", "gpt-4"])
        assert cheapest == "gpt-4o-mini"

    def test_self_hosted_free(self):
        manager = CostSyncManager()
        cost = manager.estimate_request_cost("llama-3-8b", 1000, 1000)
        assert cost == 0.0

    def test_config_sync(self):
        manager = CostSyncManager()
        models = [
            ModelConfig(name="my-model", endpoint="http://a:8000",
                        cost_per_1k_input_tokens=0.005, cost_per_1k_output_tokens=0.01),
        ]
        count = manager.sync_from_config(models)
        assert count == 1

    def test_summary(self):
        manager = CostSyncManager()
        summary = manager.summary()
        assert "gpt-4o" in summary


# ───────────────────── Multimodal Routing ─────────────────────


class TestModalityDetection:
    def test_text_only(self):
        messages = [ChatMessage(role="user", content="Hello, world!")]
        result = detect_modality(messages)
        assert not result.is_multimodal
        assert result.primary_modality == "text"

    def test_image_in_content_parts(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png", "detail": "high"}},
            ]},
        ]
        result = detect_modality(messages)
        assert result.has_images
        assert result.image_count == 1
        assert result.is_multimodal
        assert result.primary_modality == "vision"
        assert result.estimated_image_tokens == 765

    def test_multiple_images(self):
        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "a.png", "detail": "low"}},
                {"type": "image_url", "image_url": {"url": "b.png", "detail": "low"}},
                {"type": "text", "text": "Compare these"},
            ]},
        ]
        result = detect_modality(messages)
        assert result.image_count == 2
        assert result.estimated_image_tokens == 170

    def test_audio_detection(self):
        messages = [{"role": "user", "content": [{"type": "input_audio", "data": "..."}]}]
        result = detect_modality(messages)
        assert result.has_audio
        assert result.primary_modality == "audio"

    def test_video_priority(self):
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "a.png"}},
            {"type": "video", "data": "..."},
        ]}]
        result = detect_modality(messages)
        assert result.primary_modality == "video"


class TestFilterVisionCapable:
    def test_text_only_returns_all(self):
        models = [
            ModelConfig(name="a", endpoint="http://a:8000"),
            ModelConfig(name="b", endpoint="http://b:8000"),
        ]
        detection = ModalityDetection(has_images=False)
        result = filter_vision_capable(models, detection)
        assert len(result) == 2

    def test_vision_filters(self):
        models = [
            ModelConfig(name="text-only", endpoint="http://a:8000",
                        capabilities=ModelCapabilities(supports_vision=False)),
            ModelConfig(name="vision-model", endpoint="http://b:8000",
                        capabilities=ModelCapabilities(supports_vision=True)),
        ]
        detection = ModalityDetection(has_images=True, image_count=1)
        result = filter_vision_capable(models, detection)
        assert len(result) == 1
        assert result[0].name == "vision-model"

    def test_vision_tag_filter(self):
        models = [
            ModelConfig(name="tagged-vision", endpoint="http://a:8000",
                        tags={"vision": "true"}),
            ModelConfig(name="no-vision", endpoint="http://b:8000"),
        ]
        detection = ModalityDetection(has_images=True, image_count=1)
        result = filter_vision_capable(models, detection)
        assert len(result) == 1
        assert result[0].name == "tagged-vision"

    def test_no_capable_returns_all(self):
        models = [
            ModelConfig(name="a", endpoint="http://a:8000"),
            ModelConfig(name="b", endpoint="http://b:8000"),
        ]
        detection = ModalityDetection(has_images=True, image_count=1)
        result = filter_vision_capable(models, detection)
        assert len(result) == 2  # Fallback returns all


class TestMultimodalCost:
    def test_no_images(self):
        detection = ModalityDetection(has_images=False)
        model = ModelConfig(name="a", endpoint="http://a:8000", cost_per_1k_input_tokens=0.01)
        assert estimate_multimodal_cost(detection, model) == 0.0

    def test_image_cost(self):
        detection = ModalityDetection(has_images=True, image_count=2, estimated_image_tokens=1530)
        model = ModelConfig(name="a", endpoint="http://a:8000", cost_per_1k_input_tokens=0.01)
        cost = estimate_multimodal_cost(detection, model)
        assert cost > 0


# ───────────────────── Dashboard State ─────────────────────


class TestDashboardState:
    def test_record_route(self):
        state = DashboardState()
        state.record_route("Hello world", "llama-8b", "semantic", 250.0)
        assert state.total_requests == 1
        assert "llama-8b" in state.model_stats

    def test_error_tracking(self):
        state = DashboardState()
        state.record_route("test", "model-a", "static", 100.0, error="timeout")
        assert state.total_errors == 1

    def test_fallback_tracking(self):
        state = DashboardState()
        state.record_route("test", "model-b", "hybrid", 500.0, fallback=True)
        assert state.total_fallbacks == 1

    def test_cache_hit_tracking(self):
        state = DashboardState()
        state.record_route("test", "model-a", "static", 5.0, cache_hit=True)
        assert state.total_cache_hits == 1

    def test_to_dict(self):
        state = DashboardState()
        state.fleet_name = "test-fleet"
        state.strategy = "semantic"
        state.record_route("prompt", "model-a", "semantic", 200.0)
        d = state.to_dict()
        assert d["fleet"]["name"] == "test-fleet"
        assert d["counters"]["total_requests"] == 1
        assert len(d["recent_routes"]) == 1

    def test_history_limit(self):
        state = DashboardState(max_history=5)
        for i in range(10):
            state.record_route(f"prompt-{i}", "m", "s", 100.0)
        assert len(state.route_history) == 5

    def test_health_update(self):
        state = DashboardState()
        state.update_health("model-a", "http://a:8000", True, 50.0)
        assert "model-a:http://a:8000" in state.health_states

    def test_policy_block(self):
        state = DashboardState()
        state.record_policy_block("pii-rule")
        assert state.total_policy_blocks == 1
