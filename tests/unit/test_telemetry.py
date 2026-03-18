"""Tests for telemetry, health manager, metrics, and GPU state."""

import pytest

from kvfleet.adapters.base import EndpointHealth
from kvfleet.telemetry.health import HealthManager
from kvfleet.telemetry.metrics import MetricsExporter
from kvfleet.telemetry.gpu import GPUState, GPUStateAggregator


# ───────────────────────── Health Manager ─────────────────────────


class TestHealthManager:
    def test_healthy_endpoint(self):
        manager = HealthManager()
        manager.update_health(EndpointHealth(endpoint="http://a:8000", healthy=True, last_checked=1e18))
        assert manager.is_healthy("http://a:8000")

    def test_unhealthy_endpoint(self):
        manager = HealthManager(stale_threshold_seconds=999999)
        manager.update_health(EndpointHealth(endpoint="http://a:8000", healthy=False, last_checked=1e18))
        assert not manager.is_healthy("http://a:8000")

    def test_circuit_breaker(self):
        manager = HealthManager(failure_threshold=2, recovery_timeout_seconds=3600, stale_threshold_seconds=999999)
        for _ in range(3):
            manager.update_health(EndpointHealth(endpoint="http://a:8000", healthy=False, last_checked=1e18))
        assert not manager.is_healthy("http://a:8000")

    def test_unknown_is_optimistic(self):
        manager = HealthManager()
        assert manager.is_healthy("http://unknown:8000")

    def test_filter_healthy(self):
        manager = HealthManager(stale_threshold_seconds=999999)
        manager.update_health(EndpointHealth(endpoint="http://a:8000", healthy=True, last_checked=1e18))
        manager.update_health(EndpointHealth(endpoint="http://b:8000", healthy=False, last_checked=1e18))
        healthy = manager.get_healthy_endpoints(["http://a:8000", "http://b:8000", "http://c:8000"])
        assert "http://a:8000" in healthy
        assert "http://b:8000" not in healthy
        assert "http://c:8000" in healthy  # unknown = optimistic

    def test_load_scores(self):
        manager = HealthManager()
        manager.update_health(EndpointHealth(
            endpoint="http://a:8000", healthy=True, queue_depth=50, active_requests=25,
        ))
        scores = manager.get_load_scores(["http://a:8000"])
        assert 0.0 <= scores["http://a:8000"] <= 1.0

    def test_warm_detection(self):
        manager = HealthManager()
        manager.update_health(EndpointHealth(
            endpoint="http://a:8000", healthy=True, active_requests=5, tokens_per_second=100,
        ))
        assert manager.is_warm("http://a:8000")
        assert not manager.is_warm("http://unknown:8000")

    def test_summary(self):
        manager = HealthManager()
        manager.update_health(EndpointHealth(endpoint="http://a:8000", healthy=True))
        summary = manager.summary()
        assert summary["tracked_endpoints"] == 1
        assert summary["healthy"] == 1


# ───────────────────────── GPU State ─────────────────────────


class TestGPUState:
    def test_has_headroom(self):
        state = GPUState(endpoint="x", gpu_memory_used_pct=50.0)
        assert state.has_headroom

    def test_no_headroom(self):
        state = GPUState(endpoint="x", gpu_memory_used_pct=90.0)
        assert not state.has_headroom

    def test_load_factor(self):
        state = GPUState(
            endpoint="x",
            gpu_memory_used_pct=50.0,
            gpu_utilization_pct=75.0,
            kv_cache_usage_pct=30.0,
        )
        assert 0.0 <= state.load_factor <= 1.0


class TestGPUStateAggregator:
    def test_update_from_health(self):
        agg = GPUStateAggregator()
        state = agg.update_from_health("http://a:8000", {
            "gpu_memory_used_pct": 60.0,
            "gpu_utilization_pct": 70.0,
        })
        assert state.gpu_memory_used_pct == 60.0
        assert agg.get_state("http://a:8000") is not None

    def test_load_scores(self):
        agg = GPUStateAggregator()
        agg.update_from_health("http://a:8000", {"gpu_memory_used_pct": 80.0})
        scores = agg.get_load_scores()
        assert "http://a:8000" in scores


# ───────────────────────── Metrics Exporter ─────────────────────────


class TestMetricsExporter:
    def test_disabled_noop(self):
        exporter = MetricsExporter(enabled=False)
        # Should not raise
        exporter.record_route("hybrid", "model-a", 0.1)
        exporter.record_fallback("a", "b")
        exporter.record_cache_hit("session")
        exporter.record_policy_block("pii")
        exporter.record_shadow("model-b")

    def test_enabled_records(self):
        exporter = MetricsExporter(enabled=True, port=0)
        # Should not raise (may fail to start server on port 0, that's ok)
        exporter.record_route("hybrid", "model-a", 0.1)
