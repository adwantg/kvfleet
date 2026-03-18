"""Prometheus metrics exporter for kvfleet routing decisions."""

from __future__ import annotations

import logging
from typing import Any

try:
    from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

logger = logging.getLogger(__name__)


class MetricsExporter:
    """Exports kvfleet routing metrics via Prometheus.

    Metrics exported:
    - kvfleet_route_requests_total: Total routing requests
    - kvfleet_route_latency_seconds: Routing decision latency
    - kvfleet_model_selected_total: Models selected count
    - kvfleet_fallback_triggered_total: Fallback trigger count
    - kvfleet_cache_affinity_hits_total: Cache affinity hits
    - kvfleet_policy_blocks_total: Policy blocks
    - kvfleet_shadow_requests_total: Shadow traffic count
    - kvfleet_model_health: Model health gauge (0/1)
    - kvfleet_model_queue_depth: Queue depth gauge
    """

    def __init__(self, port: int = 9090, enabled: bool = True) -> None:
        self.port = port
        self.enabled = enabled and HAS_PROMETHEUS
        self._started = False

        if self.enabled:
            self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize Prometheus metric objects."""
        self.route_requests = Counter(
            "kvfleet_route_requests_total",
            "Total routing requests",
            ["strategy", "status"],
        )
        self.route_latency = Histogram(
            "kvfleet_route_latency_seconds",
            "Routing decision latency",
            ["strategy"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )
        self.model_selected = Counter(
            "kvfleet_model_selected_total",
            "Number of times each model was selected",
            ["model"],
        )
        self.fallback_triggered = Counter(
            "kvfleet_fallback_triggered_total",
            "Number of fallback triggers",
            ["from_model", "to_model"],
        )
        self.cache_hits = Counter(
            "kvfleet_cache_affinity_hits_total",
            "Cache affinity hits",
            ["type"],  # "session", "prefix", "hash"
        )
        self.policy_blocks = Counter(
            "kvfleet_policy_blocks_total",
            "Requests blocked by policy",
            ["rule"],
        )
        self.shadow_requests = Counter(
            "kvfleet_shadow_requests_total",
            "Shadow traffic requests",
            ["model"],
        )
        self.model_health = Gauge(
            "kvfleet_model_health",
            "Model endpoint health (1=healthy, 0=unhealthy)",
            ["model", "endpoint"],
        )
        self.model_queue_depth = Gauge(
            "kvfleet_model_queue_depth",
            "Model endpoint queue depth",
            ["model", "endpoint"],
        )
        self.model_gpu_usage = Gauge(
            "kvfleet_model_gpu_memory_pct",
            "GPU memory usage percentage",
            ["model", "endpoint"],
        )
        self.fleet_info = Info(
            "kvfleet_fleet",
            "Fleet information",
        )

    def start_server(self) -> None:
        """Start the Prometheus metrics HTTP server."""
        if not self.enabled or self._started:
            return
        try:
            start_http_server(self.port)
            self._started = True
            logger.info("Prometheus metrics server started on port %d", self.port)
        except OSError as e:
            logger.warning("Could not start metrics server on port %d: %s", self.port, e)

    def record_route(self, strategy: str, model: str, latency_seconds: float, success: bool = True) -> None:
        """Record a routing decision."""
        if not self.enabled:
            return
        self.route_requests.labels(strategy=strategy, status="success" if success else "error").inc()
        self.route_latency.labels(strategy=strategy).observe(latency_seconds)
        self.model_selected.labels(model=model).inc()

    def record_fallback(self, from_model: str, to_model: str) -> None:
        """Record a fallback event."""
        if not self.enabled:
            return
        self.fallback_triggered.labels(from_model=from_model, to_model=to_model).inc()

    def record_cache_hit(self, hit_type: str) -> None:
        """Record a cache affinity hit."""
        if not self.enabled:
            return
        self.cache_hits.labels(type=hit_type).inc()

    def record_policy_block(self, rule: str) -> None:
        """Record a policy block."""
        if not self.enabled:
            return
        self.policy_blocks.labels(rule=rule).inc()

    def record_shadow(self, model: str) -> None:
        """Record a shadow traffic request."""
        if not self.enabled:
            return
        self.shadow_requests.labels(model=model).inc()

    def update_health(self, model: str, endpoint: str, healthy: bool, queue_depth: int = 0, gpu_pct: float = 0.0) -> None:
        """Update health gauges for a model endpoint."""
        if not self.enabled:
            return
        self.model_health.labels(model=model, endpoint=endpoint).set(1 if healthy else 0)
        self.model_queue_depth.labels(model=model, endpoint=endpoint).set(queue_depth)
        self.model_gpu_usage.labels(model=model, endpoint=endpoint).set(gpu_pct)

    def set_fleet_info(self, **kwargs: str) -> None:
        """Set fleet info labels."""
        if not self.enabled:
            return
        self.fleet_info.info(kwargs)
