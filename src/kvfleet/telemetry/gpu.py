"""GPU state reader — runtime GPU telemetry from inference backends."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GPUState:
    """Snapshot of GPU state for an endpoint."""

    endpoint: str
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_pct: float = 0.0
    gpu_memory_used_pct: float = 0.0
    kv_cache_usage_pct: float = 0.0
    active_batch_size: int = 0
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_headroom(self) -> bool:
        """Check if GPU has capacity headroom (< 80% memory)."""
        return self.gpu_memory_used_pct < 80.0

    @property
    def load_factor(self) -> float:
        """Compute a 0-1 load factor for scoring."""
        mem = self.gpu_memory_used_pct / 100.0
        util = self.gpu_utilization_pct / 100.0
        cache = self.kv_cache_usage_pct / 100.0
        return min(1.0, (mem * 0.4 + util * 0.3 + cache * 0.3))


class GPUStateAggregator:
    """Aggregates GPU state from telemetry data.

    Converts raw metrics (from vLLM /metrics, Triton stats, etc.)
    into structured GPUState objects for the scoring engine.
    """

    def __init__(self) -> None:
        self._states: dict[str, GPUState] = {}

    def update_from_health(self, endpoint: str, health_data: dict[str, Any]) -> GPUState:
        """Update GPU state from health check data."""
        state = GPUState(
            endpoint=endpoint,
            gpu_memory_used_pct=health_data.get("gpu_memory_used_pct", 0.0),
            gpu_utilization_pct=health_data.get("gpu_utilization_pct", 0.0),
            kv_cache_usage_pct=health_data.get("kv_cache_usage_pct", 0.0),
            active_batch_size=health_data.get("active_requests", 0),
            tokens_per_second=health_data.get("tokens_per_second", 0.0),
        )
        self._states[endpoint] = state
        return state

    def update_from_vllm_metrics(self, endpoint: str, metrics: dict[str, float]) -> GPUState:
        """Update GPU state from vLLM Prometheus metrics."""
        state = GPUState(
            endpoint=endpoint,
            gpu_memory_used_pct=metrics.get("vllm:gpu_cache_usage_perc", 0.0) * 100,
            kv_cache_usage_pct=metrics.get("vllm:gpu_cache_usage_perc", 0.0) * 100,
            active_batch_size=int(metrics.get("vllm:num_requests_running", 0)),
            tokens_per_second=metrics.get("vllm:avg_generation_throughput_toks_per_s", 0.0),
        )
        self._states[endpoint] = state
        return state

    def get_state(self, endpoint: str) -> GPUState | None:
        """Get GPU state for an endpoint."""
        return self._states.get(endpoint)

    def get_all_states(self) -> dict[str, GPUState]:
        """Get all GPU states."""
        return dict(self._states)

    def get_load_scores(self) -> dict[str, float]:
        """Get load factor scores for all endpoints."""
        return {ep: state.load_factor for ep, state in self._states.items()}
