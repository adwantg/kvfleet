"""vLLM inference adapter.

Extends OpenAI-compatible adapter with vLLM-specific features:
- KV-cache state inspection via /metrics endpoint
- GPU utilization metrics
- Prefix caching awareness
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from kvfleet.adapters.base import EndpointHealth
from kvfleet.adapters.openai_compat import OpenAICompatAdapter

logger = logging.getLogger(__name__)


class VLLMAdapter(OpenAICompatAdapter):
    """Adapter for vLLM inference servers.

    vLLM exposes an OpenAI-compatible API plus additional metrics
    at /metrics (Prometheus format) that we parse for routing signals.
    """

    async def health_check(self) -> EndpointHealth:
        """Enhanced health check with vLLM metrics."""
        client = self._get_client()
        start = time.monotonic()
        health = EndpointHealth(
            endpoint=self.endpoint,
            healthy=False,
            last_checked=time.time(),
        )

        # Basic health check
        try:
            resp = await client.get("/health")
            health.healthy = resp.status_code == 200
            health.latency_ms = (time.monotonic() - start) * 1000
        except httpx.RequestError as e:
            health.error = str(e)
            return health

        # Parse Prometheus metrics for routing signals
        try:
            metrics_resp = await client.get("/metrics")
            if metrics_resp.status_code == 200:
                metrics = self._parse_prometheus_metrics(metrics_resp.text)
                health.queue_depth = int(metrics.get("vllm:num_requests_waiting", 0))
                health.active_requests = int(metrics.get("vllm:num_requests_running", 0))
                health.gpu_memory_used_pct = metrics.get("vllm:gpu_cache_usage_perc", 0.0) * 100
                health.kv_cache_usage_pct = metrics.get("vllm:gpu_cache_usage_perc", 0.0) * 100
                health.tokens_per_second = metrics.get(
                    "vllm:avg_generation_throughput_toks_per_s", 0.0
                )
        except httpx.RequestError:
            pass  # Metrics are optional

        return health

    async def get_cache_state(self) -> dict[str, Any]:
        """Get vLLM KV-cache state from metrics endpoint."""
        client = self._get_client()
        try:
            resp = await client.get("/metrics")
            if resp.status_code == 200:
                metrics = self._parse_prometheus_metrics(resp.text)
                return {
                    "kv_cache_usage_pct": metrics.get("vllm:gpu_cache_usage_perc", 0.0) * 100,
                    "cpu_cache_usage_pct": metrics.get("vllm:cpu_cache_usage_perc", 0.0) * 100,
                    "num_requests_waiting": int(metrics.get("vllm:num_requests_waiting", 0)),
                    "num_requests_running": int(metrics.get("vllm:num_requests_running", 0)),
                    "prefix_cache_hit_rate": metrics.get("vllm:prefix_cache_hit_rate", 0.0),
                }
        except httpx.RequestError:
            pass
        return {}

    async def get_metrics(self) -> dict[str, Any]:
        """Get full vLLM metrics."""
        client = self._get_client()
        try:
            resp = await client.get("/metrics")
            if resp.status_code == 200:
                return self._parse_prometheus_metrics(resp.text)
        except httpx.RequestError:
            pass
        return {}

    @staticmethod
    def _parse_prometheus_metrics(text: str) -> dict[str, float]:
        """Parse Prometheus text format into a dict of metric name → value."""
        metrics: dict[str, float] = {}
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                try:
                    value = float(parts[-1])
                    # Use a simplified key (replace colons, strip labels)
                    key = name.split("{")[0]
                    metrics[key] = value
                except ValueError:
                    continue
        return metrics
