"""Telemetry collector — gathers runtime signals from inference backends."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from kvfleet.adapters.base import EndpointHealth, InferenceAdapter

logger = logging.getLogger(__name__)


class TelemetryCollector:
    """Collects and caches telemetry data from inference backends.

    Periodically polls registered endpoints for health, queue depth,
    GPU metrics, and cache state. Data is cached and served to the
    routing engine for hardware-aware decisions.
    """

    def __init__(self, poll_interval_seconds: int = 30) -> None:
        self.poll_interval = poll_interval_seconds
        self._health_cache: dict[str, EndpointHealth] = {}
        self._metrics_cache: dict[str, dict[str, Any]] = {}
        self._cache_state: dict[str, dict[str, Any]] = {}
        self._adapters: dict[str, InferenceAdapter] = {}
        self._running = False
        self._task: asyncio.Task[None] | None = None

    def register_adapter(self, key: str, adapter: InferenceAdapter) -> None:
        """Register an adapter for telemetry collection."""
        self._adapters[key] = adapter

    async def collect_once(self) -> dict[str, EndpointHealth]:
        """Collect telemetry from all registered adapters once."""
        tasks = {
            key: asyncio.create_task(self._collect_from(key, adapter))
            for key, adapter in self._adapters.items()
        }
        for key, task in tasks.items():
            try:
                await task
            except Exception as e:
                logger.error("Telemetry collection failed for %s: %s", key, e)
        return dict(self._health_cache)

    async def _collect_from(self, key: str, adapter: InferenceAdapter) -> None:
        """Collect health, metrics, and cache state from a single adapter."""
        try:
            # E-7: Deduplicate health probes by endpoint
            endpoint = getattr(adapter, "endpoint", key)
            cached = self._health_cache.get(endpoint)
            if cached and (time.time() - cached.last_checked) < 5.0:
                # Recently probed — reuse cached result
                health = cached
            else:
                health = await adapter.health_check()
            self._health_cache[health.endpoint] = health
        except Exception as e:
            self._health_cache[key] = EndpointHealth(
                endpoint=key, healthy=False, error=str(e), last_checked=time.time()
            )

        try:
            metrics = await adapter.get_metrics()
            if metrics:
                self._metrics_cache[key] = metrics
        except Exception:
            pass

        try:
            cache = await adapter.get_cache_state()
            if cache:
                self._cache_state[key] = cache
        except Exception:
            pass

    def get_health(self, endpoint: str) -> EndpointHealth | None:
        """Get cached health for an endpoint."""
        return self._health_cache.get(endpoint)

    def get_all_health(self) -> dict[str, EndpointHealth]:
        """Get all cached health data."""
        return dict(self._health_cache)

    def get_metrics(self, key: str) -> dict[str, Any]:
        """Get cached metrics for an adapter key."""
        return self._metrics_cache.get(key, {})

    def get_cache_state(self, key: str) -> dict[str, Any]:
        """Get cached KV-cache state for an adapter key."""
        return self._cache_state.get(key, {})

    def get_healthy_endpoints(self) -> list[str]:
        """Return list of healthy endpoints."""
        return [ep for ep, h in self._health_cache.items() if h.healthy]

    async def start_polling(self) -> None:
        """Start periodic telemetry collection."""
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop_polling(self) -> None:
        """Stop periodic telemetry collection."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _poll_loop(self) -> None:
        """Periodic polling loop."""
        while self._running:
            await self.collect_once()
            await asyncio.sleep(self.poll_interval)

    def summary(self) -> dict[str, Any]:
        """Return telemetry summary."""
        return {
            "registered_adapters": len(self._adapters),
            "cached_health_entries": len(self._health_cache),
            "healthy_endpoints": len(self.get_healthy_endpoints()),
            "poll_interval_seconds": self.poll_interval,
        }
