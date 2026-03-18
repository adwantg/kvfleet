"""Health check manager for inference endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from kvfleet.adapters.base import EndpointHealth, InferenceAdapter

logger = logging.getLogger(__name__)


class HealthManager:
    """Manages health state for all inference endpoints.

    Features:
    - Periodic async health polling
    - Circuit breaker pattern (mark unhealthy after N failures)
    - Warm/cold model detection
    - Stale health data detection
    """

    def __init__(
        self,
        check_interval_seconds: int = 30,
        failure_threshold: int = 3,
        recovery_timeout_seconds: int = 60,
        stale_threshold_seconds: int = 120,
    ) -> None:
        self.check_interval = check_interval_seconds
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.stale_threshold = stale_threshold_seconds
        self._health: dict[str, EndpointHealth] = {}
        self._failure_counts: dict[str, int] = {}
        self._circuit_open_until: dict[str, float] = {}

    def update_health(self, health: EndpointHealth) -> None:
        """Update health status for an endpoint."""
        ep = health.endpoint
        if health.healthy:
            self._failure_counts[ep] = 0
            self._circuit_open_until.pop(ep, None)
        else:
            self._failure_counts[ep] = self._failure_counts.get(ep, 0) + 1
            if self._failure_counts[ep] >= self.failure_threshold:
                self._circuit_open_until[ep] = time.time() + self.recovery_timeout
                logger.warning("Circuit breaker OPEN for %s (failures: %d)", ep, self._failure_counts[ep])
        self._health[ep] = health

    def is_healthy(self, endpoint: str) -> bool:
        """Check if an endpoint is healthy and circuit is closed."""
        # Check circuit breaker
        if endpoint in self._circuit_open_until:
            if time.time() < self._circuit_open_until[endpoint]:
                return False
            # Recovery timeout elapsed — allow probe
            del self._circuit_open_until[endpoint]

        health = self._health.get(endpoint)
        if health is None:
            return True  # Unknown = optimistic
        if time.time() - health.last_checked > self.stale_threshold:
            return True  # Stale = optimistic
        return health.healthy

    def is_warm(self, endpoint: str) -> bool:
        """Check if a model/endpoint is warm (recently active)."""
        health = self._health.get(endpoint)
        if health is None:
            return False
        return health.active_requests > 0 or health.tokens_per_second > 0

    def get_health(self, endpoint: str) -> EndpointHealth | None:
        """Get health data for an endpoint."""
        return self._health.get(endpoint)

    def get_healthy_endpoints(self, endpoints: list[str]) -> list[str]:
        """Filter to only healthy endpoints."""
        return [ep for ep in endpoints if self.is_healthy(ep)]

    def get_load_scores(self, endpoints: list[str]) -> dict[str, float]:
        """Get load scores for endpoints (0-1 scale, higher = more loaded)."""
        scores: dict[str, float] = {}
        for ep in endpoints:
            health = self._health.get(ep)
            if health:
                scores[ep] = health.load_score
            else:
                scores[ep] = 0.5  # Neutral for unknown
        return scores

    def summary(self) -> dict[str, Any]:
        """Health manager summary."""
        return {
            "tracked_endpoints": len(self._health),
            "healthy": sum(1 for h in self._health.values() if h.healthy),
            "unhealthy": sum(1 for h in self._health.values() if not h.healthy),
            "circuit_breakers_open": len(self._circuit_open_until),
        }
