"""Rate limit awareness — track and respect provider rate limits."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RateLimitState:
    """Current rate limit state for an endpoint/model."""

    endpoint: str
    model_id: str = ""

    # Limits (from provider headers)
    requests_per_minute: int = 0       # 0 = unknown/unlimited
    tokens_per_minute: int = 0
    requests_per_day: int = 0

    # Current usage (tracked by us)
    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    requests_today: int = 0

    # Remaining (from provider response headers)
    remaining_requests: int | None = None
    remaining_tokens: int | None = None

    # Retry-after (from 429 responses)
    retry_after_seconds: float = 0.0
    retry_after_until: float = 0.0      # timestamp

    # Timestamps
    minute_window_start: float = field(default_factory=time.time)
    day_window_start: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    @property
    def is_throttled(self) -> bool:
        """Check if currently rate limited (429 cooldown)."""
        return time.time() < self.retry_after_until

    @property
    def usage_pct(self) -> float:
        """Per-minute request usage as percentage (0-1)."""
        if self.requests_per_minute <= 0:
            return 0.0
        self._maybe_reset_minute()
        return self.requests_this_minute / self.requests_per_minute

    @property
    def token_usage_pct(self) -> float:
        """Per-minute token usage as percentage (0-1)."""
        if self.tokens_per_minute <= 0:
            return 0.0
        self._maybe_reset_minute()
        return self.tokens_this_minute / self.tokens_per_minute

    @property
    def available_capacity(self) -> float:
        """Available capacity as 0-1 score (1 = fully available, 0 = exhausted)."""
        if self.is_throttled:
            return 0.0
        if self.remaining_requests is not None and self.requests_per_minute > 0:
            return self.remaining_requests / self.requests_per_minute
        return max(0.0, 1.0 - self.usage_pct)

    def _maybe_reset_minute(self) -> None:
        """Reset minute window if expired."""
        now = time.time()
        if now - self.minute_window_start > 60:
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
            self.minute_window_start = now
        if now - self.day_window_start > 86400:
            self.requests_today = 0
            self.day_window_start = now


class RateLimitTracker:
    """Track rate limits across all endpoints.

    Thread-safe. Parses rate limit headers from provider responses
    and tracks local request counts.
    """

    def __init__(
        self,
        default_rpm: int = 0,
        default_tpm: int = 0,
        throttle_threshold: float = 0.85,
        cooldown_seconds: float = 10.0,
    ) -> None:
        self._states: dict[str, RateLimitState] = {}
        self._lock = threading.Lock()
        self.default_rpm = default_rpm
        self.default_tpm = default_tpm
        self.throttle_threshold = throttle_threshold
        self.cooldown_seconds = cooldown_seconds

    def record_request(
        self,
        endpoint: str,
        model_id: str = "",
        tokens_used: int = 0,
    ) -> None:
        """Record a request for rate tracking."""
        key = f"{endpoint}:{model_id}"
        with self._lock:
            state = self._get_or_create(key, endpoint, model_id)
            state._maybe_reset_minute()
            state.requests_this_minute += 1
            state.requests_today += 1
            state.tokens_this_minute += tokens_used
            state.last_updated = time.time()

    def record_rate_limit_headers(
        self,
        endpoint: str,
        model_id: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        """Parse rate limit info from response headers.

        Supports standard headers:
        - x-ratelimit-limit-requests
        - x-ratelimit-limit-tokens
        - x-ratelimit-remaining-requests
        - x-ratelimit-remaining-tokens
        - retry-after
        """
        if not headers:
            return
        key = f"{endpoint}:{model_id}"
        with self._lock:
            state = self._get_or_create(key, endpoint, model_id)

            if "x-ratelimit-limit-requests" in headers:
                state.requests_per_minute = int(headers["x-ratelimit-limit-requests"])
            if "x-ratelimit-limit-tokens" in headers:
                state.tokens_per_minute = int(headers["x-ratelimit-limit-tokens"])
            if "x-ratelimit-remaining-requests" in headers:
                state.remaining_requests = int(headers["x-ratelimit-remaining-requests"])
            if "x-ratelimit-remaining-tokens" in headers:
                state.remaining_tokens = int(headers["x-ratelimit-remaining-tokens"])
            if "retry-after" in headers:
                try:
                    retry_secs = float(headers["retry-after"])
                    state.retry_after_seconds = retry_secs
                    state.retry_after_until = time.time() + retry_secs
                except ValueError:
                    pass

            state.last_updated = time.time()

    def record_429(self, endpoint: str, model_id: str = "", retry_after: float = 0) -> None:
        """Record a 429 rate limit response."""
        key = f"{endpoint}:{model_id}"
        cooldown = retry_after if retry_after > 0 else self.cooldown_seconds
        with self._lock:
            state = self._get_or_create(key, endpoint, model_id)
            state.retry_after_seconds = cooldown
            state.retry_after_until = time.time() + cooldown
            state.last_updated = time.time()
        logger.warning(
            "Rate limited: %s/%s — cooldown %.1fs",
            endpoint, model_id, cooldown,
        )

    def should_throttle(self, endpoint: str, model_id: str = "") -> bool:
        """Check if requests to this endpoint should be throttled."""
        key = f"{endpoint}:{model_id}"
        with self._lock:
            state = self._states.get(key)
            if not state:
                return False
            if state.is_throttled:
                return True
            return state.usage_pct >= self.throttle_threshold

    def get_capacity_score(self, endpoint: str, model_id: str = "") -> float:
        """Get available capacity as routing signal (0-1, higher = more available)."""
        key = f"{endpoint}:{model_id}"
        with self._lock:
            state = self._states.get(key)
            if not state:
                return 1.0  # Unknown = assume available
            return state.available_capacity

    def get_state(self, endpoint: str, model_id: str = "") -> RateLimitState | None:
        """Get rate limit state for an endpoint."""
        key = f"{endpoint}:{model_id}"
        with self._lock:
            return self._states.get(key)

    def get_all_states(self) -> dict[str, RateLimitState]:
        """Get all tracked rate limit states."""
        with self._lock:
            return dict(self._states)

    def summary(self) -> dict[str, Any]:
        """Summary of rate limit states."""
        with self._lock:
            result = {}
            for key, state in self._states.items():
                result[key] = {
                    "usage_pct": f"{state.usage_pct:.0%}",
                    "throttled": state.is_throttled,
                    "capacity": f"{state.available_capacity:.0%}",
                    "rpm": state.requests_per_minute,
                    "requests_this_min": state.requests_this_minute,
                }
            return result

    def _get_or_create(self, key: str, endpoint: str, model_id: str) -> RateLimitState:
        if key not in self._states:
            self._states[key] = RateLimitState(
                endpoint=endpoint,
                model_id=model_id,
                requests_per_minute=self.default_rpm,
                tokens_per_minute=self.default_tpm,
            )
        return self._states[key]
