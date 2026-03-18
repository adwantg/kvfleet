"""Shadow traffic — mirror requests to candidate models for offline comparison."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from kvfleet.adapters.base import ChatRequest, ChatResponse, InferenceAdapter

logger = logging.getLogger(__name__)


@dataclass
class ShadowResult:
    """Result of a shadow traffic request."""

    model: str
    response: ChatResponse | None = None
    error: str | None = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ShadowComparison:
    """Comparison between primary and shadow responses."""

    request_id: str
    primary_model: str
    primary_response: str
    shadow_results: list[ShadowResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class ShadowTrafficManager:
    """Manages shadow traffic for model evaluation.

    Features:
    - Mirror a configurable percentage of traffic to candidate models
    - Keep user-facing answer on the stable/primary model
    - Log shadow outputs for offline comparison
    - Non-blocking — shadow requests run in background
    """

    def __init__(
        self,
        sample_rate: float = 0.1,
        shadow_models: list[str] | None = None,
        log_outputs: bool = True,
        enabled: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.shadow_models = shadow_models or []
        self.log_outputs = log_outputs
        self.enabled = enabled
        self._results: list[ShadowComparison] = []
        self._request_count = 0
        self._shadow_count = 0

    def should_shadow(self) -> bool:
        """Determine if the current request should be shadowed."""
        if not self.enabled or not self.shadow_models:
            return False
        import random

        return random.random() < self.sample_rate

    async def execute_shadow(
        self,
        request: ChatRequest,
        primary_model: str,
        primary_response: ChatResponse,
        adapters: dict[str, InferenceAdapter],
        request_id: str = "",
    ) -> ShadowComparison:
        """Execute shadow requests to candidate models.

        This runs in the background and does not block the primary response.

        Args:
            request: The original request.
            primary_model: The model that served the user.
            primary_response: The primary response.
            adapters: Model name → adapter map.
            request_id: Request identifier.

        Returns:
            ShadowComparison with all shadow results.
        """
        self._request_count += 1
        comparison = ShadowComparison(
            request_id=request_id,
            primary_model=primary_model,
            primary_response=primary_response.content,
        )

        shadow_tasks = []
        for model_name in self.shadow_models:
            if model_name == primary_model or model_name not in adapters:
                continue
            shadow_tasks.append(self._shadow_request(model_name, adapters[model_name], request))

        if shadow_tasks:
            results = await asyncio.gather(*shadow_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, ShadowResult):
                    comparison.shadow_results.append(result)
                    self._shadow_count += 1
                elif isinstance(result, Exception):
                    logger.warning("Shadow request failed: %s", result)

        if self.log_outputs:
            self._results.append(comparison)
            logger.info(
                "Shadow comparison: primary=%s, shadows=%d",
                primary_model,
                len(comparison.shadow_results),
            )

        return comparison

    async def _shadow_request(
        self,
        model_name: str,
        adapter: InferenceAdapter,
        request: ChatRequest,
    ) -> ShadowResult:
        """Execute a single shadow request."""
        start = time.monotonic()
        try:
            shadow_req = ChatRequest(
                messages=request.messages,
                model=model_name,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stream=False,
            )
            response = await adapter.chat(shadow_req)
            latency = (time.monotonic() - start) * 1000
            return ShadowResult(model=model_name, response=response, latency_ms=latency)
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return ShadowResult(model=model_name, error=str(e), latency_ms=latency)

    def get_comparisons(self, last_n: int = 100) -> list[ShadowComparison]:
        """Get recent shadow comparisons."""
        return self._results[-last_n:]

    def stats(self) -> dict[str, Any]:
        """Return shadow traffic statistics."""
        return {
            "enabled": self.enabled,
            "sample_rate": self.sample_rate,
            "shadow_models": self.shadow_models,
            "total_requests": self._request_count,
            "shadow_requests": self._shadow_count,
            "stored_comparisons": len(self._results),
        }

    def clear_results(self) -> None:
        """Clear stored comparisons."""
        self._results.clear()
