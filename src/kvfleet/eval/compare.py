"""Model comparison suite and offline replay engine."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from kvfleet.adapters.base import ChatRequest, ChatResponse, InferenceAdapter

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing multiple models on the same prompt."""

    prompt: str
    results: dict[str, ChatResponse] = field(default_factory=dict)  # model → response
    latencies: dict[str, float] = field(default_factory=dict)  # model → ms
    errors: dict[str, str] = field(default_factory=dict)  # model → error
    timestamp: float = field(default_factory=time.time)


class ModelComparator:
    """Compare multiple models on the same prompts."""

    async def compare(
        self,
        request: ChatRequest,
        adapters: dict[str, InferenceAdapter],
        model_names: list[str] | None = None,
    ) -> ComparisonResult:
        """Run the same request against multiple models.

        Args:
            request: The chat request.
            adapters: Model name → adapter map.
            model_names: Models to compare (defaults to all).

        Returns:
            ComparisonResult with all model responses.
        """
        models = model_names or list(adapters.keys())
        result = ComparisonResult(
            prompt=request.messages[-1].content if request.messages else "",
        )

        tasks = {}
        for name in models:
            adapter = adapters.get(name)
            if adapter:
                tasks[name] = asyncio.create_task(self._run(name, adapter, request))

        for name, task in tasks.items():
            try:
                response, latency = await task
                result.results[name] = response
                result.latencies[name] = latency
            except Exception as e:
                result.errors[name] = str(e)

        return result

    async def _run(
        self, name: str, adapter: InferenceAdapter, request: ChatRequest,
    ) -> tuple[ChatResponse, float]:
        start = time.monotonic()
        response = await adapter.chat(request)
        latency = (time.monotonic() - start) * 1000
        return response, latency


@dataclass
class ReplayRecord:
    """A recorded request for replay."""

    request: ChatRequest
    original_model: str
    original_response: str = ""
    original_latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayResult:
    """Result of replaying a recorded request."""

    record: ReplayRecord
    responses: dict[str, ChatResponse] = field(default_factory=dict)
    latencies: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    cost_comparison: dict[str, float] = field(default_factory=dict)


class ReplayEngine:
    """Replay production prompt logs against alternate models.

    Features:
    - Replay recorded requests against candidate models
    - Compare cost, latency, and quality across models
    - Estimate rollout impact before deployment
    """

    def __init__(self) -> None:
        self._records: list[ReplayRecord] = []
        self._comparator = ModelComparator()

    def record(self, request: ChatRequest, model: str, response: ChatResponse) -> None:
        """Record a request for future replay."""
        self._records.append(ReplayRecord(
            request=request,
            original_model=model,
            original_response=response.content,
            original_latency_ms=response.latency_ms,
        ))

    async def replay(
        self,
        adapters: dict[str, InferenceAdapter],
        model_names: list[str] | None = None,
        limit: int = 100,
    ) -> list[ReplayResult]:
        """Replay recorded requests against specified models.

        Args:
            adapters: Model name → adapter map.
            model_names: Models to replay against.
            limit: Max number of records to replay.

        Returns:
            List of ReplayResults.
        """
        results: list[ReplayResult] = []
        records = self._records[-limit:]

        for record in records:
            comparison = await self._comparator.compare(
                record.request, adapters, model_names,
            )
            result = ReplayResult(
                record=record,
                responses=comparison.results,
                latencies=comparison.latencies,
                errors=comparison.errors,
            )
            results.append(result)

        return results

    def get_records(self, last_n: int = 100) -> list[ReplayRecord]:
        """Get recent recorded requests."""
        return self._records[-last_n:]

    def clear_records(self) -> None:
        """Clear all recorded requests."""
        self._records.clear()

    @property
    def record_count(self) -> int:
        """Number of recorded requests."""
        return len(self._records)


@dataclass
class CalibrationResult:
    """Result of route calibration."""

    original_weights: dict[str, float]
    recommended_weights: dict[str, float]
    improvement_pct: float = 0.0
    sample_size: int = 0


class RouteCalibrator:
    """Calibrate routing weights based on observed outcomes.

    Analyzes replay results to adjust scoring thresholds and weights.
    """

    def calibrate_from_replay(
        self,
        replay_results: list[ReplayResult],
        current_weights: dict[str, float],
    ) -> CalibrationResult:
        """Analyze replay results and suggest weight adjustments.

        Args:
            replay_results: Results from replay engine.
            current_weights: Current scoring weights.

        Returns:
            CalibrationResult with recommended weights.
        """
        # Simple heuristic: find the model with best latency/quality ratio
        # and adjust weights to favor similar characteristics
        recommended = dict(current_weights)

        if not replay_results:
            return CalibrationResult(
                original_weights=current_weights,
                recommended_weights=recommended,
                sample_size=0,
            )

        # Count which models performed best on each dimension
        latency_wins: dict[str, int] = {}
        quality_wins: dict[str, int] = {}

        for result in replay_results:
            if result.latencies:
                fastest = min(result.latencies, key=result.latencies.get)  # type: ignore[arg-type]
                latency_wins[fastest] = latency_wins.get(fastest, 0) + 1

        # Adjust weights based on observed patterns
        total = len(replay_results)
        if total > 0:
            # If latency is consistently the differentiator, increase its weight
            latency_variance = len(set(latency_wins.values())) if latency_wins else 0
            if latency_variance > 1:
                recommended["latency"] = min(0.5, current_weights.get("latency", 0.3) * 1.1)

        return CalibrationResult(
            original_weights=current_weights,
            recommended_weights=recommended,
            sample_size=total,
        )
