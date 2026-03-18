"""Multi-objective scoring engine for model candidate ranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from kvfleet.adapters.base import EndpointHealth
from kvfleet.config.schema import ModelConfig, ScoringWeights
from kvfleet.router.explain import CandidateScore

logger = logging.getLogger(__name__)


@dataclass
class ScoringContext:
    """Context passed to the scoring engine for each request."""

    data_class: str = "internal"
    required_capabilities: dict[str, bool] | None = None
    tenant_id: str | None = None
    cache_affinity_scores: dict[str, float] | None = None  # model_name → affinity score
    endpoint_health: dict[str, EndpointHealth] | None = None  # endpoint → health
    budget_remaining_pct: float = 100.0
    tags: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None


class ScoringEngine:
    """Multi-objective scoring engine.

    Scores each candidate model on multiple dimensions and produces
    a weighted composite score for ranking.
    """

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights or ScoringWeights()

    def score_candidates(
        self,
        candidates: list[ModelConfig],
        context: ScoringContext | None = None,
    ) -> list[CandidateScore]:
        """Score all candidate models.

        Args:
            candidates: List of candidate model configs.
            context: Scoring context with runtime signals.

        Returns:
            List of CandidateScore sorted by total_score descending.
        """
        ctx = context or ScoringContext()
        scores: list[CandidateScore] = []

        for model in candidates:
            score = self._score_model(model, ctx)
            scores.append(score)

        # Sort by total score descending
        scores.sort(key=lambda s: s.total_score, reverse=True)

        # Mark the top as selected
        if scores:
            scores[0].selected = True

        return scores

    def _score_model(self, model: ModelConfig, ctx: ScoringContext) -> CandidateScore:
        """Score a single model across all dimensions."""
        score = CandidateScore(model_name=model.name, endpoint=model.endpoint)

        # Cost score (lower cost = higher score, normalized 0-1)
        score.cost_score = self._score_cost(model)

        # Latency score (lower latency = higher score, normalized 0-1)
        score.latency_score = self._score_latency(model)

        # Quality score (direct from model config, 0-1)
        score.quality_score = model.quality_score

        # Cache affinity score
        if ctx.cache_affinity_scores and model.name in ctx.cache_affinity_scores:
            score.cache_affinity_score = ctx.cache_affinity_scores[model.name]

        # Hardware load score (lower load = higher score)
        score.hardware_load_score = self._score_hardware(model, ctx)

        # Compliance score
        score.compliance_score = self._score_compliance(model, ctx)

        # Compute weighted total
        score.total_score = (
            self.weights.cost * score.cost_score
            + self.weights.latency * score.latency_score
            + self.weights.quality * score.quality_score
            + self.weights.cache_affinity * score.cache_affinity_score
            + self.weights.hardware_load * score.hardware_load_score
            + self.weights.compliance * score.compliance_score
        )

        # Add signal details
        score.signals = {
            "cost_per_1k": model.cost_per_1k_input_tokens,
            "latency_p50": model.latency_p50_ms,
            "quality_prior": model.quality_score,
            "weight": model.weight,
        }

        return score

    def _score_cost(self, model: ModelConfig) -> float:
        """Score cost (0-1, higher = cheaper)."""
        cost = model.cost_per_1k_input_tokens
        if cost <= 0:
            return 1.0  # Free
        # Normalize: $0.01/1K → 0.9, $0.10/1K → 0.5, $1.00/1K → 0.1
        return max(0.0, min(1.0, 1.0 - (cost / 1.0)))

    def _score_latency(self, model: ModelConfig) -> float:
        """Score latency (0-1, higher = faster)."""
        p50 = model.latency_p50_ms
        if p50 <= 0:
            return 1.0
        # Normalize: 100ms → 0.95, 500ms → 0.75, 2000ms → 0.33, 5000ms → 0.0
        return max(0.0, min(1.0, 1.0 - (p50 / 5000.0)))

    def _score_hardware(self, model: ModelConfig, ctx: ScoringContext) -> float:
        """Score hardware load (0-1, higher = less loaded)."""
        if not ctx.endpoint_health:
            return 0.5  # Neutral when no data
        health = ctx.endpoint_health.get(model.endpoint)
        if not health or not health.healthy:
            return 0.0
        return max(0.0, 1.0 - health.load_score)

    def _score_compliance(self, model: ModelConfig, ctx: ScoringContext) -> float:
        """Score compliance fit (0-1)."""
        if ctx.data_class not in model.allowed_data_classes:
            return 0.0
        return 1.0

    def update_weights(self, **kwargs: float) -> None:
        """Update scoring weights dynamically."""
        for key, value in kwargs.items():
            if hasattr(self.weights, key):
                setattr(self.weights, key, value)
