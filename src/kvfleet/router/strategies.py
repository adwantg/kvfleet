"""Routing strategies — pluggable model selection logic."""

from __future__ import annotations

import abc
import logging
import random
from typing import Any

from kvfleet.config.schema import ModelConfig, RouteRuleConfig
from kvfleet.router.explain import CandidateScore
from kvfleet.router.scoring import ScoringContext, ScoringEngine

logger = logging.getLogger(__name__)


class RoutingStrategy(abc.ABC):
    """Abstract base class for routing strategies."""

    name: str = "base"

    @abc.abstractmethod
    def select(
        self,
        candidates: list[ModelConfig],
        context: ScoringContext | None = None,
        **kwargs: Any,
    ) -> list[CandidateScore]:
        """Select and rank models from candidates.

        Args:
            candidates: Available models.
            context: Scoring context with runtime signals.

        Returns:
            Ranked CandidateScores (first = selected).
        """


class StaticStrategy(RoutingStrategy):
    """Always route to a specific model."""

    name = "static"

    def __init__(self, default_model: str = "") -> None:
        self.default_model = default_model

    def select(self, candidates: list[ModelConfig], context: ScoringContext | None = None, **kwargs: Any) -> list[CandidateScore]:
        scores = []
        for model in candidates:
            selected = model.name == self.default_model
            scores.append(CandidateScore(
                model_name=model.name,
                endpoint=model.endpoint,
                total_score=1.0 if selected else 0.0,
                selected=selected,
                rejected_reason="" if selected else "Not the configured static target",
            ))
        if not any(s.selected for s in scores) and scores:
            scores[0].selected = True
            scores[0].total_score = 1.0
            scores[0].rejected_reason = ""
        return scores


class WeightedStrategy(RoutingStrategy):
    """Weighted random selection based on model weights."""

    name = "weighted"

    def select(self, candidates: list[ModelConfig], context: ScoringContext | None = None, **kwargs: Any) -> list[CandidateScore]:
        if not candidates:
            return []
        total_weight = sum(m.weight for m in candidates)
        if total_weight <= 0:
            total_weight = len(candidates)

        scores = []
        # Weighted random pick
        pick = random.uniform(0, total_weight)
        cumulative = 0.0
        picked_name = candidates[0].name

        for model in candidates:
            cumulative += model.weight
            if cumulative >= pick and picked_name == candidates[0].name:
                picked_name = model.name

        for model in candidates:
            selected = model.name == picked_name
            scores.append(CandidateScore(
                model_name=model.name,
                endpoint=model.endpoint,
                total_score=model.weight / total_weight if total_weight > 0 else 0.0,
                selected=selected,
                signals={"weight": model.weight},
            ))
        return scores


class RulesStrategy(RoutingStrategy):
    """Route based on declarative rules matching request metadata."""

    name = "rules"

    def __init__(self, rules: list[RouteRuleConfig] | None = None) -> None:
        self.rules = sorted(rules or [], key=lambda r: r.priority)

    def select(self, candidates: list[ModelConfig], context: ScoringContext | None = None, **kwargs: Any) -> list[CandidateScore]:
        ctx = context or ScoringContext()
        request_tags = ctx.tags or {}
        candidate_map = {m.name: m for m in candidates}

        # Try rules in priority order
        for rule in self.rules:
            if self._matches(rule, request_tags, ctx):
                if rule.target_model in candidate_map:
                    scores = []
                    for model in candidates:
                        selected = model.name == rule.target_model
                        scores.append(CandidateScore(
                            model_name=model.name,
                            endpoint=model.endpoint,
                            total_score=1.0 if selected else 0.0,
                            selected=selected,
                            rejected_reason="" if selected else f"Rule '{rule.name}' selected {rule.target_model}",
                            signals={"matched_rule": rule.name},
                        ))
                    return scores

        # No rule matched — score by quality
        engine = ScoringEngine()
        return engine.score_candidates(candidates, ctx)

    def _matches(self, rule: RouteRuleConfig, tags: dict[str, str], ctx: ScoringContext) -> bool:
        """Check if a rule's conditions match the request."""
        for key, value in rule.condition.items():
            if key.startswith("tags."):
                tag_key = key[5:]
                if tags.get(tag_key) != str(value):
                    return False
            elif key == "data_class":
                if ctx.data_class != str(value):
                    return False
            elif key == "min_quality":
                pass  # handled in scoring
            else:
                if tags.get(key) != str(value):
                    return False
        return True


class CostFirstStrategy(RoutingStrategy):
    """Route to the cheapest model."""

    name = "cost_first"

    def select(self, candidates: list[ModelConfig], context: ScoringContext | None = None, **kwargs: Any) -> list[CandidateScore]:
        engine = ScoringEngine()
        engine.weights.cost = 0.9
        engine.weights.latency = 0.05
        engine.weights.quality = 0.05
        engine.weights.cache_affinity = 0.0
        return engine.score_candidates(candidates, context)


class LatencyFirstStrategy(RoutingStrategy):
    """Route to the fastest model."""

    name = "latency_first"

    def select(self, candidates: list[ModelConfig], context: ScoringContext | None = None, **kwargs: Any) -> list[CandidateScore]:
        engine = ScoringEngine()
        engine.weights.cost = 0.05
        engine.weights.latency = 0.9
        engine.weights.quality = 0.05
        engine.weights.cache_affinity = 0.0
        return engine.score_candidates(candidates, context)


class QualityFirstStrategy(RoutingStrategy):
    """Route to the highest quality model."""

    name = "quality_first"

    def select(self, candidates: list[ModelConfig], context: ScoringContext | None = None, **kwargs: Any) -> list[CandidateScore]:
        engine = ScoringEngine()
        engine.weights.cost = 0.05
        engine.weights.latency = 0.05
        engine.weights.quality = 0.9
        engine.weights.cache_affinity = 0.0
        return engine.score_candidates(candidates, context)


class CheapCascadeStrategy(RoutingStrategy):
    """Try cheap models first, escalate on failure."""

    name = "cheap_cascade"

    def select(self, candidates: list[ModelConfig], context: ScoringContext | None = None, **kwargs: Any) -> list[CandidateScore]:
        """Order by cost (cheapest first), quality as tiebreaker."""
        sorted_models = sorted(candidates, key=lambda m: (m.cost_per_1k_input_tokens, -m.quality_score))
        scores = []
        for i, model in enumerate(sorted_models):
            scores.append(CandidateScore(
                model_name=model.name,
                endpoint=model.endpoint,
                total_score=1.0 - (i * 0.1),
                cost_score=1.0 - (model.cost_per_1k_input_tokens / max(m.cost_per_1k_input_tokens for m in candidates) if max(m.cost_per_1k_input_tokens for m in candidates) > 0 else 1.0),
                quality_score=model.quality_score,
                selected=(i == 0),
                signals={"cascade_order": i},
            ))
        return scores


class HybridScoreStrategy(RoutingStrategy):
    """Multi-objective weighted scoring with all dimensions."""

    name = "hybrid_score"

    def __init__(self, scoring_engine: ScoringEngine | None = None) -> None:
        self.scoring_engine = scoring_engine or ScoringEngine()

    def select(self, candidates: list[ModelConfig], context: ScoringContext | None = None, **kwargs: Any) -> list[CandidateScore]:
        return self.scoring_engine.score_candidates(candidates, context)


def get_strategy(strategy_name: str, **kwargs: Any) -> RoutingStrategy:
    """Factory function to create a routing strategy by name.

    Supports 14 strategies:
    - static, weighted, rules, cost_first, latency_first, quality_first
    - cheap_cascade, hybrid_score
    - semantic, domain (content-aware routing)
    - bandit (epsilon-greedy), learned (UCB1), thompson, exp3 (bandit strategies)

    Args:
        strategy_name: Name of the strategy (matches RouteStrategy enum values).
        **kwargs: Strategy-specific arguments.

    Returns:
        Configured RoutingStrategy instance.
    """
    from kvfleet.router.semantic import SemanticStrategy, DomainStrategy
    from kvfleet.router.learned import (
        EpsilonGreedyStrategy, UCB1Strategy,
        ThompsonSamplingStrategy, Exp3Strategy,
    )

    strategies: dict[str, type[RoutingStrategy]] = {
        "static": StaticStrategy,
        "weighted": WeightedStrategy,
        "rules": RulesStrategy,
        "cost_first": CostFirstStrategy,
        "latency_first": LatencyFirstStrategy,
        "quality_first": QualityFirstStrategy,
        "cheap_cascade": CheapCascadeStrategy,
        "hybrid_score": HybridScoreStrategy,
        "semantic": SemanticStrategy,
        "domain": DomainStrategy,
        "bandit": EpsilonGreedyStrategy,
        "learned": UCB1Strategy,
        "thompson": ThompsonSamplingStrategy,
        "exp3": Exp3Strategy,
    }

    cls = strategies.get(strategy_name)
    if cls is None:
        logger.warning("Unknown strategy '%s', falling back to hybrid_score", strategy_name)
        cls = HybridScoreStrategy

    return cls(**kwargs)
