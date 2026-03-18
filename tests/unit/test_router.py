"""Tests for routing strategies and scoring engine."""

import pytest

from kvfleet.config.schema import ModelConfig, RouteRuleConfig
from kvfleet.router.scoring import ScoringContext, ScoringEngine
from kvfleet.router.strategies import (
    CheapCascadeStrategy,
    CostFirstStrategy,
    HybridScoreStrategy,
    LatencyFirstStrategy,
    QualityFirstStrategy,
    RulesStrategy,
    StaticStrategy,
    WeightedStrategy,
    get_strategy,
)


@pytest.fixture
def candidates():
    return [
        ModelConfig(
            name="cheap",
            endpoint="http://a:8000",
            quality_score=0.5,
            cost_per_1k_input_tokens=0.001,
            latency_p50_ms=200,
        ),
        ModelConfig(
            name="balanced",
            endpoint="http://b:8000",
            quality_score=0.7,
            cost_per_1k_input_tokens=0.01,
            latency_p50_ms=500,
        ),
        ModelConfig(
            name="premium",
            endpoint="http://c:8000",
            quality_score=0.95,
            cost_per_1k_input_tokens=0.05,
            latency_p50_ms=400,
        ),
    ]


class TestScoringEngine:
    def test_score_candidates(self, candidates):
        engine = ScoringEngine()
        scores = engine.score_candidates(candidates)
        assert len(scores) == 3
        assert scores[0].selected is True
        assert all(s.total_score >= 0 for s in scores)

    def test_cost_score_free(self):
        engine = ScoringEngine()
        model = ModelConfig(name="x", endpoint="http://x", cost_per_1k_input_tokens=0.0)
        score = engine._score_cost(model)
        assert score == 1.0

    def test_cost_score_expensive(self):
        engine = ScoringEngine()
        model = ModelConfig(name="x", endpoint="http://x", cost_per_1k_input_tokens=0.9)
        score = engine._score_cost(model)
        assert 0.0 <= score <= 0.2

    def test_latency_score_fast(self):
        engine = ScoringEngine()
        model = ModelConfig(name="x", endpoint="http://x", latency_p50_ms=100)
        score = engine._score_latency(model)
        assert score > 0.9

    def test_compliance_filter(self, candidates):
        engine = ScoringEngine()
        ctx = ScoringContext(data_class="restricted")
        scores = engine.score_candidates(candidates, ctx)
        # All candidates should have compliance_score=0 since "restricted" not in allowed
        for s in scores:
            assert s.compliance_score == 0.0

    def test_update_weights(self):
        engine = ScoringEngine()
        engine.update_weights(cost=0.9, quality=0.1)
        assert engine.weights.cost == 0.9
        assert engine.weights.quality == 0.1


class TestStaticStrategy:
    def test_select_default(self, candidates):
        strategy = StaticStrategy(default_model="balanced")
        scores = strategy.select(candidates)
        selected = next(s for s in scores if s.selected)
        assert selected.model_name == "balanced"

    def test_select_fallback(self, candidates):
        strategy = StaticStrategy(default_model="nonexistent")
        scores = strategy.select(candidates)
        assert any(s.selected for s in scores)


class TestWeightedStrategy:
    def test_weighted_selection(self, candidates):
        strategy = WeightedStrategy()
        scores = strategy.select(candidates)
        assert len(scores) == 3
        assert any(s.selected for s in scores)

    def test_empty_candidates(self):
        strategy = WeightedStrategy()
        assert strategy.select([]) == []


class TestRulesStrategy:
    def test_rule_match(self, candidates):
        rules = [
            RouteRuleConfig(
                name="code-rule",
                condition={"tags.domain": "coding"},
                target_model="premium",
                priority=1,
            )
        ]
        strategy = RulesStrategy(rules=rules)
        ctx = ScoringContext(tags={"domain": "coding"})
        scores = strategy.select(candidates, ctx)
        selected = next(s for s in scores if s.selected)
        assert selected.model_name == "premium"

    def test_no_rule_match(self, candidates):
        rules = [
            RouteRuleConfig(
                name="code-rule",
                condition={"tags.domain": "coding"},
                target_model="premium",
            )
        ]
        strategy = RulesStrategy(rules=rules)
        ctx = ScoringContext(tags={"domain": "general"})
        scores = strategy.select(candidates, ctx)
        assert any(s.selected for s in scores)


class TestCostFirstStrategy:
    def test_cheapest_wins(self, candidates):
        strategy = CostFirstStrategy()
        scores = strategy.select(candidates)
        selected = next(s for s in scores if s.selected)
        assert selected.model_name == "cheap"


class TestLatencyFirstStrategy:
    def test_fastest_wins(self, candidates):
        strategy = LatencyFirstStrategy()
        scores = strategy.select(candidates)
        selected = next(s for s in scores if s.selected)
        assert selected.model_name == "cheap"  # 200ms is lowest


class TestQualityFirstStrategy:
    def test_best_quality_wins(self, candidates):
        strategy = QualityFirstStrategy()
        scores = strategy.select(candidates)
        selected = next(s for s in scores if s.selected)
        assert selected.model_name == "premium"


class TestCheapCascadeStrategy:
    def test_cheapest_first(self, candidates):
        strategy = CheapCascadeStrategy()
        scores = strategy.select(candidates)
        assert scores[0].model_name == "cheap"
        assert scores[0].selected is True


class TestGetStrategy:
    def test_valid_strategies(self):
        for name in [
            "static",
            "weighted",
            "rules",
            "cost_first",
            "latency_first",
            "quality_first",
            "cheap_cascade",
            "hybrid_score",
        ]:
            strategy = get_strategy(name)
            assert strategy is not None

    def test_unknown_fallback(self):
        strategy = get_strategy("unknown")
        assert isinstance(strategy, HybridScoreStrategy)
