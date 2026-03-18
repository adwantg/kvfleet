"""Tests for semantic routing and bandit strategies."""

import pytest

from kvfleet.config.schema import ModelConfig
from kvfleet.router.learned import (
    EpsilonGreedyStrategy,
    Exp3Strategy,
    ThompsonSamplingStrategy,
    UCB1Strategy,
    compute_reward,
)
from kvfleet.router.scoring import ScoringContext
from kvfleet.router.semantic import (
    DomainStrategy,
    SemanticStrategy,
    classify_domain,
    estimate_complexity,
)
from kvfleet.router.strategies import get_strategy


@pytest.fixture
def candidates():
    return [
        ModelConfig(
            name="code-model",
            endpoint="http://a:8000",
            quality_score=0.9,
            tags={"domain": "coding"},
        ),
        ModelConfig(
            name="general",
            endpoint="http://b:8000",
            quality_score=0.7,
            tags={"domain": "general"},
        ),
        ModelConfig(
            name="creative-model",
            endpoint="http://c:8000",
            quality_score=0.8,
            tags={"domain": "creative"},
        ),
    ]


# ───────────────────── Domain Classifier ─────────────────────


class TestClassifyDomain:
    def test_coding(self):
        domain, conf = classify_domain("Write a Python function to sort a list")
        assert domain == "coding"
        assert conf > 0.5

    def test_math(self):
        domain, _conf = classify_domain("Solve this integral of x^2 from 0 to 1")
        assert domain == "math"

    def test_creative(self):
        domain, _conf = classify_domain("Write a poem about the ocean")
        assert domain == "creative"

    def test_medical(self):
        domain, _conf = classify_domain("What are the symptoms of diabetes treatment?")
        assert domain == "medical"

    def test_general_fallback(self):
        domain, _conf = classify_domain("What time is it?")
        assert domain == "general"

    def test_empty_text(self):
        domain, _conf = classify_domain("")
        assert domain == "general"


class TestEstimateComplexity:
    def test_simple(self):
        c = estimate_complexity("What is Python?")
        assert c < 0.5

    def test_complex(self):
        c = estimate_complexity(
            "Implement a distributed consensus algorithm step by step, "
            "comparing trade-offs between Raft and Paxos. "
            "How do they handle network partitions? "
            "What are the performance implications?"
        )
        assert c > 0.3


# ───────────────────── Semantic Strategy ─────────────────────


class TestSemanticStrategy:
    def test_domain_match(self, candidates):
        strategy = SemanticStrategy()
        ctx = ScoringContext(metadata={"prompt_text": "Write a Python class for API client"})
        scores = strategy.select(candidates, ctx)
        selected = next(s for s in scores if s.selected)
        assert selected.model_name == "code-model"

    def test_creative_match(self, candidates):
        strategy = SemanticStrategy()
        ctx = ScoringContext(metadata={"prompt_text": "Write a poem about stars"})
        scores = strategy.select(candidates, ctx)
        selected = next(s for s in scores if s.selected)
        assert selected.model_name == "creative-model"

    def test_no_context(self, candidates):
        strategy = SemanticStrategy()
        scores = strategy.select(candidates)
        assert any(s.selected for s in scores)


class TestDomainStrategy:
    def test_mapped_domain(self, candidates):
        strategy = DomainStrategy(domain_model_map={"coding": "code-model"})
        ctx = ScoringContext(metadata={"prompt_text": "Debug this Python error"})
        scores = strategy.select(candidates, ctx)
        selected = next(s for s in scores if s.selected)
        assert selected.model_name == "code-model"

    def test_unmapped_domain(self, candidates):
        strategy = DomainStrategy(domain_model_map={})
        ctx = ScoringContext(metadata={"prompt_text": "Write code in Python"})
        scores = strategy.select(candidates, ctx)
        assert any(s.selected for s in scores)


# ───────────────────── Bandit Strategies ─────────────────────


class TestEpsilonGreedy:
    def test_select(self, candidates):
        strategy = EpsilonGreedyStrategy(epsilon=0.1)
        scores = strategy.select(candidates)
        assert any(s.selected for s in scores)

    def test_update_and_exploit(self, candidates):
        strategy = EpsilonGreedyStrategy(epsilon=0.0, min_epsilon=0.0)
        # Seed with rewards
        for _ in range(10):
            strategy.update("code-model", 0.9)
            strategy.update("general", 0.3)
            strategy.update("creative-model", 0.5)
        scores = strategy.select(candidates)
        selected = next(s for s in scores if s.selected)
        assert selected.model_name == "code-model"

    def test_stats(self, candidates):
        strategy = EpsilonGreedyStrategy()
        strategy.update("test", 1.0)
        stats = strategy.stats()
        assert "test" in stats


class TestUCB1:
    def test_explores_all_first(self, candidates):
        strategy = UCB1Strategy()
        # First 3 calls should explore each arm
        names = set()
        for _ in range(3):
            scores = strategy.select(candidates)
            selected = next(s for s in scores if s.selected)
            names.add(selected.model_name)
            strategy.update(selected.model_name, 0.5)
        assert len(names) == 3  # All arms pulled at least once

    def test_converges(self, candidates):
        strategy = UCB1Strategy()
        for m in candidates:
            strategy.update(m.name, 0.5)  # Seed
        for _ in range(100):
            strategy.update("code-model", 0.95)
            strategy.update("general", 0.3)
            strategy.update("creative-model", 0.5)
        scores = strategy.select(candidates)
        selected = next(s for s in scores if s.selected)
        assert selected.model_name == "code-model"


class TestThompsonSampling:
    def test_select(self, candidates):
        strategy = ThompsonSamplingStrategy()
        scores = strategy.select(candidates)
        assert any(s.selected for s in scores)

    def test_update(self, candidates):
        strategy = ThompsonSamplingStrategy()
        strategy.update("code-model", 0.9)
        strategy.update("code-model", 0.8)
        assert strategy._arms["code-model"].successes == 2


class TestExp3:
    def test_select(self, candidates):
        strategy = Exp3Strategy(gamma=0.1)
        scores = strategy.select(candidates)
        assert any(s.selected for s in scores)
        assert all(s.signals.get("probability", 0) > 0 for s in scores)

    def test_update(self, candidates):
        strategy = Exp3Strategy()
        strategy.select(candidates)
        strategy.update("code-model", 0.9)
        assert strategy._arms["code-model"].pulls == 1


class TestComputeReward:
    def test_success(self):
        r = compute_reward(latency_ms=500, quality_score=0.8, cost=0.001)
        assert 0 < r <= 1.0

    def test_failure(self):
        assert compute_reward(latency_ms=100, quality_score=0.9, cost=0, success=False) == 0.0

    def test_fast_cheap_high_quality(self):
        r = compute_reward(latency_ms=100, quality_score=0.95, cost=0.0)
        assert r > 0.8

    def test_slow_expensive(self):
        r = compute_reward(latency_ms=5000, quality_score=0.5, cost=0.1)
        assert r < 0.5


# ───────────────────── Factory Integration ─────────────────────


class TestStrategyFactory:
    def test_semantic_factory(self):
        strategy = get_strategy("semantic")
        assert isinstance(strategy, SemanticStrategy)

    def test_domain_factory(self):
        strategy = get_strategy("domain")
        assert isinstance(strategy, DomainStrategy)

    def test_bandit_factory(self):
        strategy = get_strategy("bandit")
        assert isinstance(strategy, EpsilonGreedyStrategy)

    def test_learned_factory(self):
        strategy = get_strategy("learned")
        assert isinstance(strategy, UCB1Strategy)

    def test_thompson_factory(self):
        strategy = get_strategy("thompson")
        assert isinstance(strategy, ThompsonSamplingStrategy)

    def test_exp3_factory(self):
        strategy = get_strategy("exp3")
        assert isinstance(strategy, Exp3Strategy)

    def test_all_14_strategies(self):
        names = [
            "static",
            "weighted",
            "rules",
            "cost_first",
            "latency_first",
            "quality_first",
            "cheap_cascade",
            "hybrid_score",
            "semantic",
            "domain",
            "bandit",
            "learned",
            "thompson",
            "exp3",
        ]
        for name in names:
            strategy = get_strategy(name)
            assert strategy is not None
