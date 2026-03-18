"""Learned and bandit-based routing strategies.

Implements:
- Epsilon-greedy multi-armed bandit
- UCB1 (Upper Confidence Bound)
- Thompson Sampling (Beta-Bernoulli)
- Exponential weights (Exp3/Hedge)
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any

from kvfleet.config.schema import ModelConfig
from kvfleet.router.explain import CandidateScore
from kvfleet.router.scoring import ScoringContext
from kvfleet.router.strategies import RoutingStrategy

logger = logging.getLogger(__name__)


@dataclass
class ArmStats:
    """Statistics for a single bandit arm (model)."""

    name: str
    pulls: int = 0
    total_reward: float = 0.0
    successes: int = 0  # For Thompson Sampling
    failures: int = 0  # For Thompson Sampling
    sum_squared_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(self.pulls, 1)

    @property
    def reward_variance(self) -> float:
        if self.pulls < 2:
            return 1.0
        mean = self.mean_reward
        return self.sum_squared_reward / self.pulls - mean * mean


class EpsilonGreedyStrategy(RoutingStrategy):
    """Epsilon-greedy bandit for model selection.

    Exploits the best-known model (1-ε) of the time, and explores
    randomly (ε) of the time. Epsilon decays over time.
    """

    name = "bandit"

    def __init__(
        self,
        epsilon: float = 0.1,
        decay: float = 0.999,
        min_epsilon: float = 0.01,
    ) -> None:
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self._arms: dict[str, ArmStats] = {}
        self._step = 0

    def select(
        self,
        candidates: list[ModelConfig],
        context: ScoringContext | None = None,
        **kwargs: Any,
    ) -> list[CandidateScore]:
        self._step += 1
        self._ensure_arms(candidates)

        # Decay epsilon
        current_epsilon = max(self.min_epsilon, self.epsilon * (self.decay**self._step))

        if random.random() < current_epsilon:
            # Explore: random selection
            selected = random.choice(candidates)
            selected_name = selected.name
        else:
            # Exploit: best mean reward
            candidate_names = {m.name for m in candidates}
            valid_arms = {k: v for k, v in self._arms.items() if k in candidate_names}
            if valid_arms:
                selected_name = max(valid_arms, key=lambda k: valid_arms[k].mean_reward)
            else:
                selected_name = candidates[0].name

        scores = []
        for model in candidates:
            arm = self._arms[model.name]
            is_selected = model.name == selected_name
            scores.append(
                CandidateScore(
                    model_name=model.name,
                    endpoint=model.endpoint,
                    total_score=arm.mean_reward if arm.pulls > 0 else 0.5,
                    selected=is_selected,
                    signals={
                        "pulls": arm.pulls,
                        "mean_reward": arm.mean_reward,
                        "epsilon": current_epsilon,
                        "explore": random.random() < current_epsilon,
                    },
                )
            )

        return scores

    def update(self, model_name: str, reward: float) -> None:
        """Update arm statistics with observed reward.

        Args:
            model_name: The model that was used.
            reward: Observed reward (0-1 scale). Compute from latency/quality/cost.
        """
        if model_name not in self._arms:
            self._arms[model_name] = ArmStats(name=model_name)
        arm = self._arms[model_name]
        arm.pulls += 1
        arm.total_reward += reward
        arm.sum_squared_reward += reward * reward
        if reward > 0.5:
            arm.successes += 1
        else:
            arm.failures += 1

    def _ensure_arms(self, candidates: list[ModelConfig]) -> None:
        for m in candidates:
            if m.name not in self._arms:
                self._arms[m.name] = ArmStats(name=m.name)

    def stats(self) -> dict[str, Any]:
        return {
            name: {"pulls": arm.pulls, "mean_reward": arm.mean_reward}
            for name, arm in self._arms.items()
        }


class UCB1Strategy(RoutingStrategy):
    """UCB1 (Upper Confidence Bound) bandit strategy.

    Selects the arm with highest upper confidence bound:
    UCB = mean_reward + C * sqrt(ln(total_pulls) / arm_pulls)

    Balances exploration and exploitation mathematically.
    """

    name = "learned"

    def __init__(self, exploration_constant: float = 1.41) -> None:
        self.c = exploration_constant
        self._arms: dict[str, ArmStats] = {}
        self._total_pulls = 0

    def select(
        self,
        candidates: list[ModelConfig],
        context: ScoringContext | None = None,
        **kwargs: Any,
    ) -> list[CandidateScore]:
        self._ensure_arms(candidates)
        self._total_pulls += 1

        scores: list[CandidateScore] = []
        for model in candidates:
            arm = self._arms[model.name]
            if arm.pulls == 0:
                ucb = float("inf")  # Pull every arm at least once
            else:
                exploitation = arm.mean_reward
                exploration = self.c * math.sqrt(math.log(self._total_pulls) / arm.pulls)
                ucb = exploitation + exploration

            scores.append(
                CandidateScore(
                    model_name=model.name,
                    endpoint=model.endpoint,
                    total_score=ucb,
                    signals={
                        "pulls": arm.pulls,
                        "mean_reward": arm.mean_reward,
                        "ucb_value": ucb,
                        "total_pulls": self._total_pulls,
                    },
                )
            )

        scores.sort(key=lambda s: s.total_score, reverse=True)
        if scores:
            scores[0].selected = True
        return scores

    def update(self, model_name: str, reward: float) -> None:
        """Update arm statistics."""
        if model_name not in self._arms:
            self._arms[model_name] = ArmStats(name=model_name)
        arm = self._arms[model_name]
        arm.pulls += 1
        arm.total_reward += reward
        arm.sum_squared_reward += reward * reward
        if reward > 0.5:
            arm.successes += 1
        else:
            arm.failures += 1

    def _ensure_arms(self, candidates: list[ModelConfig]) -> None:
        for m in candidates:
            if m.name not in self._arms:
                self._arms[m.name] = ArmStats(name=m.name)


class ThompsonSamplingStrategy(RoutingStrategy):
    """Thompson Sampling bandit strategy (Beta-Bernoulli).

    Maintains a Beta distribution for each arm and samples from it.
    Natural exploration-exploitation trade-off through Bayesian updating.
    """

    name = "thompson"

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self._arms: dict[str, ArmStats] = {}

    def select(
        self,
        candidates: list[ModelConfig],
        context: ScoringContext | None = None,
        **kwargs: Any,
    ) -> list[CandidateScore]:
        self._ensure_arms(candidates)

        scores: list[CandidateScore] = []
        for model in candidates:
            arm = self._arms[model.name]
            alpha = self.prior_alpha + arm.successes
            beta = self.prior_beta + arm.failures
            sample = random.betavariate(alpha, beta)

            scores.append(
                CandidateScore(
                    model_name=model.name,
                    endpoint=model.endpoint,
                    total_score=sample,
                    signals={
                        "pulls": arm.pulls,
                        "successes": arm.successes,
                        "failures": arm.failures,
                        "alpha": alpha,
                        "beta": beta,
                        "sampled_value": sample,
                    },
                )
            )

        scores.sort(key=lambda s: s.total_score, reverse=True)
        if scores:
            scores[0].selected = True
        return scores

    def update(self, model_name: str, reward: float) -> None:
        """Update arm with binary reward (threshold at 0.5)."""
        if model_name not in self._arms:
            self._arms[model_name] = ArmStats(name=model_name)
        arm = self._arms[model_name]
        arm.pulls += 1
        arm.total_reward += reward
        if reward > 0.5:
            arm.successes += 1
        else:
            arm.failures += 1

    def _ensure_arms(self, candidates: list[ModelConfig]) -> None:
        for m in candidates:
            if m.name not in self._arms:
                self._arms[m.name] = ArmStats(name=m.name)


class Exp3Strategy(RoutingStrategy):
    """Exp3 (Exponential-weight algorithm for Exploration and Exploitation).

    Works in adversarial settings — doesn't assume stationary rewards.
    Maintains probability weights and updates multiplicatively.
    """

    name = "exp3"

    def __init__(self, gamma: float = 0.1) -> None:
        self.gamma = gamma
        self._weights: dict[str, float] = {}
        self._arms: dict[str, ArmStats] = {}

    def select(
        self,
        candidates: list[ModelConfig],
        context: ScoringContext | None = None,
        **kwargs: Any,
    ) -> list[CandidateScore]:
        n = len(candidates)
        if n == 0:
            return []

        for m in candidates:
            if m.name not in self._weights:
                self._weights[m.name] = 1.0
            if m.name not in self._arms:
                self._arms[m.name] = ArmStats(name=m.name)

        # Compute probabilities
        total_weight = sum(self._weights.get(m.name, 1.0) for m in candidates)
        probs: dict[str, float] = {}
        for m in candidates:
            w = self._weights.get(m.name, 1.0)
            probs[m.name] = (1 - self.gamma) * (w / total_weight) + self.gamma / n

        # Sample from distribution
        r = random.random()
        cumulative = 0.0
        selected_name = candidates[0].name
        for name, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                selected_name = name
                break

        scores: list[CandidateScore] = []
        for model in candidates:
            arm = self._arms.get(model.name, ArmStats(name=model.name))
            scores.append(
                CandidateScore(
                    model_name=model.name,
                    endpoint=model.endpoint,
                    total_score=probs.get(model.name, 0.0),
                    selected=model.name == selected_name,
                    signals={
                        "probability": probs.get(model.name, 0.0),
                        "weight": self._weights.get(model.name, 1.0),
                        "pulls": arm.pulls,
                    },
                )
            )
        return scores

    def update(self, model_name: str, reward: float) -> None:
        """Update weights with observed reward."""
        if model_name not in self._arms:
            self._arms[model_name] = ArmStats(name=model_name)
        arm = self._arms[model_name]
        arm.pulls += 1
        arm.total_reward += reward

        n = max(len(self._weights), 1)
        total_weight = sum(self._weights.values()) or 1.0
        w = self._weights.get(model_name, 1.0)
        prob = (1 - self.gamma) * (w / total_weight) + self.gamma / n

        # Estimated reward (importance-weighted)
        estimated = reward / max(prob, 1e-8)
        # Multiplicative weight update
        self._weights[model_name] = w * math.exp(self.gamma * estimated / n)


def compute_reward(
    latency_ms: float,
    quality_score: float,
    cost: float,
    success: bool = True,
    target_latency_ms: float = 2000.0,
) -> float:
    """Compute a 0-1 reward signal from observed outcomes.

    Used to update bandit strategies after each request.

    Args:
        latency_ms: Observed latency.
        quality_score: Output quality estimate (0-1).
        cost: Cost of the request.
        success: Whether the request succeeded.
        target_latency_ms: Target latency for normalization.

    Returns:
        Reward value 0-1.
    """
    if not success:
        return 0.0

    latency_reward = max(0.0, 1.0 - (latency_ms / target_latency_ms))
    cost_reward = max(0.0, 1.0 - min(cost * 100, 1.0))  # $0.01 → 0.0 penalty
    quality_reward = quality_score

    return 0.4 * latency_reward + 0.3 * quality_reward + 0.3 * cost_reward
