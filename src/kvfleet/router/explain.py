"""Routing decision explainer — structured traces for every decision."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateScore:
    """Score breakdown for a single candidate model."""

    model_name: str
    endpoint: str = ""
    total_score: float = 0.0
    cost_score: float = 0.0
    latency_score: float = 0.0
    quality_score: float = 0.0
    cache_affinity_score: float = 0.0
    hardware_load_score: float = 0.0
    compliance_score: float = 0.0
    selected: bool = False
    rejected_reason: str = ""
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDecision:
    """Record of a policy gate evaluation."""

    rule_name: str
    passed: bool
    reason: str = ""
    action: str = ""


@dataclass
class RouteExplanation:
    """Full explanation of a routing decision.

    Every routing decision produces an explanation trace containing:
    - Why the selected model was chosen
    - Why other models were rejected
    - What signals were considered
    - What policy gates were applied
    - Whether cache affinity influenced the decision
    """

    request_id: str = ""
    timestamp: float = field(default_factory=time.time)
    strategy: str = ""
    selected_model: str = ""
    selected_endpoint: str = ""
    candidates: list[CandidateScore] = field(default_factory=list)
    policy_decisions: list[PolicyDecision] = field(default_factory=list)
    fallback_triggered: bool = False
    fallback_chain: list[str] = field(default_factory=list)
    cache_affinity_used: bool = False
    cache_hit: bool = False
    strategy_overridden: bool = False
    shadow_models: list[str] = field(default_factory=list)
    total_latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of the routing decision."""
        parts = [f"Strategy: {self.strategy}"]
        parts.append(f"Selected: {self.selected_model}")
        if self.selected_endpoint:
            parts.append(f"Endpoint: {self.selected_endpoint}")
        if self.cache_affinity_used:
            parts.append(f"Cache affinity: {'HIT' if self.cache_hit else 'MISS'}")
        if self.fallback_triggered:
            parts.append(f"Fallback chain: {' → '.join(self.fallback_chain)}")

        # Top candidates
        sorted_candidates = sorted(self.candidates, key=lambda c: c.total_score, reverse=True)
        if sorted_candidates:
            parts.append("Candidate scores:")
            for c in sorted_candidates[:5]:
                status = "✓" if c.selected else "✗"
                line = f"  {status} {c.model_name}: {c.total_score:.3f}"
                if c.rejected_reason:
                    line += f" (rejected: {c.rejected_reason})"
                parts.append(line)

        # Policy decisions
        if self.policy_decisions:
            parts.append("Policy gates:")
            for p in self.policy_decisions:
                status = "PASS" if p.passed else "BLOCK"
                parts.append(f"  [{status}] {p.rule_name}: {p.reason}")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for logging/API responses."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "strategy": self.strategy,
            "selected_model": self.selected_model,
            "selected_endpoint": self.selected_endpoint,
            "candidates": [
                {
                    "model": c.model_name,
                    "endpoint": c.endpoint,
                    "total_score": c.total_score,
                    "cost_score": c.cost_score,
                    "latency_score": c.latency_score,
                    "quality_score": c.quality_score,
                    "cache_affinity_score": c.cache_affinity_score,
                    "hardware_load_score": c.hardware_load_score,
                    "compliance_score": c.compliance_score,
                    "selected": c.selected,
                    "rejected_reason": c.rejected_reason,
                }
                for c in self.candidates
            ],
            "policy_decisions": [
                {"rule": p.rule_name, "passed": p.passed, "reason": p.reason, "action": p.action}
                for p in self.policy_decisions
            ],
            "fallback_triggered": self.fallback_triggered,
            "cache_affinity_used": self.cache_affinity_used,
            "cache_hit": self.cache_hit,
            "shadow_models": self.shadow_models,
            "total_latency_ms": self.total_latency_ms,
        }
