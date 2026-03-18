"""Policy engine — enforce routing constraints before model selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from kvfleet.config.schema import ModelConfig, PolicyConfig, PolicyRule
from kvfleet.router.explain import PolicyDecision

logger = logging.getLogger(__name__)


@dataclass
class PolicyContext:
    """Context for policy evaluation."""

    data_class: str = "internal"
    tenant_id: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    has_pii: bool = False
    source_region: str = ""
    prompt_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class PolicyEngine:
    """Evaluates policy rules to filter and constrain model candidates.

    Policies enforce hard enterprise guardrails:
    - PII-bearing traffic must go to private models
    - Certain teams can only use approved models
    - Regulated prompts require specific route classes
    - Confidential data must not leave VPC
    - Data residency rules by geography/business unit
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()
        self._rules = sorted(self.config.rules, key=lambda r: r.priority)

    def evaluate(
        self,
        candidates: list[ModelConfig],
        context: PolicyContext,
    ) -> tuple[list[ModelConfig], list[PolicyDecision]]:
        """Evaluate policies and filter candidates.

        Args:
            candidates: Available model candidates.
            context: Policy evaluation context.

        Returns:
            Tuple of (filtered_candidates, policy_decisions).
        """
        if not self.config.enabled:
            return candidates, []

        decisions: list[PolicyDecision] = []
        filtered = list(candidates)

        # PII detection
        if self.config.pii_detection and context.has_pii:
            before = len(filtered)
            filtered = [m for m in filtered if "confidential" in m.allowed_data_classes or self._is_private_model(m)]
            if len(filtered) < before:
                decisions.append(PolicyDecision(
                    rule_name="pii_detection",
                    passed=len(filtered) > 0,
                    reason=f"PII detected — restricted to private models ({before - len(filtered)} models removed)",
                    action="require_private",
                ))

        # Data class enforcement
        if context.data_class:
            before = len(filtered)
            filtered = [m for m in filtered if context.data_class in m.allowed_data_classes]
            if len(filtered) < before:
                decisions.append(PolicyDecision(
                    rule_name="data_class_enforcement",
                    passed=len(filtered) > 0,
                    reason=f"Data class '{context.data_class}' filter removed {before - len(filtered)} models",
                    action="filter",
                ))

        # Custom rules
        for rule in self._rules:
            result = self._evaluate_rule(rule, filtered, context)
            if result is not None:
                filtered, decision = result
                decisions.append(decision)

        return filtered, decisions

    def _evaluate_rule(
        self,
        rule: PolicyRule,
        candidates: list[ModelConfig],
        context: PolicyContext,
    ) -> tuple[list[ModelConfig], PolicyDecision] | None:
        """Evaluate a single policy rule."""
        if not self._condition_matches(rule.condition, context):
            return None

        if rule.action == "require_private":
            filtered = [m for m in candidates if self._is_private_model(m)]
            return filtered, PolicyDecision(
                rule_name=rule.name, passed=len(filtered) > 0,
                reason=rule.description or "Require private model", action=rule.action,
            )
        elif rule.action == "block":
            filtered = [m for m in candidates if m.name not in rule.target_models]
            return filtered, PolicyDecision(
                rule_name=rule.name, passed=len(filtered) > 0,
                reason=f"Blocked models: {rule.target_models}", action=rule.action,
            )
        elif rule.action == "require_model":
            filtered = [m for m in candidates if m.name in rule.target_models]
            return filtered, PolicyDecision(
                rule_name=rule.name, passed=len(filtered) > 0,
                reason=f"Required specific models: {rule.target_models}", action=rule.action,
            )
        elif rule.action == "allow":
            return None  # No filtering

        return None

    @staticmethod
    def _condition_matches(condition: str, context: PolicyContext) -> bool:
        """Simple condition matching."""
        condition = condition.strip().lower()
        if "==" in condition:
            key, value = condition.split("==", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key == "data_class":
                return context.data_class == value
            elif key == "has_pii":
                return str(context.has_pii).lower() == value
            elif key == "source_region":
                return context.source_region == value
            elif key.startswith("tags."):
                tag_key = key[5:]
                return context.tags.get(tag_key, "") == value
            elif key == "tenant_id":
                return context.tenant_id == value
        elif condition == "always":
            return True
        return False

    @staticmethod
    def _is_private_model(model: ModelConfig) -> bool:
        """Check if a model is considered private/on-premises."""
        private_indicators = {"vllm", "triton", "tgi", "ollama", "custom_http"}
        return model.provider.value in private_indicators
