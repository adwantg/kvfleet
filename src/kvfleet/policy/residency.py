"""Data residency rules for geographic/compliance-aware routing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResidencyRule:
    """A data residency rule."""

    name: str
    source_regions: list[str] = field(default_factory=list)
    allowed_model_regions: list[str] = field(default_factory=list)
    blocked_providers: list[str] = field(default_factory=list)
    description: str = ""


class ResidencyEngine:
    """Enforces data residency constraints on routing.

    Ensures that data from specific regions is only processed
    by models deployed in approved regions/providers.
    """

    def __init__(self, rules: list[ResidencyRule] | None = None) -> None:
        self.rules = rules or []

    def get_allowed_regions(self, source_region: str) -> list[str] | None:
        """Get allowed model regions for a source region."""
        for rule in self.rules:
            if source_region in rule.source_regions:
                return rule.allowed_model_regions
        return None  # No restriction

    def get_blocked_providers(self, source_region: str) -> list[str]:
        """Get blocked providers for a source region."""
        blocked = []
        for rule in self.rules:
            if source_region in rule.source_regions:
                blocked.extend(rule.blocked_providers)
        return list(set(blocked))

    def is_compliant(self, source_region: str, model_region: str, provider: str) -> bool:
        """Check if routing is compliant with residency rules."""
        for rule in self.rules:
            if source_region in rule.source_regions:
                if rule.allowed_model_regions and model_region not in rule.allowed_model_regions:
                    return False
                if provider in rule.blocked_providers:
                    return False
        return True
