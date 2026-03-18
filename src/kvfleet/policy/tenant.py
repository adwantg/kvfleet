"""Tenant-aware routing and budget management."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from kvfleet.config.schema import BudgetConfig, TenantConfig

logger = logging.getLogger(__name__)


@dataclass
class SpendRecord:
    """Tracks spending for a tenant."""

    total_usd: float = 0.0
    requests: int = 0
    period_start: float = field(default_factory=time.time)
    last_request: float = field(default_factory=time.time)


class BudgetTracker:
    """Tracks spending against budgets for rate limiting.

    Thread-safe via threading.Lock for concurrent access.
    """

    def __init__(self) -> None:
        self._spend: dict[str, SpendRecord] = {}  # tenant_id → spend
        self._lock = threading.Lock()

    def record_spend(self, tenant_id: str, cost_usd: float) -> None:
        """Record a spending event."""
        with self._lock:
            if tenant_id not in self._spend:
                self._spend[tenant_id] = SpendRecord()
            record = self._spend[tenant_id]
            record.total_usd += cost_usd
            record.requests += 1
            record.last_request = time.time()

    def get_remaining_budget(self, tenant_id: str, budget: BudgetConfig) -> float:
        """Get remaining budget for a tenant."""
        record = self._spend.get(tenant_id)
        if not record:
            return budget.monthly_budget_usd
        return max(0.0, budget.monthly_budget_usd - record.total_usd)

    def is_over_budget(self, tenant_id: str, budget: BudgetConfig) -> bool:
        """Check if tenant has exceeded budget."""
        return self.get_remaining_budget(tenant_id, budget) <= 0

    def should_throttle(self, tenant_id: str, budget: BudgetConfig) -> bool:
        """Check if spending should be throttled (approaching limit)."""
        remaining = self.get_remaining_budget(tenant_id, budget)
        threshold = budget.monthly_budget_usd * (budget.alert_threshold_pct / 100.0)
        return remaining < (budget.monthly_budget_usd - threshold)

    def get_spend(self, tenant_id: str) -> SpendRecord:
        """Get spending record for a tenant."""
        return self._spend.get(tenant_id, SpendRecord())

    def reset(self, tenant_id: str) -> None:
        """Reset spending for a tenant (new billing period)."""
        self._spend[tenant_id] = SpendRecord()

    def summary(self) -> dict[str, Any]:
        """Return spending summary."""
        return {
            tenant_id: {
                "total_usd": record.total_usd,
                "requests": record.requests,
            }
            for tenant_id, record in self._spend.items()
        }


class TenantManager:
    """Manages tenant-specific routing preferences and constraints.

    Each tenant (business unit) can define:
    - Preferred models
    - Blocked models
    - Latency priorities
    - Compliance constraints
    - Spending limits
    """

    def __init__(self, tenants: dict[str, TenantConfig] | None = None) -> None:
        self.tenants = tenants or {}
        self.budget_tracker = BudgetTracker()

    def get_tenant(self, tenant_id: str) -> TenantConfig | None:
        """Get tenant config by ID."""
        return self.tenants.get(tenant_id)

    def filter_models_for_tenant(
        self,
        tenant_id: str,
        model_names: list[str],
    ) -> list[str]:
        """Filter model names based on tenant preferences.

        Returns only models that are preferred (if set) and not blocked.
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return model_names

        filtered = model_names

        # Apply blocked list
        if tenant.blocked_models:
            filtered = [m for m in filtered if m not in tenant.blocked_models]

        # Apply preferred list (if set, restrict to preferred)
        if tenant.preferred_models:
            preferred = [m for m in filtered if m in tenant.preferred_models]
            if preferred:
                filtered = preferred

        return filtered

    def check_budget(self, tenant_id: str, estimated_cost: float) -> bool:
        """Check if a request fits within tenant budget."""
        tenant = self.tenants.get(tenant_id)
        if not tenant or not tenant.budget.enabled:
            return True

        # Check per-request limit
        if tenant.max_cost_per_request and estimated_cost > tenant.max_cost_per_request:
            return False

        # Check monthly budget
        if self.budget_tracker.is_over_budget(tenant_id, tenant.budget):
            return False

        return True

    def record_request(self, tenant_id: str, cost_usd: float) -> None:
        """Record a request cost for budget tracking."""
        self.budget_tracker.record_spend(tenant_id, cost_usd)
