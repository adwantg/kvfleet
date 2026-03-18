"""Model cost synchronization — fetch and track pricing from providers."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ───────────────────── Built-in Cost Database ─────────────────────

# Default pricing per 1K tokens (input/output) in USD
# Updated: March 2025
KNOWN_MODEL_COSTS: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
    # Anthropic
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3.5-haiku": {"input": 0.0008, "output": 0.004},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    # Google
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    # Meta (self-hosted — free, but track GPU cost)
    "llama-3-8b": {"input": 0.0, "output": 0.0},
    "llama-3-70b": {"input": 0.0, "output": 0.0},
    "llama-3.1-8b": {"input": 0.0, "output": 0.0},
    "llama-3.1-70b": {"input": 0.0, "output": 0.0},
    "llama-3.1-405b": {"input": 0.0, "output": 0.0},
    # Mistral
    "mistral-7b": {"input": 0.0, "output": 0.0},
    "mixtral-8x7b": {"input": 0.0, "output": 0.0},
    "mistral-large": {"input": 0.002, "output": 0.006},
    "mistral-small": {"input": 0.0002, "output": 0.0006},
    # DeepSeek
    "deepseek-v3": {"input": 0.00027, "output": 0.0011},
    "deepseek-r1": {"input": 0.00055, "output": 0.00219},
    "deepseek-coder": {"input": 0.0, "output": 0.0},
    # Groq-hosted (pricing varies)
    "groq/llama-3-8b": {"input": 0.00005, "output": 0.00008},
    "groq/llama-3-70b": {"input": 0.00059, "output": 0.00079},
    "groq/mixtral-8x7b": {"input": 0.00024, "output": 0.00024},
    # Together AI
    "together/llama-3-8b": {"input": 0.0002, "output": 0.0002},
    "together/llama-3-70b": {"input": 0.0009, "output": 0.0009},
}


@dataclass
class ModelCostInfo:
    """Pricing information for a model."""

    model_id: str
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    image_cost_per_image: float = 0.0
    cached_input_cost_per_1k: float = 0.0  # Prompt caching discount
    source: str = "unknown"
    last_updated: float = field(default_factory=time.time)

    @property
    def avg_cost_per_1k(self) -> float:
        """Average of input and output cost."""
        return (self.input_cost_per_1k + self.output_cost_per_1k) / 2

    def estimate_request_cost(
        self,
        input_tokens: int = 500,
        output_tokens: int = 500,
        images: int = 0,
    ) -> float:
        """Estimate cost for a single request."""
        cost = (input_tokens / 1000) * self.input_cost_per_1k
        cost += (output_tokens / 1000) * self.output_cost_per_1k
        cost += images * self.image_cost_per_image
        return cost


class CostSyncManager:
    """Manages model cost data with sync from built-in database and external sources.

    Thread-safe. Provides cost lookup for scoring and budget tracking.
    """

    def __init__(self) -> None:
        self._costs: dict[str, ModelCostInfo] = {}
        self._lock = threading.Lock()
        self._load_builtin_costs()

    def _load_builtin_costs(self) -> None:
        """Load built-in cost database."""
        for model_id, costs in KNOWN_MODEL_COSTS.items():
            self._costs[model_id] = ModelCostInfo(
                model_id=model_id,
                input_cost_per_1k=costs.get("input", 0.0),
                output_cost_per_1k=costs.get("output", 0.0),
                source="builtin",
            )

    def get_cost(self, model_id: str) -> ModelCostInfo | None:
        """Look up cost info for a model.

        Tries exact match first, then partial match.
        """
        with self._lock:
            # Exact match
            if model_id in self._costs:
                return self._costs[model_id]

            # Partial match (e.g., "meta-llama/Llama-3-8B" → "llama-3-8b")
            model_lower = model_id.lower()
            for key, info in self._costs.items():
                if key in model_lower or model_lower in key:
                    return info

            return None

    def set_cost(
        self,
        model_id: str,
        input_cost: float,
        output_cost: float,
        source: str = "manual",
    ) -> None:
        """Manually set cost for a model."""
        with self._lock:
            self._costs[model_id] = ModelCostInfo(
                model_id=model_id,
                input_cost_per_1k=input_cost,
                output_cost_per_1k=output_cost,
                source=source,
            )

    def sync_from_config(self, models: list[Any]) -> int:
        """Sync costs from fleet config model definitions.

        Args:
            models: List of ModelConfig objects.

        Returns:
            Number of models synced.
        """
        count = 0
        with self._lock:
            for model in models:
                model_id = getattr(model, "model_id", "") or getattr(model, "name", "")
                if not model_id:
                    continue
                input_cost = getattr(model, "cost_per_1k_input_tokens", 0.0)
                output_cost = getattr(model, "cost_per_1k_output_tokens", 0.0)
                self._costs[model_id] = ModelCostInfo(
                    model_id=model_id,
                    input_cost_per_1k=input_cost,
                    output_cost_per_1k=output_cost,
                    source="config",
                )
                count += 1
        return count

    def sync_from_litellm(self) -> int:
        """Attempt to sync costs from litellm's model cost map.

        Returns:
            Number of models synced.
        """
        try:
            import litellm

            cost_map = getattr(litellm, "model_cost", {})
            count = 0
            with self._lock:
                for model_id, info in cost_map.items():
                    if isinstance(info, dict):
                        self._costs[model_id] = ModelCostInfo(
                            model_id=model_id,
                            input_cost_per_1k=info.get("input_cost_per_token", 0) * 1000,
                            output_cost_per_1k=info.get("output_cost_per_token", 0) * 1000,
                            source="litellm",
                        )
                        count += 1
            logger.info("Synced %d model costs from litellm", count)
            return count
        except ImportError:
            logger.debug("litellm not installed, skipping cost sync")
            return 0

    def estimate_request_cost(
        self,
        model_id: str,
        input_tokens: int = 500,
        output_tokens: int = 500,
    ) -> float:
        """Estimate cost for a request to a specific model."""
        info = self.get_cost(model_id)
        if info:
            return info.estimate_request_cost(input_tokens, output_tokens)
        return 0.0

    def get_cheapest_model(self, model_ids: list[str]) -> str | None:
        """Find the cheapest model from a list."""
        cheapest: str | None = None
        lowest_cost = float("inf")
        for model_id in model_ids:
            info = self.get_cost(model_id)
            if info and info.avg_cost_per_1k < lowest_cost:
                lowest_cost = info.avg_cost_per_1k
                cheapest = model_id
        return cheapest

    def summary(self) -> dict[str, dict[str, Any]]:
        """Return cost summary."""
        with self._lock:
            return {
                model_id: {
                    "input_per_1k": f"${info.input_cost_per_1k:.5f}",
                    "output_per_1k": f"${info.output_cost_per_1k:.5f}",
                    "source": info.source,
                }
                for model_id, info in sorted(self._costs.items())
            }

    @property
    def model_count(self) -> int:
        return len(self._costs)
