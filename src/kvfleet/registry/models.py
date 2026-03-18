"""Unified model registry — single source of truth for all routable models."""

from __future__ import annotations

import logging
from typing import Any

from kvfleet.config.schema import ModelConfig, ProviderType

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing the fleet of available models.

    The registry stores model configurations and provides lookup, filtering,
    and lifecycle management for the routing engine.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelConfig] = {}

    def register(self, config: ModelConfig) -> None:
        """Register a model in the fleet.

        Args:
            config: Model configuration to register.

        Raises:
            ValueError: If a model with the same name already exists.
        """
        if config.name in self._models:
            raise ValueError(f"Model '{config.name}' is already registered")
        self._models[config.name] = config
        logger.info("Registered model '%s' (%s @ %s)", config.name, config.provider.value, config.endpoint)

    def unregister(self, name: str) -> None:
        """Remove a model from the registry.

        Args:
            name: Model name to remove.

        Raises:
            KeyError: If the model is not registered.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered")
        del self._models[name]
        logger.info("Unregistered model '%s'", name)

    def get(self, name: str) -> ModelConfig:
        """Get a model by name.

        Args:
            name: Model name.

        Returns:
            Model configuration.

        Raises:
            KeyError: If model not found.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self._models[name]

    def list_models(
        self,
        *,
        enabled_only: bool = True,
        provider: ProviderType | None = None,
        tags: dict[str, str] | None = None,
        data_class: str | None = None,
        min_quality: float | None = None,
        max_cost: float | None = None,
        supports_tools: bool | None = None,
        supports_json_mode: bool | None = None,
        supports_streaming: bool | None = None,
    ) -> list[ModelConfig]:
        """List models with optional filtering.

        Args:
            enabled_only: Only return enabled models.
            provider: Filter by provider type.
            tags: Filter by matching tags (all must match).
            data_class: Filter by allowed data class.
            min_quality: Minimum quality score.
            max_cost: Maximum cost per 1K input tokens.
            supports_tools: Filter by tool calling support.
            supports_json_mode: Filter by JSON mode support.
            supports_streaming: Filter by streaming support.

        Returns:
            List of matching model configs.
        """
        result: list[ModelConfig] = []
        for model in self._models.values():
            if enabled_only and not model.enabled:
                continue
            if provider is not None and model.provider != provider:
                continue
            if tags:
                if not all(model.tags.get(k) == v for k, v in tags.items()):
                    continue
            if data_class is not None and data_class not in model.allowed_data_classes:
                continue
            if min_quality is not None and model.quality_score < min_quality:
                continue
            if max_cost is not None and model.cost_per_1k_input_tokens > max_cost:
                continue
            if supports_tools is not None and model.capabilities.supports_tools != supports_tools:
                continue
            if supports_json_mode is not None and model.capabilities.supports_json_mode != supports_json_mode:
                continue
            if supports_streaming is not None and model.capabilities.supports_streaming != supports_streaming:
                continue
            result.append(model)
        return result

    def update(self, name: str, **kwargs: Any) -> ModelConfig:
        """Update a model's configuration fields.

        Args:
            name: Model to update.
            **kwargs: Fields to update.

        Returns:
            Updated model config.
        """
        model = self.get(name)
        updated = model.model_copy(update=kwargs)
        self._models[name] = updated
        return updated

    def enable(self, name: str) -> None:
        """Enable a model for routing."""
        self.update(name, enabled=True)

    def disable(self, name: str) -> None:
        """Disable a model from routing."""
        self.update(name, enabled=False)

    @property
    def count(self) -> int:
        """Total number of registered models."""
        return len(self._models)

    @property
    def enabled_count(self) -> int:
        """Number of enabled models."""
        return sum(1 for m in self._models.values() if m.enabled)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the registry."""
        models = list(self._models.values())
        return {
            "total": len(models),
            "enabled": sum(1 for m in models if m.enabled),
            "providers": list({m.provider.value for m in models}),
            "models": [
                {
                    "name": m.name,
                    "provider": m.provider.value,
                    "endpoint": m.endpoint,
                    "enabled": m.enabled,
                    "quality": m.quality_score,
                    "cost_1k_input": m.cost_per_1k_input_tokens,
                }
                for m in models
            ],
        }

    @classmethod
    def from_configs(cls, configs: list[ModelConfig]) -> ModelRegistry:
        """Create a registry from a list of model configs.

        Args:
            configs: List of model configurations.

        Returns:
            Populated ModelRegistry.
        """
        registry = cls()
        for config in configs:
            registry.register(config)
        return registry
