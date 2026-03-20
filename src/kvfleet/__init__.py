"""kvfleet — KV-cache-aware intelligent routing for self-hosted and hybrid LLM fleets."""

__version__ = "0.11.2"

from kvfleet.config.loader import load_config, save_config
from kvfleet.config.schema import FleetConfig, ModelConfig, RouteStrategy
from kvfleet.registry.models import ModelRegistry
from kvfleet.router.engine import Router
from kvfleet.router.explain import RouteExplanation
from kvfleet.sdk.async_client import AsyncFleetClient
from kvfleet.sdk.sync_client import SyncFleetClient

__all__ = [
    "AsyncFleetClient",
    "FleetConfig",
    "ModelConfig",
    "ModelRegistry",
    "RouteExplanation",
    "RouteStrategy",
    "Router",
    "SyncFleetClient",
    "__version__",
    "load_config",
    "save_config",
]
