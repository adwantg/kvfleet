"""Synchronous SDK client for kvfleet."""

from __future__ import annotations

import asyncio
from typing import Any

from kvfleet.adapters.base import ChatResponse
from kvfleet.config.loader import load_config
from kvfleet.config.schema import FleetConfig
from kvfleet.router.explain import RouteExplanation
from kvfleet.sdk.async_client import AsyncFleetClient


def _get_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop."""
    try:
        loop = asyncio.get_running_loop()
        # If there's a running loop, we can't use run_until_complete
        raise RuntimeError(
            "Cannot use SyncFleetClient inside an async context. "
            "Use AsyncFleetClient instead."
        )
    except RuntimeError as e:
        if "no current event loop" in str(e).lower() or "no running event loop" in str(e).lower():
            return asyncio.new_event_loop()
        raise


class SyncFleetClient:
    """Synchronous Python SDK for kvfleet.

    Wraps the async client for use in synchronous code paths.

    Usage:
        with SyncFleetClient.from_yaml("fleet.yaml") as client:
            response = client.chat("What is Python?")
            print(response.content)
    """

    def __init__(self, config: FleetConfig) -> None:
        self._async_client = AsyncFleetClient(config)
        self._loop = asyncio.new_event_loop()

    @classmethod
    def from_yaml(cls, path: str) -> SyncFleetClient:
        """Create client from a YAML config file."""
        config = load_config(path)
        return cls(config)

    @classmethod
    def from_config(cls, config: FleetConfig) -> SyncFleetClient:
        """Create client from a FleetConfig object."""
        return cls(config)

    def chat(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        data_class: str | None = None,
        tenant_id: str | None = None,
        tags: dict[str, str] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> ChatResponse:
        """Send a chat request (synchronous)."""
        return self._loop.run_until_complete(
            self._async_client.chat(
                prompt=prompt,
                messages=messages,
                data_class=data_class,
                tenant_id=tenant_id,
                tags=tags,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                response_format=response_format,
            )
        )

    def chat_with_explanation(
        self,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> tuple[ChatResponse, RouteExplanation]:
        """Chat and get routing explanation (synchronous)."""
        return self._loop.run_until_complete(
            self._async_client.chat_with_explanation(prompt=prompt, **kwargs)
        )

    def simulate(self, prompt: str | None = None, **kwargs: Any) -> RouteExplanation:
        """Simulate routing (synchronous)."""
        return self._loop.run_until_complete(
            self._async_client.simulate(prompt=prompt, **kwargs)
        )

    def health(self) -> dict[str, Any]:
        """Health check all endpoints (synchronous)."""
        return self._loop.run_until_complete(self._async_client.health())

    def close(self) -> None:
        """Close the client."""
        self._loop.run_until_complete(self._async_client.close())
        self._loop.close()

    def __enter__(self) -> SyncFleetClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
