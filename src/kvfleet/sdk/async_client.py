"""Async SDK client for kvfleet."""

from __future__ import annotations

from typing import Any

from kvfleet.adapters.base import ChatResponse
from kvfleet.config.loader import load_config
from kvfleet.config.schema import FleetConfig
from kvfleet.router.engine import Router
from kvfleet.router.explain import RouteExplanation


class AsyncFleetClient:
    """Async Python SDK for kvfleet.

    Usage:
        async with AsyncFleetClient.from_yaml("fleet.yaml") as client:
            response = await client.chat("What is Python?")
            print(response.content)
    """

    def __init__(self, config: FleetConfig) -> None:
        self._config = config
        self._router = Router(config)

    @classmethod
    def from_yaml(cls, path: str) -> AsyncFleetClient:
        """Create client from a YAML config file."""
        config = load_config(path)
        return cls(config)

    @classmethod
    def from_config(cls, config: FleetConfig) -> AsyncFleetClient:
        """Create client from a FleetConfig object."""
        return cls(config)

    async def chat(
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
        """Send a chat request and get a response.

        Args:
            prompt: Simple prompt string.
            messages: Chat messages (list of role/content dicts).
            data_class: Data classification for policy.
            tenant_id: Tenant for routing.
            tags: Request tags.
            temperature: Generation temperature.
            max_tokens: Max tokens.
            tools: Tool definitions.
            response_format: Response format.

        Returns:
            ChatResponse from the selected model.
        """
        response, _ = await self._router.route(
            messages=messages,
            prompt=prompt,
            data_class=data_class,
            tenant_id=tenant_id,
            tags=tags,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            response_format=response_format,
        )
        return response

    async def chat_with_explanation(
        self,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> tuple[ChatResponse, RouteExplanation]:
        """Chat and also get the routing explanation."""
        return await self._router.route(prompt=prompt, **kwargs)

    async def simulate(
        self,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> RouteExplanation:
        """Simulate routing without calling any backend."""
        return await self._router.simulate(prompt=prompt, **kwargs)

    async def health(self) -> dict[str, Any]:
        """Health check all endpoints."""
        return await self._router.health_check_all()

    @property
    def router(self) -> Router:
        """Access the underlying router for advanced use."""
        return self._router

    async def close(self) -> None:
        """Close the client and its connections."""
        await self._router.close()

    async def __aenter__(self) -> AsyncFleetClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
