"""OpenAI-compatible inference adapter.

Works with any endpoint that implements the OpenAI chat completions API,
including vLLM in OpenAI-compat mode, LiteLLM, FastChat, etc.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

import httpx

from kvfleet.adapters.base import (
    ChatRequest,
    ChatResponse,
    EndpointHealth,
    InferenceAdapter,
    StreamChunk,
    Usage,
)

logger = logging.getLogger(__name__)


class OpenAICompatAdapter(InferenceAdapter):
    """Adapter for OpenAI-compatible inference endpoints."""

    def __init__(
        self,
        endpoint: str,
        model_id: str = "",
        timeout: float = 60.0,
        api_key: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(endpoint, model_id, timeout, **kwargs)
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.endpoint,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat completion request."""
        client = self._get_client()
        payload = request.to_openai_dict()
        if self.model_id:
            payload["model"] = self.model_id
        payload["stream"] = False

        start = time.monotonic()
        try:
            resp = await client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error from %s: %s", self.endpoint, e)
            raise
        except httpx.RequestError as e:
            logger.error("Request error to %s: %s", self.endpoint, e)
            raise
        latency = (time.monotonic() - start) * 1000

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage_data = data.get("usage", {})

        return ChatResponse(
            content=message.get("content", ""),
            model=data.get("model", self.model_id),
            finish_reason=choice.get("finish_reason", "stop"),
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            tool_calls=message.get("tool_calls"),
            latency_ms=latency,
            endpoint=self.endpoint,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Send a streaming chat completion request."""
        client = self._get_client()
        payload = request.to_openai_dict()
        if self.model_id:
            payload["model"] = self.model_id
        payload["stream"] = True

        async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    yield StreamChunk(
                        content=delta.get("content", ""),
                        finish_reason=choice.get("finish_reason"),
                        model=data.get("model", self.model_id),
                    )
                except json.JSONDecodeError:
                    continue

    async def health_check(self) -> EndpointHealth:
        """Check endpoint health via /health or /v1/models."""
        client = self._get_client()
        start = time.monotonic()
        try:
            resp = await client.get("/health")
            healthy = resp.status_code == 200
            latency = (time.monotonic() - start) * 1000
            return EndpointHealth(
                endpoint=self.endpoint,
                healthy=healthy,
                latency_ms=latency,
                last_checked=time.time(),
            )
        except httpx.RequestError as e:
            return EndpointHealth(
                endpoint=self.endpoint,
                healthy=False,
                error=str(e),
                last_checked=time.time(),
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
