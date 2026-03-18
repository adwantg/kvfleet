"""Custom HTTP adapter for proprietary internal inference services."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from kvfleet.adapters.base import (
    ChatRequest,
    ChatResponse,
    EndpointHealth,
    InferenceAdapter,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class CustomHTTPAdapter(InferenceAdapter):
    """Generic HTTP adapter for custom inference endpoints.

    This adapter sends requests to a configurable HTTP endpoint. It expects
    the response to contain a 'content' or 'text' field in JSON format.
    Headers, paths, and payload structure are configurable via kwargs.
    """

    def __init__(
        self,
        endpoint: str,
        model_id: str = "",
        timeout: float = 60.0,
        chat_path: str = "/generate",
        health_path: str = "/health",
        headers: dict[str, str] | None = None,
        request_template: dict[str, Any] | None = None,
        response_content_key: str = "content",
        **kwargs: Any,
    ) -> None:
        super().__init__(endpoint, model_id, timeout, **kwargs)
        self.chat_path = chat_path
        self.health_path = health_path
        self.extra_headers = headers or {}
        self.request_template = request_template or {}
        self.response_content_key = response_content_key
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            all_headers = {"Content-Type": "application/json", **self.extra_headers}
            self._client = httpx.AsyncClient(
                base_url=self.endpoint, headers=all_headers, timeout=self.timeout
            )
        return self._client

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to the custom endpoint."""
        client = self._get_client()
        payload = {
            **self.request_template,
            "prompt": request.messages[-1].content if request.messages else "",
            "model": self.model_id or request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        start = time.monotonic()
        resp = await client.post(self.chat_path, json=payload)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.monotonic() - start) * 1000

        content = data.get(self.response_content_key, data.get("text", data.get("output", "")))

        return ChatResponse(
            content=str(content),
            model=self.model_id,
            latency_ms=latency,
            endpoint=self.endpoint,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Stream responses via SSE (Server-Sent Events).

        Falls back to single-chunk response if streaming endpoint
        doesn't return SSE format.
        """
        client = self._get_client()
        payload = {
            **self.request_template,
            "prompt": request.messages[-1].content if request.messages else "",
            "model": self.model_id or request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        try:
            async with client.stream("POST", self.chat_path, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            return
                        try:
                            import json

                            data = json.loads(data_str)
                            content = data.get(self.response_content_key, data.get("text", ""))
                            yield StreamChunk(
                                content=str(content),
                                model=self.model_id,
                                finish_reason=data.get("finish_reason"),
                            )
                        except Exception:
                            yield StreamChunk(content=data_str, model=self.model_id)
                    elif line:
                        # Non-SSE streaming: treat each line as a token
                        yield StreamChunk(content=line, model=self.model_id)
        except httpx.RequestError:
            # Fallback to non-streaming
            logger.warning("Streaming failed for custom endpoint, falling back to non-streaming")
            response = await self.chat(request)
            yield StreamChunk(content=response.content, finish_reason="stop", model=response.model)

    async def health_check(self) -> EndpointHealth:
        """Check custom endpoint health."""
        client = self._get_client()
        start = time.monotonic()
        try:
            resp = await client.get(self.health_path)
            return EndpointHealth(
                endpoint=self.endpoint,
                healthy=resp.status_code == 200,
                latency_ms=(time.monotonic() - start) * 1000,
                last_checked=time.time(),
            )
        except httpx.RequestError as e:
            return EndpointHealth(
                endpoint=self.endpoint, healthy=False, error=str(e), last_checked=time.time()
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
