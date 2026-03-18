"""Ollama inference adapter."""

from __future__ import annotations

import json
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
    Usage,
)

logger = logging.getLogger(__name__)


class OllamaAdapter(InferenceAdapter):
    """Adapter for Ollama inference servers.

    Ollama uses its own API format at /api/chat.
    """

    def __init__(
        self, endpoint: str, model_id: str = "", timeout: float = 60.0, **kwargs: Any
    ) -> None:
        super().__init__(endpoint, model_id, timeout, **kwargs)
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.endpoint, timeout=self.timeout)
        return self._client

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to Ollama."""
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self.model_id or request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
        }
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens

        start = time.monotonic()
        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.monotonic() - start) * 1000

        message = data.get("message", {})
        return ChatResponse(
            content=message.get("content", ""),
            model=data.get("model", self.model_id),
            finish_reason="stop" if data.get("done") else "length",
            usage=Usage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            ),
            latency_ms=latency,
            endpoint=self.endpoint,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Stream chat from Ollama."""
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self.model_id or request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": True,
            "options": {"temperature": request.temperature, "top_p": request.top_p},
        }

        async with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    message = data.get("message", {})
                    yield StreamChunk(
                        content=message.get("content", ""),
                        finish_reason="stop" if data.get("done") else None,
                        model=data.get("model", self.model_id),
                    )
                except json.JSONDecodeError:
                    continue

    async def health_check(self) -> EndpointHealth:
        """Check Ollama health via /api/tags."""
        client = self._get_client()
        start = time.monotonic()
        try:
            resp = await client.get("/api/tags")
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
