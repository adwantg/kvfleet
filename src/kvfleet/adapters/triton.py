"""NVIDIA Triton Inference Server adapter."""

from __future__ import annotations

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


class TritonAdapter(InferenceAdapter):
    """Adapter for NVIDIA Triton Inference Server.

    Triton with the vLLM backend supports the OpenAI-compatible
    /v1/chat/completions API. This adapter uses that endpoint and
    adds Triton-specific health/metrics collection.
    """

    def __init__(self, endpoint: str, model_id: str = "", timeout: float = 60.0, **kwargs: Any) -> None:
        super().__init__(endpoint, model_id, timeout, **kwargs)
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.endpoint,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
        return self._client

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send chat completion via Triton's OpenAI-compat endpoint."""
        client = self._get_client()
        payload = request.to_openai_dict()
        if self.model_id:
            payload["model"] = self.model_id
        payload["stream"] = False

        start = time.monotonic()
        resp = await client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
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
            latency_ms=latency,
            endpoint=self.endpoint,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Stream chat from Triton."""
        client = self._get_client()
        payload = request.to_openai_dict()
        if self.model_id:
            payload["model"] = self.model_id
        payload["stream"] = True

        import json as json_mod
        async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json_mod.loads(data_str)
                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    yield StreamChunk(
                        content=delta.get("content", ""),
                        finish_reason=choice.get("finish_reason"),
                        model=data.get("model", self.model_id),
                    )
                except json_mod.JSONDecodeError:
                    continue

    async def health_check(self) -> EndpointHealth:
        """Check Triton health via /v2/health/ready."""
        client = self._get_client()
        start = time.monotonic()
        try:
            resp = await client.get("/v2/health/ready")
            latency = (time.monotonic() - start) * 1000
            return EndpointHealth(
                endpoint=self.endpoint,
                healthy=resp.status_code == 200,
                latency_ms=latency,
                last_checked=time.time(),
            )
        except httpx.RequestError as e:
            return EndpointHealth(endpoint=self.endpoint, healthy=False, error=str(e), last_checked=time.time())

    async def get_metrics(self) -> dict[str, Any]:
        """Get Triton metrics."""
        client = self._get_client()
        try:
            resp = await client.get("/metrics")
            if resp.status_code == 200:
                return {"raw_metrics": resp.text}
        except httpx.RequestError:
            pass
        return {}

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
