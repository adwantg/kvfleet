"""TGI (Text Generation Inference) adapter."""

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


class TGIAdapter(InferenceAdapter):
    """Adapter for HuggingFace Text Generation Inference (TGI) servers.

    TGI supports the OpenAI-compatible /v1/chat/completions API as well
    as its native /generate and /generate_stream endpoints.
    This adapter uses the OpenAI-compat API for consistency and also
    reads TGI-specific health/metrics endpoints.
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
        """Send chat completion via TGI's generate endpoint."""
        client = self._get_client()
        # Build TGI /generate payload
        prompt = self._messages_to_prompt(request.messages)
        payload: dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_new_tokens": request.max_tokens or 512,
                "return_full_text": False,
            },
        }
        if request.stop:
            payload["parameters"]["stop"] = request.stop

        start = time.monotonic()
        resp = await client.post("/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.monotonic() - start) * 1000

        generated = data.get("generated_text", "")
        details = data.get("details", {})

        return ChatResponse(
            content=generated,
            model=self.model_id,
            finish_reason=details.get("finish_reason", "stop"),
            usage=Usage(
                prompt_tokens=details.get("prefill_tokens", 0) if details else 0,
                completion_tokens=details.get("generated_tokens", 0) if details else 0,
                total_tokens=(details.get("prefill_tokens", 0) + details.get("generated_tokens", 0)) if details else 0,
            ),
            latency_ms=latency,
            endpoint=self.endpoint,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Stream generation from TGI."""
        client = self._get_client()
        prompt = self._messages_to_prompt(request.messages)
        payload: dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_new_tokens": request.max_tokens or 512,
            },
        }

        async with client.stream("POST", "/generate_stream", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    import json
                    try:
                        data = json.loads(line[5:])
                        token = data.get("token", {})
                        yield StreamChunk(
                            content=token.get("text", ""),
                            finish_reason="stop" if data.get("details") else None,
                            model=self.model_id,
                        )
                    except json.JSONDecodeError:
                        continue

    async def health_check(self) -> EndpointHealth:
        """Check TGI health."""
        client = self._get_client()
        start = time.monotonic()
        try:
            resp = await client.get("/health")
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
        """Get TGI metrics from /metrics endpoint."""
        client = self._get_client()
        try:
            resp = await client.get("/metrics")
            if resp.status_code == 200:
                return {"raw_metrics": resp.text}
        except httpx.RequestError:
            pass
        return {}

    @staticmethod
    def _messages_to_prompt(messages: list[Any]) -> str:
        """Convert chat messages to a single prompt string for TGI."""
        parts = []
        for msg in messages:
            role = msg.role.capitalize()
            parts.append(f"{role}: {msg.content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
