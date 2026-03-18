"""Abstract base adapter for inference backends."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class ChatRequest:
    """Request to a chat completion endpoint."""

    messages: list[ChatMessage]
    model: str = ""
    temperature: float = 1.0
    max_tokens: int | None = None
    stream: bool = False
    top_p: float = 1.0
    stop: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        data: dict[str, Any] = {
            "messages": [
                {k: v for k, v in {"role": m.role, "content": m.content, "name": m.name}.items() if v is not None}
                for m in self.messages
            ],
            "model": self.model,
            "temperature": self.temperature,
            "stream": self.stream,
            "top_p": self.top_p,
        }
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.stop:
            data["stop"] = self.stop
        if self.tools:
            data["tools"] = self.tools
        if self.response_format:
            data["response_format"] = self.response_format
        return data


@dataclass
class Usage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    """Response from a chat completion endpoint."""

    content: str
    model: str = ""
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[dict[str, Any]] | None = None
    latency_ms: float = 0.0
    endpoint: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_cost(self) -> float:
        """Rough cost estimate (placeholder, actual pricing set by model config)."""
        return 0.0


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    content: str = ""
    finish_reason: str | None = None
    model: str = ""


@dataclass
class EndpointHealth:
    """Health status of an endpoint."""

    endpoint: str
    healthy: bool
    latency_ms: float = 0.0
    queue_depth: int = 0
    active_requests: int = 0
    gpu_memory_used_pct: float = 0.0
    gpu_utilization_pct: float = 0.0
    error: str | None = None
    last_checked: float = 0.0
    kv_cache_usage_pct: float = 0.0
    tokens_per_second: float = 0.0

    @property
    def load_score(self) -> float:
        """Compute a 0-1 load score (higher = more loaded)."""
        factors = [
            min(self.queue_depth / 100.0, 1.0) * 0.3,
            min(self.active_requests / 50.0, 1.0) * 0.3,
            self.gpu_memory_used_pct / 100.0 * 0.2,
            self.gpu_utilization_pct / 100.0 * 0.2,
        ]
        return min(sum(factors), 1.0)


class InferenceAdapter(abc.ABC):
    """Abstract base class for all inference backend adapters.

    Each adapter translates kvfleet's internal request/response format
    to the backend's API format and handles communication.
    """

    def __init__(self, endpoint: str, model_id: str = "", timeout: float = 60.0, **kwargs: Any) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout

    @abc.abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat completion request.

        Args:
            request: Chat request with messages.

        Returns:
            Chat response with content and usage.
        """

    @abc.abstractmethod
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Send a streaming chat completion request.

        Args:
            request: Chat request with stream=True.

        Yields:
            Stream chunks with content deltas.
        """
        yield StreamChunk()  # pragma: no cover

    @abc.abstractmethod
    async def health_check(self) -> EndpointHealth:
        """Check the health status of this endpoint.

        Returns:
            Health status with metrics.
        """

    async def get_cache_state(self) -> dict[str, Any]:
        """Get KV-cache state from the backend (optional).

        Returns:
            Cache state dict (e.g., cache_usage_pct, cached_sequences).
            Returns empty dict if backend doesn't support cache inspection.
        """
        return {}

    async def get_metrics(self) -> dict[str, Any]:
        """Get runtime metrics from the backend (optional).

        Returns:
            Metrics dict (e.g., queue_depth, tokens_per_second).
        """
        return {}

    def _now_ms(self) -> float:
        """Current time in milliseconds."""
        return time.time() * 1000
