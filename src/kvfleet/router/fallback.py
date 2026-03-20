"""Fallback and retry chain logic."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from kvfleet.adapters.base import ChatRequest, ChatResponse, InferenceAdapter
from kvfleet.config.schema import FallbackConfig
from kvfleet.router.explain import RouteExplanation

logger = logging.getLogger(__name__)


class FallbackChain:
    """Manages retry and fallback logic for failed routing attempts.

    Features:
    - Retry on transient failures with exponential backoff
    - Promote to stronger model on timeout
    - Structured fallback chains by route class
    - Graceful degradation to safe default model
    """

    def __init__(self, config: FallbackConfig | None = None) -> None:
        self.config = config or FallbackConfig()

    async def execute_with_fallback(
        self,
        primary_model: str,
        adapters: dict[str, InferenceAdapter],
        request: ChatRequest,
        explanation: RouteExplanation,
        fallback_order: list[str] | None = None,
    ) -> ChatResponse:
        """Execute a request with fallback chain.

        Tries the primary model first, then falls back through the chain
        if transient errors occur.

        Args:
            primary_model: Name of the primary model to try.
            adapters: Map of model name → adapter.
            request: The chat request.
            explanation: Route explanation to update.
            fallback_order: Override fallback order.

        Returns:
            Response from whichever model succeeded.

        Raises:
            RuntimeError: If all models in the chain fail.
        """
        if not self.config.enabled:
            adapter = adapters.get(primary_model)
            if not adapter:
                raise RuntimeError(f"No adapter for model '{primary_model}'")
            return await adapter.chat(request)

        chain = [primary_model]
        fb_order = fallback_order or self.config.fallback_order
        for model_name in fb_order:
            if model_name != primary_model and model_name in adapters:
                chain.append(model_name)

        explanation.fallback_chain = chain
        last_error: Exception | None = None

        for attempt, model_name in enumerate(chain):
            if attempt >= self.config.max_attempts:
                break

            adapter = adapters.get(model_name)
            if not adapter:
                logger.warning("No adapter for fallback model '%s', skipping", model_name)
                continue

            try:
                request_copy = ChatRequest(
                    messages=request.messages,
                    model=model_name,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream=request.stream,
                    top_p=request.top_p,
                    stop=request.stop,
                    tools=request.tools,
                    response_format=request.response_format,
                    metadata=request.metadata,
                )

                # E-9: Use per-request timeout if provided, else config default
                timeout_ms = request.metadata.get("_timeout_ms", self.config.timeout_ms)
                response = await asyncio.wait_for(
                    adapter.chat(request_copy),
                    timeout=timeout_ms / 1000.0,
                )

                if attempt > 0:
                    explanation.fallback_triggered = True
                    logger.info(
                        "Fallback succeeded: %s → %s (attempt %d)",
                        primary_model,
                        model_name,
                        attempt + 1,
                    )

                return response

            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Timeout after {self.config.timeout_ms}ms on {model_name}"
                )
                logger.warning("Timeout on model '%s', trying next in chain", model_name)
            except Exception as e:
                last_error = e
                logger.warning("Error on model '%s': %s, trying next", model_name, e)
                # Brief backoff before retry
                if attempt < len(chain) - 1:
                    await asyncio.sleep(min(0.5 * (2**attempt), 5.0))

        explanation.fallback_triggered = True
        raise RuntimeError(
            f"All {len(chain)} models in fallback chain failed. Last error: {last_error}"
        )


class EscalationChain:
    """Confidence-based auto-escalation chain.

    If output confidence is low, escalate through increasingly
    capable models.
    """

    def __init__(
        self,
        chain: list[str] | None = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        self.chain = chain or []
        self.confidence_threshold = confidence_threshold

    async def execute_with_escalation(
        self,
        adapters: dict[str, InferenceAdapter],
        request: ChatRequest,
        confidence_scorer: Any | None = None,
    ) -> tuple[ChatResponse, str]:
        """Execute request with potential escalation.

        Args:
            adapters: Model name → adapter map.
            request: The chat request.
            confidence_scorer: Optional callable(response) → float.

        Returns:
            Tuple of (response, model_name_that_answered).
        """
        for model_name in self.chain:
            adapter = adapters.get(model_name)
            if not adapter:
                continue

            response = await adapter.chat(request)

            # If no confidence scorer, return first result
            if confidence_scorer is None:
                return response, model_name

            confidence = confidence_scorer(response)
            if confidence >= self.confidence_threshold:
                return response, model_name

            logger.info(
                "Model '%s' confidence %.2f < threshold %.2f, escalating",
                model_name,
                confidence,
                self.confidence_threshold,
            )

        # Return last response if all below threshold
        if self.chain and self.chain[-1] in adapters:
            adapter = adapters[self.chain[-1]]
            response = await adapter.chat(request)
            return response, self.chain[-1]

        raise RuntimeError("No models available in escalation chain")
