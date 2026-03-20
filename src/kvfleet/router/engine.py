"""Core routing engine — the main entry point for kvfleet routing."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from kvfleet.adapters.base import ChatMessage, ChatRequest, ChatResponse, InferenceAdapter
from kvfleet.adapters.custom_http import CustomHTTPAdapter
from kvfleet.adapters.ollama import OllamaAdapter
from kvfleet.adapters.openai_compat import OpenAICompatAdapter
from kvfleet.adapters.tgi import TGIAdapter
from kvfleet.adapters.triton import TritonAdapter
from kvfleet.adapters.vllm import VLLMAdapter
from kvfleet.cache.fingerprints import PromptFingerprinter
from kvfleet.cache.kv_affinity import KVAffinityScorer
from kvfleet.config.schema import FleetConfig, ProviderType, RouteStrategy
from kvfleet.eval.shadow import ShadowTrafficManager
from kvfleet.policy.engine import PolicyContext, PolicyEngine
from kvfleet.policy.pii import PIIDetector
from kvfleet.policy.tenant import TenantManager
from kvfleet.registry.models import ModelRegistry
from kvfleet.router.explain import RouteExplanation
from kvfleet.router.fallback import FallbackChain
from kvfleet.router.multimodal import filter_json_mode_capable, filter_tool_capable
from kvfleet.router.scoring import ScoringContext, ScoringEngine
from kvfleet.router.strategies import RoutingStrategy, get_strategy
from kvfleet.telemetry.collector import TelemetryCollector
from kvfleet.telemetry.health import HealthManager
from kvfleet.telemetry.metrics import MetricsExporter

logger = logging.getLogger(__name__)

# Adapter factory map
ADAPTER_MAP: dict[ProviderType, type[InferenceAdapter]] = {
    ProviderType.VLLM: VLLMAdapter,
    ProviderType.OPENAI_COMPAT: OpenAICompatAdapter,
    ProviderType.OLLAMA: OllamaAdapter,
    ProviderType.TGI: TGIAdapter,
    ProviderType.TRITON: TritonAdapter,
    ProviderType.CUSTOM_HTTP: CustomHTTPAdapter,
}


class Router:
    """The main kvfleet routing engine.

    Orchestrates all subsystems to intelligently route each request
    to the best model and best replica.

    Usage:
        config = load_config("fleet.yaml")
        router = Router(config)
        response, explanation = await router.route(messages=[...])
    """

    def __init__(self, config: FleetConfig) -> None:
        self.config = config

        # Core components
        self.registry = ModelRegistry.from_configs(config.models)
        self.scoring_engine = ScoringEngine(config.scoring_weights)
        self.strategy = self._create_strategy()
        self.fallback_chain = FallbackChain(config.fallback)
        self.fingerprinter = PromptFingerprinter(
            prefix_tokens=config.cache_affinity.prefix_hash_tokens
        )
        self.affinity_scorer = KVAffinityScorer(
            virtual_nodes=config.cache_affinity.consistent_hash_replicas,
            session_ttl=config.cache_affinity.session_ttl_seconds,
            min_affinity_score=config.cache_affinity.min_affinity_score,
        )

        # Subsystems
        self.policy_engine = PolicyEngine(config.policy)
        self.pii_detector = PIIDetector()
        self.tenant_manager = TenantManager(config.tenants)
        self.health_manager = HealthManager(
            check_interval_seconds=config.telemetry.health_check_interval_seconds
        )
        self.telemetry = TelemetryCollector(
            poll_interval_seconds=config.telemetry.health_check_interval_seconds
        )
        self.metrics = MetricsExporter(
            port=config.telemetry.prometheus_port, enabled=config.telemetry.prometheus_enabled
        )
        self.shadow_manager = ShadowTrafficManager(
            sample_rate=config.shadow.sample_rate,
            shadow_models=config.shadow.shadow_models,
            log_outputs=config.shadow.log_outputs,
            enabled=config.shadow.enabled,
        )

        # Adapters
        self._adapters: dict[str, InferenceAdapter] = {}
        self._init_adapters()
        self._init_affinity_ring()

    def _create_strategy(self) -> RoutingStrategy:
        """Create the routing strategy from config."""
        kwargs: dict[str, Any] = {}
        if self.config.strategy.value == "static":
            kwargs["default_model"] = self.config.default_model
        elif self.config.strategy.value == "rules":
            kwargs["rules"] = self.config.route_rules
        elif self.config.strategy.value == "hybrid_score":
            kwargs["scoring_engine"] = self.scoring_engine
        return get_strategy(self.config.strategy.value, **kwargs)

    def _init_adapters(self) -> None:
        """Initialize adapters for all registered models."""
        for model in self.config.models:
            if not model.enabled:
                continue
            adapter_cls = ADAPTER_MAP.get(model.provider, OpenAICompatAdapter)
            adapter = adapter_cls(
                endpoint=model.endpoint,
                model_id=model.get_model_id(),
                timeout=model.timeout_seconds,
            )
            self._adapters[model.name] = adapter
            self.telemetry.register_adapter(model.name, adapter)

    def _init_affinity_ring(self) -> None:
        """Initialize the KV-cache affinity hash ring."""
        for model in self.config.models:
            if model.enabled:
                self.affinity_scorer.register_endpoints(model.name, model.all_endpoints())

    async def route(
        self,
        messages: list[dict[str, str]] | list[ChatMessage] | None = None,
        prompt: str | None = None,
        *,
        request: ChatRequest | None = None,
        data_class: str | None = None,
        tenant_id: str | None = None,
        tags: dict[str, str] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, str] | None = None,
        request_id: str | None = None,
        strategy_override: RouteStrategy | None = None,
        model_allowlist: list[str] | None = None,
    ) -> tuple[ChatResponse, RouteExplanation]:
        """Route a request to the best model.

        Args:
            messages: Chat messages (list of dicts or ChatMessage objects).
            prompt: Simple prompt string (alternative to messages).
            request: Pre-built ChatRequest (overrides messages/prompt).
            data_class: Data classification for policy evaluation.
            tenant_id: Tenant identifier for tenant-aware routing.
            tags: Request tags for rule matching.
            temperature: Temperature for generation.
            max_tokens: Max tokens to generate.
            stream: Whether to stream the response.
            tools: Tool definitions.
            response_format: Response format spec.
            request_id: Optional request ID for tracing.

        Returns:
            Tuple of (ChatResponse, RouteExplanation).
        """
        start_time = time.monotonic()
        req_id = request_id or str(uuid.uuid4())[:12]

        # Build request
        if request is None:
            chat_messages = self._normalize_messages(messages, prompt)
            request = ChatRequest(
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                tools=tools,
                response_format=response_format,
            )

        # Create explanation
        explanation = RouteExplanation(
            request_id=req_id,
            strategy=self.config.strategy.value,
        )

        # Resolve active strategy (E-3: per-request override)
        active_strategy = self.strategy
        if strategy_override is not None:
            try:
                kwargs_s: dict[str, Any] = {}
                if strategy_override.value == "hybrid_score":
                    kwargs_s["scoring_engine"] = self.scoring_engine
                active_strategy = get_strategy(strategy_override.value, **kwargs_s)
                explanation.strategy = strategy_override.value
                explanation.strategy_overridden = True
            except (ValueError, KeyError):
                logger.warning(
                    "Invalid strategy override '%s', using default",
                    strategy_override.value,
                )

        # Step 1: Get candidate models (E-6: only chat models)
        candidates = self.registry.list_models(enabled_only=True, model_type="chat")
        if not candidates:
            raise RuntimeError("No enabled models in registry")

        # Step 1.5: Model allowlist filtering (E-4)
        if model_allowlist:
            candidates = [c for c in candidates if c.name in model_allowlist]
            if not candidates:
                raise RuntimeError(f"No enabled models match allowlist: {model_allowlist}")
            explanation.metadata["model_allowlist"] = model_allowlist

        # Step 2: Policy evaluation
        if self.config.policy.enabled:
            prompt_text = " ".join(m.content for m in request.messages)
            has_pii = self.config.policy.pii_detection and self.pii_detector.has_pii(prompt_text)
            policy_ctx = PolicyContext(
                data_class=data_class or self.config.policy.default_data_class,
                tenant_id=tenant_id,
                tags=tags or {},
                has_pii=has_pii,
                prompt_text=prompt_text,
            )
            candidates, policy_decisions = self.policy_engine.evaluate(candidates, policy_ctx)
            explanation.policy_decisions = policy_decisions

            if not candidates:
                raise RuntimeError("All models filtered by policy — no candidates remain")

        # Step 3: Tenant filtering
        if tenant_id:
            candidate_names = self.tenant_manager.filter_models_for_tenant(
                tenant_id, [c.name for c in candidates]
            )
            candidates = [c for c in candidates if c.name in candidate_names]

        # Step 3.5: Capability filtering (E-2, E-8)
        candidates = filter_tool_capable(candidates, request)
        candidates = filter_json_mode_capable(candidates, request)
        if not candidates:
            raise RuntimeError("All models filtered by capability checks — no candidates remain")

        # Step 4: Fingerprint and compute cache affinity
        fingerprint = self.fingerprinter.fingerprint(request.messages)
        cache_affinity_scores: dict[str, float] = {}

        if self.config.cache_affinity.enabled:
            for model in candidates:
                endpoints = model.all_endpoints()
                affinity = self.affinity_scorer.score_affinity(fingerprint, model.name, endpoints)
                max_affinity = max(affinity.values()) if affinity else 0.0
                cache_affinity_scores[model.name] = max_affinity
            explanation.cache_affinity_used = True
            explanation.cache_hit = any(
                s > self.config.cache_affinity.min_affinity_score
                for s in cache_affinity_scores.values()
            )

        # Step 5: Build scoring context
        scoring_ctx = ScoringContext(
            data_class=data_class or "internal",
            tenant_id=tenant_id,
            tags=tags,
            cache_affinity_scores=cache_affinity_scores,
            endpoint_health=dict(self.health_manager._health.items())
            if self.health_manager._health
            else None,
        )

        # Step 6: Strategy selection
        candidate_scores = active_strategy.select(candidates, scoring_ctx)
        explanation.candidates = candidate_scores

        # Find selected model
        selected = next(
            (c for c in candidate_scores if c.selected),
            candidate_scores[0] if candidate_scores else None,
        )
        if selected is None:
            raise RuntimeError("Strategy did not select any model")

        selected_model = selected.model_name
        explanation.selected_model = selected_model
        explanation.selected_endpoint = selected.endpoint

        # Step 7: Execute with fallback
        try:
            response = await self.fallback_chain.execute_with_fallback(
                primary_model=selected_model,
                adapters=self._adapters,
                request=request,
                explanation=explanation,
                fallback_order=self.config.fallback.fallback_order,
            )
        except RuntimeError:
            explanation.fallback_triggered = True
            raise

        # Step 8: Record affinity
        if self.config.cache_affinity.enabled:
            self.affinity_scorer.record_routing(fingerprint, response.endpoint or selected.endpoint)

        # Step 9: Shadow traffic
        if self.shadow_manager.should_shadow():
            explanation.shadow_models = self.shadow_manager.shadow_models
            import asyncio

            self._shadow_task = asyncio.create_task(
                self.shadow_manager.execute_shadow(
                    request=request,
                    primary_model=selected_model,
                    primary_response=response,
                    adapters=self._adapters,
                    request_id=req_id,
                )
            )

        # Step 10: Record metrics
        total_latency = (time.monotonic() - start_time) * 1000
        explanation.total_latency_ms = total_latency
        self.metrics.record_route(
            strategy=self.config.strategy.value,
            model=selected_model,
            latency_seconds=total_latency / 1000,
        )

        # Step 11: Record spend for tenant
        if tenant_id:
            estimated_cost = (
                response.usage.prompt_tokens * selected.signals.get("cost_per_1k", 0) / 1000
                + response.usage.completion_tokens * selected.signals.get("cost_per_1k", 0) / 1000
            )
            self.tenant_manager.record_request(tenant_id, estimated_cost)

        if self.config.telemetry.log_route_decisions:
            logger.info("Route decision: %s → %s (%.1fms)", req_id, selected_model, total_latency)

        return response, explanation

    async def simulate(
        self,
        messages: list[dict[str, str]] | list[ChatMessage] | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> RouteExplanation:
        """Simulate routing without sending the actual request.

        Returns just the route explanation without calling any backend.
        """
        start_time = time.monotonic()
        req_id = kwargs.get("request_id") or str(uuid.uuid4())[:12]
        chat_messages = self._normalize_messages(messages, prompt)
        request = ChatRequest(messages=chat_messages)

        explanation = RouteExplanation(
            request_id=req_id,
            strategy=self.config.strategy.value,
        )

        candidates = self.registry.list_models(enabled_only=True)
        fingerprint = self.fingerprinter.fingerprint(request.messages)

        cache_affinity_scores: dict[str, float] = {}
        if self.config.cache_affinity.enabled:
            for model in candidates:
                affinity = self.affinity_scorer.score_affinity(
                    fingerprint, model.name, model.all_endpoints()
                )
                cache_affinity_scores[model.name] = max(affinity.values()) if affinity else 0.0
            explanation.cache_affinity_used = True

        scoring_ctx = ScoringContext(
            data_class=kwargs.get("data_class", "internal"),
            tags=kwargs.get("tags"),
            cache_affinity_scores=cache_affinity_scores,
        )

        candidate_scores = self.strategy.select(candidates, scoring_ctx)
        explanation.candidates = candidate_scores
        selected = next(
            (c for c in candidate_scores if c.selected),
            candidate_scores[0] if candidate_scores else None,
        )
        if selected:
            explanation.selected_model = selected.model_name
            explanation.selected_endpoint = selected.endpoint

        explanation.total_latency_ms = (time.monotonic() - start_time) * 1000
        return explanation

    def get_adapters(self) -> dict[str, InferenceAdapter]:
        """Get all registered adapters."""
        return dict(self._adapters)

    async def health_check_all(self) -> dict[str, Any]:
        """Run health checks on all endpoints."""
        results = await self.telemetry.collect_once()
        for _ep, health in results.items():
            self.health_manager.update_health(health)
        return {ep: {"healthy": h.healthy, "latency_ms": h.latency_ms} for ep, h in results.items()}

    async def close(self) -> None:
        """Close all adapters and stop polling."""
        await self.telemetry.stop_polling()
        for adapter in self._adapters.values():
            if hasattr(adapter, "close"):
                await adapter.close()

    def _normalize_messages(
        self,
        messages: list[dict[str, str]] | list[ChatMessage] | None,
        prompt: str | None,
    ) -> list[ChatMessage]:
        """Normalize input to list of ChatMessage."""
        if messages:
            result = []
            for msg in messages:
                if isinstance(msg, ChatMessage):
                    result.append(msg)
                else:
                    result.append(ChatMessage(role=msg["role"], content=msg["content"]))
            return result
        elif prompt:
            return [ChatMessage(role="user", content=prompt)]
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")
