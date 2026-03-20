"""OpenAI-compatible gateway server.

Exposes kvfleet routing as an OpenAI-compatible /v1/chat/completions endpoint,
acting as a drop-in replacement for any OpenAI client.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Check if starlette/uvicorn are available (optional dependency)
try:
    from starlette.applications import Starlette
    from starlette.middleware.cors import CORSMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    HAS_GATEWAY_DEPS = True
except ImportError:
    HAS_GATEWAY_DEPS = False


def create_gateway_app(
    router: Any, api_key: str = "", cors_origins: list[str] | None = None
) -> Any:
    """Create a Starlette app that exposes kvfleet as an OpenAI-compatible API.

    Args:
        router: kvfleet Router instance.
        api_key: Optional API key for authentication.
        cors_origins: CORS allowed origins.

    Returns:
        Starlette app instance.

    Raises:
        ImportError: If starlette is not installed.
    """
    if not HAS_GATEWAY_DEPS:
        raise ImportError(
            "Gateway requires 'starlette' and 'uvicorn'. Install with: pip install kvfleet[gateway]"
        )

    from kvfleet.adapters.base import ChatMessage, ChatRequest
    from kvfleet.router.engine import Router

    typed_router: Router = router

    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        """Handle /v1/chat/completions requests."""
        # Auth check
        if api_key:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {api_key}":
                return JSONResponse({"error": "Invalid API key"}, status_code=401)

        body = await request.json()
        messages_raw = body.get("messages", [])
        messages = [
            ChatMessage(
                role=m["role"],
                content=m.get("content", ""),
                name=m.get("name"),
                tool_call_id=m.get("tool_call_id"),
                tool_calls=m.get("tool_calls"),
            )
            for m in messages_raw
        ]

        # E-1: Extract configured passthrough headers
        passthrough: dict[str, str] = {}
        gw_config = typed_router.config.gateway
        for header_name in gw_config.passthrough_headers:
            value = request.headers.get(header_name.lower())
            if value:
                passthrough[header_name] = value

        metadata: dict[str, Any] = {}
        if passthrough:
            metadata["_passthrough_headers"] = passthrough

        # E-9: Timeout override via header
        if gw_config.timeout_header:
            timeout_val = request.headers.get(gw_config.timeout_header.lower())
            if timeout_val:
                try:
                    metadata["_timeout_ms"] = int(timeout_val)
                except ValueError:
                    logger.warning("Invalid timeout header value: %s", timeout_val)

        # E-10: Request ID propagation
        inbound_request_id = request.headers.get("x-request-id")

        chat_request = ChatRequest(
            messages=messages,
            model=body.get("model", ""),
            temperature=body.get("temperature", 1.0),
            max_tokens=body.get("max_tokens"),
            stream=body.get("stream", False),
            top_p=body.get("top_p", 1.0),
            stop=body.get("stop"),
            tools=body.get("tools"),
            response_format=body.get("response_format"),
            metadata=metadata,
        )

        request_id = inbound_request_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
        tags = {"model_hint": body.get("model", "")} if body.get("model") else {}

        # E-3: Strategy override via header
        strategy_override = None
        if gw_config.strategy_header:
            strategy_val = request.headers.get(gw_config.strategy_header.lower())
            if strategy_val:
                try:
                    from kvfleet.config.schema import RouteStrategy

                    strategy_override = RouteStrategy(strategy_val)
                except ValueError:
                    logger.warning("Invalid strategy override: %s", strategy_val)

        # E-4: Model allowlist via header
        model_allowlist = None
        if gw_config.model_allowlist_header:
            allowlist_val = request.headers.get(gw_config.model_allowlist_header.lower())
            if allowlist_val:
                model_allowlist = [m.strip() for m in allowlist_val.split(",") if m.strip()]

        # E-5: Tenant ID from header
        tenant_id = None
        if gw_config.tenant_header:
            tenant_id = request.headers.get(gw_config.tenant_header.lower())

        try:
            response, explanation = await typed_router.route(
                request=chat_request,
                tags=tags,
                request_id=request_id,
                tenant_id=tenant_id,
                strategy_override=strategy_override,
                model_allowlist=model_allowlist,
            )
        except RuntimeError as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "routing_error"}}, status_code=503
            )

        # Build OpenAI-format response
        result = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.model or explanation.selected_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        k: v
                        for k, v in {
                            "role": "assistant",
                            "content": response.content,
                            "tool_calls": response.tool_calls,
                        }.items()
                        if v is not None
                    },
                    "finish_reason": response.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "kvfleet_metadata": {
                "selected_model": explanation.selected_model,
                "strategy": explanation.strategy,
                "strategy_overridden": explanation.strategy_overridden,
                "cache_affinity_used": explanation.cache_affinity_used,
                "total_latency_ms": explanation.total_latency_ms,
            },
        }

        # E-10: Include X-Request-ID in response headers
        return JSONResponse(result, headers={"X-Request-ID": request_id})

    async def models_list(request: Request) -> JSONResponse:
        """Handle /v1/models requests."""
        models = typed_router.registry.list_models(enabled_only=True)
        data = [
            {
                "id": m.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "kvfleet",
            }
            for m in models
        ]
        return JSONResponse({"object": "list", "data": data})

    async def health(request: Request) -> JSONResponse:
        """Handle /health requests."""
        results = await typed_router.health_check_all()
        all_healthy = all(r.get("healthy", False) for r in results.values()) if results else True
        return JSONResponse(
            {"status": "ok" if all_healthy else "degraded", "endpoints": results},
            status_code=200 if all_healthy else 503,
        )

    async def explain(request: Request) -> JSONResponse:
        """Handle /v1/route/explain — simulate without executing."""
        body = await request.json()
        messages_raw = body.get("messages", [])
        messages = [{"role": m["role"], "content": m.get("content", "")} for m in messages_raw]

        explanation = await typed_router.simulate(
            messages=messages,
            data_class=body.get("data_class"),
            tags=body.get("tags"),
        )
        return JSONResponse(explanation.to_dict())

    routes = [
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/models", models_list, methods=["GET"]),
        Route("/v1/route/explain", explain, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
    ]

    app = Starlette(routes=routes)
    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
    )

    return app


def run_gateway(router: Any, host: str = "0.0.0.0", port: int = 8000, api_key: str = "") -> None:
    """Run the gateway server.

    Args:
        router: kvfleet Router instance.
        host: Bind host.
        port: Bind port.
        api_key: Optional API key.
    """
    import uvicorn

    app = create_gateway_app(router, api_key=api_key)
    logger.info("Starting kvfleet gateway on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
