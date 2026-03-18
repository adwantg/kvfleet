"""Configuration schema for kvfleet using Pydantic models."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class RouteStrategy(str, enum.Enum):
    """Available routing strategies."""

    STATIC = "static"
    WEIGHTED = "weighted"
    RULES = "rules"
    COST_FIRST = "cost_first"
    LATENCY_FIRST = "latency_first"
    QUALITY_FIRST = "quality_first"
    CHEAP_CASCADE = "cheap_cascade"
    HYBRID_SCORE = "hybrid_score"
    SEMANTIC = "semantic"
    DOMAIN = "domain"
    LEARNED = "learned"
    BANDIT = "bandit"
    THOMPSON = "thompson"
    EXP3 = "exp3"


class ProviderType(str, enum.Enum):
    """Supported inference providers."""

    VLLM = "vllm"
    TRITON = "triton"
    OLLAMA = "ollama"
    TGI = "tgi"
    OPENAI_COMPAT = "openai_compat"
    CUSTOM_HTTP = "custom_http"
    BEDROCK = "bedrock"


class ModelCapabilities(BaseModel):
    """Capabilities of a model."""

    supports_tools: bool = False
    supports_json_mode: bool = False
    supports_streaming: bool = True
    supports_vision: bool = False
    max_context_window: int = 4096
    supported_languages: list[str] = Field(default_factory=lambda: ["en"])


class ModelConfig(BaseModel):
    """Configuration for a single model in the fleet."""

    name: str = Field(..., description="Unique model identifier")
    endpoint: str = Field(..., description="Inference endpoint URL")
    provider: ProviderType = Field(default=ProviderType.OPENAI_COMPAT)
    model_id: str = Field(default="", description="Model ID sent to the provider (e.g., 'meta-llama/Llama-3-8B')")
    replicas: list[str] = Field(default_factory=list, description="Additional replica endpoints")
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)

    # Priors for scoring
    cost_per_1k_input_tokens: float = Field(default=0.0, description="Estimated input cost per 1K tokens")
    cost_per_1k_output_tokens: float = Field(default=0.0, description="Estimated output cost per 1K tokens")
    latency_p50_ms: float = Field(default=500.0, description="Expected p50 latency in ms")
    latency_p95_ms: float = Field(default=2000.0, description="Expected p95 latency in ms")
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Prior quality score (0–1)")

    # Constraints
    allowed_data_classes: list[str] = Field(
        default_factory=lambda: ["public", "internal", "confidential"],
        description="Data classification levels this model may process",
    )
    tags: dict[str, str] = Field(default_factory=dict, description="Arbitrary tags for rule matching")
    weight: float = Field(default=1.0, ge=0.0, description="Weight for weighted routing")
    enabled: bool = Field(default=True, description="Whether this model is available for routing")

    # Timeout
    timeout_seconds: float = Field(default=60.0, description="Request timeout")
    max_retries: int = Field(default=2, description="Retries on transient failure")

    def get_model_id(self) -> str:
        """Return the model ID to use when calling the provider."""
        return self.model_id or self.name

    def all_endpoints(self) -> list[str]:
        """Return primary + replica endpoints."""
        return [self.endpoint, *self.replicas]


class FallbackConfig(BaseModel):
    """Fallback chain configuration."""

    enabled: bool = True
    max_attempts: int = Field(default=3, ge=1)
    promote_on_timeout: bool = Field(default=True, description="Escalate to higher-quality model on timeout")
    fallback_order: list[str] = Field(default_factory=list, description="Ordered list of model names to try")
    timeout_ms: float = Field(default=10000.0, description="Timeout before triggering fallback")


class ScoringWeights(BaseModel):
    """Weights for multi-objective scoring."""

    cost: float = Field(default=0.3, ge=0.0, le=1.0)
    latency: float = Field(default=0.3, ge=0.0, le=1.0)
    quality: float = Field(default=0.3, ge=0.0, le=1.0)
    cache_affinity: float = Field(default=0.1, ge=0.0, le=1.0)
    hardware_load: float = Field(default=0.0, ge=0.0, le=1.0)
    compliance: float = Field(default=0.0, ge=0.0, le=1.0)


class CacheAffinityConfig(BaseModel):
    """KV-cache affinity routing config."""

    enabled: bool = True
    session_ttl_seconds: int = Field(default=3600, description="TTL for session affinity mappings")
    prefix_hash_tokens: int = Field(default=128, description="Number of prefix tokens to hash for affinity")
    min_affinity_score: float = Field(default=0.3, ge=0.0, le=1.0, description="Min score to prefer affinity match")
    consistent_hash_replicas: int = Field(default=150, description="Virtual nodes for consistent hashing ring")


class ShadowConfig(BaseModel):
    """Shadow traffic configuration."""

    enabled: bool = False
    shadow_models: list[str] = Field(default_factory=list, description="Models to shadow-route to")
    sample_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Fraction of traffic to shadow")
    log_outputs: bool = True
    compare_outputs: bool = False


class TelemetryConfig(BaseModel):
    """Observability configuration."""

    prometheus_enabled: bool = True
    prometheus_port: int = Field(default=9090)
    health_check_interval_seconds: int = Field(default=30)
    structured_logs: bool = True
    log_route_decisions: bool = True


class PolicyRule(BaseModel):
    """A policy rule for routing enforcement."""

    name: str
    description: str = ""
    condition: str = Field(..., description="Condition expression (e.g., 'data_class == confidential')")
    action: str = Field(..., description="Action: 'require_private', 'block', 'allow', 'require_model'")
    target_models: list[str] = Field(default_factory=list, description="Models this rule targets")
    priority: int = Field(default=100, description="Lower = higher priority")


class PolicyConfig(BaseModel):
    """Policy engine configuration."""

    enabled: bool = False
    pii_detection: bool = False
    rules: list[PolicyRule] = Field(default_factory=list)
    default_data_class: str = "internal"


class BudgetConfig(BaseModel):
    """Budget and quota configuration."""

    enabled: bool = False
    monthly_budget_usd: float = Field(default=1000.0, ge=0.0)
    per_request_max_usd: float = Field(default=1.0, ge=0.0)
    alert_threshold_pct: float = Field(default=80.0, ge=0.0, le=100.0)
    throttle_on_exceed: bool = True


class TenantConfig(BaseModel):
    """Tenant-level routing configuration."""

    name: str
    preferred_models: list[str] = Field(default_factory=list)
    blocked_models: list[str] = Field(default_factory=list)
    max_cost_per_request: float | None = None
    max_latency_ms: float | None = None
    allowed_data_classes: list[str] = Field(default_factory=lambda: ["public", "internal"])
    budget: BudgetConfig = Field(default_factory=BudgetConfig)


class SLOConfig(BaseModel):
    """SLO-aware routing configuration."""

    enabled: bool = False
    target_p95_latency_ms: float = Field(default=5000.0)
    max_cost_per_request: float = Field(default=0.10)
    min_quality_score: float = Field(default=0.5)
    load_shed_threshold: float = Field(default=0.9, description="Queue saturation threshold for load shedding")


class GatewayConfig(BaseModel):
    """OpenAI-compatible gateway configuration."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class RouteRuleConfig(BaseModel):
    """A route rule for rules-based routing."""

    name: str
    condition: dict[str, Any] = Field(
        default_factory=dict,
        description="Match conditions (e.g., {'tags.domain': 'coding', 'min_quality': 0.8})",
    )
    target_model: str = Field(..., description="Model to route to when conditions match")
    priority: int = Field(default=100, description="Lower = higher priority")


class FleetConfig(BaseModel):
    """Top-level fleet configuration."""

    fleet_name: str = Field(default="default", description="Fleet identifier")
    models: list[ModelConfig] = Field(default_factory=list)
    strategy: RouteStrategy = Field(default=RouteStrategy.HYBRID_SCORE)
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    cache_affinity: CacheAffinityConfig = Field(default_factory=CacheAffinityConfig)
    shadow: ShadowConfig = Field(default_factory=ShadowConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    slo: SLOConfig = Field(default_factory=SLOConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tenants: dict[str, TenantConfig] = Field(default_factory=dict)
    route_rules: list[RouteRuleConfig] = Field(default_factory=list)

    # Static route (for strategy=static)
    default_model: str = Field(default="", description="Default model for static routing")
