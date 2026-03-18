# kvfleet — Full Implementation Checklist

## Phase 1 — MVP (v0.1)

### Sprint 1: Core Foundation
- [/] Project scaffold ([pyproject.toml](file:///Users/goutamadwant/Documents/OpenSource/PythonProjects/kvfleet/skills/pyproject.toml), `__init__.py`, CLI)
- [ ] Config schema + loader (`config/schema.py`, `config/loader.py`)
- [ ] Model registry (`registry/models.py`)
- [ ] Adapter base class (`adapters/base.py`)
- [ ] vLLM adapter (`adapters/vllm.py`)
- [ ] OpenAI-compat adapter (`adapters/openai_compat.py`)
- [ ] CLI scaffold (`cli.py`)

### Sprint 2: Routing Engine
- [ ] Router engine (`router/engine.py`)
- [ ] Routing strategies (`router/strategies.py`)
- [ ] Cost/latency scoring (`router/scoring.py`)
- [ ] Fallback chains (`router/fallback.py`)
- [ ] Routing explainer (`router/explain.py`)

### Sprint 3: KV-Cache Affinity
- [ ] Prompt fingerprinting (`cache/fingerprints.py`)
- [ ] KV-cache affinity scoring (`cache/kv_affinity.py`)
- [ ] Affinity-aware routing (`router/affinity.py`)
- [ ] Health check manager (`telemetry/health.py`)
- [ ] Telemetry collector (`telemetry/collector.py`)

### Sprint 4: Shadow & Observability
- [ ] Shadow traffic mode (`eval/shadow.py`)
- [ ] Prometheus metrics (`telemetry/metrics.py`)
- [ ] Sync SDK (`sdk/sync_client.py`)
- [ ] Async SDK (`sdk/async_client.py`)

### Sprint 5: Testing & Polish
- [ ] Unit tests
- [ ] Integration tests
- [ ] Production README with examples

## Phase 2 — Differentiators (v1.0)
- [ ] Multi-objective scoring engine
- [ ] Task/domain classification
- [ ] Policy-aware routing (`policy/engine.py`, `policy/pii.py`, `policy/residency.py`)
- [ ] Prompt fingerprinting v2
- [ ] Semantic similarity router
- [ ] Request budgeting & quotas
- [ ] Triton adapter
- [ ] TGI adapter
- [ ] Ollama adapter
- [ ] Custom HTTP adapter
- [ ] GPU state reader (`telemetry/gpu.py`)

## Phase 3 — Enterprise (v2.0)
- [ ] Canary rollout manager
- [ ] Model comparison suite (`eval/compare.py`)
- [ ] Offline replay engine (`eval/replay.py`)
- [ ] Route calibration (`eval/calibrate.py`)
- [ ] Safety/compliance pipeline
- [ ] SLO-aware routing
- [ ] Tenant-aware routing (`policy/tenant.py`)
- [ ] OpenAI-compatible gateway (`gateway/`)

## Phase 4 — Advanced Moat (v3.0)
- [ ] Learned intelligent router
- [ ] Bandit-based routing
- [ ] Generative semantic cache
- [ ] Route verification hooks
- [ ] Auto-escalation chains

## Documentation
- [ ] Comprehensive README with examples for every feature
- [ ] Document constraints and limitations
