# Use Case: Intelligent Routing for In-House Hosted Models

> **Scenario**: You have multiple self-hosted LLMs behind a single internal API (`/chat/completions` in OpenAI format) and want kvfleet to intelligently route each request to the best model based on the prompt content.

---

## Architecture

```
┌─────────────┐      ┌───────────────────┐      ┌──────────────────────────┐
│             │      │                   │      │   Internal Inference API  │
│  Your App   │─────▶│  kvfleet Router   │─────▶│   /v1/chat/completions   │
│  (any lang) │      │                   │      │   /v1/models             │
│             │      │  Picks the best   │      │                          │
└─────────────┘      │  model per prompt │      │  ┌──────────────────┐    │
                     └───────────────────┘      │  │ llama-3-8b       │    │
                                                │  │ llama-3-70b      │    │
                     Strategies:                │  │ deepseek-coder   │    │
                     • semantic (domain-aware)  │  │ mistral-7b       │    │
                     • bandit (auto-learning)   │  └──────────────────┘    │
                     • hybrid_score (weighted)  └──────────────────────────┘
```

---

## Option A: Static Config (fleet.yaml)

### 1. Define your fleet

```yaml
# fleet.yaml
fleet_name: internal-fleet
strategy: semantic   # auto-detect domain from prompt

models:
  - name: llama-3-8b
    endpoint: http://your-internal-api:8080
    provider: openai_compat
    model_id: llama-3-8b           # model name sent in request body
    quality_score: 0.7
    latency_p50_ms: 200
    tags:
      domain: general
      tier: fast

  - name: llama-3-70b
    endpoint: http://your-internal-api:8080
    provider: openai_compat
    model_id: llama-3-70b
    quality_score: 0.9
    latency_p50_ms: 800
    tags:
      domain: general
      tier: quality

  - name: deepseek-coder
    endpoint: http://your-internal-api:8080
    provider: openai_compat
    model_id: deepseek-coder-v2
    quality_score: 0.85
    latency_p50_ms: 400
    tags:
      domain: coding
      tier: specialized

scoring_weights:
  cost: 0.0          # all free (self-hosted), cost irrelevant
  latency: 0.3
  quality: 0.5
  cache_affinity: 0.2
```

### 2. Route requests

```python
import asyncio
from kvfleet import Router, load_config

async def main():
    config = load_config("fleet.yaml")
    router = Router(config)

    # Coding prompt → deepseek-coder selected
    response, explanation = await router.route(
        prompt="Write a Python async HTTP client with retry logic",
    )
    print(f"Selected: {explanation.selected_model}")  # → deepseek-coder
    print(f"Response: {response.content}")

    # General prompt → llama-3-70b selected (highest quality)
    response2, explanation2 = await router.route(
        prompt="Explain quantum physics to a 5-year-old",
    )
    print(f"Selected: {explanation2.selected_model}")  # → llama-3-70b

    await router.close()

asyncio.run(main())
```

### What happens under the hood

```
Request: "Write a Python async HTTP client"

  1. Fingerprint → hash for cache affinity
  2. Policy check → PII? data class? → all pass
  3. Semantic classifier → domain = "coding" (confidence 0.70)
  4. Score candidates:
       deepseek-coder: 0.85  ← domain=coding tag matches!
       llama-3-70b:    0.62  ← high quality but no coding tag
       llama-3-8b:     0.45  ← fast but lower quality
  5. Selected: deepseek-coder
  6. Forward → POST http://your-internal-api:8080/v1/chat/completions
       Body: { "model": "deepseek-coder-v2", "messages": [...] }
```

---

## Option B: Dynamic Model Discovery

If your API has a `/v1/models` endpoint, fetch models at startup:

```python
import httpx
from kvfleet import Router
from kvfleet.config.schema import FleetConfig, ModelConfig

async def discover_and_route():
    # 1. Fetch from your models API
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://your-internal-api:8080/v1/models")
        models_data = resp.json()["data"]

    # 2. Build config programmatically
    models = []
    for m in models_data:
        model_id = m["id"]
        models.append(ModelConfig(
            name=model_id,
            endpoint="http://your-internal-api:8080",
            provider="openai_compat",
            model_id=model_id,
            tags=infer_tags(model_id),
            quality_score=infer_quality(model_id),
            latency_p50_ms=infer_latency(model_id),
        ))

    config = FleetConfig(
        fleet_name="auto-discovered",
        strategy="semantic",
        models=models,
    )

    # 3. Route
    router = Router(config)
    response, explanation = await router.route(
        prompt="Write a REST API in FastAPI",
    )
    print(f"Routed to: {explanation.selected_model}")
    await router.close()


def infer_tags(model_id: str) -> dict[str, str]:
    """Infer domain tags from model name."""
    lower = model_id.lower()
    if any(k in lower for k in ("code", "deepseek", "starcoder")):
        return {"domain": "coding", "tier": "specialized"}
    if any(k in lower for k in ("70b", "72b", "405b")):
        return {"domain": "general", "tier": "quality"}
    if any(k in lower for k in ("8b", "7b", "3b")):
        return {"domain": "general", "tier": "fast"}
    return {"domain": "general", "tier": "standard"}


def infer_quality(model_id: str) -> float:
    lower = model_id.lower()
    if "70b" in lower or "72b" in lower:
        return 0.9
    if "code" in lower or "deepseek" in lower:
        return 0.85
    if "8b" in lower or "7b" in lower:
        return 0.7
    return 0.75


def infer_latency(model_id: str) -> float:
    lower = model_id.lower()
    if "70b" in lower or "72b" in lower:
        return 800.0
    if "8b" in lower or "7b" in lower:
        return 200.0
    return 400.0
```

---

## Option C: Drop-In Gateway (Zero Code Changes)

Run kvfleet as a proxy — no changes to your application code:

```bash
kvfleet serve fleet.yaml --port 9000
```

Then point your app at kvfleet instead of your internal API:

```python
from openai import OpenAI

# Before (direct)
# client = OpenAI(base_url="http://your-internal-api:8080/v1")

# After (routed through kvfleet)
client = OpenAI(base_url="http://localhost:9000/v1", api_key="optional")

response = client.chat.completions.create(
    model="auto",  # kvfleet picks the best model
    messages=[{"role": "user", "content": "Debug this segfault in Rust"}],
)
# → kvfleet routes to deepseek-coder (detected domain: coding)
```

Works with **any** OpenAI-compatible SDK (Python, Node, Go, curl).

---

## Option D: Auto-Learning with Bandit Strategy

If you don't want to manually tag models, let kvfleet learn:

```yaml
# fleet.yaml
strategy: bandit   # epsilon-greedy: explores 10%, exploits 90%
```

```python
from kvfleet.router.learned import compute_reward

config = load_config("fleet.yaml")
router = Router(config)

# Route
response, explanation = await router.route(prompt="Write unit tests")

# Feed back reward so bandit learns which model is best
reward = compute_reward(
    latency_ms=response.latency_ms,
    quality_score=0.85,   # your quality assessment
    cost=0.0,
    success=True,
)
# Over time, the bandit converges to optimal routing
```

| Bandit Variant | Config Value | Best For |
|---|---|---|
| Epsilon-Greedy | `bandit` | Simple, good default |
| UCB1 | `learned` | Mathematically optimal exploration |
| Thompson Sampling | `thompson` | Fast convergence, Bayesian |
| Exp3 | `exp3` | Non-stationary environments |

---

## Strategy Comparison

| Your Situation | Strategy | How It Works |
|---|---|---|
| You know model strengths | `semantic` | Tags models by domain, classifies prompts, matches |
| Explicit routing rules | `rules` | "if coding → deepseek, if medical → med-llama" |
| Want auto-learning | `bandit` / `thompson` | Explores → converges to best model per domain |
| Want fastest response | `latency_first` | Always picks lowest-latency model |
| Want best quality | `quality_first` | Always picks highest-quality model |
| Balanced trade-off | `hybrid_score` | Weighted scoring across latency + quality + affinity |
| Domain → model map | `domain` | Explicit domain-to-model mapping |

---

## Domain Classification (Built-In)

The semantic strategy auto-detects 8 domains with zero dependencies:

| Domain | Example Prompts |
|--------|----------------|
| `coding` | "Write a Python function", "Debug this error", "Implement an API" |
| `math` | "Solve this integral", "Prove this theorem", "Calculate eigenvalues" |
| `creative` | "Write a poem", "Create a story", "Compose lyrics" |
| `medical` | "Symptoms of diabetes", "Treatment options", "Drug interactions" |
| `legal` | "Draft a contract", "Legal liability", "Patent filing" |
| `scientific` | "Research methodology", "Peer review", "Hypothesis testing" |
| `translation` | "Translate to Spanish", "Localize this text" |
| `summarization` | "Summarize this article", "Key points", "TL;DR" |

Tag your models with `domain: coding` (or `math`, `medical`, etc.) and the classifier handles the rest.

---

## Full Working Example

```python
"""Complete internal fleet routing example."""

import asyncio
from kvfleet import Router, load_config

async def main():
    config = load_config("fleet.yaml")
    router = Router(config)

    prompts = [
        "Write a binary search in Rust",           # → coding model
        "Explain the theory of relativity",          # → quality model
        "Summarize this quarterly report",           # → fast model
        "Translate this paragraph to French",        # → general model
        "What are the side effects of ibuprofen?",   # → medical model (if tagged)
    ]

    for prompt in prompts:
        response, explanation = await router.route(prompt=prompt)
        print(f"Prompt:   {prompt[:50]}...")
        print(f"Model:    {explanation.selected_model}")
        print(f"Latency:  {response.latency_ms:.0f}ms")
        print(f"Reason:   {explanation.summary()}")
        print()

    await router.close()

asyncio.run(main())
```

**Output:**
```
Prompt:   Write a binary search in Rust...
Model:    deepseek-coder
Latency:  350ms
Reason:   Domain=coding matched model tag

Prompt:   Explain the theory of relativity...
Model:    llama-3-70b
Latency:  780ms
Reason:   Highest quality for general domain

Prompt:   Summarize this quarterly report...
Model:    llama-3-8b
Latency:  180ms
Reason:   Domain=summarization, fast tier preferred

...
```
