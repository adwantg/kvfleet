"""Affinity-aware routing integration — connects KV-cache scoring to the router."""

from __future__ import annotations

from kvfleet.cache.fingerprints import PromptFingerprint
from kvfleet.cache.kv_affinity import KVAffinityScorer
from kvfleet.config.schema import ModelConfig


def compute_affinity_scores(
    fingerprint: PromptFingerprint,
    candidates: list[ModelConfig],
    scorer: KVAffinityScorer,
) -> dict[str, float]:
    """Compute cache affinity scores for all candidate models.

    Args:
        fingerprint: Pre-computed prompt fingerprint.
        candidates: Model candidates.
        scorer: KV affinity scorer instance.

    Returns:
        Dict of model_name → max affinity score across endpoints.
    """
    scores: dict[str, float] = {}
    for model in candidates:
        endpoints = model.all_endpoints()
        if not endpoints:
            scores[model.name] = 0.0
            continue
        endpoint_scores = scorer.score_affinity(fingerprint, model.name, endpoints)
        scores[model.name] = max(endpoint_scores.values()) if endpoint_scores else 0.0
    return scores


def select_best_endpoint(
    fingerprint: PromptFingerprint,
    model: ModelConfig,
    scorer: KVAffinityScorer,
    health_scores: dict[str, float] | None = None,
) -> tuple[str, float]:
    """Select the best endpoint for a model using cache affinity.

    Args:
        fingerprint: Prompt fingerprint.
        model: Selected model.
        scorer: KV affinity scorer.
        health_scores: Optional endpoint health scores.

    Returns:
        Tuple of (endpoint, affinity_score).
    """
    endpoints = model.all_endpoints()
    if not endpoints:
        return model.endpoint, 0.0
    return scorer.best_endpoint(fingerprint, model.name, endpoints, health_scores)
