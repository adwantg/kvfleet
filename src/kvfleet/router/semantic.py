"""Semantic and domain-based routing strategies.

Uses lightweight text classification to route requests to the best
model based on prompt content (domain, complexity, language, etc.).
Optionally integrates with sentence-transformers for embedding-based routing.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from kvfleet.config.schema import ModelConfig
from kvfleet.router.explain import CandidateScore
from kvfleet.router.scoring import ScoringContext
from kvfleet.router.strategies import RoutingStrategy

logger = logging.getLogger(__name__)

# ───────────────────── Built-in Domain Classifier ─────────────────────

# Keyword-based domain detection (works without any ML dependencies)
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "code",
        "function",
        "class",
        "def ",
        "import ",
        "variable",
        "debug",
        "error",
        "compile",
        "syntax",
        "algorithm",
        "api",
        "database",
        "sql",
        "python",
        "javascript",
        "rust",
        "golang",
        "java",
        "typescript",
        "git",
        "docker",
        "kubernetes",
        "ci/cd",
        "deploy",
        "refactor",
        "unittest",
        "pytest",
        "npm",
        "pip",
        "cargo",
        "webpack",
    ],
    "math": [
        "equation",
        "integral",
        "derivative",
        "matrix",
        "vector",
        "theorem",
        "proof",
        "calculus",
        "algebra",
        "geometry",
        "statistics",
        "probability",
        "eigenvalue",
        "polynomial",
        "logarithm",
        "trigonometry",
        "factorial",
    ],
    "creative": [
        "poem",
        "story",
        "novel",
        "write a",
        "creative",
        "fiction",
        "haiku",
        "song",
        "lyrics",
        "dialogue",
        "character",
        "plot",
        "narrative",
        "metaphor",
        "imagery",
        "prose",
        "screenplay",
    ],
    "medical": [
        "diagnosis",
        "symptom",
        "treatment",
        "disease",
        "patient",
        "clinical",
        "medical",
        "health",
        "doctor",
        "prescription",
        "surgery",
        "therapy",
        "dosage",
        "pharmaceutical",
        "pathology",
        "radiology",
    ],
    "legal": [
        "legal",
        "law",
        "court",
        "contract",
        "statute",
        "regulation",
        "compliance",
        "liability",
        "attorney",
        "jurisdiction",
        "precedent",
        "litigation",
        "arbitration",
        "intellectual property",
        "patent",
    ],
    "scientific": [
        "research",
        "experiment",
        "hypothesis",
        "data analysis",
        "methodology",
        "peer review",
        "citation",
        "journal",
        "abstract",
        "conclusion",
        "laboratory",
        "specimen",
        "observation",
        "scientific method",
    ],
    "translation": [
        "translate",
        "translation",
        "en español",
        "en français",
        "auf deutsch",
        "in japanese",
        "in chinese",
        "in korean",
        "localize",
        "multilingual",
    ],
    "summarization": [
        "summarize",
        "summary",
        "tldr",
        "key points",
        "bullet points",
        "condense",
        "brief",
        "overview",
        "highlights",
        "recap",
    ],
    "general": [],  # Fallback
}


def classify_domain(text: str) -> tuple[str, float]:
    """Classify text into a domain using keyword matching.

    Args:
        text: Input text to classify.

    Returns:
        Tuple of (domain_name, confidence_score).
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        if not keywords:
            continue
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[domain] = score

    if not scores:
        return "general", 0.5

    best_domain = max(scores, key=scores.get)  # type: ignore[arg-type]
    # Confidence based on keyword density
    max_score = scores[best_domain]
    confidence = min(1.0, 0.5 + (max_score * 0.1))
    return best_domain, confidence


def estimate_complexity(text: str) -> float:
    """Estimate prompt complexity (0-1 scale).

    Uses heuristics: length, question depth, technical terms, etc.
    """
    score = 0.0

    # Length factor
    word_count = len(text.split())
    if word_count > 200:
        score += 0.3
    elif word_count > 50:
        score += 0.2
    elif word_count > 20:
        score += 0.1

    # Multi-part questions
    question_marks = text.count("?")
    if question_marks > 2:
        score += 0.2
    elif question_marks > 0:
        score += 0.1

    # Technical indicators
    technical_patterns = [
        r"\b(implement|optimize|architecture|distributed|concurrent)\b",
        r"\b(trade-?offs?|compare|pros?\s+and\s+cons?)\b",
        r"\b(step\s*by\s*step|in\s*detail|thoroughly|comprehensive)\b",
    ]
    for pattern in technical_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += 0.1

    return min(1.0, score)


# ───────────────────── Semantic Strategy ─────────────────────


class SemanticStrategy(RoutingStrategy):
    """Route based on semantic similarity between prompt and model descriptions.

    Uses lightweight keyword classification by default.
    When sentence-transformers is available, uses embedding cosine similarity.
    """

    name = "semantic"

    def __init__(
        self,
        model_descriptions: dict[str, str] | None = None,
        use_embeddings: bool = False,
    ) -> None:
        self.model_descriptions = model_descriptions or {}
        self.use_embeddings = use_embeddings
        self._embedder: Any = None
        self._description_embeddings: dict[str, Any] = {}

        if use_embeddings:
            self._init_embedder()

    def _init_embedder(self) -> None:
        """Try to initialize sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            # Pre-compute description embeddings
            for model_name, desc in self.model_descriptions.items():
                self._description_embeddings[model_name] = self._embedder.encode(desc)
            logger.info("Semantic routing initialized with sentence-transformers")
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to keyword matching")
            self.use_embeddings = False

    def select(
        self,
        candidates: list[ModelConfig],
        context: ScoringContext | None = None,
        **kwargs: Any,
    ) -> list[CandidateScore]:
        ctx = context or ScoringContext()
        prompt_text = ctx.metadata.get("prompt_text", "") if ctx.metadata else ""

        if self.use_embeddings and self._embedder and prompt_text:
            return self._select_by_embedding(candidates, prompt_text)
        else:
            return self._select_by_keywords(candidates, ctx)

    def _select_by_keywords(
        self,
        candidates: list[ModelConfig],
        ctx: ScoringContext,
    ) -> list[CandidateScore]:
        """Route using keyword-based domain classification."""
        prompt_text = ctx.metadata.get("prompt_text", "") if ctx.metadata else ""
        domain, confidence = classify_domain(prompt_text) if prompt_text else ("general", 0.5)
        complexity = estimate_complexity(prompt_text) if prompt_text else 0.5

        scores: list[CandidateScore] = []
        for model in candidates:
            model_domain = model.tags.get("domain", "general")
            model.tags.get("tier", "")

            # Domain match score
            domain_score = 1.0 if model_domain == domain else 0.3

            # Quality match for complexity (complex → higher quality model)
            quality_match = 1.0 - abs(model.quality_score - complexity)

            total = domain_score * 0.6 + quality_match * 0.4
            scores.append(
                CandidateScore(
                    model_name=model.name,
                    endpoint=model.endpoint,
                    total_score=total,
                    quality_score=model.quality_score,
                    signals={
                        "detected_domain": domain,
                        "domain_confidence": confidence,
                        "complexity": complexity,
                        "domain_match": domain_score,
                    },
                )
            )

        scores.sort(key=lambda s: s.total_score, reverse=True)
        if scores:
            scores[0].selected = True
        return scores

    def _select_by_embedding(
        self,
        candidates: list[ModelConfig],
        prompt_text: str,
    ) -> list[CandidateScore]:
        """Route using embedding cosine similarity."""
        import numpy as np

        prompt_emb = self._embedder.encode(prompt_text)
        scores: list[CandidateScore] = []

        for model in candidates:
            desc_emb = self._description_embeddings.get(model.name)
            if desc_emb is not None:
                similarity = float(
                    np.dot(prompt_emb, desc_emb)
                    / (np.linalg.norm(prompt_emb) * np.linalg.norm(desc_emb) + 1e-8)
                )
            else:
                similarity = 0.5

            scores.append(
                CandidateScore(
                    model_name=model.name,
                    endpoint=model.endpoint,
                    total_score=similarity,
                    signals={"embedding_similarity": similarity},
                )
            )

        scores.sort(key=lambda s: s.total_score, reverse=True)
        if scores:
            scores[0].selected = True
        return scores


class DomainStrategy(RoutingStrategy):
    """Route based on detected task domain.

    Maps detected domains to preferred models via a configurable mapping.
    """

    name = "domain"

    def __init__(self, domain_model_map: dict[str, str] | None = None) -> None:
        self.domain_model_map = domain_model_map or {}

    def select(
        self,
        candidates: list[ModelConfig],
        context: ScoringContext | None = None,
        **kwargs: Any,
    ) -> list[CandidateScore]:
        ctx = context or ScoringContext()
        prompt_text = ctx.metadata.get("prompt_text", "") if ctx.metadata else ""
        domain, confidence = classify_domain(prompt_text) if prompt_text else ("general", 0.5)

        # Check if we have a model mapping for this domain
        preferred_model = self.domain_model_map.get(domain)
        candidate_names = {m.name for m in candidates}

        scores: list[CandidateScore] = []
        for model in candidates:
            if (
                preferred_model
                and model.name == preferred_model
                and preferred_model in candidate_names
            ):
                score = 1.0
            elif model.tags.get("domain") == domain:
                score = 0.8
            else:
                score = model.quality_score * 0.5

            scores.append(
                CandidateScore(
                    model_name=model.name,
                    endpoint=model.endpoint,
                    total_score=score,
                    signals={"detected_domain": domain, "confidence": confidence},
                )
            )

        scores.sort(key=lambda s: s.total_score, reverse=True)
        if scores:
            scores[0].selected = True
        return scores
