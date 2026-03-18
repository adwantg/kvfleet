"""KV-cache affinity scoring and consistent hashing for replica selection."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from typing import Any

from kvfleet.cache.fingerprints import PromptFingerprint

logger = logging.getLogger(__name__)


class ConsistentHashRing:
    """Consistent hash ring for KV-cache-aware replica routing.

    Maps session/prefix fingerprints to specific replicas using
    consistent hashing, ensuring that similar prompts route to
    the same worker (where KV-cache is likely warm).
    """

    def __init__(self, virtual_nodes: int = 150) -> None:
        self.virtual_nodes = virtual_nodes
        self._ring: dict[int, str] = {}
        self._sorted_keys: list[int] = []
        self._nodes: set[str] = set()

    def add_node(self, node: str) -> None:
        """Add a node (endpoint) to the ring."""
        self._nodes.add(node)
        for i in range(self.virtual_nodes):
            key = self._hash(f"{node}:{i}")
            self._ring[key] = node
        self._sorted_keys = sorted(self._ring.keys())

    def remove_node(self, node: str) -> None:
        """Remove a node from the ring."""
        self._nodes.discard(node)
        for i in range(self.virtual_nodes):
            key = self._hash(f"{node}:{i}")
            self._ring.pop(key, None)
        self._sorted_keys = sorted(self._ring.keys())

    def get_node(self, key: str) -> str | None:
        """Get the node responsible for a given key."""
        if not self._ring:
            return None
        hash_val = self._hash(key)
        # Find the first node clockwise from the hash position
        for ring_key in self._sorted_keys:
            if ring_key >= hash_val:
                return self._ring[ring_key]
        return self._ring[self._sorted_keys[0]]

    @staticmethod
    def _hash(key: str) -> int:
        """Hash a key to a position on the ring."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


class SessionAffinityStore:
    """Tracks session → endpoint mappings for sticky routing.

    Thread-safe via threading.Lock for concurrent access.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[str, float]] = {}  # session_key → (endpoint, timestamp)
        self._lock = threading.Lock()

    def get(self, session_key: str) -> str | None:
        """Get the endpoint for a session, if still valid."""
        with self._lock:
            if session_key not in self._store:
                return None
            endpoint, ts = self._store[session_key]
            if time.time() - ts > self.ttl_seconds:
                del self._store[session_key]
                return None
            return endpoint

    def set(self, session_key: str, endpoint: str) -> None:
        """Map a session to an endpoint."""
        with self._lock:
            self._store[session_key] = (endpoint, time.time())

    def clear_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        with self._lock:
            now = time.time()
            expired = [k for k, (_, ts) in self._store.items() if now - ts > self.ttl_seconds]
            for k in expired:
                del self._store[k]
            return len(expired)


class KVAffinityScorer:
    """Scores models/endpoints by KV-cache affinity for a given request.

    This is the flagship feature of kvfleet. It uses:
    1. Session affinity — route same conversation to same worker
    2. Prefix affinity — route similar system prompts to same worker
    3. Consistent hashing — deterministic replica selection by fingerprint
    4. Cache locality scoring — combine affinity signals into a score
    """

    def __init__(
        self,
        virtual_nodes: int = 150,
        session_ttl: int = 3600,
        min_affinity_score: float = 0.3,
    ) -> None:
        self.hash_ring = ConsistentHashRing(virtual_nodes=virtual_nodes)
        self.session_store = SessionAffinityStore(ttl_seconds=session_ttl)
        self.min_affinity_score = min_affinity_score
        self._prefix_cache: dict[str, str] = {}  # prefix_hash → endpoint

    def register_endpoints(self, model_name: str, endpoints: list[str]) -> None:
        """Register endpoints for a model on the hash ring."""
        for endpoint in endpoints:
            self.hash_ring.add_node(f"{model_name}:{endpoint}")

    def score_affinity(
        self,
        fingerprint: PromptFingerprint,
        model_name: str,
        available_endpoints: list[str],
    ) -> dict[str, float]:
        """Score each endpoint's cache affinity for this request.

        Args:
            fingerprint: Prompt fingerprint.
            model_name: Model name.
            available_endpoints: List of available endpoints.

        Returns:
            Dict of endpoint → affinity score (0-1).
        """
        scores: dict[str, float] = {}

        for endpoint in available_endpoints:
            score = 0.0

            # 1. Session affinity — highest signal
            session_endpoint = self.session_store.get(fingerprint.session_key)
            if session_endpoint == endpoint:
                score += 0.5

            # 2. Prefix affinity
            cached_endpoint = self._prefix_cache.get(fingerprint.prefix_key)
            if cached_endpoint == endpoint:
                score += 0.3

            # 3. Consistent hash match
            hash_target = self.hash_ring.get_node(fingerprint.prefix_key)
            if hash_target and hash_target == f"{model_name}:{endpoint}":
                score += 0.2

            scores[endpoint] = min(score, 1.0)

        return scores

    def best_endpoint(
        self,
        fingerprint: PromptFingerprint,
        model_name: str,
        available_endpoints: list[str],
        health_scores: dict[str, float] | None = None,
    ) -> tuple[str, float]:
        """Select the best endpoint considering cache affinity and health.

        Args:
            fingerprint: Prompt fingerprint.
            model_name: Model name.
            available_endpoints: Available endpoints.
            health_scores: Optional endpoint → health score (0-1, higher = healthier).

        Returns:
            Tuple of (best_endpoint, affinity_score).
        """
        affinity = self.score_affinity(fingerprint, model_name, available_endpoints)

        # Combine affinity with health
        combined: dict[str, float] = {}
        for ep in available_endpoints:
            aff = affinity.get(ep, 0.0)
            health = health_scores.get(ep, 1.0) if health_scores else 1.0
            # Weight: 60% affinity, 40% health
            combined[ep] = aff * 0.6 + health * 0.4

        if not combined:
            return available_endpoints[0] if available_endpoints else "", 0.0

        best_ep = max(combined, key=combined.get)  # type: ignore[arg-type]
        return best_ep, affinity.get(best_ep, 0.0)

    def record_routing(self, fingerprint: PromptFingerprint, endpoint: str) -> None:
        """Record a routing decision for future affinity."""
        self.session_store.set(fingerprint.session_key, endpoint)
        self._prefix_cache[fingerprint.prefix_key] = endpoint

    def get_cache_stats(self) -> dict[str, Any]:
        """Return cache affinity statistics."""
        return {
            "active_sessions": len(self.session_store._store),
            "prefix_cache_entries": len(self._prefix_cache),
            "hash_ring_nodes": len(self.hash_ring._nodes),
        }
