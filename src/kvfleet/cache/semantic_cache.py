"""Semantic dedup cache — bypass heavy models for near-duplicate prompts."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from kvfleet.cache.fingerprints import PromptFingerprint

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """A cached response from a previous request."""

    content: str
    model: str
    fingerprint: PromptFingerprint
    timestamp: float = field(default_factory=time.time)
    usage_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticCache:
    """Cache for near-duplicate prompts.

    Thread-safe via threading.Lock for concurrent access.
    Supports hash-based exact/prefix matching, TTL expiry,
    and LRU eviction.
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, CachedResponse] = {}
        self._lock = threading.Lock()

    def get(self, fingerprint: PromptFingerprint) -> CachedResponse | None:
        """Look up a cached response by fingerprint.

        Checks full hash match first, then prefix hash.
        """
        with self._lock:
            # Exact match
            entry = self._cache.get(fingerprint.full_hash)
            if entry and not self._is_expired(entry):
                entry.usage_count += 1
                logger.debug("Semantic cache HIT (exact) for %s", fingerprint.full_hash[:8])
                return entry

            # Prefix match (same system prompt + context prefix)
            entry = self._cache.get(fingerprint.prefix_key)
            if entry and not self._is_expired(entry):
                entry.usage_count += 1
                logger.debug("Semantic cache HIT (prefix) for %s", fingerprint.prefix_key[:8])
                return entry

            return None

    def put(self, fingerprint: PromptFingerprint, content: str, model: str) -> None:
        """Store a response in the cache."""
        with self._lock:
            if len(self._cache) >= self.max_size:
                self._evict()

            entry = CachedResponse(
                content=content,
                model=model,
                fingerprint=fingerprint,
            )
            self._cache[fingerprint.full_hash] = entry
            # Also index by prefix key for approximate matching
            self._cache[fingerprint.prefix_key] = entry

    def invalidate(self, fingerprint: PromptFingerprint) -> None:
        """Remove a cached entry."""
        with self._lock:
            self._cache.pop(fingerprint.full_hash, None)
            self._cache.pop(fingerprint.prefix_key, None)

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return len(self._cache)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            now = time.time()
            active = sum(1 for e in self._cache.values() if not self._is_expired(e))
            return {
                "total_entries": len(self._cache),
                "active_entries": active,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
            }

    def _is_expired(self, entry: CachedResponse) -> bool:
        return time.time() - entry.timestamp > self.ttl_seconds

    def _evict(self) -> None:
        """Evict oldest/least-used entries to make room."""
        # Remove expired first
        expired = [k for k, v in self._cache.items() if self._is_expired(v)]
        for k in expired:
            del self._cache[k]

        # If still full, remove least recently used
        if len(self._cache) >= self.max_size:
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].timestamp)
            to_remove = len(self._cache) - self.max_size + (self.max_size // 10)  # Remove 10% extra
            for k, _ in sorted_entries[:to_remove]:
                del self._cache[k]
