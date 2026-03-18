"""Prompt fingerprinting for cache affinity and route memory."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from kvfleet.adapters.base import ChatMessage


@dataclass
class PromptFingerprint:
    """A fingerprint of a prompt for cache affinity routing."""

    full_hash: str
    system_hash: str
    prefix_hash: str
    conversation_hash: str
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def session_key(self) -> str:
        """Key for session-level affinity (system + conversation context)."""
        return f"{self.system_hash}:{self.conversation_hash}"

    @property
    def prefix_key(self) -> str:
        """Key for prefix-level affinity (system + prefix)."""
        return f"{self.system_hash}:{self.prefix_hash}"


class PromptFingerprinter:
    """Generate fingerprints from chat messages for cache affinity routing.

    Fingerprinting enables:
    - Session-aware routing (same conversation → same worker)
    - Prefix-aware routing (similar system prompts → cache reuse)
    - Route memory (similar requests → similar treatment)
    """

    def __init__(self, prefix_tokens: int = 128) -> None:
        self.prefix_tokens = prefix_tokens

    def fingerprint(self, messages: list[ChatMessage]) -> PromptFingerprint:
        """Generate a fingerprint from chat messages.

        Args:
            messages: Chat messages to fingerprint.

        Returns:
            PromptFingerprint with multiple hash levels.
        """
        system_text = ""
        user_texts: list[str] = []
        all_text_parts: list[str] = []

        for msg in messages:
            all_text_parts.append(f"{msg.role}:{msg.content}")
            if msg.role == "system":
                system_text += msg.content
            elif msg.role == "user":
                user_texts.append(msg.content)

        full_text = "\n".join(all_text_parts)
        prefix_text = self._extract_prefix(full_text)

        # Conversation context = all user messages (captures multi-turn context)
        conversation_text = "\n".join(user_texts)

        return PromptFingerprint(
            full_hash=self._hash(full_text),
            system_hash=self._hash(system_text) if system_text else "",
            prefix_hash=self._hash(prefix_text),
            conversation_hash=self._hash(conversation_text),
            token_estimate=self._estimate_tokens(full_text),
        )

    def similarity(self, fp1: PromptFingerprint, fp2: PromptFingerprint) -> float:
        """Compute similarity between two fingerprints.

        Returns:
            Score 0.0 (different) to 1.0 (identical).
        """
        score = 0.0
        if fp1.full_hash == fp2.full_hash:
            return 1.0
        if fp1.system_hash and fp1.system_hash == fp2.system_hash:
            score += 0.5
        if fp1.prefix_hash == fp2.prefix_hash:
            score += 0.3
        if fp1.conversation_hash == fp2.conversation_hash:
            score += 0.2
        return min(score, 1.0)

    def _extract_prefix(self, text: str) -> str:
        """Extract prefix text (approximate first N tokens)."""
        words = text.split()
        prefix_words = words[: self.prefix_tokens]
        return " ".join(prefix_words)

    @staticmethod
    def _hash(text: str) -> str:
        """Compute SHA-256 hash of normalized text."""
        normalized = _normalize_text(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (~4 chars per token for English)."""
        return max(1, len(text) // 4)


def _normalize_text(text: str) -> str:
    """Normalize text for consistent hashing.

    - Lowercase
    - Collapse whitespace
    - Strip leading/trailing whitespace
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text
