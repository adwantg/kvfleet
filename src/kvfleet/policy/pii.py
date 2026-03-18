"""PII detection for policy enforcement."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PIIDetectionResult:
    """Result of PII detection scan."""

    has_pii: bool = False
    pii_types: list[str] = field(default_factory=list)
    matches: list[dict[str, str]] = field(default_factory=list)
    redacted_text: str = ""


# Patterns for common PII types
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "date_of_birth": re.compile(
        r"\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b"
    ),
    "passport": re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
}


class PIIDetector:
    """Pattern-based PII detection for routing policy enforcement.

    Detects common PII patterns (email, phone, SSN, credit card, etc.)
    in prompt text to enforce privacy routing policies.

    Note: This is a heuristic-based detector. For production use with
    high sensitivity requirements, integrate a dedicated NER-based PII
    detection service.
    """

    def __init__(self, patterns: dict[str, re.Pattern[str]] | None = None) -> None:
        self.patterns = patterns or PII_PATTERNS

    def detect(self, text: str) -> PIIDetectionResult:
        """Scan text for PII patterns.

        Args:
            text: Input text to scan.

        Returns:
            PIIDetectionResult with findings.
        """
        result = PIIDetectionResult()
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                result.has_pii = True
                result.pii_types.append(pii_type)
                result.matches.append(
                    {
                        "type": pii_type,
                        "value": match.group(),
                        "start": str(match.start()),
                        "end": str(match.end()),
                    }
                )
        # Deduplicate types
        result.pii_types = list(set(result.pii_types))
        return result

    def redact(self, text: str, replacement: str = "[REDACTED]") -> PIIDetectionResult:
        """Scan and redact PII from text.

        Args:
            text: Input text.
            replacement: Replacement string for PII.

        Returns:
            PIIDetectionResult with redacted_text.
        """
        result = self.detect(text)
        redacted = text
        for pii_type, pattern in self.patterns.items():
            redacted = pattern.sub(f"{replacement}({pii_type})", redacted)
        result.redacted_text = redacted
        return result

    def has_pii(self, text: str) -> bool:
        """Quick check if text contains PII."""
        return any(pattern.search(text) for pattern in self.patterns.values())
