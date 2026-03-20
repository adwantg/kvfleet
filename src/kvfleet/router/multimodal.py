"""Vision and multimodal routing — route based on input modality."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from kvfleet.adapters.base import ChatMessage
from kvfleet.config.schema import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ModalityDetection:
    """Result of modality detection on a request."""

    has_images: bool = False
    has_audio: bool = False
    has_video: bool = False
    has_documents: bool = False
    image_count: int = 0
    modalities: list[str] | None = None
    estimated_image_tokens: int = 0

    def __post_init__(self) -> None:
        if self.modalities is None:
            self.modalities = []
            if self.has_images:
                self.modalities.append("vision")
            if self.has_audio:
                self.modalities.append("audio")
            if self.has_video:
                self.modalities.append("video")
            if self.has_documents:
                self.modalities.append("document")
            if not self.modalities:
                self.modalities.append("text")

    @property
    def is_multimodal(self) -> bool:
        return self.has_images or self.has_audio or self.has_video

    @property
    def primary_modality(self) -> str:
        if self.has_video:
            return "video"
        if self.has_audio:
            return "audio"
        if self.has_images:
            return "vision"
        return "text"


def detect_modality(messages: list[ChatMessage] | list[dict[str, Any]]) -> ModalityDetection:
    """Detect input modalities from chat messages.

    Supports OpenAI multi-modal message format where content
    can be a list of content parts with types like "image_url",
    "input_audio", etc.

    Args:
        messages: List of chat messages (ChatMessage or dicts).

    Returns:
        ModalityDetection result.
    """
    has_images = False
    has_audio = False
    has_video = False
    has_documents = False
    image_count = 0
    estimated_tokens = 0

    for msg in messages:
        content = msg.content if isinstance(msg, ChatMessage) else msg.get("content", "")

        if isinstance(content, list):
            # Multi-part content (OpenAI vision format)
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    if part_type == "image_url":
                        has_images = True
                        image_count += 1
                        # Estimate tokens: ~85 tokens for low detail, ~765 for high
                        detail = part.get("image_url", {}).get("detail", "auto")
                        estimated_tokens += 85 if detail == "low" else 765
                    elif part_type == "input_audio":
                        has_audio = True
                    elif part_type == "video":
                        has_video = True
                    elif part_type == "file" or part_type == "document":
                        has_documents = True
        elif isinstance(content, str) and _has_image_markers(content):
            # Check for inline image references
            has_images = True
            image_count += content.count("data:image/")
            image_count += len(re.findall(r"!\[.*?\]\(.*?\)", content))

    return ModalityDetection(
        has_images=has_images,
        has_audio=has_audio,
        has_video=has_video,
        has_documents=has_documents,
        image_count=image_count,
        estimated_image_tokens=estimated_tokens,
    )


def _has_image_markers(text: str) -> bool:
    """Check if text contains image markers."""
    markers = [
        "data:image/",
        "![",  # Markdown image
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
    ]
    return any(m in text.lower() for m in markers)


def filter_vision_capable(
    models: list[ModelConfig],
    detection: ModalityDetection,
) -> list[ModelConfig]:
    """Filter models to those capable of handling the detected modalities.

    Uses model capabilities and tags to determine support.

    Args:
        models: All candidate models.
        detection: Detected modalities.

    Returns:
        Filtered list of capable models.
    """
    if not detection.is_multimodal:
        return models  # Text-only, all models qualify

    capable: list[ModelConfig] = []
    for model in models:
        supports_needed = True

        if (
            detection.has_images
            and not model.capabilities.supports_vision
            and model.tags.get("vision") != "true"
        ):
            # Check capabilities.supports_vision or tags
            supports_needed = False

        if detection.has_audio and model.tags.get("audio") != "true":
            supports_needed = False

        if detection.has_video and model.tags.get("video") != "true":
            supports_needed = False

        if supports_needed:
            capable.append(model)

    if not capable:
        # Fallback: return all models, let them handle the error
        logger.warning(
            "No models support required modalities %s, returning all candidates",
            detection.modalities,
        )
        return models

    return capable


def estimate_multimodal_cost(
    detection: ModalityDetection,
    model: ModelConfig,
) -> float:
    """Estimate additional cost for multimodal content.

    Args:
        detection: Detected modalities.
        model: Model to estimate cost for.

    Returns:
        Estimated additional cost in USD.
    """
    extra_tokens = detection.estimated_image_tokens
    if extra_tokens > 0:
        return (extra_tokens / 1000) * model.cost_per_1k_input_tokens
    return 0.0


def filter_tool_capable(
    models: list[ModelConfig],
    request: Any,
) -> list[ModelConfig]:
    """Filter models to those supporting tool/function calling.

    If the request contains tools, only models with
    ``supports_tools=True`` are eligible.  When no capable model
    exists the full list is returned with a warning so routing
    can degrade gracefully rather than crash.

    Args:
        models: Candidate models.
        request: ChatRequest (or any object with a ``tools`` attribute).

    Returns:
        Filtered list of tool-capable models.
    """
    tools = getattr(request, "tools", None)
    if not tools:
        return models

    capable = [m for m in models if m.capabilities.supports_tools]
    if not capable:
        logger.warning(
            "Request contains tools but no models are marked "
            "supports_tools=True; returning all candidates"
        )
        return models
    return capable


def filter_json_mode_capable(
    models: list[ModelConfig],
    request: Any,
) -> list[ModelConfig]:
    """Filter models to those supporting JSON response mode.

    If the request specifies ``response_format`` with type
    ``json_object``, only models with ``supports_json_mode=True``
    are eligible.  Graceful fallback when none are capable.

    Args:
        models: Candidate models.
        request: ChatRequest (or any object with a ``response_format``
            attribute).

    Returns:
        Filtered list of JSON-mode-capable models.
    """
    fmt = getattr(request, "response_format", None)
    if not fmt or not isinstance(fmt, dict):
        return models
    if fmt.get("type") != "json_object":
        return models

    capable = [m for m in models if m.capabilities.supports_json_mode]
    if not capable:
        logger.warning(
            "Request requires json_object response format but no models "
            "are marked supports_json_mode=True; returning all candidates"
        )
        return models
    return capable
