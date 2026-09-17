# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit policy for provider payload that text rails do not inspect.

NeMo Guardrails text checks consume text. Provider payloads can also carry
images, files, audio, and private model reasoning. Keeping those values in the
native request or response is safe only when the deployment explicitly accepts
that they are outside the text-rail decision. This module owns that choice and
the narrow validation of the provider-neutral values Relay's codecs expose.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MultimodalPolicy(str, Enum):
    """How text rails handle supported non-text content."""

    STRICT = "strict"
    TEXT_ONLY = "text_only"


class ReasoningPolicy(str, Enum):
    """How output rails handle recognized private reasoning fields."""

    REJECT = "reject"
    FINAL_ANSWER_ONLY = "final_answer_only"
    CHECK_OUTPUT = "check_output"


@dataclass(frozen=True, slots=True)
class PayloadPolicy:
    """Coverage decisions applied without changing the native payload."""

    multimodal: MultimodalPolicy = MultimodalPolicy.STRICT
    reasoning: ReasoningPolicy = ReasoningPolicy.REJECT


@dataclass(frozen=True, slots=True)
class ProjectedOutputText:
    """Every independently selectable text candidate in a provider response."""

    candidates: tuple[str, ...]
    codec_name: str | None = None
    codec_variant: str | None = None
    reasoning: tuple[str, ...] = ()

    def one(self) -> str | None:
        """Return the legacy single-candidate projection."""

        if not self.candidates:
            return None
        if len(self.candidates) != 1:
            raise ValueError("response contains multiple text candidates")
        return self.candidates[0]


def payload_policy_from_config(value: object) -> PayloadPolicy:
    """Parse the bounded plugin setting or raise a content-free error."""

    if value is None:
        return PayloadPolicy()
    if not isinstance(value, dict) or set(value) - {"multimodal", "reasoning"}:
        raise ValueError("payload_policy must contain only multimodal and reasoning")
    try:
        multimodal = MultimodalPolicy(value.get("multimodal", MultimodalPolicy.STRICT.value))
        reasoning = ReasoningPolicy(value.get("reasoning", ReasoningPolicy.REJECT.value))
    except (TypeError, ValueError):
        raise ValueError("payload_policy contains an unsupported value") from None
    return PayloadPolicy(multimodal=multimodal, reasoning=reasoning)


def is_supported_normalized_modality(codec_name: str, part: dict[str, Any]) -> bool:
    """Recognize one non-text value produced by Relay's built-in codecs.

    This deliberately uses an allowlist. A future provider-native block may
    contain prompt text or executable tool state, so an unknown block must not
    become accepted merely because ``text_only`` was selected.
    """

    part_type = part.get("type")
    if codec_name == "openai_chat":
        payload_key = {
            "image_url": "image_url",
            "audio": "audio",
            "file": "file",
        }.get(part_type)
        return (
            payload_key is not None and set(part) == {"type", payload_key} and isinstance(part.get(payload_key), dict)
        )

    if codec_name in {"openai_responses", "anthropic_messages"}:
        payload_key = {"image": "image", "audio": "audio", "file": "file"}.get(part_type)
        return (
            payload_key is not None and set(part) == {"type", payload_key} and isinstance(part.get(payload_key), dict)
        )

    if codec_name == "gemini_generate_content" and part_type == "provider_native":
        kind = part.get("kind")
        if kind not in {"inlineData", "fileData"}:
            return False
        value = part.get("value")
        return (
            set(part) == {"kind", "provider", "type", "value"}
            and part.get("provider") == "gemini"
            and isinstance(value, dict)
            and set(value) == {kind}
            and isinstance(value.get(kind), dict)
        )

    return False


def validate_openai_responses_reasoning(value: object) -> bool:
    """Validate the documented bounded shape of one Responses reasoning item."""

    if not isinstance(value, dict) or value.get("type") != "reasoning":
        return False
    if set(value) - {"content", "encrypted_content", "id", "status", "summary", "type"}:
        return False
    if not isinstance(value.get("id"), str) or not value["id"]:
        return False
    if value.get("encrypted_content") is not None and not isinstance(value["encrypted_content"], str):
        return False
    if value.get("status") not in {None, "completed", "in_progress", "incomplete"}:
        return False
    for field, block_type in (("summary", "summary_text"), ("content", "reasoning_text")):
        blocks = value.get(field, [])
        if not isinstance(blocks, list):
            return False
        if any(
            not isinstance(block, dict)
            or set(block) != {"text", "type"}
            or block.get("type") != block_type
            or not isinstance(block.get("text"), str)
            for block in blocks
        ):
            return False
    return True


def validate_anthropic_reasoning(value: object) -> bool:
    """Validate one Anthropic thinking or redacted-thinking block."""

    if not isinstance(value, dict):
        return False
    block_type = value.get("type")
    if block_type == "thinking":
        return (
            set(value) == {"signature", "thinking", "type"}
            and isinstance(value.get("thinking"), str)
            and isinstance(value.get("signature"), str)
        )
    if block_type == "redacted_thinking":
        return set(value) == {"data", "type"} and isinstance(value.get("data"), str)
    return False
