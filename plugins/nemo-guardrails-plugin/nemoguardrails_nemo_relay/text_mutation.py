# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless request and response text rewrites.

Text checks operate on a provider-neutral projection. Applying a transformed
string is safe only when that projected value has one unambiguous text leaf.
Relay's active request codec maps a normalized edit back to the provider body;
provider response inspectors retain exact text paths because Relay does not
expose a response encoder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from .provider_coverage.common import JsonPath, ResponseInspection


class MutationMode(str, Enum):
    """Action to take when Guardrails returns ``MODIFIED``."""

    REJECT = "reject"
    APPLY = "apply"


@dataclass(frozen=True, slots=True)
class MutationPolicy:
    """Independent input and output mutation decisions."""

    input: MutationMode = MutationMode.REJECT
    output: MutationMode = MutationMode.REJECT


class TextMutationError(ValueError):
    """The projected text cannot be rewritten without losing information."""


def mutation_policy_from_config(value: object) -> MutationPolicy:
    """Parse the closed mutation policy; rejection remains the default."""

    if value is None:
        return MutationPolicy()
    if not isinstance(value, Mapping) or set(value) - {"input", "output"}:
        raise ValueError("mutation_policy must contain only input and output")
    try:
        input_mode = MutationMode(value.get("input", MutationMode.REJECT.value))
        output_mode = MutationMode(value.get("output", MutationMode.REJECT.value))
    except (TypeError, ValueError):
        raise ValueError("mutation_policy contains an unsupported value") from None
    return MutationPolicy(input=input_mode, output=output_mode)


def rewrite_annotated_request_text(
    annotated: object,
    *,
    expected: str,
    replacement: str,
    max_text_characters: int,
) -> dict[str, Any]:
    """Copy an annotated request and replace one exact latest-user text leaf."""

    root = _object_copy(annotated, "annotated request")
    _validate_replacement((replacement,), max_text_characters)
    index, message = _latest_message(root.get("messages"), frozenset({"user"}))
    relative = _message_content_target(
        message,
        expected=expected,
        part_types=frozenset({"text"}),
    )
    _set_string(root, ("messages", index, *relative), expected, replacement)
    return root


def rewrite_response_texts(
    response: object,
    *,
    inspection: ResponseInspection,
    expected: Sequence[str],
    replacements: Sequence[str],
    max_text_characters: int,
) -> dict[str, Any]:
    """Copy a completed response and rewrite all checked text candidates.

    Each replacement requires one exact native text leaf. All candidates are
    updated atomically.
    """

    if len(expected) != len(replacements):
        raise TextMutationError("provider response candidate counts disagree")
    if not inspection.mutation_safe:
        raise TextMutationError("provider response contains text-bearing metadata")
    _validate_replacement(replacements, max_text_characters)
    root = _object_copy(response, "provider response")
    candidates = tuple(candidate for candidate in inspection.candidates if candidate.text is not None)
    if len(candidates) != len(expected):
        raise TextMutationError("provider response candidate counts disagree")
    if tuple(candidate.text for candidate in candidates) != tuple(expected):
        raise TextMutationError("native text does not match the checked candidates")
    for candidate, old, new in zip(candidates, expected, replacements, strict=True):
        if len(candidate.fragments) != 1:
            raise TextMutationError("projected text is not one exact native text leaf")
        target = candidate.fragments[0]
        _set_string(root, target.path, old, new)
        for mirror in target.mirrors:
            _set_string(root, mirror, old, new)
    return root


def _object_copy(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TextMutationError(f"{label} is not an object")
    return cast(dict[str, Any], deepcopy(value))


def _validate_replacement(values: Sequence[str], maximum: int) -> None:
    if maximum <= 0 or any(not isinstance(value, str) for value in values):
        raise TextMutationError("transformed text is invalid")
    if sum(len(value) for value in values) > maximum:
        raise TextMutationError("transformed text exceeds the supported size")


def _value_at(root: object, path: JsonPath) -> object:
    value = root
    for part in path:
        if isinstance(part, int):
            if not isinstance(value, list) or part < 0 or part >= len(value):
                raise TextMutationError("native text path no longer exists")
            value = value[part]
        else:
            if not isinstance(value, dict) or part not in value:
                raise TextMutationError("native text path no longer exists")
            value = value[part]
    return value


def _set_string(root: object, path: JsonPath, expected: str, replacement: str) -> None:
    if not path or _value_at(root, path) != expected:
        raise TextMutationError("native text does not match the checked projection")
    parent = _value_at(root, path[:-1])
    leaf = path[-1]
    if isinstance(leaf, int):
        if not isinstance(parent, list):
            raise TextMutationError("native text path is invalid")
        parent[leaf] = replacement
    else:
        if not isinstance(parent, dict):
            raise TextMutationError("native text path is invalid")
        parent[leaf] = replacement


def _latest_message(messages: object, roles: frozenset[str]) -> tuple[int, dict[str, Any]]:
    if not isinstance(messages, list):
        raise TextMutationError("provider messages are missing")
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") in roles:
            return index, cast(dict[str, Any], message)
    raise TextMutationError("provider request has no user text message")


def _one_part_path(
    parts: object,
    *,
    expected: str,
    text_types: frozenset[str] | None,
    type_key: str = "type",
    text_key: str = "text",
    exclude_thought: bool = False,
) -> JsonPath:
    if not isinstance(parts, list):
        raise TextMutationError("provider text parts are invalid")
    paths: list[JsonPath] = []
    for index, part in enumerate(parts):
        if not isinstance(part, dict) or not isinstance(part.get(text_key), str):
            continue
        if text_types is not None and part.get(type_key) not in text_types:
            continue
        if exclude_thought and part.get("thought") is True:
            continue
        paths.append((index, text_key))
    if len(paths) != 1 or _value_at(parts, paths[0]) != expected:
        raise TextMutationError("projected text is not one exact native text leaf")
    return paths[0]


def _message_content_target(
    message: Mapping[str, Any],
    *,
    expected: str,
    part_types: frozenset[str],
) -> JsonPath:
    content = message.get("content")
    if isinstance(content, str):
        if content != expected:
            raise TextMutationError("native text does not match the checked projection")
        return ("content",)
    return ("content", *_one_part_path(content, expected=expected, text_types=part_types))
