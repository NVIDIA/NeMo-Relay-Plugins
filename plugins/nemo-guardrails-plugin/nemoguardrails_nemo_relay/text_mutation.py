# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless native-payload rewrites for Guardrails ``MODIFIED`` results.

Text checks operate on a provider-neutral projection. Applying a transformed
string is safe only when that projected value has one unambiguous native text
leaf. This module deliberately knows the five Relay codecs qualified by the
worker and changes no other field in their request or response payloads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias, cast

PathPart: TypeAlias = str | int
JsonPath: TypeAlias = tuple[PathPart, ...]


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


@dataclass(frozen=True, slots=True)
class _TextTarget:
    """One semantic text value and any exact provider mirror of that value."""

    path: JsonPath
    mirrors: tuple[JsonPath, ...] = ()


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


def rewrite_request_text(
    request: object,
    *,
    codec_names: Sequence[str],
    expected: str,
    replacement: str,
    max_text_characters: int,
) -> dict[str, Any]:
    """Copy a request and replace its latest projected user text.

    Ambiguous OpenAI Chat/Anthropic request envelopes are accepted only when
    every plausible codec points at the exact same native leaf.
    """

    root = _object_copy(request, "provider request")
    content = root.get("content")
    if not isinstance(content, dict):
        raise TextMutationError("provider request content is not an object")
    _validate_replacement((replacement,), max_text_characters)

    paths: list[JsonPath] = []
    for codec_name in codec_names:
        try:
            relative = _request_target(codec_name, content, expected)
        except TextMutationError:
            raise TextMutationError("provider request text mapping is ambiguous or unsupported") from None
        paths.append(("content", *relative))
    if not paths or any(path != paths[0] for path in paths[1:]):
        raise TextMutationError("provider request text mapping is ambiguous")
    _set_string(root, paths[0], expected, replacement)
    return root


def rewrite_response_texts(
    response: object,
    *,
    codec_names: Sequence[str],
    codec_variants: Sequence[str | None],
    expected: Sequence[str],
    replacements: Sequence[str],
    max_text_characters: int,
) -> dict[str, Any]:
    """Copy a unary response and rewrite every selectable text candidate.

    Exactly one plausible response codec must match. Each selectable candidate
    must consist of one native scalar text leaf. Multiple choices are rewritten
    independently and atomically.
    """

    if len(expected) != len(replacements):
        raise TextMutationError("provider response candidate counts disagree")
    _validate_replacement(replacements, max_text_characters)
    root = _object_copy(response, "provider response")
    variants = tuple(codec_variants) or (None,) * len(codec_names)
    if len(variants) != len(codec_names):
        raise TextMutationError("provider response codec identity is invalid")

    matches: list[tuple[_TextTarget, ...]] = []
    for codec_name, codec_variant in zip(codec_names, variants, strict=True):
        try:
            targets = _response_targets(codec_name, codec_variant, root, tuple(expected))
        except TextMutationError:
            continue
        matches.append(targets)
    if len(matches) != 1:
        raise TextMutationError("provider response text mapping is ambiguous or unsupported")

    targets = matches[0]
    if len(targets) != len(replacements):
        raise TextMutationError("provider response candidate counts disagree")
    for target, old, new in zip(targets, expected, replacements, strict=True):
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


def _request_target(codec_name: str, content: dict[str, Any], expected: str) -> JsonPath:
    if codec_name == "openai_chat":
        index, message = _latest_message(content.get("messages"), frozenset({"user"}))
        return ("messages", index, *_message_content_target(message, expected=expected, part_types=frozenset({"text"})))

    if codec_name == "anthropic_messages":
        index, message = _latest_message(content.get("messages"), frozenset({"user"}))
        return ("messages", index, *_message_content_target(message, expected=expected, part_types=frozenset({"text"})))

    if codec_name == "openai_responses":
        input_value = content.get("input")
        if isinstance(input_value, str):
            if input_value != expected:
                raise TextMutationError("native text does not match the checked projection")
            return ("input",)
        index, message = _latest_message(input_value, frozenset({"user"}))
        return (
            "input",
            index,
            *_message_content_target(message, expected=expected, part_types=frozenset({"input_text", "text"})),
        )

    if codec_name == "gemini_generate_content":
        index, message = _latest_message(content.get("contents"), frozenset({"user"}))
        return (
            "contents",
            index,
            "parts",
            *_one_part_path(
                message.get("parts"),
                expected=expected,
                text_types=None,
                exclude_thought=True,
            ),
        )

    if codec_name == "oci_genai":
        prefix: JsonPath = ("chatRequest",) if isinstance(content.get("chatRequest"), dict) else ()
        chat_request = cast(dict[str, Any], _value_at(content, prefix) if prefix else content)
        api_format = str(chat_request.get("apiFormat", "GENERIC")).upper()
        if api_format == "COHERE":
            if chat_request.get("message") != expected:
                raise TextMutationError("native text does not match the checked projection")
            return (*prefix, "message")
        if api_format not in {"GENERIC", "COHEREV2"}:
            raise TextMutationError("OCI request dialect is unsupported")
        index, message = _latest_message(chat_request.get("messages"), frozenset({"USER", "user"}))
        return (
            *prefix,
            "messages",
            index,
            *_message_content_target(message, expected=expected, part_types=frozenset({"TEXT", "text"})),
        )

    raise TextMutationError("provider request codec is unsupported")


def _response_targets(
    codec_name: str,
    codec_variant: str | None,
    response: dict[str, Any],
    expected: tuple[str, ...],
) -> tuple[_TextTarget, ...]:
    if codec_name == "openai_chat":
        return _openai_chat_response_targets(response, expected)
    if codec_name == "openai_responses":
        return _openai_responses_response_targets(response, expected)
    if codec_name == "anthropic_messages":
        return _anthropic_response_targets(response, expected)
    if codec_name == "gemini_generate_content":
        return _gemini_response_targets(response, expected)
    if codec_name == "oci_genai":
        return _oci_response_targets(response, codec_variant, expected)
    raise TextMutationError("provider response codec is unsupported")


def _openai_chat_response_targets(response: dict[str, Any], expected: tuple[str, ...]) -> tuple[_TextTarget, ...]:
    choices = response.get("choices")
    if not isinstance(choices, list):
        raise TextMutationError("OpenAI Chat choices are missing")
    targets: list[_TextTarget] = []
    values: list[str] = []
    for index, choice in enumerate(choices):
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise TextMutationError("OpenAI Chat message is invalid")
        content = message.get("content")
        refusal = message.get("refusal")
        if isinstance(content, str) and not isinstance(refusal, str):
            targets.append(_TextTarget(("choices", index, "message", "content")))
            values.append(content)
        elif isinstance(refusal, str) and content is None:
            targets.append(_TextTarget(("choices", index, "message", "refusal")))
            values.append(refusal)
        elif content is not None or refusal is not None:
            raise TextMutationError("OpenAI Chat candidate is not one scalar text leaf")
    return _matching_targets(targets, values, expected)


def _openai_responses_response_targets(response: dict[str, Any], expected: tuple[str, ...]) -> tuple[_TextTarget, ...]:
    paths: list[JsonPath] = []
    values: list[str] = []
    output = response.get("output")
    if output is not None and not isinstance(output, list):
        raise TextMutationError("OpenAI Responses output is invalid")
    for item_index, item in enumerate(output or []):
        if not isinstance(item, dict):
            raise TextMutationError("OpenAI Responses output item is invalid")
        if item.get("type") == "output_text" and isinstance(item.get("text"), str):
            paths.append(("output", item_index, "text"))
            values.append(cast(str, item["text"]))
        elif item.get("type") == "message":
            blocks = item.get("content")
            if not isinstance(blocks, list):
                continue
            for block_index, block in enumerate(blocks):
                if not isinstance(block, dict):
                    raise TextMutationError("OpenAI Responses content block is invalid")
                if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                    paths.append(("output", item_index, "content", block_index, "text"))
                    values.append(cast(str, block["text"]))
                elif block.get("type") == "refusal" and isinstance(block.get("refusal"), str):
                    paths.append(("output", item_index, "content", block_index, "refusal"))
                    values.append(cast(str, block["refusal"]))

    aggregate = response.get("output_text")
    if aggregate is not None and not isinstance(aggregate, str):
        raise TextMutationError("OpenAI Responses aggregate text is invalid")
    if len(paths) == 1:
        mirrors = (("output_text",),) if aggregate == values[0] else ()
        if aggregate not in (None, "", values[0]):
            raise TextMutationError("OpenAI Responses aggregate text disagrees")
        return _matching_targets([_TextTarget(paths[0], mirrors)], values, expected)
    if not paths and isinstance(aggregate, str) and aggregate:
        return _matching_targets([_TextTarget(("output_text",))], [aggregate], expected)
    raise TextMutationError("OpenAI Responses output is not one exact native text leaf")


def _anthropic_response_targets(response: dict[str, Any], expected: tuple[str, ...]) -> tuple[_TextTarget, ...]:
    if len(expected) != 1:
        raise TextMutationError("Anthropic response must contain one text candidate")
    content = response.get("content")
    path = _one_part_path(content, expected=expected[0], text_types=frozenset({"text"}))
    return _matching_targets(
        [_TextTarget(("content", *path))],
        [cast(str, _value_at(content, path))],
        expected,
    )


def _gemini_response_targets(response: dict[str, Any], expected: tuple[str, ...]) -> tuple[_TextTarget, ...]:
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        raise TextMutationError("Gemini candidates are missing")
    targets: list[_TextTarget] = []
    values: list[str] = []
    for index, candidate in enumerate(candidates):
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            raise TextMutationError("Gemini candidate content is invalid")
        visible = [
            (part_index, cast(str, part["text"]))
            for part_index, part in enumerate(parts)
            if isinstance(part, dict) and isinstance(part.get("text"), str) and part.get("thought") is not True
        ]
        if not visible:
            continue
        if len(visible) != 1:
            raise TextMutationError("Gemini candidate is not one exact native text leaf")
        part_index, value = visible[0]
        targets.append(_TextTarget(("candidates", index, "content", "parts", part_index, "text")))
        values.append(value)
    return _matching_targets(targets, values, expected)


def _oci_response_targets(
    response: dict[str, Any], codec_variant: str | None, expected: tuple[str, ...]
) -> tuple[_TextTarget, ...]:
    prefix: JsonPath = ("chatResponse",) if isinstance(response.get("chatResponse"), dict) else ()
    chat_response = cast(dict[str, Any], _value_at(response, prefix) if prefix else response)
    api_format = str(chat_response.get("apiFormat", "GENERIC")).upper()
    if codec_variant is not None and api_format != codec_variant:
        raise TextMutationError("OCI response dialect does not match its request")
    if api_format == "COHERE":
        text = chat_response.get("text")
        if not isinstance(text, str):
            raise TextMutationError("OCI COHERE response text is missing")
        return _matching_targets([_TextTarget((*prefix, "text"))], [text], expected)
    if api_format == "COHEREV2":
        if len(expected) != 1:
            raise TextMutationError("OCI COHEREV2 response must contain one text candidate")
        message = chat_response.get("message")
        if not isinstance(message, dict):
            raise TextMutationError("OCI COHEREV2 response message is missing")
        relative = _message_content_target(
            message,
            expected=expected[0],
            part_types=frozenset({"TEXT", "text"}),
        )
        value = cast(str, _value_at(message, relative))
        return _matching_targets([_TextTarget((*prefix, "message", *relative))], [value], expected)
    if api_format != "GENERIC":
        raise TextMutationError("OCI response dialect is unsupported")
    choices = chat_response.get("choices")
    if not isinstance(choices, list):
        raise TextMutationError("OCI GENERIC choices are missing")
    targets: list[_TextTarget] = []
    values: list[str] = []
    for index, choice in enumerate(choices):
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise TextMutationError("OCI GENERIC response message is invalid")
        refusal = message.get("refusal")
        content = message.get("content")
        if isinstance(refusal, str) and content in (None, "", []):
            targets.append(_TextTarget((*prefix, "choices", index, "message", "refusal")))
            values.append(refusal)
            continue
        if isinstance(content, str) and refusal is None:
            targets.append(_TextTarget((*prefix, "choices", index, "message", "content")))
            values.append(content)
            continue
        if isinstance(content, list) and refusal is None:
            visible = [
                (part_index, cast(str, part["text"]))
                for part_index, part in enumerate(content)
                if isinstance(part, dict) and part.get("type") in {"TEXT", "text"} and isinstance(part.get("text"), str)
            ]
            if not visible:
                continue
            if len(visible) != 1:
                raise TextMutationError("OCI candidate is not one exact native text leaf")
            part_index, value = visible[0]
            targets.append(_TextTarget((*prefix, "choices", index, "message", "content", part_index, "text")))
            values.append(value)
            continue
        if content is not None or refusal is not None:
            raise TextMutationError("OCI candidate is not one exact native text leaf")
    return _matching_targets(targets, values, expected)


def _matching_targets(
    targets: Sequence[_TextTarget], values: Sequence[str], expected: tuple[str, ...]
) -> tuple[_TextTarget, ...]:
    if tuple(values) != expected or len(targets) != len(expected):
        raise TextMutationError("native text does not match the checked candidates")
    return tuple(targets)
