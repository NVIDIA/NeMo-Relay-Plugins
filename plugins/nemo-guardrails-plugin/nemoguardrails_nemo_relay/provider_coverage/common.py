# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small shared types for provider response inspection."""

from __future__ import annotations

import json
import re
from collections.abc import Set
from dataclasses import dataclass
from typing import Any, TypeAlias

from ..structural_tools import MAX_JSON_CHARACTERS, MAX_JSON_NODES, ToolCall

PathPart: TypeAlias = str | int
JsonPath: TypeAlias = tuple[PathPart, ...]
MAX_RESPONSE_SEGMENTS = 32


class _UnsupportedRequest(ValueError):
    """The selected policy cannot safely inspect required payload content."""


@dataclass(frozen=True, slots=True)
class TextFragment:
    """One checked string and its location in the provider response."""

    text: str
    path: JsonPath
    mirrors: tuple[JsonPath, ...] = ()


@dataclass(frozen=True, slots=True)
class ResponseCandidate:
    """Text and function calls returned by one selectable candidate."""

    fragments: tuple[TextFragment, ...] = ()
    calls: tuple[ToolCall, ...] = ()
    separator: str = "\n"
    mutation_safe: bool = True

    @property
    def text(self) -> str | None:
        if not self.fragments:
            return None
        return self.separator.join(fragment.text for fragment in self.fragments)


@dataclass(frozen=True, slots=True)
class ResponseInspection:
    """Provider response data consumed by output and structural rails."""

    codec_name: str
    codec_variant: str | None
    candidates: tuple[ResponseCandidate, ...]
    reasoning: tuple[str, ...] = ()

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(text for candidate in self.candidates if (text := candidate.text) is not None)

    @property
    def tool_candidates(self) -> tuple[tuple[ToolCall, ...], ...]:
        return tuple(candidate.calls for candidate in self.candidates)

    @property
    def has_calls(self) -> bool:
        return any(candidate.calls for candidate in self.candidates)

    @property
    def mutation_safe(self) -> bool:
        return all(candidate.mutation_safe for candidate in self.candidates)


def text_fragment(text: object, path: JsonPath, *mirrors: JsonPath) -> TextFragment:
    if not isinstance(text, str):
        raise _UnsupportedRequest("provider response text is invalid")
    return TextFragment(text, path, tuple(mirrors))


def tool_call(call_id: object, name: object, arguments: object) -> ToolCall:
    """Normalize one validated provider function call."""

    if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
        raise _UnsupportedRequest("provider response tool call is incomplete")
    if isinstance(arguments, str):
        if len(arguments) > MAX_JSON_CHARACTERS:
            raise _UnsupportedRequest("provider response tool arguments are too large")
        try:
            arguments = json.loads(
                arguments,
                parse_constant=_reject_non_json_constant,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise _UnsupportedRequest("provider response tool arguments are invalid") from exc
    if not isinstance(arguments, dict):
        raise _UnsupportedRequest("provider response tool arguments must be an object")
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _reject_non_json_constant(token: str) -> None:
    raise ValueError(token)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(key)
        value[key] = child
    return value


def _structural_key(value: str) -> bool:
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).lower().replace("-", "_").split("_")
    return bool({"call", "calls", "function", "functions", "mcp", "tool", "tools", "toolset"}.intersection(words))


def _reject_structural_extensions(value: dict[str, Any], allowed: Set[str], context: str) -> bool:
    """Reject hidden tool traffic and report meaningful unknown metadata."""

    extensions = [(key, child) for key, child in value.items() if key not in allowed]
    stack = [child for _, child in extensions]
    if any(not isinstance(key, str) or _structural_key(key) for key, _ in extensions):
        raise _UnsupportedRequest(f"{context} contains unsupported tool data")
    has_metadata = any(child not in (None, "", [], {}) for _, child in extensions)
    visited = 0
    while stack:
        current = stack.pop()
        visited += 1
        if visited > MAX_JSON_NODES:
            raise _UnsupportedRequest(f"{context} extension data is too large")
        if isinstance(current, dict):
            for key, child in current.items():
                if (
                    not isinstance(key, str)
                    or _structural_key(key)
                    or (
                        key in {"event", "kind", "object", "role", "type"}
                        and isinstance(child, str)
                        and _structural_key(child)
                    )
                ):
                    raise _UnsupportedRequest(f"{context} contains unsupported tool data")
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)
    return has_metadata


def _reject_unmodeled_fields(value: dict[str, Any], allowed: Set[str], context: str) -> None:
    if set(value) - allowed:
        raise _UnsupportedRequest(f"{context} contains unmodeled fields")
