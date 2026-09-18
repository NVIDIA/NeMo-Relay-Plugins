# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize Relay codec annotations for structural tool validation."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from .codec_adapters.common import (
    _reject_duplicate_json_keys,
    _reject_non_json_constant,
    _structural_key,
)
from .structural_tools import (
    MAX_JSON_DEPTH,
    MAX_SCHEMA_NODES,
    ToolCall,
    ToolDefinition,
    ToolExchange,
    ToolResult,
)


class ToolProjectionError(ValueError):
    """Relay's annotation cannot prove complete structural tool coverage."""


@dataclass(frozen=True, slots=True)
class ToolRequestProjection:
    """Provider-neutral tool declarations and turn-local result exchanges."""

    definitions: tuple[ToolDefinition, ...]
    exchanges: tuple[ToolExchange, ...]

    @property
    def has_results(self) -> bool:
        return any(exchange.results for exchange in self.exchanges)


def _contains_structural_data(value: object) -> bool:
    stack = [value]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, (dict, list)):
            if id(current) in seen:
                continue
            seen.add(id(current))
        if isinstance(current, dict):
            for key, child in current.items():
                if not isinstance(key, str) or _structural_key(key):
                    return True
                if key in {"role", "type"} and isinstance(child, str) and _structural_key(child):
                    return True
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)
    return False


def _reject_definition_extras(value: dict[str, Any], allowed_keys: set[str]) -> None:
    for key, child in value.items():
        if key in allowed_keys:
            continue
        if _structural_key(key) or _contains_structural_data(child):
            raise ToolProjectionError("tool definition contains unmodeled structural data")


# Relay preserves provider-owned declaration controls that Guardrails' tool
# checker does not validate. Their values must be treated as opaque because an
# output schema may legitimately define a property named ``tool``. Keep the
# allowlist codec-specific, however: accepting one provider's control on every
# codec could hide future structural traffic from this projection.
_FUNCTION_DEFINITION_CONTROLS_BY_CODEC = {
    "anthropic_messages": frozenset(),
    "gemini_generate_content": frozenset({"behavior", "response", "responseJsonSchema"}),
    "openai_chat": frozenset(),
    "openai_responses": frozenset({"allowed_callers", "async", "defer_loading", "output_schema"}),
    "oci_genai": frozenset(),
}
_TOOL_DEFINITION_CONTROLS_BY_CODEC = {
    "anthropic_messages": frozenset(
        {
            "allowed_callers",
            "cache_control",
            "defer_loading",
            "eager_input_streaming",
            "input_examples",
        }
    ),
    "gemini_generate_content": frozenset(),
    "openai_chat": frozenset(),
    "openai_responses": frozenset(),
    "oci_genai": frozenset(),
}

_GEMINI_SCHEMA_KEYS = frozenset(
    {
        "anyOf",
        "default",
        "description",
        "enum",
        "example",
        "format",
        "items",
        "maxItems",
        "maximum",
        "maxLength",
        "maxProperties",
        "minimum",
        "minItems",
        "minLength",
        "minProperties",
        "nullable",
        "pattern",
        "properties",
        "propertyOrdering",
        "required",
        "title",
        "type",
    }
)
_GEMINI_SCHEMA_TYPES = {
    "ARRAY": "array",
    "BOOLEAN": "boolean",
    "INTEGER": "integer",
    "NULL": "null",
    "NUMBER": "number",
    "OBJECT": "object",
    "STRING": "string",
}
_OCI_COHERE_SCALAR_PARAMETER_TYPES = {
    "bool": "boolean",
    "float": "number",
    "int": "integer",
    "str": "string",
}
_OCI_COHERE_MAX_PARAMETER_TYPE_DEPTH = 8
_OCI_COHERE_MAX_PARAMETER_TYPE_LENGTH = 256
_GEMINI_INT64_SCHEMA_KEYS = frozenset(
    {
        "maxItems",
        "maxLength",
        "maxProperties",
        "minItems",
        "minLength",
        "minProperties",
    }
)
_MAX_INT64 = (1 << 63) - 1


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ToolProjectionError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, field)


def _arguments(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                parse_constant=_reject_non_json_constant,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ToolProjectionError("tool-call arguments are not valid JSON") from exc
    if not isinstance(value, dict):
        raise ToolProjectionError("tool-call arguments must be an object")
    return value


def _result_content(value: object) -> str | list[dict[str, Any]] | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        if not all(isinstance(item, dict) for item in value):
            raise ToolProjectionError("tool-result content blocks must be objects")
        return value
    raise ToolProjectionError("tool-result content has an unsupported shape")


def _gemini_schema_to_json_schema(value: object) -> dict[str, Any]:
    """Translate Gemini's bounded OpenAPI-style Schema into JSON Schema."""

    remaining_nodes = [MAX_SCHEMA_NODES]
    return _translate_gemini_schema(value, depth=0, remaining_nodes=remaining_nodes)


def _translate_gemini_schema(
    value: object,
    *,
    depth: int,
    remaining_nodes: list[int],
) -> dict[str, Any]:
    """Translate one Gemini schema node without exceeding worker limits."""

    if not isinstance(value, dict):
        raise ToolProjectionError("Gemini tool input schema must be an object")
    if depth > MAX_JSON_DEPTH or remaining_nodes[0] <= 0:
        raise ToolProjectionError("Gemini tool input schema exceeds worker limits")
    remaining_nodes[0] -= 1
    unknown = value.keys() - _GEMINI_SCHEMA_KEYS
    if unknown:
        raise ToolProjectionError("Gemini tool input schema uses unsupported fields")

    translated: dict[str, Any] = {}
    for key, child in value.items():
        if key == "type":
            normalized_type = child.upper() if isinstance(child, str) else None
            if normalized_type not in _GEMINI_SCHEMA_TYPES:
                raise ToolProjectionError("Gemini tool input schema has an unsupported type")
            translated[key] = _GEMINI_SCHEMA_TYPES[normalized_type]
        elif key == "properties":
            if not isinstance(child, dict) or any(not isinstance(name, str) for name in child):
                raise ToolProjectionError("Gemini tool schema properties must be an object")
            translated[key] = {
                name: _translate_gemini_schema(
                    schema,
                    depth=depth + 1,
                    remaining_nodes=remaining_nodes,
                )
                for name, schema in child.items()
            }
        elif key == "items":
            translated[key] = _translate_gemini_schema(
                child,
                depth=depth + 1,
                remaining_nodes=remaining_nodes,
            )
        elif key == "anyOf":
            if not isinstance(child, list):
                raise ToolProjectionError("Gemini tool schema anyOf must be an array")
            translated[key] = [
                _translate_gemini_schema(
                    schema,
                    depth=depth + 1,
                    remaining_nodes=remaining_nodes,
                )
                for schema in child
            ]
        elif key == "nullable":
            if not isinstance(child, bool):
                raise ToolProjectionError("Gemini tool schema nullable must be a boolean")
            # Apply this after translating the explicit type below.
            continue
        elif key == "propertyOrdering":
            if not isinstance(child, list) or any(not isinstance(name, str) for name in child):
                raise ToolProjectionError("Gemini tool schema propertyOrdering must be a string array")
            # Ordering is a serialization hint, not an input-validation rule.
            continue
        elif key in _GEMINI_INT64_SCHEMA_KEYS:
            if isinstance(child, bool):
                raise ToolProjectionError("Gemini tool schema bound must be a non-negative int64")
            if isinstance(child, int):
                bound = child
            elif isinstance(child, str) and re.fullmatch(r"0|[1-9][0-9]*", child):
                bound = int(child)
            else:
                raise ToolProjectionError("Gemini tool schema bound must be a non-negative int64")
            if bound < 0 or bound > _MAX_INT64:
                raise ToolProjectionError("Gemini tool schema bound must be a non-negative int64")
            translated[key] = bound
        else:
            translated[key] = child

    if value.get("nullable") is True:
        schema_type = translated.get("type")
        if not isinstance(schema_type, str):
            raise ToolProjectionError("nullable Gemini tool schema must declare a type")
        if schema_type != "null":
            translated["type"] = [schema_type, "null"]
    return translated


def _function_definition(value: dict[str, Any], codec_name: str) -> ToolDefinition:
    controls = _FUNCTION_DEFINITION_CONTROLS_BY_CODEC[codec_name]
    schema_keys = {"parametersJsonSchema"} if codec_name == "gemini_generate_content" else set()
    _reject_definition_extras(
        value,
        {
            "description",
            "name",
            "parameters",
            "strict",
            *schema_keys,
            *controls,
        },
    )
    name = _non_empty_string(value.get("name"), "tool name")
    description = value.get("description")
    if description is not None and not isinstance(description, str):
        raise ToolProjectionError("tool description must be a string")
    parameters = value.get("parameters")
    parameters_json_schema = value.get("parametersJsonSchema")
    if parameters is not None and parameters_json_schema is not None:
        raise ToolProjectionError("tool definition has multiple input schemas")
    schema = parameters if parameters is not None else parameters_json_schema
    if schema is not None and not isinstance(schema, dict):
        raise ToolProjectionError("tool input schema must be an object")
    if schema is not None and codec_name == "gemini_generate_content" and parameters is not None:
        schema = _gemini_schema_to_json_schema(schema)
    strict = value.get("strict")
    if strict is not None and not isinstance(strict, bool):
        raise ToolProjectionError("tool strict flag must be a boolean")
    return ToolDefinition(
        name=name,
        description=description,
        input_schema=schema,
        strict=strict,
    )


def _oci_cohere_function_definition(value: dict[str, Any]) -> ToolDefinition:
    """Translate OCI COHERE v1's bounded Python-type parameter map."""

    if set(value) - {"description", "name", "parameterDefinitions"}:
        raise ToolProjectionError("OCI COHERE tool definition contains unmodeled fields")
    name = _non_empty_string(value.get("name"), "tool name")
    description = value.get("description")
    if description is not None and not isinstance(description, str):
        raise ToolProjectionError("tool description must be a string")
    raw_parameters = value.get("parameterDefinitions", {})
    if raw_parameters is None:
        raw_parameters = {}
    if not isinstance(raw_parameters, dict) or any(not isinstance(key, str) or not key for key in raw_parameters):
        raise ToolProjectionError("OCI COHERE parameterDefinitions must be an object")

    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter_name, raw_definition in raw_parameters.items():
        if not isinstance(raw_definition, dict) or set(raw_definition) - {"description", "isRequired", "type"}:
            raise ToolProjectionError("OCI COHERE parameter definition has an unsupported shape")
        raw_type = raw_definition.get("type")
        try:
            type_expression = (
                ast.parse(raw_type, mode="eval").body
                if isinstance(raw_type, str) and len(raw_type) <= _OCI_COHERE_MAX_PARAMETER_TYPE_LENGTH
                else None
            )
        except SyntaxError as exc:
            raise ToolProjectionError("OCI COHERE parameter definition has an unsupported Python type") from exc
        if type_expression is None:
            raise ToolProjectionError("OCI COHERE parameter definition has an unsupported Python type")
        property_schema = _oci_cohere_python_type_to_schema(type_expression)
        parameter_description = raw_definition.get("description")
        if parameter_description is not None and not isinstance(parameter_description, str):
            raise ToolProjectionError("OCI COHERE parameter description must be a string")
        is_required = raw_definition.get("isRequired", False)
        if not isinstance(is_required, bool):
            raise ToolProjectionError("OCI COHERE parameter isRequired must be a boolean")

        if parameter_description is not None:
            property_schema["description"] = parameter_description
        properties[parameter_name] = property_schema
        if is_required:
            required.append(parameter_name)

    schema: dict[str, Any] = {
        "additionalProperties": False,
        "properties": properties,
        "type": "object",
    }
    if required:
        schema["required"] = required
    return ToolDefinition(
        name=name,
        description=description,
        input_schema=schema,
        strict=None,
    )


def _oci_cohere_python_type_to_schema(value: ast.expr, depth: int = 0) -> dict[str, Any]:
    """Translate the JSON-compatible subset of OCI's Python type notation."""

    if depth > _OCI_COHERE_MAX_PARAMETER_TYPE_DEPTH:
        raise ToolProjectionError("OCI COHERE parameter type is nested too deeply")
    if isinstance(value, ast.Name):
        json_type = _OCI_COHERE_SCALAR_PARAMETER_TYPES.get(value.id)
        if json_type is not None:
            return {"type": json_type}
        if value.id in {"list", "List"}:
            return {"items": {}, "type": "array"}
        if value.id in {"dict", "Dict"}:
            return {"additionalProperties": True, "type": "object"}
        raise ToolProjectionError("OCI COHERE parameter definition has an unsupported Python type")
    if not isinstance(value, ast.Subscript) or not isinstance(value.value, ast.Name):
        raise ToolProjectionError("OCI COHERE parameter definition has an unsupported Python type")

    container = value.value.id
    if container in {"list", "List"}:
        if isinstance(value.slice, ast.Tuple):
            raise ToolProjectionError("OCI COHERE list parameter requires one element type")
        return {
            "items": _oci_cohere_python_type_to_schema(value.slice, depth + 1),
            "type": "array",
        }
    if container in {"dict", "Dict"}:
        if not isinstance(value.slice, ast.Tuple) or len(value.slice.elts) != 2:
            raise ToolProjectionError("OCI COHERE dict parameter requires key and value types")
        key_type, item_type = value.slice.elts
        if not isinstance(key_type, ast.Name) or key_type.id != "str":
            raise ToolProjectionError("OCI COHERE dict parameter keys must be strings")
        return {
            "additionalProperties": _oci_cohere_python_type_to_schema(item_type, depth + 1),
            "type": "object",
        }
    raise ToolProjectionError("OCI COHERE parameter definition has an unsupported Python type")


def _definition(
    value: object,
    codec_name: str,
    codec_variant: str | None,
) -> ToolDefinition | None:
    if not isinstance(value, dict):
        raise ToolProjectionError("tool definition must be an object")
    tool_type = value.get("type")
    if tool_type == "function":
        _reject_definition_extras(
            value,
            {"function", "type", *_TOOL_DEFINITION_CONTROLS_BY_CODEC[codec_name]},
        )
        function = value.get("function")
        if not isinstance(function, dict):
            raise ToolProjectionError("function tool definition is missing")
        return _function_definition(function, codec_name)

    if tool_type == "provider_native" and value.get("provider") == "oci_genai":
        _reject_definition_extras(value, {"kind", "provider", "type", "value"})
        native = value.get("value")
        if not isinstance(native, dict):
            raise ToolProjectionError("provider-native tool is not a supported function tool")
        if codec_variant == "COHERE":
            return _oci_cohere_function_definition(native)
        if str(native.get("type", value.get("kind", ""))).upper() != "FUNCTION":
            raise ToolProjectionError("provider-native tool is not a supported function tool")
        function = native.get("function")
        if function is not None:
            _reject_definition_extras(native, {"function", "type"})
            if not isinstance(function, dict):
                raise ToolProjectionError("provider-native function tool is invalid")
            return _function_definition(function, codec_name)
        # OCI GENERIC flattens the function fields; COHEREV2 nests them.
        return _function_definition(native, codec_name)

    if (
        tool_type == "provider_native"
        and value.get("provider") == "anthropic_messages"
        and value.get("kind") == "custom"
    ):
        _reject_definition_extras(value, {"kind", "provider", "type", "value"})
        native = value.get("value")
        if not isinstance(native, dict) or native.get("type") != "custom":
            raise ToolProjectionError("provider-native Anthropic function tool is invalid")
        # Anthropic's explicit `type: custom` tool is still a client-defined
        # function. Invocation controls and examples are provider metadata;
        # Guardrails' structural rail checks the name and input schema only.
        _reject_definition_extras(
            native,
            {
                "allowed_callers",
                "cache_control",
                "defer_loading",
                "description",
                "eager_input_streaming",
                "input_examples",
                "input_schema",
                "name",
                "strict",
                "type",
            },
        )
        return _function_definition(
            {
                "description": native.get("description"),
                "name": native.get("name"),
                "parameters": native.get("input_schema"),
                "strict": native.get("strict"),
            },
            codec_name,
        )

    # Relay preserves hosted/server tools as provider-native values. They may
    # coexist with function declarations, but its normalized response lacks
    # the call type needed to validate a hosted invocation. Keep them outside
    # the function allowlist; raw response checks reject one if it is emitted.
    if tool_type == "provider_native":
        return None
    raise ToolProjectionError("tool definition is not structurally covered")


def _standard_call(value: object) -> ToolCall:
    if not isinstance(value, dict):
        raise ToolProjectionError("tool call must be an object")
    function = value.get("function")
    if not isinstance(function, dict):
        raise ToolProjectionError("tool call function is missing")
    call_type = value.get("type")
    return ToolCall(
        call_id=_non_empty_string(value.get("id"), "tool call id"),
        name=_non_empty_string(function.get("name"), "tool call name"),
        arguments=_arguments(function.get("arguments", {})),
        type=_non_empty_string(call_type, "tool call type"),
    )


def _normalized_call(value: object) -> ToolCall:
    if not isinstance(value, dict):
        raise ToolProjectionError("normalized tool call must be an object")
    return ToolCall(
        call_id=_non_empty_string(value.get("call_id", value.get("id")), "tool call id"),
        name=_non_empty_string(value.get("name"), "tool call name"),
        arguments=_arguments(value.get("arguments", {})),
    )


def _tool_use_call(value: object) -> ToolCall:
    if not isinstance(value, dict):
        raise ToolProjectionError("tool-use block must be an object")
    return ToolCall(
        call_id=_non_empty_string(value.get("id"), "tool call id"),
        name=_non_empty_string(value.get("name"), "tool call name"),
        arguments=_arguments(value.get("input", {})),
    )


def _tool_result(
    *,
    call_id: object,
    name: object,
    content: object,
    is_error: object = False,
) -> ToolResult:
    if is_error is None:
        is_error = False
    if not isinstance(is_error, bool):
        raise ToolProjectionError("tool-result error flag must be a boolean")
    return ToolResult(
        call_id=_non_empty_string(call_id, "tool result call id"),
        name=_optional_string(name, "tool result name"),
        content=_result_content(content),
        is_error=is_error,
    )


def _message_calls(message: dict[str, Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    raw_calls = message.get("tool_calls")
    if raw_calls is not None:
        if not isinstance(raw_calls, list):
            raise ToolProjectionError("message tool_calls must be an array")
        calls.extend(_standard_call(call) for call in raw_calls)

    if message.get("role") == "tool_call":
        calls.append(_normalized_call(message))

    # Relay 0.9 preserves a valid OpenAI Chat assistant message as provider
    # native when it carries fields such as ``refusal`` or ``audio`` that its
    # normalized Message type does not model. The raw request validator has
    # already checked every call's exact wire shape, so retain those calls for
    # structural validation without changing the forwarded provider payload.
    if (
        message.get("role") == "provider_native"
        and message.get("provider") == "openai_chat"
        and message.get("kind") == "assistant"
    ):
        value = message.get("value")
        if isinstance(value, dict):
            native_calls = value.get("tool_calls")
            if native_calls is not None:
                if not isinstance(native_calls, list):
                    raise ToolProjectionError("provider-native OpenAI tool_calls must be an array")
                calls.extend(_standard_call(call) for call in native_calls)

    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append(_tool_use_call(block))
    return calls


def _message_results(message: dict[str, Any]) -> list[ToolResult]:
    role = message.get("role")
    if role == "tool":
        return [
            _tool_result(
                call_id=message.get("tool_call_id"),
                name=message.get("name"),
                content=message.get("content"),
                is_error=message.get("is_error", False),
            )
        ]
    if role == "tool_result":
        return [
            _tool_result(
                call_id=message.get("call_id"),
                name=message.get("name"),
                content=message.get("output"),
                is_error=message.get("is_error", False),
            )
        ]

    if role == "provider_native":
        value = message.get("value")
        if isinstance(value, dict) and str(value.get("role", "")).lower() == "tool":
            return [
                _tool_result(
                    call_id=value.get("tool_call_id", value.get("toolCallId")),
                    name=value.get("name"),
                    content=value.get("content"),
                    is_error=value.get("is_error", value.get("isError", False)),
                )
            ]

    results: list[ToolResult] = []
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results.append(
                    _tool_result(
                        call_id=block.get("tool_use_id"),
                        name=block.get("name"),
                        content=block.get("content"),
                        is_error=block.get("is_error", False),
                    )
                )
    return results


def _has_unprojected_tool_shape(message: dict[str, Any], calls: list[ToolCall], results: list[ToolResult]) -> bool:
    role = message.get("role")
    if role in {"tool", "tool_call", "tool_result"}:
        return not (calls or results)
    if role == "provider_native":
        value = message.get("value")
        if isinstance(value, dict):
            structural_keys = {
                key
                for key in {
                    "functionCall",
                    "functionResponse",
                    "function_call",
                    "toolCalls",
                    "tool_calls",
                }
                if value.get(key) is not None
            }
            projected_openai_calls = (
                message.get("provider") == "openai_chat"
                and message.get("kind") == "assistant"
                and structural_keys <= {"tool_calls"}
                and "tool_calls" in value
            )
            if structural_keys and not projected_openai_calls:
                return True
            discriminator = str(value.get("role", value.get("type", ""))).lower()
            is_projected_result = str(value.get("role", "")).lower() == "tool" and bool(results)
            if ("tool" in discriminator or "function" in discriminator) and not is_projected_result:
                return True
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            discriminator = str(block.get("type", "")).lower()
            if ("tool" in discriminator or "function" in discriminator) and discriminator not in {
                "tool_use",
                "tool_result",
            }:
                return True
    return False


def _is_responses_reasoning_item(message: dict[str, Any]) -> bool:
    value = message.get("value")
    return (
        message.get("role") == "provider_native"
        and message.get("provider") == "openai_responses"
        and message.get("kind") == "reasoning"
        and isinstance(value, dict)
        and value.get("type") == "reasoning"
    )


def project_request_tools(
    annotated: dict[str, Any],
    *,
    codec_name: str,
    codec_variant: str | None = None,
    include_definitions: bool = True,
) -> ToolRequestProjection:
    """Project one Relay request annotation into the stable adapter contract."""

    raw_definitions = annotated.get("tools")
    if not include_definitions or raw_definitions is None:
        definitions: tuple[ToolDefinition, ...] = ()
    else:
        if not isinstance(raw_definitions, list):
            raise ToolProjectionError("normalized tools must be an array")
        definitions = tuple(
            definition
            for value in raw_definitions
            if (definition := _definition(value, codec_name, codec_variant)) is not None
        )

    messages = annotated.get("messages")
    if not isinstance(messages, list):
        raise ToolProjectionError("normalized messages are missing")

    mutable_exchanges: list[tuple[list[ToolCall], list[ToolResult]]] = []
    open_exchange: tuple[list[ToolCall], list[ToolResult]] | None = None
    open_exchange_is_responses_items = False
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            raise ToolProjectionError("normalized message must be an object")
        calls = _message_calls(raw_message)
        results = _message_results(raw_message)
        if calls and results:
            raise ToolProjectionError("one normalized message mixes tool calls and results")
        if _has_unprojected_tool_shape(raw_message, calls, results):
            raise ToolProjectionError("normalized message contains unsupported tool traffic")

        if calls:
            is_responses_item = raw_message.get("role") == "tool_call"
            if (
                is_responses_item
                and open_exchange_is_responses_items
                and open_exchange is not None
                and not open_exchange[1]
            ):
                # OpenAI Responses emits parallel function calls as adjacent
                # normalized items rather than one assistant message.
                open_exchange[0].extend(calls)
            else:
                open_exchange = (calls, [])
                mutable_exchanges.append(open_exchange)
                open_exchange_is_responses_items = is_responses_item
        elif results:
            if open_exchange is None:
                open_exchange = ([], [])
                mutable_exchanges.append(open_exchange)
            open_exchange[1].extend(results)
        elif _is_responses_reasoning_item(raw_message):
            # Responses may place a reasoning item between a function call and
            # its output. It belongs to the same model turn and carries no
            # structural link after the check above has accepted its shape.
            continue
        else:
            open_exchange = None
            open_exchange_is_responses_items = False

    exchanges = tuple(ToolExchange(calls=tuple(calls), results=tuple(results)) for calls, results in mutable_exchanges)
    return ToolRequestProjection(definitions=definitions, exchanges=exchanges)


def project_response_calls(annotated: dict[str, Any]) -> tuple[ToolCall, ...]:
    """Project Relay's normalized response tool calls into the adapter contract."""

    raw_calls = annotated.get("tool_calls")
    if raw_calls is None:
        return ()
    if not isinstance(raw_calls, list):
        raise ToolProjectionError("normalized response tool_calls must be an array")
    return tuple(_normalized_call(call) for call in raw_calls)
