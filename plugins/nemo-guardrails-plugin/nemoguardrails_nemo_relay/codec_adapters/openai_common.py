# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation shared by the OpenAI Chat and Responses protocols."""

from __future__ import annotations

from .common import _reject_unmodeled_fields, _required_json_object_text, _UnsupportedRequest


def validate_function_call(value: object, *, responses: bool) -> None:
    if not isinstance(value, dict):
        raise _UnsupportedRequest("OpenAI function call is not an object")
    if responses:
        if value.get("type") != "function_call":
            raise _UnsupportedRequest("OpenAI Responses call is not a function call")
        call_id = value.get("call_id")
        name = value.get("name")
        arguments = value.get("arguments")
        _reject_unmodeled_fields(
            value,
            {"arguments", "call_id", "id", "name", "status", "type"},
            "OpenAI Responses function call",
        )
    else:
        if value.get("type") != "function":
            raise _UnsupportedRequest("OpenAI Chat call is not a function call")
        function = value.get("function")
        if not isinstance(function, dict):
            raise _UnsupportedRequest("OpenAI Chat function call body is missing")
        call_id = value.get("id")
        name = function.get("name")
        arguments = function.get("arguments")
        _reject_unmodeled_fields(value, {"function", "id", "type"}, "OpenAI Chat tool call")
        _reject_unmodeled_fields(function, {"arguments", "name"}, "OpenAI Chat function")
    if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
        raise _UnsupportedRequest("OpenAI function call identity is incomplete")
    _required_json_object_text(arguments, "OpenAI function-call arguments")
