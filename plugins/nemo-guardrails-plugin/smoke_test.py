# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the installed Guardrails worker against a local provider."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.environ["REPO_DIR"])
from scripts.smoke import installed_gateway, json_http_fixture, post_json


def expect_rejected(gateway: str, request: dict) -> None:
    try:
        post_json(gateway + "/v1/chat/completions", request)
    except RuntimeError as exc:
        if "NeMo Guardrails" not in str(exc):
            raise AssertionError("request failed outside Guardrails") from exc
    else:
        raise AssertionError("Guardrails unexpectedly allowed the request")


def smoke(bundle: Path, relay: Path) -> None:
    response = {
        "id": "smoke",
        "object": "chat.completion",
        "created": 1,
        "model": "fixture-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "bundle-smoke-ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    with tempfile.TemporaryDirectory(prefix="relay-guardrails-smoke-") as temporary:
        config_dir = Path(temporary)
        (config_dir / "config.yml").write_text(
            "rails:\n"
            "  input:\n    flows: [input rail]\n"
            "  output:\n    flows: [output rail]\n"
            "  tool_output:\n    flows: [tool call validation]\n",
            encoding="utf-8",
        )
        (config_dir / "rails.co").write_text(
            'define bot refuse to respond\n  "blocked"\n\n'
            'define flow input rail\n  if $user_message == "block input"\n'
            "    bot refuse to respond\n    stop\n"
            '  else if $user_message == "modify input"\n'
            '    $user_message = "modified input"\n\n'
            'define flow output rail\n  if $bot_message == "block output"\n'
            "    bot refuse to respond\n    stop\n",
            encoding="utf-8",
        )
        with json_http_fixture(response) as provider:
            request = {
                "model": "smoke-model",
                "messages": [{"role": "user", "content": "hello"}],
            }
            config = {
                "version": 2,
                "config_path": str(config_dir),
                "mutation_policy": {"input": "apply"},
            }
            with installed_gateway(
                bundle,
                relay,
                config,
                gateway_args=("--openai-base-url", provider.url + "/v1"),
            ) as gateway:
                body = post_json(gateway + "/v1/chat/completions", request)
                if body["choices"][0]["message"]["content"] != "bundle-smoke-ok":
                    raise AssertionError("allowed response changed")
                if len(provider.calls) != 1 or provider.calls[0][0] != request:
                    raise AssertionError("allowed request did not reach the fixture unchanged")

                request["messages"][0]["content"] = "modify input"
                post_json(gateway + "/v1/chat/completions", request)
                forwarded = provider.calls[-1][0]
                if forwarded["messages"][0]["content"] != "modified input":
                    raise AssertionError("input mutation did not reach the provider")

                request["messages"][0]["content"] = "block input"
                expect_rejected(gateway, request)
                if len(provider.calls) != 2:
                    raise AssertionError("blocked input reached the provider")

                request["messages"][0]["content"] = "hello"
                response["choices"][0]["message"]["content"] = "block output"
                expect_rejected(gateway, request)
                if len(provider.calls) != 3:
                    raise AssertionError("output was rejected before reaching the provider")

                tool_request = {
                    **request,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                },
                            },
                        }
                    ],
                }
                message = response["choices"][0]["message"]
                message["content"] = None
                message["tool_calls"] = [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                    }
                ]
                response["choices"][0]["finish_reason"] = "tool_calls"
                body = post_json(gateway + "/v1/chat/completions", tool_request)
                if body["choices"][0]["message"]["tool_calls"] != message["tool_calls"]:
                    raise AssertionError("valid tool call changed")

                message["tool_calls"][0]["function"]["name"] = "delete_everything"
                expect_rejected(gateway, tool_request)
                if len(provider.calls) != 5:
                    raise AssertionError("tool call was rejected before reaching the provider")


if __name__ == "__main__":
    smoke(Path(os.environ["BUNDLE_DIR"]), Path(os.environ["RELAY_BIN"]))
