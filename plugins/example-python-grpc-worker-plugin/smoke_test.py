# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify this plugin's behavior using its installed distribution and a local provider."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["REPO_DIR"])
from scripts.smoke import installed_gateway, json_http_fixture, post_json


def smoke(bundle: Path, relay: Path):
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
    with json_http_fixture(response) as provider:
        upstream = provider.url + "/v1"
        config = {"requests": {"header_value": "bundle-smoke"}}
        with installed_gateway(
            bundle, relay, config, gateway_args=("--openai-base-url", upstream)
        ) as gateway:
            body = post_json(
                gateway + "/v1/chat/completions",
                {"model": "smoke-model", "messages": [{"role": "user", "content": "test"}]},
            )
            if (
                body["choices"][0]["message"]["content"] != "bundle-smoke-ok"
                or len(provider.calls) != 1
            ):
                raise AssertionError("managed request did not reach the fixture exactly once")
            if provider.calls[0][1].get("x-nemo-relay-plugin") != "bundle-smoke":
                raise AssertionError("plugin did not intercept the managed request")


if __name__ == "__main__":
    smoke(Path(os.environ["BUNDLE_DIR"]), Path(os.environ["RELAY_BIN"]))
