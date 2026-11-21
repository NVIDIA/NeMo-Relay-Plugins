# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the installed bundle serves a repeated request from cache."""

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
    request = {
        "model": "smoke-model",
        "temperature": 0,
        "messages": [{"role": "user", "content": "cache me"}],
    }
    with json_http_fixture(response) as provider:
        upstream = provider.url + "/v1"
        config = {"namespace": "bundle-smoke"}
        with installed_gateway(
            bundle, relay, config, gateway_args=("--openai-base-url", upstream)
        ) as gateway:
            for _ in range(2):
                body = post_json(gateway + "/v1/chat/completions", request)
                if body["choices"][0]["message"]["content"] != "bundle-smoke-ok":
                    raise AssertionError("gateway did not return the fixture response")
            if len(provider.calls) != 1:
                raise AssertionError(
                    f"the repeated request reached the provider {len(provider.calls)} times"
                )


if __name__ == "__main__":
    smoke(Path(os.environ["BUNDLE_DIR"]), Path(os.environ["RELAY_BIN"]))
