# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Verify the installed Guardrails worker against a local provider."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.environ["REPO_DIR"])
from scripts.smoke import installed_gateway, json_http_fixture, post_json


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
            "rails:\n  input:\n    flows:\n      - input rail\n",
            encoding="utf-8",
        )
        (config_dir / "rails.co").write_text(
            'define bot refuse to respond\n  "blocked"\n\n'
            'define flow input rail\n  if $user_message == "block input"\n'
            "    bot refuse to respond\n    stop\n",
            encoding="utf-8",
        )
        with json_http_fixture(response) as provider:
            request = {
                "model": "smoke-model",
                "messages": [{"role": "user", "content": "hello"}],
            }
            config = {"version": 2, "config_path": str(config_dir)}
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

                request["messages"][0]["content"] = "block input"
                try:
                    post_json(gateway + "/v1/chat/completions", request)
                except RuntimeError as exc:
                    if "NeMo Guardrails" not in str(exc):
                        raise AssertionError("blocked input failed for an unrelated reason") from exc
                else:
                    raise AssertionError("blocked input unexpectedly passed")
                if len(provider.calls) != 1:
                    raise AssertionError("blocked input reached the provider")


if __name__ == "__main__":
    smoke(Path(os.environ["BUNDLE_DIR"]), Path(os.environ["RELAY_BIN"]))
