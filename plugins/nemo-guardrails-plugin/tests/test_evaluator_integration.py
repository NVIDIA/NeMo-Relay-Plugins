# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from worker_test_helpers import registered_worker

from nemoguardrails_nemo_relay import (
    configuration,
    runtime_selection,
)


def _request(messages: list[dict[str, object]], **content: object) -> dict[str, object]:
    return {
        "headers": {"authorization": "not-forwarded-to-guardrails"},
        "content": {
            "model": "fixture-model",
            "messages": messages,
            **content,
        },
    }


@pytest.fixture
def evaluator_server() -> tuple[str, list[dict[str, object]]]:
    calls: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP hook
            body = self.rfile.read(int(self.headers["content-length"]))
            parsed = json.loads(body)
            calls.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("authorization"),
                    "body": parsed,
                }
            )
            serialized = body.decode("utf-8")
            status = 200
            if "HTTP_500_EVALUATOR" in serialized:
                status = 500
                response = json.dumps({"error": {"message": "fixture evaluator failure"}}).encode("utf-8")
            elif "AUTH_ERROR_EVALUATOR" in serialized:
                status = 401
                response = json.dumps({"error": {"message": "fixture evaluator authentication failure"}}).encode(
                    "utf-8"
                )
            elif "SLOW_EVALUATOR" in serialized:
                time.sleep(0.2)
                response = b""
            elif "MALFORMED_EVALUATOR" in serialized:
                response = b"{not-json"
            else:
                verdict = "Yes" if "User input: BLOCK_EVALUATOR" in serialized else "No"
                response = json.dumps(
                    {
                        "id": "chatcmpl-local",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "local-evaluator",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": verdict},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                ).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(response)))
            self.end_headers()
            try:
                self.wfile.write(response)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _write_evaluator_config(config_path: Path, base_url: str) -> None:
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        f"""
models:
  - type: main
    engine: openai
    model: local-evaluator
    api_key_env_var: NEMO_RELAY_TEST_EVALUATOR_KEY
    parameters:
      base_url: {base_url}
      max_attempts: 1
rails:
  input:
    flows:
      - self check input
prompts:
  - task: self_check_input
    content: |
      Return Yes only when the user input contains BLOCK_EVALUATOR. Otherwise return No.
      User input: {{{{ user_input }}}}
""".strip()
        + "\n",
        encoding="utf-8",
    )


async def test_real_openai_compatible_evaluator_contract_and_failures(
    tmp_path: Path,
    evaluator_server: tuple[str, list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url, calls = evaluator_server
    config_path = tmp_path / "evaluator"
    _write_evaluator_config(config_path, base_url)
    monkeypatch.setenv("NEMO_RELAY_TEST_EVALUATOR_KEY", "evaluator-secret")
    plugin, callback = await registered_worker(
        {
            "config_path": str(config_path),
            "check_timeout_ms": configuration.MIN_CHECK_TIMEOUT_MS,
        }
    )
    assert plugin._text_policy is not None
    plugin._text_policy._timeout_seconds = 0.5

    allowed = await callback(_request([{"role": "user", "content": "hello evaluator"}]))
    blocked = await callback(_request([{"role": "user", "content": "BLOCK_EVALUATOR"}]))
    malformed = await callback(_request([{"role": "user", "content": "MALFORMED_EVALUATOR"}]))
    server_error = await callback(_request([{"role": "user", "content": "HTTP_500_EVALUATOR"}]))
    auth_error = await callback(_request([{"role": "user", "content": "AUTH_ERROR_EVALUATOR"}]))
    plugin._text_policy._timeout_seconds = 0.075
    timed_out = await callback(_request([{"role": "user", "content": "SLOW_EVALUATOR"}]))
    recovered = await callback(_request([{"role": "user", "content": "after evaluator failures"}]))

    assert allowed is None
    assert blocked == "NeMo Guardrails prevented execution in input rail 'self check input'"
    # LLMRails 0.24 turns these evaluator failures into a BLOCKED RailsResult.
    # The standalone result does not preserve enough information for this
    # worker to distinguish them from a deliberate self-check rejection.
    blocked_reason = "NeMo Guardrails prevented execution in input rail 'self check input'"
    assert malformed == blocked_reason
    assert server_error == "NeMo Guardrails input check failed"
    assert auth_error == "NeMo Guardrails input check failed"
    assert timed_out == "NeMo Guardrails input check timed out"
    assert recovered is None
    # IORails counts total attempts, so ``max_attempts: 1`` disables retries
    # without leaking a client option into the provider request body.
    assert len(calls) == 7
    assert sum("HTTP_500_EVALUATOR" in json.dumps(call["body"]) for call in calls) == 1
    assert all(call["path"] == "/v1/chat/completions" for call in calls)
    assert all(call["authorization"] == "Bearer evaluator-secret" for call in calls)
    assert all("not-forwarded-to-guardrails" not in json.dumps(call["body"]) for call in calls)
    assert plugin._rails is not None
    assert plugin._rails.events_history_cache == {}
    if runtime_selection.runtime_uses_explain_state(plugin._rails):
        assert plugin._rails.explain().llm_calls == []
        assert plugin._rails.explain().colang_history is None
    await plugin.close()
