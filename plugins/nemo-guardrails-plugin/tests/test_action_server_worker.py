# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_relay_plugin import PluginContext
from registration_helpers import registered_llm_execution

from nemoguardrails_nemo_relay import configuration, execution_policy, worker


def _context() -> MagicMock:
    context = MagicMock(spec=PluginContext)
    context.runtime = SimpleNamespace(list_runtime_registrations=AsyncMock(return_value=[]))
    return context


@pytest.fixture
def action_server() -> Iterator[tuple[str, list[dict[str, object]]]]:
    calls: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP hook
            body = self.rfile.read(int(self.headers.get("content-length", "0")))
            parsed = json.loads(body)
            calls.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("authorization"),
                    "private_header": self.headers.get("x-private-secret"),
                    "body": parsed,
                }
            )
            text = parsed.get("action_parameters", {}).get("text")
            status = 200
            if text == "allow":
                response = json.dumps({"status": "success", "result": True}).encode()
            elif text == "block":
                response = json.dumps({"status": "success", "result": False}).encode()
            elif text == "action failure":
                status = 503
                response = b"service unavailable"
            elif text == "malformed action":
                response = b"{not-json"
            elif text == "slow action":
                time.sleep(0.2)
                response = json.dumps({"status": "success", "result": True}).encode()
            else:
                response = json.dumps({"status": "success", "result": True}).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(response)))
            self.end_headers()
            try:
                self.wfile.write(response)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _write_remote_action_config(path: Path, endpoint: str) -> None:
    path.mkdir()
    (path / "config.yml").write_text(
        f"actions_server_url: {endpoint}\nrails:\n  input:\n    flows:\n      - remote action rail\n",
        encoding="utf-8",
    )
    (path / "rails.co").write_text(
        """
define bot refuse
  "blocked"

define flow remote action rail
  $allowed = execute remote_policy_action(text=$user_message)
  if not $allowed
    bot refuse
    stop
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (path / "config.py").write_text(
        """
from nemoguardrails.actions import action

@action(name="remote_policy_action")
def remote_policy_action(text: str) -> bool:
    return True

def init(app: object) -> None:
    app.register_action(remote_policy_action)
""".strip()
        + "\n",
        encoding="utf-8",
    )


async def test_real_action_server_allow_block_failure_malformed_timeout_and_recovery(
    tmp_path: Path,
    action_server: tuple[str, list[dict[str, object]]],
) -> None:
    endpoint, calls = action_server
    config = tmp_path / "remote-actions"
    _write_remote_action_config(config, endpoint)
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(
        context,
        {
            "config_path": str(config),
            "check_timeout_ms": configuration.MIN_CHECK_TIMEOUT_MS,
            "secret_env": {"PRIVATE_ACTION_SERVER_CANARY": "must-not-become-an-http-header"},
        },
    )
    assert plugin._text_policy is not None
    plugin._text_policy._timeout_seconds = 0.05
    callback = registered_llm_execution(context).callback

    async def check(text: str) -> str | None:
        request = {
            "headers": {"authorization": "provider-secret", "x-private-secret": "provider-canary"},
            "content": {"model": "fixture", "messages": [{"role": "user", "content": text}]},
        }
        next_call = SimpleNamespace(call=AsyncMock(return_value={"choices": []}))
        try:
            await callback("openai.chat_completions", request, next_call)
        except execution_policy._LlmPolicyError as error:
            return str(error)
        return None

    assert await check("allow") is None
    assert await check("block") == "NeMo Guardrails prevented execution in input rail 'remote action rail'"
    # Guardrails 0.24 represents action transport and parse failures as a
    # blocked/internal-error result. Relay still fails closed, but the public
    # check result cannot distinguish it from a deliberate policy block.
    assert await check("action failure") == "NeMo Guardrails prevented execution in input rail 'remote action rail'"
    assert await check("malformed action") == "NeMo Guardrails prevented execution in input rail 'remote action rail'"
    assert await check("slow action") == "NeMo Guardrails input check timed out"
    assert await check("recovered") is None

    assert len(calls) == 6
    assert all(call["path"] == "/v1/actions/run" for call in calls)
    assert all(call["authorization"] is None for call in calls)
    assert all(call["private_header"] is None for call in calls)
    assert all("provider-secret" not in json.dumps(call["body"]) for call in calls)
    assert all("must-not-become-an-http-header" not in json.dumps(call["body"]) for call in calls)
    await plugin.close()
