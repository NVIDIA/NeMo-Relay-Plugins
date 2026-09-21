# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from guardrails_config import write_guardrails_config
from http_test_server import HttpRequest, loopback_http_server
from registration_helpers import registered_llm_execution
from worker_test_helpers import worker_context

from nemoguardrails_nemo_relay import configuration, execution_policy, worker


@pytest.fixture
def action_server() -> Iterator[tuple[str, list[dict[str, object]]]]:
    calls: list[dict[str, object]] = []

    def respond(request: HttpRequest) -> tuple[int, bytes]:
        body = json.loads(request.body)
        calls.append(
            {
                "path": request.path,
                "authorization": request.headers.get("authorization"),
                "private_header": request.headers.get("x-private-secret"),
                "body": body,
            }
        )
        text = body.get("action_parameters", {}).get("text")
        if text == "block":
            return 200, json.dumps({"status": "success", "result": False}).encode()
        if text == "action failure":
            return 503, b"service unavailable"
        if text == "malformed action":
            return 200, b"{not-json"
        if text == "slow action":
            time.sleep(0.2)
        return 200, json.dumps({"status": "success", "result": True}).encode()

    with loopback_http_server(respond) as endpoint:
        yield endpoint, calls


def _write_remote_action_config(path: Path, endpoint: str) -> None:
    write_guardrails_config(
        path.parent,
        f"actions_server_url: {endpoint}\nrails:\n  input:\n    flows:\n      - remote action rail\n",
        name=path.name,
        rails_co="""
define bot refuse
  "blocked"

define flow remote action rail
  $allowed = execute remote_policy_action(text=$user_message)
  if not $allowed
    bot refuse
    stop
""".strip()
        + "\n",
        config_py="""
from nemoguardrails.actions import action

@action(name="remote_policy_action")
def remote_policy_action(text: str) -> bool:
    return True

def init(app: object) -> None:
    app.register_action(remote_policy_action)
""".strip()
        + "\n",
    )


async def test_real_action_server_allow_block_failure_malformed_timeout_and_recovery(
    tmp_path: Path,
    action_server: tuple[str, list[dict[str, object]]],
) -> None:
    endpoint, calls = action_server
    config = tmp_path / "remote-actions"
    _write_remote_action_config(config, endpoint)
    context = worker_context()
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
