# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify this plugin's behavior using its installed distribution and a local hooks server."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["REPO_DIR"])
from scripts.smoke import installed_relay_gateway, json_http_fixture


TOOL_INPUT = {"command": "rm -rf build"}
# What the plugin asks the NeMo Guardrails service about each Bash call.
CHECK_REQUEST = {
    "hook_event_name": "NemoRelayToolConditionalExecutionGuardrail",
    "tool_name": "Bash",
    "tool_input": TOOL_INPUT,
}


def forward_tool_call(gateway, service, bundle: Path, status: int, answer: dict):
    """Send a Claude Code PreToolUse hook through the gateway while the service gives this answer."""
    service.status, service.response = status, answer
    before = len(service.calls)
    hook = {
        "session_id": "smoke",
        "transcript_path": str(bundle / "transcript.jsonl"),
        "cwd": str(bundle),
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": TOOL_INPUT,
    }
    result = gateway.cli.run(
        "hook-forward",
        "claude",
        "--gateway-url",
        gateway.url,
        "--forward-only",
        "--fail-closed",
        input=json.dumps(hook),
        timeout=60,
    )
    checks = [body for body, _headers in service.calls[before:]]
    require(checks == [CHECK_REQUEST], f"expected one check request, got {checks}", result)
    return result


def require(condition: bool, problem: str, result):
    if not condition:
        raise AssertionError(
            f"{problem} (exit code {result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def hook_output(result) -> dict:
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def smoke(bundle: Path, relay: Path):
    with json_http_fixture({}) as service:
        with installed_relay_gateway(bundle, relay, {"endpoint": service.url}) as gateway:
            # Relay's own wording around the plugin's messages can change between
            # releases, so match only the plugin's text.
            denied = forward_tool_call(
                gateway, service, bundle, 200, {"decision": "deny", "reason": "smoke"}
            )
            require(
                denied.returncode == 2
                and "NeMo Guardrails blocked tool call 'Bash': smoke" in denied.stderr,
                "the service's deny did not block the tool call",
                denied,
            )

            allowed = forward_tool_call(gateway, service, bundle, 200, {"decision": "allow"})
            require(
                allowed.returncode == 0 and hook_output(allowed).get("continue") is True,
                "the service's allow did not let the tool call run",
                allowed,
            )

            failure = {
                "error": {"code": "rail_failed", "message": "a tool-call rail failed: smoke"}
            }
            failed = forward_tool_call(gateway, service, bundle, 500, failure)
            require(
                failed.returncode == 2
                and "NeMo Guardrails check failed for tool call 'Bash'" in failed.stderr
                and "rail_failed" in failed.stderr
                # A failed check is an error, not a guardrail rejection.
                and "guardrail rejected" not in failed.stderr,
                "the failed check did not block the tool call as an error",
                failed,
            )


if __name__ == "__main__":
    smoke(Path(os.environ["BUNDLE_DIR"]), Path(os.environ["RELAY_BIN"]))
