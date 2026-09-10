# SPDX-License-Identifier: Apache-2.0
"""Exercise installed bundles through a real Relay gateway and loopback provider."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import tomli_w


def smoke(bundle: Path, relay: Path, switchyard: bool = False):
    manifest_path = bundle / "relay-plugin.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    plugin_id = manifest["plugin"]["id"]
    calls = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((body, self.headers))
            response = json.dumps(
                {
                    "id": "smoke",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "bundle-smoke-ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args):
            pass

    with tempfile.TemporaryDirectory(prefix="relay-install-") as temporary:
        state = Path(temporary)
        config = state / "config.toml"
        config.write_text("")
        plugins = state / "plugins.toml"
        plugins.write_text(
            tomli_w.dumps(
                {
                    "plugins": {
                        "policy": {
                            "overrides": {
                                plugin_id: {"attestation": "integrity_only", "startup": "required"}
                            }
                        }
                    }
                }
            )
        )
        # Keep credentials, implicit plugin settings, and project import paths out of the test.
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("NEMO_RELAY_", "OTEL_"))
            and k not in {"PYTHONPATH", "PYTHONHOME", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
        }
        env.update(
            {
                "XDG_CONFIG_HOME": str(state / "xdg"),
                "NEMO_RELAY_PYTHON": os.environ.get("PLUGIN_PYTHON", sys.executable),
                "NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG": "1",
            }
        )
        base = [str(relay), "--config", str(config), "--plugin-config-path", str(plugins)]

        def cli(*args, success=True):
            result = subprocess.run(
                [*base, *args], env=env, cwd=state, capture_output=True, text=True, timeout=240
            )
            if (result.returncode == 0) != success:
                raise RuntimeError(f"Relay {' '.join(args)}: {result.stdout}\n{result.stderr}")
            return result.stdout

        def validate(target, valid=True):
            result = json.loads(cli("plugins", "validate", target, "--json"))
            data = result.get("data") or {}
            if data.get("valid") is not valid:
                raise AssertionError(f"unexpected validation result: {result}")
            if not valid and data.get("integrity_state") != "invalid":
                raise AssertionError(f"tampering did not fail integrity verification: {result}")

        validate(str(manifest_path))
        cli("plugins", "add", str(manifest_path))
        provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
        thread = threading.Thread(target=provider.serve_forever, daemon=True)
        thread.start()
        proc = None
        try:
            upstream = f"http://127.0.0.1:{provider.server_port}/v1"
            document = tomllib.loads(plugins.read_text(encoding="utf-8"))
            record = next(
                r
                for r in document["plugins"]["dynamic"]
                if Path(r["manifest"]).resolve() == manifest_path.resolve()
            )
            if switchyard:
                record["config"] = {
                    "switchyard_config": {
                        "schema_version": 1,
                        "llm_clients": {"primary": {"format": "openai_chat", "base_url": upstream}},
                        "targets": {"default": {"id": "example/model", "llm_client": "primary"}},
                        "routes": {
                            "default": {
                                "id": "switchyard/default",
                                "type": "passthrough",
                                "target": "default",
                            }
                        },
                    }
                }
            else:
                record["config"] = {"requests": {"header_value": "bundle-smoke"}}
            plugins.write_text(tomli_w.dumps(document))
            cli("plugins", "enable", plugin_id)
            validate(plugin_id)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            log_path = state / "gateway.log"
            with log_path.open("w") as log:
                proc = subprocess.Popen(
                    [*base, "--bind", f"127.0.0.1:{port}", "--openai-base-url", upstream],
                    env=env,
                    cwd=state,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 60
                while True:
                    if proc.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(
                            "gateway failed to start:\n" + log_path.read_text(encoding="utf-8")
                        )
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                            break
                    except OSError:
                        time.sleep(0.2)
                model = "switchyard/default" if switchyard else "smoke-model"
                request = Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=json.dumps(
                        {"model": model, "messages": [{"role": "user", "content": "test"}]}
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urlopen(request, timeout=30) as response:
                        body = json.load(response)
                except HTTPError as error:
                    raise RuntimeError(
                        error.read().decode() + "\n" + log_path.read_text(encoding="utf-8")
                    ) from error
                if body["choices"][0]["message"]["content"] != "bundle-smoke-ok" or len(calls) != 1:
                    raise AssertionError("managed request did not reach the fixture exactly once")
                if switchyard:
                    if calls[0][0]["model"] != "example/model":
                        raise AssertionError("Switchyard did not resolve the configured route")
                elif calls[0][1].get("x-nemo-relay-plugin") != "bundle-smoke":
                    raise AssertionError("plugin did not intercept the managed request")
                if os.name == "nt":
                    subprocess.run(
                        [str(relay), "--bind", f"127.0.0.1:{port}", "gateway", "stop"],
                        env=env,
                        cwd=state,
                        check=True,
                        timeout=30,
                    )
                else:
                    proc.send_signal(signal.SIGINT)
                proc.wait(timeout=30)
                if os.name != "nt" and proc.returncode != 0:
                    raise AssertionError(
                        "gateway did not shut down cleanly:\n"
                        + log_path.read_text(encoding="utf-8")
                    )
            artifact = manifest_path.parent / manifest["source"]["artifact"]
            original = artifact.read_bytes()
            try:
                artifact.write_bytes(original + b"tampered")
                validate(str(manifest_path), valid=False)
                blocked = subprocess.run(
                    [*base, "--bind", "127.0.0.1:0", "--openai-base-url", upstream],
                    env=env,
                    cwd=state,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if (
                    blocked.returncode == 0
                    or "integrity" not in (blocked.stdout + blocked.stderr).lower()
                ):
                    raise AssertionError(
                        "tampered plugin did not block startup: " + blocked.stdout + blocked.stderr
                    )
            finally:
                artifact.write_bytes(original)
            cli("plugins", "disable", plugin_id)
            cli("plugins", "remove", plugin_id)
            listed = json.loads(cli("plugins", "list", "--json"))
            if plugin_id in json.dumps(listed):
                raise AssertionError("removed plugin still appears in registry")
        finally:
            if proc is not None and proc.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True
                    )
                else:
                    proc.kill()
                proc.wait(timeout=15)
            provider.shutdown()
            provider.server_close()
            thread.join(timeout=5)
