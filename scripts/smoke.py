# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import tomli_w


def registered_manifest(document: dict, manifest_path: Path) -> dict:
    """Find the installed file, including Windows extended-length path aliases."""
    records = document.get("plugins", {}).get("dynamic", [])
    for record in records:
        # Rust canonicalization can retain the Windows \\?\ prefix whereas
        # Python's resolve() removes it. Compare filesystem identity instead.
        if Path(record["manifest"]).samefile(manifest_path):
            return record
    raise RuntimeError(
        f"Relay did not register {manifest_path}; "
        f"registered manifests: {', '.join(record['manifest'] for record in records)}"
    )


@dataclass
class HttpFixture:
    url: str
    calls: list
    response: dict
    status: int = 200


@contextmanager
def json_http_fixture(response: dict, status: int = 200):
    """Serve a caller-supplied JSON response and record POST bodies and headers.

    Each request gets the fixture's current status and response, so callers can
    change either between requests.
    """
    calls = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((body, self.headers))
            payload = json.dumps(fixture.response).encode()
            self.send_response(fixture.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    fixture = HttpFixture(f"http://127.0.0.1:{provider.server_port}", calls, response, status)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    try:
        yield fixture
    finally:
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=5)


def post_json(url: str, body: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def wait_until_serving(url: str, proc, log_path: Path, timeout: float = 60):
    """Wait until a starting gateway answers HTTP requests.

    Relay accepts connections as soon as it binds its port, but answers them only
    after it activates its plugins. Hook clients give up on a gateway that takes
    longer than half a second to answer.
    """
    deadline = time.monotonic() + timeout
    while True:
        if proc.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError("gateway failed to start:\n" + log_path.read_text(encoding="utf-8"))
        try:
            # A refused connection fails at once; an accepted one waits for the answer.
            with urlopen(url + "/healthz", timeout=max(deadline - time.monotonic(), 0.1)):
                return
        except HTTPError:
            # Any HTTP status means the gateway is serving.
            return
        except OSError:
            time.sleep(0.2)


@dataclass
class RelayCli:
    """Relay's command line with one installation's settings, environment, and directory."""

    command: tuple[str, ...]
    env: dict[str, str]
    cwd: Path

    def run(
        self, *args: str, input: str | None = None, timeout: float = 240
    ) -> subprocess.CompletedProcess:
        """Run a Relay command against this installation and return its result, whatever its exit code."""
        return subprocess.run(
            [*self.command, *args],
            env=self.env,
            cwd=self.cwd,
            input=input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


@dataclass
class RelayGateway:
    url: str
    cli: RelayCli


@contextmanager
def installed_gateway(
    bundle: Path,
    relay: Path,
    config: dict,
    *,
    gateway_args: tuple[str, ...] = (),
    runtime_manifest: str = "relay-plugin.toml",
):
    """Install and activate a bundle, yielding its gateway URL for caller-owned tests."""
    with installed_relay_gateway(
        bundle, relay, config, gateway_args=gateway_args, runtime_manifest=runtime_manifest
    ) as gateway:
        yield gateway.url


@contextmanager
def installed_relay_gateway(
    bundle: Path,
    relay: Path,
    config: dict,
    *,
    gateway_args: tuple[str, ...] = (),
    runtime_manifest: str = "relay-plugin.toml",
):
    """Install and activate a bundle, yielding its gateway for caller-owned tests.

    The gateway's command line shares its settings and per-user credentials, so
    tests can run other Relay commands, such as hook clients, against it.
    Successful tests must also pass graceful shutdown, tamper rejection, and
    removal. On failure, retain gateway diagnostics and clean up child processes.
    """
    manifest_path = bundle / runtime_manifest
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    plugin_id = manifest["plugin"]["id"]
    plugin_config = config
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
        relay_cli = RelayCli(tuple(base), env, state)

        def cli(*args, success=True):
            result = relay_cli.run(*args)
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
        proc = None
        try:
            document = tomllib.loads(plugins.read_text(encoding="utf-8"))
            record = registered_manifest(document, manifest_path)
            record["config"] = plugin_config
            plugins.write_text(tomli_w.dumps(document))
            cli("plugins", "enable", plugin_id)
            validate(plugin_id)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            log_path = state / "gateway.log"
            with log_path.open("w") as log:
                command = [*base, "--bind", f"127.0.0.1:{port}", *gateway_args]
                if os.name == "nt":
                    command = [
                        # Avoid the venv redirector, which adds another process
                        # with its own console signal handling.
                        sys._base_executable,
                        str(Path(__file__).with_name("windows_console.py")),
                        "run",
                        *command,
                    ]
                proc = subprocess.Popen(
                    command,
                    env=env,
                    cwd=state,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    # Ctrl+C is console-wide on Windows. Give Relay its own
                    # console so graceful shutdown cannot interrupt the CI runner.
                    creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
                )
                url = f"http://127.0.0.1:{port}"
                wait_until_serving(url, proc, log_path)
                yield RelayGateway(url, relay_cli)
                if os.name == "nt":
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("windows_console.py")),
                            str(proc.pid),
                        ],
                        env=env,
                        cwd=state,
                        check=True,
                        timeout=40,
                    )
                else:
                    proc.send_signal(signal.SIGINT)
                proc.wait(timeout=30)
                if proc.returncode != 0:
                    raise AssertionError(
                        f"gateway did not shut down cleanly (exit code {proc.returncode}):\n"
                        + log_path.read_text(encoding="utf-8")
                    )
            artifact = manifest_path.parent / manifest["source"]["artifact"]
            original = artifact.read_bytes()
            try:
                artifact.write_bytes(original + b"tampered")
                validate(str(manifest_path), valid=False)
                blocked = subprocess.run(
                    [*base, "--bind", "127.0.0.1:0", *gateway_args],
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
        except Exception:
            # Temporary state is removed on failure; retain the gateway evidence
            # in CI logs before cleanup, including shutdown timeouts.
            if (state / "gateway.log").exists():
                print((state / "gateway.log").read_text(encoding="utf-8"), file=sys.stderr)
            raise
        finally:
            if proc is not None and proc.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True
                    )
                else:
                    proc.kill()
                proc.wait(timeout=15)
