# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from scripts.smoke import json_http_fixture, post_json, registered_manifest


def test_http_fixture_accepts_unrelated_request_and_response_shapes():
    with json_http_fixture({"reply": 42}) as fixture:
        assert post_json(fixture.url + "/custom-operation", {"event": "custom"}) == {"reply": 42}
        assert len(fixture.calls) == 1
        assert fixture.calls[0][0] == {"event": "custom"}


def test_registered_manifest_matches_file_identity(tmp_path):
    manifest = tmp_path / "relay-plugin.toml"
    manifest.write_text("fixture")
    alias = tmp_path / "alias.toml"
    alias.hardlink_to(manifest)
    if os.name == "nt":
        # Match the extended-length spelling written by Rust canonicalize().
        alias = Path("\\\\?\\" + str(manifest.resolve()))
    unrelated = tmp_path / "other.toml"
    unrelated.write_text("fixture")
    record = {"manifest": str(alias)}
    document = {"plugins": {"dynamic": [{"manifest": str(unrelated)}, record]}}

    assert registered_manifest(document, manifest) is record


def test_registered_manifest_reports_missing_registration(tmp_path):
    manifest = tmp_path / "relay-plugin.toml"
    manifest.write_text("fixture")
    other = tmp_path / "other.toml"
    other.write_text("fixture")
    document = {"plugins": {"dynamic": [{"manifest": str(other)}]}}

    with pytest.raises(RuntimeError, match="Relay did not register") as error:
        registered_manifest(document, manifest)
    assert str(other) in str(error.value)


@pytest.mark.parametrize("wait_result", [0, 258, 0xFFFFFFFF])
def test_windows_console_waits_for_exit_before_detaching(monkeypatch, wait_result):
    from scripts import windows_console

    kernel = Mock()
    kernel.OpenProcess.return_value = 42
    kernel.WaitForSingleObject.return_value = wait_result
    monkeypatch.setattr(windows_console.ctypes, "WinDLL", lambda *a, **kw: kernel, raising=False)
    monkeypatch.setattr(windows_console.ctypes, "get_last_error", lambda: 6, raising=False)
    monkeypatch.setattr(windows_console.ctypes, "WinError", OSError, raising=False)

    if wait_result:
        with pytest.raises(TimeoutError if wait_result == 258 else OSError):
            windows_console.interrupt_console(123)
    else:
        windows_console.interrupt_console(123)

    kernel.assert_has_calls(
        [
            call.AttachConsole(123),
            call.SetConsoleCtrlHandler(None, True),
            call.GenerateConsoleCtrlEvent(0, 0),
            call.WaitForSingleObject(42, 30_000),
            call.FreeConsole(),
            call.CloseHandle(42),
        ]
    )


def test_windows_gateway_enables_interrupts_before_launch(monkeypatch):
    from scripts import windows_console

    kernel = Mock()
    child = Mock()
    child.wait.return_value = 7
    monkeypatch.setattr(windows_console.ctypes, "WinDLL", lambda *a, **kw: kernel, raising=False)

    def spawn(argv):
        assert argv == ["relay", "--bind", "127.0.0.1:1234"]
        kernel.SetConsoleCtrlHandler.assert_called_once_with(None, False)
        return child

    monkeypatch.setattr(windows_console.subprocess, "Popen", spawn)
    assert windows_console.run_gateway(["relay", "--bind", "127.0.0.1:1234"]) == 7
    kernel.SetConsoleCtrlHandler.assert_called_with(None, True)


@pytest.mark.skipif(os.name != "nt", reason="Windows console API")
def test_windows_console_interrupts_real_child(tmp_path):
    from scripts import windows_console

    helper = str(Path(windows_console.__file__).resolve())
    ready = tmp_path / "ready"
    code = (
        "import pathlib, signal, sys, time; "
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
        "pathlib.Path(sys.argv[1]).touch(); time.sleep(120)"
    )
    proc = subprocess.Popen(
        [sys._base_executable, helper, "run", sys._base_executable, "-c", code, str(ready)],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert proc.poll() is None, "console child exited before readiness"
            assert time.monotonic() < deadline, "console child did not become ready"
            time.sleep(0.1)
        subprocess.run([sys._base_executable, helper, str(proc.pid)], check=True, timeout=40)
        assert proc.wait(timeout=5) == 0
    finally:
        if proc.poll() is None:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            proc.wait(timeout=15)
