# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from scripts.smoke import registered_manifest


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
