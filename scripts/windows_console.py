# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send Ctrl+C to the private console of a smoke-test gateway.

Run in a helper process so attaching to Relay never detaches the test runner
from its own console. Relay handles Ctrl+C, but does not handle Ctrl+Break.
https://learn.microsoft.com/en-us/windows/console/generateconsolectrlevent
"""

import ctypes
import subprocess
import sys
from ctypes import wintypes


def run_gateway(argv: list[str]) -> int:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetConsoleCtrlHandler.argtypes = [ctypes.c_void_p, wintypes.BOOL]
    kernel.SetConsoleCtrlHandler.restype = wintypes.BOOL
    # Hosted runners may ignore Ctrl+C. That flag is inherited independently
    # of registered handlers, so explicitly clear it before spawning Relay.
    if not kernel.SetConsoleCtrlHandler(None, False):
        raise ctypes.WinError(ctypes.get_last_error())
    child = subprocess.Popen(argv)
    # Only the child should handle the console event; relay's exit status is
    # forwarded once its own graceful teardown has completed.
    if not kernel.SetConsoleCtrlHandler(None, True):
        child.kill()
        child.wait()
        raise ctypes.WinError(ctypes.get_last_error())
    return child.wait()


def interrupt_console(pid: int):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.FreeConsole.argtypes = []
    kernel.FreeConsole.restype = wintypes.BOOL
    kernel.AttachConsole.argtypes = [wintypes.DWORD]
    kernel.AttachConsole.restype = wintypes.BOOL
    kernel.SetConsoleCtrlHandler.argtypes = [ctypes.c_void_p, wintypes.BOOL]
    kernel.SetConsoleCtrlHandler.restype = wintypes.BOOL
    kernel.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.GenerateConsoleCtrlEvent.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL

    process = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not process:
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.FreeConsole()  # A helper launched without a console may already be detached.
    try:
        if not kernel.AttachConsole(pid):
            raise ctypes.WinError(ctypes.get_last_error())
        # Attaching resets handlers; ignore Ctrl+C in this helper only.
        if not kernel.SetConsoleCtrlHandler(None, True):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.GenerateConsoleCtrlEvent(0, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        # Delivery is asynchronous. Keep the helper attached until Relay exits;
        # detaching immediately can race console event dispatch.
        result = kernel.WaitForSingleObject(process, 30_000)
        if result == 258:  # WAIT_TIMEOUT
            raise TimeoutError(f"gateway PID {pid} did not exit after Ctrl+C")
        if result != 0:  # WAIT_OBJECT_0
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.FreeConsole()
        kernel.CloseHandle(process)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        sys.exit(run_gateway(sys.argv[2:]))
    interrupt_console(int(sys.argv[1]))
