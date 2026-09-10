# SPDX-License-Identifier: Apache-2.0
"""Send Ctrl+C to the private console of a smoke-test gateway.

Run in a helper process so attaching to Relay never detaches the test runner
from its own console. Relay handles Ctrl+C, but does not handle Ctrl+Break.
https://learn.microsoft.com/en-us/windows/console/generateconsolectrlevent
"""

import ctypes
import sys
from ctypes import wintypes


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

    kernel.FreeConsole()  # A helper launched without a console may already be detached.
    if not kernel.AttachConsole(pid):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        # Attaching resets handlers; ignore Ctrl+C in this helper only.
        if not kernel.SetConsoleCtrlHandler(None, True):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.GenerateConsoleCtrlEvent(0, 0):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.FreeConsole()


if __name__ == "__main__":
    interrupt_console(int(sys.argv[1]))
