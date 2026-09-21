# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small loopback HTTP server shared by integration tests."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass(frozen=True)
class HttpRequest:
    path: str
    headers: dict[str, str]
    body: bytes


HttpResponder = Callable[[HttpRequest], tuple[int, bytes]]


@contextmanager
def loopback_http_server(responder: HttpResponder) -> Iterator[str]:
    """Serve POST requests until the surrounding test exits."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP hook
            body = self.rfile.read(int(self.headers.get("content-length", "0")))
            headers = {name.lower(): value for name, value in self.headers.items()}
            status, response = responder(HttpRequest(self.path, headers, body))
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
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
