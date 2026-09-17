# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation for Guardrails' remote action-server connection.

Guardrails 0.24 accepts an arbitrary string and later joins it with
``/v1/actions/run``.  A malformed URL therefore fails only when a rail first
executes an action.  Keep the policy worker's activation boundary explicit and
reject ambiguous or credential-bearing URLs before constructing Guardrails.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit, urlunsplit

_MAX_ACTION_SERVER_URL_CHARACTERS = 2_048


def validated_actions_server_url(value: object) -> str | None:
    """Return one canonical action-server origin or raise ``ValueError``.

    Guardrails replaces the configured path with ``/v1/actions/run``.  Requiring
    an origin here prevents an operator from believing a path prefix is used
    when Guardrails would silently discard it.  Plain HTTP is limited to a
    loopback listener; remote action parameters can contain policy inputs and
    must otherwise travel over TLS.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("actions_server_url must be a URL")
    candidate = value.strip()
    if candidate != value or not candidate or len(candidate) > _MAX_ACTION_SERVER_URL_CHARACTERS:
        raise ValueError("actions_server_url must be a bounded URL")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in candidate):
        raise ValueError("actions_server_url contains control characters")

    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError:
        raise ValueError("actions_server_url is invalid") from None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("actions_server_url must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("actions_server_url cannot contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("actions_server_url cannot contain a query or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("actions_server_url must not contain a path")
    if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
        raise ValueError("actions_server_url must use https unless it targets loopback")

    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _is_loopback(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
