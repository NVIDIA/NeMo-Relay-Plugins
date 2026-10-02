# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded local and remote text-check backends.

The execution policy owns projection, timeout, and fail-open/fail-closed behavior. Backends in
this module do exactly one thing: evaluate an already projected Guardrails message list for one
explicit input or output phase and return a provider-neutral result.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}$")
_REMOTE_RESPONSE_LIMIT = 64 * 1024
_MAX_CONFIG_IDS = 32
_MAX_CONFIG_ID_LENGTH = 256
_MAX_MODEL_LENGTH = 256
_MAX_REMOTE_TIMEOUT_MS = 25_000


class CheckBackendError(RuntimeError):
    """A content-free check-backend failure safe to map through worker policy."""


class CheckStatus(str, Enum):
    """Statuses returned by Guardrails' standalone check API."""

    PASSED = "passed"
    MODIFIED = "modified"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Provider-neutral result from one explicit Guardrails phase."""

    status: CheckStatus
    content: str
    rail: str | None = None

    @property
    def transformed_content(self) -> str | None:
        """Return content only when Guardrails explicitly reports a rewrite."""

        return self.content if self.status is CheckStatus.MODIFIED else None


class CheckBackend(Protocol):
    """Backend contract used by the Relay execution policy."""

    async def check(self, messages: Sequence[Mapping[str, str]], phase: str) -> CheckResult:
        """Run one explicit ``input`` or ``output`` phase."""

    async def close(self) -> None:
        """Release backend-owned resources."""


class LocalCheckBackend:
    """Adapter over a local ``Guardrails.check_async`` engine."""

    def __init__(self, rails: Any, rail_type: type[Any]) -> None:
        self._rails = rails
        self._rail_type = rail_type

    async def check(self, messages: Sequence[Mapping[str, str]], phase: str) -> CheckResult:
        try:
            selected = self._rail_type.INPUT if phase == "input" else self._rail_type.OUTPUT
            result = await self._rails.check_async(list(messages), rail_types=[selected])
            status_value = getattr(getattr(result, "status", None), "value", None)
            status = CheckStatus(status_value)
            content = getattr(result, "content", None)
            rail = getattr(result, "rail", None)
            if not isinstance(content, str) or (rail is not None and not isinstance(rail, str)):
                raise ValueError("invalid local result")
            return CheckResult(status=status, content=content, rail=rail)
        except CheckBackendError:
            raise
        except Exception:
            raise CheckBackendError("local Guardrails check failed") from None

    async def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class RemoteChecksSettings:
    """Validated connection settings for Guardrails ``POST /v1/checks``."""

    endpoint: str
    config_ids: tuple[str, ...]
    phases: frozenset[str]
    model: str
    header_env: tuple[tuple[str, str], ...] = ()
    timeout_seconds: float = 25.0
    max_response_bytes: int = _REMOTE_RESPONSE_LIMIT

    @classmethod
    def from_config(cls, value: object) -> "RemoteChecksSettings":
        """Parse the plugin's remote-check mapping without reading secret values."""

        if not isinstance(value, Mapping):
            raise ValueError("remote_checks must be an object")
        allowed = {
            "endpoint",
            "config_ids",
            "phases",
            "model",
            "allow_remote_content_logging_and_retention",
            "header_env",
            "timeout_ms",
            "max_response_bytes",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"remote_checks contains unknown fields: {', '.join(unknown)}")

        endpoint = _checks_endpoint(value.get("endpoint"))
        config_ids = _config_ids(value.get("config_ids"))
        phases = _phases(value.get("phases"))
        # Guardrails' /v1/checks implementation injects this value into the
        # selected configuration and replaces its configured main model. It is
        # therefore an evaluator choice, not cosmetic request metadata.
        model = value.get("model")
        if not isinstance(model, str) or not model.strip() or len(model) > _MAX_MODEL_LENGTH:
            raise ValueError("remote_checks.model must be a non-empty bounded string")
        acknowledgement = value.get("allow_remote_content_logging_and_retention")
        if acknowledgement is not True:
            raise ValueError(
                "remote_checks.allow_remote_content_logging_and_retention must be true because "
                "the Guardrails 0.24 server logs checked content and retains conversation "
                "events in its process memory"
            )

        header_env = _header_env(value.get("header_env", {}))
        timeout_ms = value.get("timeout_ms", 25_000)
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or not 1_000 <= timeout_ms <= _MAX_REMOTE_TIMEOUT_MS
        ):
            raise ValueError(f"remote_checks.timeout_ms must be between 1000 and {_MAX_REMOTE_TIMEOUT_MS}")
        max_response_bytes = value.get("max_response_bytes", _REMOTE_RESPONSE_LIMIT)
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or not 1_024 <= max_response_bytes <= 1_048_576
        ):
            raise ValueError("remote_checks.max_response_bytes must be between 1024 and 1048576")
        return cls(
            endpoint=endpoint,
            config_ids=config_ids,
            phases=phases,
            model=model.strip(),
            header_env=header_env,
            timeout_seconds=timeout_ms / 1_000,
            max_response_bytes=max_response_bytes,
        )

    def headers(self, environment: Mapping[str, str] | None = None) -> dict[str, str]:
        """Resolve configured headers from environment variables at activation time."""

        source = os.environ if environment is None else environment
        resolved: dict[str, str] = {}
        for header, env_name in self.header_env:
            secret = source.get(env_name)
            if not secret:
                raise ValueError(f"remote check credential environment variable {env_name!r} is not set")
            resolved[header] = secret
        return resolved


class RemoteChecksBackend:
    """Bounded client for Guardrails' explicit-phase ``/v1/checks`` endpoint."""

    def __init__(
        self,
        settings: RemoteChecksSettings,
        *,
        headers: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            headers=dict(headers or settings.headers()),
            timeout=httpx.Timeout(settings.timeout_seconds),
            follow_redirects=False,
            # The configured endpoint and credentials form one explicit trust
            # boundary. Ambient proxy variables must not silently redirect a
            # worker's policy traffic or authorization headers.
            trust_env=False,
            transport=transport,
        )

    async def check(self, messages: Sequence[Mapping[str, str]], phase: str) -> CheckResult:
        if phase not in {"input", "output"}:
            raise CheckBackendError("unsupported Guardrails check phase")
        payload = {
            "model": self._settings.model,
            "messages": [dict(message) for message in messages],
            "guardrails": {
                "config_ids": list(self._settings.config_ids),
                "rail_types": [phase],
            },
        }
        try:
            async with self._client.stream("POST", self._settings.endpoint, json=payload) as response:
                if response.status_code != 200:
                    raise CheckBackendError("remote Guardrails check failed")
                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        declared_length = int(content_length)
                    except ValueError:
                        raise CheckBackendError("remote Guardrails response is invalid") from None
                    if declared_length < 0 or declared_length > self._settings.max_response_bytes:
                        raise CheckBackendError("remote Guardrails response is oversized")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self._settings.max_response_bytes:
                        raise CheckBackendError("remote Guardrails response is oversized")
        except CheckBackendError:
            raise
        except (httpx.HTTPError, TimeoutError):
            raise CheckBackendError("remote Guardrails check failed") from None

        return _decode_remote_result(bytes(body))

    async def close(self) -> None:
        await self._client.aclose()


def _checks_endpoint(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("remote_checks.endpoint must be a non-empty URL")
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("remote_checks.endpoint must use http or https")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("remote_checks.endpoint cannot contain credentials, a query, or a fragment")
    if parsed.scheme == "http" and not _loopback_host(parsed.hostname):
        raise ValueError("remote_checks.endpoint must use https unless it targets loopback")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1/checks"):
        path = f"{path}/v1/checks"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _config_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("remote_checks.config_ids must be a non-empty list")
    if len(value) > _MAX_CONFIG_IDS:
        raise ValueError(f"remote_checks.config_ids cannot contain more than {_MAX_CONFIG_IDS} entries")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item) > _MAX_CONFIG_ID_LENGTH:
            raise ValueError("remote_checks.config_ids entries must be non-empty bounded strings")
        result.append(item.strip())
    if len(set(result)) != len(result):
        raise ValueError("remote_checks.config_ids cannot contain duplicates")
    return tuple(result)


def _header_env(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise ValueError("remote_checks.header_env must be an object")
    if len(value) > 16:
        raise ValueError("remote_checks.header_env cannot contain more than 16 headers")
    result: list[tuple[str, str]] = []
    forbidden = {"content-length", "host", "transfer-encoding"}
    seen: set[str] = set()
    for header, env_name in value.items():
        if not isinstance(header, str) or not _HEADER_NAME.fullmatch(header):
            raise ValueError("remote_checks.header_env contains an invalid header name")
        normalized = header.lower()
        if normalized in forbidden or normalized in seen:
            raise ValueError("remote_checks.header_env contains a forbidden or duplicate header")
        if not isinstance(env_name, str) or not _ENV_NAME.fullmatch(env_name):
            raise ValueError("remote_checks.header_env values must be environment variable names")
        seen.add(normalized)
        result.append((header, env_name))
    return tuple(result)


def _phases(value: object) -> frozenset[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("remote_checks.phases must be a non-empty list")
    if any(item not in {"input", "output"} for item in value):
        raise ValueError("remote_checks.phases supports only input and output")
    if len(set(value)) != len(value):
        raise ValueError("remote_checks.phases cannot contain duplicates")
    return frozenset(value)


def _decode_remote_result(body: bytes) -> CheckResult:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CheckBackendError("remote Guardrails response is invalid") from None
    if not isinstance(value, dict) or set(value) - {"status", "content", "rail"}:
        raise CheckBackendError("remote Guardrails response is invalid")
    status = value.get("status")
    content = value.get("content")
    rail = value.get("rail")
    try:
        parsed_status = CheckStatus(status)
    except (TypeError, ValueError):
        raise CheckBackendError("remote Guardrails response is invalid") from None
    if not isinstance(content, str) or (rail is not None and not isinstance(rail, str)):
        raise CheckBackendError("remote Guardrails response is invalid")
    return CheckResult(status=parsed_status, content=content, rail=rail)
