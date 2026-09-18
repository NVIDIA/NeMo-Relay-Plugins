# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemoguardrails_nemo_relay.remote_actions import validated_actions_server_url


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("http://localhost:8080/", "http://localhost:8080"),
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080"),
        ("http://[::1]:8080", "http://[::1]:8080"),
        ("https://actions.example.com", "https://actions.example.com"),
    ],
)
def test_actions_server_url_accepts_unambiguous_origins(value: object, expected: str | None) -> None:
    assert validated_actions_server_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        7,
        "ftp://actions.example.com",
        "http://actions.example.com",
        "https://user:secret@actions.example.com",
        "https://actions.example.com/prefix",
        "https://actions.example.com?token=secret",
        "https://actions.example.com#fragment",
        " https://actions.example.com",
        "https://actions.example.com\n.invalid",
        "http://[::1",
    ],
)
def test_actions_server_url_rejects_unsafe_or_ignored_parts_without_echoing_secrets(value: object) -> None:
    with pytest.raises(ValueError) as captured:
        validated_actions_server_url(value)

    assert "secret" not in str(captured.value)
    assert "token" not in str(captured.value)
