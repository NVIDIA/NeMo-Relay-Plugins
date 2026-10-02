# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One policy-rejection base for LLM and tool worker callbacks."""

try:
    from nemo_relay_plugin import GuardrailRejectedError as _PolicyErrorBase
except ImportError:  # Relay has not released a typed worker rejection yet.
    _PolicyErrorBase = RuntimeError


class PolicyRejectedError(_PolicyErrorBase):
    """Reject guarded work using Relay's typed error when available."""
