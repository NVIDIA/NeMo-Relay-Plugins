# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One policy-rejection base for LLM and tool worker callbacks."""

try:  # Relay 0.9 adds typed rejection without breaking older SDKs.
    from nemo_relay_plugin import GuardrailRejectedError as _RelayGuardrailRejectedError
except ImportError:  # pragma: no cover - exercised by the legacy SDK matrix.
    _RelayGuardrailRejectedError = RuntimeError


class PolicyRejectedError(_RelayGuardrailRejectedError):
    """Reject guarded work using the strongest error type the host supports."""
