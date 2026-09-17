# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-specific validation and projection helpers.

The public worker-facing projection API remains in :mod:`codec_projection`.
These modules keep wire-protocol details isolated so extending one provider
does not change the policy orchestration shared by every codec.
"""
