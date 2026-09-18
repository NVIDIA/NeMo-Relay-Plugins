<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Original source

This example was synced from NVIDIA NeMo Relay's `release/0.9` branch at commit
`073c913f47d4c8ed10f3af5bf781ae58f6dee20a`.

[View the original source](https://github.com/NVIDIA/NeMo-Relay/tree/073c913f47d4c8ed10f3af5bf781ae58f6dee20a/examples/rust-native-plugin).

The release name is specific to this repository. The original package name and
the plugin ID used by Relay have not changed.

This source needs Relay 0.9. Its SDK packages are not published yet, so the
package files and lockfiles use the exact Git commit above. The `release.toml`
file selects the same commit for the test host. These are separate settings;
changing the host does not update the SDK lock.

Local changes keep Rust builds in release mode and save each plugin's Cargo
lockfile. They also add this repository's packaging and bundle tests.
The README uses plain language and links to the build and release guides here.
Pre-commit formats the copied source using this repository's style.
