<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Source of the license tools

`attributions_lockfile_md.py`, `license_diff.py`, and `about.toml` are based on
[NVIDIA NeMo Relay](https://github.com/NVIDIA/NeMo-Relay/tree/231da6c643fe5c4ffae91e2eb49715262932a127).
Their original paths are `scripts/licensing/attributions_lockfile_md.py`,
`scripts/licensing/license_diff.py`, and `about.toml`.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under Apache-2.0; see the repository's `LICENSE`.

The local versions keep the Python and Rust attribution formats. They remove
unused Node support and accept paths for each project's lockfiles and output.
Cargo commands keep locked package versions. If a package has no license text,
the tools fetch it from an exact upstream commit. Missing license text causes
an error.

The repository scripts combine package lists from all plugins. The license diff
also checks remote workspaces at their saved source commits.
