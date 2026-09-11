<!-- SPDX-License-Identifier: Apache-2.0 -->
# Attribution tooling provenance

`attributions_lockfile_md.py`, `license_diff.py`, and `about.toml` are adapted from
[NVIDIA NeMo Relay](https://github.com/NVIDIA/NeMo-Relay/tree/231da6c643fe5c4ffae91e2eb49715262932a127),
specifically `scripts/licensing/attributions_lockfile_md.py`,
`scripts/licensing/license_diff.py`, and `about.toml`.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under Apache-2.0; see the repository's `LICENSE`.

Local adaptations retain the Python and Rust attribution formats, remove unused
Node support, accept project/output paths for independent lockfiles, explicitly
lock Cargo operations, and fetch missing license text programmatically from exact upstream revisions.
Missing license text fails generation. Repository orchestration aggregates all
plugin locks, and the adapted diff collector includes pinned remote workspaces.
