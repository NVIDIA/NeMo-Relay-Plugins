<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Switchyard plugin third-party notices

This plugin is built from [NVIDIA-NeMo/Switchyard](https://github.com/NVIDIA-NeMo/Switchyard)
at commit `d1454f304e68b082684b19c6d217231d823f81ae`, using the
`crates/switchyard-nemo-relay-plugin` Rust package and the other workspace packages it uses.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Switchyard uses Apache-2.0. The bundle keeps its original `LICENSE` and `NOTICE`
files. It also includes the licenses and notices for the `protocol` and
`switchyard-translation` crates in the `notices/` folder.

The generated `ATTRIBUTIONS-Rust.md` in the bundle contains license text for the Rust
packages in the full workspace lockfile. It includes build, test, optional, and
platform-specific packages. This file adds to the original notice. That notice
lists Python packages, not the Rust packages used by this native plugin.

The release uses Switchyard's workspace manifest and lockfile. They select the
published NeMo Relay 0.10.0 SDK from crates.io. The generated attributions
include this SDK and its dependencies.

The published `valuable` 0.1.1 archive has no license file. The generator fetches its license
from [the archive's recorded upstream revision](https://github.com/tokio-rs/valuable/blob/9efc29b6e58cef28f6566a47aa7e142a55fead77/LICENSE)
and adds it to the attribution file.

See the source repository for release documentation. Update these notices and
the root aggregate attribution files whenever you change the source commit in
`release.toml`. Packaging generates the bundle's attribution file from that commit.
