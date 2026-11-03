<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Guardrails plugin third-party notices

This plugin is built from [NVIDIA-NeMo/Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)
at commit `91bc4419d87a9673ed7d5bbc6c820d0149585427`, using the
`crates/nemo-guardrails-relay-plugin` Rust package.

Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NeMo Guardrails uses Apache-2.0. The bundle keeps its original `LICENSE.md` and
`LICENSE-Apache-2.0.txt` files. The plugin is the only package in the NeMo
Guardrails Rust workspace, so the bundle has no other workspace notices.

The generated `ATTRIBUTIONS-Rust.md` in the bundle contains license text for the Rust
packages in the full workspace lockfile. It includes build, test, optional, and
platform-specific packages. This file adds to the original license files. NeMo
Guardrails' `LICENCES-3rd-party` file covers other code in that repository, not the
Rust packages used by this native plugin.

The release uses NeMo Guardrails' workspace manifest and lockfile. They select the
published NeMo Relay 0.11.0 SDK from crates.io. The generated attributions
include this SDK and its dependencies.

The published `valuable` 0.1.1 archive has no license file. The generator fetches its license
from [the archive's recorded upstream revision](https://github.com/tokio-rs/valuable/blob/9efc29b6e58cef28f6566a47aa7e142a55fead77/LICENSE)
and adds it to the attribution file.

See the source repository for release documentation. Update these notices and
the root aggregate attribution files whenever you change the source commit in
`release.toml`. Packaging generates the bundle's attribution file from that commit.
