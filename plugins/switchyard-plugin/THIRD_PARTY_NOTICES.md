<!-- SPDX-License-Identifier: Apache-2.0 -->
# Switchyard plugin third-party notices

This plugin is built from [NVIDIA-NeMo/Switchyard](https://github.com/NVIDIA-NeMo/Switchyard)
at commit `8dc891195a5fa71350f5a03c19f9eecc0f9fcb09`, using the
`crates/switchyard-nemo-relay-plugin` crate and its workspace dependencies.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Switchyard is licensed under Apache-2.0. The bundle includes the upstream `LICENSE`
and `NOTICE` unchanged, plus the license and notice files for its linked `protocol`
and `switchyard-translation` crates under `notices/`.

[ATTRIBUTIONS-Rust.md](ATTRIBUTIONS-Rust.md) reproduces dependency licenses for the
entire pinned Rust workspace lockfile, including build, test, optional, and
platform-specific dependencies. It supplements the upstream notice; the upstream
notice's Python dependency list does not describe this native plugin's Rust dependencies.

The published `valuable` 0.1.1 archive omits its license file. Its license is copied
from [the archive's recorded upstream revision](https://github.com/tokio-rs/valuable/blob/9efc29b6e58cef28f6566a47aa7e142a55fead77/LICENSE)
and included in the attribution file.

Refer to the upstream repository for release documentation. Update these notices
and regenerate attributions whenever the registered source revision changes.
