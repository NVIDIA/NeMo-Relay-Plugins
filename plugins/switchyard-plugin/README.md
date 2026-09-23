<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Switchyard plugin

## Summary

This plugin sends supported model requests through routes in a
[Switchyard](https://github.com/NVIDIA-NeMo/Switchyard/tree/v0.3.0/crates/switchyard-nemo-relay-plugin)
deployment. It uses Switchyard's targets, client pools, routing algorithms,
retry policies, and route validation. Relay loads the Rust shared library
directly into its own process.

**Load and enable.** The bundled `relay-plugin.toml` registers
`nvidia.switchyard`. The manifest starts disabled. Register the manifest, add
a Switchyard deployment and Relay trust policy to the plugin configuration,
then enable it. Relay loads the library the next time it starts:

```sh
nemo-relay plugins add --user ./switchyard-plugin/relay-plugin.toml
nemo-relay plugins enable nvidia.switchyard
```

Relay cannot enable this plugin without a valid deployment. See the
[upstream plugin documentation](https://github.com/NVIDIA-NeMo/Switchyard/tree/v0.3.0/crates/switchyard-nemo-relay-plugin#configure-relay)
for configuration fields and examples.

**Distributed artifacts.** A release provides one `.tar.gz` archive for each
supported Linux or macOS platform and one `.zip` archive for each supported
Windows platform. Each archive has a `.sha256` checksum and a `.json`
build-metadata sidecar. The archive contains the platform shared library, the
completed runtime manifest, the configuration schema, license notices, Rust
dependency attributions, and third-party notices for linked Switchyard crates.

## Build and test

This folder builds and releases the Switchyard routing plugin. Its code lives
in the Switchyard repository. The `release.toml` file sets the exact source
commit and selects the plugin's Rust package, called a crate. The build
downloads the full Switchyard workspace so the crate can use other packages
there.

From this repository's root:

```sh
uv run --locked python -m scripts.plugins run switchyard-plugin
```

This command builds the plugin, runs Switchyard’s plugin tests, and uses its package script to create a bundle. It then extracts and installs the bundle for tests. These tests send a routed request to a local server, check shutdown and removal, and confirm that Relay rejects changed files. CI requires tests to pass on all five supported platforms.

To use newer Switchyard code, update `source.sha` to the commit you want. Set
`source.ref` to a branch name or to `refs/tags/<tag>`. Planning checks that a
tag resolves to the pinned commit. The ref records what to track; it does not
update the source on its own.

The release version here is separate from the Switchyard workspace version.
SDK package versions come from Switchyard's Cargo lockfile. A `[relay]` setting
changes only the Relay host used for tests.

See the upstream documentation for deployment settings and supported features.
