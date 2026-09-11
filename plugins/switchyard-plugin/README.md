<!-- SPDX-License-Identifier: Apache-2.0 -->
# Switchyard plugin

This folder builds and releases the [Switchyard routing plugin](https://github.com/NVIDIA-NeMo/Switchyard/tree/main/crates/switchyard-nemo-relay-plugin). Its code lives in the Switchyard repository. The `release.toml` file sets the exact source commit and selects the plugin’s Rust package, called a crate. The build downloads the full Switchyard workspace so the crate can use other packages there.

From this repository's root:

```sh
uv run --locked python -m scripts.plugins run switchyard-plugin
```

This command builds the plugin, runs Switchyard’s plugin tests, and uses its package script to create a bundle. It then extracts and installs the bundle for tests. These tests send a routed request to a local server, check shutdown and removal, and confirm that Relay rejects changed files. CI requires tests to pass on all five supported platforms.

To use newer Switchyard code, update `source.sha` to the commit you want. `source.ref` records the branch or tag you plan to track; it does not update the source on its own.

The release version here is separate from the Switchyard workspace version. SDK package versions come from Switchyard’s Cargo lockfile. A `[relay]` setting changes only the Relay host used for tests.

Bundles include Switchyard’s `LICENSE` and `NOTICE`, plus `ATTRIBUTIONS-Rust.md` for its Rust packages. See the upstream documentation for deployment settings and supported features.
