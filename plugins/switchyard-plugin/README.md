<!-- SPDX-License-Identifier: Apache-2.0 -->
# Switchyard plugin

This folder independently builds and releases the [Switchyard routing plugin](https://github.com/NVIDIA-NeMo/Switchyard/tree/main/crates/switchyard-nemo-relay-plugin). The implementation remains in Switchyard; `release.toml` pins its complete source workspace to a commit and selects the plugin crate.

From this repository's root:

```sh
uv run --locked python -m scripts.plugins run switchyard-plugin
```

The pipeline runs the upstream plugin tests and packager, then verifies installation, route execution against a loopback provider, shutdown/removal, and integrity rejection using the extracted bundle. All five repository platforms are required.

Update `source.sha` deliberately when adopting upstream changes. `source.ref` records tracking intent and never advances the checkout automatically. The release version is maintained here independently of the upstream workspace version. SDK dependencies use the upstream Cargo lockfile; `[relay]` can override only the test host.

Bundles preserve Switchyard's `LICENSE` and `NOTICE`. Consult the upstream documentation for deployment configuration and supported behavior.
