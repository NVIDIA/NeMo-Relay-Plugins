<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust Native Dynamic Plugin

This plugin’s release name is `example-rust-native-plugin`.
[`release.toml`](release.toml) defines its build and release settings.
`relay-plugin.toml` tells Relay how to load the plugin.

From the repository root, use this command to build and test the plugin, create
a bundle, and check that the installed bundle works:

```sh
uv run --locked python -m scripts.plugins run example-rust-native-plugin
```

See [UPSTREAM.md](UPSTREAM.md) for the original source commit.
Use [the release guide](../../RELEASE.md) to install a bundle and set Relay’s trust rules.

This is the native plugin from the authoring guide. Relay loads it as a shared
library inside the host process. Separate source modules handle settings,
event tracking, request rules, execution wrappers, and runtime helpers.

The example uses these hooks from the typed 0.8.0 SDK:

- One subscriber to receive events.
- Three event sanitizers to clean event data.
- Five hooks for tool calls.
- Six hooks for large language model (LLM) calls.

Run the tests and build the shared library from this folder. The settings tests
check valid values and schema rules. The lifecycle test checks the plugin from
start to finish. It builds a fresh `cdylib` (a Rust shared library), writes a
manifest with its hash, and loads it into a host. The test runs request handlers
and checks an event mark. It then clears the callbacks before unloading the library:

```bash
cargo test --release
cargo build --release
```

Copy `relay-plugin.toml` to `relay-plugin.local.toml`. Replace
`<platform-library-file>` with the path to the built library:

| Platform | Library Path |
|---|---|
| macOS | `target/release/libnemo_relay_rust_native_plugin_example.dylib` |
| Linux | `target/release/libnemo_relay_rust_native_plugin_example.so` |
| Windows | `target/release/nemo_relay_rust_native_plugin_example.dll` |

Get the file’s SHA-256 hash with `shasum -a 256`, `sha256sum`, or
`Get-FileHash -Algorithm SHA256`. Use that hash to replace `<artifact-sha256>`,
keeping the `sha256:` prefix. Set `source.artifact` and `load.library` to the
same library path, relative to the manifest.

The schema lists the allowed settings for each feature. It also includes
`executor.worker_threads`, which sets the worker thread count used by the SDK.

The optional `registration_control` setting adds callbacks that can block or
allow selected middleware while the plugin is active. Middleware handles
requests or events as they pass through Relay. These are the defaults:

- `enabled: false`
- `kinds: ["subscriber"]`
- `registration_name: "documentation-controlled-subscriber"`
- `allowed_registration_name: "documentation-observed-subscriber"`
- `reason: "disabled by documentation plugin"`

When enabled, one callback blocks `registration_name` and returns the reason.
Another returns `None` to allow `allowed_registration_name`. This shows both
decisions in one plugin run. The kinds, target names, and reason must not be
empty. The two target names must differ when this control is enabled.

Read [Conditional Middleware Guardrails](https://github.com/NVIDIA/NeMo-Relay/blob/0.8.4/docs/about-nemo-relay/concepts/conditional-middleware-guardrails.mdx)
before enabling the control for a target found at runtime.
