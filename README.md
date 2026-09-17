<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Plugins

This repository builds and releases plugins for
[NVIDIA NeMo Relay](https://github.com/NVIDIA/NeMo-Relay). Each folder under
`plugins/` is an independent plugin with its own version, supported platforms,
dependencies, tests, and release artifacts.

This project is currently not accepting contributions.

## Plugins

| Plugin | What it does | How Relay loads it | Relay plugin ID |
| --- | --- | --- | --- |
| [Switchyard](plugins/switchyard-plugin) | Routes supported model requests through a Switchyard deployment. | Loads a Rust shared library into the Relay process. | `nvidia.switchyard` |
| [Python gRPC worker example](plugins/example-python-grpc-worker-plugin) | Demonstrates settings, request and event hooks, streaming, state cleanup, and middleware controls in Python. | Creates a Python environment and starts a separate `grpc-v1` worker process. | `examples.python_grpc_worker` |
| [Rust gRPC worker example](plugins/example-rust-grpc-worker-plugin) | Demonstrates the same worker-plugin lifecycle in Rust, including request, stream, event, and state hooks. | Starts a separate Rust `grpc-v1` worker executable. | `examples.rust_grpc_worker` |
| [Rust native example](plugins/example-rust-native-plugin) | Demonstrates typed event, tool-call, and LLM-call hooks in a native Rust plugin. | Loads a Rust shared library into the Relay process. | `examples.rust_native_policy` |

A **worker** is a separate process. A **native plugin** is a library loaded into
Relay itself. Switchyard supports all five platforms listed below. The Python
worker does not support Windows ARM64; the other examples support all five.

- `linux-x86_64`
- `linux-arm64`
- `windows-x86_64`
- `windows-arm64`
- `macos-arm64`

## Install and enable a plugin

Extract a release archive to a permanent folder, then register its runtime
manifest with Relay:

```sh
nemo-relay plugins validate ./<bundle>/relay-plugin.toml
nemo-relay plugins add --user ./<bundle>/relay-plugin.toml
nemo-relay plugins enable <plugin-id>
```

Every supplied manifest sets `enabled = false`, so registering a plugin does
not activate it. Configure the plugin and Relay's trust policy before enabling
it. Native plugins run inside Relay and should only be loaded from a trusted
bundle. Switchyard also requires a deployment configuration. See the plugin's
README and the [release guide](RELEASE.md) for the exact setup.

## Distributed artifacts

Each plugin release contains one platform-specific archive: `.tar.gz` on Linux
and macOS, or `.zip` on Windows. Its name is
`<name>-<version>-<platform>.<format>`. Every archive has two sidecar files:

- `.sha256` records the archive checksum.
- `.json` records the source commit, tested Relay commit, tool versions,
  platform, archive hash, and verification result.

Every archive contains a completed `relay-plugin.toml`, the runtime artifact,
`config.schema.json`, license notices, and generated dependency attributions.
The runtime artifact depends on the plugin:

| Plugin | Runtime and additional bundle contents |
| --- | --- |
| Switchyard | Platform shared library and Switchyard third-party notices. |
| Python gRPC worker | Python source package, `pyproject.toml`, and `uv.lock`. |
| Rust gRPC worker | Platform worker executable. |
| Rust native example | Platform shared library. |

Relay verifies the runtime artifact's SHA-256 hash from `relay-plugin.toml`
before loading it. Bundles are integrity checked but are not digitally signed.

## Local development

Install Git, Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and the
[GitHub CLI](https://cli.github.com/). Rust plugins also require the Rust
version in their `release.toml` and `cargo-about`.

From the repository root:

```sh
uv sync --locked --group test
rustup toolchain install 1.96.1 --profile minimal --component rustfmt
cargo +1.96.1 install cargo-about --version 0.9.1 --locked --features cli
uv run --locked --group test pre-commit install
uv run --locked --group test pre-commit run --all-files
uv run --locked --group test pytest
uv run --locked python -m scripts.plugins validate
uv run --locked python -m scripts.plugins list
uv run --locked python -m scripts.plugins run example-rust-native-plugin
```

The `run` command builds and tests one plugin in release mode, creates its
bundle, installs the extracted bundle, and performs a smoke test. Results go to
`dist/<name>/<platform>/`; downloads and build caches go to `.cache/`. It can
only test the current machine's platform.

Before pushing a plugin change, run:

```sh
uv run --locked python -m scripts.plugins run <name>
```

## Plugin manifests and releases

`plugins/<name>/release.toml` tells this repository how to obtain, build, test,
package, and release a plugin. The bundled `relay-plugin.toml` tells Relay how
to validate and load it. See [the release schema](schemas/release.schema.json)
for all manifest fields and [package sources](docs/package-sources.md) for
published Python wheels and Rust crates.

Source, package locks, build commands, smoke tests, and output names belong in
the plugin's folder. Shared scripts must not choose behavior from a plugin name.
Rust builds and tests must use release mode, and every declared platform must
be tested.

See [RELEASE.md](RELEASE.md) to publish one plugin, [SECURITY.md](SECURITY.md)
to report a security issue, and [the licensing guide](scripts/licensing/README.md)
when dependencies or remote sources change. Source code uses the
[Apache-2.0 license](LICENSE), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
links to dependency license information.
