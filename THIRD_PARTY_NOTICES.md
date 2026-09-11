<!-- SPDX-License-Identifier: Apache-2.0 -->
# Third-party notices

NVIDIA NeMo Relay Plugins uses the [Apache-2.0 license](LICENSE). Packages from
other projects keep their own copyrights, license terms, and notices.

## Package credits and licenses

These attribution files follow NeMo Relay's format. Each file lists exact
package versions and their license text. The lists come from each project's
lockfile and include build, test, optional, and platform-specific packages.
A listed package may not be part of every platform's binary. Switchyard's list
covers its full Rust workspace at the saved commit.

| Project | Attribution file |
| --- | --- |
| All Python dependencies (plugins and tooling) | [ATTRIBUTIONS-Python.md](ATTRIBUTIONS-Python.md) |
| All Rust dependencies (all plugins) | [ATTRIBUTIONS-Rust.md](ATTRIBUTIONS-Rust.md) |
| Repository Python tooling | [ATTRIBUTIONS-Python.md](scripts/licensing/ATTRIBUTIONS-Python.md) |
| Python worker example | [ATTRIBUTIONS-Python.md](plugins/example-python-grpc-worker-plugin/ATTRIBUTIONS-Python.md) |
| Rust worker example | [ATTRIBUTIONS-Rust.md](plugins/example-rust-grpc-worker-plugin/ATTRIBUTIONS-Rust.md) |
| Rust native example | [ATTRIBUTIONS-Rust.md](plugins/example-rust-native-plugin/ATTRIBUTIONS-Rust.md) |
| Switchyard plugin | [ATTRIBUTIONS-Rust.md](plugins/switchyard-plugin/ATTRIBUTIONS-Rust.md) |

## Copied and remote source code

- The three examples come from [NVIDIA NeMo Relay 0.8.4](https://github.com/NVIDIA/NeMo-Relay/tree/6f0caf7aefc1363943badf51a643d4c5e52f020a/examples), under Apache-2.0. Each example keeps its license. Its `UPSTREAM.md` and `NOTICE` files record the original source and local changes.
- The license tools and their settings come from NVIDIA NeMo Relay, under Apache-2.0. See [the tools' source details](scripts/licensing/UPSTREAM.md).
- Switchyard is built from a saved source commit, under Apache-2.0. See [Switchyard's notices](plugins/switchyard-plugin/THIRD_PARTY_NOTICES.md). Its bundle keeps the original license and notice files.

If a package archive has no license text, the generator gets it from the source
project. Rust archives record the commit used to publish the package. For Python,
the generator finds the commit for the matching version tag. Each generated entry
links to that exact source commit.

There are no separate package license copies to maintain in this repository.
The generator fails if it cannot find the source or license text.

See [the update instructions](scripts/licensing/README.md) when changing locks or sources.
