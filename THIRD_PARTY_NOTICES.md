<!-- SPDX-License-Identifier: Apache-2.0 -->
# Third-party notices

NVIDIA NeMo Relay Plugins is licensed under [Apache-2.0](LICENSE). Third-party
components retain their respective copyrights, license terms, and notices.

## Dependency attributions

These inventories follow NeMo Relay's attribution format and include exact locked
versions and license text. They cover the dependencies in each project's lockfile,
including development, build, optional, and platform-specific dependencies. An
entry does not imply that the dependency is included in every platform's binary.
Switchyard's inventory covers its full pinned Rust workspace.

| Project | Attribution file |
| --- | --- |
| All Python dependencies (plugins and tooling) | [ATTRIBUTIONS-Python.md](ATTRIBUTIONS-Python.md) |
| All Rust dependencies (all plugins) | [ATTRIBUTIONS-Rust.md](ATTRIBUTIONS-Rust.md) |
| Repository Python tooling | [ATTRIBUTIONS-Python.md](scripts/licensing/ATTRIBUTIONS-Python.md) |
| Python worker example | [ATTRIBUTIONS-Python.md](plugins/example-python-grpc-worker-plugin/ATTRIBUTIONS-Python.md) |
| Rust worker example | [ATTRIBUTIONS-Rust.md](plugins/example-rust-grpc-worker-plugin/ATTRIBUTIONS-Rust.md) |
| Rust native example | [ATTRIBUTIONS-Rust.md](plugins/example-rust-native-plugin/ATTRIBUTIONS-Rust.md) |
| Switchyard plugin | [ATTRIBUTIONS-Rust.md](plugins/switchyard-plugin/ATTRIBUTIONS-Rust.md) |

## Adapted and remote sources

- The three example plugins were adapted from [NVIDIA NeMo Relay 0.8.4](https://github.com/NVIDIA/NeMo-Relay/tree/6f0caf7aefc1363943badf51a643d4c5e52f020a/examples), licensed under Apache-2.0. Each example preserves its license and records its origin and adaptations in `UPSTREAM.md` and `NOTICE`.
- The attribution generator and configuration were adapted from NVIDIA NeMo Relay under Apache-2.0. See [tooling provenance](scripts/licensing/UPSTREAM.md).
- Switchyard is built from an independently pinned upstream checkout under Apache-2.0. See [Switchyard's notices](plugins/switchyard-plugin/THIRD_PARTY_NOTICES.md). Its bundle preserves the upstream license and notice files.

When a published archive omits license text, the generator obtains it from an
exact upstream revision identified by package metadata: the publication commit
recorded in Rust archives, or a matching Python version tag resolved to its commit.
The generated entry links to that immutable source. No per-package license copies
are maintained in this repository. Missing or unresolvable licenses fail generation.

See [regeneration instructions](scripts/licensing/README.md) when updating locks or sources.
