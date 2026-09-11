<!-- SPDX-License-Identifier: Apache-2.0 -->
# NeMo Relay Plugins

Independently built and released plugins for [NVIDIA NeMo Relay](https://github.com/NVIDIA/NeMo-Relay). Each directory under `plugins/` owns its release version, source selection, supported platforms, dependencies, tests, and packaging commands.

This project is currently not accepting contributions.

The five platform identifiers are `linux-x86_64`, `linux-arm64`, `windows-x86_64`, `windows-arm64`, and `macos-arm64`. macOS x86_64 is not included. Python workers explicitly exclude Windows ARM64 because their gRPC dependency does not provide a supported wheel there. CI requires every declared combination, totaling 19 jobs for a full build.

## Official plugins

| Plugin | Type | Source | Supported platforms |
| --- | --- | --- | --- |
| [switchyard-plugin](plugins/switchyard-plugin) | Native | Remote Switchyard workspace | All five |

## Example plugins

| Plugin | Type | Source | Supported platforms |
| --- | --- | --- | --- |
| [example-python-grpc-worker-plugin](plugins/example-python-grpc-worker-plugin) | Worker | In-tree Python | Linux x86_64/ARM64, Windows x86_64, macOS ARM64 |
| [example-rust-grpc-worker-plugin](plugins/example-rust-grpc-worker-plugin) | Worker | In-tree Rust | All five |
| [example-rust-native-plugin](plugins/example-rust-native-plugin) | Native | In-tree Rust | All five |

## Local development

Install Git, Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and the [GitHub CLI](https://cli.github.com/). Rust plugins also require rustup and the manifest's Rust toolchain. Building a Relay host from source requires the toolchain specified by that Relay revision. GitHub access is needed for host resolution/downloads and remote plugins.

From the repository root:

```sh
uv sync --locked --group test
uv run --locked --group test pytest
uv run --locked python -m scripts.plugins validate
uv run --locked python -m scripts.plugins list
rustup toolchain install 1.96.1 --profile minimal
uv run --locked python -m scripts.plugins run example-rust-native-plugin
```

`run` detects the local platform and executes build, plugin tests, packaging, and a smoke test against an extracted bundle. It refuses cross-platform execution and unsupported targets. Output appears under `dist/<name>/<platform>/`. Temporary source/host checkouts and compilation caches live under `.cache/`.

SDK versions are controlled by each plugin's package manifest and lockfile. The test host defaults to the latest published stable Relay release, resolved once for the CI run. Override only the host in a plugin's `release.toml` when needed:

```toml
[relay]
tag = "0.8.4"
# Alternatively use sha = "<full 40-character Relay commit SHA>".
```

Tag overrides must name Relay tags. A SHA selects a source build. A tag without downloadable binaries also selects a source build. An incompatible host fails the tests; selection does not silently fall back to an older release or change SDK dependencies.

## Release manifests

`plugins/<name>/release.toml` describes repository builds and releases. `relay-plugin.toml` describes how Relay loads a packaged plugin. Release names, language package names, and runtime IDs are separate; the release name must match its folder.

The machine-readable contract is [schemas/release.schema.json](schemas/release.schema.json). The Python tooling validates TOML against that schema and emits JSON for Actions.

```toml
schema_version = 1
name = "my-plugin"
version = "0.1.0"
type = "native" # or "worker"
# Omit platforms to select all five.
platforms = ["linux-x86_64", "linux-arm64"]

[metadata]
description = "My Relay plugin"
license = "Apache-2.0"
authors = ["Plugin maintainers"]
# Optional repository, documentation, and arbitrary additional metadata.

[source]
location = "in-tree"
path = "."

[toolchains]
python = "3.11"
rust = "1.96.1" # optional for non-Rust plugins

[commands.build]
argv = ["${PYTHON}", "${PLUGIN_DIR}/tasks.py", "build"]
cwd = "source"

[commands.test]
argv = ["${PYTHON}", "${PLUGIN_DIR}/tasks.py", "test"]
cwd = "source"

[commands.package]
argv = ["${PYTHON}", "${PLUGIN_DIR}/tasks.py", "package"]
cwd = "source"

[commands.smoke]
argv = ["${PYTHON}", "${PLUGIN_DIR}/smoke_test.py"]
cwd = "plugin"

[artifacts]
bundle = "bundle"
manifest = "relay-plugin.toml"
```

Commands are argument arrays, not shell expressions. `${VARIABLE}` substitution happens within individual arguments. `cwd = "source"` means the full source checkout root (the plugin source directory for in-tree entries); `cwd = "plugin"` means this repository's registration folder. For remote entries, `SOURCE_DIR` identifies the selected subdirectory within `SOURCE_ROOT`.

| Variable | Value |
| --- | --- |
| `REPO_DIR`, `PLUGIN_DIR` | This repository and the plugin's registration folder |
| `SOURCE_ROOT`, `SOURCE_DIR` | Source checkout root and selected source subdirectory |
| `OUTPUT_DIR` | Fresh per-plugin/platform working output directory |
| `PLUGIN_PLATFORM` | Canonical platform identifier |
| `PYTHON` | Tooling Python executable, with repository tooling dependencies |
| `PLUGIN_PYTHON` | Python interpreter selected by the plugin toolchain |
| `RELAY_BIN` | Resolved test-host executable |
| `CARGO_TARGET_DIR` | Persistent plugin/platform Rust compilation cache |
| `BUNDLE_DIR` | Extracted archive root, available during smoke tests |

The package command must create `artifacts.bundle` relative to `OUTPUT_DIR`. The runtime manifest path is relative to that bundle. The runner verifies integrity and archives it, then invokes the smoke command against `BUNDLE_DIR`. Smoke tests must use that extracted distribution, not source/build outputs. A failure in any stage prevents artifact delivery.

Each plugin owns its command implementations. The initial registrations use `tasks.py` for build, test, and packaging, and `smoke_test.py` for their installed-bundle scenarios. Cargo package names, artifact filenames, runtime files, configuration, requests, and behavior assertions belong in those plugin folders. Switchyard's registration invokes its upstream packager from its pinned workspace.

Shared helpers are optional: `scripts/tasks.py` provides command execution, environment context, platform filename conventions, and manifest digest writing. `scripts/smoke.py` provides a JSON HTTP fixture and the `installed_gateway` context manager for installation, activation, shutdown, tamper rejection, and removal. A plugin supplies its own configuration and gateway arguments, then exercises the yielded gateway URL. Commands may instead use any scripts or tools that satisfy the manifest contract; shared code has no plugin-name dispatch or central recipe registry.

## Adding or updating a plugin

1. Add `plugins/<name>/release.toml`, source or remote reference, documentation, task scripts, and tests. Adding an unrelated plugin requires no changes to shared orchestration, no root Cargo workspace, and no central plugin registry.
2. Choose `worker` or `native` and declare any platform restrictions. Keep all declared platforms mandatory.
3. Lock SDK/runtime dependencies independently. Preserve upstream licensing and attribution.
4. Implement configuration/behavior tests and an installed-bundle test that proves activation, a representative managed request, shutdown/removal, and integrity rejection. The examples use a loopback HTTP provider without credentials.
5. Run manifest validation, shared tooling tests, and the local plugin pipeline. CI verifies the remaining platforms.

Remote manifests use this source shape:

```toml
[source]
location = "remote"
repository = "https://github.com/NVIDIA-NeMo/Switchyard"
ref = "main"
sha = "8dc891195a5fa71350f5a03c19f9eecc0f9fcb09"
path = "crates/switchyard-nemo-relay-plugin"
```

`sha` is mandatory and authoritative. `ref` records the branch/tag/ref being tracked; CI does not resolve it to newer code. Update the SHA through a reviewed manifest change and test it before releasing. The entire remote repository is fetched so sibling workspace dependencies remain available. The initial remote integration uses a publicly readable repository; private upstream access requires separately provisioned read credentials.

## CI

External GitHub Actions use full commit SHAs for their latest stable releases, with exact version tags in comments. When updating an action, resolve the newest stable release to its commit SHA. CI and the initial plugin manifests use Python 3.11, NeMo Relay’s minimum supported version; plugins can select a newer Python when their dependencies require it.

PRs validate every manifest and run shared tooling tests. Branch pushes do not trigger separate CI runs; tag pushes trigger plugin release validation. Plugin-local changes select that plugin. Changes to shared scripts, tests, schemas, workflow code, root dependency locks, or licensing select all plugins. Root documentation alone selects none. Missing comparison history conservatively selects all plugins; renames and deletions are included.

Every selected plugin builds, tests, packages, and installs on all declared native platforms. Configure branch protection to require **Plugin checks**, the stable aggregate check. Build jobs have read-only permissions. Only the tag release job can write release data.

See [RELEASE.md](RELEASE.md) for single-plugin tagging and draft publication. Report security issues using [SECURITY.md](SECURITY.md). Source code is licensed under [Apache-2.0](LICENSE); bundles retain their applicable upstream notices.

## Third-party notices and attribution

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) identifies copied sources and the
attribution inventory for each independently locked project. The attribution files
use NeMo Relay's format, including dependency versions and full license text.
The root Python and Rust attribution files aggregate all plugins and repository
tooling, retaining each distinct dependency version.
Plugin bundles include their license, notices, and dependency attributions.

When updating a dependency lockfile or remote source revision, regenerate the
plugin and aggregate attribution files using [the licensing instructions](scripts/licensing/README.md).
