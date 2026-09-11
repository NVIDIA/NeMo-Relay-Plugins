<!-- SPDX-License-Identifier: Apache-2.0 -->
# NeMo Relay Plugins

This repository contains plugins for [NVIDIA NeMo Relay](https://github.com/NVIDIA/NeMo-Relay). You can build and release each plugin on its own. Each folder under `plugins/` defines the plugin’s version, source code, supported platforms, required packages, tests, and build steps.

This project is currently not accepting contributions.

The supported platforms are `linux-x86_64`, `linux-arm64`, `windows-x86_64`, `windows-arm64`, and `macos-arm64`. macOS x86_64 is not supported. The Python worker does not support Windows ARM64 because its gRPC package has no supported wheel (a prebuilt Python package) for that platform. Automated checks run for every supported plugin and platform pair: 19 jobs for a full build.

A **worker** runs as a separate process. A **native plugin** loads a library into the Relay process. **In-tree** source code lives in this repository. **Remote** source code lives in another repository.

Plugins can also use published Python wheels or Rust crates. See
[package sources](docs/package-sources.md) for `location = "wheel"` and
`location = "crate"`, locked downloads, packaging, and tests.

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

Install Git, Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and the [GitHub CLI](https://cli.github.com/). For Rust plugins, also install rustup and the Rust version listed in the plugin’s `release.toml` file. These language tools are called a **toolchain**.

Tests use a Relay executable called the **test host**. You need GitHub access to find and download the host and any remote plugin sources. If the host must be built from source, use the Rust version required by that Relay commit.

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

`run` detects your platform, builds the plugin, runs its tests, and creates a bundle. A **bundle** is an archive with the files needed to install the plugin. The command then extracts and installs the bundle for a **smoke test**, which checks basic behavior. It only runs for supported platforms and cannot test a different platform from your own.

Build results go under `dist/<name>/<platform>/`. Downloaded source code and saved build files go under `.cache/`.

### Checks before each commit

The `pre-commit install` command adds a **Git hook**: a check that runs when you
commit. Run this command once in each clone. The hook uses the versions saved in
`uv.lock`. It also installs a pinned version of actionlint, which checks GitHub
Actions files. The first run needs network access and may take a few minutes.

The hooks format Python and Rust, check file syntax, check package lockfiles,
validate release manifests, and run the shared tests. When formatting changes a
source artifact, they update its hash in the runtime manifest. Each commit also
rebuilds the root attribution files from all plugin locks, including after file deletions.
This part needs network access and can take a few minutes.
These checks need the Rust tools installed above. They do not edit remote source
code or license text.

Formatting checks cover all tracked Python files and all in-tree Rust plugins
on each commit. This also catches changes caused by new formatter settings or
tool versions. Untracked Python files are left alone.
Workflow checks also run on each commit, so changes to actionlint settings are covered.

If a hook changes files, review those changes, stage them, and commit again.
To run every hook yourself, use:

```sh
uv run --locked --group test pre-commit run --all-files --show-diff-on-failure
```

If a package lockfile is out of date, run `uv lock` or `cargo update` in the
folder that owns it. Review the package changes before staging the lockfile.
Use `cargo update -p <package>` to update one Rust package.

CI runs the same hooks on all tracked files. Plugin builds and installed-bundle
tests still run in the separate platform jobs. Before pushing a plugin change,
run its full local pipeline with `uv run --locked python -m scripts.plugins run <name>`.

Each plugin’s package file and **lockfile** set its SDK versions. The SDK provides tools for writing plugins; the lockfile records exact package versions.

Tests use the latest stable Relay release by default. The three examples now need the unpublished Relay 0.9 API, so their manifests select a host built from source. Their SDK locks also use an exact upstream Git commit. See each example’s `UPSTREAM.md` for that commit.

CI, the automated checks in GitHub Actions, selects the host once per run. To choose a different test host, add this to the plugin’s `release.toml`:

```toml
[relay]
sha = "65eb82bf3d788986512246abf7f8ab7a520f28d9"
# Or use tag = "<a compatible published Relay tag>".
```

The `tag` value must name a Relay tag. A **SHA** is the full ID of a Git commit. Using a SHA makes CI build the host from source. CI also builds from source if a tag has no suitable download. If the host does not work with the plugin, tests fail. CI does not switch to an older host or change the SDK packages.

## Release manifests

A **manifest** is a file that describes a plugin. `plugins/<name>/release.toml` tells the build scripts how to build and release it. `relay-plugin.toml` tells Relay how to load it. The release name must match the plugin’s folder. Its Python or Rust package name and the ID used inside Relay may differ.

[schemas/release.schema.json](schemas/release.schema.json) defines the allowed fields and values. The Python scripts check each TOML file against these rules and produce JSON for GitHub Actions.

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

Write each command as a list of arguments. The scripts replace `${VARIABLE}` with its value inside each argument. Commands do not run through a shell.

`cwd` sets the folder where a command runs. `cwd = "source"` uses the source root: the full remote checkout or the in-tree plugin’s source folder. `cwd = "plugin"` uses this repository’s `plugins/<name>/` folder. For remote plugins, `SOURCE_DIR` points to the chosen subfolder within `SOURCE_ROOT`.

| Variable | Value |
| --- | --- |
| `REPO_DIR`, `PLUGIN_DIR` | This repository and its `plugins/<name>/` folder |
| `SOURCE_ROOT`, `SOURCE_DIR` | Root of the source code and the chosen source folder |
| `OUTPUT_DIR` | New output folder for this plugin and platform |
| `PLUGIN_PLATFORM` | Platform name, such as `linux-x86_64` |
| `PYTHON` | Python used to run this repository’s scripts and their packages |
| `PLUGIN_PYTHON` | Python version chosen for the plugin |
| `RELAY_BIN` | Relay executable used for tests |
| `CARGO_TARGET_DIR` | Saved Rust build files for this plugin and platform |
| `BUNDLE_DIR` | Folder with the extracted bundle, used by smoke tests |

The package command must create the `artifacts.bundle` folder inside `OUTPUT_DIR`. The runtime manifest path starts from that bundle folder. The runner checks the files’ hashes to detect changes, then creates an archive. It extracts the archive into `BUNDLE_DIR` and runs the smoke test there. Tests must use those extracted files. If any step fails, the runner does not deliver the bundle.

Each plugin keeps its build, test, and package steps in `tasks.py`. Its `smoke_test.py` checks the installed bundle. Keep package names, output filenames, runtime files, settings, test requests, and expected results in the plugin’s folder. The Switchyard plugin uses the package script from its saved upstream commit. **Upstream** means the source project that this repository builds or copies from.

Plugins can use these shared helpers:

- `scripts/tasks.py` runs commands, reads build settings, chooses filenames for each platform, and writes file hashes into manifests.
- `scripts/smoke.py` provides a local HTTP server for test data. Its `installed_gateway` helper installs and starts a plugin, then checks shutdown, changed-file rejection, and removal. The plugin supplies settings and startup arguments, then sends test requests to the gateway URL.

You can use other scripts or tools if they follow the manifest rules. Shared scripts do not select special behavior based on a plugin’s name.

## Adding or updating a plugin

1. Add `plugins/<name>/release.toml`, source code or a remote source link, documentation, task scripts, and tests. Shared scripts find plugins from these folders, so you do not need to add a central entry or a root Cargo workspace.
2. Choose `worker` or `native` and list any platform limits. Tests must pass on every platform you list.
3. Give the plugin its own SDK and runtime package locks. Keep the upstream license and credit files.
4. Test the plugin’s settings and behavior. Also test an installed bundle: start it, send a typical request, shut it down, and remove it. Check that Relay rejects a bundle whose files were changed. The examples use a local HTTP server that needs no credentials.
5. Check the manifests, run the shared tests, and run the local plugin build and tests. CI checks the other platforms.

For a remote plugin, set its source like this:

```toml
[source]
location = "remote"
repository = "https://github.com/NVIDIA-NeMo/Switchyard"
ref = "main"
sha = "8dc891195a5fa71350f5a03c19f9eecc0f9fcb09"
path = "crates/switchyard-nemo-relay-plugin"
```

`sha` is required and sets the exact source commit to build. `ref` records the branch or tag you plan to track. It does not make CI fetch newer code. To update the source, change the SHA in a pull request and test it before release.

CI downloads the full remote repository so the plugin can use other packages in that workspace. Switchyard’s repository is public. For a private source repository, you must provide credentials with read access.

## CI

External GitHub Actions use the full commit SHA of their latest stable release. Each action also has a comment with its exact version tag. When you update an action, find the commit SHA for its newest stable release.

CI and the current plugins use Python 3.11, the oldest version NeMo Relay supports. A plugin can choose a newer Python version if its packages need one.

Each pull request (PR) checks all manifests and runs the shared tests. A branch push does not start a separate CI run. A tag push starts release checks for the named plugin.

CI selects plugin builds from the changed files:

- A change inside a plugin’s folder selects that plugin.
- A change to shared scripts, tests, schemas, workflows, root package locks, or license files selects all plugins.
- A change only to root documentation does not select a plugin build.
- If Git history is missing, CI selects all plugins.

The file comparison includes renamed and deleted files.

CI builds, tests, packages, and installs each selected plugin on every platform it supports. Each job runs on that platform. In the repository’s branch protection settings, require **Plugin checks**. This check combines the required job results. Build jobs have read-only access. Only the tag release job can write release data.

See [RELEASE.md](RELEASE.md) to tag one plugin and publish its draft release. Follow [SECURITY.md](SECURITY.md) to report security issues. Source code uses the [Apache-2.0 license](LICENSE). Bundles also include the license notices from their source projects.

## Third-party notices and attribution

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists copied sources and links to
each project’s **attributions**: credits and license text for the packages it uses.
These files follow NeMo Relay’s format. The root Python and Rust files combine
entries from all plugins and repository tools, keeping each package version.
Each package script generates its plugin's attribution files from its locked
dependencies and includes them in the bundle. Per-plugin attribution files are
not committed. Every bundle still includes its license and source notices.

When you change a package lockfile or remote source commit, update the root
attribution files. Follow [the licensing instructions](scripts/licensing/README.md).
