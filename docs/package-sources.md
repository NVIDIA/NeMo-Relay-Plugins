<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Plugins from wheels and crates

A wheel (`.whl`) is a built Python package. A crate (`.crate`) contains Rust
source code. This repository can use either as a plugin source. The package
must include `relay-plugin.toml` and the files that its manifest names, such as
its configuration schema. A crate can use placeholders for files produced by
the build. Its package script must fill in those paths and hashes.

The existing four plugin registrations keep their current sources. These new
source types are available when adding another plugin or changing its source.

## Choose a source

For a wheel, put this in the plugin's `release.toml`:

```toml
[source]
location = "wheel"
package = "my-worker"
version = "1.2.3"
manifest = "my_worker/relay-plugin.toml"
# Optional; defaults to source.lock in this plugin's folder.
lock = "source.lock"
```

For a crate:

```toml
[source]
location = "crate"
package = "my-native-plugin"
version = "1.2.3"
manifest = "relay-plugin.toml"
```

The manifest path is relative to the wheel's root, or the crate's package
folder after extraction. Keep the other release settings, including commands
and tests. Crate sources require a Rust toolchain. The package version above
can differ from the release version at the top of `release.toml`.

A crate can include its runtime manifest alongside `Cargo.toml`. Use Cargo's
`include` setting if needed, and check the result with `cargo package --list`.
The published crate must contain `Cargo.lock`. Its dependencies must work from
the published archive; unpublished sibling folders will not be available.

For a wheel, keep the runtime manifest inside its Python package. Configure
the Python build tool to include it and any schema files as package data.
A root-level `relay-plugin.toml` is also accepted, but that location can collide
with another package's manifest during normal Python installation. All wheel
files, including the manifest and schemas, must appear in the wheel's `RECORD`
file with valid hashes. `RECORD` is the wheel's file inventory.

## Lock the downloads

Commit a `source.lock` file beside `release.toml`. It contains the plugin
package and, for wheels, every runtime and test dependency needed by its
commands. Each entry records an exact file, URL, checksum, and list of platforms.
Use the download URLs and SHA-256 values from the package registry. Replace
the sample URL and checksum below with the real values before validation.

```toml
schema_version = 1

[[artifacts]]
name = "my-worker"
version = "1.2.3"
filename = "my_worker-1.2.3-py3-none-any.whl"
url = "https://files.pythonhosted.org/path/to/my_worker-1.2.3-py3-none-any.whl"
sha256 = "<64 lowercase hexadecimal characters>"
platforms = ["linux-x86_64", "linux-arm64", "windows-x86_64", "windows-arm64", "macos-arm64"]
```

Add an entry for each dependency wheel. If a package needs different wheels
on different platforms, give each file its own entry and platform list. Each
platform must have exactly one plugin wheel and one version of each dependency.
A pure Python wheel can list several platforms. A compiled wheel must match the
Python version and platform used by the job. Installation checks this match.

For a crate, use the same entry format with the crate's name, version, `.crate`
filename, HTTPS download URL, and SHA-256. There is one crate artifact per
platform. The same archive can list all platforms. Cargo dependencies come
from its packaged `Cargo.lock`; do not list them as extra crate artifacts.

CI never resolves a newer package version or substitutes a source distribution
for a missing wheel. It checks download hashes, package names, and versions.
Wheel environments install only the listed wheels, without a package index,
and fail if required dependencies are missing or have conflicting versions.
Wheel dependencies must use normal package requirements. Direct URL or Git
requirements inside a wheel are rejected; publish those dependencies as wheels
and list their exact files in `source.lock`.

## Build, test, and package

Package sources use the same four required commands as other plugins. Keep
their implementations in the plugin folder. The runner adds these variables:

| Variable | Meaning |
| --- | --- |
| `SOURCE_ROOT`, `SOURCE_DIR` | Extracted package folder |
| `SOURCE_MANIFEST` | Runtime manifest inside that folder |
| `PACKAGE_SOURCE_LOCK` | JSON record of the selected files and hashes |
| `WHEELHOUSE` | Downloaded wheels for this platform; wheel sources only |
| `PACKAGE_PYTHON` | Isolated Python environment with all locked wheels installed |

For wheels, installation and dependency checks run before the plugin's build
command. Use `PACKAGE_PYTHON` to run plugin tests against the installed package.
The build command can check required imports or prepare local test fixtures.
It does not need to rebuild the upstream wheel.

For a Python worker wheel, the package command can call the shared helper:

```python
from scripts.tasks import Context, package_wheel

package_wheel(Context.from_environment())
```

The helper copies the package files, generates attributions, and writes
`relay-plugin.toml` at the bundle root. It adjusts paths and computes the runtime
artifact hash. Set `artifacts.manifest = "relay-plugin.toml"` when using it.
Python worker entrypoint modules must use the usual package layout at the wheel
root, such as `my_worker/worker.py`. Paths in the upstream runtime manifest are
relative to that manifest. They may point to a parent folder within the package
archive, but may not leave the archive.

Relay currently installs Python workers from a project directory. The helper
adds a small Python build adapter and the original wheels to the bundle. The
adapter verifies their hashes and installs those same wheels after the bundle
has moved. It has no build dependencies and does not change upstream wheel
bytes. It requires the Python version and platform recorded for that bundle.
Its files use reserved paths: `pyproject.toml`, `relay_wheel_backend.py`,
`package-source.json`, `wheelhouse/`, and `notices/relay-wheel-adapter/`.
The upstream wheel must leave these paths available.

For crates, build and test from `SOURCE_ROOT` with the selected Rust toolchain:

```python
from scripts.tasks import Context, run

ctx = Context.from_environment()
run(["cargo", "build", "--locked", "--release"], cwd=ctx.source_root)
run(["cargo", "test", "--locked", "--release"], cwd=ctx.source_root)
```

Keep crate-specific artifact filenames and packaging steps in that plugin's
script. Copy the needed library or worker executable, configuration schemas,
and notices. Fill in the runtime manifest and call `write_attributions(ctx)`.
Published packages may omit upstream tests, so keep the required behavior tests
in this repository when needed.

The smoke command must still install the extracted bundle in Relay, send a
representative request, and check shutdown, removal, and tamper rejection. A
successful package download or compilation does not replace these tests.

## Licenses and updates

Wheel attributions come from the exact locked wheels and their included license
files. Wheels must carry their license text. Crate attributions include the
published crate itself and all locked dependencies. Root attribution files
combine entries across all declared platforms. Bundle attributions are generated
for that bundle's platform. No per-plugin attribution files need to be committed.

To update a package, change `source.version` and its `source.lock` entries in one
pull request. Keep the release version separate unless you also intend to release
a new bundle. Run:

```sh
uv run --locked python -m scripts.plugins validate
uv run --locked python -m scripts.plugins run <plugin-name>
```

Pre-commit updates root attributions and checks the locks. CI tests every declared
platform. Draft release notes link to the upstream package and documentation.
Build metadata records the selected package URLs, versions, hashes, platform,
and Python version. Release checks compare that metadata with the committed lock.
