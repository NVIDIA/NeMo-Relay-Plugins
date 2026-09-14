<!-- SPDX-License-Identifier: Apache-2.0 -->
# Updating package credits and licenses

Attribution files list the packages a project uses and include their license
text. To update these files, start at the repository root. Install the Rust
versions listed in the plugin manifests and the `cargo-about` tool. Then run
the generator to update the committed aggregate files:

```sh
rustup toolchain install 1.96.1 --profile minimal
cargo +1.96.1 install cargo-about --version 0.9.1 --locked --features cli
uv run --locked python -m scripts.licensing.generate
```

The generator finds plugins through their `release.toml` files and reads each
plugin's package lockfile. For remote plugins, it downloads the exact commit
set in `source.sha`.
Remote Rust plugins also include the packages in their source workspace. A
workspace is a group of packages built together. This covers upstream code
that the plugin uses through local paths, as well as downloaded dependencies.

For Python, the generator starts at the plugin's `source.path` and looks for
`uv.lock`. If it is missing there, it checks each parent folder up to the
source repository's root. This supports both a plugin's own lockfile and a
shared workspace lockfile. It never searches outside the source checkout.

Remote Python notices include local-path and editable packages from that
lockfile. An editable package loads code from its source folder. The generator
reads each package's `pyproject.toml` and license files from the pinned checkout.
It supports `license-files` patterns and the older `license.file` and
`license.text` fields. Without these fields, it looks for `LICENSE`, `LICENCE`,
or `COPYING` files in the package and then its parent folders. It also keeps
nearby `NOTICE` files. Paths must stay inside the checkout. Missing license
text or a package name or version that conflicts with the lockfile stops
generation. For a dynamic version, the lockfile supplies the version without
running the package's build code.

Virtual workspace entries contain no installable package, so they are excluded.
Local packages in in-tree plugins remain excluded from third-party notices.
Downloaded registry and pinned Git dependencies are included in both cases.

For `wheel` and `crate` sources, the generator reads the exact downloads listed
in the plugin's `source.lock`. Wheels supply their own license text. Crates
supply their package license and a `Cargo.lock` for their dependencies. Root
attributions cover every declared platform; bundles cover their own platform.
See [package sources](../../docs/package-sources.md) for the lockfile format.

Plugin attribution files are generated during packaging and placed inside the
bundle. They are not committed in the plugin folders. The list for the
repository's Python tools is in `scripts/licensing/ATTRIBUTIONS-Python.md`.
The committed root `ATTRIBUTIONS-Python.md` and `ATTRIBUTIONS-Rust.md` combine
fresh data from all plugins and tools. They remove exact duplicates but keep
different versions and license text. Building a bundle does not read these
aggregate files; it generates notices from that plugin's own locked source.

Files follow NeMo Relay's format and include full license text. They cover all
platforms and packages listed in the locks, including build and test packages.
You need network access to run the generator. It checks Python downloads against
the hashes in `uv.lock`. Cargo commands use `--locked` to keep package versions fixed.

If a package archive has no license text, the generator looks for it in the
source repository listed by the package. Rust's `.cargo_vcs_info.json` records
the commit used to publish the package. For Python packages from a registry,
the generator finds the commit for the exact version tag. For a Python Git
dependency, it reads the package metadata and license at the commit recorded
in `uv.lock`. It downloads license text from that commit and adds source links
to the generated entry.

There are no separate package license files to maintain by hand. The generator
also has no special rules for named packages. It fails if it cannot find the
source or license text.

Commit the generated files along with package or source updates. To check that
the committed aggregate and tooling files are up to date without editing them, run:

```sh
uv run --locked python -m scripts.licensing.generate --check
```

The pre-commit attribution hook runs the same generator on each commit, including
commits that delete files. It updates the committed files for you. Review and
stage those changes before committing again. CI runs every hook
and fails if generated files differ from the committed copies. To run just this
hook locally, use:

```sh
uv run --locked --group test pre-commit run attributions --all-files
```

Every bundle must include a nonempty `ATTRIBUTIONS*.md` file. Each plugin's
package script calls `write_attributions(ctx)` from `scripts/tasks.py` to generate
its attribution files. It also copies the source project notices into the bundle.
The helper reads the package lockfile and, for remote plugins, the checked-out
workspace at the pinned commit. If license collection fails, packaging fails.
Rust packaging needs `cargo-about`; CI installs the same version shown above.
The shared checks reject missing attribution files, both before packing the
bundle and after extracting it.

## Comparing licenses

The license diff lists added, removed, and changed package versions and licenses.
It uses NeMo Relay's report format. It reads the lockfiles on both sides of the
comparison, including each remote plugin's saved source commit. The collector
does not run plugin code.

To compare your current files with a base commit or branch, run:

```sh
uv run --locked python -m scripts.licensing.license_diff --base-ref origin/main
```

Use `--format json` for output that other scripts can read. Use `--base-json`
and `--current-json` to compare saved package lists. The generator's
`--inventory-output <path>` option saves such a list while checking attribution files.

PR CI puts the report in the Actions summary and the downloadable `license-report`
file set. It does not post PR comments. Attribution files must be up to date for
CI to pass. The license diff is a report for review; like NeMo Relay's check,
it does not block CI if the comparison fails.

## Dependency inventory

PR CI also creates a `dependency-inventory` artifact and links to it from the
repository-check job summary. The artifact contains `dependencies.csv`. Each
row gives the package name, exact locked version, language, and dependency type.
It lists dependencies declared directly by the repository tools and plugins.
Transitive packages remain in the attribution files but are not repeated in
this direct-dependency report.

The dependency types are `direct + required`, `direct + optional`,
`development-only`, and `test-only`. Python dependencies in the `test` or
`tests` development group are test-only. Other Python development groups are
development-only. Rust build dependencies are development-only, and Rust dev
dependencies are test-only. If the same exact package has more than one use,
the report keeps its broadest use in this order: required, optional,
development, then test.

To create the same CSV locally, run:

```sh
uv run --locked python -m scripts.licensing.dependency_inventory \
  --output .cache/dependencies.csv
```
