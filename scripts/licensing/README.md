<!-- SPDX-License-Identifier: Apache-2.0 -->
# Updating package credits and licenses

Attribution files list the packages a project uses and include their license
text. To update these files, start at the repository root. Install the Rust
versions listed in the plugin manifests and the `cargo-about` tool. Then run
the generator for all plugins and both root files:

```sh
rustup toolchain install 1.96.1 --profile minimal
cargo +1.96.1 install cargo-about --version 0.9.1 --locked --features cli
uv run --locked python -m scripts.licensing.generate
```

The generator finds plugins through their `release.toml` files and reads each
plugin's package lockfile. For remote plugins, it downloads the exact commit
set in `source.sha`.

Each plugin keeps its attribution files in its own folder. The list for the
repository's Python tools is in `scripts/licensing/ATTRIBUTIONS-Python.md`.
The root `ATTRIBUTIONS-Python.md` and `ATTRIBUTIONS-Rust.md` combine all these
lists. They remove exact duplicates but keep different versions and license text.

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

Commit the generated files along with package or source updates. CI checks that
all plugin and root attribution files are up to date with:

```sh
uv run --locked python -m scripts.licensing.generate --check
```

Every bundle must include a nonempty `ATTRIBUTIONS*.md` file. Each plugin's
package script copies its attribution files and source project notices into
the bundle. The shared checks reject missing attribution files, both before
packing the bundle and after extracting it.

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
