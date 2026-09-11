<!-- SPDX-License-Identifier: Apache-2.0 -->
# Updating dependency attributions

From the repository root, install the Rust toolchains selected in plugin manifests
and cargo-about, then regenerate every plugin and both root aggregates:

```sh
rustup toolchain install 1.96.1 --profile minimal
cargo +1.96.1 install cargo-about --version 0.9.1 --locked --features cli
uv run --locked python -m scripts.licensing.generate
```

The generator discovers `release.toml` registrations, reads each independent lock,
and checks out remote workspaces only at `source.sha`. The tooling inventory is
stored in `scripts/licensing/ATTRIBUTIONS-Python.md`; each plugin owns its inventory
in its registration directory. The root `ATTRIBUTIONS-Python.md` and
`ATTRIBUTIONS-Rust.md` combine all of those inventories, deduplicating identical
entries while preserving different versions, licenses, and license text.

Files follow NeMo Relay's format, including full license text. Inventories cover
all locked targets and build/test dependencies, rather than only the current
platform's installed packages. Network access is required. Python artifacts are
verified against `uv.lock` hashes, and Cargo operations use `--locked`.

If an archive omits license text, the generator reads upstream repository metadata
from the package. Rust's `.cargo_vcs_info.json` provides its publication commit;
Python's exact version tag is resolved to a commit. License files are fetched from
that immutable revision and their source URLs appear in the generated entries.
There are no hand-maintained package license files or package-name exceptions.
An unresolvable source or missing license text fails generation.

Commit generated changes with dependency or source updates. CI verifies all
per-plugin inventories and root aggregates using:

```sh
uv run --locked python -m scripts.licensing.generate --check
```

Every bundle must include a nonempty `ATTRIBUTIONS*.md` file. Plugin packaging
copies its own inventories and upstream notices; generic bundle verification
rejects missing attribution files before distribution and after extraction.

## License diff

The comparison uses NeMo Relay's added, removed, and updated/changed license report
format. Both sides collect actual lockfile dependencies, including the remote
source SHA recorded at each revision. No plugin code is executed by the collector.
To compare the current tree with a base revision:

```sh
uv run --locked python -m scripts.licensing.license_diff --base-ref origin/main
```

`--format json` provides machine-readable output. `--base-json` and
`--current-json` accept pre-generated inventories. The generation command accepts
`--inventory-output <path>` to save an inventory alongside attribution checks.
PR CI includes the informational diff in the Actions summary and downloadable
`license-report` artifact. It does not post PR comments. Attribution freshness is
a required check; the informational diff follows NeMo Relay's nonblocking behavior.
