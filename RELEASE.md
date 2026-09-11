<!-- SPDX-License-Identifier: Apache-2.0 -->
# Releasing one plugin

Release one plugin at a time. Each release has one Git tag and one draft GitHub Release. The repository has no shared version or command to tag several plugins at once.

## Prepare

1. Update the plugin’s version in `release.toml`. Use a SemVer version, such as `0.1.0`, without a leading `v`. Update the Python or Rust package version too, if needed. A remote plugin’s release version here can differ from its source project’s version.
2. Update package lockfiles and the remote source commit (`source.sha`) as needed. Check that the runtime manifest lists the correct supported Relay versions. Update the [attribution files](scripts/licensing/README.md) when packages or sources change.
3. Merge the change after all required checks pass on every supported platform. Tests use the latest stable Relay release by default. A `[relay]` tag or SHA setting changes only that test host.
4. Check out the merged commit you want to release, with full Git history. Then check and push one tag:

```sh
uv run --locked python -m scripts.plugins validate-tag example-rust-native-plugin-0.1.0
git tag example-rust-native-plugin-0.1.0
git push origin refs/tags/example-rust-native-plugin-0.1.0
```

The tag must match a plugin name and the exact version in its manifest at that commit. CI checks every tag and fails if the tag is invalid. It builds the named plugin on every supported platform.

## Review and publish

After all builds and tests pass, including tests of the installed bundle, CI creates a draft release with:

- One installable archive per platform: `.tar.gz` for Linux/macOS or `.zip` for Windows.
- A `.sha256` file with a checksum for each archive. Use it to check that the downloaded file has not changed.
- A `.json` file that records the source and repository commits, tested Relay commit, tool versions, platform, file hash, and test result.
- Release notes that list PRs affecting this plugin. Remote plugins also link to their source commit and upstream documentation.

CI looks back through the tagged commit’s history to find the closest published release tag for the same plugin. It lists merged PRs added since that release. For the first release, it looks through all earlier repository history.

Notes include only PRs that changed the plugin or shared build files. They leave out unrelated PRs and direct commits with no merged PR. If GitHub cannot return the full list of changed files, release creation fails.

Local bundles are marked if the checkout has uncommitted changes or uses a local Relay executable. You can use these bundles for tests, but the release script rejects them. Release files must come from a clean checkout and a checked test host.

Review the files and notes, then publish the draft yourself in GitHub. CI never publishes it. Wait for the workflow to finish before editing or publishing the draft.

## Install a bundle

Check the archive’s checksum, then extract it to a folder you plan to keep. Add the plugin to Relay using the extracted runtime manifest:

```sh
nemo-relay plugins validate ./example-rust-native-plugin/relay-plugin.toml
nemo-relay plugins add --user ./example-rust-native-plugin/relay-plugin.toml
```

The plugin starts disabled. Set its options and Relay’s trust rules before you enable it. Bundles include SHA-256 hashes to detect changed files. They do not include signing keys or digital signatures.

For a bundle you trust, set the plugin’s `attestation` policy to `"integrity_only"` to allow it to run. See the [Relay discoverable plugin guide](https://github.com/NVIDIA/NeMo-Relay/blob/main/docs/configure-plugins/discoverable-plugins.mdx). Switchyard also needs a deployment configuration file; see its linked upstream documentation.

For Python workers, `plugins add` creates a Python environment from the extracted source package. Keep that source and the runtime manifest in the install folder. The install step needs access to your Python package index to download required packages. The current Python example also needs Git and access to GitHub to install its SDK from the locked source commit.

## Failure and retry

The workflow fails if a build or runtime test fails, a release file is missing, or release notes cannot be completed. A build failure does not create a draft. A failed upload may leave a partial draft. Fix the cause, then rerun the failed workflow to complete the draft.

Each draft has a marker with the plugin name, version, and tag commit. On a retry, CI only updates a draft with a matching marker. It then checks that all release files were uploaded.

CI refuses to change published releases or drafts it did not create. It also rejects tags that moved to another commit and drafts with a different commit marker. Keep the marker so retries can check the draft.

To fix a published release, create a new plugin version and tag. Save remote source SHAs in the manifest; a retry never switches to newer source code.

The default test host may change if Relay publishes a new stable release between runs. Each run records the exact host commit in the release files. Set a Relay tag or SHA in the manifest if you need the same host on every run.

## Repository setup

Allow the actions listed in `.github/workflows/plugins.yml` and all five configured GitHub runner types. The release job needs `contents: write` and `pull-requests: read`. This repository’s action policy allows the GitHub-owned actions used here.

In branch protection settings, require **Plugin checks**. Maintainers manage these settings and publish releases. Running the local build and test commands does not create tags or releases.
