<!-- SPDX-License-Identifier: Apache-2.0 -->
# Releasing one plugin

Each release has exactly one plugin tag and one draft GitHub Release. There is no repository-wide version or batch-tagging command.

## Prepare

1. Update the selected plugin's `release.toml` version using SemVer without a leading `v`. Coordinate language-package versions when appropriate; the remote plugin's release version is controlled independently here.
2. Update dependency locks and, for remote plugins, the committed source SHA as needed. Keep the runtime compatibility declaration accurate.
3. Merge the change after all required plugin/platform checks pass. The test host defaults to the latest stable Relay release; a `[relay]` tag/SHA override changes only the test host.
4. Check out the intended merged commit with full history, then validate and push one tag:

```sh
uv run --locked python -m scripts.plugins validate-tag example-rust-native-plugin-0.1.0
git tag example-rust-native-plugin-0.1.0
git push origin refs/tags/example-rust-native-plugin-0.1.0
```

The tag must match an existing plugin name and its exact manifest version at that commit. Every tag triggers validation; invalid tags fail CI. The workflow builds only the named plugin, across every supported platform.

## Review and publish

After all builds, plugin tests, and installed-bundle smoke tests pass, CI creates a draft release with:

- One installable archive per platform: `.tar.gz` for Linux/macOS or `.zip` for Windows.
- A `.sha256` sidecar for each archive.
- A `.json` sidecar recording source and repository commits, the tested Relay revision, toolchains, platform, digest, and verification result.
- Plugin-specific PR release notes. Remote plugins also link to their source revision and upstream documentation.

Notes compare with the nearest ancestral published tag for this plugin. The first release considers repository history. Only merged PRs represented in that interval and affecting the plugin or shared infrastructure are listed. Unrelated PRs and direct commits without a merged PR are not added. File-list pagination failures stop release generation rather than silently producing incomplete notes.

Locally built bundles are marked when the checkout is dirty or a local host override is used. Those bundles remain useful for testing, but the publisher refuses them. Release assets must come from a clean checkout and a verified host selection.

Review the assets and notes, then publish the draft manually in GitHub. CI never publishes it. Avoid editing/publishing a draft while its workflow is running.

## Install a bundle

Verify its archive checksum, extract it to a persistent directory, and register the extracted runtime manifest:

```sh
nemo-relay plugins validate ./example-rust-native-plugin/relay-plugin.toml
nemo-relay plugins add --user ./example-rust-native-plugin/relay-plugin.toml
```

The bundle defaults to disabled. Configure the plugin and host trust policy before enabling it. These bundles include SHA-256 integrity metadata and do not include signing keys or artifact signatures. For a bundle you trust, Relay's per-plugin `attestation = "integrity_only"` policy permits activation; see the [Relay discoverable plugin guide](https://github.com/NVIDIA/NeMo-Relay/blob/main/docs/configure-plugins/discoverable-plugins.mdx). Switchyard additionally requires a deployment configuration; refer to its linked upstream documentation.

For Python workers, `plugins add` provisions the managed Python environment from the extracted package source. Retain that source and the runtime manifest at the installed location. Package dependency installation requires access to the configured Python package index.

## Failure and retry

A failed build, missing artifact, failed runtime test, or incomplete release-note lookup fails the workflow. Build failures do not create a draft. Upload failures may leave an incomplete draft; rerun the failed workflow to repair it after diagnosing the cause.

Drafts carry a machine-readable identity marker with the plugin name, version, and tag commit. Retrying updates only a matching draft and verifies the complete uploaded asset set. Published releases, unmanaged drafts, moved tags, and mismatched commit markers are rejected. Do not remove the marker while a draft may need a retry.

Use a new plugin version and tag to correct a published release. Keep remote SHAs committed; reruns never advance an upstream branch. The default latest-stable host may advance between workflow runs, and its exact revision is recorded in the assets. Use a tag/SHA host override when a fixed host is required.

## Repository setup

GitHub Actions must permit the pinned actions in `.github/workflows/plugins.yml`, the five configured hosted runner labels, and `contents: write` plus `pull-requests: read` for the release job. The repository's current action policy permits the GitHub-owned actions used here. Require the **Plugin checks** branch-protection check. Repository settings and publication remain maintainer operations; local implementation does not create tags or releases.
