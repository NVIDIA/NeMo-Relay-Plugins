<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Guardrails plugin

## Summary

This plugin checks tool calls before they run. A tool call is an AI agent's
request to use a tool, such as running a shell command. Before each tool call
that Relay manages, the plugin sends the tool's name and arguments to a
[NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/crates/nemo-guardrails-relay-plugin)
service. Relay loads the Rust shared library directly into its own process.

The plugin has no rules of its own. The NeMo Guardrails service holds the
rules, called rails, and decides every call. You start the service with
`nemoguardrails hooks-server --config <path>`.

- If the service allows the call, the call runs.
- If the service denies the call, Relay blocks it and shows the service's
  reason.
- If the check fails, Relay reports an error and, in most cases, blocks the
  call. A check fails when the service answers with an error, takes too long,
  or cannot be reached. The upstream documentation describes the one
  exception.

**Load and enable.** The bundled `relay-plugin.toml` registers
`nvidia.nemo_guardrails`. The manifest starts disabled. Register the manifest,
set the service's address in the plugin configuration, and add a Relay trust
policy. Then enable it. Relay loads the library the next time it starts:

```sh
nemo-relay plugins add --user ./nemo-guardrails-plugin/relay-plugin.toml
nemo-relay plugins enable nvidia.nemo_guardrails
```

The configuration requires one field, `endpoint`, which is the service's base
URL, such as `http://127.0.0.1:8100`. See the
[upstream plugin documentation](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/crates/nemo-guardrails-relay-plugin#configuration)
for the optional fields.

**Distributed artifacts.** A release provides one `.tar.gz` archive for each
supported Linux or macOS platform and one `.zip` archive for each supported
Windows platform. Each archive has a `.sha256` checksum and a `.json`
build-metadata sidecar. The archive contains the platform shared library, the
completed runtime manifest, the configuration schema, NeMo Guardrails' license
files, Rust dependency attributions, and third-party notices.

## Build and test

This folder builds and releases the NeMo Guardrails plugin. Its code lives in
the NeMo Guardrails repository. The `release.toml` file sets the exact source
commit and selects the plugin's Rust package, called a crate. The build
downloads the NeMo Guardrails repository at that commit, because the crate's
Cargo workspace and lockfile are at the repository root.

From this repository's root:

```sh
uv run --locked python -m scripts.plugins run nemo-guardrails-plugin
```

This command builds the plugin, runs the crate's tests, and uses its package
script to create a bundle. It then extracts and installs the bundle for tests.
These tests start a small local server that stands in for the NeMo Guardrails
service. They send three tool calls through Relay's Claude Code hook: one that
the server denies, one that it allows, and one whose check fails. Each call
must reach the server once and get the right result. The tests also check
shutdown and removal, and confirm that Relay rejects changed files. CI requires
tests to pass on all seven supported platforms.

To use newer NeMo Guardrails code, update `source.sha` to the commit you want.
Set `source.ref` to a branch name or to `refs/tags/<tag>`. Planning checks that
a tag resolves to the pinned commit. The ref records what to track; it does not
update the source on its own.

The release version here matches the crate's version. It is separate from the
NeMo Guardrails Python package version. SDK package versions come from NeMo
Guardrails' Cargo lockfile. A `[relay]` setting changes only the Relay host
used for tests.

Plugin release 0.1.0 requires Relay 0.11.0 or newer, below 1.0. It also accepts
the 0.11.0 release candidates. The plugin's runtime manifest sets this range.
Its workspace uses the published Relay 0.11.0 SDK from crates.io. This
registration tracks NeMo Guardrails' `develop` branch, builds the exact commit
saved in `source.sha`, and tests against Relay 0.11.0.

So far, each Relay minor release has changed the native plugin interface. A
minor release is a version such as 0.11 or 0.12. Test the plugin again before
you move the `[relay]` setting to a new minor release.

See the upstream documentation for configuration settings and supported
agents.
