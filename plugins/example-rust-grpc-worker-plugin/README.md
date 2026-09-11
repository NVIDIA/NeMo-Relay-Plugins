<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust gRPC Worker Plugin

This plugin’s release name is `example-rust-grpc-worker-plugin`.
[`release.toml`](release.toml) defines its build and release settings.
`relay-plugin.toml` tells Relay how to load the plugin.

From the repository root, use this command to build and test the plugin, create
a bundle, and check that the installed bundle works:

```sh
uv run --locked python -m scripts.plugins run example-rust-grpc-worker-plugin
```

See [UPSTREAM.md](UPSTREAM.md) for the original source commit.
Use [the release guide](../../RELEASE.md) to install a bundle and set Relay’s trust rules.

This is the Rust worker from the NeMo Relay plugin authoring guide. It checks
the example’s settings and uses the safe hooks in `grpc-v1`, the protocol for
talking to Relay. A hook lets a plugin handle an event or change a request.

The example passes requests to the next handler and processes stream items as
they arrive. It gives each call its own helpers to encode and decode data. It
also shows how to record event marks and clean up state when a call ends.

Run `cargo test --release` and `cargo build --release` from this folder. The
settings and schema tests can run in any order. The lifecycle test checks the
worker from start to finish: it builds a fresh worker, writes a manifest with
the file’s hash, and starts the worker through `grpc-v1`. It then runs request
handlers, checks an event mark in the host, and checks shutdown.

To set up a local manifest:

1. Copy `relay-plugin.toml` to `relay-plugin.local.toml`.
2. Replace the platform worker placeholder with the built executable’s name.
3. Replace only `<artifact-sha256>` with the executable’s SHA-256 hash in lowercase hexadecimal. Keep the `sha256:` prefix.

You can get the hash with `shasum -a 256`, `sha256sum`, or `Get-FileHash`. Copy
only the hash, not the filename column.

The optional `registration_control` setting adds a callback that can block
selected middleware while the worker is active. Middleware handles requests or
events as they pass through Relay. The control starts disabled, with these defaults:

- Kinds: `["subscriber"]`
- Target: `documentation-controlled-subscriber`
- Reason: `disabled by documentation plugin`

The callback blocks targets whose names start with `documentation-controlled-`
and returns the reason. It returns `None` to leave other matching targets enabled.
All three values must not be empty. See
[Conditional Middleware Guardrails](https://github.com/NVIDIA/NeMo-Relay/blob/65eb82bf3d788986512246abf7f8ab7a520f28d9/docs/about-nemo-relay/concepts/conditional-middleware-guardrails.mdx)
for how to find target names and how Relay removes the control when the worker stops.

## SDK and test host

This example needs the Relay 0.9 API from upstream `main`. Until those SDK
packages are published, its package locks use an exact Git commit. Its
`release.toml` selects a test host built from that commit. See [UPSTREAM.md](UPSTREAM.md)
for the source version and local changes.
