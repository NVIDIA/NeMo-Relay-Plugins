<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python gRPC Worker Plugin

This plugin’s release name is `example-python-grpc-worker-plugin`.
[`release.toml`](release.toml) defines its build and release settings.
`relay-plugin.toml` tells Relay how to load the plugin.

From the repository root, use this command to build and test the plugin, create
a bundle, and check that the installed bundle works:

```sh
uv run --locked python -m scripts.plugins run example-python-grpc-worker-plugin
```

See [UPSTREAM.md](UPSTREAM.md) for the original source commit.
Use [the release guide](../../RELEASE.md) to install a bundle and set Relay’s trust rules.

This is the Python worker from the plugin authoring guide. It checks the example’s
settings and uses the safe plugin hooks in `grpc-v1`, the protocol for talking to Relay.
A hook lets a plugin handle an event or change a request.

The example shows how to keep result annotations and Relay’s usage records intact.
It gives each call its own codec helpers, which encode and decode data. It also
changes stream items as they arrive and cleans up event marks, call state,
isolated stacks, and cancelled tasks.

The worker uses Relay 0.9’s `grpc-v1` result format. When it passes a tool call to
the next handler, it returns `ToolExecutionResult`. Its execution hook keeps the
application’s result and stores the original annotation in worker metadata. It
also adds pending marks owned by Relay.

Run the example's own test project from this directory:

```bash
uv run --locked --group test pytest
```

Each test checks one behavior and can run on its own. The tests build a wheel
(an installable Python package) in a fresh temporary project. They check the
required source hash, the settings schema, valid settings, and all 17 registered
hooks. Separate tests cover policies, data cleaning, request changes, calls to
the next handler, streams, event marks, and call state.

The seventeenth hook can block selected middleware while the plugin is active.
Middleware handles requests or events as they pass through Relay. This control
is optional: `registration_control.enabled` defaults to `false`. Its other defaults are:

- Kinds: `["subscriber"]`
- Target: `documentation-controlled-subscriber`
- Reason: `disabled by documentation plugin`

The callback blocks targets whose names start with `documentation-controlled-`
and returns the reason. It returns `None` to leave other matching targets enabled.
The kinds, target name, and reason must not be empty. See
[Conditional Middleware Guardrails](https://github.com/NVIDIA/NeMo-Relay/blob/073c913f47d4c8ed10f3af5bf781ae58f6dee20a/docs/about-nemo-relay/concepts/conditional-middleware-guardrails.mdx)
for the full rules, including which plugin owns each control.

To try the plugin from this folder, create temporary Relay settings, add the
manifest, and enable the plugin:

```bash
relay_tmp="$(mktemp -d)"
relay_config="$relay_tmp/gateway.toml"
: > "$relay_config"
nemo-relay --config "$relay_config" plugins add ./relay-plugin.toml
nemo-relay --config "$relay_config" plugins enable examples.python_grpc_worker
nemo-relay --config "$relay_config" --bind 127.0.0.1:4040
```

After stopping Relay, run these cleanup commands in the same shell. This keeps
`relay_config` and `relay_tmp` set to the temporary paths. Removing the plugin
also deletes the Python environment that Relay created.

```bash
nemo-relay --config "$relay_config" plugins remove examples.python_grpc_worker
rm -rf -- "$relay_tmp"
```

## SDK and test host

This example needs the Relay 0.9 API from upstream `release/0.9`. Until those SDK
packages are published, its package locks use an exact Git commit. Its
`release.toml` selects a test host built from that commit. See [UPSTREAM.md](UPSTREAM.md)
for the source version and local changes.
