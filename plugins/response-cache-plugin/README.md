<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Response Cache Plugin

## Summary

This plugin saves large language model (LLM) answers and replays them when
Relay sees the exact same request again. A repeated request then skips the
model provider, so it is faster and costs nothing. The plugin can also save
results from tools that you mark as safe to reuse.

The plugin used to be the `response_cache` section of Relay's adaptive plugin.
It is now a separate native plugin. A **native plugin** is a library that Relay
loads into its own process. See [UPSTREAM.md](UPSTREAM.md) for what changed in
the move.

**Load and enable.** The bundled `relay-plugin.toml` registers
`nvidia.response_cache`. The manifest starts disabled. Configure the plugin and
Relay's trust policy, then run:

```sh
nemo-relay plugins add --user ./response-cache-plugin/relay-plugin.toml
nemo-relay plugins enable nvidia.response_cache
```

**Distributed artifacts.** A release provides one archive for each supported
platform. Each archive has a `.sha256` checksum and a `.json` build-metadata
file. The archive contains the platform shared library, the completed runtime
manifest, the configuration schema, upstream notes, license notices, and Rust
dependency attributions.

## What it caches

- **LLM answers.** Normal and streaming calls share one entry. A streamed answer
  is saved as one complete response and replayed as stream chunks.
- **Tool results.** Off by default. Turn it on only for tools that never change
  anything and give the same answer for the time the entry is kept.

Only requests pinned to `temperature` 0 are cached unless you set
`cache_nondeterministic = true`. Failed calls, incomplete streams, and answers
that cannot be replayed exactly are never saved. When anything goes wrong with
the store, the call runs live.

## Configure

Every setting is in [config.schema.json](config.schema.json). The keys are the
same as the old adaptive `response_cache` section. `namespace` is required. It
names one group of users that may share answers. Never share a namespace
between tenants or upstreams that should not see each other's data.

```toml
[[plugins.dynamic]]
manifest = "./response-cache-plugin/relay-plugin.toml"

[plugins.dynamic.config]
namespace = "team-dev"
ttl_seconds = 3600

[plugins.dynamic.config.backend]
kind = "redis"
config = { url = "redis://127.0.0.1:6379", key_prefix = "relay-cache:" }
```

The default backend is `in_memory`, which keeps up to 256 MiB in the Relay
process. The `redis` backend shares one cache across processes.

Every cache decision emits a `response_cache` mark. A hit reports the tokens
and cost it saved. Cost comes from Relay's active pricing catalog, so this
plugin needs a Relay version that provides plugin cost estimates.

## Build and test

This plugin's release name is `response-cache-plugin`.
[`release.toml`](release.toml) defines its build and release settings.
`relay-plugin.toml` is the template for the bundled runtime manifest.

From the repository root, build and test the plugin, create a bundle, and check
the installed bundle:

```sh
uv run --locked python -m scripts.plugins run response-cache-plugin
```

From this folder, run the tests directly:

```sh
cargo test --release
```

The unit tests check cache keys, storage, replay, and marks without Relay. The
integration tests build the library, load it into a Relay host in the test
process, and run real LLM and tool calls through it. One test that leaves the
plugin loaded on purpose runs in its own test program.

The library names are:

| Platform | Library |
|---|---|
| macOS | `libnemo_relay_response_cache_plugin.dylib` |
| Linux | `libnemo_relay_response_cache_plugin.so` |
| Windows | `nemo_relay_response_cache_plugin.dll` |

Use [the release guide](../../RELEASE.md) to install a bundle and set Relay's
trust rules.
