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
loads into its own process.

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
same as the old adaptive `response_cache` section, but unknown keys now fail
validation instead of producing warnings. `namespace` is required. It
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

## Record and replay

Record/replay is a separate mode of this plugin. It saves model answers to a
**trace** (a folder of files) once. Later runs replay those answers, chunk by
chunk, on their original schedule. Tools always run live. When
`record_replay` is set, it replaces the normal cache: there is no TTL, bypass,
tool-result cache, or fallback to the live provider.

```toml
[[plugins.dynamic]]
manifest = "./response-cache-plugin/relay-plugin.toml"

[plugins.dynamic.config]
namespace = "workload"

[plugins.dynamic.config.record_replay]
mode = "record" # record, verify, or replay
trace_directory = "/data/trace"
output_directory = "/data/out"
latency_scale = 1.0

[plugins.dynamic.config.record_replay.key_filters]
drop = ["/prompt_cache_key"]
redact_lines = ['^Session ID: .*$']
tool_results = "placeholder"
```

| Setting | What it does |
|---|---|
| `mode` | `record` calls the live provider. `verify` and `replay` only serve recorded answers. |
| `latency_scale` | Multiplies each recorded chunk time. `0` turns off waiting. |
| `start_gate` | Optional `path` and `timeout_seconds`. The first request must match before the run reports ready. |
| `key_filters` | Drops or redacts parts of a request before it is matched, and can replace tool results with placeholders. |
| `divergence` | Which live tool results to compare with the trace, and how much output size may change. Only tools in `tool_names` have their result shape checked. |

The trace folder must already exist. Recording never overwrites trace files.
After the recording process exits, the launcher removes `record.lock` before
starting a replay. A locked, corrupt, or mismatched trace makes every model
call fail with an `RR-*` error instead of calling the provider. Repeated
requests use their recorded answers in order.

The plugin reads the provider's stream as it arrives, even when the caller reads
more slowly. Each chunk's time is when the plugin received it from the provider,
so a replay follows the provider's pace. A call counts as complete when the caller reads the whole
answer; a call the caller cancels or drops first is recorded as
`consumer_dropped`.

Record/replay runs before every other execution intercept, so no other plugin
can answer a model call around it.

Record/replay needs a Unix system. Do not combine it with tool caching, cache
bypass, header allowlists, a Redis backend, or a custom `priority`. Those
settings, and mistakes in the `record_replay` section, produce warnings and
leave model calls failing closed. Run plugin validation first and refuse a
configuration with warnings.

Traces do not store request headers, but prompts and answers can still contain
sensitive data. Use test or reviewed workloads. The plugin does not measure CPU
time; use Relay's `resource_metrics` plugin for that.

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

The unit tests check cache keys, storage, replay, marks, and record/replay
sessions without Relay. The integration tests build the library, load it into a
Relay host in the test process, and run real LLM and tool calls through it,
including a record and replay with caller timing. One test that leaves the
plugin loaded on purpose runs in its own test program.

The library names are:

| Platform | Library |
|---|---|
| macOS | `libnemo_relay_response_cache_plugin.dylib` |
| Linux | `libnemo_relay_response_cache_plugin.so` |
| Windows | `nemo_relay_response_cache_plugin.dll` |

Use [the release guide](../../RELEASE.md) to install a bundle and set Relay's
trust rules.
