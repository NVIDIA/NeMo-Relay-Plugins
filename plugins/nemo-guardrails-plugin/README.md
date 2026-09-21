<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<!-- SPDX-License-Identifier: Apache-2.0 -->
# NeMo Guardrails for NeMo Relay

This plugin runs NeMo Guardrails 0.24.1 as a Relay-managed Python worker. It
checks LLM and tool traffic while Relay continues to own routing, caching,
provider calls, tool execution, and observability.

## What it supports

| Capability | Support | Notes |
|---|---|---|
| Input text rails | Supported | Checks the latest nonblank user turn. Earlier conversation text is context, not a separate check. |
| Unary output text rails | Supported | Holds the response and checks every fully represented candidate before returning it. |
| Text transformation | Conditional | `MODIFIED` rejects by default. Opt-in mutation applies only when the replacement maps losslessly to exact native text fields. |
| Streaming | Input only | Input and prior-result checks run before the stream opens. Output or model-call enforcement rejects before the provider. |
| Structural function tools | Supported | Checks emitted ordinary function calls against their declarations and validates call/result structure and linkage. Unused declarations are not prevalidated. |
| Semantic tool checks | Conditional | Can check bounded JSON around Relay-managed tools or earlier transcript results. The plugin never executes tools or rewrites JSON. |
| Multiple candidates | Supported | Every candidate is checked when the provider payload can be represented completely. |
| Multimodal payloads | Text only | Recognized media can be rejected or preserved outside the text-policy claim. Images, audio, and files are not inspected. |
| Reasoning and thinking | Conditional | Can reject it, preserve it unchecked, or check visible text with ordinary output rails. This is not native `$bot_thinking`. |
| Custom actions | Async supported | Synchronous actions reject by default and require an explicit unsafe opt-in. |
| Remote actions | Conditional | Supports loopback HTTP or credential-free HTTPS. Guardrails can report transport failures as `BLOCKED`. |
| Observability | Best effort | Emits content-free policy-decision marks. They share the Relay turn but are not linked to an exact LLM invocation. |

The plugin returns provider payloads unchanged after a pass. It fails closed on
policy blocks, timeouts, check failures, and payloads it cannot inspect safely.

### Application providers

| Relay codec | Text | Ordinary function tools | Main limitation |
|---|---|---|---|
| OpenAI Chat Completions | Input and unary output | Supported | Legacy `functions` and `function_call` are not supported. |
| OpenAI Responses | Input and unary output | Supported | Hosted and free-form custom tools are not supported. |
| Anthropic Messages | Input, unary output, and `count_tokens` input checks | Client functions | Server tools, MCP tools, and computer use are not supported. |
| Gemini `generateContent` | Input and unary output | Supported | Generator-style tool responses are not supported. |
| OCI Generative AI | `GENERIC`, `COHERE`, and `COHEREV2` | Supported subset | COHERE v1 result linkage is not supported. |
| Bedrock Converse | Not supported | Not supported | The worker has no Bedrock projector. |

Custom, runtime, opaque, unknown, and malformed codecs are unsupported. They
reject when a configured policy needs to inspect that request or response
direction. Request and response codecs are independent, so supported router
transcodes can still be checked in both directions.

### Guardrails backends and evaluators

Choose one backend:

- **Local configuration:** loads one Guardrails directory when the worker
  starts. It supports stateless input/output checks, Colang 1 flows, structural
  function rails, and trusted custom actions.
- **Remote `/v1/checks`:** calls a separately operated Guardrails service for
  stateless input and output decisions. It does not provide structural tool
  rails, local actions, dialog state, retrieval state, or generation.

The evaluator can differ from the application provider. The base archive
supports OpenAI-compatible evaluators, including compatible NVIDIA NIM
endpoints. Native Anthropic evaluation and catalog integrations that require
Presidio, YARA, Hugging Face, Cleanlab, Google Cloud moderation, Guardrails AI,
or other optional packages are not included in the standard archive. A custom
build may supply compatible dependencies, but those combinations are outside
this release's support and test matrix.

## Known limits

This worker does not support:

- dialog rails, stateful conversation flows, or Colang 2;
- retrieval rails or checks that require trusted retrieved chunks;
- Guardrails-owned generation, refusal responses, or thread state;
- output-guarded streaming, speculative generation, or multi-step generation;
- native image, audio, or file inspection;
- provider-native hosted tools, MCP calls, computer use, server tools, other
  non-function tool protocols, or tool activity hidden behind an MCP server;
- arbitrary mutation of tool arguments or results;
- per-request local configuration, tenant, rail-set, or context selection;
- Guardrails metrics or tracing;
- Relay raw worker delivery, which bypasses execution interceptors;
- a guarantee that rejected output was never written to an inner cache; or
- running beside Relay's deprecated built-in Guardrails component.

Some Guardrails catalog rails also need context that this attachment point does
not receive. This includes all retrieval rails and output factuality checks such
as self-check facts, self-check hallucination, AlignScore, AutoAlign, Fiddler
faithfulness, and Patronus factuality checks.

For the pinned 0.24.1 catalog, this attachment admits 31 of 32 input surfaces,
28 of 35 output surfaces, and none of the 11 retrieval surfaces. The missing
input surface is `jailbreak detection heuristics`. “Admitted” means the rail can
attach here; it does not mean its optional package, model, or service is bundled
or qualified for production.

## How it works

```text
application request
        |
Relay request interceptors
        |
Guardrails input and prior-result checks
        |
router / cache / provider
        |
Guardrails unary output and function-call checks
        |
application response
```

The plugin always registers one outer LLM execution interceptor. Semantic tool
checking adds one outer tool-execution interceptor. Input checks run before
inner routers such as Switchyard, while output checks see the routed unary
response. Middleware that changes already-checked content later in the chain is
part of the deployment trust boundary.

## Install

Verify the downloaded archive against its `.sha256` sidecar, extract it to a
directory you will keep, then install its runtime manifest:

```bash
nemo-relay plugins validate ./nemo-guardrails-plugin/relay-plugin.toml
nemo-relay plugins add --user ./nemo-guardrails-plugin/relay-plugin.toml
nemo-relay plugins edit
```

Configure Relay to fail startup rather than run without the requested policy:

```toml
[plugins.policy.overrides."nemoguardrails.nemo_relay"]
attestation = "integrity_only"
startup = "required"
```

Use `plugins add`; a hand-written `[[plugins.dynamic]]` record does not create
the Python worker's managed environment. Complete the configuration below
before enabling the plugin.

## Configure

Choose exactly one backend: `config_path` or `remote_checks`.

### Local Guardrails

Point the generated plugin entry at an existing absolute directory:

```toml
[plugins.dynamic.config]
version = 2
config_path = "/absolute/path/to/guardrails-config"
```

Restart or reactivate the plugin after changing that directory. The worker does
not watch it. The directory must configure at least one supported input rail,
output rail, or exact structural flow: `tool result validation` or `tool call
validation`.

### Remote checks

Use `remote_checks` instead of `config_path`:

```toml
[plugins.dynamic.config]
version = 2

[plugins.dynamic.config.remote_checks]
endpoint = "https://guardrails.example.com"
config_ids = ["policy-a"]
phases = ["input", "output"]
model = "guardrails-evaluator"
allow_remote_content_logging_and_retention = true
```

For an authenticated service, map HTTP headers to worker environment values:

```toml
[plugins.dynamic.config.remote_checks.header_env]
Authorization = "GUARDRAILS_AUTHORIZATION"

[plugins.dynamic.config.secret_env]
GUARDRAILS_AUTHORIZATION = "Bearer replace-with-the-deployment-secret"
```

The acknowledgement is required because the stock Guardrails 0.24 checks
server logs checked content and can retain conversation events in memory.
Treat that service as a separate content trust boundary.
`remote_checks.timeout_ms` defaults to 25,000 and cannot exceed
`check_timeout_ms`. Its response limit defaults to 65,536 bytes.

### Common options

| Setting | Default | Purpose |
|---|---|---|
| `check_timeout_ms` | `25000` | Policy timeout from 1,000 to 25,000 ms |
| `payload_policy.multimodal` | `strict` | `strict` or `text_only` |
| `payload_policy.reasoning` | `reject` | `reject`, `final_answer_only`, or `check_output` |
| `mutation_policy.input` / `.output` | `reject` | Set either to `apply` for safe text-only mutation |
| `semantic_tool_policy.check_arguments` | `false` | Check canonical tool-argument JSON |
| `semantic_tool_policy.result_source` | `off` | `off`, `execution`, `history`, or `both` |
| `evaluator_framework` | `default` | Local input/output rails only. Use `langchain` for native Anthropic evaluation. |
| `action_safety.synchronous` | `reject` | Local LLMRails input/output only. `allow_unsafe` permits synchronous custom actions. |
| `allow_ignored_rail_families` | `false` | Acknowledge configured rail families that will not run |
| `allow_known_fail_open_rails` | `false` | Acknowledge Guardrails rails with known fail-open behavior |
| `secret_env` | Empty | Environment values exposed only to the worker process |

Managed workers do not inherit arbitrary credentials from Relay. Add only the
values the Guardrails configuration needs:

```toml
[plugins.dynamic.config.secret_env]
NVIDIA_API_KEY = "replace-with-the-deployment-secret"
```

These values remain plaintext in Relay's plugin TOML. Restrict file access and
never commit credentials.

Options that inspect a phase require that phase to exist. `check_output` and
output mutation require output rails. Input mutation and semantic argument
checks require input rails. Semantic result checks require output rails.

After configuring the backend and policy, enable and validate the plugin:

```bash
nemo-relay plugins enable nemoguardrails.nemo_relay
nemo-relay plugins validate nemoguardrails.nemo_relay
```

## Examples

| Example | Purpose |
|---|---|
| [`no-model-rails`](examples/no-model-rails) | First install and policy-path smoke test |
| [`input-output-rails`](examples/input-output-rails) | Input and unary output text rails |
| [`structural-tool-rails`](examples/structural-tool-rails) | Ordinary function-call and result validation |
| [`migrated-local-rails`](examples/migrated-local-rails) | Colang 1 with a trusted custom Python action |

## Security and operations

- Local `config.py` files and imported actions are trusted code. They can read
  credentials, access files, log content, and make network calls.
- Evaluator traffic must not route back through the same guarded Relay policy;
  recursive LLMRails checks can deadlock the worker.
- Anonymous Guardrails usage reporting is disabled. Guardrails metrics and
  tracing configurations reject during activation.
- Timeouts are cooperative and cannot stop synchronous Python code that has
  blocked the worker event loop.
- The worker admits at most 32 text checks and 32 structural validations at a
  time. Structural validation uses at most four worker threads.
- Text projection is limited to 512 messages and 1,000,000 characters. Output
  checks reserve part of that budget for the held assistant response.
- Policy marks contain decisions and timing, not prompts, responses, tool data,
  errors, or credentials.
- The external Guardrails configuration directory is not covered by the plugin
  archive checksum. Version and protect it separately.

## Compatibility

| Item | Current value |
|---|---|
| NeMo Guardrails | Exactly 0.24.1 |
| Relay API and Python SDK | 0.9.0; CI host temporarily includes [NeMo-Relay#1110](https://github.com/NVIDIA/NeMo-Relay/pull/1110) |
| Python | 3.11–3.13 |
| Declared release targets | Linux x86-64/ARM64, Windows x86-64, macOS ARM64 |
| Plugin ID | `nemoguardrails.nemo_relay` |

See [MIGRATION.md](MIGRATION.md) before replacing an existing built-in
`nemo_guardrails` component.

## Develop

From the repository root:

```bash
uv run --locked python -m scripts.plugins run nemo-guardrails-plugin
```

This runs package tests, builds the archive, installs it with Relay, exercises
the installed bundle, checks tamper rejection, and removes the plugin.

See the [NeMo Guardrails documentation](https://docs.nvidia.com/nemo/guardrails/),
[NeMo Relay](https://github.com/NVIDIA/NeMo-Relay), and the repository
[release guide](https://github.com/NVIDIA/NeMo-Relay-Plugins/blob/main/RELEASE.md)
for the surrounding systems and release process.
