<!-- SPDX-License-Identifier: Apache-2.0 -->
# Migrate from Relay's built-in NeMo Guardrails component

Relay's dynamic worker replaces the supported subset of the deprecated
`nemo_guardrails` component. This guide is for users moving an existing local
or remote configuration during the Relay 0.9 removal window.

The worker is not a drop-in replacement. Review the unsupported behavior below
before removing the built-in component. The current Relay 0.9.0-rc.2 bundle is
for compatibility testing only; see the [README](README.md#status) for the
release requirement.

## Before you migrate

Record whether the old component uses:

- local or remote mode;
- input, output, or tool rails;
- streamed output enforcement;
- inline YAML or Colang;
- custom Python modules, paths, actions, or third-party packages;
- a custom codec; and
- remote generation, thread state, or request defaults.

Do not activate both implementations in one Relay process. Overlap can run
custom actions twice and apply different projection rules. The worker refuses
activation when it sees an active built-in `nemo_guardrails` component.

## Configuration mapping

| Built-in component | Dynamic worker |
|---|---|
| `[[components]] kind = "nemo_guardrails"` | Install the manifest with `plugins add --user`, edit its generated dynamic record, and enable `nemoguardrails.nemo_relay`. |
| Component `enabled` | Use `plugins enable` and `plugins disable`. |
| Component config `version = 1` | Use worker config `version = 2`. |
| `mode = "local"` | Remove `mode` and use an absolute `config_path`. |
| `config_path` | Keep the directory, make its path absolute, and retest it with Guardrails 0.24.1. |
| `config_yaml` | Write the YAML to `config.yml` in the Guardrails directory. |
| `colang_content` | Write Colang 1 content to a `.co` file in the directory. Colang 2 is not supported. |
| `codec` | Remove it. A release-ready Relay host supplies the request and response codec for each invocation. The worker supports OpenAI Chat, OpenAI Responses, Anthropic Messages, Gemini, and OCI GenAI. |
| `input = true` | Configure one or more `rails.input.flows`. The latest nonblank user turn is checked. |
| `output = true` | Configure one or more `rails.output.flows`. Complete unary responses are held until every supported candidate passes. |
| `input = false` / `output = false` | Omit that rail family. Keep at least one supported input, output, or structural tool flow. |
| `tool_input = true` | Configure `tool call validation` under `rails.tool_output`. This checks model-emitted function calls. |
| `tool_output = true` | Configure `tool result validation` under `rails.tool_input`. This checks linked results in later LLM history. |
| Other tool flows | No direct successor. The worker runs only the two structural flows above. |
| `priority` | Remove it. The worker owns its outer execution-interceptor priority. Later middleware must not replace checked content. |
| `local.python_module` | Remove it. The worker package owns the Guardrails import. |
| `local.python_executable` | Remove it. Relay provisions the managed Python environment. |
| `local.python_path` | No general replacement. Keep self-contained actions beside `config.yml`; package other dependencies in a custom worker build. |
| `mode = "remote"` | No automatic replacement. Use `remote_checks` only if stateless `/v1/checks` input/output decisions are sufficient. |
| `remote.*` | Reconfigure with `remote_checks.endpoint`, `config_ids`, `phases`, `model`, and optional `header_env`. Put credential values in `secret_env`. |
| `request_defaults.*` | No successor. Configure the evaluator model and prompts in the Guardrails configuration or remote service. |
| `policy.*` | No field-for-field mapping. Use the worker's fail-closed defaults and explicit payload, mutation, and acknowledgement settings. |
| `rails.output.streaming.*` / `stream_first` | No successor. Input-only streams can pass; output or model-call enforcement rejects before the provider stream opens. |
| Input or output `MODIFIED` | Rejects by default. Set the matching `mutation_policy` field to `apply` only for a lossless native text rewrite. |
| Guardrails 0.22 | Move to the pinned Guardrails 0.24.1 environment and retest every flow and custom action. |

The old component enabled input and output checks by default. The worker follows
the rail families in the Guardrails configuration, so confirm both phases
explicitly when you need the old default behavior.

## Behavior that does not carry over

| Old behavior or dependency | Migration result |
|---|---|
| Remote generation gateway, thread state, and request defaults | Not supported. `remote_checks` is a stateless policy client; Relay still owns generation. |
| Dialog rails, canonical conversation flows, or Colang 2 | Not supported by this worker's input/output attachment point. |
| Retrieval rails and factuality checks that need retrieved chunks | Not supported because the worker does not receive trusted retrieval context. |
| Streamed output enforcement | Not supported. The worker does not buffer, check, and replay generic provider streams. |
| Bedrock Converse, runtime, opaque, or custom codecs | Not supported when the configured policy needs that request or response direction. |
| Hosted tools, MCP calls, computer use, server tools, or free-form custom calls | Not covered by the structural function-tool adapter. |
| Arbitrary tool argument or result mutation | Not supported. Semantic tool checks are allow/block only. |
| Guardrails refusal responses | Not returned. Relay receives a policy rejection instead. |
| Guardrails metrics or tracing | Rejected during activation. |
| “Rejected output was never persisted” | Not guaranteed. An inner cache can write a fresh response before the outer output check rejects it. |

## Local migration

An old component may look like this:

```toml
[[components]]
kind = "nemo_guardrails"
enabled = true

[components.config]
version = 1
mode = "local"
config_path = "/absolute/path/to/guardrails-config"
codec = "openai_chat"
input = true
output = false
```

Install the worker using the [README instructions](README.md#install), then set
the generated record to:

```toml
[plugins.dynamic.config]
version = 2
config_path = "/absolute/path/to/guardrails-config"
```

Configure the worker as a required policy before enabling it:

```toml
[plugins.policy.overrides."nemoguardrails.nemo_relay"]
attestation = "integrity_only"
startup = "required"
```

If the old component used tool rails, add the supported structural flows to the
Guardrails directory:

```yaml
rails:
  tool_input:
    flows:
      - tool result validation
  tool_output:
    flows:
      - tool call validation
```

The names cross the old booleans: old `tool_input = true` maps to new
`rails.tool_output`, and old `tool_output = true` maps to new
`rails.tool_input`.

Custom `config.py` files run as trusted worker code. Async actions are
supported. Synchronous actions reject unless the migration explicitly sets
`action_safety.synchronous = "allow_unsafe"`; a blocked synchronous action can
stall the worker. See [`examples/migrated-local-rails`](examples/migrated-local-rails)
for a self-contained Colang 1 example.

## Remote migration

There is no automatic migration from old `mode = "remote"`. The old backend
was a generation gateway. The new remote backend calls Guardrails 0.24
`POST /v1/checks` around Relay's provider call.

Use it only when stateless input/output checks are sufficient. Follow the
[remote configuration](README.md#remote-checks) and explicitly acknowledge
that the stock checks server logs checked content and can retain conversation
events in memory. Keep the built-in component until another system replaces
any generation, thread-state, dialog, retrieval, or streamed-output behavior
your deployment still needs.

## Validate the replacement

Use a separate Relay configuration while testing. For every provider shape in
the deployment, verify:

- one allowed and one deterministically blocked latest-user input;
- allowed, blocked, and transformed unary output when output rails are used;
- input-only streaming and pre-provider rejection of output-guarded streaming;
- a valid function call, invalid arguments, and linked and orphaned results
  when structural tool rails are used;
- any multimodal or reasoning policy the deployment enables;
- timeout and evaluator-failure behavior; and
- provider calls that exceed 30 seconds once the release-target Relay host is
  available.

Do not cut over until the exact release archive, Relay version, Guardrails
configuration, evaluator, credentials, and representative payloads pass
together.

## Cut over and roll back

Disable the built-in component before enabling the worker. If the migration
fails, disable and remove the worker, then restore the previous Relay profile:

```bash
nemo-relay plugins disable nemoguardrails.nemo_relay
nemo-relay plugins remove nemoguardrails.nemo_relay
```

Removal deletes Relay's managed Python environment. It does not delete the
external Guardrails configuration directory.
