<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Original source

This plugin is the response cache that used to be part of NVIDIA NeMo Relay's
adaptive plugin. It moved here from the Relay 1.0 source tree at
`crates/adaptive/src/response_cache` and its tests.

Changes made in the move:

- The cache is its own native plugin with the ID `nvidia.response_cache`. Its
  configuration keys are the same as the old adaptive `response_cache` section.
- Codec work uses the `nemo-relay-codecs` crate. Cost estimates come from
  Relay through `PluginRuntime`, so they use Relay's active pricing catalog.
- `config.schema.json` replaces Relay's configuration-editor metadata. Unknown
  fields now fail validation instead of producing warnings.
- Integration tests load the built library into an in-process Relay host.
