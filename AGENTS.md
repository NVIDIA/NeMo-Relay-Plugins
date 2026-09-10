# Repository guidance

- Never comment on or respond to GitHub/GitLab PRs, MRs, or issues without explicit approval.
- Each `plugins/<name>/release.toml` owns its release metadata. `relay-plugin.toml` is the runtime contract.
- Run `uv run --group test pytest` for shared tooling changes and `uv run python -m scripts.plugins validate` for manifests.
- Use `uv run python -m scripts.plugins run <name>` for a complete local plugin build, tests, packaging, and installed-bundle smoke test.
- Keep plugin SDK lockfiles independent of the Relay test-host selector.
- Never skip required platform tests or publish a draft automatically.
