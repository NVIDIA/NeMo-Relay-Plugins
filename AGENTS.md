# Repository guidance

- Never comment on or respond to GitHub/GitLab PRs, MRs, or issues without explicit approval.
- Each `plugins/<name>/release.toml` defines that plugin's release settings. `relay-plugin.toml` tells Relay how to load it.
- Keep each plugin's commands, packaging, and smoke tests in its folder. Shared scripts must not choose behavior based on plugin names or contain a plugin's package or output filenames.
- Run `uv run --group test pytest` for shared tooling changes and `uv run python -m scripts.plugins validate` for manifests.
- Use `uv run python -m scripts.plugins run <name>` for a complete local plugin build, tests, packaging, and installed-bundle smoke test.
- Keep plugin SDK lockfiles separate from the setting that chooses the Relay test host.
- Always build and test Rust in release mode. This includes builds started by tests that load and shut down a plugin.
- Never skip required platform tests or publish a draft automatically.
- Pin external GitHub Actions to the full commit SHA of their latest stable release. Record the exact release tag in a comment.
- Prefer NeMo Relay’s minimum supported Python version (currently 3.11) for CI and plugin toolchains.
- Write README files and documentation for a high-school reader. Use short sentences, explain technical terms when first used, and keep commands exact. Keep license terms and required copyright notices verbatim.
