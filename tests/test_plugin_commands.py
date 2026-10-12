# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An unrelated registration can run without changing any shared recipe or registry."""

import json
import sys
from pathlib import Path

import pytest

from scripts import plugins


@pytest.mark.parametrize("encoded", [None, "", "-C\x1fopt-level=2"])
@pytest.mark.parametrize(
    "kind,platform,rust,flag",
    [
        ("worker", plugins.local_platform(), None, ""),
        ("native", "linux-x86_64", "1.96.1", "link-arg=-Wl,-z,nodelete"),
        ("native", "linux-arm64", "1.96.1", "link-arg=-Wl,-z,nodelete"),
        ("native", "linux-musl-x86_64", "1.96.1", "target-feature=-crt-static"),
        ("native", "linux-musl-arm64", "1.96.1", "target-feature=-crt-static"),
        ("worker", "linux-x86_64", "1.96.1", ""),
    ],
)
def test_unrelated_plugin_owns_every_stage(
    tmp_path, monkeypatch, kind, platform, rust, flag, encoded
):
    name = "unrelated-custom-plugin"
    folder = tmp_path / "plugins" / name
    folder.mkdir(parents=True)
    (folder / "custom-command.py").write_text(
        """import hashlib, os, sys
from pathlib import Path

stage = sys.argv[1]
plugin = Path(os.environ["PLUGIN_DIR"])
assert Path.cwd() == plugin
assert sys.argv[2] == "literal argument with spaces"
flags = os.environ.get("RUSTFLAGS", "") + os.environ.get("CARGO_ENCODED_RUSTFLAGS", "")
expected_flag = os.environ["EXPECTED_RUST_FLAG"]
if expected_flag:
    assert expected_flag in flags
    if "nodelete" in expected_flag and os.environ.get("EXPECTED_ENCODED"):
        assert os.environ["EXPECTED_ENCODED"] in flags
else:
    assert "nodelete" not in flags and "crt-static" not in flags
native = os.environ["EXPECTED_KIND"] == "native"
with (plugin / "stages").open("a") as log:
    log.write(stage + "\\n")
if stage == "package":
    bundle = Path(os.environ["OUTPUT_DIR"]) / "custom-bundle"
    bundle.mkdir()
    (bundle / "custom-worker").write_bytes(b"custom artifact")
    (bundle / "ATTRIBUTIONS.md").write_text("Fixture license text")
    digest = hashlib.sha256(b"custom artifact").hexdigest()
    runtime_kind = "rust_dynamic" if native else "worker"
    load = 'library = "custom-worker"' if native else 'runtime = "rust"\\nentrypoint = "custom-worker"'
    (bundle / "runtime.toml").write_text(
        '[plugin]\\nkind = "' + runtime_kind + '"\\nid = "custom.runtime"\\n'
        '[source]\\nartifact = "custom-worker"\\n'
        '[load]\\n' + load + '\\n'
        '[integrity]\\nsha256 = "sha256:' + digest + '"\\n'
    )
elif stage == "smoke":
    bundle = Path(os.environ["BUNDLE_DIR"])
    assert not bundle.is_relative_to(Path(os.environ["REPO_DIR"]))
    assert (bundle / "custom-worker").read_bytes() == b"custom artifact"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(plugins, "local_platform", lambda: platform)
    monkeypatch.setenv("EXPECTED_KIND", kind)
    monkeypatch.setenv("EXPECTED_RUST_FLAG", flag)
    # Cargo prioritizes encoded flags. Verify that the GNU workaround is added
    # without discarding the caller's existing options.
    monkeypatch.setenv("EXPECTED_ENCODED", encoded or "")
    if encoded is None:
        monkeypatch.delenv("CARGO_ENCODED_RUSTFLAGS", raising=False)
    else:
        monkeypatch.setenv("CARGO_ENCODED_RUSTFLAGS", encoded)
    monkeypatch.delenv("RUSTFLAGS", raising=False)
    manifest = {
        "name": name,
        "version": "1.2.3",
        "type": kind,
        "platforms": [platform],
        "source": {"location": "in-tree", "path": "."},
        "toolchains": {"python": "3.11", **({"rust": rust} if rust else {})},
        "artifacts": {"bundle": "custom-bundle", "manifest": "runtime.toml"},
        "commands": {
            stage: {
                "argv": [
                    "${PYTHON}",
                    "${PLUGIN_DIR}/custom-command.py",
                    stage,
                    "literal argument with spaces",
                ],
                "cwd": "plugin",
            }
            for stage in ["build", "test", "package", "smoke"]
        },
    }
    monkeypatch.setattr(plugins, "ROOT", tmp_path)
    monkeypatch.setattr(plugins, "git", lambda *args: "")
    check_output = plugins.subprocess.check_output

    def tool_output(argv, **kwargs):
        if argv[:2] == ["uv", "python"]:
            return sys.executable + "\n"
        if argv == ["uv", "--version"]:
            return "uv fixture\n"
        if argv == ["rustc", "--version"]:
            return "rustc fixture\n"
        return check_output(argv, **kwargs)

    monkeypatch.setattr(plugins.subprocess, "check_output", tool_output)
    output = plugins.run_plugin(
        manifest,
        platform,
        {"sha": "a" * 40, "tag": None},
        "b" * 40,
        relay=Path(sys.executable),
    )
    assert (folder / "stages").read_text().splitlines() == ["build", "test", "package", "smoke"]
    metadata = json.loads(next(output.glob("*.json")).read_text())
    assert metadata["name"] == name
    assert metadata["verified"] is True
    assert metadata["local_host_override"] is True
