# SPDX-License-Identifier: Apache-2.0
"""An unrelated registration can run without changing any shared recipe or registry."""

import json
import sys
from pathlib import Path

from scripts import plugins


def test_unrelated_plugin_owns_every_stage(tmp_path, monkeypatch):
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
with (plugin / "stages").open("a") as log:
    log.write(stage + "\\n")
if stage == "package":
    bundle = Path(os.environ["OUTPUT_DIR"]) / "custom-bundle"
    bundle.mkdir()
    (bundle / "custom-worker").write_bytes(b"custom artifact")
    (bundle / "ATTRIBUTIONS.md").write_text("Fixture license text")
    digest = hashlib.sha256(b"custom artifact").hexdigest()
    (bundle / "runtime.toml").write_text(
        '[plugin]\\nkind = "worker"\\nid = "custom.runtime"\\n'
        '[source]\\nartifact = "custom-worker"\\n'
        '[load]\\nruntime = "rust"\\nentrypoint = "custom-worker"\\n'
        '[integrity]\\nsha256 = "sha256:' + digest + '"\\n'
    )
elif stage == "smoke":
    bundle = Path(os.environ["BUNDLE_DIR"])
    assert not bundle.is_relative_to(Path(os.environ["REPO_DIR"]))
    assert (bundle / "custom-worker").read_bytes() == b"custom artifact"
""",
        encoding="utf-8",
    )
    platform = plugins.local_platform()
    manifest = {
        "name": name,
        "version": "1.2.3",
        "type": "worker",
        "platforms": [platform],
        "source": {"location": "in-tree", "path": "."},
        "toolchains": {"python": "3.11"},
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
