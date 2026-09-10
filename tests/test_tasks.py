# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
import tomllib

import pytest

from scripts.bundles import sha256
from scripts.tasks import (
    Context,
    executable_filename,
    library_filename,
    run,
    write_runtime_manifest,
)


@pytest.mark.parametrize(
    "platform,library,executable",
    [
        ("linux-x86_64", "libcustom.so", "custom"),
        ("linux-arm64", "libcustom.so", "custom"),
        ("macos-arm64", "libcustom.dylib", "custom"),
        ("windows-x86_64", "custom.dll", "custom.exe"),
        ("windows-arm64", "custom.dll", "custom.exe"),
    ],
)
def test_platform_filenames(platform, library, executable):
    assert library_filename("custom", platform) == library
    assert executable_filename("custom", platform) == executable


def test_context_uses_registration_metadata_without_known_plugin_name(tmp_path, monkeypatch):
    plugin = tmp_path / "unrelated-plugin"
    plugin.mkdir()
    (plugin / "release.toml").write_text('[artifacts]\nbundle = "custom-distribution"\n')
    for name, path in {
        "PLUGIN_DIR": plugin,
        "SOURCE_DIR": tmp_path / "checkout" / "nested-crate",
        "SOURCE_ROOT": tmp_path / "checkout",
        "OUTPUT_DIR": tmp_path / "output",
        "CARGO_TARGET_DIR": tmp_path / "target",
    }.items():
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("PLUGIN_PLATFORM", "windows-arm64")

    context = Context.from_environment()
    assert context.plugin == plugin
    assert context.source == tmp_path / "checkout" / "nested-crate"
    assert context.source_root == tmp_path / "checkout"
    assert context.bundle == tmp_path / "output" / "custom-distribution"


def test_run_preserves_argument_boundaries_and_explicit_directory(tmp_path):
    argument = "spaces and $(no shell expansion)"
    run(
        [
            sys.executable,
            "-c",
            "import pathlib, sys; pathlib.Path('result').write_text(sys.argv[1])",
            argument,
        ],
        cwd=tmp_path,
    )
    assert (tmp_path / "result").read_text() == argument
    with pytest.raises(subprocess.CalledProcessError):
        run([sys.executable, "-c", "raise SystemExit(3)"], cwd=tmp_path)


def test_materialized_manifest_digests_the_packaged_artifact(tmp_path):
    artifact = tmp_path / "custom.bin"
    artifact.write_bytes(b"packaged content")
    manifest = {
        "plugin": {"id": "unrelated.runtime"},
        "source": {"artifact": "custom.bin"},
        "integrity": {"sha256": "placeholder"},
        "load": {"library": "custom.bin"},
    }
    write_runtime_manifest(tmp_path, manifest)
    result = tomllib.loads((tmp_path / "relay-plugin.toml").read_text())
    assert result["integrity"]["sha256"] == "sha256:" + sha256(artifact)
    assert result["plugin"]["id"] == "unrelated.runtime"
    assert result["load"]["library"] == "custom.bin"


def test_materialized_manifest_requires_declared_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_runtime_manifest(tmp_path, {"source": {"artifact": "absent"}, "integrity": {}})
