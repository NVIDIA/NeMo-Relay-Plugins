# SPDX-License-Identifier: Apache-2.0
"""Optional primitives for plugin-owned build scripts; no plugin registry or recipes."""

import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

import tomli_w

from scripts.bundles import sha256


@dataclass
class Context:
    plugin: Path
    source: Path
    source_root: Path
    output: Path
    target: Path
    platform: str
    release: dict

    @classmethod
    def from_environment(cls):
        plugin = Path(os.environ["PLUGIN_DIR"])
        return cls(
            plugin=plugin,
            source=Path(os.environ["SOURCE_DIR"]),
            source_root=Path(os.environ["SOURCE_ROOT"]),
            output=Path(os.environ["OUTPUT_DIR"]),
            target=Path(os.environ["CARGO_TARGET_DIR"]),
            platform=os.environ["PLUGIN_PLATFORM"],
            release=tomllib.loads((plugin / "release.toml").read_text(encoding="utf-8")),
        )

    @property
    def bundle(self) -> Path:
        return self.output / self.release["artifacts"]["bundle"]


def run(argv, *, cwd: Path):
    subprocess.run([str(arg) for arg in argv], cwd=cwd, check=True)


def library_filename(stem: str, platform: str) -> str:
    if platform.startswith("windows-"):
        return stem + ".dll"
    return "lib" + stem + (".dylib" if platform.startswith("macos-") else ".so")


def executable_filename(stem: str, platform: str) -> str:
    return stem + (".exe" if platform.startswith("windows-") else "")


def write_runtime_manifest(bundle: Path, manifest: dict, filename: str = "relay-plugin.toml"):
    """Digest the declared artifact after packaging and write the runtime manifest."""
    manifest["integrity"]["sha256"] = "sha256:" + sha256(bundle / manifest["source"]["artifact"])
    (bundle / filename).write_text(tomli_w.dumps(manifest), encoding="utf-8")


def write_attributions(ctx: Context):
    """Generate this plugin's notices from its own source and lockfiles."""
    from scripts.licensing.generate import write_project_attributions

    write_project_attributions(
        ctx.source_root,
        ctx.source,
        ctx.bundle,
        toolchain=ctx.release["toolchains"].get("rust"),
    )
