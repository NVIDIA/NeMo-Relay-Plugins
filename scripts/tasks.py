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

    if ctx.release["source"]["location"] in {"wheel", "crate"}:
        from scripts.licensing.packages import collect_packages

        documents, _ = collect_packages(
            ctx.plugin,
            ctx.release,
            ctx.plugin.parent.parent / ".cache/packages",
            platforms=[ctx.platform],
        )
        ctx.bundle.mkdir(parents=True, exist_ok=True)
        for language, text in documents.items():
            (ctx.bundle / f"ATTRIBUTIONS-{language}.md").write_text(
                text, encoding="utf-8", newline="\n"
            )
        return
    write_project_attributions(
        ctx.source_root,
        ctx.source,
        ctx.bundle,
        toolchain=ctx.release["toolchains"].get("rust"),
        include_workspace=ctx.release["source"]["location"] == "remote",
    )


def package_wheel(ctx: Context):
    """Materialize a wheel as an installable Relay bundle without changing its bytes."""
    import shutil
    from scripts import wheel_backend
    from scripts.licensing.python_local import checked_path

    shutil.copytree(ctx.source_root, ctx.bundle)
    original = checked_path(ctx.source_root, ctx.source_root, ctx.release["source"]["manifest"])
    manifest = tomllib.loads(original.read_text(encoding="utf-8"))
    for section, field in [
        ("source", "artifact"),
        ("source", "manifest_root"),
        ("config_schema", "path"),
        ("load", "library"),
    ]:
        if field in manifest.get(section, {}):
            manifest[section][field] = (
                checked_path(ctx.source_root, original.parent, manifest[section][field])
                .relative_to(ctx.source_root)
                .as_posix()
            )
    if manifest["load"].get("runtime") == "python":
        manifest["source"]["manifest_root"] = "."
        for name in [
            "pyproject.toml",
            "relay_wheel_backend.py",
            "package-source.json",
            "wheelhouse",
        ]:
            if (ctx.bundle / name).exists():
                raise ValueError(f"wheel uses a reserved adapter path: {name}")
        shutil.copytree(ctx.output / "wheels", ctx.bundle / "wheelhouse")
        shutil.copy2(ctx.output / "package-source.json", ctx.bundle / "package-source.json")
        shutil.copy2(wheel_backend.__file__, ctx.bundle / "relay_wheel_backend.py")
        notices = ctx.bundle / "notices/relay-wheel-adapter"
        notices.mkdir(parents=True)
        repository = Path(__file__).resolve().parents[1]
        for filename in ["LICENSE", "NOTICE"]:
            if (repository / filename).is_file():
                shutil.copy2(repository / filename, notices / filename)
        (ctx.bundle / "pyproject.toml").write_text(
            tomli_w.dumps(
                {
                    "build-system": {
                        "requires": [],
                        "build-backend": "relay_wheel_backend",
                        "backend-path": ["."],
                    }
                }
            ),
            encoding="utf-8",
        )
    elif "entrypoint" in manifest["load"]:
        manifest["load"]["entrypoint"] = (
            checked_path(ctx.source_root, original.parent, manifest["load"]["entrypoint"])
            .relative_to(ctx.source_root)
            .as_posix()
        )
    if ctx.release["artifacts"]["manifest"] != "relay-plugin.toml":
        raise ValueError("package_wheel writes relay-plugin.toml at the bundle root")
    write_runtime_manifest(ctx.bundle, manifest)
    write_attributions(ctx)
