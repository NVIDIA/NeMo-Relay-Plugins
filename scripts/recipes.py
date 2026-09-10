# SPDX-License-Identifier: Apache-2.0
"""Build recipes selected by the four initial plugins' own task entrypoints."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import tomli_w

from scripts.bundles import sha256


def run(args, **kwargs):
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def run_recipe(name: str, stage: str):
    source = Path(os.environ["SOURCE_DIR"])
    source_root = Path(os.environ["SOURCE_ROOT"])
    output = Path(os.environ["OUTPUT_DIR"])
    platform = os.environ["PLUGIN_PLATFORM"]
    plugin = Path(os.environ["PLUGIN_DIR"])
    release = tomllib.loads((plugin / "release.toml").read_text(encoding="utf-8"))
    python_worker = name == "example-python-grpc-worker-plugin"
    remote = name == "switchyard-plugin"
    if stage == "smoke":
        from scripts.smoke import smoke

        smoke(Path(os.environ["BUNDLE_DIR"]), Path(os.environ["RELAY_BIN"]), remote)
        return
    if stage in ("build", "test"):
        if python_worker:
            if stage == "build":
                run(
                    [
                        "uv",
                        "sync",
                        "--locked",
                        "--python",
                        release["toolchains"]["python"],
                        "--group",
                        "test",
                        "--project",
                        source,
                    ]
                )
                run(["uv", "build", "--project", source, "--out-dir", output / "python-dist"])
            else:
                run(
                    [
                        "uv",
                        "run",
                        "--locked",
                        "--python",
                        release["toolchains"]["python"],
                        "--group",
                        "test",
                        "--project",
                        source,
                        "pytest",
                        "-c",
                        source / "pyproject.toml",
                        source / "tests",
                        "-q",
                    ]
                )
        else:
            args = ["cargo", stage, "--locked", "--release"]
            if remote:
                args += ["-p", "switchyard-nemo-relay-plugin"]
            run(args, cwd=source_root if remote else source)
        return
    if stage != "package":
        raise ValueError(f"unknown stage: {stage}")
    bundle = output / release["artifacts"]["bundle"]
    suffix = (
        ".dll"
        if platform.startswith("windows")
        else ".dylib"
        if platform.startswith("macos")
        else ".so"
    )
    prefix = "" if platform.startswith("windows") else "lib"
    if remote:
        library = (
            Path(os.environ["CARGO_TARGET_DIR"])
            / "release"
            / (prefix + "switchyard_nemo_relay_plugin" + suffix)
        )
        run(
            [
                sys.executable,
                source / "scripts/package_bundle.py",
                "--library",
                library,
                "--output",
                bundle,
            ]
        )
        return
    bundle.mkdir(parents=True)
    manifest = tomllib.loads((source / "relay-plugin.toml").read_text(encoding="utf-8"))
    if python_worker:
        package = "nemo_relay_python_grpc_worker_example"
        shutil.copytree(
            source / package,
            bundle / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for filename in ["pyproject.toml", "uv.lock"]:
            shutil.copy2(source / filename, bundle / filename)
        artifact = bundle / manifest["source"]["artifact"]
    else:
        if release["type"] == "native":
            filename = prefix + "nemo_relay_rust_native_plugin_example" + suffix
            manifest["load"]["library"] = filename
        else:
            filename = "nemo-relay-rust-grpc-worker-plugin-example" + (
                ".exe" if platform.startswith("windows") else ""
            )
            manifest["load"]["entrypoint"] = filename
        artifact = bundle / filename
        shutil.copy2(Path(os.environ["CARGO_TARGET_DIR"]) / "release" / filename, artifact)
        manifest["source"]["artifact"] = filename
    manifest["integrity"]["sha256"] = "sha256:" + sha256(artifact)
    for filename in ["config.schema.json", "LICENSE", "NOTICE"]:
        if (source / filename).exists():
            shutil.copy2(source / filename, bundle / filename)
    (bundle / "relay-plugin.toml").write_text(tomli_w.dumps(manifest))
