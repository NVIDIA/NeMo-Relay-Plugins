# SPDX-License-Identifier: Apache-2.0
"""Build, test, and package the NeMo Guardrails worker."""

import argparse
import os
import shutil
import sys
import tomllib

sys.path.insert(0, os.environ["REPO_DIR"])
from scripts.tasks import (
    Context,
    package_locked_python_project,
    run,
    write_attributions,
    write_runtime_manifest,
)


def build(ctx: Context) -> None:
    run(
        [
            "uv",
            "sync",
            "--locked",
            "--python",
            ctx.release["toolchains"]["python"],
            "--group",
            "test",
            "--project",
            ctx.source,
        ],
        cwd=ctx.source,
    )
    run(
        [
            "uv",
            "build",
            "--wheel",
            "--project",
            ctx.source,
            "--out-dir",
            ctx.output / "python-dist",
            "--build-constraints",
            "build-constraints.txt",
            "--require-hashes",
        ],
        cwd=ctx.source,
    )


def test(ctx: Context) -> None:
    run(
        [
            "uv",
            "run",
            "--locked",
            "--python",
            ctx.release["toolchains"]["python"],
            "--group",
            "test",
            "--project",
            ctx.source,
            "pytest",
            "-c",
            ctx.source / "pyproject.toml",
            ctx.source / "tests",
            "-q",
        ],
        cwd=ctx.source,
    )


def package(ctx: Context) -> None:
    ctx.bundle.mkdir(parents=True)
    package_name = "nemoguardrails_nemo_relay"
    shutil.copytree(
        ctx.source / package_name,
        ctx.bundle / package_name,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for filename in [
        "uv.lock",
        "config.schema.json",
        "LICENSE",
        "NOTICE",
        "README.md",
        "MIGRATION.md",
    ]:
        shutil.copy2(ctx.source / filename, ctx.bundle / filename)
    shutil.copytree(
        ctx.source / "examples",
        ctx.bundle / "examples",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    manifest = tomllib.loads((ctx.source / "relay-plugin.toml").read_text(encoding="utf-8"))
    wheels = list((ctx.output / "python-dist").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("build must produce exactly one project wheel")
    package_locked_python_project(ctx, wheels[0], manifest)
    write_runtime_manifest(ctx.bundle, manifest)
    write_attributions(ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "test", "package"])
    args = parser.parse_args()
    {"build": build, "test": test, "package": package}[args.stage](Context.from_environment())
