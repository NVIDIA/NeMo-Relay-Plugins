# SPDX-License-Identifier: Apache-2.0
"""Build, test, and package this plugin independently of other registrations."""

import argparse
import os
import sys
import shutil
import tomllib

sys.path.insert(0, os.environ["REPO_DIR"])
from scripts.tasks import Context, run, write_runtime_manifest


def build(ctx: Context):
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
        ["uv", "build", "--project", ctx.source, "--out-dir", ctx.output / "python-dist"],
        cwd=ctx.source,
    )


def test(ctx: Context):
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


def package(ctx: Context):
    ctx.bundle.mkdir(parents=True)
    package = "nemo_relay_python_grpc_worker_example"
    shutil.copytree(
        ctx.source / package,
        ctx.bundle / package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for filename in ["pyproject.toml", "uv.lock", "config.schema.json", "LICENSE"]:
        shutil.copy2(ctx.source / filename, ctx.bundle / filename)
    if (ctx.source / "NOTICE").exists():
        shutil.copy2(ctx.source / "NOTICE", ctx.bundle / "NOTICE")
    manifest = tomllib.loads((ctx.source / "relay-plugin.toml").read_text(encoding="utf-8"))
    write_runtime_manifest(ctx.bundle, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "test", "package"])
    args = parser.parse_args()
    {"build": build, "test": test, "package": package}[args.stage](Context.from_environment())
