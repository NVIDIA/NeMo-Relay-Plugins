# SPDX-License-Identifier: Apache-2.0
"""Build, test, and package this plugin independently of other registrations."""

import argparse
import os
import sys
import shutil
import tomllib

sys.path.insert(0, os.environ["REPO_DIR"])
from scripts.tasks import Context, executable_filename, run, write_runtime_manifest


def build(ctx: Context):
    run(["cargo", "build", "--locked", "--release"], cwd=ctx.source)


def test(ctx: Context):
    run(["cargo", "test", "--locked", "--release"], cwd=ctx.source)


def package(ctx: Context):
    ctx.bundle.mkdir(parents=True)
    filename = executable_filename("nemo-relay-rust-grpc-worker-plugin-example", ctx.platform)
    shutil.copy2(ctx.target / "release" / filename, ctx.bundle / filename)
    manifest = tomllib.loads((ctx.source / "relay-plugin.toml").read_text(encoding="utf-8"))
    manifest["source"]["artifact"] = filename
    manifest["load"]["entrypoint"] = filename
    for filename in [
        "config.schema.json",
        "LICENSE",
        "NOTICE",
        "UPSTREAM.md",
        "ATTRIBUTIONS-Rust.md",
    ]:
        shutil.copy2(ctx.source / filename, ctx.bundle / filename)
    write_runtime_manifest(ctx.bundle, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "test", "package"])
    args = parser.parse_args()
    {"build": build, "test": test, "package": package}[args.stage](Context.from_environment())
