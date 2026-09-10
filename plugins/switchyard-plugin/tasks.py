# SPDX-License-Identifier: Apache-2.0
"""Build, test, and package this plugin independently of other registrations."""

import argparse
import os
import sys

sys.path.insert(0, os.environ["REPO_DIR"])
from scripts.tasks import Context, library_filename, run


CRATE = "switchyard-nemo-relay-plugin"


def build(ctx: Context):
    run(["cargo", "build", "--locked", "--release", "-p", CRATE], cwd=ctx.source_root)


def test(ctx: Context):
    run(["cargo", "test", "--locked", "--release", "-p", CRATE], cwd=ctx.source_root)


def package(ctx: Context):
    library = (
        ctx.target / "release" / library_filename("switchyard_nemo_relay_plugin", ctx.platform)
    )
    run(
        [
            sys.executable,
            ctx.source / "scripts/package_bundle.py",
            "--library",
            library,
            "--output",
            ctx.bundle,
        ],
        cwd=ctx.source_root,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "test", "package"])
    args = parser.parse_args()
    {"build": build, "test": test, "package": package}[args.stage](Context.from_environment())
