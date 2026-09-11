# SPDX-License-Identifier: Apache-2.0
"""Local checks shared by pre-commit and CI, using each plugin's own settings."""

import argparse
import hashlib
import re
import subprocess
import tomllib
from pathlib import Path

from scripts.catalog import ROOT, discover, inside


def in_tree_sources(root: Path):
    for name, manifest in discover(root).items():
        if manifest["source"]["location"] == "in-tree":
            yield inside(root / "plugins" / name, manifest["source"]["path"]), manifest


def rust_fmt(root: Path):
    for source, manifest in in_tree_sources(root):
        if (source / "Cargo.toml").exists():
            subprocess.run(
                ["cargo", f"+{manifest['toolchains']['rust']}", "fmt", "--all"],
                cwd=source,
                check=True,
            )


def lockfiles(root: Path):
    subprocess.run(["uv", "lock", "--check"], cwd=root, check=True)
    for source, manifest in in_tree_sources(root):
        if (source / "pyproject.toml").exists():
            subprocess.run(["uv", "lock", "--check"], cwd=source, check=True)
        if (source / "Cargo.toml").exists():
            # Resolving the full graph is necessary: --no-deps can overlook a
            # stale lockfile. Keep stdout quiet; Cargo prints failures to stderr.
            subprocess.run(
                [
                    "cargo",
                    f"+{manifest['toolchains']['rust']}",
                    "metadata",
                    "--locked",
                    "--format-version",
                    "1",
                ],
                cwd=source,
                stdout=subprocess.DEVNULL,
                check=True,
            )


def update_source_hash(path: Path):
    text = path.read_text(encoding="utf-8")
    manifest = tomllib.loads(text)
    artifact = manifest["source"]["artifact"]
    # Build templates contain placeholders filled by plugin packaging scripts.
    if "<" in artifact and ">" in artifact:
        return
    artifact_root = inside(path.parent, manifest["source"].get("manifest_root", "."))
    digest = "sha256:" + hashlib.sha256(inside(artifact_root, artifact).read_bytes()).hexdigest()
    previous = manifest["integrity"]["sha256"]
    if previous == digest:
        return
    # Change only the digest, preserving comments and the rest of the manifest.
    pattern = r"(?m)^(\s*sha256\s*=\s*['\"])" + re.escape(previous) + r"(['\"])"
    updated, count = re.subn(pattern, lambda m: m[1] + digest + m[2], text)
    if count != 1:
        raise ValueError(f"Expected exactly one integrity digest in {path}")
    path.write_text(updated, encoding="utf-8")
    print(f"Updated artifact hash: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def runtime_integrity(root: Path):
    for source, manifest in in_tree_sources(root):
        path = inside(source, manifest["artifacts"]["manifest"])
        if path.exists():
            update_source_hash(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=["rust-fmt", "lockfiles", "runtime-integrity"])
    args = parser.parse_args()
    {"rust-fmt": rust_fmt, "lockfiles": lockfiles, "runtime-integrity": runtime_integrity}[
        args.check
    ](ROOT)


if __name__ == "__main__":
    main()
