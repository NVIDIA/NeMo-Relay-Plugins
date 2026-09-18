# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a Relay test host once, then install the exact resolved revision."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from urllib.parse import quote

from scripts.catalog import PLATFORMS
from scripts.github import GitHub

REPOSITORY = "NVIDIA/NeMo-Relay"


def resolve_tag(tag: str, github: GitHub, repository: str = REPOSITORY) -> str:
    reference = github.api(f"repos/{repository}/git/ref/tags/{quote(tag, safe='')}")
    obj = reference["object"]
    seen = set()
    while obj["type"] == "tag":
        if obj["sha"] in seen:
            raise ValueError("cyclic annotated tag")
        seen.add(obj["sha"])
        obj = github.api(f"repos/{repository}/git/tags/{obj['sha']}")["object"]
    if obj["type"] != "commit":
        raise ValueError("Relay/release tag must identify a commit")
    return obj["sha"]


def resolve_host(selector: dict, github: GitHub) -> dict:
    if selector.get("sha"):
        sha = selector["sha"]
        commit = github.api(f"repos/{REPOSITORY}/commits/{sha}")
        if commit["sha"] != sha:
            raise ValueError("Relay SHA did not resolve exactly")
        return {"sha": sha, "tag": None, "assets": []}
    endpoint = "latest" if not selector.get("tag") else "tags/" + quote(selector["tag"], safe="")
    if selector.get("tag"):
        # A tag override need not have an associated GitHub Release.
        commit_sha = resolve_tag(selector["tag"], github)
        releases = list(github.pages(f"repos/{REPOSITORY}/releases"))
        release = next((r for r in releases if r["tag_name"] == selector["tag"]), None)
        return {
            "sha": commit_sha,
            "tag": selector["tag"],
            "assets": (release or {}).get("assets", []),
        }
    release = github.api(f"repos/{REPOSITORY}/releases/{endpoint}")
    if release.get("draft") or release.get("prerelease"):
        raise ValueError("latest Relay release must be stable and published")
    commit_sha = resolve_tag(release["tag_name"], github)
    return {"sha": commit_sha, "tag": release["tag_name"], "assets": release["assets"]}


def checkout(repository: str, sha: str, destination: Path):
    destination.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", str(destination)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "--depth=1", repository, sha], check=True
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != sha:
        raise ValueError(f"checkout mismatch: expected {sha}, got {actual}")


def install_host(resolution: dict, platform: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if platform.startswith("windows") else ""
    binary = destination / ("nemo-relay" + suffix)
    version = (resolution["tag"] or "").removeprefix("v")
    asset = f"nemo-relay-cli-{PLATFORMS[platform]['target']}-{version}{suffix}"
    assets = {a["name"] for a in resolution["assets"]}
    if asset in assets and asset + ".sha256" in assets:
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                resolution["tag"],
                "--repo",
                REPOSITORY,
                "--pattern",
                asset,
                "--pattern",
                asset + ".sha256",
                "--dir",
                str(destination),
            ],
            check=True,
        )
        artifact = destination / asset
        digest = (destination / (asset + ".sha256")).read_text(encoding="utf-8").split()[0]
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
            raise ValueError("Relay binary checksum mismatch")
        artifact.rename(binary)
    else:
        source = destination / "source"
        checkout(f"https://github.com/{REPOSITORY}.git", resolution["sha"], source)
        # The pinned source's rust-toolchain.toml selects the host build toolchain.
        subprocess.run(
            ["cargo", "build", "--release", "--locked", "-p", "nemo-relay-cli"],
            cwd=source,
            check=True,
            env={
                k: v
                for k, v in os.environ.items()
                if k not in {"CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"}
            },
        )
        import shutil

        shutil.copy2(source / "target/release" / ("nemo-relay" + suffix), binary)
    binary.chmod(0o755)
    subprocess.run([str(binary), "--version"], check=True)
    return binary.resolve()
