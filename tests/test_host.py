# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess

import pytest

from scripts.github import GitHub
from scripts.host import checkout, resolve_host


class API:
    def __init__(self):
        self.paths = []

    def api(self, path):
        self.paths.append(path)
        if path.endswith("/releases/latest"):
            return {"tag_name": "0.8.4", "draft": False, "prerelease": False, "assets": []}
        return {"sha": "a" * 40, "object": {"type": "commit", "sha": "a" * 40}}

    def pages(self, path):
        self.paths.append(path)
        return iter([])


def test_latest_host_resolved_to_commit():
    api = API()
    host = resolve_host({}, api)
    assert host == {"tag": "0.8.4", "sha": "a" * 40, "assets": []}
    assert api.paths[0].endswith("/releases/latest")


def test_tag_override_without_release():
    api = API()
    assert resolve_host({"tag": "custom-tag"}, api)["tag"] == "custom-tag"
    assert not any(p.endswith("/latest") for p in api.paths)


def test_sha_override_and_mismatch():
    assert resolve_host({"sha": "a" * 40}, API()) == {"sha": "a" * 40, "tag": None, "assets": []}
    with pytest.raises(ValueError):
        resolve_host({"sha": "b" * 40}, API())


def test_checkout_uses_sha_even_when_branch_advances(repo, tmp_path):
    root, git, commit = repo
    sha = git("rev-parse", "HEAD")
    commit("README.md", "new upstream state")
    destination = tmp_path / "pinned"
    checkout(str(root), sha, destination)
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual == sha
    assert not (destination / "README.md").exists()


def test_github_pagination(monkeypatch):
    pages = []

    def api(self, path):
        pages.append(path)
        return list(range(100)) if path.endswith("page=1") else [100]

    monkeypatch.setattr(GitHub, "api", api)
    assert len(list(GitHub().pages("repos/a/b/pulls?state=closed"))) == 101
    assert pages[-1].endswith("&per_page=100&page=2")


def test_annotated_tag_is_peeled():
    from scripts.host import resolve_tag

    class Annotated:
        def api(self, path):
            if "/git/ref/" in path:
                return {"object": {"type": "tag", "sha": "b" * 40}}
            return {"object": {"type": "commit", "sha": "a" * 40}}

    assert resolve_tag("0.8.4", Annotated()) == "a" * 40


def test_host_download_checksum_and_source_fallback(tmp_path, monkeypatch):
    from scripts.host import install_host
    from pathlib import Path
    from types import SimpleNamespace
    import hashlib

    name = "nemo-relay-cli-aarch64-apple-darwin-0.8.4"
    downloaded = tmp_path / "download"

    def download(args, **kwargs):
        if args[0] == "gh":
            (downloaded / name).write_bytes(b"host")
            (downloaded / (name + ".sha256")).write_text(
                hashlib.sha256(b"host").hexdigest() + "  " + name
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.host.subprocess.run", download)
    resolution = {
        "sha": "a" * 40,
        "tag": "0.8.4",
        "assets": [{"name": name}, {"name": name + ".sha256"}],
    }
    assert install_host(resolution, "macos-arm64", downloaded).read_bytes() == b"host"
    source_build = tmp_path / "source-build"
    calls = []

    def checkout(repository, sha, destination):
        assert sha == "a" * 40
        destination.mkdir(parents=True)

    def build(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "cargo":
            binary = Path(kwargs["cwd"]) / "target/release/nemo-relay"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"compiled")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.host.checkout", checkout)
    monkeypatch.setattr("scripts.host.subprocess.run", build)
    monkeypatch.setenv("RUSTUP_TOOLCHAIN", "plugin-toolchain")
    monkeypatch.setenv("CARGO_TARGET_DIR", "/plugin/target")
    binary = install_host({"sha": "a" * 40, "tag": None, "assets": []}, "macos-arm64", source_build)
    assert binary.read_bytes() == b"compiled"
    build_env = calls[0][1]["env"]
    assert "RUSTUP_TOOLCHAIN" not in build_env and "CARGO_TARGET_DIR" not in build_env
