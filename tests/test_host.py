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


@pytest.mark.parametrize(
    "message,failures,expected_attempts",
    [
        ("Failed to connect to github.com port 443", 1, 2),
        ("The requested URL returned error: 503", 3, 3),
        ("fatal: remote error: upload-pack: not our ref", 1, 1),
    ],
)
def test_checkout_retries_only_transient_fetch_errors(
    tmp_path, monkeypatch, message, failures, expected_attempts
):
    sha = "a" * 40
    fetches = []
    sleeps = []

    def run(argv, **kwargs):
        if "fetch" in argv:
            fetches.append(argv)
            if len(fetches) <= failures:
                raise subprocess.CalledProcessError(128, argv, stderr=message)

    monkeypatch.setattr("scripts.host.subprocess.run", run)
    monkeypatch.setattr("scripts.host.subprocess.check_output", lambda *args, **kwargs: sha)
    monkeypatch.setattr("scripts.host.time.sleep", sleeps.append)
    if failures >= expected_attempts:
        with pytest.raises(subprocess.CalledProcessError):
            checkout("https://github.com/example/repo.git", sha, tmp_path / "checkout")
    else:
        checkout("https://github.com/example/repo.git", sha, tmp_path / "checkout")
    assert len(fetches) == expected_attempts
    assert all(argv[-1] == sha for argv in fetches)
    assert len(sleeps) == expected_attempts - 1


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


@pytest.mark.parametrize("source_tag", [None, "custom-tag"])
def test_host_download_checksum_and_source_fallback(tmp_path, monkeypatch, source_tag):
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
    monkeypatch.setattr("scripts.host.shutil.which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(
        "scripts.host.subprocess.check_output",
        lambda *args, **kwargs: pytest.fail(
            "downloaded or custom-tag host needs no version rewrite"
        ),
    )
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
    binary = install_host(
        {"sha": "a" * 40, "tag": source_tag, "assets": []}, "macos-arm64", source_build
    )
    assert binary.read_bytes() == b"compiled"
    assert [arguments[0] for arguments, _ in calls] == ["cargo", str(binary)]
    build_env = calls[0][1]["env"]
    assert "RUSTUP_TOOLCHAIN" not in build_env and "CARGO_TARGET_DIR" not in build_env


@pytest.mark.parametrize("platform", ["linux-musl-x86_64", "linux-musl-arm64"])
def test_native_musl_host_is_dynamic_even_when_static_release_exists(
    tmp_path, monkeypatch, platform
):
    from pathlib import Path
    from scripts.catalog import PLATFORMS
    from scripts.host import install_host

    target = PLATFORMS[platform]["target"]
    asset = f"nemo-relay-cli-{target}-0.10.2"
    calls = []
    version_commands = []

    def checkout(repository, sha, destination):
        destination.mkdir()

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == "cargo":
            assert argv[-2:] == ["--target", target]
            assert "--release" in argv and "--locked" in argv
            env = kwargs["env"]
            assert "target-feature=-crt-static" in env["RUSTFLAGS"]
            assert "CARGO_ENCODED_RUSTFLAGS" not in env
            assert "RUSTUP_TOOLCHAIN" not in env
            binary = Path(kwargs["cwd"]) / "target" / target / "release/nemo-relay"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"dynamic host")

    def version_output(arguments, **kwargs):
        version_commands.append(arguments)
        return "nemo-relay 0.10.2\n"

    monkeypatch.setattr("scripts.host.checkout", checkout)
    monkeypatch.setattr("scripts.host.subprocess.run", run)
    monkeypatch.setattr("scripts.host.subprocess.check_output", version_output)
    monkeypatch.setenv("CARGO_ENCODED_RUSTFLAGS", "-Ctarget-feature=+crt-static")
    monkeypatch.setenv("RUSTFLAGS", "--cfg=plugin_configuration")
    monkeypatch.setenv("RUSTUP_TOOLCHAIN", "plugin-toolchain")
    monkeypatch.setenv("CARGO_TARGET_DIR", "/plugin/target")
    monkeypatch.setenv("CARGO_BUILD_TARGET", "plugin-target")
    resolution = {
        "sha": "a" * 40,
        "tag": "0.10.2",
        "assets": [{"name": asset}, {"name": asset + ".sha256"}],
    }
    assert (
        install_host(resolution, platform, tmp_path, dynamic_musl=True).read_bytes()
        == b"dynamic host"
    )
    assert calls[0][0] == ["just", "set-cargo-version", "0.10.2"]
    assert calls[1][0][0] == "cargo"
    assert len(calls) == 2
    assert version_commands == [[str(tmp_path / "nemo-relay"), "--version"]]
    build_environment = calls[1][1]["env"]
    assert "--cfg=plugin_configuration" in build_environment["RUSTFLAGS"]
    assert "CARGO_TARGET_DIR" not in build_environment
    assert "CARGO_BUILD_TARGET" not in build_environment


@pytest.mark.parametrize("tag", ["0.10.2", "v0.10.2", "0.10.2-rc.1"])
def test_release_source_build_prepares_and_verifies_selected_version(tmp_path, monkeypatch, tag):
    from pathlib import Path
    from scripts.host import install_host

    version = tag.removeprefix("v")
    commands = []

    def checkout(repository, sha, destination):
        destination.mkdir()

    def run(arguments, **kwargs):
        commands.append(arguments)
        if arguments[0] == "cargo":
            assert commands[0] == ["just", "set-cargo-version", version]
            binary = Path(kwargs["cwd"]) / "target/release/nemo-relay"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"release host")

    def version_output(arguments, **kwargs):
        commands.append(arguments)
        return f"nemo-relay {version}\n"

    monkeypatch.setattr("scripts.host.checkout", checkout)
    monkeypatch.setattr("scripts.host.subprocess.run", run)
    monkeypatch.setattr("scripts.host.subprocess.check_output", version_output)

    binary = install_host({"sha": "a" * 40, "tag": tag, "assets": []}, "macos-arm64", tmp_path)

    assert binary.read_bytes() == b"release host"
    assert commands == [
        ["just", "set-cargo-version", version],
        ["cargo", "build", "--release", "--locked", "-p", "nemo-relay-cli"],
        [str(binary), "--version"],
    ]


@pytest.mark.parametrize(
    "version_output", ["nemo-relay 0.10.0", "nemo-relay 0.10.20", "nemo-relay 0.10.2-extra"]
)
def test_release_source_build_rejects_wrong_executable_version(
    tmp_path, monkeypatch, version_output
):
    from pathlib import Path
    from scripts.host import install_host

    def checkout(repository, sha, destination):
        destination.mkdir()

    def run(arguments, **kwargs):
        if arguments[0] == "cargo":
            binary = Path(kwargs["cwd"]) / "target/release/nemo-relay"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"wrong version host")

    monkeypatch.setattr("scripts.host.checkout", checkout)
    monkeypatch.setattr("scripts.host.subprocess.run", run)
    monkeypatch.setattr(
        "scripts.host.subprocess.check_output", lambda *args, **kwargs: version_output
    )

    with pytest.raises(ValueError, match="Relay source build version mismatch"):
        install_host({"sha": "a" * 40, "tag": "0.10.2", "assets": []}, "macos-arm64", tmp_path)


def test_release_version_preparation_failure_stops_before_build(tmp_path, monkeypatch):
    from scripts.host import install_host

    commands = []

    def checkout(repository, sha, destination):
        destination.mkdir()

    def run(arguments, **kwargs):
        commands.append(arguments)
        raise subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr("scripts.host.checkout", checkout)
    monkeypatch.setattr("scripts.host.subprocess.run", run)

    with pytest.raises(subprocess.CalledProcessError):
        install_host({"sha": "a" * 40, "tag": "0.10.2", "assets": []}, "macos-arm64", tmp_path)

    assert commands == [["just", "set-cargo-version", "0.10.2"]]


def test_public_host_download_without_github_cli_checks_digest(tmp_path, monkeypatch):
    import hashlib
    import io
    from scripts.host import install_host

    name = "nemo-relay-cli-x86_64-unknown-linux-musl-0.8.4"
    payloads = {
        name: b"static musl host",
        name + ".sha256": hashlib.sha256(b"static musl host").hexdigest().encode(),
    }
    monkeypatch.setattr("scripts.host.shutil.which", lambda name: None)
    monkeypatch.setattr("scripts.host.subprocess.run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "scripts.host.urlopen", lambda url, **kwargs: io.BytesIO(payloads[url.rsplit("/", 1)[-1]])
    )
    resolution = {
        "sha": "a" * 40,
        "tag": "0.8.4",
        "assets": [
            {"name": filename, "browser_download_url": f"https://github.com/example/{filename}"}
            for filename in payloads
        ],
    }
    assert (
        install_host(resolution, "linux-musl-x86_64", tmp_path / "good").read_bytes()
        == payloads[name]
    )
    payloads[name] = b"wrong binary"
    with pytest.raises(ValueError, match="checksum mismatch"):
        install_host(resolution, "linux-musl-x86_64", tmp_path / "bad")
