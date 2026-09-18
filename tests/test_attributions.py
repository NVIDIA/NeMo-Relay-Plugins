# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locked inventories and verified license sources must survive regeneration."""

import io
import json
import re
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

from scripts.licensing import attributions_lockfile_md as attribution
from scripts.licensing import generate
from scripts.bundles import create_archive, extract_archive

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("remote", [False, True])
def test_remote_workspace_packages_reach_notices_and_license_inventory(
    tmp_path, monkeypatch, remote
):
    workspace = {
        "id": "path+file:///checkout/crate#0.2.0",
        "name": "upstream-crate",
        "version": "0.2.0",
    }
    registry = {"id": "registry+example#dependency@1.0.0", "name": "dependency", "version": "1.0.0"}
    data = {
        "licenses": [
            {
                "id": "MIT",
                "text": "Copyright Authors\nPermission is hereby granted",
                "used_by": [{"crate": workspace}, {"crate": registry}],
            }
        ]
    }
    monkeypatch.setattr(attribution, "_cargo_about_json", lambda: data)
    monkeypatch.setattr(attribution, "_cargo_workspace_members", lambda: {workspace["id"]})
    monkeypatch.setattr(attribution, "_rust_missing_cargo_about_packages", lambda keys: [])
    (tmp_path / "Cargo.toml").touch()
    (tmp_path / "Cargo.lock").write_text(
        '[[package]]\nname = "upstream-crate"\nversion = "0.2.0"\n'
        '[[package]]\nname = "dependency"\nversion = "1.0.0"\n'
    )
    documents, inventory = generate.collect_project(
        tmp_path, None, ["Rust"], include_workspace=remote
    )
    assert ("## upstream-crate - 0.2.0" in documents["Rust"]) is remote
    assert ("upstream-crate" in {row["package"] for row in inventory["rust"]}) is remote
    assert "## dependency - 1.0.0" in documents["Rust"]
    generate.write_project_attributions(
        tmp_path, tmp_path, tmp_path / "bundle", include_workspace=remote
    )
    assert (tmp_path / "bundle/ATTRIBUTIONS-Rust.md").read_text() == documents["Rust"]
    if remote:
        data["licenses"][0]["used_by"] = [{"crate": registry}]
        with pytest.raises(
            ValueError, match="Missing remote workspace attributions.*upstream-crate"
        ):
            generate.collect_project(tmp_path, None, ["Rust"], include_workspace=True)


@pytest.mark.parametrize(
    "reader", ["_cargo_about_json", "_cargo_metadata", "_cargo_workspace_members"]
)
def test_cargo_json_preserves_unicode_on_legacy_windows_encoding(tmp_path, monkeypatch, reader):
    payload = {"workspace_members": ["Bjørn/Łódź"], "license": "Copyright Bjørn – Łódź"}
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    run = attribution.subprocess.run

    def emit_cargo_json(args, **kwargs):
        # Simulate the legacy Windows default unless the collector chooses UTF-8.
        kwargs.setdefault("encoding", "cp1252")
        return run(
            [sys.executable, "-c", f"import sys; sys.stdout.buffer.write({encoded!r})"], **kwargs
        )

    monkeypatch.setattr(attribution, "ROOT", tmp_path)
    monkeypatch.setattr(attribution, "_cargo_fetch_locked", lambda: None)
    monkeypatch.setattr(attribution.subprocess, "run", emit_cargo_json)
    expected = (
        set(payload["workspace_members"]) if reader == "_cargo_workspace_members" else payload
    )
    assert getattr(attribution, reader)() == expected


@pytest.mark.parametrize(
    "language,manifest,lock",
    [
        ("Python", "pyproject.toml", "uv.lock"),
        ("Rust", "Cargo.toml", "Cargo.lock"),
    ],
)
@pytest.mark.parametrize("extension", [".tar.gz", ".zip"])
def test_bundle_notices_are_generated_from_current_locks(
    tmp_path, monkeypatch, language, manifest, lock, extension
):
    source_root = tmp_path / "workspace"
    package = source_root / "package"
    package.mkdir(parents=True)
    (package / manifest).touch()
    # Unrelated workspace languages must not leak into this plugin's notices.
    (source_root / "pyproject.toml").touch()
    (source_root / lock).write_text("dependency version 1")
    filename = f"ATTRIBUTIONS-{language}.md"
    (package / filename).write_text("Stale source copy: do not use")
    (tmp_path / filename).write_text("Unrelated root aggregate: do not use")
    bundle = tmp_path / "bundle"

    def collect(source, toolchain, languages, **kwargs):
        assert source == source_root
        assert toolchain == "1.96.1"
        assert languages == [language]
        return {
            language: "Copyright Bjørn – Łódź\nLicense for " + (source / lock).read_text() + "\n"
        }, {}

    monkeypatch.setattr(generate, "collect_project", collect)
    for version in ["1", "2"]:
        (source_root / lock).write_text(f"dependency version {version}")
        generate.write_project_attributions(source_root, package, bundle, toolchain="1.96.1")
        archive = tmp_path / f"bundle-{version}{extension}"
        create_archive(bundle, archive, "plugin")
        extracted = extract_archive(archive, tmp_path / f"extracted-{version}")
        assert (extracted / filename).read_bytes() == (
            f"Copyright Bjørn – Łódź\nLicense for dependency version {version}\n".encode("utf-8")
        )
        assert sorted(p.name for p in extracted.iterdir()) == [filename]
    assert (package / filename).read_text() == "Stale source copy: do not use"


def test_packaging_stops_when_license_generation_fails(tmp_path, monkeypatch):
    (tmp_path / "Cargo.toml").touch()

    def fail(*args, **kwargs):
        raise ValueError("Missing license text")

    monkeypatch.setattr(generate, "collect_project", fail)
    with pytest.raises(ValueError, match="Missing license text"):
        generate.write_project_attributions(tmp_path, tmp_path, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("lock", [ROOT / "uv.lock", *ROOT.glob("plugins/*/uv.lock")])
def test_python_attributions_cover_locked_registry_versions(lock):
    packages = tomllib.loads(lock.read_text())["package"]
    expected = {
        (p["name"], p["version"])
        for p in packages
        if any(key in p["source"] for key in ("registry", "git"))
    }
    path = ROOT / "ATTRIBUTIONS-Python.md"
    text = path.read_text()
    assert expected <= set(re.findall(r"^## (.+) \((.+)\)$", text, re.MULTILINE))
    assert "No license file found" not in text


@pytest.mark.parametrize("lock", list(ROOT.glob("plugins/*/Cargo.lock")))
def test_rust_attributions_cover_locked_registry_versions(lock):
    packages = tomllib.loads(lock.read_text())["package"]
    expected = {(p["name"], p["version"]) for p in packages if "source" in p}
    text = (ROOT / "ATTRIBUTIONS-Rust.md").read_text()
    assert expected <= set(re.findall(r"^## (.+) - (.+)$", text, re.MULTILINE))
    assert "No package license file" not in text


def test_download_rejects_artifact_hash_mismatch(monkeypatch):
    monkeypatch.setattr(
        attribution.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"tampered")
    )
    with pytest.raises(ValueError, match="Hash mismatch"):
        attribution._download_locked_artifact("https://example.com/pkg.whl", "sha256:" + "0" * 64)


def test_wheel_license_is_rendered_in_relay_format():
    data = io.BytesIO()
    license_text = "Copyright Example Authors\nPermission granted.\n"
    with zipfile.ZipFile(data, "w") as wheel:
        wheel.writestr(
            "example-1.0.dist-info/METADATA",
            "Name: example\nVersion: 1.0\nLicense-Expression: MIT\nLicense-File: LICENSE\n",
        )
        wheel.writestr("example-1.0.dist-info/licenses/LICENSE", license_text)
    license_name, texts = attribution._wheel_metadata_from_bytes(
        data.getvalue(), package_name="example"
    )
    parts = []
    attribution._render_python_package(
        parts, name="example", version="1.0", license_name=license_name, license_texts=texts
    )
    text = "".join(parts)
    assert "## example (1.0)\n\n### Licenses\nLicense: `MIT`" in text
    assert "`licenses/LICENSE`" in text
    assert license_text in text


def test_rust_upstream_fallback_uses_publication_commit(tmp_path, monkeypatch):
    import json

    sha = "a" * 40
    (tmp_path / ".cargo_vcs_info.json").write_text(
        json.dumps({"git": {"sha1": sha}, "path_in_vcs": "crate"})
    )
    calls = []

    def fetch(repository, revision, subdir):
        calls.append((repository, revision, subdir))
        return [("upstream/LICENSE", "Copyright Authors\nLicense text")]

    monkeypatch.setattr(attribution, "_upstream_licenses", fetch)
    crate = {
        "name": "example",
        "version": "1.0",
        "repository": "https://github.com/owner/repo",
        "manifest_path": str(tmp_path / "Cargo.toml"),
        "license": "MIT",
    }
    rendered = attribution._render_rust_metadata_fallback_attribution(crate)[2]
    assert calls == [(crate["repository"], sha, "crate")]
    assert "Copyright Authors" in rendered


@pytest.mark.parametrize("sha", ["main", "v1.0", "a" * 39])
def test_upstream_license_fetch_requires_exact_commit(sha):
    with pytest.raises(ValueError, match="immutable"):
        attribution._upstream_licenses("https://github.com/owner/repo", sha)


def test_python_fallback_resolves_version_tag_to_commit(monkeypatch):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as wheel:
        wheel.writestr(
            "example-1.0.dist-info/METADATA",
            "Name: example\nVersion: 1.0\nProject-URL: Repository, https://github.com/owner/repo\n",
        )
    monkeypatch.setattr(attribution, "_download_locked_artifact", lambda *args: data.getvalue())
    tag_sha, commit_sha = "a" * 40, "b" * 40
    calls = []

    def refs(argv, **kwargs):
        calls.append(argv)
        return f"{tag_sha}\trefs/tags/1.0\n{commit_sha}\trefs/tags/1.0^{{}}\n"

    monkeypatch.setattr(attribution.subprocess, "check_output", refs)
    monkeypatch.setattr(
        attribution, "_upstream_licenses", lambda repo, sha: [(sha, "License text")]
    )
    package = {
        "version": "1.0",
        "wheels": [{"url": "https://example.com/pkg.whl", "hash": "verified"}],
    }
    assert attribution._python_upstream_licenses(package) == [(commit_sha, "License text")]
    assert calls[0][-2:] == ["refs/tags/1.0", "refs/tags/1.0^{}"]
    monkeypatch.setattr(attribution.subprocess, "check_output", lambda *args, **kwargs: "")
    assert attribution._python_upstream_licenses(package) == []


def test_missing_rust_license_fails_without_placeholder(tmp_path, monkeypatch):
    crate = {"name": "example", "version": "1.0", "manifest_path": str(tmp_path / "Cargo.toml")}
    with pytest.raises(ValueError, match="Missing upstream license text"):
        attribution._render_rust_metadata_fallback_attribution(crate)


def test_workspace_crates_are_excluded_from_dependency_attributions():
    assert (
        attribution._render_rust_crate_attribution(
            {"id": "local", "name": "local", "version": "1.0"},
            license_id="MIT",
            license_text="license",
            workspace_members={"local"},
        )
        is None
    )


def test_python_git_license_uses_locked_commit_without_building(monkeypatch):
    sha = "a" * 40
    package = {
        "name": "external-sdk",
        "version": "2.0.0",
        "source": {
            "git": f"https://github.com/owner/repo.git?subdirectory=python%2Fsdk&rev=main#{sha}"
        },
    }
    urls = []

    def download(url, **kwargs):
        urls.append(url)
        return io.BytesIO(b'[project]\nname = "external-sdk"\nversion = "2.0.0"\nlicense = "MIT"\n')

    monkeypatch.setattr(attribution.urllib.request, "urlopen", download)
    sources = []

    def licenses(repository, commit, subdir):
        sources.append((repository, commit, subdir))
        return [("upstream/LICENSE", "Copyright Authors\nLicense text")]

    monkeypatch.setattr(attribution, "_upstream_licenses", licenses)
    result = attribution._lockfile_only_python_package(package)
    assert result["license_name"] == "MIT"
    assert "Copyright Authors" in result["license_texts"][0][1]
    assert urls == [f"https://raw.githubusercontent.com/owner/repo/{sha}/python/sdk/pyproject.toml"]
    assert sources == [("https://github.com/owner/repo", sha, "python/sdk")]
    package["version"] = "3.0.0"
    with pytest.raises(ValueError, match="identity differs"):
        attribution._git_python_license(package)


@pytest.mark.parametrize("revision", ["main", "", "a" * 39])
def test_python_git_license_rejects_unlocked_source(revision):
    with pytest.raises(ValueError, match="exact supported source"):
        attribution._git_python_license(
            {"name": "example", "source": {"git": f"https://github.com/owner/repo#{revision}"}}
        )
