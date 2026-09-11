# SPDX-License-Identifier: Apache-2.0
"""Locked inventories and verified license sources must survive regeneration."""

import io
import re
import tomllib
import zipfile
from pathlib import Path

import pytest

from scripts.licensing import attributions_lockfile_md as attribution

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("lock", [ROOT / "uv.lock", *ROOT.glob("plugins/*/uv.lock")])
def test_python_attributions_cover_locked_registry_versions(lock):
    packages = tomllib.loads(lock.read_text())["package"]
    expected = {(p["name"], p["version"]) for p in packages if "registry" in p["source"]}
    path = (
        ROOT / "scripts/licensing/ATTRIBUTIONS-Python.md"
        if lock.parent == ROOT
        else lock.with_name("ATTRIBUTIONS-Python.md")
    )
    text = path.read_text()
    assert set(re.findall(r"^## (.+) \((.+)\)$", text, re.MULTILINE)) == expected
    assert "No license file found" not in text


@pytest.mark.parametrize("lock", list(ROOT.glob("plugins/*/Cargo.lock")))
def test_rust_attributions_cover_locked_registry_versions(lock):
    packages = tomllib.loads(lock.read_text())["package"]
    expected = {(p["name"], p["version"]) for p in packages if "source" in p}
    text = lock.with_name("ATTRIBUTIONS-Rust.md").read_text()
    assert set(re.findall(r"^## (.+) - (.+)$", text, re.MULTILINE)) == expected
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
