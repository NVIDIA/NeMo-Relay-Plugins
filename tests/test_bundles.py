# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile
import zipfile

import pytest
import tomli_w

from scripts.bundles import (
    archive_name,
    create_archive,
    extract_archive,
    sha256,
    verify_assets,
    verify_bundle,
)


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "ATTRIBUTIONS.md").write_text("Third-party fixture license text\n")
    (root / "worker").write_bytes(b"worker fixture")
    (root / "worker").chmod(0o755)
    manifest = {
        "plugin": {"kind": "worker", "id": "test.worker"},
        "source": {"artifact": "worker"},
        "integrity": {"sha256": "sha256:" + sha256(root / "worker")},
        "load": {"runtime": "rust", "entrypoint": "worker"},
    }
    (root / "relay-plugin.toml").write_text(tomli_w.dumps(manifest))
    return root


@pytest.mark.parametrize("extension", [".zip", ".tar.gz"])
def test_archive_round_trip_and_tampering(bundle, tmp_path, extension):
    archive = tmp_path / ("plugin" + extension)
    create_archive(bundle, archive, "plugin")
    extracted = extract_archive(archive, tmp_path / "extracted")
    assert (extracted / "ATTRIBUTIONS.md").read_bytes() == (bundle / "ATTRIBUTIONS.md").read_bytes()
    verify_bundle(extracted, "relay-plugin.toml", "worker")
    if extension == ".tar.gz":
        assert (extracted / "worker").stat().st_mode & 0o111
    (extracted / "worker").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        verify_bundle(extracted, "relay-plugin.toml", "worker")


@pytest.mark.parametrize("contents", [None, "", "   \n"])
def test_bundle_requires_attributions(bundle, contents):
    path = bundle / "ATTRIBUTIONS.md"
    if contents is None:
        path.unlink()
    else:
        path.write_text(contents)
    with pytest.raises(ValueError, match="ATTRIBUTIONS"):
        verify_bundle(bundle, "relay-plugin.toml", "worker")


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "C:/escape", "a/../../escape", "a\\escape"]
)
def test_zip_path_escape_rejected(tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(name, "bad")
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "out")


def test_tar_symlinks_rejected(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("bundle/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        stream.addfile(member)
    with pytest.raises(ValueError, match="links"):
        extract_archive(archive, tmp_path / "out")


def test_duplicates_rejected(tmp_path):
    archive = tmp_path / "duplicate.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for _ in range(2):
            member = tarfile.TarInfo("bundle/file")
            member.size = 1
            stream.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="duplicate"):
        extract_archive(archive, tmp_path / "out")


def assets_for(manifest, root, commit):
    root.mkdir(exist_ok=True)
    for platform in manifest["platforms"]:
        name = archive_name(manifest, platform)
        archive = root / name
        archive.write_bytes(b"archive fixture")
        digest = sha256(archive)
        metadata = {
            "name": manifest["name"],
            "version": manifest["version"],
            "platform": platform,
            "repository_commit": commit,
            "repository_dirty": False,
            "local_host_override": False,
            "source_commit": manifest["source"].get("sha", commit),
            "relay": {"sha": "b" * 40},
            "verified": True,
            "sha256": digest,
        }
        (root / (name + ".json")).write_text(json.dumps(metadata))
        (root / (name + ".sha256")).write_text(f"{digest}  {name}\n")


def test_complete_asset_set_required(catalog, tmp_path):
    from scripts.catalog import discover

    manifest = discover(catalog)["example-rust-native-plugin"]
    root = tmp_path / "assets"
    assets_for(manifest, root, "a" * 40)
    assert len(verify_assets(manifest, root, "a" * 40)) == 15
    with pytest.raises(ValueError, match="identity"):
        verify_assets(manifest, root, "c" * 40)
    extra = root / "other-plugin.zip"
    extra.write_text("unrelated")
    with pytest.raises(ValueError, match="unexpected"):
        verify_assets(manifest, root, "a" * 40)
    extra.unlink()
    metadata = next(root.glob("*.json"))
    data = json.loads(metadata.read_text())
    data["verified"] = False
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="runtime"):
        verify_assets(manifest, root, "a" * 40)
    metadata.unlink()
    with pytest.raises(FileNotFoundError):
        verify_assets(manifest, root, "a" * 40)
