# SPDX-License-Identifier: Apache-2.0
"""Portable archives, runtime-manifest verification, and asset integrity."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

from scripts.catalog import inside


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bundle(bundle: Path, manifest_path: str, kind: str) -> dict:
    manifest = tomllib.loads(inside(bundle, manifest_path).read_text(encoding="utf-8"))
    root = inside(bundle, manifest_path).parent
    artifact = inside(root, manifest["source"]["artifact"])
    if not artifact.is_file():
        raise ValueError("runtime artifact is missing")
    if manifest["integrity"]["sha256"] != "sha256:" + sha256(artifact):
        raise ValueError("runtime artifact digest mismatch")
    expected_kind = "rust_dynamic" if kind == "native" else "worker"
    if manifest["plugin"]["kind"] != expected_kind:
        raise ValueError("release type and runtime kind disagree")
    load = manifest["load"]
    if kind == "native":
        if inside(root, load["library"]) != artifact:
            raise ValueError("loaded library must match integrity artifact")
    elif load.get("runtime") == "python":
        package = inside(root, manifest["source"]["manifest_root"])
        if not (package / "pyproject.toml").is_file():
            raise ValueError("Python bundle lacks installable source")
        module = load["entrypoint"].split(":")[0].replace(".", "/")
        if artifact not in [package / (module + ".py"), package / module / "__init__.py"]:
            raise ValueError("Python entrypoint does not match integrity artifact")
    elif inside(root, load["entrypoint"]) != artifact:
        raise ValueError("worker executable must match integrity artifact")
    if "config_schema" in manifest:
        json.loads(inside(root, manifest["config_schema"]["path"]).read_text(encoding="utf-8"))
    for path in bundle.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError(f"unsupported bundle member: {path}")
    attributions = list(bundle.glob("ATTRIBUTIONS*.md"))
    if not attributions or any(
        not path.is_file() or not path.read_text(encoding="utf-8").strip() for path in attributions
    ):
        raise ValueError("bundle must include nonempty ATTRIBUTIONS files")
    return manifest


def archive_name(manifest: dict, platform: str) -> str:
    suffix = ".zip" if platform.startswith("windows") else ".tar.gz"
    return f"{manifest['name']}-{manifest['version']}-{platform}{suffix}"


def create_archive(bundle: Path, archive: Path, prefix: str):
    if archive.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("archive must be outside its input bundle")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    stream.write(
                        path, str(PurePosixPath(prefix) / path.relative_to(bundle).as_posix())
                    )
    else:
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(bundle, arcname=prefix)


def _member(destination: Path, name: str, seen: set[str]):
    normalized = name.rstrip("/")
    if not normalized or normalized in seen:
        raise ValueError(f"duplicate or empty archive member: {name}")
    if ".." in PurePosixPath(normalized).parts:
        raise ValueError(f"unsafe archive member: {name}")
    inside(destination, normalized)
    seen.add(normalized)


def extract_archive(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("archive extraction directory must be empty")
    seen: set[str] = set()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as stream:
            for info in stream.infolist():
                _member(destination, info.filename, seen)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError("archive links are not allowed")
            stream.extractall(destination)
    else:
        with tarfile.open(archive) as stream:
            for info in stream.getmembers():
                _member(destination, info.name, seen)
                if not (info.isfile() or info.isdir()):
                    raise ValueError("archive links/devices are not allowed")
            stream.extractall(destination, filter="data")
    children = list(destination.iterdir())
    if len(children) != 1 or not children[0].is_dir():
        raise ValueError("archive must contain one bundle root directory")
    return children[0]


def verify_assets(manifest: dict, directory: Path, commit: str) -> list[Path]:
    files = []
    expected = set()
    for platform in manifest["platforms"]:
        name = archive_name(manifest, platform)
        archive = directory / name
        metadata = directory / (name + ".json")
        checksum = directory / (name + ".sha256")
        expected.update([name, metadata.name, checksum.name])
        data = json.loads(metadata.read_text(encoding="utf-8"))
        identity = (
            data.get("name"),
            data.get("version"),
            data.get("platform"),
            data.get("repository_commit"),
        )
        if identity != (manifest["name"], manifest["version"], platform, commit):
            raise ValueError("release asset identity mismatch")
        if data.get("local_host_override") is not False:
            raise ValueError("release assets cannot use a local host override")
        if data.get("repository_dirty") is not False:
            raise ValueError("release assets must come from a clean checkout")
        if data.get("verified") is not True:
            raise ValueError("bundle has not passed runtime verification")
        if (
            manifest["source"]["location"] == "remote"
            and data.get("source_commit") != manifest["source"]["sha"]
        ):
            raise ValueError("remote source commit mismatch")
        if manifest["source"]["location"] in {"wheel", "crate"}:
            from scripts.catalog import ROOT
            from scripts.package_sources import provenance

            expected_source = provenance(ROOT / "plugins" / manifest["name"], manifest, platform)
            if data.get("package_source") != expected_source:
                raise ValueError("package source provenance mismatch")
        if not re.fullmatch("[0-9a-f]{40}", data.get("relay", {}).get("sha", "")):
            raise ValueError("missing tested Relay revision")
        digest = sha256(archive)
        if (
            digest != data.get("sha256")
            or checksum.read_text(encoding="utf-8") != f"{digest}  {name}\n"
        ):
            raise ValueError("release asset checksum mismatch")
        files.extend([archive, metadata, checksum])
    actual = {p.name for p in directory.iterdir() if p.is_file()}
    if actual != expected:
        raise ValueError(
            f"unexpected/missing release assets: {actual.symmetric_difference(expected)}"
        )
    return files
