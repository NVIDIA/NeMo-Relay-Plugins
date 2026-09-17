# SPDX-License-Identifier: Apache-2.0
"""Pinned package downloads shared by builds, licensing, and release verification."""

import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tomllib
import urllib.request
import zipfile
from email.parser import Parser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags, sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from scripts.bundles import _member, extract_archive, sha256
from scripts.catalog import inside

KINDS = {"wheel", "crate"}


def marker_environment_for_python(python: str) -> dict[str, str]:
    """Return native-platform marker values for a declared CPython major/minor."""
    version = tuple(map(int, python.split(".")))
    if len(version) != 2:
        raise ValueError("Python toolchains must declare a major.minor version")
    environment = default_environment()
    environment.update(
        {
            "implementation_name": "cpython",
            "implementation_version": f"{python}.0",
            "platform_python_implementation": "CPython",
            "python_full_version": f"{python}.0",
            "python_version": python,
        }
    )
    return environment


def compatible_tags_for_python(python: str):
    """Return native-platform CPython tags for the declared plugin interpreter."""
    version = tuple(map(int, python.split(".")))
    if len(version) != 2:
        raise ValueError("Python toolchains must declare a major.minor version")
    platforms = list(dict.fromkeys(tag.platform for tag in sys_tags() if tag.platform != "any"))
    interpreter = f"cp{version[0]}{version[1]}"
    return [
        *cpython_tags(python_version=version, platforms=platforms),
        *compatible_tags(python_version=version, interpreter=interpreter, platforms=platforms),
    ]


def locked_runtime_requirements(project: Path, python: str) -> list[Requirement]:
    """Export the base runtime closure from a uv lock without optional profiles."""
    exported = subprocess.check_output(
        [
            "uv",
            "export",
            "--locked",
            "--python",
            python,
            "--project",
            str(project),
            "--no-default-groups",
            "--no-emit-project",
            "--no-header",
            "--no-annotate",
            "--no-hashes",
        ],
        text=True,
    )
    return parse_locked_runtime_requirements(
        exported, environment=marker_environment_for_python(python)
    )


def parse_locked_runtime_requirements(
    exported: str, *, environment: dict[str, str] | None = None
) -> list[Requirement]:
    """Parse an exact, marker-filtered base export suitable for a wheelhouse."""
    environment = environment or default_environment()
    requirements = []
    for line in exported.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") or line.endswith("\\"):
            raise ValueError("locked runtime export contains an unsupported option or continuation")
        requirement = Requirement(line)
        if requirement.url or requirement.extras:
            raise ValueError("locked runtime requirements must be registry packages without extras")
        specifiers = list(requirement.specifier)
        if (
            len(specifiers) != 1
            or specifiers[0].operator != "=="
            or specifiers[0].version.endswith(".*")
        ):
            raise ValueError("locked runtime requirements must contain one exact version")
        if requirement.marker is not None and not requirement.marker.evaluate(environment):
            continue
        requirements.append(requirement)
    names = [canonicalize_name(requirement.name) for requirement in requirements]
    if len(names) != len(set(names)):
        raise ValueError("locked runtime export selected more than one version of a package")
    return sorted(requirements, key=lambda requirement: canonicalize_name(requirement.name))


def select_locked_wheel_artifacts(
    lock: dict,
    requirements: list[Requirement],
    platform: str,
    *,
    compatible_tags=None,
) -> list[dict]:
    """Choose one exact compatible wheel for each selected locked requirement."""
    ranked_tags = {tag: index for index, tag in enumerate(compatible_tags or sys_tags())}
    artifacts = []
    for requirement in requirements:
        name = canonicalize_name(requirement.name)
        version = Version(next(iter(requirement.specifier)).version)
        packages = [
            package
            for package in lock.get("package", [])
            if canonicalize_name(package.get("name", "")) == name
            and Version(package.get("version", "0")) == version
            and package.get("source", {}).get("registry")
        ]
        if len(packages) != 1:
            raise ValueError(f"uv.lock does not contain one registry package for {name}=={version}")
        choices = []
        for wheel in packages[0].get("wheels", []):
            url = urlsplit(wheel.get("url", ""))
            filename = unquote(Path(url.path).name)
            if url.scheme != "https" or not url.netloc or url.username or url.fragment:
                raise ValueError("locked wheel URLs require HTTPS without credentials or fragments")
            try:
                distribution, wheel_version, _, tags = parse_wheel_filename(filename)
            except ValueError:
                continue
            if canonicalize_name(distribution) != name or wheel_version != version:
                raise ValueError("locked wheel filename does not match its package identity")
            score = min((ranked_tags[tag] for tag in tags if tag in ranked_tags), default=None)
            if score is None:
                continue
            digest = wheel.get("hash", "")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise ValueError("locked wheels require a full SHA-256")
            choices.append(
                (
                    score,
                    filename,
                    {
                        "name": requirement.name,
                        "version": str(version),
                        "filename": filename,
                        "url": wheel["url"],
                        "sha256": digest.removeprefix("sha256:"),
                        "platforms": [platform],
                    },
                )
            )
        if not choices:
            raise ValueError(f"uv.lock has no compatible wheel for {name}=={version}")
        choices.sort(key=lambda choice: (choice[0], choice[1]))
        if len(choices) > 1 and choices[0][0] == choices[1][0]:
            raise ValueError(f"uv.lock has ambiguous best wheels for {name}=={version}")
        artifacts.append(choices[0][2])
    return artifacts


def prepare_locked_project_wheelhouse(
    project: Path,
    project_wheel: Path,
    python: str,
    platform: str,
    work: Path,
    cache: Path,
) -> dict:
    """Materialize and preflight a platform wheelhouse from an in-tree uv lock."""
    lock_path = project / "uv.lock"
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    requirements = locked_runtime_requirements(project, python)
    artifacts = select_locked_wheel_artifacts(
        lock,
        requirements,
        platform,
        compatible_tags=compatible_tags_for_python(python),
    )
    wheels = work / "wheels"
    wheels.mkdir()
    for item in artifacts:
        shutil.copy2(download(item, cache), wheels / item["filename"])

    distribution, version, _, _ = parse_wheel_filename(project_wheel.name)
    project_item = {
        "name": str(distribution),
        "version": str(version),
        "filename": project_wheel.name,
        "source": "in-tree",
        "sha256": sha256(project_wheel),
        "platforms": [platform],
    }
    wheel_metadata(project_wheel, project_item)
    if canonicalize_name(project_item["name"]) in {
        canonicalize_name(item["name"]) for item in artifacts
    }:
        raise ValueError("runtime export must omit the in-tree project wheel")
    shutil.copy2(project_wheel, wheels / project_wheel.name)
    identity = {
        "platform": platform,
        "python": python,
        "location": "in-tree",
        "package": project_item["name"],
        "version": project_item["version"],
        "lock_sha256": sha256(lock_path),
        "artifacts": [project_item, *artifacts],
    }
    (work / "package-source.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )
    install_wheels(work, python)
    return identity


def install_wheels(work: Path, python: str) -> Path:
    """Install every locked wheel and check dependency closure on the native runner."""
    environment = work / "package-env"
    subprocess.run(["uv", "venv", "--python", python, str(environment)], check=True)
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    wheels = sorted((work / "wheels").glob("*.whl"))
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(executable),
            "--no-index",
            *map(str, wheels),
        ],
        check=True,
    )
    subprocess.run(["uv", "pip", "check", "--python", str(executable)], check=True)
    return executable


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock(plugin: Path, manifest: dict) -> dict:
    source = manifest["source"]
    path = inside(plugin, source.get("lock", "source.lock"))
    lock = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(lock) != {"schema_version", "artifacts"} or lock["schema_version"] != 1:
        raise ValueError("source.lock requires schema_version = 1 and artifacts")
    artifacts = lock["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("source.lock must contain artifacts")
    filenames = set()
    for item in artifacts:
        if set(item) != {"name", "version", "filename", "url", "sha256", "platforms"}:
            raise ValueError("invalid source.lock artifact fields")
        if not all(
            isinstance(item[k], str) and item[k]
            for k in ["name", "version", "filename", "url", "sha256"]
        ):
            raise ValueError("artifact identity fields must be nonempty strings")
        filename = item["filename"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", item["name"]) or not re.fullmatch(
            r"[0-9][A-Za-z0-9_.+!-]*", item["version"]
        ):
            raise ValueError("invalid package name or version in source.lock")
        if normalized(item["name"]) == "relay-wheel-environment":
            raise ValueError("package name is reserved for the Relay wheel adapter")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+!-]*", filename) or filename in filenames:
            raise ValueError("artifact filenames must be unique portable basenames")
        filenames.add(filename)
        if not filename.endswith(".whl" if source["location"] == "wheel" else ".crate"):
            raise ValueError("source artifact has the wrong file format")
        url = urlsplit(item["url"])
        if url.scheme != "https" or not url.netloc or url.username or url.fragment:
            raise ValueError(
                "package downloads require HTTPS URLs without credentials or fragments"
            )
        if not re.fullmatch("[0-9a-f]{64}", item["sha256"]):
            raise ValueError("package downloads require a full SHA-256")
        platforms = item["platforms"]
        if (
            not isinstance(platforms, list)
            or not platforms
            or len(set(platforms)) != len(platforms)
            or not set(platforms) <= set(manifest["platforms"])
        ):
            raise ValueError("artifact platforms must be declared plugin platforms")
    for platform in manifest["platforms"]:
        selected = [a for a in artifacts if platform in a["platforms"]]
        names = [normalized(a["name"]) for a in selected]
        roots = [
            a
            for a in selected
            if normalized(a["name"]) == normalized(source["package"])
            and a["version"] == source["version"]
        ]
        if len(roots) != 1 or len(names) != len(set(names)):
            raise ValueError(
                f"source.lock needs one exact plugin and one version per dependency for {platform}"
            )
        if source["location"] == "crate" and len(selected) != 1:
            raise ValueError("crate dependencies belong in the packaged Cargo.lock")
    return lock


def provenance(plugin: Path, manifest: dict, platform: str) -> dict:
    lock = read_lock(plugin, manifest)
    return {
        "platform": platform,
        "python": manifest["toolchains"]["python"],
        "location": manifest["source"]["location"],
        "package": manifest["source"]["package"],
        "version": manifest["source"]["version"],
        "artifacts": [a for a in lock["artifacts"] if platform in a["platforms"]],
    }


def download(item: dict, cache: Path) -> Path:
    path = cache / item["sha256"] / item["filename"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        try:
            with (
                urllib.request.urlopen(item["url"], timeout=60) as response,
                temporary.open("wb") as output,
            ):
                if urlsplit(response.url).scheme != "https":
                    raise ValueError("package download redirected away from HTTPS")
                shutil.copyfileobj(response, output)
            if sha256(temporary) != item["sha256"]:
                raise ValueError("package download digest mismatch")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(path) != item["sha256"]:
        raise ValueError("cached package digest mismatch")
    return path


def wheel_metadata(archive: Path, item: dict):
    with zipfile.ZipFile(archive) as stream:
        seen = set()
        folded = set()
        for member in stream.infolist():
            _member(Path("/wheel-validation"), member.filename, seen)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("wheel links are not allowed")
            if member.filename.rstrip("/").casefold() in folded:
                raise ValueError("wheel contains case-colliding paths")
            folded.add(member.filename.rstrip("/").casefold())
        metadata = [n for n in seen if n.endswith(".dist-info/METADATA") and n.count("/") == 1]
        if len(metadata) != 1:
            raise ValueError("wheel must contain one distribution metadata directory")
        info = Parser().parsestr(stream.read(metadata[0]).decode("utf-8"))
        if any("@" in requirement for requirement in info.get_all("Requires-Dist", [])):
            raise ValueError(
                "wheel dependencies must use package requirements, not direct URLs; lock their wheels in source.lock"
            )
        if (
            normalized(info["Name"]) != normalized(item["name"])
            or info["Version"] != item["version"]
        ):
            raise ValueError("wheel package identity mismatch")
        dist = metadata[0].removesuffix("METADATA")
        record = dist + "RECORD"
        rows = list(csv.reader(io.StringIO(stream.read(record).decode("utf-8"))))
        recorded = set()
        for name, digest, size in rows:
            if name in recorded:
                raise ValueError("duplicate wheel RECORD entry")
            recorded.add(name)
            if name == record and not digest and not size:
                continue
            algorithm, _, encoded = digest.partition("=")
            if algorithm not in {"sha256", "sha384", "sha512"}:
                raise ValueError("wheel RECORD requires strong file digests")
            data = stream.read(name)
            actual = (
                base64.urlsafe_b64encode(hashlib.new(algorithm, data).digest())
                .rstrip(b"=")
                .decode()
            )
            if actual != encoded or str(len(data)) != size:
                raise ValueError("wheel RECORD digest mismatch")
        if recorded != {n.filename for n in stream.infolist() if not n.is_dir()}:
            raise ValueError("wheel RECORD does not cover every file")
        return info


def prepare(plugin: Path, manifest: dict, platform: str, work: Path, cache: Path):
    identity = provenance(plugin, manifest, platform)
    source = manifest["source"]
    archives = [(a, download(a, cache)) for a in identity["artifacts"]]
    primary, archive = next(
        (a, p) for a, p in archives if normalized(a["name"]) == normalized(source["package"])
    )
    if source["location"] == "crate":
        root = extract_archive(archive, work / "source")
        package = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))["package"]
        if package["name"] != source["package"] or package["version"] != source["version"]:
            raise ValueError("crate package identity mismatch")
        if not (root / "Cargo.lock").is_file():
            raise ValueError("published crate must include Cargo.lock for locked builds")
    else:
        root = work / "source"
        root.mkdir(parents=True)
        wheelhouse = work / "wheels"
        wheelhouse.mkdir()
        for item, path in archives:
            wheel_metadata(path, item)
            shutil.copy2(path, wheelhouse / item["filename"])
        with zipfile.ZipFile(archive) as stream:
            stream.extractall(root)
            for member in stream.infolist():
                if not member.is_dir() and member.external_attr >> 16 & 0o111:
                    path = inside(root, member.filename)
                    path.chmod(path.stat().st_mode | (member.external_attr >> 16 & 0o111))
    runtime = inside(root, source["manifest"])
    data = tomllib.loads(runtime.read_text(encoding="utf-8"))
    expected = "rust_dynamic" if manifest["type"] == "native" else "worker"
    if data["plugin"]["kind"] != expected:
        raise ValueError("package runtime manifest and release type disagree")
    (work / "package-source.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )
    return root, runtime, identity
