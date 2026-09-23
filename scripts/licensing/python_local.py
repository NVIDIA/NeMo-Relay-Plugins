# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read Python locks and local package licenses without executing package code."""

import tomllib
from pathlib import Path

from scripts.licensing import attributions_lockfile_md as collector


def checked_path(root: Path, base: Path, value: str) -> Path:
    """Allow sibling packages, but keep all paths inside the pinned checkout."""
    if Path(value).is_absolute() or "\\" in value or ":" in value:
        raise ValueError(f"Expected a relative source path: {value}")
    path = (base / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Attribution path escapes source checkout: {value}")
    return path


def lock_root(source: Path, package: Path) -> Path:
    source = source.resolve()
    current = package.resolve()
    if not current.is_relative_to(source):
        raise ValueError("Python package is outside source checkout")
    while True:
        lock = checked_path(source, current, "uv.lock")
        if lock.is_file():
            return current
        if current == source:
            raise ValueError(
                f"No uv.lock between Python package {package} and source root {source}"
            )
        current = current.parent


def license_texts(root: Path, package: Path, project: dict) -> tuple[str, list[tuple[str, str]]]:
    license_value = project.get("license", "UNKNOWN")
    label = license_value if isinstance(license_value, str) else "UNKNOWN"
    texts = []
    paths = []
    explicit = "license-files" in project
    if isinstance(license_value, dict):
        if "text" in license_value:
            explicit = True
            if not collector._is_useful_license_text(license_value["text"]):
                raise ValueError(f"Empty inline license text for {project.get('name')}")
            texts.append(("pyproject.toml: project.license.text", license_value["text"]))
        if "file" in license_value:
            explicit = True
            paths.append(checked_path(root, package, license_value["file"]))
    for pattern in project.get("license-files", []):
        checked_path(root, package, pattern)
        matches = sorted(package.glob(pattern))
        if not matches:
            raise ValueError(f"License file pattern has no matches: {pattern}")
        paths.extend(matches)
    if not explicit:
        current = package
        while True:
            candidates = sorted(
                p
                for p in current.iterdir()
                if p.is_file() and p.name.lower().startswith(("license", "licence", "copying"))
            )
            if candidates:
                paths.extend(candidates)
                break
            if current == root:
                break
            current = current.parent
    # Preserve notices alongside both the package and any inherited license.
    for path in paths:
        checked_path(root, path.parent, path.name)
    for directory in {package, *(p.parent for p in paths)}:
        paths.extend(
            p for p in directory.iterdir() if p.is_file() and p.name.lower().startswith("notice")
        )
    has_license = bool(texts) or any(not p.name.lower().startswith("notice") for p in paths)
    for path in sorted(set(paths)):
        safe = checked_path(root, path.parent, path.name)
        text = safe.read_text(encoding="utf-8")
        if not collector._is_useful_license_text(text):
            raise ValueError(f"Empty license or notice file: {path.relative_to(root)}")
        texts.append((path.relative_to(root).as_posix(), text))
    if not has_license:
        raise ValueError(f"Missing local license text for {project.get('name')}")
    return collector._normalize_license_name(collector._infer_license_name(label, texts)), texts


def local_package(root: Path, locked_root: Path, pkg: dict) -> collector.RenderedPythonPackage:
    source = pkg["source"]
    relative = source.get("editable", source.get("directory"))
    package = checked_path(root, locked_root, relative)
    manifest = checked_path(root, package, "pyproject.toml")
    project = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]
    if collector._normalize_package_name(project["name"]) != collector._normalize_package_name(
        pkg["name"]
    ) or (project.get("version") != pkg["version"] and "version" not in project.get("dynamic", [])):
        raise ValueError(f"Local package identity differs from uv.lock: {pkg['name']}")
    label, texts = license_texts(root, package, project)
    return collector.RenderedPythonPackage(
        name=pkg["name"], version=pkg["version"], license_name=label, license_texts=texts
    )


def locked_packages(
    source: Path, package: Path, *, include_local=False
) -> list[collector.RenderedPythonPackage]:
    """Use the nearest lock, preserving every locked version and source entry."""
    root = source.resolve()
    locked_root = lock_root(root, package)
    lock = tomllib.loads((locked_root / "uv.lock").read_text(encoding="utf-8"))
    packages = []
    for pkg in lock.get("package", []):
        kind = pkg.get("source", {})
        if "registry" in kind or "git" in kind:
            packages.append(collector._lockfile_only_python_package(pkg))
        elif "editable" in kind or "directory" in kind:
            if include_local:
                packages.append(local_package(root, locked_root, pkg))
        elif "virtual" not in kind:
            raise ValueError(f"Unsupported Python attribution source for {pkg.get('name')}: {kind}")
    return sorted(packages, key=lambda p: (p["name"].lower(), p["version"], p["license_texts"]))
