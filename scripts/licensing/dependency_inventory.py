# SPDX-License-Identifier: Apache-2.0
"""Write exact package dependencies and their inherited declared use to CSV."""

import argparse
import csv
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scripts.catalog import discover
from scripts.licensing import attributions_lockfile_md as collector
from scripts.licensing.python_local import lock_root
from scripts.package_sources import prepare, read_lock


DIRECT_REQUIRED = "direct + required"
DIRECT_OPTIONAL = "direct + optional"
INDIRECT_REQUIRED = "indirect + required"
INDIRECT_OPTIONAL = "indirect + optional"
DEVELOPMENT = "development-only"
TEST = "test-only"
TYPE_PRECEDENCE = {
    DIRECT_REQUIRED: 0,
    INDIRECT_REQUIRED: 1,
    DIRECT_OPTIONAL: 2,
    INDIRECT_OPTIONAL: 3,
    DEVELOPMENT: 4,
    TEST: 5,
}
TYPE_SORT_ORDER = {
    DIRECT_REQUIRED: 0,
    DIRECT_OPTIONAL: 1,
    INDIRECT_REQUIRED: 2,
    INDIRECT_OPTIONAL: 3,
    TEST: 4,
    DEVELOPMENT: 5,
}


@dataclass(frozen=True)
class Dependency:
    name: str
    version: str
    language: str
    dependency_type: str


def _normalized_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(".", "_")


def _deduplicate(rows: Iterable[Dependency]) -> list[Dependency]:
    """Keep the broadest declared use for each exact language package version."""
    selected: dict[tuple[str, str, str], Dependency] = {}
    for row in rows:
        key = (_normalized_name(row.name), row.version, row.language)
        current = selected.get(key)
        if (
            current is None
            or TYPE_PRECEDENCE[row.dependency_type] < TYPE_PRECEDENCE[current.dependency_type]
        ):
            selected[key] = row
    return sorted(
        selected.values(),
        key=lambda row: (
            row.language.lower(),
            TYPE_SORT_ORDER[row.dependency_type],
            row.name.lower(),
            row.version,
        ),
    )


def _indirect_type(dependency_type: str) -> str:
    return {
        DIRECT_REQUIRED: INDIRECT_REQUIRED,
        DIRECT_OPTIONAL: INDIRECT_OPTIONAL,
    }.get(dependency_type, dependency_type)


def _python_package(lock_packages: list[dict], dependency: dict) -> dict:
    candidates = [
        package
        for package in lock_packages
        if _normalized_name(str(package.get("name", "")))
        == _normalized_name(str(dependency.get("name", "")))
    ]
    if dependency.get("version"):
        candidates = [p for p in candidates if p.get("version") == dependency["version"]]
    if dependency.get("source"):
        candidates = [p for p in candidates if p.get("source") == dependency["source"]]
    if len(candidates) != 1:
        identity = dependency.get("name", "unknown")
        raise ValueError(f"Expected one locked Python package for dependency: {identity}")
    return candidates[0]


def _python_roots(lock_packages: list[dict], package: Path, include_workspace: bool) -> list[dict]:
    project = tomllib.loads((package / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if include_workspace:
        roots = [
            item
            for item in lock_packages
            if any(key in item.get("source", {}) for key in ("editable", "directory"))
        ]
    else:
        roots = [
            item
            for item in lock_packages
            if _normalized_name(str(item.get("name", ""))) == _normalized_name(project["name"])
        ]
    if not roots:
        raise ValueError(f"No locked Python package matches {project['name']}")
    return roots


def python_dependencies(
    source: Path, package: Path, *, include_workspace: bool = False
) -> list[Dependency]:
    """Return dependency closures from the package entries in the nearest uv.lock."""
    locked = lock_root(source, package)
    lock_packages = tomllib.loads((locked / "uv.lock").read_text(encoding="utf-8")).get(
        "package", []
    )
    roots = _python_roots(lock_packages, package, include_workspace)
    queue = []
    for root in roots:
        scopes = [(root.get("dependencies", []), DIRECT_REQUIRED)]
        scopes.extend(
            (dependencies, DIRECT_OPTIONAL)
            for dependencies in root.get("optional-dependencies", {}).values()
        )
        scopes.extend(
            (dependencies, TEST if group.lower() in {"test", "tests"} else DEVELOPMENT)
            for group, dependencies in root.get("dev-dependencies", {}).items()
        )
        for dependencies, dependency_type in scopes:
            for dependency in dependencies:
                queue.append((dependency, dependency_type))
    rows = []
    seen = set()
    while queue:
        dependency, dependency_type = queue.pop()
        resolved = _python_package(lock_packages, dependency)
        extras = dependency.get("extra", dependency.get("extras", []))
        if isinstance(extras, str):
            extras = [extras]
        key = (
            resolved["name"],
            resolved["version"],
            str(resolved.get("source", {})),
            dependency_type,
            tuple(sorted(extras)),
        )
        if key in seen:
            continue
        seen.add(key)
        source_kind = resolved.get("source", {})
        is_downloaded = "registry" in source_kind or "git" in source_kind
        is_external_workspace = include_workspace and any(
            key in source_kind for key in ("editable", "directory")
        )
        if is_downloaded or is_external_workspace:
            rows.append(
                Dependency(
                    name=str(resolved["name"]),
                    version=str(resolved["version"]),
                    language="Python",
                    dependency_type=dependency_type,
                )
            )
        child_type = _indirect_type(dependency_type)
        queue.extend((child, child_type) for child in resolved.get("dependencies", []))
        optional = resolved.get("optional-dependencies", {})
        for extra in extras:
            queue.extend((child, child_type) for child in optional.get(extra, []))
    return _deduplicate(rows)


def _rust_resolved_packages(metadata: dict, package_id: str, declaration: dict) -> list[dict]:
    packages = {item["id"]: item for item in metadata.get("packages", [])}
    node = next(item for item in metadata["resolve"]["nodes"] if item["id"] == package_id)
    dependency_name = declaration.get("rename") or declaration["name"]
    kind = declaration.get("kind")
    target = declaration.get("target")
    matches = []
    for resolved in node.get("deps", []):
        if _normalized_name(resolved["name"]) != _normalized_name(dependency_name):
            continue
        if not any(
            dependency_kind.get("kind") == kind and dependency_kind.get("target") == target
            for dependency_kind in resolved.get("dep_kinds", [])
        ):
            continue
        candidate = packages[resolved["pkg"]]
        if candidate["name"] == declaration["name"]:
            matches.append(candidate)
    if not matches:
        raise ValueError(f"No resolved Rust package for dependency: {dependency_name}")
    return matches


def rust_dependencies(
    package: Path, metadata: dict, *, include_workspace: bool = False, direct: bool = True
):
    """Return dependency closures resolved by Cargo for the package or workspace."""
    manifest = (package / "Cargo.toml").resolve()
    packages = metadata.get("packages", [])
    selected = [item for item in packages if Path(item["manifest_path"]).resolve() == manifest]
    if not selected:
        raise ValueError(f"No Cargo package matches manifest: {manifest}")
    if include_workspace:
        member_ids = set(metadata.get("workspace_members", []))
        selected = [item for item in packages if item["id"] in member_ids]
    workspace_ids = set(metadata.get("workspace_members", []))
    queue = []
    for root in selected:
        for declaration in root.get("dependencies", []):
            kind = declaration.get("kind")
            if kind == "dev":
                dependency_type = TEST
            elif kind == "build":
                dependency_type = DEVELOPMENT
            elif kind is None:
                if declaration.get("optional"):
                    dependency_type = DIRECT_OPTIONAL if direct else INDIRECT_OPTIONAL
                else:
                    dependency_type = DIRECT_REQUIRED if direct else INDIRECT_REQUIRED
            else:
                raise ValueError(f"Unsupported Cargo dependency kind: {kind}")
            for resolved in _rust_resolved_packages(metadata, root["id"], declaration):
                queue.append((resolved["id"], dependency_type))
    package_by_id = {item["id"]: item for item in packages}
    node_by_id = {item["id"]: item for item in metadata["resolve"]["nodes"]}
    rows = []
    seen = set()
    while queue:
        package_id, dependency_type = queue.pop()
        key = (package_id, dependency_type)
        if key in seen:
            continue
        seen.add(key)
        resolved = package_by_id[package_id]
        if include_workspace or package_id not in workspace_ids:
            rows.append(
                Dependency(
                    name=str(resolved["name"]),
                    version=str(resolved["version"]),
                    language="Rust",
                    dependency_type=dependency_type,
                )
            )
        for dependency in node_by_id.get(package_id, {}).get("deps", []):
            # Registry package dev-dependencies are not part of its usable graph. Workspace
            # dev-dependencies are already seeded above with their own test-only scope.
            if any(kind.get("kind") != "dev" for kind in dependency.get("dep_kinds", [])):
                queue.append((dependency["pkg"], _indirect_type(dependency_type)))
    return _deduplicate(rows)


def collect(root: Path) -> list[Dependency]:
    """Collect dependencies from repository tools and every registered plugin."""
    from scripts.licensing.generate import project_context, projects

    rows = []
    for source, _, toolchain, languages, include_workspace, package in projects(root):
        if "Python" in languages:
            rows.extend(python_dependencies(source, package, include_workspace=include_workspace))
        if "Rust" in languages:
            with project_context(source, toolchain):
                rows.extend(
                    rust_dependencies(
                        package,
                        collector._cargo_metadata(),
                        include_workspace=include_workspace,
                    )
                )
    for name, manifest in discover(root).items():
        location = manifest["source"]["location"]
        if location not in {"wheel", "crate"}:
            continue
        plugin = root / "plugins" / name
        lock = read_lock(plugin, manifest)
        if location == "wheel":
            # A wheel source lock contains the root wheel and its complete locked closure.
            rows.extend(
                Dependency(
                    item["name"],
                    item["version"],
                    "Python",
                    DIRECT_REQUIRED
                    if _normalized_name(item["name"])
                    == _normalized_name(manifest["source"]["package"])
                    and item["version"] == manifest["source"]["version"]
                    else INDIRECT_REQUIRED,
                )
                for item in lock["artifacts"]
            )
            continue
        seen = set()
        for platform in manifest["platforms"]:
            with tempfile.TemporaryDirectory(prefix="relay-package-dependencies-") as directory:
                source, _, identity = prepare(
                    plugin, manifest, platform, Path(directory), root / ".cache/packages"
                )
                artifact = identity["artifacts"][0]
                if artifact["sha256"] in seen:
                    continue
                seen.add(artifact["sha256"])
                rows.append(
                    Dependency(artifact["name"], artifact["version"], "Rust", DIRECT_REQUIRED)
                )
                with project_context(source, manifest["toolchains"]["rust"]):
                    rows.extend(
                        rust_dependencies(
                            source,
                            collector._cargo_metadata(),
                            include_workspace=True,
                            direct=False,
                        )
                    )
    return _deduplicate(rows)


def write_csv(rows: Iterable[Dependency], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["name", "version", "language", "dependency_type"])
        for row in _deduplicate(rows):
            writer.writerow([row.name, row.version, row.language, row.dependency_type])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_csv(collect(args.root.resolve()), args.output)


if __name__ == "__main__":
    main()
