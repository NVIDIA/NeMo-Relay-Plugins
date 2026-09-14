# SPDX-License-Identifier: Apache-2.0
"""Write exact, direct package dependencies and their declared use to CSV."""

import argparse
import csv
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scripts.catalog import discover
from scripts.licensing import attributions_lockfile_md as collector
from scripts.licensing.python_local import lock_root
from scripts.package_sources import read_lock


REQUIRED = "direct + required"
OPTIONAL = "direct + optional"
DEVELOPMENT = "development-only"
TEST = "test-only"
TYPE_ORDER = {REQUIRED: 0, OPTIONAL: 1, DEVELOPMENT: 2, TEST: 3}


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
        if current is None or TYPE_ORDER[row.dependency_type] < TYPE_ORDER[current.dependency_type]:
            selected[key] = row
    return sorted(
        selected.values(),
        key=lambda row: (row.language.lower(), row.name.lower(), row.version),
    )


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
        raise ValueError(f"Expected one locked Python package for direct dependency: {identity}")
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
    """Return direct dependencies from the package entries in the nearest uv.lock."""
    locked = lock_root(source, package)
    lock_packages = tomllib.loads((locked / "uv.lock").read_text(encoding="utf-8")).get(
        "package", []
    )
    roots = _python_roots(lock_packages, package, include_workspace)
    rows = []
    for root in roots:
        scopes = [(root.get("dependencies", []), REQUIRED)]
        scopes.extend(
            (dependencies, OPTIONAL)
            for dependencies in root.get("optional-dependencies", {}).values()
        )
        scopes.extend(
            (dependencies, TEST if group.lower() in {"test", "tests"} else DEVELOPMENT)
            for group, dependencies in root.get("dev-dependencies", {}).items()
        )
        for dependencies, dependency_type in scopes:
            for dependency in dependencies:
                resolved = _python_package(lock_packages, dependency)
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
        raise ValueError(f"No resolved Rust package for direct dependency: {dependency_name}")
    return matches


def rust_dependencies(package: Path, metadata: dict, *, include_workspace: bool = False):
    """Return direct dependencies resolved by Cargo for the selected package or workspace."""
    manifest = (package / "Cargo.toml").resolve()
    packages = metadata.get("packages", [])
    selected = [item for item in packages if Path(item["manifest_path"]).resolve() == manifest]
    if not selected:
        raise ValueError(f"No Cargo package matches manifest: {manifest}")
    if include_workspace:
        member_ids = set(metadata.get("workspace_members", []))
        selected = [item for item in packages if item["id"] in member_ids]
    workspace_ids = set(metadata.get("workspace_members", []))
    rows = []
    for root in selected:
        for declaration in root.get("dependencies", []):
            kind = declaration.get("kind")
            if kind == "dev":
                dependency_type = TEST
            elif kind == "build":
                dependency_type = DEVELOPMENT
            elif kind is None:
                dependency_type = OPTIONAL if declaration.get("optional") else REQUIRED
            else:
                raise ValueError(f"Unsupported Cargo dependency kind: {kind}")
            for resolved in _rust_resolved_packages(metadata, root["id"], declaration):
                if include_workspace or resolved["id"] not in workspace_ids:
                    rows.append(
                        Dependency(
                            name=str(resolved["name"]),
                            version=str(resolved["version"]),
                            language="Rust",
                            dependency_type=dependency_type,
                        )
                    )
    return _deduplicate(rows)


def collect(root: Path) -> list[Dependency]:
    """Collect direct dependencies from repository tools and every registered plugin."""
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
        if manifest["source"]["location"] not in {"wheel", "crate"}:
            continue
        lock = read_lock(root / "plugins" / name, manifest)
        language = "Python" if manifest["source"]["location"] == "wheel" else "Rust"
        rows.extend(
            Dependency(item["name"], item["version"], language, REQUIRED)
            for item in lock["artifacts"]
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
