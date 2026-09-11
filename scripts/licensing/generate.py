# SPDX-License-Identifier: Apache-2.0
"""Generate per-project notices and deduplicated repository-wide attributions."""

import argparse
import os
import re
import subprocess
import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path

from scripts.catalog import discover, inside
from scripts.host import checkout
from scripts.licensing import attributions_lockfile_md as collector


@contextmanager
def project_context(root, toolchain=None):
    previous_root = collector.ROOT
    previous_rust = os.environ.get("RUSTUP_TOOLCHAIN")
    collector.ROOT = root
    if toolchain:
        os.environ["RUSTUP_TOOLCHAIN"] = toolchain
    try:
        yield
    finally:
        collector.ROOT = previous_root
        if previous_rust is None:
            os.environ.pop("RUSTUP_TOOLCHAIN", None)
        else:
            os.environ["RUSTUP_TOOLCHAIN"] = previous_rust


def projects(root: Path):
    """Discover independent locks, checking remote workspaces out only at their locked SHAs."""
    if (root / "uv.lock").exists():
        yield root, Path("scripts/licensing"), None, ["Python"], False, root
    if not (root / "plugins").exists():
        return
    for name, manifest in discover(root).items():
        source = manifest["source"]
        registration = Path("plugins") / name
        if source["location"] in {"wheel", "crate"}:
            continue
        if source["location"] == "remote":
            source_root = root / ".cache/licensing" / name / source["sha"]
            if not source_root.exists():
                checkout(source["repository"], source["sha"], source_root)
            actual = subprocess.check_output(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
            ).strip()
            dirty = subprocess.check_output(
                ["git", "-C", str(source_root), "status", "--porcelain"], text=True
            ).strip()
            if actual != source["sha"] or dirty:
                raise ValueError(f"Remote attribution checkout differs from locked source: {name}")
        else:
            source_root = inside(root / registration, source["path"])
        package = (
            inside(source_root, source["path"]) if source["location"] == "remote" else source_root
        )
        languages = project_languages(package)
        yield (
            source_root,
            registration,
            manifest["toolchains"].get("rust"),
            languages,
            source["location"] == "remote",
            package,
        )


def project_languages(package: Path) -> list[str]:
    languages = [
        language
        for filename, language in [("pyproject.toml", "Python"), ("Cargo.toml", "Rust")]
        if (package / filename).exists()
    ]
    if not languages:
        raise ValueError(f"No supported package manifest at plugin source: {package}")
    return languages


def collect_project(
    source: Path,
    toolchain,
    languages,
    *,
    inventory_only=False,
    include_workspace=False,
    package: Path | None = None,
):
    """Read one project's locked dependencies for either bundles or aggregation."""
    documents = {}
    inventory = {"python": [], "rust": []}
    with project_context(source, toolchain):
        if "Python" in languages:
            from scripts.licensing.python_local import locked_packages

            packages = locked_packages(source, package or source, include_local=include_workspace)
            inventory["python"] = [
                collector._rendered_python_package_inventory(p) for p in packages
            ]
            if not inventory_only:
                parts = [collector.PYTHON_HEADER]
                for package in packages:
                    collector._render_python_package(parts, **package)
                documents["Python"] = "".join(parts).rstrip() + "\n"
        if "Rust" in languages:
            data = collector._cargo_about_json()
            # A remote workspace is external source, including its path dependencies.
            members = set() if include_workspace else collector._cargo_workspace_members()
            inventory["rust"] = collector._rust_license_inventory(data, members)
            if include_workspace:
                locked = tomllib.loads((source / "Cargo.lock").read_text(encoding="utf-8"))
                expected = {(p["name"], p["version"]) for p in locked["package"]}
                actual = {(p["package"], p["version"]) for p in inventory["rust"]}
                missing = expected - actual
                if missing:
                    raise ValueError(f"Missing remote workspace attributions: {sorted(missing)}")
            if not inventory_only:
                documents["Rust"] = collector._render_rust_attributions(data, members)
    return documents, inventory


def write_project_attributions(
    source_root: Path, package: Path, output: Path, *, toolchain=None, include_workspace=False
):
    """Generate bundle notices directly from this project's locked source."""
    documents, _ = collect_project(
        source_root,
        toolchain,
        project_languages(package),
        include_workspace=include_workspace,
        package=package,
    )
    output.mkdir(parents=True, exist_ok=True)
    for language, text in documents.items():
        (output / f"ATTRIBUTIONS-{language}.md").write_text(text, encoding="utf-8", newline="\n")


def sections(text: str, language: str) -> list[str]:
    pattern = r"(?m)(?=^## .+ \(.+\)$)" if language == "Python" else r"(?m)(?=^## .+ - .+$)"
    return [part.strip() for part in re.split(pattern, text)[1:]]


def aggregate(documents: list[str], language: str) -> str:
    """Retain every version/license text while deduplicating identical dependency entries."""
    entries = {entry for document in documents for entry in sections(document, language)}
    header = (
        collector.ATTRIBUTIONS_MD_LICENSE_PREFIX
        + f"\n# Third-Party Software Attributions{' (Python)' if language == 'Python' else ''}\n\n"
        + "This file lists packages used by all plugins and repository tools. It includes all "
        + "locked versions and platforms, plus build and test packages. Each package keeps its own license terms.\n\n"
        + "Automatically generated. Regenerate with `uv run --locked python -m scripts.licensing.generate`.\n\n"
    )
    return (
        header
        + "\n\n".join(sorted(entries, key=lambda entry: (entry.splitlines()[0].lower(), entry)))
        + "\n"
    )


def collect(root: Path, *, inventory_only=False) -> tuple[dict[Path, str], dict]:
    outputs = {}
    documents = {"Python": [], "Rust": []}
    inventory = {"python": [], "rust": []}
    for source, destination, toolchain, languages, include_workspace, package in projects(root):
        print(f"Collecting licenses: {destination}", file=sys.stderr)
        project_documents, project_inventory = collect_project(
            source,
            toolchain,
            languages,
            inventory_only=inventory_only,
            include_workspace=include_workspace,
            package=package,
        )
        for language, text in project_documents.items():
            # Per-plugin notices belong in generated bundles, not source control.
            if destination.parts[0] != "plugins":
                outputs[destination / f"ATTRIBUTIONS-{language}.md"] = text
            documents[language].append(text)
        for language, rows in project_inventory.items():
            inventory[language].extend(rows)
    if (root / "plugins").exists():
        for name, manifest in discover(root).items():
            if manifest["source"]["location"] not in {"wheel", "crate"}:
                continue
            from scripts.licensing.packages import collect_packages

            print(f"Collecting package licenses: plugins/{name}", file=sys.stderr)
            package_documents, package_inventory = collect_packages(
                root / "plugins" / name,
                manifest,
                root / ".cache/packages",
                inventory_only=inventory_only,
            )
            for language, text in package_documents.items():
                documents[language].append(text)
            for language, rows in package_inventory.items():
                inventory[language].extend(rows)
    for language, texts in documents.items():
        if texts:
            outputs[Path(f"ATTRIBUTIONS-{language}.md")] = aggregate(texts, language)
    for language, rows in inventory.items():
        unique = {(row["package"], row["version"], row["license"]): row for row in rows}
        inventory[language] = [unique[key] for key in sorted(unique)]
    return outputs, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--check", action="store_true", help="Fail if any committed attribution is stale"
    )
    parser.add_argument(
        "--inventory-output", type=Path, help="Also save a license inventory for diffing"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    outputs, inventory = collect(root)
    stale = []
    for relative, text in outputs.items():
        path = root / relative
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(str(relative))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
    if args.inventory_output:
        import json

        args.inventory_output.parent.mkdir(parents=True, exist_ok=True)
        args.inventory_output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    if stale:
        raise SystemExit("Stale attributions; regenerate and commit: " + ", ".join(stale))


if __name__ == "__main__":
    main()
