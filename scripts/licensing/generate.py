# SPDX-License-Identifier: Apache-2.0
"""Generate per-project notices and deduplicated repository-wide attributions."""

import argparse
import os
import re
import subprocess
import sys
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
        yield root, Path("scripts/licensing"), None, ["Python"]
    if not (root / "plugins").exists():
        return
    for name, manifest in discover(root).items():
        source = manifest["source"]
        registration = Path("plugins") / name
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
        languages = [
            language
            for filename, language in [("pyproject.toml", "Python"), ("Cargo.toml", "Rust")]
            if (package / filename).exists()
        ]
        if not languages:
            raise ValueError(f"No supported package manifest at plugin source: {name}")
        yield source_root, registration, manifest["toolchains"].get("rust"), languages


def sections(text: str, language: str) -> list[str]:
    pattern = r"(?m)(?=^## .+ \(.+\)$)" if language == "Python" else r"(?m)(?=^## .+ - .+$)"
    return [part.strip() for part in re.split(pattern, text)[1:]]


def aggregate(documents: list[str], language: str) -> str:
    """Retain every version/license text while deduplicating identical dependency entries."""
    entries = {entry for document in documents for entry in sections(document, language)}
    header = (
        collector.ATTRIBUTIONS_MD_LICENSE_PREFIX
        + f"\n# Third-Party Software Attributions{' (Python)' if language == 'Python' else ''}\n\n"
        + "Aggregated across all plugins and repository tooling, including all locked versions, "
        + "platforms, and build/test dependencies. Entries retain their individual license terms.\n\n"
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
    for source, destination, toolchain, languages in projects(root):
        print(f"Collecting licenses: {destination}", file=sys.stderr)
        with project_context(source, toolchain):
            if "Python" in languages:
                packages = collector._python_attribution_packages()
                inventory["python"].extend(
                    collector._rendered_python_package_inventory(p) for p in packages
                )
                if not inventory_only:
                    parts = [collector.PYTHON_HEADER]
                    for package in packages:
                        collector._render_python_package(parts, **package)
                    text = "".join(parts).rstrip() + "\n"
                    outputs[destination / "ATTRIBUTIONS-Python.md"] = text
                    documents["Python"].append(text)
            if "Rust" in languages:
                data = collector._cargo_about_json()
                members = collector._cargo_workspace_members()
                inventory["rust"].extend(collector._rust_license_inventory(data, members))
                if not inventory_only:
                    text = collector._render_rust_attributions(data, members)
                    outputs[destination / "ATTRIBUTIONS-Rust.md"] = text
                    documents["Rust"].append(text)
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
            path.write_text(text, encoding="utf-8")
    if args.inventory_output:
        import json

        args.inventory_output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    if stale:
        raise SystemExit("Stale attributions; regenerate and commit: " + ", ".join(stale))


if __name__ == "__main__":
    main()
