# SPDX-License-Identifier: Apache-2.0
"""Validated release manifests and shared path-impact rules."""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = {
    "linux-x86_64": {"runner": "ubuntu-24.04", "target": "x86_64-unknown-linux-gnu"},
    "linux-arm64": {"runner": "ubuntu-24.04-arm", "target": "aarch64-unknown-linux-gnu"},
    "windows-x86_64": {"runner": "windows-2022", "target": "x86_64-pc-windows-msvc"},
    "windows-arm64": {"runner": "windows-11-arm", "target": "aarch64-pc-windows-msvc"},
    "macos-arm64": {"runner": "macos-15", "target": "aarch64-apple-darwin"},
}
SHARED_PREFIXES = ("scripts/", "schemas/", "tests/", ".github/workflows/", ".github/actions/")
SHARED_FILES = {
    "pyproject.toml",
    "uv.lock",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "ATTRIBUTIONS-Python.md",
    "ATTRIBUTIONS-Rust.md",
    ".python-version",
    ".gitattributes",
    ".pre-commit-config.yaml",
}


def inside(root: Path, value: str) -> Path:
    """Resolve a portable relative path without permitting directory escapes."""
    if Path(value).is_absolute() or "\\" in value or ":" in value:
        raise ValueError(f"expected a portable relative path: {value}")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes {root}: {value}")
    return path


def discover(root: Path = ROOT) -> dict[str, dict]:
    schema = json.loads((root / "schemas/release.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    result = {}
    for directory in sorted((root / "plugins").iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        path = directory / "release.toml"
        with path.open("rb") as stream:
            manifest = tomllib.load(stream)
        errors = sorted(validator.iter_errors(manifest), key=lambda e: str(e.path))
        if errors:
            raise ValueError(
                f"{path}: " + "; ".join(f"{list(e.path)}: {e.message}" for e in errors)
            )
        name = manifest["name"]
        if name != directory.name or name in result:
            raise ValueError(f"{path}: name must be unique and match folder")
        for value in (
            manifest["source"]["path"],
            manifest["artifacts"]["bundle"],
            manifest["artifacts"]["manifest"],
        ):
            inside(directory, value)
        if (
            manifest["source"]["location"] == "in-tree"
            and not inside(directory, manifest["source"]["path"]).is_dir()
        ):
            raise ValueError(f"{path}: source directory does not exist")
        manifest.setdefault("platforms", list(PLATFORMS))
        manifest.setdefault("relay", {})
        result[name] = manifest
    if not result:
        raise ValueError("no plugins found")
    return result


def tag_plugin(tag: str, manifests: dict[str, dict]) -> dict:
    matches = [m for m in manifests.values() if tag == f"{m['name']}-{m['version']}"]
    if len(matches) != 1:
        raise ValueError(
            f"invalid plugin tag {tag!r}; expected an existing name and its exact manifest version"
        )
    return matches[0]


def affects(name: str, paths: list[str]) -> bool:
    return any(
        p.startswith(f"plugins/{name}/") or p.startswith(SHARED_PREFIXES) or p in SHARED_FILES
        for p in paths
    )


def git(*args: str, root: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def changed_paths(
    base: str | None, head: str, *, pull_request: bool = False, root: Path = ROOT
) -> list[str] | None:
    if not base or re.fullmatch("0+", base):
        return None
    try:
        # Resolve to hashes before passing event-controlled values to revision arguments.
        base = git("rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}", root=root)
        head = git("rev-parse", "--verify", "--end-of-options", f"{head}^{{commit}}", root=root)
        if pull_request:
            base = git("merge-base", base, head, root=root)
        # --no-renames reports both old and new paths, including cross-plugin moves.
        raw = subprocess.check_output(
            ["git", "diff", "--name-only", "--no-renames", "-z", base, head, "--"], cwd=root
        )
        return [s.decode() for s in raw.split(b"\0") if s]
    except subprocess.CalledProcessError:
        return None


def select(manifests: dict[str, dict], paths: list[str] | None) -> list[dict]:
    return [m for m in manifests.values() if paths is None or affects(m["name"], paths)]
