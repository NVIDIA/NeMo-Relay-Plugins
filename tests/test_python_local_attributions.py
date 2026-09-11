# SPDX-License-Identifier: Apache-2.0
"""Remote Python notices use the selected lock and pinned local package sources."""

import re
import subprocess
from pathlib import Path

import pytest
import tomli_w

from scripts.bundles import create_archive, extract_archive
from scripts.licensing import generate, python_local, license_diff

LICENSE = "MIT License\nCopyright Bjørn Authors\nPermission is hereby granted.\n"


def write_toml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(data), encoding="utf-8")


def package(root, relative, name, **metadata):
    folder = root / relative
    write_toml(
        folder / "pyproject.toml",
        {"project": {"name": name, "version": "1.0", "license": "MIT", **metadata}},
    )
    (folder / "LICENSE").write_text(LICENSE, encoding="utf-8")
    return folder


def lock_entry(name, kind, relative, version="1.0"):
    return {"name": name, "version": version, "source": {kind: relative}}


@pytest.mark.parametrize("layout", ["subdirectory", "workspace"])
@pytest.mark.parametrize("extension", [".zip", ".tar.gz"])
def test_remote_python_root_inventory_and_bundle_use_same_locked_sources(
    tmp_path, monkeypatch, layout, extension
):
    upstream = tmp_path / "upstream"
    plugin = package(upstream, "python/plugin", "remote-plugin")
    package(upstream, "python/helper", "local-helper")
    package(upstream, "python/editable", "editable-helper")
    locked_root = plugin if layout == "subdirectory" else upstream
    relative = "." if layout == "subdirectory" else "python/plugin"
    helper = "../helper" if layout == "subdirectory" else "python/helper"
    editable = "../editable" if layout == "subdirectory" else "python/editable"
    entries = [
        lock_entry("remote-plugin", "editable", relative),
        lock_entry("local-helper", "directory", helper),
        lock_entry("editable-helper", "editable", editable),
        lock_entry("workspace-tools", "virtual", "."),
    ]
    write_toml(locked_root / "uv.lock", {"version": 1, "package": entries})
    if layout == "subdirectory":
        # A different root project must never override the plugin's own lock.
        write_toml(
            upstream / "uv.lock",
            {"version": 1, "package": [lock_entry("unrelated", "directory", "missing")]},
        )
    subprocess.run(["git", "init", "-q", str(upstream)], check=True)
    subprocess.run(["git", "-C", str(upstream), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(upstream),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Pinned fixture",
        ],
        check=True,
    )
    sha = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    repository = tmp_path / "repository"
    (repository / "plugins").mkdir(parents=True)
    monkeypatch.setattr(
        generate,
        "discover",
        lambda root: {
            "remote-plugin": {
                "source": {
                    "location": "remote",
                    "repository": str(upstream),
                    "sha": sha,
                    "ref": "main",
                    "path": "python/plugin",
                },
                "toolchains": {},
            }
        },
    )
    # Later working-tree changes must not affect the checkout at the pinned SHA.
    (plugin / "LICENSE").write_text("Uncommitted license change")
    outputs, inventory = generate.collect(repository)
    text = outputs[Path("ATTRIBUTIONS-Python.md")]
    names = {"remote-plugin", "local-helper", "editable-helper"}
    assert set(re.findall(r"^## (.+) \(1.0\)$", text, re.M)) == names
    assert {p["package"] for p in inventory["python"]} == names
    assert "Uncommitted" not in text
    assert LICENSE in text
    assert str(tmp_path) not in text
    assert set(outputs) == {Path("ATTRIBUTIONS-Python.md")}
    _, inventory_only = generate.collect(repository, inventory_only=True)
    assert inventory_only == inventory
    assert len(license_diff.compare_inventories({}, inventory)["python"]["added"]) == 3
    checkout = repository / ".cache/licensing/remote-plugin" / sha
    bundle = tmp_path / "bundle"
    generate.write_project_attributions(
        checkout, checkout / "python/plugin", bundle, include_workspace=True
    )
    archive = tmp_path / ("bundle" + extension)
    create_archive(bundle, archive, "plugin")
    extracted = extract_archive(archive, tmp_path / "extracted")
    bundled = (extracted / "ATTRIBUTIONS-Python.md").read_text(encoding="utf-8")
    assert generate.aggregate([bundled], "Python") == text
    assert not list(checkout.rglob("ATTRIBUTIONS*.md"))


@pytest.mark.parametrize("source_kind", ["editable", "directory"])
def test_local_packages_remain_excluded_for_in_tree_plugins(tmp_path, source_kind):
    own = package(tmp_path, ".", "own")
    write_toml(own / "uv.lock", {"package": [lock_entry("own", source_kind, ".")]})
    assert python_local.locked_packages(tmp_path, own) == []
    assert len(python_local.locked_packages(tmp_path, own, include_local=True)) == 1


def test_nearest_lock_stays_within_checkout(tmp_path):
    root = tmp_path / "checkout"
    plugin = package(root, "nested/plugin", "own")
    write_toml(tmp_path / "uv.lock", {"package": []})
    with pytest.raises(ValueError, match="No uv.lock"):
        python_local.locked_packages(root, plugin)
    with pytest.raises(ValueError, match="outside source checkout"):
        python_local.locked_packages(root, tmp_path)


@pytest.mark.parametrize(
    "mode",
    [
        "escape",
        "symlink",
        "name",
        "version",
        "missing-license",
        "missing-declared",
        "empty",
        "unsupported",
    ],
)
def test_bad_remote_local_package_fails_instead_of_omitting_notice(tmp_path, mode):
    root = tmp_path / "checkout"
    own = package(root, "plugin", "own")
    entries = [lock_entry("own", "editable", "plugin")]
    if mode == "escape":
        entries[0]["source"] = {"directory": "../outside"}
    elif mode == "symlink":
        outside = package(tmp_path, "outside", "own")
        (own / "LICENSE").unlink()
        (own / "LICENSE").symlink_to(outside / "LICENSE")
    elif mode in {"name", "version"}:
        entries[0][mode] = "different"
    elif mode == "missing-license":
        (own / "LICENSE").unlink()
        (own / "NOTICE").write_text("A notice alone is not a license.")
    elif mode == "missing-declared":
        package(root, "plugin", "own", **{"license-files": ["legal/*.txt"]})
    elif mode == "empty":
        (own / "LICENSE").write_text("")
    else:
        entries[0]["source"] = {"unknown": "somewhere"}
    write_toml(root / "uv.lock", {"package": entries})
    with pytest.raises(ValueError):
        generate.write_project_attributions(root, own, root / "bundle", include_workspace=True)
    assert not (root / "bundle").exists()


@pytest.mark.parametrize(
    "metadata",
    [
        {"license": {"file": "legal/terms.txt"}},
        {"license": {"text": LICENSE}},
        {"license": "MIT", "license-files": ["legal/*.txt"]},
        {"license": "MIT"},
    ],
)
def test_local_license_metadata_and_parent_license_fallback(tmp_path, metadata):
    own = package(tmp_path, "plugin", "own", **metadata)
    (own / "LICENSE").unlink()
    (tmp_path / "LICENSE").write_text(LICENSE, encoding="utf-8")
    (own / "NOTICE").write_text("Keep this notice", encoding="utf-8")
    (own / "legal").mkdir()
    (own / "legal/terms.txt").write_text(LICENSE, encoding="utf-8")
    write_toml(own / "uv.lock", {"package": [lock_entry("own", "editable", ".")]})
    rendered = python_local.locked_packages(tmp_path, own, include_local=True)[0]
    assert rendered["license_name"] == "MIT"
    assert any(text == LICENSE for _, text in rendered["license_texts"])
    assert any(text == "Keep this notice" for _, text in rendered["license_texts"])
    assert all(str(tmp_path) not in label for label, _ in rendered["license_texts"])


def test_external_locked_versions_are_not_collapsed(tmp_path, monkeypatch):
    write_toml(
        tmp_path / "uv.lock",
        {
            "package": [
                lock_entry("dep", "registry", "https://example.com", version=v) for v in ["1", "2"]
            ]
        },
    )
    monkeypatch.setattr(
        python_local.collector,
        "_lockfile_only_python_package",
        lambda p: {
            "name": p["name"],
            "version": p["version"],
            "license_name": "MIT",
            "license_texts": [("LICENSE", LICENSE)],
        },
    )
    assert [p["version"] for p in python_local.locked_packages(tmp_path, tmp_path)] == ["1", "2"]
