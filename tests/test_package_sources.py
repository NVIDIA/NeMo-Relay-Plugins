# SPDX-License-Identifier: Apache-2.0
"""Published package identities, immutable downloads, installation and notices."""

import base64
import csv
import hashlib
import io
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import tomli_w
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import Tag

from scripts import package_sources as sources, plugins
from scripts.bundles import sha256, verify_bundle, create_archive, extract_archive
from scripts.licensing import generate
from scripts.tasks import Context, package_wheel


def test_locked_runtime_export_selects_exact_base_requirements():
    environment = default_environment()
    environment["python_full_version"] = "3.11.9"
    requirements = sources.parse_locked_runtime_requirements(
        "base==1.0\nold==2.0 ; python_full_version < '3.12'\n"
        "new==3.0 ; python_full_version >= '3.12'\n",
        environment=environment,
    )
    assert [(requirement.name, str(requirement.specifier)) for requirement in requirements] == [
        ("base", "==1.0"),
        ("old", "==2.0"),
    ]
    for invalid in ["range>=1", "profile[extra]==1", "direct @ https://example.com/a.whl"]:
        with pytest.raises(ValueError, match="locked runtime"):
            sources.parse_locked_runtime_requirements(invalid, environment=environment)


def test_declared_python_drives_markers_and_wheel_tags():
    environment = sources.marker_environment_for_python("3.11")
    assert environment["python_version"] == "3.11"
    assert environment["python_full_version"] == "3.11.0"
    tags = sources.compatible_tags_for_python("3.11")
    assert any(tag.interpreter == "cp311" and tag.abi == "cp311" for tag in tags)
    assert any(
        tag.interpreter == "py3" and tag.abi == "none" and tag.platform == "any" for tag in tags
    )
    assert not any(
        tag.interpreter == "cp311" and tag.abi == "cp311" and tag.platform == "any" for tag in tags
    )


def test_locked_wheel_selection_uses_compatible_hashed_artifact():
    lock = {
        "package": [
            {
                "name": "demo-package",
                "version": "1.0",
                "source": {"registry": "https://pypi.org/simple"},
                "wheels": [
                    {
                        "url": "https://files.example/demo_package-1.0-cp311-cp311-win_amd64.whl",
                        "hash": "sha256:" + "1" * 64,
                    },
                    {
                        "url": "https://files.example/demo_package-1.0-py3-none-any.whl",
                        "hash": "sha256:" + "2" * 64,
                    },
                ],
            }
        ]
    }
    artifacts = sources.select_locked_wheel_artifacts(
        lock,
        [Requirement("demo-package==1.0")],
        "linux-x86_64",
        compatible_tags=[Tag("py3", "none", "any")],
    )
    assert artifacts == [
        {
            "name": "demo-package",
            "version": "1.0",
            "filename": "demo_package-1.0-py3-none-any.whl",
            "url": "https://files.example/demo_package-1.0-py3-none-any.whl",
            "sha256": "2" * 64,
            "platforms": ["linux-x86_64"],
        }
    ]


def wheel(path, name="published_worker", version="2.0", requires=()):
    dist = f"{name}-{version}.dist-info"
    manifest = {
        "plugin": {"kind": "worker", "id": "fixture.published"},
        "source": {"artifact": "worker.py", "manifest_root": ".."},
        "integrity": {
            "sha256": "sha256:"
            + hashlib.sha256(b'def main(): return "request passed"\n').hexdigest()
        },
        "load": {"runtime": "python", "entrypoint": f"{name}.worker:main"},
    }
    files = {
        f"{name}/__init__.py": b"",
        f"{name}/worker.py": b'def main(): return "request passed"\n',
        f"{name}/relay-plugin.toml": tomli_w.dumps(manifest).encode(),
        f"{dist}/METADATA": (
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\nLicense-Expression: MIT\nLicense-File: LICENSE\n"
            + "".join(f"Requires-Dist: {r}\n" for r in requires)
        ).encode(),
        f"{dist}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{dist}/licenses/LICENSE": b"MIT License\nCopyright Fixture Authors\nPermission is hereby granted.\n",
    }
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for filename, data in files.items():
        writer.writerow(
            [
                filename,
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(),
                len(data),
            ]
        )
    writer.writerow([f"{dist}/RECORD", "", ""])
    files[f"{dist}/RECORD"] = record.getvalue().encode()
    with zipfile.ZipFile(path, "w") as stream:
        for filename, data in files.items():
            stream.writestr(filename, data)
    return path


@pytest.fixture
def published(tmp_path, monkeypatch):
    plugin = tmp_path / "plugins/registered-worker"
    plugin.mkdir(parents=True)
    filename = "published_worker-2.0-py3-none-any.whl"
    artifact = wheel(tmp_path / filename)
    platform = plugins.local_platform()
    manifest = {
        "schema_version": 1,
        "name": plugin.name,
        "version": "1.0.0",
        "type": "worker",
        "metadata": {"description": "Fixture", "license": "MIT"},
        "platforms": [platform],
        "relay": {},
        "source": {
            "location": "wheel",
            "package": "published-worker",
            "version": "2.0",
            "manifest": "published_worker/relay-plugin.toml",
        },
        "toolchains": {"python": "3.11"},
        "artifacts": {"bundle": "bundle", "manifest": "relay-plugin.toml"},
        "commands": {
            s: {"argv": ["true"], "cwd": "plugin"} for s in ["build", "test", "package", "smoke"]
        },
    }
    item = {
        "name": "published-worker",
        "version": "2.0",
        "filename": filename,
        "url": "https://example.com/" + filename,
        "sha256": sha256(artifact),
        "platforms": [platform],
    }
    (plugin / "source.lock").write_text(tomli_w.dumps({"schema_version": 1, "artifacts": [item]}))
    (plugin / "release.toml").write_text(tomli_w.dumps(manifest))

    def download(url, **kwargs):
        assert url == item["url"]
        response = io.BytesIO(artifact.read_bytes())
        response.url = url
        return response

    monkeypatch.setattr(sources.urllib.request, "urlopen", download)
    return plugin, manifest, item, artifact


@pytest.mark.parametrize(
    "mutation", ["missing-platform", "hash", "duplicate", "version", "url", "filename", "extra"]
)
def test_package_lock_validation(published, mutation):
    plugin, manifest, item, _ = published
    items = [item]
    if mutation == "missing-platform":
        item["platforms"] = []
    elif mutation == "hash":
        item["sha256"] = "a"
    elif mutation == "duplicate":
        items.append(item)
    elif mutation == "version":
        item["version"] = "3.0"
    elif mutation == "url":
        item["url"] = "http://example.com/file"
    elif mutation == "filename":
        item["filename"] = "../file.whl"
    else:
        item["extra"] = True
    (plugin / "source.lock").write_text(tomli_w.dumps({"schema_version": 1, "artifacts": items}))
    with pytest.raises(ValueError):
        sources.read_lock(plugin, manifest)


def test_download_and_cache_digests(published, tmp_path):
    _, _, item, _ = published
    downloaded = sources.download(item, tmp_path / "cache")
    downloaded.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cached.*digest"):
        sources.download(item, tmp_path / "cache")
    item["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="download digest"):
        sources.download(item, tmp_path / "fresh")


def test_wheel_identity_and_record(published, tmp_path):
    _, _, item, artifact = published
    wrong = dict(item, version="8")
    with pytest.raises(ValueError, match="identity"):
        sources.wheel_metadata(artifact, wrong)
    with zipfile.ZipFile(artifact) as source:
        files = {n: source.read(n) for n in source.namelist()}
    files["published_worker/worker.py"] = b"tampered"
    with zipfile.ZipFile(artifact, "w") as stream:
        for name, data in files.items():
            stream.writestr(name, data)
    with pytest.raises(ValueError, match="RECORD digest"):
        sources.wheel_metadata(artifact, item)


def test_wheel_bundle_installs_after_relocation_and_checks_tampering(published, tmp_path):
    import subprocess

    plugin, manifest, _, _ = published
    work = tmp_path / "work"
    root, runtime, identity = sources.prepare(
        plugin, manifest, manifest["platforms"][0], work, tmp_path / "cache"
    )
    assert runtime == root / manifest["source"]["manifest"]
    interpreter = sources.install_wheels(work, sys.executable)
    assert (
        subprocess.check_output(
            [str(interpreter), "-c", "from published_worker.worker import main; print(main())"],
            text=True,
        ).strip()
        == "request passed"
    )
    ctx = Context(plugin, root, root, work, work / "target", manifest["platforms"][0], manifest)
    package_wheel(ctx)
    verify_bundle(ctx.bundle, "relay-plugin.toml", "worker")
    for extension in [".zip", ".tar.gz"]:
        archive = tmp_path / ("bundle" + extension)
        create_archive(ctx.bundle, archive, "plugin")
        moved = extract_archive(archive, tmp_path / ("moved" + extension))
        assert "published-worker (2.0)" in (moved / "ATTRIBUTIONS-Python.md").read_text()
        venv = tmp_path / ("environment" + extension)
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        # Exercise the same project-install interface used by Relay, without an index.
        subprocess.run([str(python), "-m", "pip", "install", "--no-index", str(moved)], check=True)
        assert (
            subprocess.check_output(
                [str(python), "-c", "from published_worker.worker import main; print(main())"],
                text=True,
            ).strip()
            == "request passed"
        )
        next((moved / "wheelhouse").glob("*.whl")).write_bytes(b"tampered")
        result = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-index",
                "--reinstall",
                str(moved),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode and "bundled wheel digest mismatch" in result.stderr


def test_wheel_missing_dependencies_fail_native_install(published, tmp_path):
    plugin, manifest, item, artifact = published
    wheel(artifact, requires=["missing-package==9.9.9"])
    item["sha256"] = sha256(artifact)
    (plugin / "source.lock").write_text(tomli_w.dumps({"schema_version": 1, "artifacts": [item]}))
    sources.prepare(
        plugin, manifest, manifest["platforms"][0], tmp_path / "work", tmp_path / "cache"
    )
    import subprocess

    with pytest.raises(subprocess.CalledProcessError):
        sources.install_wheels(tmp_path / "work", sys.executable)


def test_package_source_schema_and_root_aggregation(published, tmp_path):
    from scripts.catalog import discover, ROOT

    plugin, manifest, _, _ = published
    (tmp_path / "schemas").mkdir()
    shutil.copy2(ROOT / "schemas/release.schema.json", tmp_path / "schemas/release.schema.json")
    assert discover(tmp_path)[manifest["name"]]["source"]["location"] == "wheel"
    docs, inventory = generate.collect(tmp_path)
    assert "published-worker (2.0)" in docs[Path("ATTRIBUTIONS-Python.md")]
    assert inventory["python"][0]["package"] == "published-worker"


def test_crate_source_identity_manifest_and_locked_dependencies(published, tmp_path, monkeypatch):
    plugin, manifest, item, _ = published
    manifest["source"] = {
        "location": "crate",
        "package": "published-crate",
        "version": "1.0.0",
        "manifest": "relay-plugin.toml",
    }
    manifest["toolchains"]["rust"] = "1.96.1"
    item.update(
        name="published-crate",
        version="1.0.0",
        filename="published-crate-1.0.0.crate",
        url="https://example.com/source.crate",
    )
    archive = tmp_path / item["filename"]
    with tarfile.open(archive, "w:gz") as stream:
        for name, data in {
            "Cargo.toml": '[package]\nname="published-crate"\nversion="1.0.0"\n',
            "Cargo.lock": "version=4\n",
            "relay-plugin.toml": '[plugin]\nkind="worker"\n',
        }.items():
            info = tarfile.TarInfo("published-crate-1.0.0/" + name)
            info.size = len(data.encode())
            stream.addfile(info, io.BytesIO(data.encode()))
    item["sha256"] = sha256(archive)
    (plugin / "source.lock").write_text(tomli_w.dumps({"schema_version": 1, "artifacts": [item]}))
    monkeypatch.setattr(sources, "download", lambda *args: archive)
    root, runtime, provenance = sources.prepare(
        plugin, manifest, manifest["platforms"][0], tmp_path / "work", tmp_path / "cache"
    )
    assert root.name == "published-crate-1.0.0"
    assert runtime.is_file() and (root / "Cargo.lock").is_file()
    assert provenance["artifacts"][0]["sha256"] == sha256(archive)


def test_published_wheel_runs_all_plugin_owned_stages(published, tmp_path, monkeypatch):
    plugin, manifest, item, _ = published
    repository = Path(__file__).resolve().parents[1]
    task = plugin / "tasks.py"
    task.write_text("""import os, subprocess, sys
from pathlib import Path
sys.path.insert(0, os.environ['REPO_DIR'])
from scripts.tasks import Context, package_wheel
ctx = Context.from_environment()
stage = sys.argv[1]
with (ctx.plugin / 'stages').open('a') as stream: stream.write(stage + '\\n')
assert Path(os.environ['SOURCE_MANIFEST']).is_file()
assert Path(os.environ['PACKAGE_SOURCE_LOCK']).is_file()
if stage in ('build', 'test'):
    subprocess.run([os.environ['PACKAGE_PYTHON'], '-c', 'from published_worker.worker import main; assert main() == "request passed"'], check=True)
elif stage == 'package':
    package_wheel(ctx)
else:
    from scripts.bundles import verify_bundle
    bundle = Path(os.environ['BUNDLE_DIR'])
    assert not bundle.is_relative_to(ctx.plugin.parent.parent)
    verify_bundle(bundle, 'relay-plugin.toml', 'worker')
""")
    # The fixture uses the real helpers while keeping its registration outside the repo.
    for stage in manifest["commands"]:
        manifest["commands"][stage] = {
            "argv": ["${PYTHON}", "${PLUGIN_DIR}/tasks.py", stage],
            "cwd": "plugin",
        }
    (plugin / "release.toml").write_text(tomli_w.dumps(manifest))
    monkeypatch.setattr(plugins, "ROOT", tmp_path)
    monkeypatch.setattr(plugins, "git", lambda *args: "")
    original_run = plugins.subprocess.run
    original_output = plugins.subprocess.check_output

    def run(argv, **kwargs):
        if "env" in kwargs and "REPO_DIR" in kwargs["env"]:
            kwargs["env"] = dict(kwargs["env"], REPO_DIR=str(repository))
        return original_run(argv, **kwargs)

    def output(argv, **kwargs):
        if argv[:2] == ["uv", "python"]:
            return sys.executable + "\n"
        return original_output(argv, **kwargs)

    monkeypatch.setattr(plugins.subprocess, "run", run)
    monkeypatch.setattr(plugins.subprocess, "check_output", output)
    destination = plugins.run_plugin(
        manifest,
        manifest["platforms"][0],
        {"sha": "a" * 40, "tag": None},
        "b" * 40,
        relay=Path(sys.executable),
    )
    assert (plugin / "stages").read_text().splitlines() == ["build", "test", "package", "smoke"]
    metadata = json.loads(next(destination.glob("*.json")).read_text())
    assert metadata["source_commit"] is None
    assert metadata["package_source"]["artifacts"][0]["sha256"] == item["sha256"]
    assert metadata["verified"] is True


def test_package_assets_reject_changed_locks(published, tmp_path, monkeypatch):
    from scripts import catalog
    from scripts.bundles import verify_assets
    from test_bundles import assets_for

    plugin, manifest, item, _ = published
    assets = tmp_path / "assets"
    assets_for(manifest, assets, "b" * 40)
    metadata_path = next(assets.glob("*.json"))
    metadata = json.loads(metadata_path.read_text())
    metadata["package_source"] = sources.provenance(plugin, manifest, manifest["platforms"][0])
    metadata_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    assert len(verify_assets(manifest, assets, "b" * 40)) == 3
    item["sha256"] = "0" * 64
    (plugin / "source.lock").write_text(tomli_w.dumps({"schema_version": 1, "artifacts": [item]}))
    with pytest.raises(ValueError, match="provenance"):
        verify_assets(manifest, assets, "b" * 40)


def test_crate_can_build_test_and_collect_licenses_from_published_archive(
    published, tmp_path, monkeypatch
):
    import subprocess
    from scripts.licensing.packages import collect_packages

    plugin, manifest, item, _ = published
    upstream = tmp_path / "upstream"
    (upstream / "src").mkdir(parents=True)
    (upstream / "Cargo.toml").write_text(
        '[package]\nname="published-crate"\nversion="1.0.0"\nedition="2021"\nlicense="MIT"\n'
    )
    (upstream / "src/main.rs").write_text(
        'fn main() { println!("request passed"); }\n#[test] fn behavior() { assert_eq!(2+2, 4); }\n'
    )
    (upstream / "LICENSE").write_text(
        "MIT License\nCopyright Fixture Authors\nPermission is hereby granted.\n"
    )
    (upstream / "relay-plugin.toml").write_text('[plugin]\nkind="worker"\n')
    cargo = ["cargo", "+1.96.1"]
    subprocess.run(cargo + ["generate-lockfile", "--offline"], cwd=upstream, check=True)
    subprocess.run(
        cargo + ["package", "--locked", "--no-verify", "--allow-dirty"], cwd=upstream, check=True
    )
    archive = upstream / "target/package/published-crate-1.0.0.crate"
    manifest["source"] = {
        "location": "crate",
        "package": "published-crate",
        "version": "1.0.0",
        "manifest": "relay-plugin.toml",
    }
    manifest["toolchains"]["rust"] = "1.96.1"
    item.update(
        name="published-crate",
        version="1.0.0",
        filename=archive.name,
        url="https://example.com/source.crate",
        sha256=sha256(archive),
    )
    (plugin / "source.lock").write_text(tomli_w.dumps({"schema_version": 1, "artifacts": [item]}))
    monkeypatch.setattr(sources, "download", lambda *args: archive)
    # The licensing module imports the same download function separately.
    monkeypatch.setattr("scripts.licensing.packages.download", lambda *args: archive)
    root, _, _ = sources.prepare(
        plugin, manifest, manifest["platforms"][0], tmp_path / "work", tmp_path / "cache"
    )
    subprocess.run(cargo + ["test", "--locked", "--release"], cwd=root, check=True)
    subprocess.run(cargo + ["build", "--locked", "--release"], cwd=root, check=True)
    executable = (
        root
        / "target/release"
        / ("published-crate.exe" if sys.platform == "win32" else "published-crate")
    )
    assert subprocess.check_output([str(executable)], text=True).strip() == "request passed"
    documents, inventory = collect_packages(plugin, manifest, tmp_path / "cache")
    assert "published-crate - 1.0.0" in documents["Rust"]
    assert inventory["rust"][0]["package"] == "published-crate"


@pytest.mark.parametrize("kind", ["wheel", "crate"])
def test_package_release_notes_link_exact_upstream_version(repo, kind):
    from scripts.catalog import discover
    from scripts.releases import release_notes
    from test_releases import FakeGitHub

    root, git, commit = repo
    manifest = discover(root)["example-python-grpc-worker-plugin"]
    manifest["source"] = {
        "location": kind,
        "package": "published-package",
        "version": "2.3.4",
        "manifest": "relay-plugin.toml",
    }
    manifest["metadata"]["documentation"] = "https://example.com/docs"
    head = commit("README.md")
    notes = release_notes(
        manifest, manifest["name"] + "-0.1.0", head, "a/b", FakeGitHub(head), [], root
    )
    assert "published-package 2.3.4" in notes
    assert "https://example.com/docs" in notes
    assert ("pypi.org" if kind == "wheel" else "crates.io") in notes


def test_wheel_direct_url_dependencies_are_rejected(published):
    _, _, item, artifact = published
    wheel(artifact, requires=["dep @ https://example.com/unlocked.whl"])
    with pytest.raises(ValueError, match="direct URLs"):
        sources.wheel_metadata(artifact, item)


def test_each_declared_platform_needs_a_locked_plugin(published):
    from scripts.catalog import PLATFORMS

    plugin, manifest, _, _ = published
    manifest["platforms"].append(next(p for p in PLATFORMS if p not in manifest["platforms"]))
    with pytest.raises(ValueError, match="one exact plugin"):
        sources.read_lock(plugin, manifest)


def test_missing_embedded_manifest_fails(published, tmp_path):
    plugin, manifest, _, _ = published
    manifest["source"]["manifest"] = "missing.toml"
    with pytest.raises(FileNotFoundError):
        sources.prepare(
            plugin, manifest, manifest["platforms"][0], tmp_path / "work", tmp_path / "cache"
        )


def test_download_redirect_must_stay_https(published, tmp_path, monkeypatch):
    _, _, item, artifact = published

    def response(*args, **kwargs):
        value = io.BytesIO(artifact.read_bytes())
        value.url = "http://example.com/insecure"
        return value

    monkeypatch.setattr(sources.urllib.request, "urlopen", response)
    with pytest.raises(ValueError, match="redirected"):
        sources.download(item, tmp_path / "cache")
