# SPDX-License-Identifier: Apache-2.0
import hashlib
import subprocess
import tomllib

import pytest
import yaml

from scripts import checks
from scripts.catalog import ROOT, select, discover


def runtime_manifest(tmp_path, artifact="worker.py"):
    path = tmp_path / "relay-plugin.toml"
    path.write_text(
        '# Keep this source notice.\n[source]\nartifact = "' + artifact + '"\n'
        '[integrity]\nsha256 = "sha256:old" # Keep this comment.\n'
    )
    return path


def test_formatting_refreshes_digest_and_preserves_manifest_comments(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_bytes(b"print('formatted')\n")
    path = runtime_manifest(tmp_path)
    checks.update_source_hash(path)
    result = path.read_bytes()
    assert tomllib.loads(result.decode())["integrity"]["sha256"] == (
        "sha256:" + hashlib.sha256(worker.read_bytes()).hexdigest()
    )
    assert b"# Keep this source notice." in result
    assert b"# Keep this comment." in result
    checks.update_source_hash(path)
    assert path.read_bytes() == result
    worker.write_bytes(b"print('changed again')\n")
    checks.update_source_hash(path)
    assert path.read_bytes() != result


def test_source_hash_rejects_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        checks.update_source_hash(runtime_manifest(tmp_path))


def test_source_hash_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="path escapes"):
        checks.update_source_hash(runtime_manifest(tmp_path, "../outside.py"))


def test_compiled_artifact_template_waits_for_packaging(tmp_path):
    path = runtime_manifest(tmp_path, "target/release/<platform-library-file>")
    original = path.read_bytes()
    checks.update_source_hash(path)
    assert path.read_bytes() == original


def test_rust_format_uses_each_in_tree_toolchain(monkeypatch, tmp_path):
    sources = []
    for version in ["1.95.0", "1.96.1"]:
        source = tmp_path / f"source with spaces {version}"
        source.mkdir()
        (source / "Cargo.toml").touch()
        sources.append((source, {"toolchains": {"rust": version}}))
    monkeypatch.setattr(checks, "in_tree_sources", lambda root: iter(sources))
    calls = []
    monkeypatch.setattr(checks.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    checks.rust_fmt(tmp_path)
    assert [args for args, _ in calls] == [
        ["cargo", "+1.95.0", "fmt", "--all"],
        ["cargo", "+1.96.1", "fmt", "--all"],
    ]
    assert [kw["cwd"] for _, kw in calls] == [source for source, _ in sources]
    assert all(kw["check"] for _, kw in calls)


def test_remote_sources_are_not_formatted_or_updated(catalog):
    assert len(list(checks.in_tree_sources(catalog))) == 3
    assert all(
        manifest["source"]["location"] == "in-tree"
        for _, manifest in checks.in_tree_sources(catalog)
    )


def test_lockfile_check_propagates_failure(monkeypatch, tmp_path):
    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(checks.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        checks.lockfiles(tmp_path)


def test_cargo_lockfile_check_resolves_dependencies(monkeypatch, tmp_path):
    (tmp_path / "Cargo.toml").touch()
    monkeypatch.setattr(
        checks, "in_tree_sources", lambda root: [(tmp_path, {"toolchains": {"rust": "1.96.1"}})]
    )
    calls = []
    monkeypatch.setattr(checks.subprocess, "run", lambda argv, **kw: calls.append(argv))
    checks.lockfiles(tmp_path)
    assert calls[-1] == ["cargo", "+1.96.1", "metadata", "--locked", "--format-version", "1"]


def test_hook_config_change_selects_all_plugins(catalog):
    assert len(select(discover(catalog), [".pre-commit-config.yaml"])) == 4


def test_attribution_hook_runs_even_when_commit_only_deletes_files():
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hook = next(
        hook for repo in config["repos"] for hook in repo["hooks"] if hook["id"] == "attributions"
    )
    assert hook["always_run"]
    assert not hook["pass_filenames"]


def test_ci_uses_same_required_hooks():
    workflow = yaml.safe_load((ROOT / ".github/workflows/plugins.yml").read_text())
    steps = workflow["jobs"]["licenses"]["steps"]
    command = next(step["run"] for step in steps if step.get("name") == "Run pre-commit checks")
    assert "pre-commit run --all-files --show-diff-on-failure" in command
    assert "licenses" in workflow["jobs"]["required"]["needs"]


def test_ci_downloads_tools_without_source_build_fallbacks():
    workflow = yaml.safe_load((ROOT / ".github/workflows/plugins.yml").read_text())
    jobs = workflow["jobs"]
    for job_name in ["plan", "build", "licenses", "release"]:
        steps = jobs[job_name]["steps"]
        assert any(step.get("uses", "").startswith("astral-sh/setup-uv@") for step in steps)
        assert all("pip install uv" not in step.get("run", "") for step in steps)

    for job_name in ["build", "licenses"]:
        steps = jobs[job_name]["steps"]
        install = next(
            step for step in steps if step.get("uses") == "./.github/actions/setup-rust-tools"
        )
        assert install["with"]["toolchain"]

    action_path = ROOT / ".github/actions/setup-rust-tools/action.yml"
    action_text = action_path.read_text()
    assert "cargo install" not in action_text
    assert "cargo-about/releases/download" in action_text
    assert "expected_sha256" in action_text


@pytest.mark.parametrize("hook_id", ["ruff-check", "ruff-format", "rust-fmt", "actionlint"])
def test_configurable_checks_run_for_settings_changes_and_deletions(hook_id):
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hook = next(hook for repo in config["repos"] for hook in repo["hooks"] if hook["id"] == hook_id)
    assert hook["always_run"]
    assert not hook["pass_filenames"]


def test_python_formatter_checks_unchanged_tracked_files_and_leaves_untracked_work(repo):
    root, git, commit = repo
    # Simulate changing settings after the source file was already committed.
    source = root / "source with spaces.py"
    commit(source.name, "answer=  42\n")
    (root / "pyproject.toml").write_text("[tool.ruff]\nline-length = 90\n")
    git("add", "pyproject.toml")
    scratch = root / "scratch.py"
    scratch.write_text("answer=  42\n")
    checks.python_check(root, "format")
    assert source.read_text() == "answer = 42\n"
    assert scratch.read_text() == "answer=  42\n"


def test_python_lint_handles_deleted_files_and_stub_files(repo):
    root, git, commit = repo
    commit("deleted.py", "answer = 42\n")
    (root / "deleted.py").unlink()
    stub = root / "types.pyi"
    stub.write_text("import unused_module\nanswer: int\n")
    git("add", stub.name)
    checks.python_check(root, "check")
    assert stub.read_text() == "answer: int\n"
