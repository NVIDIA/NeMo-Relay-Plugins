# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import json

import pytest
import tomli_w

from scripts.catalog import PLATFORMS, affects, changed_paths, discover, select, tag_plugin
from scripts.plugins import expand, make_plan

NAME = "example-python-grpc-worker-plugin"


def rewrite(root, mutate):
    manifests = discover(root)
    manifest = deepcopy(manifests[NAME])
    mutate(manifest)
    (root / "plugins" / NAME / "release.toml").write_text(tomli_w.dumps(manifest))


def test_catalog_has_19_targets(catalog):
    manifests = discover(catalog)
    assert sum(len(m["platforms"]) for m in manifests.values()) == 19
    assert "windows-arm64" not in manifests[NAME]["platforms"]


def test_omitted_platforms_defaults_to_all(catalog):
    rewrite(catalog, lambda m: m.pop("platforms"))
    assert discover(catalog)[NAME]["platforms"] == list(PLATFORMS)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(platforms=[]),
        lambda m: m.update(platforms=["linux-fake"]),
        lambda m: m.update(platforms=["linux-arm64", "linux-arm64"]),
        lambda m: m.update(name="other"),
        lambda m: m.update(version="v1.0.0"),
        lambda m: m.update(version="1.0.0-01"),
        lambda m: m.update(type="unknown"),
        lambda m: m["source"].update(path="../.."),
        lambda m: m["source"].update(
            location="remote", repository="https://github.com/a/b", ref="main"
        ),
        lambda m: m["relay"].update(tag="0.8.4", sha="a" * 40),
        lambda m: m["artifacts"].update(bundle="/tmp/escape"),
        lambda m: m.update(unknown="typo"),
    ],
)
def test_invalid_manifests(catalog, mutate):
    rewrite(catalog, mutate)
    with pytest.raises(ValueError):
        discover(catalog)


def test_remote_selector_has_committed_sha(catalog):
    m = discover(catalog)["switchyard-plugin"]
    assert m["source"]["ref"] == "main"
    assert len(m["source"]["sha"]) == 40


def test_hyphenated_tags_and_prereleases(catalog):
    rewrite(catalog, lambda m: m.update(version="1.2.3-rc.1+build.4"))
    manifests = discover(catalog)
    assert tag_plugin(NAME + "-1.2.3-rc.1+build.4", manifests)["name"] == NAME
    for tag in ["v1.0.0", "unknown-1.0.0", NAME + "-1.2.3", NAME + "-v1.2.3-rc.1+build.4"]:
        with pytest.raises(ValueError):
            tag_plugin(tag, manifests)


def test_shared_and_local_changes(catalog):
    manifests = discover(catalog)
    assert select(manifests, ["README.md"]) == []
    assert len(select(manifests, ["scripts/new.py"])) == 4
    assert len(select(manifests, ["uv.lock"])) == 4
    assert len(select(manifests, None)) == 4
    assert [m["name"] for m in select(manifests, [f"plugins/{NAME}/README.md"])] == [NAME]
    assert not affects(NAME, [f"plugins/{NAME}-other/code.py"])


def test_git_diff_handles_rename_delete_and_missing_history(repo):
    root, git, commit = repo
    old = f"plugins/{NAME}/source.py"
    first = commit(old)
    git("mv", old, "plugins/example-rust-native-plugin/moved.py")
    git("commit", "-qm", "move")
    paths = changed_paths(first, "HEAD", root=root)
    assert set(paths) == {old, "plugins/example-rust-native-plugin/moved.py"}
    previous = git("rev-parse", "HEAD")
    git("rm", "plugins/example-rust-native-plugin/moved.py")
    git("commit", "-qm", "delete")
    assert changed_paths(previous, "HEAD", root=root) == [
        "plugins/example-rust-native-plugin/moved.py"
    ]
    assert changed_paths("0" * 40, "HEAD", root=root) is None
    assert changed_paths("not-a-commit", "HEAD", root=root) is None


def test_pr_diff_uses_merge_base(repo):
    root, git, commit = repo
    git("checkout", "-qb", "feature")
    head = commit(f"plugins/{NAME}/new.py")
    git("checkout", "main")
    base = commit("scripts/change.py")
    paths = changed_paths(base, head, pull_request=True, root=root)
    assert paths == [f"plugins/{NAME}/new.py"]


def test_plan_resolves_host_once_and_isolates_tag(catalog, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "scripts.plugins.resolve_host",
        lambda selector, github: calls.append(selector) or {"sha": "a" * 40},
    )
    manifests = discover(catalog)
    plan = make_plan(manifests, {}, "push", "refs/heads/main", "a" * 40, None, catalog)
    assert len(plan["matrix"]["include"]) == 19
    assert len(calls) == 1
    plan = make_plan(manifests, {}, "push", f"refs/tags/{NAME}-0.1.0", "a" * 40, None, catalog)
    assert len(plan["matrix"]["include"]) == 4
    assert set(plan["hosts"]) == {NAME}
    with pytest.raises(ValueError):
        make_plan(manifests, {}, "push", "refs/tags/nope", "a" * 40, None, catalog)


def test_commands_are_argument_arrays():
    assert expand(
        ["${PYTHON}", "${PLUGIN_DIR}/tasks.py", "$(echo unsafe)"],
        {"PYTHON": "python", "PLUGIN_DIR": "/has spaces"},
    ) == ["python", "/has spaces/tasks.py", "$(echo unsafe)"]
    with pytest.raises(ValueError):
        expand(["${MISSING}"], {})


def test_schema_is_valid(catalog):
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(
        json.loads((catalog / "schemas/release.schema.json").read_text())
    )


def test_initial_lockfiles_are_not_ignored():
    import subprocess
    from scripts.catalog import ROOT

    for folder in ["example-rust-native-plugin", "example-rust-grpc-worker-plugin"]:
        path = ROOT / "plugins" / folder / "Cargo.lock"
        assert path.is_file()
        result = subprocess.run(["git", "check-ignore", str(path)], cwd=ROOT, capture_output=True)
        assert result.returncode == 1


def test_pipeline_rejects_non_native_or_undeclared_targets(catalog, monkeypatch):
    from scripts.plugins import run_plugin

    monkeypatch.setattr("scripts.plugins.local_platform", lambda: "macos-arm64")
    with pytest.raises(ValueError, match="natively"):
        run_plugin(discover(catalog)[NAME], "windows-arm64", {}, "a" * 40)
