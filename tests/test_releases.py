# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts.catalog import discover
from scripts.releases import predecessor, publish_draft, release_notes
from test_bundles import assets_for

NAME = "example-python-grpc-worker-plugin"


class FakeGitHub:
    def __init__(self, commit, prs=None, releases=None):
        self.commit = commit
        self.prs = prs or []
        self.releases = releases or []
        self.mutations = []
        self.uploads = []

    def api(self, path, method="GET", body=None):
        if method in ("POST", "PATCH"):
            self.mutations.append((method, body))
            release = {"id": 42, **body}
            self.releases = [release]
            return release
        if "/commits/" in path or "/git/ref/tags/" in path:
            return {"sha": self.commit, "object": {"type": "commit", "sha": self.commit}}
        if "/pulls/" in path:
            number = int(path.rsplit("/", 1)[1])
            return {
                "changed_files": len(next(p for p in self.prs if p["number"] == number)["files"])
            }
        return self.releases[0]

    def pages(self, path):
        if path.endswith("/releases"):
            return iter(self.releases)
        if path.endswith("/assets"):
            return iter({"name": p.name, "size": p.stat().st_size} for p in self.uploads)
        if "/commits/" in path:
            commit = path.split("/commits/")[1].split("/")[0]
            return iter(p for p in self.prs if p["merge_commit_sha"] == commit)
        number = int(path.split("/pulls/")[1].split("/")[0])
        return iter(next(p for p in self.prs if p["number"] == number)["files"])

    def upload(self, repository, tag, files):
        self.uploads = files


def pr(number, sha, path):
    return {
        "number": number,
        "merge_commit_sha": sha,
        "merged_at": "2026-09-10",
        "title": f"Change {number}",
        "user": {"login": f"user{number}"},
        "html_url": f"https://github.com/a/b/pull/{number}",
        "files": [{"filename": path}],
    }


def test_notes_filter_plugin_shared_and_unrelated_changes(repo):
    root, git, commit = repo
    manifest = discover(root)[NAME]
    initial = git("rev-parse", "HEAD")
    git("tag", NAME + "-0.0.1")
    local = commit(f"plugins/{NAME}/code.py")
    other = commit("plugins/example-rust-native-plugin/code.rs")
    shared = commit("scripts/build.py")
    docs = commit("README.md")
    releases = [
        {"tag_name": NAME + "-0.0.1", "draft": False},
        {"tag_name": "example-rust-native-plugin-1.0.0", "draft": False},
    ]
    api = FakeGitHub(
        docs,
        [
            pr(1, initial, f"plugins/{NAME}/old.py"),
            pr(2, local, f"plugins/{NAME}/code.py"),
            pr(3, other, "plugins/example-rust-native-plugin/code.rs"),
            pr(4, shared, "scripts/build.py"),
            pr(5, docs, "README.md"),
        ],
    )
    notes = release_notes(manifest, NAME + "-0.1.0", docs, "a/b", api, releases, root)
    assert "Change 2" in notes and "Change 4" in notes
    assert all(f"Change {i}" not in notes for i in [1, 3, 5])
    assert "@user3" not in notes
    assert "/compare/" not in notes


def test_first_remote_release_links_upstream(repo):
    root, git, commit = repo
    manifest = discover(root)["switchyard-plugin"]
    head = commit("README.md")
    notes = release_notes(
        manifest, manifest["name"] + "-0.2.0", head, "a/b", FakeGitHub(head), [], root
    )
    assert "First release" in notes
    assert manifest["source"]["sha"] in notes
    assert "upstream release documentation" in notes


def test_predecessor_ignores_drafts_unrelated_and_nonancestors(repo):
    root, git, commit = repo
    manifest = discover(root)[NAME]
    git("tag", NAME + "-0.0.1")
    git("checkout", "-qb", "other")
    commit("unrelated.md")
    git("tag", NAME + "-0.0.9")
    git("checkout", "main")
    commit("README.md")
    git("tag", NAME + "-0.0.2")
    head = commit("RELEASE.md")
    releases = [
        {"tag_name": NAME + "-0.0.1", "draft": False},
        {"tag_name": NAME + "-0.0.9", "draft": False},
        {"tag_name": NAME + "-0.0.2", "draft": True},
    ]
    assert predecessor(manifest, NAME + "-0.1.0", head, releases, root) == NAME + "-0.0.1"


@pytest.mark.parametrize("strategy", ["merge", "squash", "rebase"])
def test_notes_for_merge_strategies(repo, strategy):
    root, git, commit = repo
    manifest = discover(root)[NAME]
    git("checkout", "-qb", "feature")
    commit(f"plugins/{NAME}/one.py")
    commit(f"plugins/{NAME}/two.py")
    git("checkout", "main")
    if strategy == "merge":
        git("merge", "--no-ff", "feature", "-m", "merge PR")
    elif strategy == "squash":
        git("merge", "--squash", "feature")
        git("commit", "-qm", "squash PR")
    else:
        commit("README.md")
        git("checkout", "feature")
        git("rebase", "main")
        git("checkout", "main")
        git("merge", "--ff-only", "feature")
    head = git("rev-parse", "HEAD")
    api = FakeGitHub(head, [pr(1, head, f"plugins/{NAME}/one.py")])
    notes = release_notes(manifest, NAME + "-0.1.0", head, "a/b", api, [], root)
    assert notes.count("Change 1") == 1


def test_create_and_retry_draft(repo, tmp_path):
    root, git, commit = repo
    manifest = discover(root)[NAME]
    head = git("rev-parse", "HEAD")
    assets = tmp_path / "assets"
    assets_for(manifest, assets, head)
    api = FakeGitHub(head)
    tag = NAME + "-0.1.0"
    publish_draft(manifest, tag, head, "a/b", assets, api, root)
    assert api.mutations[0][0] == "POST"
    assert api.mutations[0][1]["draft"] is True
    assert len(api.uploads) == 12
    publish_draft(manifest, tag, head, "a/b", assets, api, root)
    assert api.mutations[-1][0] == "PATCH"


@pytest.mark.parametrize(
    "existing",
    [
        {"draft": False, "body": ""},
        {"draft": True, "body": "unmanaged draft"},
        {"draft": True, "body": '<!-- relay-plugin-release: {"commit":"different"} -->'},
    ],
)
def test_published_and_mismatched_releases_untouched(repo, tmp_path, existing):
    root, git, commit = repo
    manifest = discover(root)[NAME]
    head = git("rev-parse", "HEAD")
    assets = tmp_path / "assets"
    assets_for(manifest, assets, head)
    api = FakeGitHub(head, releases=[{"id": 42, "tag_name": NAME + "-0.1.0", **existing}])
    with pytest.raises(ValueError, match="refusing"):
        publish_draft(manifest, NAME + "-0.1.0", head, "a/b", assets, api, root)
    assert api.mutations == [] and api.uploads == []


def test_missing_asset_and_moved_tag_cannot_create_draft(repo, tmp_path):
    root, git, commit = repo
    manifest = discover(root)[NAME]
    head = git("rev-parse", "HEAD")
    assets = tmp_path / "assets"
    assets_for(manifest, assets, head)
    api = FakeGitHub("a" * 40)
    with pytest.raises(ValueError, match="moved"):
        publish_draft(manifest, NAME + "-0.1.0", head, "a/b", assets, api, root)
    next(assets.glob("*.json")).unlink()
    with pytest.raises(FileNotFoundError):
        publish_draft(manifest, NAME + "-0.1.0", head, "a/b", assets, api, root)
    assert api.mutations == []


def test_incomplete_pr_file_list_fails(repo):
    root, git, commit = repo
    head = git("rev-parse", "HEAD")
    api = FakeGitHub(head, [pr(1, head, f"plugins/{NAME}/code.py")])
    api.api = lambda path: {"changed_files": 3001}
    with pytest.raises(ValueError, match="incomplete"):
        release_notes(discover(root)[NAME], NAME + "-0.1.0", head, "a/b", api, [], root)
