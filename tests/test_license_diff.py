# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from scripts.licensing import generate, license_diff


def row(name, version, license_name="MIT"):
    return {"package": name, "version": version, "license": license_name}


def test_diff_distinguishes_add_remove_version_and_license_changes():
    before = {"python": [row("removed", "1"), row("updated", "1"), row("relicensed", "1")]}
    after = {
        "python": [row("added", "1"), row("updated", "2"), row("relicensed", "1", "Apache-2.0")]
    }
    diff = license_diff.compare_inventories(before, after)
    assert diff["python"]["added"] == [row("added", "1")]
    assert diff["python"]["removed"] == [row("removed", "1")]
    assert {entry["package"] for entry in diff["python"]["updated_changed"]} == {
        "updated",
        "relicensed",
    }
    report = license_diff.render_markdown(diff)
    assert report.startswith("# Lockfile License Changes")
    assert all(
        heading in report
        for heading in ["### Added", "### Removed", "### Updated/Changed", "Before:", "After:"]
    )


def test_diff_preserves_multiple_versions_and_first_release():
    inventory = {"rust": [row("shared", "1"), row("shared", "2")]}
    assert len(license_diff.compare_inventories({}, inventory)["rust"]["added"]) == 2
    assert license_diff.compare_inventories(inventory, inventory)["rust"]["updated_changed"] == []


@pytest.mark.parametrize(
    "language,heading", [("Python", "## shared (1)"), ("Rust", "## shared - 1")]
)
def test_aggregation_deduplicates_but_preserves_distinct_license_text(language, heading):
    first = f"# Header\n\n{heading}\n\nFirst license text\n"
    second = f"# Header\n\n{heading}\n\nSecond license text\n"
    result = generate.aggregate([first, first, second], language)
    assert result.count("First license text") == 1
    assert result.count("Second license text") == 1
    assert result.count(heading) == 2


def test_committed_aggregates_cover_every_plugin_and_tooling_inventory():
    root = Path(__file__).resolve().parents[1]
    for language in ["Python", "Rust"]:
        sources = list(root.glob(f"plugins/*/ATTRIBUTIONS-{language}.md"))
        tooling = root / "scripts/licensing" / f"ATTRIBUTIONS-{language}.md"
        if tooling.exists():
            sources.append(tooling)
        expected = generate.aggregate([path.read_text() for path in sources], language)
        assert (root / f"ATTRIBUTIONS-{language}.md").read_text() == expected


def test_remote_inventory_uses_locked_sha_and_rejects_modified_checkout(tmp_path, monkeypatch):
    (tmp_path / "plugins").mkdir()
    sha = "a" * 40
    monkeypatch.setattr(
        generate,
        "discover",
        lambda root: {
            "remote-plugin": {
                "source": {
                    "location": "remote",
                    "repository": "https://example.com/repo",
                    "sha": sha,
                    "ref": "main",
                    "path": "crate",
                },
                "toolchains": {"rust": "1.96.1"},
            }
        },
    )
    calls = []

    def checkout(repository, commit, destination):
        calls.append((repository, commit))
        destination.mkdir(parents=True)
        (destination / "crate").mkdir()
        (destination / "crate/Cargo.toml").write_text("[package]\nname = 'example'\n")
        (destination / "uv.lock").write_text("# Unrelated upstream Python application\n")

    monkeypatch.setattr(generate, "checkout", checkout)
    monkeypatch.setattr(
        generate.subprocess, "check_output", lambda args, **kw: sha if args[-1] == "HEAD" else ""
    )
    projects = list(generate.projects(tmp_path))
    assert calls == [("https://example.com/repo", sha)]
    assert projects[0][0].name == sha
    assert projects[0][3] == ["Rust"]
    monkeypatch.setattr(
        generate.subprocess,
        "check_output",
        lambda args, **kw: sha if args[-1] == "HEAD" else " M Cargo.lock",
    )
    with pytest.raises(ValueError, match="differs from locked source"):
        list(generate.projects(tmp_path))


def test_json_diff_cli_matches_markdown_inventory(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base.json"
    current = tmp_path / "current.json"
    base.write_text(json.dumps({"python": []}))
    current.write_text(json.dumps({"python": [row("added", "1")]}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "license-diff",
            "--base-json",
            str(base),
            "--current-json",
            str(current),
            "--format",
            "json",
        ],
    )
    assert license_diff.main() == 0
    assert json.loads(capsys.readouterr().out)["python"]["added"] == [row("added", "1")]
