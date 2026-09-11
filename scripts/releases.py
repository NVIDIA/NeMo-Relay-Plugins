# SPDX-License-Identifier: Apache-2.0
"""Plugin-scoped release notes and draft-only publishing."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from scripts.bundles import verify_assets
from scripts.catalog import ROOT, affects, git
from scripts.github import GitHub
from scripts.host import resolve_tag

MARKER = re.compile(r"<!-- relay-plugin-release: (.*?) -->")


def predecessor(
    manifest: dict, tag: str, head: str, releases: list[dict], root: Path = ROOT
) -> str | None:
    pattern = json.loads((root / "schemas/release.schema.json").read_text(encoding="utf-8"))[
        "properties"
    ]["version"]["pattern"]
    candidates = {}
    prefix = manifest["name"] + "-"
    for release in releases:
        previous = release["tag_name"]
        if release["draft"] or previous == tag or not previous.startswith(prefix):
            continue
        if not re.fullmatch(pattern, previous[len(prefix) :]):
            continue
        try:
            sha = git(
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"refs/tags/{previous}^{{commit}}",
                root=root,
            )
        except subprocess.CalledProcessError:
            raise ValueError(
                f"missing release tag {previous}; fetch full history and tags"
            ) from None
        if sha != head:
            candidates.setdefault(sha, previous)
    for sha in git("rev-list", "--topo-order", head, root=root).splitlines():
        if sha in candidates:
            return candidates[sha]
    return None


def release_notes(
    manifest: dict,
    tag: str,
    head: str,
    repository: str,
    github: GitHub,
    releases: list[dict],
    root: Path = ROOT,
) -> str:
    previous = predecessor(manifest, tag, head, releases, root)
    revision = f"refs/tags/{previous}..{head}" if previous else head
    commits = set(git("rev-list", revision, root=root).splitlines())
    prs = {}
    for commit in sorted(commits):
        for pr in github.pages(f"repos/{repository}/commits/{commit}/pulls"):
            if pr.get("merged_at") and pr.get("merge_commit_sha") in commits:
                prs[pr["number"]] = pr
    included = []
    for number, pr in sorted(prs.items()):
        files = list(github.pages(f"repos/{repository}/pulls/{number}/files"))
        detail = github.api(f"repos/{repository}/pulls/{number}")
        if len(files) != detail["changed_files"]:
            raise ValueError(f"incomplete changed-file list for PR #{number}")
        paths = [f["filename"] for f in files] + [
            f["previous_filename"] for f in files if "previous_filename" in f
        ]
        if affects(manifest["name"], paths):
            included.append(pr)
    lines = ["## What’s Changed", ""]
    for pr in included:
        title = " ".join(pr["title"].splitlines())
        lines.append(f"* {title} by @{pr['user']['login']} in {pr['html_url']}")
    if not included:
        lines.append("No merged pull requests affected this plugin in this release interval.")
    if included:
        lines += [
            "",
            "## Contributors",
            "",
            ", ".join("@" + name for name in sorted({p["user"]["login"] for p in included})),
        ]
    lines += ["", f"Changes since `{previous}`." if previous else "First release of this plugin."]
    if manifest["source"]["location"] == "remote":
        source = manifest["source"]
        lines += [
            "",
            "## Upstream documentation",
            "",
            f"Source: [{source['repository']}]({source['repository']}) at "
            f"[{source['sha'][:12]}]({source['repository']}/tree/{source['sha']}). "
            f"Refer to the [upstream release documentation]({manifest['metadata'].get('documentation', source['repository'])}) "
            "for plugin behavior and upstream changes.",
        ]
    if manifest["source"]["location"] in {"wheel", "crate"}:
        source = manifest["source"]
        url = (
            f"https://pypi.org/project/{source['package']}/{source['version']}/"
            if source["location"] == "wheel"
            else f"https://crates.io/crates/{source['package']}/{source['version']}"
        )
        documentation = manifest["metadata"].get("documentation", url)
        lines += [
            "",
            "## Upstream documentation",
            "",
            f"Package: [{source['package']} {source['version']}]({url}). "
            f"Refer to the [upstream release documentation]({documentation}) for plugin behavior and upstream changes. "
            "Exact package URLs and checksums are recorded in the attached build metadata.",
        ]
    return "\n".join(lines) + "\n"


def publish_draft(
    manifest: dict,
    tag: str,
    head: str,
    repository: str,
    assets: Path,
    github: GitHub,
    root: Path = ROOT,
) -> dict:
    if tag != f"{manifest['name']}-{manifest['version']}":
        raise ValueError("invalid plugin release tag")
    files = verify_assets(manifest, assets, head)
    actual = resolve_tag(tag, github, repository)
    if actual != head:
        raise ValueError("remote tag moved from the built commit")
    releases = list(github.pages(f"repos/{repository}/releases"))
    existing = next((r for r in releases if r["tag_name"] == tag), None)
    identity = {"name": manifest["name"], "version": manifest["version"], "commit": head}
    if existing:
        marker = MARKER.search(existing.get("body") or "")
        if not existing["draft"] or not marker or json.loads(marker.group(1)) != identity:
            raise ValueError("refusing to modify published or mismatched release")
    notes = release_notes(manifest, tag, head, repository, github, releases, root)
    body = notes + "\n<!-- relay-plugin-release: " + json.dumps(identity, sort_keys=True) + " -->\n"
    payload = {
        "tag_name": tag,
        "target_commitish": head,
        "name": tag,
        "body": body,
        "draft": True,
        "prerelease": "-" in manifest["version"].split("+")[0],
        "make_latest": "false",
    }
    if existing:
        release = github.api(f"repos/{repository}/releases/{existing['id']}")
        if not release["draft"]:
            raise ValueError("release was published while preparing assets")
        release = github.api(f"repos/{repository}/releases/{existing['id']}", "PATCH", payload)
    else:
        release = github.api(f"repos/{repository}/releases", "POST", payload)
    if not github.api(f"repos/{repository}/releases/{release['id']}")["draft"]:
        raise ValueError("release was published before asset upload")
    github.upload(repository, tag, files)
    uploaded = list(github.pages(f"repos/{repository}/releases/{release['id']}/assets"))
    expected = {f.name: f.stat().st_size for f in files}
    actual_files = {f["name"]: f["size"] for f in uploaded}
    if actual_files != expected:
        raise ValueError("draft release assets do not match the verified asset set")
    return release
