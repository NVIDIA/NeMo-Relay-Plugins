# SPDX-License-Identifier: Apache-2.0
"""CLI shared by local development and GitHub Actions."""

from __future__ import annotations

import argparse
import json
import os
import platform as machine
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.bundles import archive_name, create_archive, extract_archive, sha256, verify_bundle
from scripts.catalog import (
    ROOT,
    PLATFORMS,
    changed_paths,
    discover,
    git,
    inside,
    select,
    tag_plugin,
)
from scripts.github import GitHub
from scripts.host import checkout, install_host, resolve_host
from scripts.releases import publish_draft, release_notes


def local_platform() -> str:
    system = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}[machine.system()]
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x86_64", "amd64": "x86_64"}[
        machine.machine().lower()
    ]
    return f"{system}-{arch}"


def make_plan(
    manifests: dict,
    event: dict,
    event_name: str,
    ref: str,
    head: str,
    github: GitHub,
    root: Path = ROOT,
) -> dict:
    if ref.startswith("refs/tags/"):
        selected = [tag_plugin(ref.removeprefix("refs/tags/"), manifests)]
    elif event_name == "pull_request":
        selected = select(
            manifests,
            changed_paths(
                event["pull_request"]["base"]["sha"],
                event["pull_request"]["head"]["sha"],
                pull_request=True,
                root=root,
            ),
        )
    else:
        selected = select(manifests, changed_paths(event.get("before"), head, root=root))
    hosts = {}
    resolutions = {}
    matrix = []
    for manifest in selected:
        selector = json.dumps(manifest["relay"], sort_keys=True)
        if selector not in resolutions:
            resolutions[selector] = resolve_host(manifest["relay"], github)
        hosts[manifest["name"]] = resolutions[selector]
        for platform in manifest["platforms"]:
            matrix.append(
                {
                    "name": manifest["name"],
                    "platform": platform,
                    **PLATFORMS[platform],
                    "python": manifest["toolchains"]["python"],
                    "rust": manifest["toolchains"].get("rust", ""),
                }
            )
    return {"commit": head, "ref": ref, "matrix": {"include": matrix}, "hosts": hosts}


def expand(argv: list[str], env: dict[str, str]) -> list[str]:
    def replace(match):
        if match[1] not in env:
            raise ValueError(f"undefined command variable {match[1]}")
        return env[match[1]]

    return [re.sub(r"\$\{([A-Z_]+)\}", replace, word) for word in argv]


def run_plugin(
    manifest: dict, platform: str, resolution: dict, commit: str, *, relay: Path | None = None
):
    name = manifest["name"]
    if platform not in manifest["platforms"] or platform != local_platform():
        raise ValueError("build/test must run natively on a declared platform")
    destination = ROOT / "dist" / name / platform
    if destination.exists():
        shutil.rmtree(destination)
    work = ROOT / ".cache/runs" / name / platform
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    plugin = ROOT / "plugins" / name
    package_identity = None
    source_manifest = None
    if manifest["source"]["location"] == "remote":
        source_root = work / "source"
        checkout(manifest["source"]["repository"] + ".git", manifest["source"]["sha"], source_root)
        source = inside(source_root, manifest["source"]["path"])
        source_commit = manifest["source"]["sha"]
    elif manifest["source"]["location"] in {"wheel", "crate"}:
        from scripts.package_sources import prepare

        source_root, source_manifest, package_identity = prepare(
            plugin, manifest, platform, work, ROOT / ".cache/packages"
        )
        source = source_root
        source_commit = None
    else:
        source_root = source = inside(plugin, manifest["source"]["path"])
        source_commit = commit
    binary = relay or install_host(resolution, platform, work / "host")
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    # Prefer setup-python's native interpreter, including Windows ARM64 where a
    # managed Python download may not exist even though a system interpreter does.
    find_python = ["uv", "python", "find", manifest["toolchains"]["python"]]
    try:
        plugin_python = subprocess.check_output(find_python, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        subprocess.run(["uv", "python", "install", manifest["toolchains"]["python"]], check=True)
        plugin_python = subprocess.check_output(find_python, text=True)
    env["PLUGIN_PYTHON"] = plugin_python.strip()
    if package_identity:
        env["SOURCE_MANIFEST"] = str(source_manifest)
        env["PACKAGE_SOURCE_LOCK"] = str(work / "package-source.json")
        if manifest["source"]["location"] == "wheel":
            from scripts.package_sources import install_wheels

            env["WHEELHOUSE"] = str(work / "wheels")
            env["PACKAGE_PYTHON"] = str(install_wheels(work, env["PLUGIN_PYTHON"]))
    env.update(
        {
            "PYTHON": sys.executable,
            "REPO_DIR": str(ROOT),
            "PLUGIN_DIR": str(plugin),
            "SOURCE_DIR": str(source),
            "SOURCE_ROOT": str(source_root),
            "OUTPUT_DIR": str(work),
            "PLUGIN_PLATFORM": platform,
            "RELAY_BIN": str(binary.resolve()),
            "CARGO_TARGET_DIR": str(ROOT / ".cache/targets" / name / platform),
        }
    )
    if manifest["toolchains"].get("rust"):
        env["RUSTUP_TOOLCHAIN"] = manifest["toolchains"]["rust"]

    def stage(which):
        command = manifest["commands"][which]
        subprocess.run(
            expand(command["argv"], env),
            cwd=source_root if command["cwd"] == "source" else plugin,
            env=env,
            check=True,
        )

    for which in ["build", "test", "package"]:
        stage(which)
    bundle = inside(work, manifest["artifacts"]["bundle"])
    verify_bundle(bundle, manifest["artifacts"]["manifest"], manifest["type"])
    staging = work / "assets"
    archive = staging / archive_name(manifest, platform)
    create_archive(bundle, archive, name)
    with tempfile.TemporaryDirectory(prefix="relay-bundle-") as temporary:
        extracted = extract_archive(archive, Path(temporary))
        verify_bundle(extracted, manifest["artifacts"]["manifest"], manifest["type"])
        env["BUNDLE_DIR"] = str(extracted)
        stage("smoke")
    digest = sha256(archive)
    metadata = {
        "name": name,
        "version": manifest["version"],
        "platform": platform,
        "repository_commit": commit,
        "repository_dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
        "source_commit": source_commit,
        **({"package_source": package_identity} if package_identity else {}),
        "relay": {
            "sha": resolution["sha"],
            "tag": resolution["tag"],
            "binary_sha256": sha256(binary),
            "version_output": subprocess.check_output(
                [str(binary), "--version"], text=True
            ).strip(),
        },
        "local_host_override": relay is not None,
        "toolchains": manifest["toolchains"],
        "tool_versions": {
            "python": subprocess.check_output(
                [env["PLUGIN_PYTHON"], "--version"], text=True
            ).strip(),
            "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
            **(
                {
                    "rustc": subprocess.check_output(
                        ["rustc", "--version"], env=env, text=True
                    ).strip()
                }
                if manifest["toolchains"].get("rust")
                else {}
            ),
        },
        "sha256": digest,
        "verified": True,
    }
    (staging / (archive.name + ".json")).write_text(json.dumps(metadata, indent=2) + "\n")
    (staging / (archive.name + ".sha256")).write_text(f"{digest}  {archive.name}\n")
    shutil.copytree(staging, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("list")
    tag = sub.add_parser("validate-tag")
    tag.add_argument("tag")
    plan = sub.add_parser("plan")
    plan.add_argument("--event", type=Path, required=True)
    plan.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME", "push"))
    plan.add_argument("--ref", default=os.environ.get("GITHUB_REF", "refs/heads/main"))
    plan.add_argument("--head", default=os.environ.get("GITHUB_SHA", "HEAD"))
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--github-output", type=Path)
    run = sub.add_parser("run")
    run.add_argument("name")
    run.add_argument("--platform", default=None)
    run.add_argument("--plan", type=Path)
    run.add_argument(
        "--relay-bin",
        type=Path,
        help="Local testing only: an already installed host matching the resolved version",
    )
    run.add_argument(
        "--host-resolution", type=Path, help="Local testing: reuse a saved host resolution"
    )
    for command in ["notes", "release"]:
        release = sub.add_parser(command)
        release.add_argument("tag")
        release.add_argument(
            "--repository", default=os.environ.get("GITHUB_REPOSITORY", "NVIDIA/NeMo-Relay-Plugins")
        )
        release.add_argument("--assets", type=Path)
    args = parser.parse_args()
    manifests = discover()
    github = GitHub()
    if args.command in ("list", "validate"):
        print(
            json.dumps(
                manifests if args.command == "list" else {"valid": list(manifests)}, indent=2
            )
        )
    elif args.command == "validate-tag":
        print(json.dumps(tag_plugin(args.tag, manifests)))
    elif args.command == "plan":
        head = git("rev-parse", "--verify", "--end-of-options", f"{args.head}^{{commit}}")
        result = make_plan(
            manifests,
            json.loads(args.event.read_text(encoding="utf-8")),
            args.event_name,
            args.ref,
            head,
            github,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.github_output:
            with args.github_output.open("a") as stream:
                stream.write("matrix=" + json.dumps(result["matrix"]) + "\n")
                stream.write("has_plugins=" + str(bool(result["matrix"]["include"])).lower() + "\n")
                stream.write("is_tag=" + str(args.ref.startswith("refs/tags/")).lower() + "\n")
    elif args.command == "run":
        manifest = manifests[args.name]
        commit = git("rev-parse", "HEAD")
        if args.plan:
            if args.relay_bin or args.host_resolution:
                raise ValueError("CI plans cannot be combined with local host overrides")
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
            if plan["commit"] != commit:
                raise ValueError("checkout differs from the planned commit")
            if not any(
                row["name"] == args.name and row["platform"] == (args.platform or local_platform())
                for row in plan["matrix"]["include"]
            ):
                raise ValueError("plugin/platform is not in the planned matrix")
            resolution = plan["hosts"][args.name]
        else:
            resolution = (
                json.loads(args.host_resolution.read_text(encoding="utf-8"))
                if args.host_resolution
                else resolve_host(manifest["relay"], github)
            )
        print(
            run_plugin(
                manifest,
                args.platform or local_platform(),
                resolution,
                commit,
                relay=args.relay_bin,
            )
        )
    else:
        manifest = tag_plugin(args.tag, manifests)
        head = git("rev-parse", "--verify", "--end-of-options", f"refs/tags/{args.tag}^{{commit}}")
        if head != git("rev-parse", "HEAD"):
            raise ValueError("checkout must match the release tag")
        if args.command == "notes":
            print(
                release_notes(
                    manifest,
                    args.tag,
                    head,
                    args.repository,
                    github,
                    list(github.pages(f"repos/{args.repository}/releases")),
                )
            )
        else:
            if args.assets is None:
                raise ValueError("--assets is required")
            print(
                json.dumps(
                    publish_draft(manifest, args.tag, head, args.repository, args.assets, github)
                )
            )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
