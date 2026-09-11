# SPDX-License-Identifier: Apache-2.0
"""Attributions from exact published package bytes and locked crate dependencies."""

import tempfile
from pathlib import Path

from scripts.package_sources import prepare, download, read_lock
from scripts.licensing import attributions_lockfile_md as collector


def collect_packages(plugin, manifest, cache, *, platforms=None, inventory_only=False):
    from scripts.licensing.generate import aggregate, collect_project

    documents = {}
    inventory = {"python": [], "rust": []}
    texts = []
    seen = set()
    lock = read_lock(plugin, manifest)
    for platform in platforms or manifest["platforms"]:
        with tempfile.TemporaryDirectory(prefix="relay-package-license-") as directory:
            source, _, identity = prepare(plugin, manifest, platform, Path(directory), cache)
            if identity["location"] == "crate":
                checksum = identity["artifacts"][0]["sha256"]
                if checksum in seen:
                    continue
                seen.add(checksum)
                docs, rows = collect_project(
                    source,
                    manifest["toolchains"]["rust"],
                    ["Rust"],
                    include_workspace=True,
                    inventory_only=inventory_only,
                )
                texts.extend(docs.values())
                inventory["rust"].extend(rows["rust"])
            else:
                for item in lock["artifacts"]:
                    if platform not in item["platforms"] or item["sha256"] in seen:
                        continue
                    seen.add(item["sha256"])
                    archive = download(item, cache)
                    license_name, license_texts = collector._wheel_metadata_from_bytes(
                        archive.read_bytes(), package_name=item["name"]
                    )
                    if not license_texts:
                        raise ValueError(f"Published wheel lacks license text: {item['filename']}")
                    package = {
                        "name": item["name"],
                        "version": item["version"],
                        "license_name": collector._infer_license_name(license_name, license_texts),
                        "license_texts": license_texts,
                    }
                    inventory["python"].append(
                        collector._rendered_python_package_inventory(package)
                    )
                    if not inventory_only:
                        parts = [collector.PYTHON_HEADER]
                        collector._render_python_package(parts, **package)
                        texts.append("".join(parts))
    if texts:
        language = "Rust" if manifest["source"]["location"] == "crate" else "Python"
        documents[language] = aggregate(texts, language)
    return documents, inventory
