# SPDX-License-Identifier: Apache-2.0
"""Standard-library-only build backend copied into a Relay wheel bundle.

The adapter installs the unchanged, checksummed wheels as direct requirements.
It uses no build dependencies and resolves paths after the bundle is moved.
"""

import base64
import csv
import hashlib
import io
import json
import platform
import sys
import zipfile
from pathlib import Path


def get_requires_for_build_wheel(config_settings=None):
    return []


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    root = Path(__file__).resolve().parent
    identity = json.loads((root / "package-source.json").read_text(encoding="utf-8"))
    expected_python = tuple(map(int, identity["python"].split(".")))
    if sys.version_info[: len(expected_python)] != expected_python:
        raise ValueError("Python interpreter differs from the locked wheel environment")
    system = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(platform.system())
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x86_64", "amd64": "x86_64"}.get(
        platform.machine().lower()
    )
    if f"{system}-{arch}" != identity["platform"]:
        raise ValueError("platform differs from the locked wheel environment")
    requirements = []
    for item in identity["artifacts"]:
        path = root / "wheelhouse" / item["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("bundled wheel digest mismatch")
        requirements.append(f"Requires-Dist: {item['name']} @ {path.as_uri()}")
    distribution = "relay_wheel_environment"
    dist_info = f"{distribution}-1.0.dist-info"
    files = {
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.3\nName: relay-wheel-environment\nVersion: 1.0\n"
            + "\n".join(requirements)
            + "\n"
        ).encode(),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: relay-wheel-adapter\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([name, "sha256=" + digest, len(data)])
    writer.writerow([f"{dist_info}/RECORD", "", ""])
    files[f"{dist_info}/RECORD"] = record.getvalue().encode()
    filename = f"{distribution}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(Path(wheel_directory) / filename, "w", zipfile.ZIP_DEFLATED) as stream:
        for name, data in files.items():
            stream.writestr(name, data)
    return filename
