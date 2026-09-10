# SPDX-License-Identifier: Apache-2.0
"""Small GitHub API adapter; gh owns authentication, never shell interpolation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


class GitHub:
    def api(self, path: str, method: str = "GET", body: dict | None = None):
        args = ["gh", "api", path, "--method", method]
        if body is not None:
            args += ["--input", "-"]
        result = subprocess.run(
            args,
            input=json.dumps(body) if body is not None else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout) if result.stdout.strip() else None

    def pages(self, path: str):
        separator = "&" if "?" in path else "?"
        page = 1
        while True:
            items = self.api(f"{path}{separator}per_page=100&page={page}")
            yield from items
            if len(items) < 100:
                return
            page += 1

    def upload(self, repository: str, tag: str, files: list[Path]):
        subprocess.run(
            ["gh", "release", "upload", tag, "--repo", repository, "--clobber", *map(str, files)],
            check=True,
        )
