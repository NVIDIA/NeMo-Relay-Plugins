# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess

import pytest
import tomli_w

from scripts.catalog import ROOT, discover


@pytest.fixture
def catalog(tmp_path):
    (tmp_path / "schemas").mkdir()
    shutil.copy2(ROOT / "schemas/release.schema.json", tmp_path / "schemas/release.schema.json")
    for name, manifest in discover().items():
        folder = tmp_path / "plugins" / name
        folder.mkdir(parents=True)
        (folder / "release.toml").write_text(tomli_w.dumps(manifest))
    return tmp_path


@pytest.fixture
def repo(catalog):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=catalog, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    git("add", ".")
    git("commit", "-qm", "initial")

    def commit(path, content="changed"):
        file = catalog / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
        git("add", ".")
        git("commit", "-qm", "change " + path)
        return git("rev-parse", "HEAD")

    return catalog, git, commit
