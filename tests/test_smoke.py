# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

import pytest

from scripts.smoke import registered_manifest


def test_registered_manifest_matches_file_identity(tmp_path):
    manifest = tmp_path / "relay-plugin.toml"
    manifest.write_text("fixture")
    alias = tmp_path / "alias.toml"
    alias.hardlink_to(manifest)
    if os.name == "nt":
        # Match the extended-length spelling written by Rust canonicalize().
        alias = Path("\\\\?\\" + str(manifest.resolve()))
    unrelated = tmp_path / "other.toml"
    unrelated.write_text("fixture")
    record = {"manifest": str(alias)}
    document = {"plugins": {"dynamic": [{"manifest": str(unrelated)}, record]}}

    assert registered_manifest(document, manifest) is record


def test_registered_manifest_reports_missing_registration(tmp_path):
    manifest = tmp_path / "relay-plugin.toml"
    manifest.write_text("fixture")
    other = tmp_path / "other.toml"
    other.write_text("fixture")
    document = {"plugins": {"dynamic": [{"manifest": str(other)}]}}

    with pytest.raises(RuntimeError, match="Relay did not register") as error:
        registered_manifest(document, manifest)
    assert str(other) in str(error.value)
