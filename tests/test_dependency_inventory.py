# SPDX-License-Identifier: Apache-2.0
"""Dependency inventory uses manifests for scope and locks for exact versions."""

import csv

from scripts.licensing.dependency_inventory import (
    DEVELOPMENT,
    OPTIONAL,
    REQUIRED,
    TEST,
    Dependency,
    python_dependencies,
    rust_dependencies,
    write_csv,
)


def test_python_dependency_types_and_locked_versions(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "example"\n')
    (tmp_path / "uv.lock").write_text(
        """
[[package]]
name = "example"
version = "1.0.0"
source = { editable = "." }
dependencies = [{ name = "required-package" }]

[package.optional-dependencies]
feature = [{ name = "optional-package" }]

[package.dev-dependencies]
lint = [{ name = "development-package" }]
test = [{ name = "test-package" }, { name = "required-package" }]

[[package]]
name = "required-package"
version = "2.0.0"
source = { registry = "https://example.invalid" }

[[package]]
name = "optional-package"
version = "3.0.0"
source = { registry = "https://example.invalid" }

[[package]]
name = "development-package"
version = "4.0.0"
source = { registry = "https://example.invalid" }

[[package]]
name = "test-package"
version = "5.0.0"
source = { registry = "https://example.invalid" }
""".lstrip()
    )

    assert python_dependencies(tmp_path, tmp_path) == [
        Dependency("development-package", "4.0.0", "Python", DEVELOPMENT),
        Dependency("optional-package", "3.0.0", "Python", OPTIONAL),
        Dependency("required-package", "2.0.0", "Python", REQUIRED),
        Dependency("test-package", "5.0.0", "Python", TEST),
    ]


def test_rust_dependency_types_and_locked_versions(tmp_path):
    manifest = tmp_path / "Cargo.toml"
    manifest.touch()
    root_id = "path+file:///example#root@1.0.0"
    packages = [
        {
            "id": root_id,
            "name": "root",
            "version": "1.0.0",
            "manifest_path": str(manifest),
            "dependencies": [
                {"name": "required", "rename": None, "kind": None, "target": None},
                {
                    "name": "optional",
                    "rename": None,
                    "kind": None,
                    "target": None,
                    "optional": True,
                },
                {"name": "builder", "rename": None, "kind": "build", "target": None},
                {"name": "tester", "rename": None, "kind": "dev", "target": None},
                {"name": "required", "rename": None, "kind": "dev", "target": None},
            ],
        }
    ]
    resolved = []
    for name, version, kinds in [
        ("required", "2.0.0", [None, "dev"]),
        ("optional", "3.0.0", [None]),
        ("builder", "4.0.0", ["build"]),
        ("tester", "5.0.0", ["dev"]),
    ]:
        package_id = f"registry+example#{name}@{version}"
        packages.append(
            {
                "id": package_id,
                "name": name,
                "version": version,
                "manifest_path": f"/registry/{name}/Cargo.toml",
                "dependencies": [],
            }
        )
        resolved.append(
            {
                "name": name,
                "pkg": package_id,
                "dep_kinds": [{"kind": kind, "target": None} for kind in kinds],
            }
        )
    metadata = {
        "packages": packages,
        "workspace_members": [root_id],
        "resolve": {"nodes": [{"id": root_id, "deps": resolved}]},
    }

    assert rust_dependencies(tmp_path, metadata) == [
        Dependency("builder", "4.0.0", "Rust", DEVELOPMENT),
        Dependency("optional", "3.0.0", "Rust", OPTIONAL),
        Dependency("required", "2.0.0", "Rust", REQUIRED),
        Dependency("tester", "5.0.0", "Rust", TEST),
    ]


def test_csv_has_requested_columns_and_one_row_per_package(tmp_path):
    output = tmp_path / "dependencies.csv"
    write_csv(
        [
            Dependency("example", "1.0", "Python", TEST),
            Dependency("example", "1.0", "Python", REQUIRED),
        ],
        output,
    )
    with output.open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {
                "name": "example",
                "version": "1.0",
                "language": "Python",
                "dependency_type": REQUIRED,
            }
        ]
