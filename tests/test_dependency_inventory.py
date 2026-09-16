# SPDX-License-Identifier: Apache-2.0
"""Dependency inventory uses manifests for scope and locks for exact versions."""

import csv

from scripts.licensing.dependency_inventory import (
    DEVELOPMENT,
    DIRECT_OPTIONAL,
    DIRECT_REQUIRED,
    INDIRECT_OPTIONAL,
    INDIRECT_REQUIRED,
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
dependencies = [{ name = "required-transitive" }]

[[package]]
name = "required-transitive"
version = "2.1.0"
source = { registry = "https://example.invalid" }

[[package]]
name = "optional-package"
version = "3.0.0"
source = { registry = "https://example.invalid" }
dependencies = [{ name = "optional-transitive" }]

[[package]]
name = "optional-transitive"
version = "3.1.0"
source = { registry = "https://example.invalid" }

[[package]]
name = "development-package"
version = "4.0.0"
source = { registry = "https://example.invalid" }

[[package]]
name = "test-package"
version = "5.0.0"
source = { registry = "https://example.invalid" }
dependencies = [{ name = "test-transitive" }]

[[package]]
name = "test-transitive"
version = "5.1.0"
source = { registry = "https://example.invalid" }
""".lstrip()
    )

    assert python_dependencies(tmp_path, tmp_path) == [
        Dependency("required-package", "2.0.0", "Python", DIRECT_REQUIRED),
        Dependency("optional-package", "3.0.0", "Python", DIRECT_OPTIONAL),
        Dependency("required-transitive", "2.1.0", "Python", INDIRECT_REQUIRED),
        Dependency("optional-transitive", "3.1.0", "Python", INDIRECT_OPTIONAL),
        Dependency("test-package", "5.0.0", "Python", TEST),
        Dependency("test-transitive", "5.1.0", "Python", TEST),
        Dependency("development-package", "4.0.0", "Python", DEVELOPMENT),
    ]
    assert {
        row.plugins for row in python_dependencies(tmp_path, tmp_path, plugin="python-plugin")
    } == {("python-plugin",)}


def test_python_workspace_inventory_starts_at_the_selected_package(tmp_path):
    package = tmp_path / "plugin"
    package.mkdir()
    (package / "pyproject.toml").write_text('[project]\nname = "plugin"\n')
    (tmp_path / "uv.lock").write_text(
        """
[[package]]
name = "plugin"
version = "1.0.0"
source = { editable = "plugin" }
dependencies = [{ name = "shared" }]

[[package]]
name = "shared"
version = "1.0.0"
source = { editable = "shared" }
dependencies = [{ name = "reachable" }]

[[package]]
name = "unrelated-workspace-package"
version = "1.0.0"
source = { editable = "unrelated" }
dependencies = [{ name = "unrelated" }]

[[package]]
name = "reachable"
version = "2.0.0"
source = { registry = "https://example.invalid" }

[[package]]
name = "unrelated"
version = "3.0.0"
source = { registry = "https://example.invalid" }
""".lstrip()
    )

    assert python_dependencies(tmp_path, package, include_workspace=True) == [
        Dependency("shared", "1.0.0", "Python", DIRECT_REQUIRED),
        Dependency("reachable", "2.0.0", "Python", INDIRECT_REQUIRED),
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
    required_transitive_id = "registry+example#required-transitive@2.1.0"
    optional_transitive_id = "registry+example#optional-transitive@3.1.0"
    test_transitive_id = "registry+example#test-transitive@5.1.0"
    packages.extend(
        [
            {
                "id": required_transitive_id,
                "name": "required-transitive",
                "version": "2.1.0",
                "manifest_path": "/registry/required-transitive/Cargo.toml",
                "dependencies": [],
            },
            {
                "id": optional_transitive_id,
                "name": "optional-transitive",
                "version": "3.1.0",
                "manifest_path": "/registry/optional-transitive/Cargo.toml",
                "dependencies": [],
            },
            {
                "id": test_transitive_id,
                "name": "test-transitive",
                "version": "5.1.0",
                "manifest_path": "/registry/test-transitive/Cargo.toml",
                "dependencies": [],
            },
        ]
    )
    metadata = {
        "packages": packages,
        "workspace_members": [root_id],
        "resolve": {
            "nodes": [
                {"id": root_id, "deps": resolved},
                {
                    "id": "registry+example#required@2.0.0",
                    "deps": [
                        {
                            "name": "required_transitive",
                            "pkg": required_transitive_id,
                            "dep_kinds": [{"kind": None, "target": None}],
                        }
                    ],
                },
                {
                    "id": "registry+example#optional@3.0.0",
                    "deps": [
                        {
                            "name": "optional_transitive",
                            "pkg": optional_transitive_id,
                            "dep_kinds": [{"kind": None, "target": None}],
                        }
                    ],
                },
                {
                    "id": "registry+example#tester@5.0.0",
                    "deps": [
                        {
                            "name": "test_transitive",
                            "pkg": test_transitive_id,
                            "dep_kinds": [{"kind": None, "target": None}],
                        }
                    ],
                },
            ]
        },
    }

    assert rust_dependencies(tmp_path, metadata) == [
        Dependency("required", "2.0.0", "Rust", DIRECT_REQUIRED),
        Dependency("optional", "3.0.0", "Rust", DIRECT_OPTIONAL),
        Dependency("required-transitive", "2.1.0", "Rust", INDIRECT_REQUIRED),
        Dependency("optional-transitive", "3.1.0", "Rust", INDIRECT_OPTIONAL),
        Dependency("test-transitive", "5.1.0", "Rust", TEST),
        Dependency("tester", "5.0.0", "Rust", TEST),
        Dependency("builder", "4.0.0", "Rust", DEVELOPMENT),
    ]
    assert {row.plugins for row in rust_dependencies(tmp_path, metadata, plugin="rust-plugin")} == {
        ("rust-plugin",)
    }


def test_rust_workspace_inventory_starts_at_the_selected_package(tmp_path):
    manifest = tmp_path / "Cargo.toml"
    manifest.touch()
    root_id = "path+file:///example#plugin@1.0.0"
    shared_id = "path+file:///example/shared#shared@1.0.0"
    unrelated_id = "path+file:///example/unrelated#unrelated@1.0.0"
    reachable_id = "registry+example#reachable@2.0.0"
    unrelated_dependency_id = "registry+example#unrelated@3.0.0"
    metadata = {
        "packages": [
            {
                "id": root_id,
                "name": "plugin",
                "version": "1.0.0",
                "manifest_path": str(manifest),
                "dependencies": [{"name": "shared", "kind": None, "target": None}],
            },
            {
                "id": shared_id,
                "name": "shared",
                "version": "1.0.0",
                "manifest_path": str(tmp_path / "shared/Cargo.toml"),
                "dependencies": [],
            },
            {
                "id": unrelated_id,
                "name": "unrelated-workspace-package",
                "version": "1.0.0",
                "manifest_path": str(tmp_path / "unrelated/Cargo.toml"),
                "dependencies": [],
            },
            {
                "id": reachable_id,
                "name": "reachable",
                "version": "2.0.0",
                "manifest_path": "/registry/reachable/Cargo.toml",
                "dependencies": [],
            },
            {
                "id": unrelated_dependency_id,
                "name": "unrelated",
                "version": "3.0.0",
                "manifest_path": "/registry/unrelated/Cargo.toml",
                "dependencies": [],
            },
        ],
        "workspace_members": [root_id, shared_id, unrelated_id],
        "resolve": {
            "nodes": [
                {
                    "id": root_id,
                    "deps": [
                        {
                            "name": "shared",
                            "pkg": shared_id,
                            "dep_kinds": [{"kind": None, "target": None}],
                        }
                    ],
                },
                {
                    "id": shared_id,
                    "deps": [
                        {
                            "name": "reachable",
                            "pkg": reachable_id,
                            "dep_kinds": [{"kind": None, "target": None}],
                        }
                    ],
                },
                {
                    "id": unrelated_id,
                    "deps": [
                        {
                            "name": "unrelated",
                            "pkg": unrelated_dependency_id,
                            "dep_kinds": [{"kind": None, "target": None}],
                        }
                    ],
                },
            ]
        },
    }

    assert rust_dependencies(tmp_path, metadata, include_workspace=True) == [
        Dependency("shared", "1.0.0", "Rust", DIRECT_REQUIRED),
        Dependency("reachable", "2.0.0", "Rust", INDIRECT_REQUIRED),
    ]


def test_csv_has_requested_columns_and_order(tmp_path):
    output = tmp_path / "dependencies.csv"
    write_csv(
        [
            Dependency("development", "1.0", "Rust", DEVELOPMENT),
            Dependency("test", "1.0", "Rust", TEST),
            Dependency("indirect-optional", "1.0", "Rust", INDIRECT_OPTIONAL),
            Dependency("indirect-required", "1.0", "Rust", INDIRECT_REQUIRED),
            Dependency("direct-optional", "1.0", "Rust", DIRECT_OPTIONAL),
            Dependency("direct-required", "1.0", "Rust", DIRECT_REQUIRED),
            Dependency("python-package", "1.0", "Python", TEST),
        ],
        output,
    )
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["language"], row["dependency_type"]) for row in rows] == [
        ("Python", TEST),
        ("Rust", DIRECT_REQUIRED),
        ("Rust", DIRECT_OPTIONAL),
        ("Rust", INDIRECT_REQUIRED),
        ("Rust", INDIRECT_OPTIONAL),
        ("Rust", TEST),
        ("Rust", DEVELOPMENT),
    ]
    assert all(row["plugins"] == "" for row in rows)


def test_broadest_use_wins_and_plugins_are_merged(tmp_path):
    output = tmp_path / "dependencies.csv"
    write_csv(
        [
            Dependency("example", "1.0", "Python", TEST, ("plugin-b",)),
            Dependency("example", "1.0", "Python", DIRECT_REQUIRED, ("plugin-a",)),
        ],
        output,
    )
    with output.open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {
                "name": "example",
                "version": "1.0",
                "language": "Python",
                "dependency_type": DIRECT_REQUIRED,
                "plugins": "plugin-a; plugin-b",
            }
        ]
