# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release metadata tests for the Guardrails worker package."""

import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

from nemoguardrails_nemo_relay import configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_manifest_matches_worker_and_current_contract() -> None:
    manifest = tomllib.loads((PROJECT_ROOT / "relay-plugin.toml").read_text(encoding="utf-8"))
    artifact = PROJECT_ROOT / manifest["source"]["artifact"]
    digest = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"

    assert manifest["plugin"] == {"id": configuration.PLUGIN_ID, "kind": "worker"}
    assert manifest["compat"] == {"relay": ">=0.9.0,<1.0", "worker_protocol": "grpc-v1"}
    assert manifest["integrity"]["sha256"] == digest


def test_schema_is_narrow_and_closed() -> None:
    schema = json.loads((PROJECT_ROOT / "config.schema.json").read_text(encoding="utf-8"))

    assert schema["additionalProperties"] is False
    assert schema["oneOf"] == [
        {"required": ["config_path"], "not": {"required": ["remote_checks"]}},
        {"required": ["remote_checks"], "not": {"required": ["config_path"]}},
    ]
    assert set(schema["properties"]) == {
        "version",
        "config_path",
        "remote_checks",
        "check_timeout_ms",
        "secret_env",
        "payload_policy",
        "mutation_policy",
        "semantic_tool_policy",
        "evaluator_framework",
        "action_safety",
        "allow_ignored_rail_families",
        "allow_known_fail_open_rails",
    }
    assert schema["properties"]["check_timeout_ms"]["minimum"] == configuration.MIN_CHECK_TIMEOUT_MS
    assert schema["properties"]["check_timeout_ms"]["maximum"] == configuration.MAX_CHECK_TIMEOUT_MS
    remote_checks = schema["properties"]["remote_checks"]
    assert "allow_remote_content_logging_and_retention" in remote_checks["required"]
    assert remote_checks["properties"]["allow_remote_content_logging_and_retention"]["const"] is True
    assert schema["properties"]["payload_policy"]["additionalProperties"] is False
    assert schema["properties"]["payload_policy"]["properties"]["multimodal"]["enum"] == [
        "strict",
        "text_only",
    ]
    assert schema["properties"]["mutation_policy"]["additionalProperties"] is False
    assert schema["properties"]["mutation_policy"]["properties"]["input"]["enum"] == [
        "reject",
        "apply",
    ]
    assert schema["properties"]["mutation_policy"]["properties"]["output"]["default"] == "reject"
    semantic_tools = schema["properties"]["semantic_tool_policy"]
    assert semantic_tools["additionalProperties"] is False
    assert semantic_tools["properties"]["result_source"]["enum"] == [
        "off",
        "execution",
        "history",
        "both",
    ]
    assert schema["properties"]["evaluator_framework"]["enum"] == ["default", "langchain"]
    assert schema["properties"]["action_safety"]["properties"]["synchronous"]["enum"] == [
        "reject",
        "allow_unsafe",
    ]
    secret_env = schema["properties"]["secret_env"]
    assert secret_env["maxProperties"] == configuration.MAX_SECRET_ENV_ENTRIES
    assert secret_env["propertyNames"]["pattern"] == configuration._SAFE_ENV_NAME.pattern
    assert secret_env["additionalProperties"] == {
        "type": "string",
        "minLength": 1,
        "maxLength": configuration.MAX_SECRET_ENV_VALUE_CHARACTERS,
        "writeOnly": True,
    }


def test_project_dependency_bounds_match_the_design() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    release = tomllib.loads((PROJECT_ROOT / "release.toml").read_text(encoding="utf-8"))
    manifest = tomllib.loads((PROJECT_ROOT / "relay-plugin.toml").read_text(encoding="utf-8"))
    assert project["requires-python"] == ">=3.11,<3.14"
    assert release["version"] == project["version"]
    assert release["relay"]["sha"] == "8122ee1f1765b1e3ef3ef706a8e8233a1630c5ad"
    assert manifest["compat"]["relay"] == ">=0.9.0,<1.0"
    assert "nemo-relay==0.9.0" in project["dependencies"]
    assert "nemo-relay-plugin==0.9.0" in project["dependencies"]
    assert "nemoguardrails==0.24.1" in project["dependencies"]


def test_base_runtime_is_pinned_without_optional_profiles() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(project["project"]["dependencies"])

    assert "optional-dependencies" not in project["project"]
    assert "httpx==0.28.1" in dependencies
    assert "nemo-relay==0.9.0" in dependencies
    assert "nemo-relay-plugin==0.9.0" in dependencies
    assert "nemoguardrails==0.24.1" in dependencies
    assert not any(
        name in requirement
        for name in [
            "cleanlab-studio",
            "fast-langdetect",
            "google-cloud-language",
            "guardrails-ai",
            "guardrails-ai-regex-match",
            "langchain-anthropic",
            "presidio-analyzer",
            "torch",
            "transformers",
            "yara-python",
        ]
        for requirement in dependencies
    )


def _export() -> str:
    command = [
        "uv",
        "export",
        "--locked",
        "--python",
        "3.11",
        "--project",
        str(PROJECT_ROOT),
        "--no-default-groups",
        "--no-emit-project",
        "--no-header",
        "--no-annotate",
        "--no-hashes",
    ]
    return subprocess.check_output(command, text=True)


def test_runtime_lock_excludes_unsupported_optional_integrations() -> None:
    exported = _export()
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))
    root = next(package for package in lock["package"] if package["name"] == "nemoguardrails-nemo-relay")
    locked_names = {package["name"] for package in lock["package"]}

    assert "optional-dependencies" not in root
    for package in {
        "cleanlab-studio",
        "fast-langdetect",
        "google-cloud-language",
        "guardrails-ai",
        "guardrails-ai-regex-match",
        "langchain-anthropic",
        "presidio-analyzer",
        "presidio-anonymizer",
        "torch",
        "transformers",
        "yara-python",
    }:
        assert f"{package}==" not in exported
        assert package not in locked_names


def test_release_host_and_runtime_contract_target_relay_09() -> None:
    release = tomllib.loads((PROJECT_ROOT / "release.toml").read_text(encoding="utf-8"))
    runtime = tomllib.loads((PROJECT_ROOT / "relay-plugin.toml").read_text(encoding="utf-8"))

    assert release["relay"]["sha"] == "8122ee1f1765b1e3ef3ef706a8e8233a1630c5ad"
    assert runtime["compat"]["relay"] == ">=0.9.0,<1.0"


def test_build_backend_is_hash_constrained() -> None:
    constraints = (PROJECT_ROOT / "build-constraints.txt").read_text(encoding="utf-8")

    assert "setuptools==84.0.0" in constraints
    assert "--hash=sha256:" in constraints
