# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit evaluator-framework selection for local Guardrails checks."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from enum import Enum
from typing import Any


class EvaluatorFramework(str, Enum):
    """Guardrails model framework used for LLM-backed rails."""

    DEFAULT = "default"
    LANGCHAIN = "langchain"


@dataclass(frozen=True, slots=True)
class EvaluatorFrameworkIssue:
    """A content-safe activation issue."""

    code: str
    message: str


def evaluator_framework_from_config(value: object) -> EvaluatorFramework:
    """Parse the bounded wrapper value."""

    try:
        return EvaluatorFramework(EvaluatorFramework.DEFAULT.value if value is None else value)
    except (TypeError, ValueError):
        raise ValueError("evaluator_framework must be default or langchain") from None


def _installed(distribution: str) -> bool:
    try:
        importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def evaluator_framework_issues(
    rails_config: Any,
    framework: EvaluatorFramework,
) -> tuple[EvaluatorFrameworkIssue, ...]:
    """Validate provider/framework combinations before runtime construction.

    Guardrails 0.24's default and IORails model clients speak OpenAI Chat
    Completions. Anthropic's native Messages API is supported through the
    optional LangChain framework instead; treating it as a generic base URL
    would send the wrong path, authentication scheme, and payload.
    """

    model_engines = {
        model.engine for model in getattr(rails_config, "models", ()) if isinstance(getattr(model, "engine", None), str)
    }
    issues: list[EvaluatorFrameworkIssue] = []
    if "anthropic" in model_engines and framework is not EvaluatorFramework.LANGCHAIN:
        issues.append(
            EvaluatorFrameworkIssue(
                "anthropic_evaluator_requires_langchain",
                "native Anthropic evaluators require evaluator_framework=langchain",
            )
        )

    if framework is EvaluatorFramework.LANGCHAIN:
        unsupported_engines = model_engines - {"anthropic"}
        if unsupported_engines:
            issues.append(
                EvaluatorFrameworkIssue(
                    "unsupported_langchain_evaluator_engine",
                    "this worker qualifies evaluator_framework=langchain only for native Anthropic evaluators",
                )
            )
        else:
            required = {"langchain", "langchain-core", "langchain-community"}
            if "anthropic" in model_engines:
                required.add("langchain-anthropic")
            missing = sorted(distribution for distribution in required if not _installed(distribution))
            if missing:
                issues.append(
                    EvaluatorFrameworkIssue(
                        "missing_evaluator_framework_dependencies",
                        "the selected evaluator framework requires missing Python distributions: " + ", ".join(missing),
                    )
                )
    return tuple(issues)
