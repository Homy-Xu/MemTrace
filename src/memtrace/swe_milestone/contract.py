"""Immutable per-project verifier contract handed to the runtime by the campaign."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "homy/swe-milestone-verifier-contract@1"

# Root build manifests the official capture copies alongside ``repo_src_dirs``.
ROOT_BUILD_FILES: tuple[str, ...] = (
    "Cargo.toml",
    "Cargo.lock",
    "go.mod",
    "go.sum",
    "go.work",
    "go.work.sum",
    "pom.xml",
)

SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {"go", "rust", "typescript", "javascript", "java", "groovy", "python"}
)


@dataclass(frozen=True, slots=True)
class SweMilestoneContract:
    project_id: str
    languages: tuple[str, ...]
    repo_src_dirs: tuple[str, ...]
    test_dirs: tuple[str, ...]
    exclude_patterns: tuple[str, ...] = ()
    build_command: str | None = None
    test_framework: str | None = None
    submission_tag_prefix: str = "agent-impl-"
    baseline_revision_file: str = ".git/homy-baseline-revision"
    # Wall clock caps.  A verifier that never finishes is worse than a
    # verifier that reports an honest timeout, which the kernel records as an
    # unavailable result rather than as evidence.
    unit_timeout_seconds: int = 900
    total_timeout_seconds: int = 2400
    max_units: int = 24
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def language(self) -> str:
        return self.languages[0] if self.languages else "unknown"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "project_id": self.project_id,
            "languages": list(self.languages),
            "repo_src_dirs": list(self.repo_src_dirs),
            "test_dirs": list(self.test_dirs),
            "exclude_patterns": list(self.exclude_patterns),
            "build_command": self.build_command,
            "test_framework": self.test_framework,
            "submission_tag_prefix": self.submission_tag_prefix,
            "baseline_revision_file": self.baseline_revision_file,
            "unit_timeout_seconds": self.unit_timeout_seconds,
            "total_timeout_seconds": self.total_timeout_seconds,
            "max_units": self.max_units,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "SweMilestoneContract":
        if not isinstance(value, dict) or value.get("schema") != SCHEMA:
            raise ValueError("unsupported SWE-Milestone verifier contract")
        languages = tuple(str(item).strip().lower() for item in value.get("languages", ()))
        if not languages or not set(languages).issubset(SUPPORTED_LANGUAGES):
            raise ValueError(f"unsupported verifier languages: {languages}")
        src_dirs = tuple(_normalize_dir(item) for item in value.get("repo_src_dirs", ()))
        if not src_dirs:
            raise ValueError("verifier contract requires repo_src_dirs")
        prefix = str(value.get("submission_tag_prefix", "agent-impl-"))
        if not prefix:
            raise ValueError("submission tag prefix must not be empty")
        return cls(
            project_id=str(value["project_id"]),
            languages=languages,
            repo_src_dirs=src_dirs,
            test_dirs=tuple(str(item) for item in value.get("test_dirs", ())),
            exclude_patterns=tuple(str(item) for item in value.get("exclude_patterns", ())),
            build_command=(str(value["build_command"]) if value.get("build_command") else None),
            test_framework=(str(value["test_framework"]) if value.get("test_framework") else None),
            submission_tag_prefix=prefix,
            baseline_revision_file=str(
                value.get("baseline_revision_file", ".git/homy-baseline-revision")
            ),
            unit_timeout_seconds=int(value.get("unit_timeout_seconds", 900)),
            total_timeout_seconds=int(value.get("total_timeout_seconds", 2400)),
            max_units=int(value.get("max_units", 24)),
            extra=dict(value.get("extra") or {}),
        )


def _normalize_dir(value: Any) -> str:
    text = str(value).strip().strip("/")
    if not text or text.startswith("..") or text.startswith("/"):
        raise ValueError(f"unsafe repo_src_dirs entry: {value!r}")
    return text


def load_contract(path: str | Path) -> SweMilestoneContract:
    return SweMilestoneContract.from_mapping(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def contract_from_official_config(
    *,
    project_id: str,
    languages: tuple[str, ...],
    raw_config: dict[str, Any],
    build_command: str | None,
    test_framework: str | None,
) -> SweMilestoneContract:
    """Build the contract from the pinned official project YAML.

    Only fields the official evaluator itself uses (source/test directories,
    build command, test framework) are copied; nothing about hidden tests.
    """

    return SweMilestoneContract(
        project_id=project_id,
        languages=tuple(languages),
        repo_src_dirs=tuple(
            _normalize_dir(item) for item in (raw_config.get("repo_src_dirs") or ()) if item
        ),
        test_dirs=tuple(str(item) for item in (raw_config.get("test_dirs") or ()) if item),
        exclude_patterns=tuple(
            str(item) for item in (raw_config.get("exclude_patterns") or ()) if item
        ),
        build_command=build_command,
        test_framework=test_framework,
    )
