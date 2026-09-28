from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ..contracts import RichGraphCapabilityReceipt
from ..references import ReferenceIdentityFactory

STRUCTURAL_RELATIONS = frozenset(
    {
        "DEFINES",
        "CALLS",
        "CALLED_BY",
        "IMPORTS",
        "IMPORTED_BY",
        "COVERS",
        "COVERED_BY",
        "PRODUCES",
        "RENAMED_TO",
        "MAY_IMPACT",
        "RESOLVES_TO",
        "MAY_RESOLVE_TO",
        "MATCHES_FAILURE",
        "MENTIONS_ENTITY",
        "SAME_TEST_AS",
        "EXTENDS",
        "IMPLEMENTS",
    }
)


def is_structural_relation(relation: str) -> bool:
    """Return whether a relation is eligible for optional Rich expansion."""

    return relation.strip().upper() in STRUCTURAL_RELATIONS


def file_reference_id(repository_id: str, repository_relative_path: str) -> str:
    """Stable logical FileReference ID, intentionally independent of revision."""

    return ReferenceIdentityFactory(repository_id).file(repository_relative_path)


def symbol_reference_id(
    repository_id: str,
    repository_relative_path: str,
    qualified_name: str,
) -> str:
    """Stable logical SymbolReference ID, intentionally independent of revision."""

    return ReferenceIdentityFactory(repository_id).symbol(repository_relative_path, qualified_name)


def test_reference_id(repository_id: str, selector: str) -> str:
    return ReferenceIdentityFactory(repository_id).test(selector)


@dataclass(frozen=True, slots=True)
class MilestoneFrontier:
    """Bounded repository evidence that may seed the Rich background queue.

    Values are repository-relative file paths. Test selectors may use
    ``path/to/test.py::test_name`` and symbols may use ``path.py:qualified.name``.
    The scheduler extracts only their file portion; it never scans the repository.
    """

    workspace_revision_id: str
    current_milestone_id: str
    current_milestone_files: tuple[str, ...] = ()
    dependency_files: tuple[str, ...] = ()
    accessed_files: tuple[str, ...] = ()
    modified_files: tuple[str, ...] = ()
    failed_tests: tuple[str, ...] = ()
    failure_signatures: tuple[str, ...] = ()
    recent_symbols: tuple[str, ...] = ()
    prefetch_files: tuple[str, ...] = ()
    prefetch_budget: int = 4
    # Optional language declared by a benchmark task.  A mixed-language
    # repository leaves this unset and each file extension remains the
    # deterministic fallback.
    language: str | None = None

    def __post_init__(self) -> None:
        if not self.workspace_revision_id:
            raise ValueError("workspace_revision_id is required")
        if not self.current_milestone_id:
            raise ValueError("current_milestone_id is required")
        if self.prefetch_budget < 0:
            raise ValueError("prefetch_budget cannot be negative")


@dataclass(frozen=True, slots=True)
class FrontierTask:
    generation_id: str
    repository_id: str
    workspace_revision_id: str
    repository_relative_path: str
    absolute_path: Path
    priority: int
    reasons: tuple[str, ...]
    failure_signatures: tuple[str, ...]
    enqueued_ns: int
    # Optional task-declared language.  The processor falls back to the
    # repository-relative file extension when this is absent.
    language: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceBinding:
    reference_id: str
    reference_kind: str
    canonical_entity_id: str
    repository_relative_path: str
    qualified_name: str | None = None


@dataclass(frozen=True, slots=True)
class RichRelation:
    relation: str
    source_reference_id: str
    target_reference_id: str
    authority: str
    provenance: tuple[str, ...]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not is_structural_relation(self.relation):
            raise ValueError(f"unsupported Rich relation: {self.relation}")
        if self.authority not in {"ASSERTED", "DERIVED", "INFERRED"}:
            raise ValueError(f"unsupported Rich authority: {self.authority}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class VersionNode:
    node_id: str
    node_kind: str
    reference_id: str | None
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FileProjection:
    repository_relative_path: str
    file_reference_id: str
    file_version_id: str
    content_digest: str
    language: str
    byte_count: int
    bindings: tuple[ReferenceBinding, ...]
    relations: tuple[RichRelation, ...]
    version_nodes: tuple[VersionNode, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RichGraphHint:
    relation: str
    source_reference_id: str
    target_reference_id: str
    authority: str
    provenance: tuple[str, ...]
    confidence: float
    generation_id: str
    workspace_revision_id: str


@dataclass(frozen=True, slots=True)
class RichGraphHintReceipt:
    """A non-blocking answer. Empty hints mean "not currently known", not false."""

    repository_id: str
    workspace_revision_id: str
    entity_id: str
    relation: str
    eligible: bool
    hints: tuple[RichGraphHint, ...]
    queued: bool
    cache_hit: bool
    capability: RichGraphCapabilityReceipt
