from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..contracts import EvidenceKey, FallbackStage, PageCandidate


@dataclass(frozen=True, slots=True)
class EntityResolutionResult:
    """Deterministic translation from model-facing aliases to internal keys."""

    keys: tuple[EvidenceKey, ...]
    resolved_entities: Mapping[str, tuple[str, ...]]
    ambiguous_entities: Mapping[str, tuple[str, ...]]
    unresolved_entities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExactEvidenceHit:
    requested_key_digest: str
    stored_key_digest: str
    evidence_id: str
    anchor_id: str
    page_id: str
    event_id: str
    event_group_id: str
    event_range: tuple[int, int]
    revision_id: str
    authority: str


@dataclass(frozen=True, slots=True)
class ExactLookupResult:
    candidates: tuple[PageCandidate, ...]
    hits: tuple[ExactEvidenceHit, ...]
    required_key_digests: tuple[str, ...]
    matched_key_digests: tuple[str, ...]
    missing_key_digests: tuple[str, ...]
    executed_stage: FallbackStage
    sql_filtered: bool = True

    @property
    def complete(self) -> bool:
        return not self.missing_key_digests


@dataclass(frozen=True, slots=True)
class FallbackQueryResult:
    candidates: tuple[PageCandidate, ...]
    executed: bool
    executed_stage: FallbackStage
    reason: str


@dataclass(frozen=True, slots=True)
class PageProjectionReceipt:
    page_id: str
    cursor: int
    node_count: int
    edge_count: int
    evidence_count: int
    anchor_count: int
    idempotent_replay: bool
