from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from ..contracts import FallbackStage, PageCandidate, RecallIntent
from ..semantic_memory.contracts import ExactEvidenceHit, ExactLookupResult


class ExactSemanticIndex(Protocol):
    def locate_exact(self, intent: RecallIntent) -> ExactLookupResult: ...


@dataclass(frozen=True, slots=True)
class LocatedCandidates:
    candidates: tuple[PageCandidate, ...]
    hits: tuple[ExactEvidenceHit, ...]
    missing_key_digests: tuple[str, ...]


class SemanticLocator:
    """Thin guard around indexed Memory Resolution."""

    def __init__(self, semantic_index: ExactSemanticIndex) -> None:
        self.semantic_index = semantic_index

    def locate(self, intent: RecallIntent) -> LocatedCandidates:
        result = self.semantic_index.locate_exact(intent)
        if not result.sql_filtered:
            raise RuntimeError("Semantic exact lookup did not apply SQL hard filters")
        if result.executed_stage is not FallbackStage.SEMANTIC_EXACT:
            raise RuntimeError("locate_exact reported an unexecuted fallback stage")
        required = {item.key_digest for item in intent.required_evidence}
        if set(result.required_key_digests) != required:
            raise RuntimeError("Semantic exact lookup changed the requested EvidenceKeys")
        candidates: list[PageCandidate] = []
        seen: set[str] = set()
        for candidate in result.candidates:
            if candidate.page_id in seen:
                continue
            if not candidate.payload_digest or candidate.estimated_tokens <= 0:
                continue
            # Exact-revision exclusion is repeated defensively at the boundary;
            # branch-lineage visibility remains the indexed Trace Store's job.
            if intent.require_exact_revision and candidate.revision_id != intent.revision_id:
                continue
            if not required.intersection(candidate.evidence_key_digests):
                continue
            seen.add(candidate.page_id)
            candidates.append(candidate)
        hits = tuple(
            item
            for item in result.hits
            if item.requested_key_digest in required
            and any(candidate.page_id == item.page_id for candidate in candidates)
            and (not intent.require_exact_revision or item.revision_id == intent.revision_id)
        )
        return LocatedCandidates(
            candidates=tuple(candidates),
            hits=hits,
            missing_key_digests=tuple(result.missing_key_digests),
        )


class CandidateRanker:
    """Rank by current marginal exact-evidence gain per estimated token."""

    @staticmethod
    def rank(
        candidates: tuple[PageCandidate, ...] | list[PageCandidate],
        missing_key_digests: frozenset[str],
        *,
        already_read: frozenset[str] = frozenset(),
        hint_key_digests: Mapping[str, frozenset[str]] | None = None,
    ) -> tuple[PageCandidate, ...]:
        scored: list[tuple[float, int, int, int, str, PageCandidate]] = []
        for candidate in candidates:
            if candidate.page_id in already_read:
                continue
            gain = len(missing_key_digests.intersection(candidate.evidence_key_digests))
            # FTS/recent-related candidates deliberately cannot claim exact key
            # Coverage.  Once such a stage really executes, a candidate is given
            # one unit of *estimated* gain solely for I/O ordering; only trace body
            # verification can turn that estimate into Coverage.
            if gain <= 0 and hint_key_digests is not None:
                gain = len(
                    missing_key_digests.intersection(
                        hint_key_digests.get(candidate.page_id, frozenset())
                    )
                )
            if gain <= 0:
                continue
            cost = max(1, candidate.estimated_tokens)
            scored.append(
                (
                    -(gain / cost),
                    -gain,
                    cost,
                    -candidate.freshness_cursor,
                    candidate.page_id,
                    candidate,
                )
            )
        scored.sort(key=lambda item: item[:-1])
        return tuple(item[-1] for item in scored)
