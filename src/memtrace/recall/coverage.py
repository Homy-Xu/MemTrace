from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ..contracts import (
    CoverageReceipt,
    CoverageState,
    EvidenceKey,
    FallbackStage,
    RichGraphState,
    digest,
)


@dataclass(frozen=True, slots=True)
class PageBodyEvidence:
    """An exact fact obtained while inspecting a verified Memory Trace body.

    The retriever is the only production constructor.  The body proof binds the
    Page, Event, exact key, revision, branch and fact content; Manifest/FTS/Rich
    hints do not have enough information to produce a valid instance.
    """

    page_id: str
    event_id: str
    evidence_id: str
    key: EvidenceKey
    revision_id: str
    branch_id: str
    content: Mapping[str, object]
    body_proof: str
    complete: bool = True
    continuation: Mapping[str, object] | None = None

    @classmethod
    def from_verified_body(
        cls,
        *,
        page_id: str,
        event_id: str,
        evidence_id: str,
        key: EvidenceKey,
        revision_id: str,
        branch_id: str,
        content: Mapping[str, object],
        complete: bool = True,
        continuation: Mapping[str, object] | None = None,
    ) -> "PageBodyEvidence":
        proof = digest(
            {
                "source": "PAGE_BODY",
                "page_id": page_id,
                "event_id": event_id,
                "evidence_id": evidence_id,
                "key": key,
                "revision_id": revision_id,
                "branch_id": branch_id,
                "content": content,
                "complete": complete,
                "continuation": continuation,
            }
        )
        return cls(
            page_id=page_id,
            event_id=event_id,
            evidence_id=evidence_id,
            key=key,
            revision_id=revision_id,
            branch_id=branch_id,
            content=dict(content),
            body_proof=proof,
            complete=complete,
            continuation=(None if continuation is None else dict(continuation)),
        )

    def proof_is_valid(self) -> bool:
        return self.body_proof == digest(
            {
                "source": "PAGE_BODY",
                "page_id": self.page_id,
                "event_id": self.event_id,
                "evidence_id": self.evidence_id,
                "key": self.key,
                "revision_id": self.revision_id,
                "branch_id": self.branch_id,
                "content": self.content,
                "complete": self.complete,
                "continuation": self.continuation,
            }
        )


class CoverageTracker:
    """Coverage over exact EvidenceKey digests, committed from validated traces only."""

    def __init__(self, required: Iterable[EvidenceKey]) -> None:
        ordered = tuple(required)
        self._required: dict[str, EvidenceKey] = {item.key_digest: item for item in ordered}
        if not self._required:
            raise ValueError("Coverage requires at least one exact EvidenceKey")
        if len(self._required) != len(ordered):
            raise ValueError("Coverage requires unique exact EvidenceKeys")
        self._covered: dict[str, PageBodyEvidence] = {}
        self._partial: dict[str, PageBodyEvidence] = {}
        self._validated_pages: list[str] = []

    @property
    def required_key_digests(self) -> tuple[str, ...]:
        return tuple(self._required)

    @property
    def covered_key_digests(self) -> tuple[str, ...]:
        return tuple(key for key in self._required if key in self._covered)

    @property
    def missing_key_digests(self) -> tuple[str, ...]:
        return tuple(key for key in self._required if key not in self._covered)

    @property
    def missing(self) -> frozenset[str]:
        return frozenset(self.missing_key_digests)

    @property
    def complete(self) -> bool:
        return len(self._covered) == len(self._required)

    def mark_page_validated(self, page_id: str) -> None:
        if page_id not in self._validated_pages:
            self._validated_pages.append(page_id)

    def commit_page_body(self, evidence: Iterable[PageBodyEvidence]) -> int:
        """Commit exact facts and reject anything lacking its Page-body proof."""

        added = 0
        for item in evidence:
            if not isinstance(item, PageBodyEvidence) or not item.proof_is_valid():
                raise ValueError("Coverage accepts only verified Page body evidence")
            expected = self._required.get(item.key.key_digest)
            # Dataclass equality is intentional: matching a digest hint or a
            # FactType is insufficient if entity/role/revision/branch differs.
            if expected is None or item.key != expected:
                continue
            if item.page_id not in self._validated_pages:
                raise ValueError("Page body must be validated before Coverage commit")
            if item.key.key_digest not in self._covered:
                if item.complete:
                    self._covered[item.key.key_digest] = item
                    self._partial.pop(item.key.key_digest, None)
                    added += 1
                else:
                    self._partial.setdefault(item.key.key_digest, item)
        return added

    def receipt(
        self,
        *,
        fallback_stage: FallbackStage,
        freshness: str,
        rich_capability_state: RichGraphState,
    ) -> CoverageReceipt:
        if self.complete:
            state = CoverageState.COMPLETE
        elif self._covered or self._partial:
            state = CoverageState.PARTIAL
        else:
            state = CoverageState.EMPTY
        return CoverageReceipt(
            state=state,
            required_keys=self.required_key_digests,
            covered_keys=self.covered_key_digests,
            missing_keys=self.missing_key_digests,
            validated_page_ids=tuple(self._validated_pages),
            fallback_stage=fallback_stage,
            freshness=freshness,
            rich_capability_state=rich_capability_state,
        )
