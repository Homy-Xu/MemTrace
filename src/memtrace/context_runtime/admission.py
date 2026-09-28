from __future__ import annotations

from dataclasses import dataclass, replace

from ..contracts import ContextArtifact, ContextImage, PressureLevel, Representation
from ..observability import MetricRecorder
from .compaction import Compactor
from .image import ContextImageBuilder
from .pressure import PressurePolicy


@dataclass(frozen=True, slots=True)
class AdmissionOutcome:
    pressure: PressureLevel
    final_pressure: PressureLevel
    image: ContextImage
    admitted_artifact_ids: tuple[str, ...]
    evicted_artifact_ids: tuple[str, ...]
    fixed_point: bool
    fixed_point_reason: str | None
    projected_tokens: int
    final_tokens: int


class ContextAdmission:
    def __init__(
        self,
        policy: PressurePolicy,
        compactor: Compactor,
        builder: ContextImageBuilder,
        metrics: MetricRecorder,
    ) -> None:
        self.policy = policy
        self.compactor = compactor
        self.builder = builder
        self.metrics = metrics

    def admit(
        self,
        *,
        current: ContextImage,
        incoming: tuple[ContextArtifact, ...],
        working_set_milestones: tuple[str, ...],
        focus_terms: tuple[str, ...] = (),
        assembly_overhead: int = 0,
    ) -> AdmissionOutcome:
        if assembly_overhead < 0:
            raise ValueError("assembly_overhead cannot be negative")
        with self.metrics.timer("context_admission_ms"):
            incoming_ids = {item.artifact_id for item in incoming}
            # A recovered Slice is protected regardless of the pressure level
            # observed at the instant it arrives.
            pinned_incoming = tuple(
                self._with_pin(item, 2)
                if item.representation == Representation.SEMANTIC_SLICE
                else item
                for item in incoming
            )
            artifacts = self._deduplicate((*current.artifacts, *pinned_incoming))
            projected = sum(item.token_count for item in artifacts) + assembly_overhead
            initial_pressure = self.policy.level(projected)
            if initial_pressure == PressureLevel.NORMAL:
                image = self.builder.build(
                    thread_id=current.thread_id,
                    artifacts=artifacts,
                    current_milestone_id=current.current_milestone_id,
                    revision_id=current.revision_id,
                )
                return AdmissionOutcome(
                    pressure=initial_pressure,
                    final_pressure=initial_pressure,
                    image=image,
                    admitted_artifact_ids=tuple(
                        item.artifact_id for item in incoming if item.artifact_id in incoming_ids
                    ),
                    evicted_artifact_ids=(),
                    fixed_point=False,
                    fixed_point_reason=None,
                    projected_tokens=projected,
                    final_tokens=image.total_tokens + assembly_overhead,
                )

            mutable = list(artifacts)
            evicted: list[str] = []
            if initial_pressure == PressureLevel.SOFT:
                mutable, removed = self._soft_cleanup(mutable, working_set_milestones)
                evicted.extend(removed)
            fixed_point = False
            fixed_reason: str | None = None
            blocked: set[str] = set()
            while self.policy.level(
                sum(item.token_count for item in mutable) + assembly_overhead
            ) in {PressureLevel.URGENT, PressureLevel.HARD}:
                candidate_index = self._candidate_index(mutable, working_set_milestones, blocked)
                if candidate_index is None:
                    fixed_point = True
                    fixed_reason = (
                        "NO_SAFE_REPRESENTATION_DEMOTION"
                        if blocked
                        else "CORE_CURRENT_MILESTONE_AND_PINNED_EVIDENCE_ONLY"
                    )
                    break
                source = mutable[candidate_index]
                compacted = self.compactor.demote(source, focus_terms=focus_terms)
                if compacted is None:
                    # One artifact being irreducible must not hide another
                    # safe candidate.  Fixed point is global, not per item.
                    blocked.add(source.artifact_id)
                    continue
                mutable[candidate_index] = compacted
                if compacted.representation == Representation.NONRESIDENT:
                    evicted.append(source.artifact_id)

            image = self.builder.build(
                thread_id=current.thread_id,
                artifacts=tuple(mutable),
                current_milestone_id=current.current_milestone_id,
                revision_id=current.revision_id,
            )
            if self.policy.level(image.total_tokens + assembly_overhead) in {
                PressureLevel.URGENT,
                PressureLevel.HARD,
            }:
                native_target = max(
                    0,
                    int(self.policy.budget.effective_limit * self.policy.budget.soft_ratio)
                    - assembly_overhead
                    - 1,
                )
                native = self.compactor.native_compact(
                    image,
                    target_tokens=native_target,
                    builder=self.builder,
                )
                if native is not None:
                    image = native
                    fixed_point = False
                    fixed_reason = None
            final_tokens = image.total_tokens + assembly_overhead
            return AdmissionOutcome(
                pressure=initial_pressure,
                final_pressure=self.policy.level(final_tokens),
                image=image,
                admitted_artifact_ids=tuple(item.artifact_id for item in incoming),
                evicted_artifact_ids=tuple(evicted),
                fixed_point=fixed_point,
                fixed_point_reason=fixed_reason,
                projected_tokens=projected,
                final_tokens=final_tokens,
            )

    def compact_to_fixed_point(
        self,
        *,
        current: ContextImage,
        working_set_milestones: tuple[str, ...],
        focus_terms: tuple[str, ...] = (),
    ) -> AdmissionOutcome:
        """Apply every safe representation downgrade after a physical failure."""

        with self.metrics.timer("context_admission_ms"):
            mutable = list(current.artifacts)
            evicted: list[str] = []
            blocked: set[str] = set()
            while True:
                candidate_index = self._candidate_index(mutable, working_set_milestones, blocked)
                if candidate_index is None:
                    break
                source = mutable[candidate_index]
                compacted = self.compactor.demote(source, focus_terms=focus_terms)
                if compacted is None:
                    blocked.add(source.artifact_id)
                    continue
                mutable[candidate_index] = compacted
                if compacted.representation == Representation.NONRESIDENT:
                    evicted.append(source.artifact_id)
            image = self.builder.build(
                thread_id=current.thread_id,
                artifacts=tuple(mutable),
                current_milestone_id=current.current_milestone_id,
                revision_id=current.revision_id,
            )
            return AdmissionOutcome(
                pressure=self.policy.level(current.total_tokens),
                final_pressure=self.policy.level(image.total_tokens),
                image=image,
                admitted_artifact_ids=(),
                evicted_artifact_ids=tuple(evicted),
                fixed_point=True,
                fixed_point_reason="ALL_SAFE_REPRESENTATION_DEMOTIONS_EXHAUSTED",
                projected_tokens=current.total_tokens,
                final_tokens=image.total_tokens,
            )

    @staticmethod
    def _with_pin(artifact: ContextArtifact, boundaries: int) -> ContextArtifact:
        return replace(
            artifact,
            soft_pin_boundaries=max(artifact.soft_pin_boundaries, boundaries),
        )

    @staticmethod
    def _deduplicate(
        artifacts: tuple[ContextArtifact, ...],
    ) -> tuple[ContextArtifact, ...]:
        """Keep one physical representation for each virtual Page section.

        Content digests deduplicate identical non-addressed artifacts.  Once
        an artifact has a MemoryRef, however, identity is its stable virtual
        address plus Page event range—not whichever summary or recovered body
        happens to be resident.  The later artifact is the new residency
        state and replaces the earlier representation while preserving scope
        metadata.  This is the ContextImage page-replacement invariant.
        """

        result: dict[tuple[object, ...], ContextArtifact] = {}
        order: list[tuple[object, ...]] = []
        for item in artifacts:
            if item.memory_ref and item.source_handles:
                section_locator = tuple(
                    sorted(
                        {
                            (handle.page_id, handle.event_range)
                            for handle in item.source_handles
                            if handle.page_id
                        }
                    )
                )
                key: tuple[object, ...] = (
                    "VIRTUAL_SECTION",
                    item.memory_ref,
                    section_locator,
                )
            else:
                key = ("CONTENT", item.content_digest)
            if key not in result:
                order.append(key)
                result[key] = item
                continue

            existing = result[key]
            if key[0] == "CONTENT":
                if (
                    existing.content != item.content
                    or existing.representation != item.representation
                ):
                    raise RuntimeError("content digest collision in ContextImage")
                selected = existing
            else:
                # ``artifacts`` is ordered current image first, incoming last;
                # the incoming representation is therefore the authoritative
                # residency transition for this stable address.
                selected = item
            result[key] = replace(
                selected,
                milestone_ids=tuple(
                    dict.fromkeys((*existing.milestone_ids, *item.milestone_ids))
                ),
                entity_refs=tuple(dict.fromkeys((*existing.entity_refs, *item.entity_refs))),
                source_handles=tuple(
                    dict.fromkeys((*selected.source_handles, *existing.source_handles))
                ),
                must_preserve=existing.must_preserve or item.must_preserve,
                current_milestone=(existing.current_milestone or item.current_milestone),
                verified=existing.verified or item.verified,
                soft_pin_boundaries=max(existing.soft_pin_boundaries, item.soft_pin_boundaries),
                derived_from=tuple(
                    dict.fromkeys(
                        (
                            *existing.derived_from,
                            *item.derived_from,
                            existing.artifact_id,
                        )
                    )
                ),
            )
        return tuple(result[key] for key in order)

    @classmethod
    def _soft_cleanup(
        cls,
        artifacts: list[ContextArtifact],
        milestones: tuple[str, ...],
    ) -> tuple[list[ContextArtifact], list[str]]:
        """Remove only expired, low-value duplicate recovered slices.

        Different Page ranges are never treated as duplicates merely because
        they are recent.  This is deliberately bounded to the supplied image;
        no historical Page scan occurs.
        """

        keep: list[ContextArtifact] = []
        by_locator: dict[tuple[object, ...], int] = {}
        removed: list[str] = []
        for item in artifacts:
            if (
                item.representation != Representation.SEMANTIC_SLICE
                or not item.source_handles
                or item.soft_pin_boundaries > 0
                or item.must_preserve
                or item.current_milestone
                or any(
                    entity.startswith("context:active_step_lease:") for entity in item.entity_refs
                )
                or cls._in_working_set(item, milestones)
            ):
                keep.append(item)
                continue
            key: tuple[object, ...] = tuple(item.source_handles)
            previous_index = by_locator.get(key)
            if previous_index is None:
                by_locator[key] = len(keep)
                keep.append(item)
                continue
            previous = keep[previous_index]
            if item.token_count < previous.token_count:
                removed.append(previous.artifact_id)
                keep[previous_index] = item
            else:
                removed.append(item.artifact_id)

        # Locator keys seen before a protected item are intentionally not
        # merged with it: protected evidence is never selected for cleanup.
        return keep, removed

    @staticmethod
    def _in_working_set(artifact: ContextArtifact, milestones: tuple[str, ...]) -> bool:
        return bool(set(artifact.milestone_ids).intersection(milestones))

    def _candidate_index(
        self,
        artifacts: list[ContextArtifact],
        milestones: tuple[str, ...],
        blocked: set[str],
    ) -> int | None:
        candidates = []
        for index, item in enumerate(artifacts):
            if (
                item.must_preserve
                or item.current_milestone
                or item.soft_pin_boundaries > 0
                or "context:milestone_handoff" in item.entity_refs
                or item.representation == Representation.NONRESIDENT
                or item.artifact_id in blocked
            ):
                continue
            outside = not self._in_working_set(item, milestones)
            candidates.append(
                (
                    0 if outside else 1,
                    -item.token_count,
                    item.artifact_id,
                    index,
                )
            )
        return min(candidates)[-1] if candidates else None
