from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from ..contracts import (
    CoverageReceipt,
    CoverageState,
    FallbackStage,
    PageCandidate,
    RecallIntent,
    RecoveredContextBlock,
    RichGraphCapabilityReceipt,
    RichGraphState,
    SemanticAnchor,
    primitive,
)
from ..observability import CounterName, MetricRecorder
from ..page_store import PageStoreError
from ..semantic_memory.contracts import ExactEvidenceHit
from .assembler import ContextAssembler
from .coverage import CoverageTracker
from .fallback import ExecutedFallback, FallbackChain, LocalSearchHit
from .intent import normalize_intent, with_required_keys
from .locator import CandidateRanker, SemanticLocator
from .retriever import PageSliceRetriever, RecallBudgetExceeded


class RichHintProvider(Protocol):
    def request_hint(
        self,
        entity_id: str,
        relation: str,
        revision_id: str,
        file_path: str | None = None,
    ) -> object: ...

    def capability(self) -> RichGraphCapabilityReceipt: ...


@dataclass(frozen=True, slots=True)
class RecallTrace:
    executed_stages: tuple[FallbackStage, ...]
    fallback_reasons: tuple[str, ...]
    pages_read: tuple[str, ...]
    slice_levels: tuple[str, ...]
    candidate_advances: int
    stopped_on_complete: bool
    rich_requested: bool
    page_bytes_read: int = 0
    blob_bytes_read: int = 0
    graph_pages_opened: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecallOutcome:
    block: RecoveredContextBlock
    trace: RecallTrace


class RecallService:
    """Formal V2 page-fault path: exact address -> minimal Page reads -> append block."""

    def __init__(
        self,
        *,
        semantic_index: object,
        page_reader: object,
        metrics: MetricRecorder | None = None,
        fallback_chain: FallbackChain | None = None,
        rich_hints: RichHintProvider | None = None,
    ) -> None:
        self.semantic_index = semantic_index
        self.metrics = metrics or MetricRecorder()
        self.locator = SemanticLocator(semantic_index)  # type: ignore[arg-type]
        self.ranker = CandidateRanker()
        self.retriever = PageSliceRetriever(  # type: ignore[arg-type]
            page_reader, self.metrics
        )
        self.assembler = ContextAssembler()
        self.fallback_chain = fallback_chain or FallbackChain(semantic_index)
        self.rich_hints = rich_hints

    def recall(self, intent: RecallIntent) -> RecoveredContextBlock:
        """Convenience production API used by the orchestration layer."""

        return self.fault(intent).block

    def fault(self, intent: RecallIntent) -> RecallOutcome:
        with self.metrics.timer("fault_total_ms"):
            normalized = normalize_intent(intent)
            if normalized.direct_page_ids:
                return self._fault_direct(normalized)
            tracker = CoverageTracker(normalized.required_evidence)
            with self.metrics.timer("semantic_lookup_ms"):
                located = self.locator.locate(normalized)

            executed = [FallbackStage.SEMANTIC_EXACT]
            reasons = ["indexed exact Evidence/Anchor lookup executed"]
            final_stage = FallbackStage.SEMANTIC_EXACT
            slices = []
            local_hits: list[LocalSearchHit] = []
            read_pages: set[str] = set()
            read_order: list[str] = []
            slice_levels: list[str] = []
            candidate_advances = 0
            page_bytes_read = 0
            blob_bytes_read = 0
            slice_tokens = 0
            evidence_by_page: dict[str, tuple[object, ...]] = {}
            relevant_page_count = 0
            first_relevant_rank: int | None = None
            hits: list[ExactEvidenceHit] = list(located.hits)
            pool = list(located.candidates)
            hint_key_digests: dict[str, frozenset[str]] = {}
            graph_candidates: tuple[PageCandidate, ...] = ()
            graph_pages_opened: list[str] = []

            rich_relations, rich_state, rich_requested, rich_reference_ids = (
                self._request_rich(normalized)
            )
            rich_page_candidates = self._rich_page_candidates(
                normalized,
                rich_reference_ids,
                limit=normalized.page_limit,
            )

            def consume_pool() -> None:
                nonlocal candidate_advances
                nonlocal page_bytes_read
                nonlocal blob_bytes_read
                nonlocal slice_tokens
                nonlocal relevant_page_count
                nonlocal first_relevant_rank
                while not tracker.complete and len(read_order) < normalized.page_limit:
                    ranked = self.ranker.rank(
                        pool,
                        tracker.missing,
                        already_read=frozenset(read_pages),
                        hint_key_digests=hint_key_digests,
                    )
                    if not ranked:
                        return
                    # Page Manifest tokens are not an admission budget. Exact
                    # Anchors permit opening a large Page and reading a tiny slice.
                    candidate = ranked[0]
                    if read_order:
                        candidate_advances += 1  # L5: advance to the next useful Page.
                    read_pages.add(candidate.page_id)
                    read_order.append(candidate.page_id)
                    if (
                        candidate.page_id in {item.page_id for item in graph_candidates}
                        and candidate.page_id not in graph_pages_opened
                    ):
                        graph_pages_opened.append(candidate.page_id)
                    try:
                        anchors = self._fallback_anchors(candidate, tracker.missing, tuple(hits))
                        scan = self.retriever.retrieve(
                            candidate=candidate,
                            intent=normalized,
                            missing_key_digests=tracker.missing,
                            hits=tuple(hits),
                            anchors=anchors,
                            max_page_bytes_read=(normalized.max_page_bytes_read - page_bytes_read),
                            max_blob_bytes_read=(normalized.max_blob_bytes_read - blob_bytes_read),
                            max_slice_tokens=min(
                                normalized.max_slice_tokens,
                                # Reserve most of the final Provider payload for
                                # the trust boundary, Coverage, provenance and
                                # delivery metadata. The Page slice is bounded
                                # independently and can never consume the whole
                                # RecoveredContextBlock budget.
                                max(
                                    64,
                                    (
                                        normalized.recovered_block_limit
                                        - slice_tokens
                                        - min(
                                            1_024,
                                            max(
                                                512,
                                                (normalized.recovered_block_limit - slice_tokens)
                                                // 4,
                                            ),
                                        )
                                        if normalized.recovered_block_limit >= 2_048
                                        else (normalized.recovered_block_limit - slice_tokens) // 4
                                    ),
                                ),
                            ),
                        )
                    except (
                        KeyError,
                        OSError,
                        PageStoreError,
                        RecallBudgetExceeded,
                        ValueError,
                    ) as exc:
                        reasons.append(
                            "Page body validation failed for "
                            f"{candidate.page_id}: {type(exc).__name__}"
                        )
                        continue
                    page_bytes_read += scan.page_bytes_read
                    blob_bytes_read += scan.blob_bytes_read
                    tracker.mark_page_validated(candidate.page_id)
                    evidence_by_page[candidate.page_id] = tuple(scan.body_evidence)
                    if scan.page_slice is None:
                        continue
                    if (
                        slice_tokens + scan.page_slice.token_count
                        > normalized.recovered_block_limit
                    ):
                        reasons.append(
                            f"Recovered slice budget excluded {candidate.page_id}; continuation retained"
                        )
                        continue
                    added = tracker.commit_page_body(scan.body_evidence)
                    if added:
                        relevant_page_count += 1
                        if first_relevant_rank is None:
                            first_relevant_rank = len(read_order)
                    slices.append(scan.page_slice)
                    slice_levels.append(scan.terminal_level)
                    slice_tokens += scan.page_slice.token_count
                    # The next loop recomputes marginal gain against *body-
                    # validated* Coverage; no original candidate order survives.

            consume_pool()
            graph_expansion_needed = (
                not tracker.complete
                or any(bool(item.continuation) for item in slices)
                or bool(normalized.required_structural_relations)
                or bool(normalized.preferred_structural_relations)
            )
            if graph_expansion_needed or rich_page_candidates:
                graph_candidates = (
                    self._page_graph_candidates(
                        normalized,
                        tuple(read_order),
                        limit=max(0, normalized.page_limit - len(read_order)),
                    )
                    if read_order
                    else ()
                )
                known = {item.page_id for item in pool}
                for candidate in graph_candidates:
                    if candidate.page_id not in known:
                        pool.append(candidate)
                        known.add(candidate.page_id)
                # A graph neighbor may carry the same missing exact key. Let
                # ordinary body-verified Coverage consume those first.
                consume_pool()
                # If Coverage remains partial, or the model explicitly asked
                # for structural continuity, include bounded neighbor slices
                # as supporting context without claiming exact Coverage.
                supporting_candidates = tuple(
                    {
                        candidate.page_id: candidate
                        for candidate in (*graph_candidates, *rich_page_candidates)
                    }.values()
                )
                for candidate in supporting_candidates:
                    if len(read_order) >= normalized.page_limit:
                        break
                    if candidate.page_id in read_pages:
                        continue
                    if tracker.complete and not (
                        normalized.required_structural_relations
                        or normalized.preferred_structural_relations
                        or normalized.rich_code_relations
                    ):
                        break
                    try:
                        support = self.retriever.retrieve_related(
                            candidate=candidate,
                            intent=normalized,
                            max_page_bytes_read=(normalized.max_page_bytes_read - page_bytes_read),
                            max_blob_bytes_read=(normalized.max_blob_bytes_read - blob_bytes_read),
                            max_slice_tokens=min(
                                normalized.max_slice_tokens,
                                max(64, normalized.recovered_block_limit - slice_tokens),
                            ),
                        )
                    except (
                        KeyError,
                        OSError,
                        PageStoreError,
                        RecallBudgetExceeded,
                        ValueError,
                    ) as exc:
                        reasons.append(
                            "Supporting Page validation failed for "
                            f"{candidate.page_id}: {type(exc).__name__}"
                        )
                        continue
                    read_pages.add(candidate.page_id)
                    read_order.append(candidate.page_id)
                    graph_pages_opened.append(candidate.page_id)
                    page_bytes_read += support.page_bytes_read
                    blob_bytes_read += support.blob_bytes_read
                    tracker.mark_page_validated(candidate.page_id)
                    if support.page_slice is None:
                        continue
                    if (
                        slice_tokens + support.page_slice.token_count
                        > normalized.recovered_block_limit
                    ):
                        reasons.append(
                            f"Recovered slice budget excluded graph Page {candidate.page_id}"
                        )
                        continue
                    slices.append(support.page_slice)
                    slice_levels.append(support.terminal_level)
                    slice_tokens += support.page_slice.token_count
                if graph_candidates:
                    reasons.append(
                        "confirmed Semantic Page relations expanded the bounded candidate set"
                    )
                    self.metrics.increment(CounterName.SEMANTIC_GRAPH_EXPANSION)
                if rich_page_candidates:
                    reasons.append(
                        "Rich Graph references translated to body-validated Semantic Page "
                        "support without claiming Evidence coverage"
                    )
                if supporting_candidates:
                    self.metrics.set("semantic_graph_pages_opened", len(graph_pages_opened))
            for stage in self.fallback_chain.stages:
                if tracker.complete or len(read_order) >= normalized.page_limit:
                    break
                narrowed = with_required_keys(normalized, tracker.missing)
                fallback = self.fallback_chain.execute(stage, narrowed)
                if fallback is None:
                    continue
                final_stage = fallback.stage
                executed.append(fallback.stage)
                reasons.append(fallback.reason)
                local_hits.extend(fallback.local_hits)
                for item in fallback.candidates:
                    hint_key_digests[item.page_id] = frozenset(
                        set(hint_key_digests.get(item.page_id, frozenset())) | set(tracker.missing)
                    )
                self._extend_candidates(pool, fallback)
                consume_pool()

            freshness = f"{normalized.temporal_scope.value}:{normalized.revision_id}"
            while True:
                coverage = tracker.receipt(
                    fallback_stage=final_stage,
                    freshness=freshness,
                    rich_capability_state=rich_state,
                )
                page_relations = self._page_relations(
                    normalized,
                    tuple(item.page_id for item in slices),
                )
                with self.metrics.timer("context_assembly_ms"):
                    block = self.assembler.assemble(
                        intent=normalized,
                        slices=tuple(slices),
                        coverage=coverage,
                        rich_relations=(*page_relations, *rich_relations),
                        local_search_hits=local_hits,
                        max_tokens=normalized.recovered_block_limit,
                    )
                if block.token_count <= normalized.recovered_block_limit or not slices:
                    break
                removed = slices.pop()
                slice_levels.pop()
                reasons.append(f"final recovered-block budget removed slice {removed.slice_id}")
                tracker = CoverageTracker(normalized.required_evidence)
                for accepted in slices:
                    tracker.mark_page_validated(accepted.page_id)
                    tracker.commit_page_body(evidence_by_page.get(accepted.page_id, ()))
            if block.token_count > normalized.recovered_block_limit:
                raise RecallBudgetExceeded("RecoveredContextBlock metadata exceeds final budget")
            self.metrics.set(
                "page_hit_rate",
                relevant_page_count / len(read_order) if read_order else None,
            )
            self.metrics.set("first_relevant_page_rank", first_relevant_rank)
            self.metrics.set("fallback_stage", final_stage.value)
            trace = RecallTrace(
                executed_stages=tuple(executed),
                fallback_reasons=tuple(reasons),
                pages_read=tuple(read_order),
                slice_levels=tuple(slice_levels),
                candidate_advances=candidate_advances,
                stopped_on_complete=tracker.complete,
                rich_requested=rich_requested,
                page_bytes_read=page_bytes_read,
                blob_bytes_read=blob_bytes_read,
                graph_pages_opened=tuple(graph_pages_opened),
            )
            return RecallOutcome(block, trace)

    def _fault_direct(self, intent: RecallIntent) -> RecallOutcome:
        """Dereference a MemoryRef as an address, never as a search seed."""

        candidate_method = getattr(self.semantic_index, "direct_page_candidates", None)
        if candidate_method is None:
            raise TypeError("Semantic Page Table does not support direct Page addresses")
        candidates = candidate_method(
            intent,
            intent.direct_page_ids,
            limit=intent.page_limit,
        )
        if not isinstance(candidates, tuple) or not all(
            isinstance(item, PageCandidate) for item in candidates
        ):
            raise TypeError("direct_page_candidates returned invalid candidates")

        slices = []
        read_order: list[str] = []
        slice_levels: list[str] = []
        reasons = ["runtime-owned MemoryRef dereferenced through the Semantic Page Table"]
        page_bytes_read = 0
        blob_bytes_read = 0
        slice_tokens = 0
        direct_validated = False

        for candidate in candidates:
            if len(read_order) >= intent.page_limit:
                break
            try:
                scan = self.retriever.retrieve_addressed(
                    candidate=candidate,
                    intent=intent,
                    max_page_bytes_read=(intent.max_page_bytes_read - page_bytes_read),
                    max_blob_bytes_read=(intent.max_blob_bytes_read - blob_bytes_read),
                    max_slice_tokens=min(
                        intent.max_slice_tokens,
                        self._available_slice_tokens(intent, slice_tokens),
                    ),
                )
            except (
                KeyError,
                OSError,
                PageStoreError,
                RecallBudgetExceeded,
                ValueError,
            ) as exc:
                reasons.append(
                    f"direct Page validation failed for {candidate.page_id}: {type(exc).__name__}"
                )
                continue
            read_order.append(candidate.page_id)
            page_bytes_read += scan.page_bytes_read
            blob_bytes_read += scan.blob_bytes_read
            direct_validated = True
            if scan.page_slice is None:
                continue
            for page_slice in (scan.page_slice, *scan.additional_slices):
                if slice_tokens + page_slice.token_count > intent.recovered_block_limit:
                    reasons.append(
                        "direct Page section budget retained a directory/continuation "
                        f"address for {candidate.page_id}"
                    )
                    continue
                slices.append(page_slice)
                slice_levels.append(scan.terminal_level)
                slice_tokens += page_slice.token_count
            # One MemoryRef fault opens the directory-selected section only.
            # A slice continuation addresses more bytes in that same section;
            # it is not permission to fan out across sibling Pages. Graph
            # traversal is entered only by an explicit semantic relation intent.
            if not (intent.required_structural_relations or intent.preferred_structural_relations):
                break

        graph_pages_opened: list[str] = []
        graph_needed = bool(slices) and bool(
            intent.required_structural_relations or intent.preferred_structural_relations
        )
        if graph_needed and len(read_order) < intent.page_limit:
            graph_candidates = self._page_graph_candidates(
                intent,
                tuple(read_order),
                limit=intent.page_limit - len(read_order),
            )
            for candidate in graph_candidates:
                if candidate.page_id in read_order or len(read_order) >= intent.page_limit:
                    continue
                try:
                    support = self.retriever.retrieve_related(
                        candidate=candidate,
                        intent=intent,
                        max_page_bytes_read=(intent.max_page_bytes_read - page_bytes_read),
                        max_blob_bytes_read=(intent.max_blob_bytes_read - blob_bytes_read),
                        max_slice_tokens=min(
                            intent.max_slice_tokens,
                            self._available_slice_tokens(intent, slice_tokens),
                        ),
                    )
                except (
                    KeyError,
                    OSError,
                    PageStoreError,
                    RecallBudgetExceeded,
                    ValueError,
                ) as exc:
                    reasons.append(
                        "confirmed graph Page validation failed for "
                        f"{candidate.page_id}: {type(exc).__name__}"
                    )
                    continue
                read_order.append(candidate.page_id)
                graph_pages_opened.append(candidate.page_id)
                page_bytes_read += support.page_bytes_read
                blob_bytes_read += support.blob_bytes_read
                if support.page_slice is None:
                    continue
                if slice_tokens + support.page_slice.token_count > intent.recovered_block_limit:
                    reasons.append(
                        f"graph Page slice budget retained continuation for {candidate.page_id}"
                    )
                    continue
                slices.append(support.page_slice)
                slice_levels.append(support.terminal_level)
                slice_tokens += support.page_slice.token_count
            if graph_pages_opened:
                reasons.append("confirmed Semantic Page relations supplied bounded continuation")
                self.metrics.increment(CounterName.SEMANTIC_GRAPH_EXPANSION)

        address = intent.source_memory_ref or ",".join(intent.direct_page_ids)
        delivered = bool(direct_validated and slices)
        incomplete = delivered and any(bool(item.continuation) for item in slices)
        if incomplete:
            reasons.append(
                "addressed semantic section was only partially delivered; use its exact "
                "continuation before treating the MemoryRef as complete"
            )
        coverage_state = (
            CoverageState.PARTIAL
            if incomplete
            else CoverageState.COMPLETE
            if delivered
            else CoverageState.EMPTY
        )
        covered = (address,) if coverage_state is CoverageState.COMPLETE else ()
        coverage = CoverageReceipt(
            state=coverage_state,
            required_keys=(address,),
            covered_keys=covered,
            missing_keys=(() if coverage_state is CoverageState.COMPLETE else (address,)),
            validated_page_ids=tuple(read_order),
            fallback_stage=FallbackStage.MEMORY_REF_DIRECT,
            freshness=f"{intent.temporal_scope.value}:{intent.revision_id}",
            rich_capability_state=RichGraphState.NOT_STARTED,
        )
        page_relations = (
            self._page_relations(
                intent,
                tuple(item.page_id for item in slices),
            )
            if (intent.required_structural_relations or intent.preferred_structural_relations)
            else ()
        )
        block = self.assembler.assemble_direct(
            intent=intent,
            slices=tuple(slices),
            coverage=coverage,
            page_relations=page_relations,
            max_tokens=intent.recovered_block_limit,
        )
        # Direct addressing bypasses generic Evidence search. Its hit metric
        # must therefore be based on actual assembled Page sections, not the
        # stale value left by an earlier generic search. Continuation remains
        # PARTIAL coverage even when the addressed section is a hit.
        delivered_pages = {item.page_id for item in block.slices}
        self.metrics.set("page_hit_rate", len(delivered_pages.intersection(read_order)) / len(read_order) if read_order else 0.0)
        self.metrics.set("first_relevant_page_rank", next((i for i, page in enumerate(read_order, 1) if page in delivered_pages), None))
        self.metrics.set("fallback_stage", FallbackStage.MEMORY_REF_DIRECT.value)
        return RecallOutcome(
            block,
            RecallTrace(
                executed_stages=(FallbackStage.MEMORY_REF_DIRECT,),
                fallback_reasons=tuple(reasons),
                pages_read=tuple(read_order),
                slice_levels=tuple(slice_levels),
                candidate_advances=max(0, len(read_order) - 1),
                stopped_on_complete=coverage_state is CoverageState.COMPLETE,
                rich_requested=False,
                page_bytes_read=page_bytes_read,
                blob_bytes_read=blob_bytes_read,
                graph_pages_opened=tuple(graph_pages_opened),
            ),
        )

    def _available_slice_tokens(self, intent: RecallIntent, used_tokens: int) -> int:
        """Reserve space for the typed delivery envelope and provenance.

        A Page address must not become an EMPTY recall merely because the
        selected slice consumed the entire admission budget before the
        RecoveredContextBlock metadata was assembled.
        """

        remaining = max(64, intent.recovered_block_limit - used_tokens)
        measured_frame = self.assembler.minimum_direct_frame_tokens()
        # Reserve the complete direct-frame envelope.  Reserving only half of
        # it lets multiple retriever slices consume the body budget before the
        # assembler has written their section addresses and exact cursors.
        reserve = min(max(0, remaining - 64), max(256, measured_frame))
        return max(64, remaining - reserve)

    def _page_graph_candidates(
        self,
        intent: RecallIntent,
        page_ids: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[PageCandidate, ...]:
        if not page_ids or limit <= 0:
            return ()
        method = getattr(self.semantic_index, "page_graph_candidates", None)
        if method is None:
            return ()
        try:
            value = method(intent, page_ids, max_hops=2, limit=limit)
        except Exception:
            return ()
        if not isinstance(value, tuple) or not all(
            isinstance(item, PageCandidate) for item in value
        ):
            raise TypeError("page_graph_candidates returned invalid candidates")
        return value

    def _page_relations(
        self,
        intent: RecallIntent,
        page_ids: tuple[str, ...],
    ) -> tuple[Mapping[str, object], ...]:
        if not page_ids:
            return ()
        method = getattr(self.semantic_index, "page_graph_context", None)
        if method is None:
            return ()
        try:
            value = method(
                page_ids,
                max_hops=1,
                limit=8,
                required_relations=intent.required_structural_relations,
                preferred_relations=intent.preferred_structural_relations,
                direction=intent.structural_relation_direction,
            )
        except Exception:
            # Exact Evidence/Anchor translation remains authoritative; Page
            # flow context is a bounded continuity aid and never blocks Page-in.
            return ()
        if not isinstance(value, tuple) or not all(isinstance(item, Mapping) for item in value):
            raise TypeError("page_graph_context returned invalid relationships")
        return value

    @staticmethod
    def _extend_candidates(pool: list[PageCandidate], fallback: ExecutedFallback) -> None:
        known = {item.page_id for item in pool}
        for candidate in fallback.candidates:
            if candidate.page_id not in known:
                known.add(candidate.page_id)
                pool.append(candidate)

    def _fallback_anchors(
        self,
        candidate: PageCandidate,
        missing_key_digests: frozenset[str],
        hits: tuple[ExactEvidenceHit, ...],
    ) -> tuple[SemanticAnchor, ...]:
        if any(hit.page_id == candidate.page_id for hit in hits):
            return ()
        method = getattr(self.semantic_index, "anchors_for_page", None)
        if method is None:
            return ()
        with self.metrics.timer("semantic_lookup_ms"):
            result = method(candidate.page_id, tuple(sorted(missing_key_digests)))
        if not isinstance(result, tuple) or not all(
            isinstance(item, SemanticAnchor) for item in result
        ):
            raise TypeError("anchors_for_page returned invalid anchors")
        return result

    def _request_rich(
        self, intent: RecallIntent
    ) -> tuple[
        tuple[Mapping[str, object], ...],
        RichGraphState,
        bool,
        tuple[str, ...],
    ]:
        if self.rich_hints is None:
            return (), RichGraphState.NOT_STARTED, False, ()
        try:
            capability = self.rich_hints.capability()
            state = capability.state
        except Exception:
            state = RichGraphState.FAILED
        # Body/evidence absence is intentionally not a Rich trigger.
        requested_relations = intent.rich_code_relations
        requests = [
            (entity, relation)
            for relation in requested_relations
            for entity in (
                intent.ambiguous_entities
                or tuple(key.canonical_entity_id for key in intent.required_evidence)
            )
        ]
        requests.extend((entity, "MAY_RESOLVE_TO") for entity in intent.ambiguous_entities)
        if not requests:
            return (), state, False, ()
        relations: list[Mapping[str, object]] = []
        reference_ids: list[str] = []
        for entity, relation in dict.fromkeys(requests):
            try:
                result = self.rich_hints.request_hint(
                    entity,
                    relation,
                    intent.revision_id,
                    file_path=None,
                )
            except Exception as exc:
                relations.append(
                    {
                        "graph": "RICH_CODE",
                        "entity_id": entity,
                        "relation": relation,
                        "state": "DEGRADED",
                        "reason": type(exc).__name__,
                        "authority": "HINT_ONLY_NOT_COVERAGE",
                    }
                )
                state = RichGraphState.FAILED
            else:
                rendered = primitive(result)
                reference_ids.extend(self._rich_reference_ids(rendered))
                relations.append(
                    {
                        "graph": "RICH_CODE",
                        "entity_id": entity,
                        "relation": relation,
                        "result": rendered,
                        "authority": "HINT_ONLY_NOT_COVERAGE",
                    }
                )
        return tuple(relations), state, True, tuple(dict.fromkeys(reference_ids))

    @classmethod
    def _rich_reference_ids(cls, value: object) -> tuple[str, ...]:
        """Extract only typed Rich reference addresses from a hint receipt."""

        found: list[str] = []
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"source_reference_id", "target_reference_id", "reference_id"}:
                    if isinstance(item, str) and item.strip():
                        found.append(item)
                else:
                    found.extend(cls._rich_reference_ids(item))
        elif isinstance(value, (list, tuple)):
            for item in value:
                found.extend(cls._rich_reference_ids(item))
        return tuple(found)

    def _rich_page_candidates(
        self,
        intent: RecallIntent,
        reference_ids: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[PageCandidate, ...]:
        if not reference_ids or limit <= 0:
            return ()
        method = getattr(self.semantic_index, "page_candidates_for_references", None)
        if method is None:
            return ()
        try:
            result = method(intent, reference_ids, limit=limit)
        except Exception:
            return ()
        if not isinstance(result, tuple) or not all(
            isinstance(item, PageCandidate) for item in result
        ):
            return ()
        return result
