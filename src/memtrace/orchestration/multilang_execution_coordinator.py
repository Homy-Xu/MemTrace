"""Opt-in non-Python execution policy built on the frozen Python coordinator.

The multilingual release wheel redirects ``RunCoordinator`` to this subclass.
The source/default Python import continues to use ``execution_coordinator``
unchanged.  Only the two policies that need multilingual recovery semantics are
overridden: predecessor Evidence visibility and bounded advisory acceptance.
"""

from __future__ import annotations

import json
import os

from ..contracts import MilestoneStatus, digest
from ..harness.dynamic_tools import DynamicToolInvocation, DynamicToolResult
from ..harness.memory_tools import CODE_GRAPH_SEARCH_TOOL
from ..planning import CurrentMilestone
from .acceptance_progress import gap_digest
from .execution_coordinator import ExecutionCoordinator as PythonExecutionCoordinator
from .verification_coordinator import MilestoneVerificationBatch, VerificationCoordinator


class MultilangExecutionCoordinator(PythonExecutionCoordinator):
    """Keep the Python memory runtime while making route gaps advisory."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._multilang_navigation_enabled = bool(
            os.environ.get("HOMY_MULTILANG_BASE_COMMIT")
        )
        if not self._multilang_navigation_enabled:
            return
        self._verifier = VerificationCoordinator(
            self.registry.database,
            self.registry,
            self.trace,
            page_store=self.page_store,
            include_predecessor_evidence=True,
        )

        self._navigation_card_key: tuple[str, str] | None = None

    def _execute_dynamic_memory_tool(
        self, invocation: DynamicToolInvocation
    ) -> DynamicToolResult:
        if self._multilang_navigation_enabled and invocation.tool == CODE_GRAPH_SEARCH_TOOL:
            arguments = invocation.arguments
            query = str(arguments.get("query", "")).strip()
            if not query:
                return DynamicToolResult(
                    False,
                    json.dumps(
                        {
                            "schema": "homy/rich-code-search@1",
                            "status": "NO_MATCH",
                            "matches": [],
                            "truncated": False,
                        },
                        sort_keys=True,
                    ),
                    runtime_metadata={"kind": "RICH_HINT"},
                )
            result = self.background.search_code_graph(
                query,
                scope_entities=tuple(
                    str(x)
                    for x in arguments.get("scope_entities", ())
                    if str(x).strip()
                ),
                relation_kinds=tuple(
                    str(x)
                    for x in arguments.get("relation_kinds", ())
                    if str(x).strip()
                ),
                max_results=int(arguments.get("max_results", 8) or 8),
                revision_id=self.revision_id,
                purpose=str(arguments.get("purpose", "navigation")),
            )
            return DynamicToolResult(
                True,
                json.dumps(
                    result, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                runtime_metadata={
                    "kind": "RICH_HINT",
                    "status": result.get("status"),
                    "matches": len(result.get("matches", ())),
                },
            )
        return super()._execute_dynamic_memory_tool(invocation)


    def _render_code_map(self, current: CurrentMilestone):
        rendered = super()._render_code_map(current)
        if not self._multilang_navigation_enabled or rendered is None:
            return rendered
        key = (current.identity_id, self.revision_id)
        if self._navigation_card_key == key:
            return {
                "authority": "RICH_HINT",
                "revision_id": self.revision_id,
                "unchanged": True,
                "note": (
                    "Navigation card unchanged for this Milestone and revision; "
                    "use the prior card or search_code_graph."
                ),
            }
        self._navigation_card_key = key
        encoded = json.dumps(rendered, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= 4096:
            return rendered
        files = list(rendered.get("files", ()))[:4]
        compact_files = []
        for file_card in files:
            item = dict(file_card)
            item["symbols"] = list(item.get("symbols", ()))[:8]
            compact_files.append(item)
        return {**rendered, "files": compact_files, "truncated": True}

    def _verify_claimed_milestones(
        self,
        source_event_id: str,
        *,
        accept_unverified_semantic: bool = False,
        accept_unverified_advisory: bool = False,
    ) -> MilestoneVerificationBatch:
        if not self._multilang_navigation_enabled or not accept_unverified_advisory:
            return super()._verify_claimed_milestones(
                source_event_id,
                accept_unverified_semantic=accept_unverified_semantic,
            )
        batch = self._verifier.verify_claimed_milestones(
            self.request.run_id,
            accept_unverified_semantic=accept_unverified_semantic,
            accept_unverified_advisory=True,
            advisory_source_event_id=source_event_id,
        )
        for canonical_id in batch.verified_canonical_ids:
            self._commit_milestone_page_set(
                canonical_id,
                terminal_state=MilestoneStatus.COMPLETED_VERIFIED.value,
                source_event_id=source_event_id,
            )
        for canonical_id in batch.failed_canonical_ids:
            self._commit_milestone_page_set(
                canonical_id,
                terminal_state=MilestoneStatus.VERIFICATION_FAILED.value,
                source_event_id=source_event_id,
            )
        return batch

    def _continue_claimed_milestone_acceptance(
        self,
        current: CurrentMilestone,
        batch: MilestoneVerificationBatch,
        *,
        durable_source_event_id: str,
    ) -> None:
        if not self._multilang_navigation_enabled:
            return super()._continue_claimed_milestone_acceptance(
                current,
                batch,
                durable_source_event_id=durable_source_event_id,
            )
        canonical_id = current.canonical_id
        unmet = tuple(batch.unmet_criteria.get(canonical_id, ()))
        blocking = tuple(batch.blocking_unmet_criteria.get(canonical_id, ()))
        semantic = tuple(batch.semantic_unmet_criteria.get(canonical_id, ()))
        if not blocking and semantic:
            # The frozen Python path already implements the desired bounded
            # semantic review and UNVERIFIED terminal behavior.
            return super()._continue_claimed_milestone_acceptance(
                current,
                batch,
                durable_source_event_id=durable_source_event_id,
            )

        missing = batch.missing_evidence_types.get(canonical_id, {})
        rejections = batch.evidence_rejection_reasons.get(canonical_id, {})
        gap = gap_digest(unmet, missing, rejections)
        bound = batch.bound_evidence_digests.get(canonical_id, "") or "none"
        progress = self.registry.observe_acceptance_progress(
            run_id=self.request.run_id,
            revision_id=self.revision_id,
            source_event_id=durable_source_event_id,
            boundary_kind="ACCEPTANCE",
            gap_digest=gap,
            bound_evidence_digest=bound,
            scoped_revision_digest=self._milestone_scope_revision_digest(current),
            unmet_criterion_ids=unmet,
        )
        self.trace.record(
            "MILESTONE_ACCEPTANCE_PROGRESS_OBSERVED",
            canonical_id=canonical_id,
            progress_class=progress.progress_class.value,
            weak_streak=progress.weak_streak,
            none_streak=progress.none_streak,
            boundary_count=progress.boundary_count,
            blocking_unmet_criteria=list(blocking),
            semantic_unmet_criteria=list(semantic),
            gap_digest=gap,
            source_event_id=durable_source_event_id,
        )
        exhausted = (
            progress.none_streak >= self.acceptance_budgets.no_progress
            or progress.weak_streak >= self.acceptance_budgets.weak_progress
        )
        if exhausted:
            accepted = self._verify_claimed_milestones(
                durable_source_event_id,
                accept_unverified_advisory=True,
            )
            if canonical_id in accepted.verified_canonical_ids:
                self.trace.record(
                    "MILESTONE_ACCEPTANCE_ADVISORY_BUDGET_EXHAUSTED",
                    canonical_id=canonical_id,
                    unverified_criterion_ids=list(
                        accepted.unverified_criteria.get(canonical_id, ())
                    ),
                    none_streak=progress.none_streak,
                    weak_streak=progress.weak_streak,
                    source_event_id=durable_source_event_id,
                    route_blocked=False,
                )
                self._advance_after_acceptance(durable_source_event_id)
                return

        state = self.registry.record_milestone_state(
            run_id=self.request.run_id,
            canonical_id=canonical_id,
            status=MilestoneStatus.IN_PROGRESS,
            revision_id=self.revision_id,
            source_event_id=durable_source_event_id,
        )
        current = self.registry.current(self.request.run_id)
        self.trace.record(
            "MILESTONE_ACCEPTANCE_NAVIGATION_DEFERRED",
            canonical_id=canonical_id,
            unmet_criterion_ids=list(unmet),
            blocking_unmet_criterion_ids=list(blocking),
            gap_digest=gap,
            state_event_id=state.state_event_id if state is not None else None,
            source_event_id=durable_source_event_id,
            focus_created=False,
            route_blocked=False,
            navigation_frontier=digest(
                {
                    "milestone_identity_id": current.identity_id,
                    "revision_id": self.revision_id,
                    "gap_digest": gap,
                }
            ),
        )
        if self.context_transport is not None:
            self.context_transport.request_task_continuation(
                self._render_current_milestone_turn(
                    current.canonical_id,
                    current.status,
                    source_event_id=durable_source_event_id,
                    missing_evidence_types=missing,
                    evidence_rejection_reasons=rejections,
                )
            )


# The extracted multilingual wheel imports this name in place of the frozen
# Python coordinator.  Keeping the alias local avoids any global monkeypatch.
ExecutionCoordinator = MultilangExecutionCoordinator
