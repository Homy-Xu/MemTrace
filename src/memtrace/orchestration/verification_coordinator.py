from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ..acceptance import criterion_contract_rejections
from ..contracts import (
    GLOBAL_REVISION_FACT_TYPES,
    ClaimType,
    CommitmentLevel,
    CriterionVerificationMode,
    FactType,
    MilestoneStatus,
    TaskStatus,
    default_verification_mode,
    digest,
)
from ..database import StateDatabase
from ..harness.command_semantics import evidence_result_selector_matches
from ..page_store import PageStore
from ..planning import PlanRegistry
from ..recovery import RecoveryAuditor
from .trace import TraceRecorder


@dataclass(frozen=True, slots=True)
class VerificationReceipt:
    current_relation_count: int
    page_count: int
    projected_page_count: int
    unobserved_delivery_count: int
    recovery_active_epoch_id: str
    recovery_workspace_revision_id: str | None
    verified_milestone_count: int
    failed_verification_count: int
    unmet_criteria: dict[str, tuple[str, ...]] = field(default_factory=dict)
    unmet_final_criteria: tuple[str, ...] = ()
    unverified_final_criteria: tuple[str, ...] = ()
    final_review_disposition: str = "CORRECT"
    targeted_requirement_ids: tuple[str, ...] = ()
    correction_requirement_ids: tuple[str, ...] = ()
    completion_verdict: str = "INCOMPLETE"


_HOST_VERIFIER_SELECTOR = "verify_current_milestone"


def _requires_host_verifier(criterion: Mapping[str, object]) -> bool:
    """Whether a Criterion row is bound to the runtime-owned host verifier."""

    evidence_types = {str(getattr(v, "value", v)) for v in criterion.get("required_evidence_types", ())}
    selectors = set(map(str, criterion.get("test_selectors", ()) or ()))
    return FactType.VERIFIER_RESULT.value in evidence_types or _HOST_VERIFIER_SELECTOR in selectors


def criterion_verification_mode(criterion: Mapping[str, object]) -> CriterionVerificationMode:
    """Return the persisted or derived decision procedure of one Criterion row."""

    raw_mode = str(criterion.get("verification_mode", "") or "")
    if raw_mode:
        try:
            return CriterionVerificationMode(raw_mode)
        except ValueError:
            pass
    try:
        claim_type = ClaimType(str(criterion.get("claim_type") or ClaimType.STRUCTURAL.value))
    except ValueError:
        claim_type = ClaimType.STRUCTURAL
    try:
        level = CommitmentLevel(
            str(criterion.get("commitment_level", CommitmentLevel.STEP.value))
        )
    except ValueError:
        level = CommitmentLevel.STEP
    evidence_types: list[FactType] = []
    for item in criterion.get("required_evidence_types", ()):
        try:
            evidence_types.append(FactType(str(item)))
        except ValueError:
            continue
    return default_verification_mode(
        claim_type,
        requirement_id=str(criterion.get("requirement_id", "") or ""),
        required_evidence_types=evidence_types,
        commitment_level=level,
    )


@dataclass(frozen=True, slots=True)
class MilestoneVerificationBatch:
    verified_count: int
    failed_count: int
    verified_canonical_ids: tuple[str, ...]
    failed_canonical_ids: tuple[str, ...]
    unmet_criteria: dict[str, tuple[str, ...]]
    missing_evidence_types: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)
    evidence_rejection_reasons: dict[str, dict[str, tuple[str, ...]]] = field(
        default_factory=dict
    )
    satisfied_evidence_event_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    failed_evidence_event_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    failed_criteria: dict[str, tuple[str, ...]] = field(default_factory=dict)
    failure_signatures: dict[str, str] = field(default_factory=dict)
    # Unmet criteria partitioned by decision procedure.  Only the blocking
    # partition may hold a route open; the semantic partition is decided by a
    # bounded model review and otherwise reported as UNVERIFIED.
    blocking_unmet_criteria: dict[str, tuple[str, ...]] = field(default_factory=dict)
    semantic_unmet_criteria: dict[str, tuple[str, ...]] = field(default_factory=dict)
    unverified_criteria: dict[str, tuple[str, ...]] = field(default_factory=dict)
    bound_evidence_digests: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StepVerificationReceipt:
    step_id: str
    satisfied: bool
    unmet_criteria: tuple[str, ...]
    missing_evidence_types: dict[str, tuple[str, ...]]
    evidence_event_ids: tuple[str, ...]
    evidence_rejection_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    failed_criteria: tuple[str, ...] = ()
    failed_evidence_event_ids: tuple[str, ...] = ()
    failure_signature: str | None = None
    # Total number of factual criteria assessed and the normalized digest of
    # the evidence bound to them; used by the acceptance-progress fence.
    assessed_criteria: tuple[str, ...] = ()
    bound_evidence_digest: str = ""


@dataclass(frozen=True, slots=True)
class StepSpanVerificationReceipt:
    """Acceptance result for the single authoritative current Step.

    ``covered_step_ids`` remains tuple-shaped for durable-schema compatibility,
    but new execution never contains more than one Step.
    """

    owner_step_id: str
    covered_step_ids: tuple[str, ...]
    satisfied: bool
    unmet_criteria: tuple[str, ...]
    missing_evidence_types: dict[str, tuple[str, ...]]
    evidence_event_ids: tuple[str, ...]
    evidence_rejection_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    failed_criteria: tuple[str, ...] = ()
    failed_evidence_event_ids: tuple[str, ...] = ()
    failure_signature: str | None = None


@dataclass(frozen=True, slots=True)
class MilestoneFailureContext:
    canonical_id: str
    revision_id: str
    criterion_ids: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    signature: str


@dataclass(frozen=True, slots=True)
class _CriterionVerdict:
    criterion_id: str
    satisfied: bool
    missing_evidence_types: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    failed_evidence_event_ids: tuple[str, ...]
    evidence_rejection_reasons: tuple[str, ...]
    matching: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _CriterionEvidenceSelection:
    """One deterministic partition of Evidence for a Criterion.

    The verifier, diagnostic surface and state transition all consume this
    same partition.  This prevents a second, prompt-only interpretation of why
    an otherwise relevant fact was not admissible acceptance Evidence.
    """

    matching: tuple[object, ...]
    stale: tuple[object, ...]
    unbound_direct: tuple[object, ...]


class VerificationCoordinator:
    """Checks cross-stage invariants on the same authoritative state database."""

    UNRELIABLE_EXIT_STATUS = "UNRELIABLE_EXIT_STATUS"
    FAILED_OBSERVATION = "FAILED_OBSERVATION"
    STALE_REVISION = "STALE_REVISION"
    UNBOUND_ENTITY_OR_SELECTOR = "UNBOUND_ENTITY_OR_SELECTOR"
    UNPROVEN_SUCCESS = "UNPROVEN_SUCCESS"
    CODE_CHANGE_ONLY_ACTION_EVIDENCE = "CODE_CHANGE_ONLY_ACTION_EVIDENCE"
    INVALID_BEHAVIORAL_COMMITMENT = "INVALID_BEHAVIORAL_COMMITMENT"
    REQUIREMENT_NOT_REVIEWED = "REQUIREMENT_NOT_REVIEWED"
    REQUIREMENT_NOT_SATISFIED = "REQUIREMENT_NOT_SATISFIED"
    STEP_SEMANTIC_CONCLUSION_REQUIRED = "STEP_SEMANTIC_CONCLUSION_REQUIRED"
    INCOMPLETE_STEP_ENTITY_COVERAGE = "INCOMPLETE_STEP_ENTITY_COVERAGE"
    COMPOSITE_OBSERVATION = "CURRENT_SUCCESS_OBSERVATION"
    COMPOSITE_OBSERVATION_REQUIRED = "COMPOSITE_OBSERVATION_REQUIRED"

    _CURRENT_REVISION_EVIDENCE_TYPES = frozenset(
        item.value for item in GLOBAL_REVISION_FACT_TYPES
    )
    _VERIFICATION_OBSERVATION_TYPES = frozenset(
        {"TEST_RESULT", "TEST_FAILURE", "VERIFIER_RESULT", "TOOL_RESULT"}
    )

    def __init__(
        self,
        database: StateDatabase,
        registry: PlanRegistry,
        trace: TraceRecorder,
        page_store: PageStore | None = None,
    ) -> None:
        self.database = database
        self.registry = registry
        self.trace = trace
        self.page_store = page_store
        # SWE-Milestone submit gate: a Milestone bound to an official ID may
        # not become COMPLETED_VERIFIED while ``git tag agent-impl-<id>`` is
        # missing.  The execution coordinator installs the callable; ``None``
        # (ordinary tasks) leaves acceptance unchanged.
        self.official_submit_gate: Callable[[str], str | None] | None = None
        self.database.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS v2_final_verification_results (
                result_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                verdict TEXT NOT NULL CHECK(verdict IN ('COMPLETED','INCOMPLETE')),
                source_event_id TEXT NOT NULL,
                detail_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS v2_final_verification_no_update
            BEFORE UPDATE ON v2_final_verification_results
            BEGIN SELECT RAISE(ABORT, 'Final verification is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS v2_final_verification_no_delete
            BEFORE DELETE ON v2_final_verification_results
            BEGIN SELECT RAISE(ABORT, 'Final verification is append-only'); END;
            """
        )

    def verify_run(self, run_id: str, *, finalize_task: bool = True) -> VerificationReceipt:
        connection = self.database.connection
        batch = self.verify_claimed_milestones(run_id)
        current_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM v2_semantic_edges "
                "WHERE run_id=? AND edge_type='CURRENT_MILESTONE' "
                "AND valid_to_cursor IS NULL",
                (run_id,),
            ).fetchone()[0]
        )
        if current_count != 1:
            raise RuntimeError(f"run must have exactly one current Milestone, got {current_count}")
        page_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM v2_pages WHERE run_id=?", (run_id,)
            ).fetchone()[0]
        )
        projected_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM v2_semantic_pages WHERE run_id=?", (run_id,)
            ).fetchone()[0]
        )
        if projected_count != page_count:
            raise RuntimeError(
                f"Page Store/Semantic projection divergence: {page_count} != {projected_count}"
            )
        unobserved = int(
            connection.execute(
                "SELECT COUNT(*) FROM context_deliveries WHERE state<>'MODEL_OBSERVED'"
            ).fetchone()[0]
        )
        self.registry.current(run_id)
        branch = connection.execute(
            "SELECT branch_id FROM v2_tasks WHERE run_id=?", (run_id,)
        ).fetchone()
        if branch is None:
            raise KeyError(run_id)
        recovery = RecoveryAuditor(self.database).audit(run_id, str(branch["branch_id"]))
        statuses = self.registry.milestone_statuses(run_id)
        all_verified = bool(statuses) and all(
            value == MilestoneStatus.COMPLETED_VERIFIED.value for value in statuses.values()
        )
        pending_side_effects = int(
            connection.execute(
                "SELECT COUNT(*) FROM v2_side_effects WHERE run_id=? "
                "AND state NOT IN ('CONFIRMED','FAILED')",
                (run_id,),
            ).fetchone()[0]
        )
        unresolved = int(
            connection.execute(
                "SELECT COUNT(*) FROM v2_unresolved_milestone_observations WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
        )
        unverified_by_milestone = self.registry.unverified_criterion_ids(run_id)
        unverified_final = tuple(
            dict.fromkeys(
                criterion_id
                for criterion_ids in unverified_by_milestone.values()
                for criterion_id in criterion_ids
            )
        )
        unmet_final = self._unmet_final_acceptance(
            run_id,
            branch_id=str(branch["branch_id"]),
            revision_id=recovery.workspace_revision_id,
            unverified_criterion_ids=unverified_final,
        )
        requirement_coverage = self.registry.requirement_coverage(run_id)
        # These are diagnostics for the bounded model review, not completion
        # gates.  The links are deterministic address translations and may be
        # incomplete for paraphrased or explanatory Task prose.  Correctness
        # still comes from the immutable Task-final review and trusted current-
        # revision Evidence, never from a lexical score.
        correction_requirement_ids = tuple(
            str(item["requirement_id"])
            for item in requirement_coverage
            if bool(item["required"])
            and item["coverage_state"] in {"UNMAPPED", "ROUTE_PENDING"}
        )
        targeted_requirement_ids = tuple(
            str(item["requirement_id"])
            for item in requirement_coverage
            if bool(item["required"])
            and item["coverage_state"] == "ROUTE_VERIFIED"
            and item["category"]
            in {"BEHAVIORAL", "VERIFICATION", "PERFORMANCE"}
            and not bool(item["verification_capable"])
        )
        base_complete = (
            all_verified
            and unobserved == 0
            and pending_side_effects == 0
            and not batch.unmet_criteria
            and not unmet_final
        )
        # Report diagnostics for every Milestone that did not reach
        # COMPLETED_VERIFIED, including stalled or failed ones the acceptance
        # batch no longer re-verifies.  Read-only: the batch above stays the
        # sole acceptance authority; this only makes ``result.json`` say exactly
        # which criteria stayed unmet.
        unmet_report: dict[str, tuple[str, ...]] = dict(batch.unmet_criteria)
        for canonical_id, status in statuses.items():
            if status == MilestoneStatus.COMPLETED_VERIFIED.value or canonical_id in unmet_report:
                continue
            try:
                facts = self.assess_milestone_facts(run_id, canonical_id)
            except (KeyError, ValueError):
                continue
            if facts.unmet_criteria:
                unmet_report[canonical_id] = tuple(facts.unmet_criteria)
        verified_milestone_count = sum(
            value == MilestoneStatus.COMPLETED_VERIFIED.value for value in statuses.values()
        )
        failed_verification_count = sum(
            value == MilestoneStatus.VERIFICATION_FAILED.value for value in statuses.values()
        )
        final_review_disposition = (
            "COMPLETE"
            if base_complete
            else "CORRECT"
            if failed_verification_count or batch.failed_criteria
            else "TARGETED_VERIFY"
        )
        completion_verdict = "COMPLETED" if base_complete else "INCOMPLETE"
        source = connection.execute(
            "SELECT source_event_id,revision_id FROM v2_milestone_state_events "
            "WHERE run_id=? ORDER BY created_cursor DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if source is not None:
            detail = {
                "milestone_statuses": statuses,
                "unmet_criteria": unmet_report,
                "unobserved_deliveries": unobserved,
                "pending_side_effects": pending_side_effects,
                # These rows are append-only receipts for native Plan input
                # which was fenced from the authoritative Registry. They are
                # useful diagnostics, but are not live work and cannot remain
                # an irreversible completion blocker after a later accepted
                # Milestone review.
                "unapplied_plan_observations": unresolved,
                "unmet_final_criteria": list(unmet_final),
                # Semantic requirements whose bounded review never reached a
                # verdict.  They are disclosed to the grader instead of holding
                # the route in an unbounded review loop.
                "unverified_final_criteria": list(unverified_final),
                "unverified_criteria_by_milestone": {
                    milestone_id: list(criterion_ids)
                    for milestone_id, criterion_ids in unverified_by_milestone.items()
                },
                "final_review_disposition": final_review_disposition,
                "targeted_requirement_ids": list(targeted_requirement_ids),
                "correction_requirement_ids": list(correction_requirement_ids),
                "immutable_requirement_coverage": requirement_coverage,
            }
            result_id = f"final:{run_id}:{source['source_event_id']}:{completion_verdict}"
            with self.database.transaction() as writer:
                writer.execute(
                    "INSERT OR IGNORE INTO v2_final_verification_results "
                    "VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)",
                    (
                        result_id,
                        run_id,
                        completion_verdict,
                        str(source["source_event_id"]),
                        json.dumps(detail, sort_keys=True),
                    ),
                )
            if completion_verdict == "COMPLETED" and finalize_task:
                current_task = self.registry.task_status(run_id)
                if current_task is TaskStatus.EXECUTING:
                    self.registry.record_task_state(
                        run_id=run_id,
                        status=TaskStatus.VERIFYING,
                        revision_id=str(source["revision_id"]),
                        source_event_id=str(source["source_event_id"]),
                    )
                    current_task = TaskStatus.VERIFYING
                if current_task is TaskStatus.VERIFYING:
                    self.registry.record_task_state(
                        run_id=run_id,
                        status=TaskStatus.COMPLETED,
                        revision_id=str(source["revision_id"]),
                        source_event_id=str(source["source_event_id"]),
                    )
        receipt = VerificationReceipt(
            current_relation_count=current_count,
            page_count=page_count,
            projected_page_count=projected_count,
            unobserved_delivery_count=unobserved,
            recovery_active_epoch_id=recovery.active_epoch_id,
            recovery_workspace_revision_id=recovery.workspace_revision_id,
            verified_milestone_count=verified_milestone_count,
            failed_verification_count=failed_verification_count,
            unmet_criteria=unmet_report,
            unmet_final_criteria=unmet_final,
            unverified_final_criteria=unverified_final,
            final_review_disposition=final_review_disposition,
            targeted_requirement_ids=targeted_requirement_ids,
            correction_requirement_ids=correction_requirement_ids,
            completion_verdict=completion_verdict,
        )
        self.trace.record(
            "FINAL_INVARIANTS_VERIFIED",
            current_relation_count=current_count,
            page_count=page_count,
            projected_page_count=projected_count,
            unobserved_delivery_count=unobserved,
            verified_milestone_count=verified_milestone_count,
            failed_verification_count=failed_verification_count,
            unmet_criteria=unmet_report,
            unmet_final_criteria=list(unmet_final),
            unverified_final_criteria=list(unverified_final),
            final_review_disposition=final_review_disposition,
            targeted_requirement_ids=list(targeted_requirement_ids),
            correction_requirement_ids=list(correction_requirement_ids),
        )
        self.trace.record(
            "FINAL_COMPLETION_VERDICT",
            verdict=completion_verdict,
            milestone_statuses=statuses,
            unmet_criteria=unmet_report,
            unmet_final_criteria=list(unmet_final),
            unverified_final_criteria=list(unverified_final),
            unobserved_delivery_count=unobserved,
            pending_side_effect_count=pending_side_effects,
            unapplied_plan_observation_count=unresolved,
            final_review_disposition=final_review_disposition,
            targeted_requirement_ids=list(targeted_requirement_ids),
            correction_requirement_ids=list(correction_requirement_ids),
        )
        return receipt

    def verify_step_acceptance(
        self,
        run_id: str,
        step: dict[str, object],
    ) -> StepVerificationReceipt:
        """Verify one Step from evidence causally attributed to its active span.

        ``criterion_ids`` are contribution links into the Milestone contract;
        they are never an alternate Step acceptance contract.  The terminal
        Milestone predicate is evaluated separately after every local Step is
        complete. This separation prevents unrelated history from advancing
        the current route node while allowing one natural Turn to satisfy a
        contiguous lightweight Step span.
        """

        span = self.verify_step_span(run_id, (step,))
        return StepVerificationReceipt(
            step_id=span.owner_step_id,
            satisfied=span.satisfied,
            unmet_criteria=span.unmet_criteria,
            missing_evidence_types=span.missing_evidence_types,
            evidence_event_ids=span.evidence_event_ids,
            evidence_rejection_reasons=span.evidence_rejection_reasons,
            failed_criteria=span.failed_criteria,
            failed_evidence_event_ids=span.failed_evidence_event_ids,
            failure_signature=span.failure_signature,
        )

    def verify_step_span(
        self,
        run_id: str,
        steps: tuple[dict[str, object], ...],
    ) -> StepSpanVerificationReceipt:
        """Evaluate a contiguous Step span from its current causal owner."""

        if not steps:
            raise ValueError("Step span must contain its causal owner")
        owner = steps[0]
        owner_step_id = str(owner["step_id"])
        if any(
            str(step["milestone_identity_id"]) != str(owner["milestone_identity_id"])
            for step in steps
        ):
            raise ValueError("Step span cannot cross a Milestone boundary")
        current_revision_id, evidence = self._step_evidence_scope(run_id, owner)
        # A natural Turn can cross the boundary between two lightweight
        # navigation cursors. The first command may already have produced the
        # audit fact that the immediately following cursor needs. Requiring a
        # second read solely because the route pointer advanced recreates the
        # old Step-approval loop. Reuse is deliberately bounded to the direct
        # predecessor in the same Milestone and to evidence without a
        # conflicting explicit Criterion binding; unrelated or older history
        # remains out of scope.
        predecessor_step_id = self._immediate_predecessor_step_id(owner_step_id)
        scoped = []
        for row in evidence:
            content = self._content(row)
            plan_step_id = str(content.get("plan_step_id", ""))
            if plan_step_id == owner_step_id:
                scoped.append(row)
                continue
            if plan_step_id != predecessor_step_id:
                continue
            explicit = content.get("criterion_ids", content.get("criterion_id", ()))
            if explicit:
                # Explicitly addressed predecessor evidence is reusable only
                # when its address is also declared by this Step; otherwise it
                # remains evidence for the predecessor alone.
                explicit_ids = (
                    {explicit}
                    if isinstance(explicit, str)
                    else set(map(str, explicit))
                    if isinstance(explicit, (list, tuple))
                    else set()
                )
                current_ids = set(map(str, owner.get("criterion_ids", ())))
                if not explicit_ids.intersection(current_ids):
                    continue
            scoped.append(row)
        scoped_ids = {str(row["evidence_id"]) for row in scoped}
        compatible_criterion_ids = tuple(
            dict.fromkeys(
                criterion_id
                for step in steps
                for criterion_id in map(str, step.get("criterion_ids", ()))
            )
        )
        unmet: list[str] = []
        missing: dict[str, tuple[str, ...]] = {}
        rejection_reasons: dict[str, tuple[str, ...]] = {}
        selected_events: list[str] = []
        failed_criteria: list[str] = []
        failed_events: list[str] = []
        required_count = 0
        for step in steps:
            step_id = str(step["step_id"])
            criteria = [
                item
                for item in step.get("minimum_acceptance", ())
                if isinstance(item, dict) and bool(item.get("required", True))
            ]
            required_count += len(criteria)
            for criterion in criteria:
                local_id = str(criterion.get("criterion_id", ""))
                qualified_id = f"{step_id}:{local_id}"
                verdict = self._evaluate_criterion(
                    evidence,
                    criterion,
                    # A Step receipt proves that its causally owned work
                    # happened.  It remains a historical execution fact after
                    # later revisions.  Current-code truth is enforced once,
                    # separately, by the Milestone/final acceptance predicate.
                    current_revision_id=None,
                    direct_evidence_ids=scoped_ids,
                    compatible_criterion_ids=compatible_criterion_ids,
                )
                if verdict.satisfied:
                    selected_events.extend(verdict.evidence_event_ids)
                    continue
                unmet.append(qualified_id)
                missing[qualified_id] = verdict.missing_evidence_types
                if verdict.evidence_rejection_reasons:
                    rejection_reasons[qualified_id] = verdict.evidence_rejection_reasons
                if verdict.failed_evidence_event_ids:
                    failed_criteria.append(qualified_id)
                    failed_events.extend(verdict.failed_evidence_event_ids)
            mutation_outcome = self._mutation_step_outcome_receipt(
                step=step,
                scoped_evidence=scoped,
                current_revision_id=current_revision_id,
                owner_step_id=owner_step_id,
                compatible_criterion_ids=compatible_criterion_ids,
            )
            if mutation_outcome is not None:
                qualified_id, semantic_event_id = mutation_outcome
                if semantic_event_id is None:
                    if qualified_id not in unmet:
                        unmet.append(qualified_id)
                    missing[qualified_id] = (FactType.CODE_OBSERVATION.value,)
                    rejection_reasons[qualified_id] = (
                        self.CODE_CHANGE_ONLY_ACTION_EVIDENCE,
                    )
                else:
                    selected_events.append(semantic_event_id)
        if required_count == 0:
            return StepSpanVerificationReceipt(
                owner_step_id=owner_step_id,
                covered_step_ids=tuple(str(step["step_id"]) for step in steps),
                satisfied=True,
                unmet_criteria=(),
                missing_evidence_types={},
                evidence_event_ids=(),
                evidence_rejection_reasons={},
            )
        unique_failed_events = tuple(dict.fromkeys(failed_events))
        failure_signature = (
            self._failure_signature(
                revision_id=current_revision_id,
                criterion_ids=tuple(dict.fromkeys(failed_criteria)),
                evidence=[
                    row for row in evidence if str(row["event_id"]) in unique_failed_events
                ],
            )
            if unique_failed_events
            else None
        )
        return StepSpanVerificationReceipt(
            owner_step_id=owner_step_id,
            covered_step_ids=tuple(str(step["step_id"]) for step in steps),
            satisfied=not unmet,
            unmet_criteria=tuple(unmet),
            missing_evidence_types=missing,
            evidence_event_ids=tuple(dict.fromkeys(selected_events)),
            evidence_rejection_reasons=rejection_reasons,
            failed_criteria=tuple(dict.fromkeys(failed_criteria)),
            failed_evidence_event_ids=unique_failed_events,
            failure_signature=failure_signature,
        )

    @staticmethod
    def _immediate_predecessor_step_id(step_id: str) -> str | None:
        """Return the prior numbered navigation cursor, if one is addressable."""

        match = re.fullmatch(r"(.+)\.S(\d+)", str(step_id))
        if match is None:
            return None
        number = int(match.group(2))
        return f"{match.group(1)}.S{number - 1:03d}" if number > 1 else None

    @classmethod
    def _mutation_step_outcome_receipt(
        cls,
        *,
        step: dict[str, object],
        scoped_evidence: list[object],
        current_revision_id: str,
        owner_step_id: str,
        compatible_criterion_ids: tuple[str, ...],
    ) -> tuple[str, str | None] | None:
        """Validate an explicitly declared post-change observation Criterion.

        The persisted Step contract is the complete Criterion address space.
        Verification must never invent a hidden ``post_change`` Criterion after
        execution has started: such an address cannot be submitted through the
        semantic-update tool and splits the acceptance authority in two.

        ``CODE_CHANGE`` remains an action receipt.  Live Plans are validated at
        ingestion so a Step which declares that receipt also declares a semantic
        outcome (a code observation or typed test/verifier result).  When the
        explicit outcome is ``CODE_OBSERVATION``, this method additionally
        proves that the observation is current, model-authored, Criterion-bound,
        and over an entity that actually changed.  Typed test/verifier Criteria
        are evaluated by the ordinary Criterion kernel and need no redundant
        model bookkeeping fact.
        """

        criteria = tuple(
            item
            for item in step.get("minimum_acceptance", ())
            if isinstance(item, dict) and bool(item.get("required", True))
        )
        current_changes = tuple(
            row
            for row in scoped_evidence
            if str(row["evidence_type"]) == FactType.CODE_CHANGE.value
            and str(row["revision_id"]) == current_revision_id
        )
        if not current_changes:
            return None
        observation_ids = tuple(
            str(criterion.get("criterion_id", ""))
            for criterion in criteria
            if FactType.CODE_OBSERVATION.value
            in set(map(str, criterion.get("required_evidence_types", ())))
        )
        if not observation_ids:
            return None
        qualified_id = f"{step['step_id']}:{observation_ids[0]}"
        changed_aliases = {
            alias
            for row in current_changes
            for alias in cls._entity_aliases(str(row["canonical_entity_id"]))
        }
        if not changed_aliases:
            return qualified_id, None
        for row in scoped_evidence:
            if (
                str(row["evidence_type"]) != FactType.CODE_OBSERVATION.value
                or str(row["revision_id"]) != current_revision_id
                or str(row["semantic_role"]) != "agent_observation"
            ):
                continue
            content = cls._content(row)
            if content.get("authority") != "STRUCTURED_AGENT_OUTPUT":
                continue
            local_binding = any(
                cls._has_explicit_criterion_binding(row, criterion_id)
                for criterion_id in observation_ids
            )
            merged_binding = (
                str(step["step_id"]) != owner_step_id
                and any(
                    cls._has_explicit_criterion_binding(row, criterion_id)
                    for criterion_id in compatible_criterion_ids
                )
            )
            if not local_binding and not merged_binding:
                continue
            observed_aliases = cls._entity_aliases(str(row["canonical_entity_id"]))
            if not changed_aliases.intersection(observed_aliases):
                continue
            return qualified_id, str(row["event_id"])
        return qualified_id, None

    def _step_evidence_scope(
        self,
        run_id: str,
        step: dict[str, object],
        *,
        include_same_revision_history: bool = False,
    ) -> tuple[str, list[object]]:
        """Load the causal Milestone history used by local Step acceptance."""

        has_revision_table = self.database.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='v2_current_workspace_revision'"
        ).fetchone()
        if has_revision_table is None:
            scope = self.database.connection.execute(
                "SELECT branch_id,initial_revision_id current_revision "
                "FROM v2_tasks WHERE run_id=?",
                (run_id,),
            ).fetchone()
        else:
            scope = self.database.connection.execute(
                "SELECT t.branch_id,COALESCE(("
                " SELECT wr.revision_id FROM v2_current_workspace_revision wr"
                " WHERE wr.run_id=t.run_id AND wr.branch_id=t.branch_id"
                "),t.initial_revision_id) current_revision "
                "FROM v2_tasks t WHERE t.run_id=?",
                (run_id,),
            ).fetchone()
        if scope is None:
            raise KeyError(run_id)
        current_revision_id = str(scope["current_revision"])
        if include_same_revision_history:
            evidence = list(
                self.database.connection.execute(
                    "SELECT e.evidence_id,e.evidence_type,e.event_id,e.revision_id,e.content_json,"
                    "e.canonical_entity_id,e.semantic_role,e.valid_to_cursor "
                    "FROM v2_semantic_evidence e "
                    "JOIN v2_semantic_events ev ON ev.event_id=e.event_id "
                    "WHERE e.run_id=? AND e.branch_id=? AND e.revision_id=? "
                    "ORDER BY ev.observed_at DESC,ev.event_position DESC,e.evidence_id DESC",
                    (run_id, str(scope["branch_id"]), current_revision_id),
                ).fetchall()
            )
        else:
            evidence = list(
                self.database.connection.execute(
                    "SELECT e.evidence_id,e.evidence_type,e.event_id,e.revision_id,e.content_json,"
                    "e.canonical_entity_id,e.semantic_role,e.valid_to_cursor "
                    "FROM v2_semantic_evidence e "
                    "JOIN v2_semantic_events ev ON ev.event_id=e.event_id "
                    "WHERE e.run_id=? AND e.branch_id=? AND ev.milestone_identity_id=? "
                    "ORDER BY ev.observed_at DESC,ev.event_position DESC,e.evidence_id DESC",
                    (run_id, str(scope["branch_id"]), str(step["milestone_identity_id"])),
                ).fetchall()
            )
        open_evidence: list[object] = []
        if self.page_store is not None:
            for group in self.page_store.open_groups():
                if include_same_revision_history:
                    if str(group.revision_id) != current_revision_id:
                        continue
                elif group.milestone_id != str(step["milestone_identity_id"]):
                    continue
                for event in group.events:
                    for fact in event.facts:
                        content = self.page_store.resolve_fact_content(fact)
                        open_evidence.append(
                            {
                                "evidence_id": f"open:{event.event_id}:{fact.key.key_digest}",
                                "evidence_type": fact.key.evidence_type.value,
                                "event_id": event.event_id,
                                "revision_id": event.revision_id or group.revision_id,
                                "content_json": json.dumps(
                                    content,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                                "canonical_entity_id": fact.key.canonical_entity_id,
                                "semantic_role": fact.key.semantic_role,
                                "valid_to_cursor": None,
                            }
                        )
        # Sealed evidence is newest-first. Open groups are chronological, so
        # reverse them and place them first: an unsealed verifier result is the
        # newest authoritative observation even before the Page reaches min size.
        evidence = [*reversed(open_evidence), *evidence]
        return current_revision_id, evidence

    def assess_milestone_facts(
        self,
        run_id: str,
        canonical_id: str,
        *,
        allow_cross_milestone_reuse: bool = False,
    ) -> StepVerificationReceipt:
        """Assess factual readiness before model requirement review.

        This does not change route state. It verifies that the requested code,
        test, process, or benchmark facts exist at the current revision. The
        model then judges whether those facts actually cover the original
        requirement; its review becomes separate durable Evidence.
        """

        current = self.registry.current(run_id)
        if current.canonical_id != canonical_id:
            raise ValueError("fact readiness can assess only the current Milestone")
        revision_id, evidence = self._step_evidence_scope(
            run_id,
            {"milestone_identity_id": current.identity_id},
            include_same_revision_history=allow_cross_milestone_reuse,
        )
        criteria = tuple(
            item
            for item in self.registry.completion_criteria(run_id, canonical_id)
            if bool(item.get("required", True))
            # Semantic requirements are exactly what the model review decides;
            # they are never a factual precondition of that review.
            and criterion_verification_mode(item) is not CriterionVerificationMode.SEMANTIC
        )
        verdicts = tuple(
            self._evaluate_criterion(
                evidence,
                criterion,
                current_revision_id=revision_id,
                require_requirement_review=False,
            )
            for criterion in criteria
        )
        unmet = tuple(item.criterion_id for item in verdicts if not item.satisfied)
        return StepVerificationReceipt(
            step_id=f"milestone:{canonical_id}",
            satisfied=not unmet,
            unmet_criteria=unmet,
            missing_evidence_types={
                item.criterion_id: item.missing_evidence_types
                for item in verdicts
                if not item.satisfied
            },
            evidence_event_ids=tuple(
                dict.fromkeys(
                    event_id
                    for item in verdicts
                    if item.satisfied
                    for event_id in item.evidence_event_ids
                )
            ),
            evidence_rejection_reasons={
                item.criterion_id: item.evidence_rejection_reasons
                for item in verdicts
                if item.evidence_rejection_reasons
            },
            failed_criteria=tuple(
                item.criterion_id for item in verdicts if item.failed_evidence_event_ids
            ),
            failed_evidence_event_ids=tuple(
                dict.fromkeys(
                    event_id
                    for item in verdicts
                    for event_id in item.failed_evidence_event_ids
                )
            ),
            assessed_criteria=tuple(item.criterion_id for item in verdicts),
            bound_evidence_digest=self._bound_evidence_digest(
                [row for item in verdicts for row in item.matching]
            ),
        )

    def verify_claimed_milestones(
        self,
        run_id: str,
        *,
        accept_unverified_semantic: bool = False,
        accept_unverified_missing: bool = False,
        allow_cross_milestone_reuse: bool = False,
        protect_host_verifier_missing: bool = False,
    ) -> MilestoneVerificationBatch:
        """Verify every claimed Milestone criterion against current-revision Evidence.

        Executable and composite criteria decide the route.  Semantic criteria
        are decided by a bounded model review; when ``accept_unverified_semantic``
        is set (review budget exhausted) a Milestone whose only remaining gap is
        semantic is accepted and those criteria are recorded as UNVERIFIED.

        ``accept_unverified_missing`` is a benchmark-runtime compatibility mode:
        it advances a claimed Milestone whose only deficit is absent evidence,
        while preserving every unmet Criterion as UNVERIFIED.  It never accepts
        an observed deterministic failure and is disabled by default, so the
        Python runtime and existing receipts retain their original behaviour.

        ``protect_host_verifier_missing`` narrows that compatibility mode: a
        criterion bound to the runtime-owned host verifier (``VERIFIER_RESULT``
        or the ``verify_current_milestone`` selector) is never accepted merely
        because its observation is absent.  A configured regression guard must
        actually run before its Milestone advances.
        """

        connection = self.database.connection
        has_revision_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='v2_current_workspace_revision'"
        ).fetchone()
        if has_revision_table is None:
            scope = connection.execute(
                "SELECT branch_id,initial_revision_id,NULL current_revision "
                "FROM v2_tasks WHERE run_id=?",
                (run_id,),
            ).fetchone()
        else:
            scope = connection.execute(
                "SELECT t.branch_id,t.initial_revision_id,"
                "(SELECT revision_id FROM v2_current_workspace_revision wr "
                " WHERE wr.run_id=t.run_id AND wr.branch_id=t.branch_id) current_revision "
                "FROM v2_tasks t WHERE t.run_id=?",
                (run_id,),
            ).fetchone()
        if scope is None:
            raise KeyError(run_id)
        revision_id = str(scope["current_revision"] or scope["initial_revision_id"])
        rows = connection.execute(
            """SELECT mi.identity_id,mi.canonical_id,mse.status,mse.source_event_id
               FROM v2_plan_milestones pm
               JOIN v2_milestone_identities mi ON mi.identity_id=pm.identity_id
               JOIN v2_milestone_state_events mse ON mse.identity_id=mi.identity_id
               WHERE pm.plan_version_id=(
                   SELECT pv.plan_version_id FROM v2_plan_versions pv
                   JOIN v2_tasks t ON t.task_id=pv.task_id WHERE t.run_id=?
                   ORDER BY pv.version_number DESC LIMIT 1
               ) AND mse.created_cursor=(
                   SELECT MAX(newer.created_cursor) FROM v2_milestone_state_events newer
                   WHERE newer.identity_id=mi.identity_id)
               ORDER BY pm.ordinal""",
            (run_id,),
        ).fetchall()
        verified = 0
        failed = 0
        verified_ids: list[str] = []
        failed_ids: list[str] = []
        unmet_by_milestone: dict[str, tuple[str, ...]] = {}
        missing_by_milestone: dict[str, dict[str, tuple[str, ...]]] = {}
        rejections_by_milestone: dict[str, dict[str, tuple[str, ...]]] = {}
        satisfied_event_ids: dict[str, tuple[str, ...]] = {}
        failed_event_ids: dict[str, tuple[str, ...]] = {}
        failed_criteria_by_milestone: dict[str, tuple[str, ...]] = {}
        failure_signatures: dict[str, str] = {}
        blocking_unmet_by_milestone: dict[str, tuple[str, ...]] = {}
        semantic_unmet_by_milestone: dict[str, tuple[str, ...]] = {}
        unverified_by_milestone: dict[str, tuple[str, ...]] = {}
        bound_digests: dict[str, str] = {}
        last_source: str | None = None
        last_revision: str | None = None
        for row in rows:
            status = str(row["status"])
            if status != MilestoneStatus.COMPLETED_CLAIMED.value:
                continue
            canonical_id = str(row["canonical_id"])
            criteria = self.registry.completion_criteria(run_id, canonical_id)
            required_criteria = [item for item in criteria if bool(item["required"])]
            # Acceptance must observe the same durable WAL-backed facts whether
            # or not the current semantic Page has reached its sealing boundary.
            # Page finalization controls external storage layout, not when a
            # fact becomes eligible for the authoritative route transaction.
            scoped_revision, evidence = self._step_evidence_scope(
                run_id,
                {"milestone_identity_id": str(row["identity_id"])},
                include_same_revision_history=allow_cross_milestone_reuse,
            )
            revision_id = scoped_revision
            satisfied: dict[str, _CriterionVerdict] = {}
            missing_for_criteria: dict[str, tuple[str, ...]] = {}
            rejection_reasons_for_criteria: dict[str, tuple[str, ...]] = {}
            failure_events: list[str] = []
            failure_criteria: list[str] = []
            semantic_unmet: list[str] = []
            blocking_unmet: list[str] = []
            bound_rows: list[object] = []
            for criterion in required_criteria:
                criterion_id = str(criterion["criterion_id"])
                verdict = self._evaluate_criterion(
                    evidence,
                    criterion,
                    current_revision_id=revision_id,
                    # Milestone acceptance is a trusted runtime fold over
                    # WAL-backed evidence.  Requirement Review is retained
                    # for explicit/manual review APIs, but must never gate the
                    # normal execution boundary.
                    require_requirement_review=False,
                )
                bound_rows.extend(verdict.matching)
                if verdict.satisfied:
                    satisfied[criterion_id] = verdict
                else:
                    missing_for_criteria[criterion_id] = verdict.missing_evidence_types
                    if verdict.evidence_rejection_reasons:
                        rejection_reasons_for_criteria[criterion_id] = (
                            verdict.evidence_rejection_reasons
                        )
                    if criterion_verification_mode(criterion) is (
                        CriterionVerificationMode.SEMANTIC
                    ):
                        semantic_unmet.append(criterion_id)
                    else:
                        blocking_unmet.append(criterion_id)
                if verdict.failed_evidence_event_ids:
                    failure_criteria.append(criterion_id)
                    failure_events.extend(verdict.failed_evidence_event_ids)
            bound_digests[canonical_id] = self._bound_evidence_digest(bound_rows)
            unmet = tuple(
                str(item["criterion_id"])
                for item in required_criteria
                if str(item["criterion_id"]) not in satisfied
            )
            unique_failure_events = tuple(dict.fromkeys(failure_events))
            unverified: tuple[str, ...] = ()
            if (
                unmet
                and accept_unverified_semantic
                and not blocking_unmet
                and not unique_failure_events
            ):
                # Every executable contract holds and the only remaining gap is
                # semantic with an exhausted review budget: the route continues
                # and the receipt names exactly what was not verified.
                unverified = tuple(semantic_unmet)
                unmet = ()
            elif unmet and accept_unverified_missing and not unique_failure_events:
                # A missing observation is not a failed target predicate.  The
                # SWE-Milestone navigation runtime records the gap and advances;
                # a later official evaluator remains the correctness authority.
                criteria_by_id = {
                    str(item["criterion_id"]): item for item in required_criteria
                }
                protected = tuple(
                    criterion_id
                    for criterion_id in unmet
                    if protect_host_verifier_missing
                    and _requires_host_verifier(criteria_by_id[criterion_id])
                )
                if protected:
                    # The regression guard has not produced an observation at
                    # this revision.  Everything else may advance as UNVERIFIED,
                    # but the host-managed contract keeps the Milestone open.
                    unverified = tuple(item for item in unmet if item not in protected)
                    unmet = protected
                    blocking_unmet[:] = [item for item in blocking_unmet if item in protected]
                    semantic_unmet[:] = [item for item in semantic_unmet if item in protected]
                    self.trace.record(
                        "MILESTONE_HOST_VERIFIER_EVIDENCE_REQUIRED",
                        canonical_id=canonical_id,
                        protected_criterion_ids=list(protected),
                        revision_id=revision_id,
                    )
                else:
                    unverified = tuple(unmet)
                    unmet = ()
                    blocking_unmet.clear()
                    semantic_unmet.clear()
            if unmet:
                unmet_by_milestone[canonical_id] = unmet
                missing_by_milestone[canonical_id] = {
                    criterion_id: missing_for_criteria[criterion_id] for criterion_id in unmet
                }
                rejected = {
                    criterion_id: rejection_reasons_for_criteria[criterion_id]
                    for criterion_id in unmet
                    if criterion_id in rejection_reasons_for_criteria
                }
                if rejected:
                    rejections_by_milestone[canonical_id] = rejected
                if blocking_unmet:
                    blocking_unmet_by_milestone[canonical_id] = tuple(blocking_unmet)
                if semantic_unmet:
                    semantic_unmet_by_milestone[canonical_id] = tuple(semantic_unmet)
            successful_events = tuple(
                dict.fromkeys(
                    event_id
                    for verdict in satisfied.values()
                    for event_id in verdict.evidence_event_ids
                )
            )
            if not successful_events and unverified and not unmet:
                successful_events = self._latest_scope_event_ids(evidence)
                if not successful_events:
                    successful_events = (str(row["source_event_id"]),)
            selected_event_id = (
                unique_failure_events[0]
                if unique_failure_events
                else (successful_events[0] if not unmet and successful_events else None)
            )
            if selected_event_id is None:
                self.trace.record(
                    "MILESTONE_VERIFICATION_DEFERRED",
                    canonical_id=canonical_id,
                    reason="UNMET_COMPLETION_CRITERIA",
                    unmet_criteria=list(unmet),
                    blocking_unmet_criteria=list(blocking_unmet),
                    semantic_unmet_criteria=list(semantic_unmet),
                    evidence_rejection_reasons={
                        criterion_id: list(reasons)
                        for criterion_id, reasons in rejection_reasons_for_criteria.items()
                    },
                    revision_id=revision_id,
                )
                continue
            target = (
                MilestoneStatus.VERIFICATION_FAILED
                if unique_failure_events
                else MilestoneStatus.COMPLETED_VERIFIED
            )
            if (
                target is MilestoneStatus.COMPLETED_VERIFIED
                and self.official_submit_gate is not None
            ):
                gate_reason = self.official_submit_gate(canonical_id)
                if gate_reason:
                    # RELEASE means available, never completed: the official
                    # tag is the only submission and the official evaluator is
                    # the only verifier.  Keep the Milestone open with an
                    # explicit unmet contract instead of accepting it.
                    self.trace.record(
                        "MILESTONE_OFFICIAL_TAG_REQUIRED",
                        canonical_id=canonical_id,
                        reason=gate_reason,
                        revision_id=revision_id,
                    )
                    unmet_ids = tuple(
                        str(item["criterion_id"]) for item in required_criteria
                    )
                    unmet_by_milestone[canonical_id] = unmet_ids
                    blocking_unmet_by_milestone[canonical_id] = unmet_ids
                    continue
            failed_criterion_ids = tuple(dict.fromkeys(failure_criteria))
            failure_signature = (
                self._failure_signature(
                    revision_id=revision_id,
                    criterion_ids=failed_criterion_ids,
                    evidence=[
                        item
                        for item in evidence
                        if str(item["event_id"]) in set(unique_failure_events)
                    ],
                )
                if unique_failure_events
                else None
            )
            self.registry.commit_milestone_acceptance(
                run_id=run_id,
                canonical_id=canonical_id,
                verdict=target,
                # Older entity-scoped facts can remain valid across an unrelated
                # workspace revision. The completion state nevertheless describes
                # the current workspace snapshot on which the full contract passed.
                revision_id=revision_id,
                source_event_id=selected_event_id,
                satisfied_criterion_ids=tuple(satisfied),
                failed_criterion_ids=failed_criterion_ids,
                evidence_event_ids=(
                    unique_failure_events
                    if target is MilestoneStatus.VERIFICATION_FAILED
                    else successful_events
                ),
                failure_signature=failure_signature,
                unverified_criterion_ids=(
                    unverified if target is MilestoneStatus.COMPLETED_VERIFIED else ()
                ),
            )
            last_source = selected_event_id
            last_revision = revision_id
            if target is MilestoneStatus.COMPLETED_VERIFIED:
                verified += 1
                verified_ids.append(canonical_id)
                unmet_by_milestone.pop(canonical_id, None)
                missing_by_milestone.pop(canonical_id, None)
                rejections_by_milestone.pop(canonical_id, None)
                blocking_unmet_by_milestone.pop(canonical_id, None)
                semantic_unmet_by_milestone.pop(canonical_id, None)
                satisfied_event_ids[canonical_id] = successful_events
                if unverified:
                    unverified_by_milestone[canonical_id] = unverified
                    self.trace.record(
                        "MILESTONE_ACCEPTED_WITH_UNVERIFIED_SEMANTIC_CRITERIA",
                        canonical_id=canonical_id,
                        unverified_criterion_ids=list(unverified),
                        revision_id=revision_id,
                        route_blocked=False,
                    )
            else:
                failed += 1
                failed_ids.append(canonical_id)
                failed_event_ids[canonical_id] = unique_failure_events
                failed_criteria_by_milestone[canonical_id] = failed_criterion_ids
                if failure_signature is not None:
                    failure_signatures[canonical_id] = failure_signature
            self.trace.record(
                "MILESTONE_VERIFICATION_PROJECTED",
                canonical_id=canonical_id,
                status=target.value,
                source_event_id=selected_event_id,
                criterion_ids=list(satisfied),
                unmet_criteria=list(unmet),
                revision_id=revision_id,
                revalidation=False,
            )

        if last_source is not None and last_revision is not None:
            current_task = self.registry.task_status(run_id)
            if current_task is TaskStatus.EXECUTING:
                self.registry.record_task_state(
                    run_id=run_id,
                    status=TaskStatus.VERIFYING,
                    revision_id=last_revision,
                    source_event_id=last_source,
                )
            remaining = connection.execute(
                """SELECT COUNT(*) FROM v2_plan_milestones pm
                   JOIN v2_milestone_identities mi ON mi.identity_id=pm.identity_id
                   WHERE pm.plan_version_id=(
                     SELECT pv.plan_version_id FROM v2_plan_versions pv
                     JOIN v2_tasks t ON t.task_id=pv.task_id WHERE t.run_id=?
                     ORDER BY pv.version_number DESC LIMIT 1
                   ) AND (
                     SELECT mse.status FROM v2_milestone_state_events mse
                     WHERE mse.identity_id=mi.identity_id
                     ORDER BY mse.created_cursor DESC LIMIT 1
                   ) <> 'COMPLETED_VERIFIED'""",
                (run_id,),
            ).fetchone()[0]
            if int(remaining) != 0:
                self.registry.record_task_state(
                    run_id=run_id,
                    status=TaskStatus.EXECUTING,
                    revision_id=last_revision,
                    source_event_id=last_source,
                )
        return MilestoneVerificationBatch(
            verified_count=verified,
            failed_count=failed,
            verified_canonical_ids=tuple(verified_ids),
            failed_canonical_ids=tuple(failed_ids),
            unmet_criteria=unmet_by_milestone,
            missing_evidence_types=missing_by_milestone,
            evidence_rejection_reasons=rejections_by_milestone,
            satisfied_evidence_event_ids=satisfied_event_ids,
            failed_evidence_event_ids=failed_event_ids,
            failed_criteria=failed_criteria_by_milestone,
            failure_signatures=failure_signatures,
            blocking_unmet_criteria=blocking_unmet_by_milestone,
            semantic_unmet_criteria=semantic_unmet_by_milestone,
            unverified_criteria=unverified_by_milestone,
            bound_evidence_digests=bound_digests,
        )

    @classmethod
    def _bound_evidence_digest(cls, rows: list[object]) -> str:
        """Digest criterion-bound evidence by its normalized observation identity.

        Event IDs, timestamps and duplicated output are excluded: repeating the
        same command with the same result at the same revision produces the
        same digest and therefore no acceptance progress.
        """

        normalized: set[tuple[object, ...]] = set()
        for row in rows:
            content = cls._content(row)
            normalized.add(
                (
                    str(row["evidence_type"]),
                    str(row["canonical_entity_id"]),
                    str(row["revision_id"]),
                    str(content.get("logical_command") or content.get("command_digest") or ""),
                    content.get("success"),
                    content.get("success_exit_status_reliable"),
                    str(content.get("complete_output_digest") or content.get("summary") or ""),
                )
            )
        return digest(sorted(map(repr, normalized)))

    @staticmethod
    def _latest_scope_event_ids(evidence: list[object]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(row["event_id"]) for row in evidence))[:1]

    def milestone_failure_context(
        self,
        run_id: str,
        canonical_id: str,
    ) -> MilestoneFailureContext | None:
        """Read the exact acceptance receipt committed with the failed route state."""

        connection = self.database.connection
        current = self.registry.current(run_id)
        if (
            current.canonical_id != canonical_id
            or current.status != MilestoneStatus.VERIFICATION_FAILED.value
        ):
            return None
        receipt = connection.execute(
            "SELECT revision_id,failed_criterion_ids_json,evidence_event_ids_json,"
            "failure_signature FROM v2_milestone_acceptance_receipts "
            "WHERE run_id=? AND milestone_identity_id=? "
            "AND verdict='VERIFICATION_FAILED' ORDER BY created_cursor DESC LIMIT 1",
            (run_id, current.identity_id),
        ).fetchone()
        if receipt is None or not receipt["failure_signature"]:
            return None
        return MilestoneFailureContext(
            canonical_id=canonical_id,
            revision_id=str(receipt["revision_id"]),
            criterion_ids=tuple(
                map(str, json.loads(str(receipt["failed_criterion_ids_json"])))
            ),
            evidence_event_ids=tuple(
                map(str, json.loads(str(receipt["evidence_event_ids_json"])))
            ),
            signature=str(receipt["failure_signature"]),
        )

    @staticmethod
    def _content(row: object) -> dict[str, object]:
        value = json.loads(str(row["content_json"]))
        return value if isinstance(value, dict) else {}

    @classmethod
    def _evaluate_criterion(
        cls,
        evidence: list[object],
        criterion: dict[str, object],
        *,
        current_revision_id: str | None = None,
        allow_aggregate_reuse: bool = False,
        compatible_criterion_ids: tuple[str, ...] = (),
        direct_evidence_ids: set[str] | None = None,
        require_requirement_review: bool = True,
    ) -> _CriterionVerdict:
        """Apply the one Criterion truth rule used at Step and Milestone scope.

        Evidence is ordered newest-first.  A latest failed result invalidates
        older positive result rows for this Criterion, while durable non-result
        facts such as a current CODE_CHANGE remain eligible.  ``TEST_FAILURE``
        is positive only when the Criterion explicitly asks to reproduce a
        failure; it can never satisfy a terminal success contract by accident.
        """

        required_types = set(map(str, criterion.get("required_evidence_types", ())))
        contract_rejections = criterion_contract_rejections(criterion)
        if contract_rejections:
            return _CriterionVerdict(
                criterion_id=str(criterion.get("criterion_id", "")),
                satisfied=False,
                missing_evidence_types=tuple(sorted(required_types)) or ("BOUND_EVIDENCE",),
                evidence_event_ids=(),
                failed_evidence_event_ids=(),
                evidence_rejection_reasons=(
                    cls.INVALID_BEHAVIORAL_COMMITMENT,
                    *contract_rejections,
                ),
                matching=(),
            )

        selection = cls._select_criterion_evidence(
            evidence,
            criterion,
            current_revision_id=current_revision_id,
            allow_aggregate_reuse=allow_aggregate_reuse,
            compatible_criterion_ids=compatible_criterion_ids,
            direct_evidence_ids=direct_evidence_ids,
        )
        matching = list(selection.matching)
        criterion_id = str(criterion.get("criterion_id", ""))
        commitment_level = CommitmentLevel(
            str(criterion.get("commitment_level", CommitmentLevel.STEP.value))
        )
        requirement_reviews = [
            row
            for row in matching
            if str(row["evidence_type"]) == FactType.REQUIREMENT_REVIEW.value
        ]
        latest_requirement_review = requirement_reviews[0] if requirement_reviews else None
        requirement_review_satisfied = (
            latest_requirement_review is not None
            and cls._evidence_success(latest_requirement_review)
        )
        mode = criterion_verification_mode(criterion)
        if mode is CriterionVerificationMode.SEMANTIC:
            # A semantic requirement is decided only by the bounded model
            # review bound to this Criterion at the current revision.  It has
            # no factual evidence category and it never blocks by itself: the
            # coordinator either requests one review round or reports it as
            # UNVERIFIED when the budget is exhausted.
            if latest_requirement_review is None:
                stale_reviews = [
                    row
                    for row in selection.stale
                    if str(row["evidence_type"]) == FactType.REQUIREMENT_REVIEW.value
                ]
                return _CriterionVerdict(
                    criterion_id=criterion_id,
                    satisfied=False,
                    missing_evidence_types=(FactType.REQUIREMENT_REVIEW.value,),
                    evidence_event_ids=(),
                    failed_evidence_event_ids=(),
                    evidence_rejection_reasons=(
                        (cls.REQUIREMENT_NOT_REVIEWED, cls.STALE_REVISION)
                        if stale_reviews
                        else (cls.REQUIREMENT_NOT_REVIEWED,)
                    ),
                    matching=tuple(matching),
                )
            if requirement_review_satisfied:
                return _CriterionVerdict(
                    criterion_id=criterion_id,
                    satisfied=True,
                    missing_evidence_types=(),
                    evidence_event_ids=(str(latest_requirement_review["event_id"]),),
                    failed_evidence_event_ids=(),
                    evidence_rejection_reasons=(),
                    matching=tuple(matching),
                )
            return _CriterionVerdict(
                criterion_id=criterion_id,
                satisfied=False,
                missing_evidence_types=(),
                evidence_event_ids=(),
                failed_evidence_event_ids=(str(latest_requirement_review["event_id"]),),
                evidence_rejection_reasons=(cls.REQUIREMENT_NOT_SATISFIED,),
                matching=tuple(matching),
            )
        factual_matching = [
            row
            for row in matching
            if str(row["evidence_type"]) != FactType.REQUIREMENT_REVIEW.value
        ]
        latest_observation = next(
            (
                row
                for row in factual_matching
                if str(row["evidence_type"]) in cls._VERIFICATION_OBSERVATION_TYPES
            ),
            None,
        )
        reproduction_contract = FactType.TEST_FAILURE.value in required_types
        latest_observation_satisfies = latest_observation is not None and (
            (
                reproduction_contract
                and str(latest_observation["evidence_type"])
                == FactType.TEST_FAILURE.value
            )
            or cls._evidence_success(latest_observation)
        )
        observation_barrier = (
            latest_observation is not None and not latest_observation_satisfies
        )
        failed_observation = (
            observation_barrier
            and not reproduction_contract
            and cls._observation_failed(latest_observation)
        )
        successful: list[object] = []
        for row in factual_matching:
            evidence_type = str(row["evidence_type"])
            if observation_barrier and evidence_type in cls._VERIFICATION_OBSERVATION_TYPES:
                continue
            if evidence_type == FactType.TEST_FAILURE.value:
                if reproduction_contract:
                    successful.append(row)
                continue
            if cls._evidence_success(row):
                successful.append(row)
        step_observation_required = (
            direct_evidence_ids is not None
            and FactType.CODE_OBSERVATION.value in required_types
        )
        if step_observation_required:
            # A file read is an addressable fact, not a semantic conclusion.
            # INSPECT/PROCESS cursors use the existing bounded semantic update
            # emitted in the same natural Turn; no model Review or extra Turn
            # is introduced.
            successful = [
                row
                for row in successful
                if str(row["evidence_type"]) != FactType.CODE_OBSERVATION.value
                or (
                    str(row["semantic_role"]) == "agent_observation"
                    and cls._content(row).get("authority") == "STRUCTURED_AGENT_OUTPUT"
                )
            ]
        successful_types = {str(row["evidence_type"]) for row in successful}
        if required_types:
            missing = tuple(sorted(required_types.difference(successful_types)))
            factual_satisfied = not missing
        else:
            missing = () if successful else ("BOUND_EVIDENCE",)
            factual_satisfied = bool(successful)
        composite_observation_missing = False
        if mode is CriterionVerificationMode.COMPOSITE and factual_satisfied:
            # A structural requirement is not done because a file changed.  The
            # change must be accompanied by one current observation of the
            # result: either a successful execution receipt in this Milestone
            # (test, verifier, build or import) or the model's structured
            # CODE_OBSERVATION that binds to this Criterion's address at this
            # revision.  "Edited the file" alone can never be the whole proof.
            bound_structured_observation = any(
                str(row["evidence_type"]) == FactType.CODE_OBSERVATION.value
                and str(row["semantic_role"]) == "agent_observation"
                and cls._content(row).get("authority") == "STRUCTURED_AGENT_OUTPUT"
                for row in successful
            )
            composite_observation_missing = (
                not bound_structured_observation
                and not cls._has_current_success_observation(
                    evidence,
                    current_revision_id=current_revision_id,
                )
            )
            if composite_observation_missing:
                missing = (*missing, cls.COMPOSITE_OBSERVATION)
                factual_satisfied = False
        missing_entity_refs: tuple[str, ...] = ()
        if step_observation_required:
            observed_aliases = {
                alias
                for row in successful
                if str(row["evidence_type"]) == FactType.CODE_OBSERVATION.value
                for alias in cls._entity_aliases(str(row["canonical_entity_id"]))
            }
            missing_entity_refs = tuple(
                entity
                for entity in map(str, criterion.get("entity_refs", ()))
                if not cls._entity_address_covered(entity, observed_aliases)
            )
            if missing_entity_refs:
                missing = tuple(dict.fromkeys((*missing, "BOUND_EVIDENCE")))
                factual_satisfied = False
        review_required = (
            require_requirement_review
            and commitment_level is CommitmentLevel.MILESTONE
        )
        if review_required and latest_requirement_review is None:
            missing = (*missing, FactType.REQUIREMENT_REVIEW.value)
        satisfied = factual_satisfied and (
            not review_required or requirement_review_satisfied
        )
        rejected_types = set(missing)
        rejection_reasons: list[str] = []
        if step_observation_required and FactType.CODE_OBSERVATION.value not in successful_types:
            rejection_reasons.append(cls.STEP_SEMANTIC_CONCLUSION_REQUIRED)
        if missing_entity_refs:
            rejection_reasons.append(cls.INCOMPLETE_STEP_ENTITY_COVERAGE)
        if composite_observation_missing:
            rejection_reasons.append(cls.COMPOSITE_OBSERVATION_REQUIRED)
        for row in factual_matching:
            evidence_type = str(row["evidence_type"])
            if evidence_type not in rejected_types and "BOUND_EVIDENCE" not in rejected_types:
                continue
            content = cls._content(row)
            if (
                evidence_type in cls._VERIFICATION_OBSERVATION_TYPES
                and content.get("success") is True
                and content.get("success_exit_status_reliable") is False
            ):
                rejection_reasons.append(cls.UNRELIABLE_EXIT_STATUS)
            elif cls._observation_failed(row) and not reproduction_contract:
                rejection_reasons.append(cls.FAILED_OBSERVATION)
            elif not cls._evidence_success(row):
                rejection_reasons.append(cls.UNPROVEN_SUCCESS)
        if any(
            str(row["evidence_type"]) in rejected_types
            or "BOUND_EVIDENCE" in rejected_types
            for row in selection.stale
        ):
            rejection_reasons.append(cls.STALE_REVISION)
        if any(
            str(row["evidence_type"]) in rejected_types
            or "BOUND_EVIDENCE" in rejected_types
            for row in selection.unbound_direct
        ):
            rejection_reasons.append(cls.UNBOUND_ENTITY_OR_SELECTOR)
        if review_required and latest_requirement_review is None:
            rejection_reasons.append(cls.REQUIREMENT_NOT_REVIEWED)
        elif review_required and not requirement_review_satisfied:
            rejection_reasons.append(cls.REQUIREMENT_NOT_SATISFIED)
        failure_events = tuple(
            dict.fromkeys(
                (
                    *(
                        (str(latest_observation["event_id"]),)
                        if failed_observation
                        else ()
                    ),
                    *(
                        (str(latest_requirement_review["event_id"]),)
                        if review_required
                        and latest_requirement_review is not None
                        and not requirement_review_satisfied
                        else ()
                    ),
                )
            )
        )
        return _CriterionVerdict(
            criterion_id=criterion_id,
            satisfied=satisfied,
            # A negative requirement review is an observed semantic failure,
            # not missing Evidence.  Keeping these states distinct is what
            # makes "continue the current Step" and "diagnose a failed
            # requirement" deterministic without an Oracle state machine.
            missing_evidence_types=(
                missing
                if missing
                else (() if satisfied or failure_events else ("BOUND_EVIDENCE",))
            ),
            evidence_event_ids=tuple(
                dict.fromkeys(str(row["event_id"]) for row in successful)
            ),
            failed_evidence_event_ids=failure_events,
            evidence_rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
            matching=tuple(matching),
        )

    @classmethod
    def _failure_signature(
        cls,
        *,
        revision_id: str,
        criterion_ids: tuple[str, ...],
        evidence: list[object],
    ) -> str:
        signatures: list[object] = []
        for row in evidence:
            content = cls._content(row)
            signatures.append(
                content.get("failure_signature")
                or {
                    "evidence_type": str(row["evidence_type"]),
                    "canonical_entity_id": str(row["canonical_entity_id"]),
                    "content": content,
                }
            )
        return digest(
            {
                "revision_id": revision_id,
                "criterion_ids": sorted(criterion_ids),
                "failures": signatures,
            }
        )

    @classmethod
    def _criterion_matching_evidence(
        cls,
        evidence: list[object],
        criterion: dict[str, object],
        *,
        current_revision_id: str | None = None,
        allow_aggregate_reuse: bool = False,
        compatible_criterion_ids: tuple[str, ...] = (),
        direct_evidence_ids: set[str] | None = None,
    ) -> list[object]:
        """Return admissible rows from the one Criterion Evidence partition."""

        return list(
            cls._select_criterion_evidence(
                evidence,
                criterion,
                current_revision_id=current_revision_id,
                allow_aggregate_reuse=allow_aggregate_reuse,
                compatible_criterion_ids=compatible_criterion_ids,
                direct_evidence_ids=direct_evidence_ids,
            ).matching
        )

    @classmethod
    def _select_criterion_evidence(
        cls,
        evidence: list[object],
        criterion: dict[str, object],
        *,
        current_revision_id: str | None = None,
        allow_aggregate_reuse: bool = False,
        compatible_criterion_ids: tuple[str, ...] = (),
        direct_evidence_ids: set[str] | None = None,
    ) -> _CriterionEvidenceSelection:
        """Partition Evidence once for acceptance and actionable diagnostics."""

        matching: list[object] = []
        stale: list[object] = []
        unbound_direct: list[object] = []
        for row in evidence:
            evidence_id = str(row["evidence_id"])
            directly_eligible = direct_evidence_ids is None or evidence_id in direct_evidence_ids
            if not directly_eligible:
                continue
            bound = cls._evidence_binds(
                row,
                criterion,
                allow_aggregate_reuse=allow_aggregate_reuse,
                compatible_criterion_ids=compatible_criterion_ids,
            )
            if not bound:
                if direct_evidence_ids is not None:
                    unbound_direct.append(row)
                continue
            if current_revision_id is not None and not cls._evidence_is_current_for_acceptance(
                row,
                current_revision_id,
            ):
                stale.append(row)
                continue
            matching.append(row)
        return _CriterionEvidenceSelection(
            matching=tuple(matching),
            stale=tuple(stale),
            unbound_direct=tuple(unbound_direct),
        )

    @classmethod
    def _evidence_binds(
        cls,
        row: object,
        criterion: dict[str, object],
        *,
        allow_aggregate_reuse: bool = False,
        compatible_criterion_ids: tuple[str, ...] = (),
    ) -> bool:
        content = cls._content(row)
        explicit = content.get("criterion_ids", content.get("criterion_id", ()))
        if isinstance(explicit, str):
            explicit_ids = {explicit}
        elif isinstance(explicit, (list, tuple)):
            explicit_ids = set(map(str, explicit))
        else:
            explicit_ids = set()
        criterion_id = str(criterion["criterion_id"])
        task_final = content.get("task_final_criterion_ids", ())
        task_final_ids = (
            {task_final}
            if isinstance(task_final, str)
            else set(map(str, task_final))
            if isinstance(task_final, (list, tuple))
            else set()
        )
        if allow_aggregate_reuse and criterion_id in task_final_ids:
            return True
        if explicit_ids:
            if criterion_id in explicit_ids:
                return True
            if not explicit_ids.intersection(compatible_criterion_ids) and not (
                allow_aggregate_reuse
            ):
                return False
        canonical_entity = str(row["canonical_entity_id"])
        entity_refs = set(map(str, criterion["entity_refs"]))
        entity_aliases = {alias for entity in entity_refs for alias in cls._entity_aliases(entity)}
        canonical_aliases = cls._entity_aliases(canonical_entity)
        if entity_aliases.intersection(canonical_aliases):
            return True
        # A trailing slash is an explicit directory address, not a fuzzy
        # symbol or basename match.  It deterministically owns descendant
        # file Evidence while preserving the exact-match rule for ordinary
        # file refs (``file:pkg/model.py`` must not match a similarly named
        # file elsewhere).  This lets a Plan safely declare a changelog or
        # fixture directory before the concrete output filename exists.
        directory_aliases = {alias for alias in entity_aliases if alias.endswith("/")}
        if any(
            candidate.startswith(directory)
            for directory in directory_aliases
            for candidate in canonical_aliases
        ):
            return True
        selectors = tuple(map(str, criterion["test_selectors"]))
        if selectors and any(
            evidence_result_selector_matches(
                selector,
                canonical_entity_id=canonical_entity,
                content=content,
            )
            for selector in selectors
        ):
            return True
        return False

    @staticmethod
    def _entity_aliases(value: str) -> set[str]:
        """Return only deterministic file-prefix aliases, never fuzzy matches."""

        normalized = value.strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        if not normalized:
            return set()
        aliases = {normalized}
        if normalized.startswith("file:"):
            aliases.add(normalized.removeprefix("file:"))
        elif normalized.startswith("symbol:"):
            symbol_address = normalized.removeprefix("symbol:")
            if ":" in symbol_address:
                path = symbol_address.rsplit(":", 1)[0]
                aliases.update({path, f"file:{path}"})
        elif ":" not in normalized and ("/" in normalized or "." in normalized):
            aliases.add(f"file:{normalized}")
        return aliases

    @classmethod
    def _entity_address_covered(cls, required: str, observed_aliases: set[str]) -> bool:
        required_aliases = cls._entity_aliases(required)
        if required_aliases.intersection(observed_aliases):
            return True
        directories = {alias for alias in required_aliases if alias.endswith("/")}
        return any(
            observed.startswith(directory)
            for directory in directories
            for observed in observed_aliases
        )

    @classmethod
    def _has_explicit_criterion_binding(cls, row: object, criterion_id: str) -> bool:
        content = cls._content(row)
        explicit = content.get("criterion_ids", content.get("criterion_id", ()))
        if isinstance(explicit, str):
            return explicit == criterion_id
        if isinstance(explicit, (list, tuple)):
            return criterion_id in set(map(str, explicit))
        return False

    @classmethod
    def _evidence_success(cls, row: object) -> bool:
        evidence_type = str(row["evidence_type"])
        if evidence_type == "TEST_FAILURE":
            return False
        content = cls._content(row)
        if content.get("authority") == "OBSERVED_AGENT_MESSAGE":
            # A free-form progress report is an assertion, never completion
            # proof.  Structured facts emitted beside it remain independently
            # eligible through their own Evidence row.
            return False
        if evidence_type in {
            "TEST_RESULT",
            "VERIFIER_RESULT",
            "TOOL_RESULT",
            "REQUIREMENT_REVIEW",
        }:
            return content.get("success") is True and (
                content.get("success_exit_status_reliable") is not False
            )
        return True

    @classmethod
    def _has_current_success_observation(
        cls,
        evidence: list[object],
        *,
        current_revision_id: str | None,
    ) -> bool:
        """Whether the scope holds one reliable successful execution receipt now."""

        for row in evidence:
            evidence_type = str(row["evidence_type"])
            if evidence_type not in {"TEST_RESULT", "VERIFIER_RESULT", "TOOL_RESULT"}:
                continue
            if current_revision_id is not None and str(row["revision_id"]) != current_revision_id:
                continue
            if cls._evidence_success(row):
                return True
        return False

    @classmethod
    def _observation_failed(cls, row: object) -> bool:
        """Distinguish a failed result from an inadmissible success claim.

        A successful test hidden behind a downstream shell pipeline has an
        unreliable process status.  It must not satisfy acceptance, but it is
        also not proof that the implementation failed and therefore must not
        authorize a Corrective Step.
        """

        if str(row["evidence_type"]) == FactType.TEST_FAILURE.value:
            return True
        return cls._content(row).get("success") is False

    @classmethod
    def _evidence_is_current_for_acceptance(
        cls,
        row: object,
        current_revision_id: str,
    ) -> bool:
        """Require physical verification outcomes to describe the current tree.

        Durable implementation facts may remain valid across unrelated
        revisions until Semantic Store closes them.  Test/tool/verifier
        outcomes are snapshot claims and must never be carried forward merely
        because their Evidence row is still addressable as execution history.
        """

        evidence_type = str(row["evidence_type"])
        return (
            evidence_type not in cls._CURRENT_REVISION_EVIDENCE_TYPES
            or str(row["revision_id"]) == current_revision_id
        )

    def _unmet_final_acceptance(
        self,
        run_id: str,
        *,
        branch_id: str,
        revision_id: str | None,
        unverified_criterion_ids: Sequence[str] = (),
    ) -> tuple[str, ...]:
        """Evaluate the live Plan's repository-level final contract.

        Semantic requirements that a COMPLETED_VERIFIED Milestone accepted as
        UNVERIFIED are disclosed separately by the caller; they are not counted
        again here, otherwise an unbounded review loop would simply move from
        the Milestone boundary to the final verdict.
        """

        plan = self.registry.active_plan(run_id)
        unverified = set(map(str, unverified_criterion_ids))
        required = [
            item
            for item in plan.final_acceptance
            if item.required
            and not (
                item.criterion_id in unverified
                and item.verification_mode is CriterionVerificationMode.SEMANTIC
            )
        ]
        if not required:
            # Older offline/imported plans predate the typed final contract.
            # Production Codex plans are rejected by the normalizer without it.
            return ()
        if not revision_id:
            return tuple(item.criterion_id for item in required)
        evidence = self.database.connection.execute(
            """SELECT e.evidence_id,e.evidence_type,e.event_id,e.revision_id,e.content_json,
                      e.canonical_entity_id,e.semantic_role
               FROM v2_semantic_evidence e
               JOIN v2_semantic_events ev ON ev.event_id=e.event_id
               WHERE e.run_id=? AND e.branch_id=? AND e.valid_to_cursor IS NULL
               ORDER BY ev.observed_at DESC,ev.event_position DESC,e.evidence_id DESC""",
            (run_id, branch_id),
        ).fetchall()
        unmet: list[str] = []
        for item in required:
            criterion = {
                "criterion_id": item.criterion_id,
                "requirement_id": item.requirement_id,
                "requirement_text": item.requirement_text,
                "claim_type": item.claim_type.value,
                "commitment_level": item.commitment_level.value,
                "required_evidence_types": tuple(
                    value.value for value in item.required_evidence_types
                ),
                "entity_refs": item.entity_refs,
                "test_selectors": item.test_selectors,
                "verification_mode": item.verification_mode.value,
            }
            verdict = self._evaluate_criterion(
                list(evidence),
                criterion,
                current_revision_id=revision_id,
                allow_aggregate_reuse=True,
                # Final evaluation is the external grader over the current
                # workspace. Milestone requirement review has already governed
                # route advancement and must not become a second final-review
                # protocol.
                require_requirement_review=False,
            )
            if not verdict.satisfied:
                unmet.append(item.criterion_id)
        return tuple(unmet)
