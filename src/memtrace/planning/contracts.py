from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..contracts import MilestoneSpec, MilestoneStatus, PlanSpec, PlanStepStatus, TaskStatus


class PlanObservationKind(StrEnum):
    SNAPSHOT = "SNAPSHOT"
    PATCH = "PATCH"


class PlanCoverage(StrEnum):
    COMPLETE = "COMPLETE"
    UNKNOWN = "UNKNOWN"


class RouteTransitionKind(StrEnum):
    """The two map positions after observing a contiguous Step span.

    These values describe logical TPG navigation only.  Neither outcome owns
    Provider Turn control or Milestone correctness.
    """

    STEP_ADVANCED = "STEP_ADVANCED"
    MILESTONE_EXECUTION_BOUNDARY = "MILESTONE_EXECUTION_BOUNDARY"


@dataclass(frozen=True, slots=True)
class RouteTransitionReceipt:
    """Semantic result of one WAL-backed route-observation transaction.

    Provider Turn identity is deliberately absent.  It is control provenance,
    not part of the Harness-independent TPG route address.
    """

    transition_kind: RouteTransitionKind
    milestone_id: str
    completed_step_id: str
    next_step_id: str | None
    source_event_id: str
    covered_step_ids: tuple[str, ...] = ()

    @property
    def committed_step_ids(self) -> tuple[str, ...]:
        return self.covered_step_ids or (self.completed_step_id,)

    @property
    def same_turn(self) -> bool:
        # Exhausting the planned Step list is not a physical context boundary.
        # The model keeps working until its natural Turn ends; only then does
        # the Milestone acceptance kernel run.
        return True

    @property
    def turn_fence_required(self) -> bool:
        # Kept as a compatibility property for old readers.  Step navigation
        # must never request, persist, or recover a Provider Turn fence.
        return False

    def semantic_payload(self) -> dict[str, object]:
        return {
            "transition_kind": self.transition_kind.value,
            "milestone_id": self.milestone_id,
            "completed_step_id": self.completed_step_id,
            "covered_step_ids": list(self.committed_step_ids),
            "next_step_id": self.next_step_id,
            "source_event_id": self.source_event_id,
        }


@dataclass(frozen=True, slots=True)
class MilestoneStateEvent:
    state_event_id: str
    run_id: str
    identity_id: str
    previous_status: MilestoneStatus | None
    status: MilestoneStatus
    source_event_id: str
    revision_id: str
    cursor: int


@dataclass(frozen=True, slots=True)
class TaskStateEvent:
    state_event_id: str
    run_id: str
    previous_status: TaskStatus | None
    status: TaskStatus
    source_event_id: str
    revision_id: str
    cursor: int


@dataclass(frozen=True, slots=True)
class UnresolvedMilestoneObservation:
    observation_id: str
    run_id: str
    source_event_id: str
    title: str
    reason: str
    candidates: tuple[str, ...]
    cursor: int


@dataclass(frozen=True, slots=True)
class MilestoneProjectionRecord:
    identity_id: str
    version_id: str
    version_number: int
    ordinal: int
    spec: MilestoneSpec
    previous_version_id: str | None
    created_new_version: bool


@dataclass(frozen=True, slots=True)
class PlanProjectionInput:
    repository_id: str
    run_id: str
    branch_id: str
    revision_id: str
    task_id: str
    goal_id: str
    plan_version_id: str
    plan_version_number: int
    previous_plan_version_id: str | None
    source_event_id: str
    cursor: int
    plan: PlanSpec
    milestones: tuple[MilestoneProjectionRecord, ...]


@dataclass(frozen=True, slots=True)
class PlanApplication:
    task_id: str
    goal_id: str
    plan_version_id: str
    plan_version_number: int
    current_milestone_id: str
    milestone_identity_ids: tuple[str, ...]
    created_new_version: bool
    cursor: int


@dataclass(frozen=True, slots=True)
class PlanObservationResult:
    application: PlanApplication | None
    unresolved: tuple[UnresolvedMilestoneObservation, ...]
    observation_kind: PlanObservationKind
    coverage: PlanCoverage
    superseded: bool


@dataclass(frozen=True, slots=True)
class PlanStepProgressClaim:
    """A native Harness Plan observation interpreted against the live route.

    ``completed_span`` is either empty or a contiguous prefix beginning at the
    authoritative current Step.  It is a navigation observation from the
    native Harness Plan, not a correctness claim.  Milestone acceptance later
    evaluates the durable workspace facts independently.
    """

    owner_step_id: str | None
    completed_span: tuple[str, ...]
    observed_statuses: tuple[tuple[str, PlanStepStatus], ...]
    unresolved: tuple[tuple[str, str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class CurrentMilestone:
    run_id: str
    plan_version_id: str
    identity_id: str
    canonical_id: str
    version_id: str
    title: str
    status: str
    source_event_id: str
    valid_from_revision: str
    valid_from_cursor: int


class AcceptanceProgressClass(StrEnum):
    """Semantic progress observed between two acceptance boundaries.

    ``STRONG`` means the unmet gap shrank or a previously unmet Criterion
    received new bound evidence.  ``WEAK`` means the repository or bound
    evidence changed without shrinking the gap.  ``NONE`` means the boundary is
    indistinguishable from the previous one, which is the durable livelock
    signal.  ``INITIAL`` is the first boundary of a Milestone acceptance.
    """

    INITIAL = "INITIAL"
    STRONG = "STRONG"
    WEAK = "WEAK"
    NONE = "NONE"


@dataclass(frozen=True, slots=True)
class AcceptanceProgress:
    observation_id: str
    milestone_identity_id: str
    gap_digest: str
    bound_evidence_digest: str
    scoped_revision_digest: str
    progress_class: AcceptanceProgressClass
    weak_streak: int
    none_streak: int
    boundary_count: int
    unmet_criterion_ids: tuple[str, ...]
    created: bool


@dataclass(frozen=True, slots=True)
class ContractFreezeReceipt:
    freeze_id: str
    milestone_identity_id: str
    canonical_id: str
    previous_plan_version_id: str
    resulting_plan_version_id: str
    previous_milestone_version_id: str
    resulting_milestone_version_id: str
    address_resolution: dict[str, object]
    created: bool


@dataclass(frozen=True, slots=True)
class WorkingSetRoot:
    identity_id: str
    canonical_id: str
    state: str
    reason: str
    ordinal: int
    source_event_id: str
    plan_version_id: str
