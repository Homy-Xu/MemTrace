from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Iterable, Mapping

# Confirmed execution-flow edges in the Semantic Page Table.  These are
# intentionally distinct from code-structure relations such as CALLS/IMPORTS,
# which belong to the optional Rich Code Graph.
SEMANTIC_PAGE_RELATIONS = frozenset(
    {
        "ADVANCES_TO",
        "CONTINUES_WITH",
        "CONTAINS_PAGE",
        "CORRECTED_BY",
        "VERIFIED_BY",
        "SUPERSEDED_BY",
        "DEPENDED_ON_BY",
        "BRANCHES_TO",
        "RESOLVED_BY",
    }
)
SEMANTIC_PAGE_QUERY_RELATIONS = SEMANTIC_PAGE_RELATIONS.difference({"CONTAINS_PAGE"})


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def primitive(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return primitive(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [primitive(item) for item in value]
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def stable_id(prefix: str, value: Any) -> str:
    return prefix + digest(value).removeprefix("sha256:")[:32]


class Authority(StrEnum):
    ASSERTED = "ASSERTED"
    DERIVED = "DERIVED"
    INFERRED = "INFERRED"


class NodeType(StrEnum):
    RUN = "Run"
    TASK = "Task"
    GOAL = "Goal"
    PLAN_VERSION = "PlanVersion"
    NATIVE_PLAN_ITEM = "NativePlanItem"
    MILESTONE_IDENTITY = "MilestoneIdentity"
    MILESTONE_VERSION = "MilestoneVersion"
    PLAN_STEP = "PlanStep"
    COMPLETION_CRITERION = "CompletionCriterion"
    VERIFICATION_RESULT = "VerificationResult"
    MILESTONE_REVIEW = "MilestoneReviewEvent"
    WORKSPACE_REVISION = "WorkspaceRevision"
    EVENT_GROUP = "EventGroup"
    EVENT = "Event"
    EVIDENCE_UNIT = "EvidenceUnit"
    SEMANTIC_ANCHOR = "SemanticAnchor"
    PAGE = "Page"
    FILE_REFERENCE = "FileReference"
    SYMBOL_REFERENCE = "SymbolReference"
    TEST_REFERENCE = "TestReference"
    FAILURE_REFERENCE = "FailureReference"
    CHANGE_REFERENCE = "ChangeReference"
    PAYLOAD_HANDLE = "PayloadHandle"
    BLOB = "Blob"
    CONTEXT_ARTIFACT = "ContextArtifact"
    FILE_VERSION = "FileVersion"
    SYMBOL_VERSION = "SymbolVersion"
    TEST = "Test"
    DIAGNOSTIC = "Diagnostic"
    CHANGE_SET = "ChangeSet"


class FactType(StrEnum):
    USER_CONSTRAINT = "USER_CONSTRAINT"
    PLAN_DECISION = "PLAN_DECISION"
    MILESTONE_STATE = "MILESTONE_STATE"
    CODE_OBSERVATION = "CODE_OBSERVATION"
    IMPLEMENTATION_DECISION = "IMPLEMENTATION_DECISION"
    CODE_CHANGE = "CODE_CHANGE"
    TOOL_RESULT = "TOOL_RESULT"
    TEST_RESULT = "TEST_RESULT"
    TEST_FAILURE = "TEST_FAILURE"
    VERIFIER_RESULT = "VERIFIER_RESULT"
    REQUIREMENT_REVIEW = "REQUIREMENT_REVIEW"
    UNRESOLVED_QUESTION = "UNRESOLVED_QUESTION"


class ClaimType(StrEnum):
    """Semantic strength of one acceptance claim.

    Evidence types describe what the runtime observed.  Claim types describe
    what that observation is allowed to prove.  Keeping the two dimensions
    separate prevents a file edit from silently proving a behavioural or
    equivalence requirement.
    """

    STRUCTURAL = "STRUCTURAL"
    BEHAVIORAL = "BEHAVIORAL"
    EQUIVALENCE = "EQUIVALENCE"
    INVARIANT = "INVARIANT"
    REGRESSION = "REGRESSION"
    PERFORMANCE = "PERFORMANCE"
    PROCESS = "PROCESS"
    DECISION = "DECISION"
    FAILURE_REPRODUCTION = "FAILURE_REPRODUCTION"


class CriterionVerificationMode(StrEnum):
    """How the runtime decides one Criterion at the Milestone boundary.

    ``EXECUTABLE`` is decided by criterion-bound execution results (tests,
    verifiers, tool receipts).  ``COMPOSITE`` combines a current code change at
    the addressed entity with a current successful observation of the same
    Milestone.  ``SEMANTIC`` cannot be executed and is decided by one bounded
    model requirement review; it never blocks the route indefinitely and is
    reported as UNVERIFIED when the review budget is exhausted.
    """

    EXECUTABLE = "EXECUTABLE"
    COMPOSITE = "COMPOSITE"
    SEMANTIC = "SEMANTIC"


class CommitmentLevel(StrEnum):
    """Lifecycle scope of one requirement-derived behavioral commitment.

    ``DIRECTION`` is the stage-level intent preserved in the initial route.
    ``MILESTONE`` is the active terminal commitment reviewed at the stage
    boundary. ``STEP`` is a local work contract and never proves the whole
    Milestone merely because an action happened.
    """

    DIRECTION = "DIRECTION"
    MILESTONE = "MILESTONE"
    STEP = "STEP"


# One temporal contract is shared by the Semantic Store, acceptance kernel,
# and route revalidation.  Result facts describe a concrete workspace
# snapshot; code facts remain current only while their addressed entities do
# not change.  Decisions and constraints are execution-history facts: they are
# replaced only by an explicit corrective/superseding event, never merely by a
# later workspace revision.
GLOBAL_REVISION_FACT_TYPES = frozenset(
    {
        FactType.TOOL_RESULT,
        FactType.TEST_RESULT,
        FactType.TEST_FAILURE,
        FactType.VERIFIER_RESULT,
        FactType.REQUIREMENT_REVIEW,
    }
)
ENTITY_REVISION_FACT_TYPES = frozenset(
    {
        FactType.CODE_CHANGE,
        FactType.CODE_OBSERVATION,
    }
)
REVISION_SENSITIVE_FACT_TYPES = frozenset(
    (*GLOBAL_REVISION_FACT_TYPES, *ENTITY_REVISION_FACT_TYPES)
)


class PageState(StrEnum):
    OPEN_BELOW_MIN = "OPEN_BELOW_MIN"
    OPEN_NORMAL = "OPEN_NORMAL"
    CLOSE_PENDING = "CLOSE_PENDING"
    SEALED = "SEALED"


class PageKind(StrEnum):
    SEMANTIC = "SEMANTIC_PAGE"
    AUDIT = "AUDIT_PAGE"


class CoverageState(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    EMPTY = "EMPTY"


class RecallTemporalScope(StrEnum):
    """Semantic time intent; revision mechanics remain runtime-private."""

    CURRENT_WORKSPACE_TRUTH = "CURRENT_WORKSPACE_TRUTH"
    HISTORICAL_EXECUTION = "HISTORICAL_EXECUTION"
    COMPARE_HISTORY_TO_CURRENT = "COMPARE_HISTORY_TO_CURRENT"
    MEMORY_REF_DETAIL = "MEMORY_REF_DETAIL"


class Representation(StrEnum):
    FULL = "FULL"
    SEMANTIC_SLICE = "SEMANTIC_SLICE"
    VERIFIED_SUMMARY = "VERIFIED_SUMMARY"
    HANDLE = "HANDLE"
    NONRESIDENT = "NONRESIDENT"


COMPRESSED_MEMORY_REPRESENTATIONS = frozenset(
    {
        Representation.SEMANTIC_SLICE,
        Representation.VERIFIED_SUMMARY,
        Representation.HANDLE,
        Representation.NONRESIDENT,
    }
)


class PressureLevel(StrEnum):
    NORMAL = "NORMAL"
    SOFT = "SOFT"
    URGENT = "URGENT"
    HARD = "HARD"


class DeliveryState(StrEnum):
    PREPARED = "PREPARED"
    TRANSPORT_ACCEPTED = "TRANSPORT_ACCEPTED"
    CONTEXT_COMMITTED = "CONTEXT_COMMITTED"
    MODEL_OBSERVED = "MODEL_OBSERVED"


class RichGraphState(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    BUILDING = "BUILDING"
    READY_PARTIAL = "READY_PARTIAL"
    READY_CURRENT = "READY_CURRENT"
    LAGGED = "LAGGED"
    FAILED = "FAILED"


class EpochReason(StrEnum):
    COMPRESSION_FIXED_POINT = "COMPRESSION_FIXED_POINT"
    PROVIDER_CONTEXT_LIMIT = "PROVIDER_CONTEXT_LIMIT"
    THREAD_UNRECOVERABLE = "THREAD_UNRECOVERABLE"
    SESSION_LOST = "SESSION_LOST"
    CRASH_RESUME_UNAVAILABLE = "CRASH_RESUME_UNAVAILABLE"


class FallbackStage(StrEnum):
    MEMORY_REF_DIRECT = "MEMORY_REF_DIRECT"
    SEMANTIC_EXACT = "SEMANTIC_EXACT"
    STRUCTURED_INDEX = "STRUCTURED_INDEX"
    PAGE_METADATA_FTS = "PAGE_METADATA_FTS"
    RECENT_RELATED_PAGE = "RECENT_RELATED_PAGE"
    LOCAL_REPOSITORY_SEARCH = "LOCAL_REPOSITORY_SEARCH"
    NONE = "NONE"


class TaskStatus(StrEnum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class MilestoneStatus(StrEnum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED_CLAIMED = "COMPLETED_CLAIMED"
    COMPLETED_VERIFIED = "COMPLETED_VERIFIED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    REPAIRING = "REPAIRING"
    # Kept only so old WAL/SQLite histories remain readable. New execution
    # never moves a completed Milestone back into the live route. Current-code
    # validity belongs to the active/final acceptance contract, not to a
    # mutation of an immutable historical Milestone.
    REQUIRES_REVALIDATION = "REQUIRES_REVALIDATION"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class PlanStepStatus(StrEnum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED_CLAIMED = "COMPLETED_CLAIMED"
    # SQLite/WAL histories before the lightweight-route refactor used the
    # value ``COMPLETED_VERIFIED`` for a terminal Step.  A Step is no longer an
    # acceptance authority, so new code uses the semantic alias below while
    # retaining the persisted value for replay compatibility.
    COMPLETED_OBSERVED = "COMPLETED_VERIFIED"
    COMPLETED_VERIFIED = "COMPLETED_VERIFIED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class PlanStepReviewDecision(StrEnum):
    """Legacy replay vocabulary for pre-lightweight-route Step reviews.

    New execution does not expose or consume a Step review tool. The enum stays
    readable so existing WAL and SQLite histories can still be recovered.
    """

    SATISFIED = "SATISFIED"
    CONTINUE = "CONTINUE"
    CORRECT = "CORRECT"
    BLOCKED = "BLOCKED"


class MilestoneReviewDecision(StrEnum):
    CONTINUE = "CONTINUE"
    REPLAN_FUTURE = "REPLAN_FUTURE"
    CORRECT_CURRENT = "CORRECT_CURRENT"
    UPDATE_FUTURE = "UPDATE_FUTURE"
    SPLIT_FUTURE = "SPLIT_FUTURE"
    MERGE_FUTURE = "MERGE_FUTURE"
    CANCEL_FUTURE = "CANCEL_FUTURE"
    REPLACE_FUTURE = "REPLACE_FUTURE"


@dataclass(frozen=True, slots=True)
class NativePlanItemSpec:
    """One item exactly as exposed by the Coding Harness native Plan."""

    source_step_id: str
    ordinal: int
    title: str
    status: str = "pending"

    def __post_init__(self) -> None:
        if not self.source_step_id.strip() or not self.title.strip() or self.ordinal < 1:
            raise ValueError("Native Plan item requires a stable ID, ordinal, and title")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], ordinal: int) -> "NativePlanItemSpec":
        return cls(
            source_step_id=str(value.get("source_step_id") or f"N{ordinal:03d}"),
            ordinal=int(value.get("ordinal", ordinal)),
            title=str(value.get("title") or value.get("step") or "").strip(),
            status=str(value.get("status", "pending")),
        )


@dataclass(frozen=True, slots=True)
class NativePlanSnapshot:
    """Immutable planning-source snapshot preserved before Milestone projection."""

    items: tuple[NativePlanItemSpec, ...]
    final_text: str = ""

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("Native Plan snapshot requires at least one Plan item")
        ids = tuple(item.source_step_id for item in self.items)
        if len(ids) != len(set(ids)):
            raise ValueError("Native Plan source IDs must be unique")

    @property
    def snapshot_digest(self) -> str:
        return digest(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NativePlanSnapshot":
        raw_items = value.get("items", ())
        if not isinstance(raw_items, (list, tuple)):
            raise TypeError("Native Plan items must be an array")
        return cls(
            items=tuple(
                NativePlanItemSpec.from_dict(item, ordinal)
                for ordinal, item in enumerate(raw_items, start=1)
                if isinstance(item, Mapping)
            ),
            final_text=str(value.get("final_text", "")),
        )


def _legacy_claim_type(evidence_types: tuple[FactType, ...]) -> ClaimType:
    selected = set(evidence_types)
    if FactType.TEST_FAILURE in selected:
        return ClaimType.FAILURE_REPRODUCTION
    if FactType.VERIFIER_RESULT in selected or FactType.TEST_RESULT in selected:
        return ClaimType.BEHAVIORAL
    if FactType.CODE_CHANGE in selected or FactType.CODE_OBSERVATION in selected:
        return ClaimType.STRUCTURAL
    if FactType.TOOL_RESULT in selected:
        return ClaimType.PROCESS
    if FactType.IMPLEMENTATION_DECISION in selected:
        return ClaimType.DECISION
    return ClaimType.STRUCTURAL


TASK_FINAL_REQUIREMENT_ID = "TASK.FINAL"


def default_verification_mode(
    claim_type: ClaimType,
    *,
    requirement_id: str = "",
    required_evidence_types: Iterable[FactType] = (),
    commitment_level: CommitmentLevel = CommitmentLevel.STEP,
) -> CriterionVerificationMode:
    """Derive the boundary decision procedure for a Criterion without one.

    The Task-final check compares the whole repository with the immutable Task
    text; no single execution result decides it, so it is semantic.  A
    structural claim whose only evidence is ``CODE_CHANGE`` is composite: the
    change must exist and the Milestone must also hold a current successful
    observation, otherwise "edited a file" would count as done.  Everything
    else is decided by criterion-bound execution results.
    """

    if requirement_id == TASK_FINAL_REQUIREMENT_ID:
        return CriterionVerificationMode.SEMANTIC
    selected = set(required_evidence_types)
    if claim_type is ClaimType.DECISION and not selected:
        # A decision that names no recordable fact can only be judged; one
        # that requires an IMPLEMENTATION_DECISION row is decided by that row.
        return CriterionVerificationMode.SEMANTIC
    if (
        claim_type is ClaimType.STRUCTURAL
        and commitment_level is not CommitmentLevel.STEP
        and selected.issubset({FactType.CODE_CHANGE, FactType.CODE_OBSERVATION})
    ):
        return CriterionVerificationMode.COMPOSITE
    return CriterionVerificationMode.EXECUTABLE


@dataclass(frozen=True, slots=True)
class CompletionCriterionSpec:
    criterion_id: str
    observable_outcome: str
    required_evidence_types: tuple[FactType, ...] = ()
    entity_refs: tuple[str, ...] = ()
    test_selectors: tuple[str, ...] = ()
    required: bool = True
    requirement_id: str = ""
    requirement_text: str = ""
    claim_type: ClaimType | None = None
    commitment_level: CommitmentLevel = CommitmentLevel.STEP
    verification_mode: CriterionVerificationMode | None = None

    def __post_init__(self) -> None:
        if not self.criterion_id.strip() or not self.observable_outcome.strip():
            raise ValueError("CompletionCriterion requires an ID and observable outcome")
        if not self.requirement_id:
            object.__setattr__(self, "requirement_id", self.criterion_id)
        if not self.requirement_text:
            object.__setattr__(self, "requirement_text", self.observable_outcome)
        claim_type = self.claim_type or _legacy_claim_type(self.required_evidence_types)
        object.__setattr__(self, "claim_type", claim_type)
        if self.verification_mode is None:
            object.__setattr__(
                self,
                "verification_mode",
                default_verification_mode(
                    claim_type,
                    requirement_id=self.requirement_id,
                    required_evidence_types=self.required_evidence_types,
                    commitment_level=self.commitment_level,
                ),
            )

    @property
    def requires_requirement_review(self) -> bool:
        return self.commitment_level is CommitmentLevel.MILESTONE

    @property
    def is_semantic(self) -> bool:
        return self.verification_mode is CriterionVerificationMode.SEMANTIC

    @property
    def blocks_route(self) -> bool:
        """Only executable/composite contracts may hold the route open."""

        return self.required and not self.is_semantic

    @property
    def has_binding_selector(self) -> bool:
        """Whether persisted Evidence can be bound without guessing from prose."""

        return bool(self.entity_refs or self.test_selectors)

    @classmethod
    def from_value(
        cls,
        value: object,
        ordinal: int,
        *,
        default_level: CommitmentLevel = CommitmentLevel.STEP,
    ) -> "CompletionCriterionSpec":
        if isinstance(value, str):
            return cls(
                criterion_id=f"C{ordinal}",
                observable_outcome=value,
                # A prose criterion does not imply that the Milestone is a
                # testing stage.  The verifier still requires current,
                # criterion-bound evidence, but the Harness must provide a
                # structured evidence type when a particular kind is
                # mandatory.
                required_evidence_types=(),
                commitment_level=default_level,
            )
        if not isinstance(value, Mapping):
            raise TypeError("CompletionCriterion must be a string or object")
        raw_types = value.get("required_evidence_types", ())
        evidence_types = tuple(FactType(str(item)) for item in raw_types)
        raw_claim_type = value.get("claim_type")
        raw_level = value.get("commitment_level")
        raw_mode = value.get("verification_mode")
        return cls(
            criterion_id=str(value.get("criterion_id") or f"C{ordinal}"),
            observable_outcome=str(value.get("observable_outcome", "")),
            required_evidence_types=evidence_types,
            entity_refs=tuple(map(str, value.get("entity_refs", ()))),
            test_selectors=tuple(map(str, value.get("test_selectors", ()))),
            required=bool(value.get("required", True)),
            requirement_id=str(value.get("requirement_id", "")),
            requirement_text=str(value.get("requirement_text", "")),
            claim_type=(ClaimType(str(raw_claim_type)) if raw_claim_type else None),
            commitment_level=(
                CommitmentLevel(str(raw_level))
                if raw_level is not None
                else default_level
            ),
            verification_mode=(
                CriterionVerificationMode(str(raw_mode)) if raw_mode else None
            ),
        )


@dataclass(frozen=True, slots=True)
class PlanStepSpec:
    """A lightweight route cursor beneath a Milestone.

    ``minimum_acceptance`` is retained only to deserialize pre-refactor Plans
    and WAL records.  Newly projected Steps leave it empty: correctness is
    decided by the enclosing Milestone criteria, while a Step records focus,
    expected direction, risks and history addresses.
    """

    step_id: str
    title: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    corrective: bool = False
    criterion_ids: tuple[str, ...] = ()
    entity_refs: tuple[str, ...] = ()
    expected_outcome: str = ""
    minimum_acceptance: tuple[CompletionCriterionSpec, ...] = ()
    failure_signals: tuple[str, ...] = ()
    historical_dependency_refs: tuple[str, ...] = ()
    source_plan_item_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.step_id.strip() or not self.title.strip():
            raise ValueError("PlanStep requires an ID and title")
        if not self.expected_outcome:
            # Imported/offline plans created before Step contracts remain
            # readable. Live Harness manifests require this field explicitly.
            object.__setattr__(self, "expected_outcome", self.title)
        acceptance_ids = [item.criterion_id for item in self.minimum_acceptance]
        if len(acceptance_ids) != len(set(acceptance_ids)):
            raise ValueError("PlanStep acceptance criterion IDs must be unique")
        if len(self.criterion_ids) != len(set(self.criterion_ids)):
            raise ValueError("PlanStep Milestone criterion mappings must be unique")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], ordinal: int) -> "PlanStepSpec":
        normalized = str(value.get("status", "PENDING")).replace("-", "_").casefold()
        raw_status = {
            "pending": "PENDING",
            "inprogress": "IN_PROGRESS",
            "in_progress": "IN_PROGRESS",
            "completed": "COMPLETED_CLAIMED",
            "completed_claimed": "COMPLETED_CLAIMED",
            "completed_verified": "COMPLETED_VERIFIED",
            "failed": "FAILED",
            "cancelled": "CANCELLED",
        }.get(normalized, normalized.upper())
        raw_acceptance = value.get("minimum_acceptance", value.get("acceptance", ()))
        return cls(
            step_id=str(value.get("step_id") or value.get("id") or f"S{ordinal:03d}"),
            title=str(value.get("title") or value.get("step") or ""),
            status=PlanStepStatus(raw_status),
            corrective=bool(value.get("corrective", False)),
            criterion_ids=tuple(map(str, value.get("criterion_ids", ()))),
            entity_refs=tuple(map(str, value.get("entity_refs", ()))),
            expected_outcome=str(
                value.get("expected_outcome", value.get("title", value.get("step", "")))
            ),
            minimum_acceptance=tuple(
                CompletionCriterionSpec.from_value(item, index)
                for index, item in enumerate(raw_acceptance, start=1)
            ),
            failure_signals=tuple(map(str, value.get("failure_signals", ()))),
            historical_dependency_refs=tuple(
                map(str, value.get("historical_dependency_refs", ()))
            ),
            source_plan_item_ids=tuple(map(str, value.get("source_plan_item_ids", ()))),
        )


@dataclass(frozen=True, slots=True)
class MilestoneSpec:
    canonical_id: str
    title: str
    description: str
    completion_criteria: tuple[str, ...]
    verification: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    status: str = "pending"
    entity_refs: tuple[str, ...] = ()
    objective: str = ""
    scope: str = ""
    target_outcome: str = ""
    downstream_assumptions: tuple[str, ...] = ()
    non_goals: tuple[str, ...] = ()
    source_plan_item_ids: tuple[str, ...] = ()
    criteria: tuple[CompletionCriterionSpec, ...] = ()
    steps: tuple[PlanStepSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.canonical_id or not self.title:
            raise ValueError("Milestone requires canonical_id and title")
        if self.canonical_id in self.depends_on:
            raise ValueError("Milestone cannot depend on itself")
        if not self.completion_criteria and not self.criteria:
            raise ValueError("Milestone requires at least one CompletionCriterion")
        if not self.verification:
            raise ValueError("Milestone requires an explicit verification method")
        normalized = self.criteria or tuple(
            CompletionCriterionSpec.from_value(
                item,
                index,
                default_level=CommitmentLevel.MILESTONE,
            )
            for index, item in enumerate(self.completion_criteria, start=1)
        )
        if len({item.criterion_id for item in normalized}) != len(normalized):
            raise ValueError("CompletionCriterion IDs must be unique within a Milestone")
        step_ids = [item.step_id for item in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("PlanStep IDs must be unique within a Milestone")
        object.__setattr__(self, "criteria", normalized)
        object.__setattr__(
            self,
            "completion_criteria",
            tuple(item.observable_outcome for item in normalized),
        )
        if not self.objective:
            object.__setattr__(self, "objective", self.description)
        if not self.scope:
            object.__setattr__(self, "scope", self.description)
        if not self.target_outcome:
            # Legacy/offline plans remain readable.  Live Harness plans are
            # validated more strictly by ``CodexPlanNormalizer`` and must
            # declare the ideal repository state explicitly.
            object.__setattr__(self, "target_outcome", self.objective)

    @property
    def minimum_acceptance(self) -> tuple[CompletionCriterionSpec, ...]:
        """Hard completion gate; kept as an alias over typed criteria."""

        return self.criteria

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MilestoneSpec":
        raw_criteria = tuple(
            value.get(
                "minimum_acceptance",
                value.get("criteria", value.get("completion_criteria", ())),
            )
        )
        return cls(
            canonical_id=str(value.get("canonical_id") or value.get("milestone_id")),
            title=str(value["title"]),
            description=str(value.get("description", "")),
            completion_criteria=tuple(
                str(item.get("observable_outcome", "")) if isinstance(item, Mapping) else str(item)
                for item in raw_criteria
            ),
            verification=tuple(map(str, value.get("verification", ()))),
            depends_on=tuple(map(str, value.get("depends_on", ()))),
            status=str(value.get("status", "pending")),
            entity_refs=tuple(map(str, value.get("entity_refs", ()))),
            objective=str(value.get("objective", value.get("description", ""))),
            scope=str(value.get("scope", value.get("description", ""))),
            target_outcome=str(
                value.get(
                    "target_outcome",
                    value.get("objective", value.get("description", "")),
                )
            ),
            downstream_assumptions=tuple(map(str, value.get("downstream_assumptions", ()))),
            non_goals=tuple(map(str, value.get("non_goals", ()))),
            source_plan_item_ids=tuple(map(str, value.get("source_plan_item_ids", ()))),
            criteria=tuple(
                CompletionCriterionSpec.from_value(
                    item,
                    index,
                    default_level=CommitmentLevel.MILESTONE,
                )
                for index, item in enumerate(raw_criteria, start=1)
            ),
            steps=tuple(
                PlanStepSpec.from_dict(item, index)
                for index, item in enumerate(value.get("steps", ()), start=1)
                if isinstance(item, Mapping)
            ),
        )


@dataclass(frozen=True, slots=True)
class PlanSpec:
    goal: str
    milestones: tuple[MilestoneSpec, ...]
    final_verification: tuple[str, ...] = ()
    final_acceptance: tuple[CompletionCriterionSpec, ...] = ()
    native_plan: NativePlanSnapshot | None = None

    def __post_init__(self) -> None:
        if not self.goal.strip() or not self.milestones:
            raise ValueError("Plan requires a goal and at least one Milestone")
        identities = [item.canonical_id for item in self.milestones]
        if len(identities) != len(set(identities)):
            raise ValueError("Milestone canonical IDs must be unique")
        known = set(identities)
        unknown = {
            dependency
            for item in self.milestones
            for dependency in item.depends_on
            if dependency not in known
        }
        if unknown:
            raise ValueError(f"unknown Milestone dependencies: {sorted(unknown)}")
        final_ids = [item.criterion_id for item in self.final_acceptance]
        if len(final_ids) != len(set(final_ids)):
            raise ValueError("Final acceptance criterion IDs must be unique")
        if self.native_plan is not None:
            native_ids = tuple(item.source_step_id for item in self.native_plan.items)
            projected_ids = tuple(
                source_id
                for milestone in self.milestones
                for source_id in milestone.source_plan_item_ids
            )
            unknown = set(projected_ids).difference(native_ids)
            if unknown:
                raise ValueError(f"Milestone projection references unknown native items: {sorted(unknown)}")
            if projected_ids != native_ids:
                raise ValueError(
                    "Milestone projection must preserve every native Plan item exactly once "
                    "and in original order"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanSpec":
        raw_final = value.get("final_acceptance", ())
        return cls(
            goal=str(value["goal"]),
            milestones=tuple(MilestoneSpec.from_dict(item) for item in value["milestones"]),
            final_verification=tuple(map(str, value.get("final_verification", ()))),
            final_acceptance=tuple(
                CompletionCriterionSpec.from_value(
                    item,
                    index,
                    default_level=CommitmentLevel.MILESTONE,
                )
                for index, item in enumerate(raw_final, start=1)
            ),
            native_plan=(
                NativePlanSnapshot.from_dict(value["native_plan"])
                if isinstance(value.get("native_plan"), Mapping)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class EvidenceKey:
    evidence_type: FactType
    canonical_entity_id: str
    semantic_role: str
    revision_constraint: str
    branch_scope: str
    validity_requirement: str = "CURRENT"

    def __post_init__(self) -> None:
        if not all(
            (
                self.canonical_entity_id,
                self.semantic_role,
                self.revision_constraint,
                self.branch_scope,
                self.validity_requirement,
            )
        ):
            raise ValueError("EvidenceKey fields must be non-empty")

    @property
    def key_digest(self) -> str:
        return digest(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceKey":
        return cls(
            evidence_type=FactType(str(value["evidence_type"])),
            canonical_entity_id=str(value["canonical_entity_id"]),
            semantic_role=str(value["semantic_role"]),
            revision_constraint=str(value["revision_constraint"]),
            branch_scope=str(value["branch_scope"]),
            validity_requirement=str(value.get("validity_requirement", "CURRENT")),
        )


@dataclass(frozen=True, slots=True)
class EvidenceDraft:
    key: EvidenceKey
    content: Mapping[str, Any]
    authority: Authority = Authority.ASSERTED
    confidence: float = 1.0
    must_preserve: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceDraft":
        return cls(
            key=EvidenceKey.from_dict(value["key"]),
            content=dict(value.get("content", {})),
            authority=Authority(str(value.get("authority", "ASSERTED"))),
            confidence=float(value.get("confidence", 1.0)),
            must_preserve=bool(value.get("must_preserve", False)),
        )


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    event_type: str
    payload: Mapping[str, Any]
    facts: tuple[EvidenceDraft, ...] = ()
    entity_refs: tuple[str, ...] = ()
    milestone_id: str | None = None
    execution_phase: str = "execution"
    revision_id: str = "unknown"
    observed_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Event":
        return cls(
            event_id=str(value["event_id"]),
            event_type=str(value["event_type"]),
            payload=dict(value.get("payload", {})),
            facts=tuple(EvidenceDraft.from_dict(item) for item in value.get("facts", ())),
            entity_refs=tuple(map(str, value.get("entity_refs", ()))),
            milestone_id=(str(value["milestone_id"]) if value.get("milestone_id") else None),
            execution_phase=str(value.get("execution_phase", "execution")),
            revision_id=str(value.get("revision_id", "unknown")),
            observed_at=str(value.get("observed_at", utc_now())),
        )


@dataclass(frozen=True, slots=True)
class EventGroup:
    group_id: str
    group_type: str
    run_id: str
    branch_id: str
    revision_id: str
    events: tuple[Event, ...]
    milestone_id: str | None = None
    semantic_boundary: bool = True
    complete: bool = True
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.complete:
            raise ValueError("only complete EventGroups may enter the WAL")
        if not self.events:
            raise ValueError("EventGroup cannot be empty")
        ids = [item.event_id for item in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("Event IDs must be unique inside an EventGroup")
        if any(item.revision_id != self.revision_id for item in self.events):
            raise ValueError("all Events must bind the EventGroup revision")

    @property
    def token_count(self) -> int:
        return max(1, (len(canonical_bytes(self)) + 2) // 3)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EventGroup":
        return cls(
            group_id=str(value["group_id"]),
            group_type=str(value["group_type"]),
            run_id=str(value["run_id"]),
            branch_id=str(value["branch_id"]),
            revision_id=str(value["revision_id"]),
            events=tuple(Event.from_dict(item) for item in value["events"]),
            milestone_id=(str(value["milestone_id"]) if value.get("milestone_id") else None),
            semantic_boundary=bool(value.get("semantic_boundary", True)),
            complete=bool(value.get("complete", True)),
            created_at=str(value.get("created_at", utc_now())),
        )


@dataclass(frozen=True, slots=True)
class PageManifest:
    page_id: str
    run_id: str
    branch_id: str
    page_seq: int
    revision_ids: tuple[str, ...]
    milestone_ids: tuple[str, ...]
    execution_phases: tuple[str, ...]
    event_range: tuple[int, int]
    event_group_ids: tuple[str, ...]
    evidence_key_digests: tuple[str, ...]
    entity_refs: tuple[str, ...]
    token_count: int
    byte_count: int
    payload_digest: str
    redaction_proof: str
    previous_page_id: str | None
    seal_reason: str
    page_kind: PageKind
    tail: bool


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    evidence_id: str
    key: EvidenceKey
    content: Mapping[str, Any]
    authority: Authority
    confidence: float
    must_preserve: bool
    content_digest: str
    event_ids: tuple[str, ...]
    event_group_id: str
    page_id: str
    run_id: str
    branch_id: str
    revision_id: str
    valid_from_revision: str
    valid_to_revision: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticAnchor:
    anchor_id: str
    evidence_id: str
    page_id: str
    page_digest: str
    event_ids: tuple[str, ...]
    event_range: tuple[int, int]
    event_group_id: str
    blob_handle: str | None
    blob_range: tuple[int, int] | None
    revision_id: str
    branch_id: str


@dataclass(frozen=True, slots=True)
class EdgeTypeSpec:
    edge_type: str
    source_types: tuple[NodeType, ...]
    target_types: tuple[NodeType, ...]
    allowed_authority: tuple[Authority, ...]
    creation_rule: str
    update_rule: str
    critical_path_allowed: bool


@dataclass(frozen=True, slots=True)
class SemanticEdge:
    edge_id: str
    edge_type: str
    source_id: str
    source_type: NodeType
    target_id: str
    target_type: NodeType
    authority: Authority
    run_id: str
    branch_id: str
    plan_version_id: str | None
    valid_from_revision: str | None
    valid_to_revision: str | None
    valid_from_cursor: int
    valid_to_cursor: int | None
    provenance: tuple[str, ...]
    supporting_event_ids: tuple[str, ...]
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecallIntent:
    recall_id: str
    repository_id: str
    run_id: str
    branch_id: str
    revision_id: str
    required_evidence: tuple[EvidenceKey, ...]
    current_milestone_id: str
    question: str
    direct_page_ids: tuple[str, ...] = ()
    source_memory_ref: str | None = None
    direct_section_handle: str | None = None
    direct_continuation_token: str | None = None
    entity_refs: tuple[str, ...] = ()
    desired_detail: str = "smallest exact fact slice"
    purpose: str = "continue the current Milestone"
    required_structural_relations: tuple[str, ...] = ()
    preferred_structural_relations: tuple[str, ...] = ()
    structural_relation_direction: str = "BOTH"
    rich_code_relations: tuple[str, ...] = ()
    ambiguous_entities: tuple[str, ...] = ()
    unresolved_entities: tuple[str, ...] = ()
    require_exact_revision: bool = True
    temporal_scope: RecallTemporalScope = RecallTemporalScope.CURRENT_WORKSPACE_TRUTH
    max_pages: int = 8
    max_tokens: int = 8192
    max_pages_opened: int | None = None
    max_page_bytes_read: int = 8 * 1024 * 1024
    max_blob_bytes_read: int = 1024 * 1024
    max_slice_tokens: int = 4096
    max_recovered_block_tokens: int | None = None
    max_context_admission_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.required_evidence and not self.direct_page_ids:
            raise ValueError(
                "RecallIntent requires an exact EvidenceKey or runtime-owned Page address"
            )
        if len({item.key_digest for item in self.required_evidence}) != len(self.required_evidence):
            raise ValueError("RecallIntent contains duplicate EvidenceKeys")
        if len(set(self.direct_page_ids)) != len(self.direct_page_ids):
            raise ValueError("RecallIntent contains duplicate direct Page addresses")
        if self.direct_page_ids and not self.source_memory_ref:
            raise ValueError("direct Page addresses require their opaque MemoryRef provenance")
        if self.direct_section_handle is not None and not self.direct_page_ids:
            raise ValueError("a direct section handle requires its MemoryRef Page address")
        if self.direct_continuation_token is not None and self.direct_section_handle is None:
            raise ValueError("a section continuation requires its semantic section handle")
        if self.structural_relation_direction.upper() not in {"INCOMING", "OUTGOING", "BOTH"}:
            raise ValueError("RecallIntent structural relation direction is invalid")
        unknown_page_relations = {
            item.strip().upper()
            for item in (
                *self.required_structural_relations,
                *self.preferred_structural_relations,
            )
            if item.strip()
        }.difference(SEMANTIC_PAGE_QUERY_RELATIONS)
        if unknown_page_relations:
            raise ValueError(
                "RecallIntent contains non-Page relations in the Semantic Page Graph intent: "
                f"{sorted(unknown_page_relations)}"
            )

    @property
    def page_limit(self) -> int:
        return self.max_pages if self.max_pages_opened is None else self.max_pages_opened

    @property
    def recovered_block_limit(self) -> int:
        return (
            self.max_tokens
            if self.max_recovered_block_tokens is None
            else self.max_recovered_block_tokens
        )

    @property
    def admission_limit(self) -> int:
        return (
            self.recovered_block_limit
            if self.max_context_admission_tokens is None
            else self.max_context_admission_tokens
        )


@dataclass(frozen=True, slots=True)
class CoverageReceipt:
    state: CoverageState
    required_keys: tuple[str, ...]
    covered_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    validated_page_ids: tuple[str, ...]
    fallback_stage: FallbackStage
    freshness: str
    rich_capability_state: RichGraphState

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CoverageReceipt":
        return cls(
            state=CoverageState(str(value["state"])),
            required_keys=tuple(map(str, value.get("required_keys", ()))),
            covered_keys=tuple(map(str, value.get("covered_keys", ()))),
            missing_keys=tuple(map(str, value.get("missing_keys", ()))),
            validated_page_ids=tuple(map(str, value.get("validated_page_ids", ()))),
            fallback_stage=FallbackStage(str(value["fallback_stage"])),
            freshness=str(value["freshness"]),
            rich_capability_state=RichGraphState(str(value["rich_capability_state"])),
        )


@dataclass(frozen=True, slots=True)
class PageCandidate:
    page_id: str
    payload_digest: str
    revision_id: str
    branch_id: str
    event_range: tuple[int, int]
    anchor_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_key_digests: tuple[str, ...]
    estimated_tokens: int
    freshness_cursor: int


@dataclass(frozen=True, slots=True)
class PageSlice:
    slice_id: str
    page_id: str
    level: str
    event_range: tuple[int, int]
    event_ids: tuple[str, ...]
    event_group_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_keys: tuple[str, ...]
    revision_id: str
    page_digest: str
    content: Mapping[str, Any]
    token_count: int
    content_digest: str
    continuation: Mapping[str, Any] = field(default_factory=dict)
    section_handle: str | None = None
    section_directory: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PageSlice":
        return cls(
            slice_id=str(value["slice_id"]),
            page_id=str(value["page_id"]),
            level=str(value["level"]),
            event_range=tuple(map(int, value["event_range"])),
            event_ids=tuple(map(str, value.get("event_ids", ()))),
            event_group_ids=tuple(map(str, value.get("event_group_ids", ()))),
            evidence_ids=tuple(map(str, value.get("evidence_ids", ()))),
            evidence_keys=tuple(map(str, value.get("evidence_keys", ()))),
            revision_id=str(value["revision_id"]),
            page_digest=str(value["page_digest"]),
            content=dict(value.get("content", {})),
            token_count=int(value["token_count"]),
            content_digest=str(value["content_digest"]),
            continuation=dict(value.get("continuation", {})),
            section_handle=(
                None if value.get("section_handle") is None else str(value["section_handle"])
            ),
            section_directory=tuple(
                dict(item) for item in value.get("section_directory", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class RecoveredContextBlock:
    block_id: str
    recall_id: str
    slices: tuple[PageSlice, ...]
    coverage: CoverageReceipt
    current_milestone_id: str
    revision_id: str
    rendered_content: str
    token_count: int
    content_digest: str
    untrusted_data: bool = True
    source_memory_ref: str | None = None
    requested_entity_refs: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecoveredContextBlock":
        return cls(
            block_id=str(value["block_id"]),
            recall_id=str(value["recall_id"]),
            slices=tuple(PageSlice.from_dict(item) for item in value.get("slices", ())),
            coverage=CoverageReceipt.from_dict(value["coverage"]),
            current_milestone_id=str(value["current_milestone_id"]),
            revision_id=str(value["revision_id"]),
            rendered_content=str(value["rendered_content"]),
            token_count=int(value["token_count"]),
            content_digest=str(value["content_digest"]),
            untrusted_data=bool(value.get("untrusted_data", True)),
            source_memory_ref=(
                None
                if value.get("source_memory_ref") is None
                else str(value["source_memory_ref"])
            ),
            requested_entity_refs=tuple(
                map(str, value.get("requested_entity_refs", ()))
            ),
        )


@dataclass(frozen=True, slots=True)
class ContextHandle:
    page_id: str
    event_range: tuple[int, int]
    blob_handle: str | None
    blob_range: tuple[int, int] | None
    content_digest: str
    revision_id: str


@dataclass(frozen=True, slots=True)
class ContextArtifact:
    artifact_id: str
    representation: Representation
    content: str
    token_count: int
    content_digest: str
    milestone_ids: tuple[str, ...]
    entity_refs: tuple[str, ...]
    source_handles: tuple[ContextHandle, ...]
    must_preserve: bool = False
    current_milestone: bool = False
    verified: bool = True
    soft_pin_boundaries: int = 0
    derived_from: tuple[str, ...] = ()
    memory_ref: str | None = None

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError("token_count cannot be negative")
        if digest({"content": self.content}) != self.content_digest:
            raise ValueError("ContextArtifact content digest mismatch")
        if self.representation == Representation.VERIFIED_SUMMARY and not self.verified:
            raise ValueError("unverified Summary cannot enter ContextImage")
        if self.representation in {Representation.HANDLE, Representation.NONRESIDENT}:
            if not self.source_handles:
                raise ValueError("Handle/Nonresident representation needs a recoverable handle")
        if self.memory_ref is not None:
            if not self.memory_ref.startswith("memoryref_"):
                raise ValueError("ContextArtifact MemoryRef has an invalid virtual address")
            if not self.source_handles:
                raise ValueError("ContextArtifact MemoryRef needs a recoverable source handle")


@dataclass(frozen=True, slots=True)
class ContextImage:
    image_id: str
    thread_id: str
    artifacts: tuple[ContextArtifact, ...]
    total_tokens: int
    image_digest: str
    current_milestone_id: str
    revision_id: str


@dataclass(frozen=True, slots=True)
class ContinuityCheckpoint:
    checkpoint_id: str
    task_goal_digest: str
    plan_version_id: str
    current_milestone_id: str
    milestone_state_digest: str
    workspace_revision_id: str
    uncommitted_changes: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    failing_tests: tuple[str, ...]
    pending_side_effects: tuple[str, ...]
    resident_context_image: ContextImage
    nonresident_handles: tuple[ContextHandle, ...]
    latest_recovery_receipt_id: str | None
    delivery_state: DeliveryState
    context_digest: str
    safe_action_boundary: bool
    # A bounded projection of durable TPG/WAL state needed to continue the
    # same line of reasoning after a physical Thread replacement.  It is not
    # a second route authority: every value is derived from the persisted
    # route, native Plan observations, and Page-backed Evidence.
    execution_handoff: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RichGraphCapabilityReceipt:
    state: RichGraphState
    repository_id: str
    workspace_revision_id: str
    supported_relations: tuple[str, ...]
    frontier: tuple[str, ...]
    freshness_lag: int
    failure_reason: str | None
    cache_hit: bool


def unique_by_digest(items: Iterable[EvidenceKey]) -> tuple[EvidenceKey, ...]:
    result: dict[str, EvidenceKey] = {}
    for item in items:
        result.setdefault(item.key_digest, item)
    return tuple(result.values())
