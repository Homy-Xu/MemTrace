from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..contracts import (
    EvidenceDraft,
    EvidenceKey,
    PlanSpec,
    RecallTemporalScope,
    digest,
    stable_id,
)
from .planning_coordinator import WorkspaceSnapshotReceipt


@dataclass(frozen=True, slots=True)
class MemoryNeed:
    question: str
    required_evidence: tuple[EvidenceKey, ...]
    entity_refs: tuple[str, ...] = ()
    memory_ref: str | None = None
    direct_page_ids: tuple[str, ...] = ()
    section_handle: str | None = None
    continuation_token: str | None = None
    desired_detail: str = "smallest exact fact slice"
    purpose: str = "continue the current Milestone"
    required_structural_relations: tuple[str, ...] = ()
    preferred_structural_relations: tuple[str, ...] = ()
    structural_relation_direction: str = "BOTH"
    rich_code_relations: tuple[str, ...] = ()
    ambiguous_entities: tuple[str, ...] = ()
    unresolved_entities: tuple[str, ...] = ()
    resolution_state: str = "ACCEPTED"
    require_exact_revision: bool = True
    temporal_scope: RecallTemporalScope = RecallTemporalScope.CURRENT_WORKSPACE_TRUTH
    trigger_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.memory_ref is not None and not self.memory_ref.startswith("memoryref_"):
            raise ValueError("MemoryNeed contains an invalid MemoryRef")
        if self.direct_page_ids and self.memory_ref is None:
            raise ValueError("direct Page addresses require MemoryRef provenance")
        if self.section_handle is not None and self.memory_ref is None:
            raise ValueError("a semantic section handle requires MemoryRef provenance")
        if self.continuation_token is not None and self.section_handle is None:
            raise ValueError("a semantic section continuation requires its section handle")
        if len(set(self.direct_page_ids)) != len(self.direct_page_ids):
            raise ValueError("MemoryNeed contains duplicate direct Page addresses")

    @property
    def address_is_recallable(self) -> bool:
        """Whether Page-in has at least one exact address and requires no guess."""

        return bool(self.required_evidence or self.direct_page_ids) and self.resolution_state in {
            "ACCEPTED",
            "RESOLVED_EXACT",
            "RESOLVED_OPEN_WAL",
            "RESOLVED_HANDLE",
            "RESOLVED_PARTIAL_EXACT",
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MemoryNeed":
        raw_scope = (
            str(value.get("temporal_purpose", value.get("temporal_scope", ""))).strip().upper()
        )
        scope_aliases = {
            "CURRENT_WORKSPACE_TRUTH": RecallTemporalScope.CURRENT_WORKSPACE_TRUTH,
            "HISTORICAL_EXECUTION": RecallTemporalScope.HISTORICAL_EXECUTION,
            "COMPARE_HISTORY_TO_CURRENT": RecallTemporalScope.COMPARE_HISTORY_TO_CURRENT,
            "MEMORY_REF_DETAIL": RecallTemporalScope.MEMORY_REF_DETAIL,
        }
        if raw_scope:
            try:
                temporal_scope = scope_aliases[raw_scope]
            except KeyError as exc:
                raise ValueError(f"unknown temporal purpose: {raw_scope}") from exc
            exact_revision = temporal_scope is RecallTemporalScope.CURRENT_WORKSPACE_TRUTH
        else:
            # Compatibility for stored/offline requests. The model-facing tool
            # no longer exposes this mechanical switch.
            exact_revision = bool(value.get("require_exact_revision", True))
            temporal_scope = (
                RecallTemporalScope.CURRENT_WORKSPACE_TRUTH
                if exact_revision
                else RecallTemporalScope.HISTORICAL_EXECUTION
            )
        return cls(
            question=str(value["question"]),
            required_evidence=tuple(
                EvidenceKey.from_dict(item) for item in value.get("required_evidence", ())
            ),
            entity_refs=tuple(map(str, value.get("entity_refs", value.get("entities", ())))),
            memory_ref=(
                str(value["memory_ref"]).strip() if value.get("memory_ref") is not None else None
            ),
            direct_page_ids=tuple(map(str, value.get("direct_page_ids", ()))),
            section_handle=(
                str(value["section_handle"]).strip()
                if value.get("section_handle") is not None
                else None
            ),
            continuation_token=(
                str(value["continuation_token"]).strip()
                if value.get("continuation_token") is not None
                else None
            ),
            desired_detail=str(
                value.get("desired_detail", value.get("detail", "smallest exact fact slice"))
            ),
            purpose=str(value.get("purpose", "continue the current Milestone")),
            required_structural_relations=tuple(
                map(
                    str,
                    value.get(
                        "required_page_relations",
                        value.get("required_structural_relations", ()),
                    ),
                )
            ),
            preferred_structural_relations=tuple(
                map(
                    str,
                    value.get(
                        "preferred_page_relations",
                        value.get("preferred_structural_relations", ()),
                    ),
                )
            ),
            structural_relation_direction=str(
                value.get(
                    "page_relation_direction",
                    value.get("structural_relation_direction", value.get("direction", "BOTH")),
                )
            ).upper(),
            rich_code_relations=tuple(map(str, value.get("rich_code_relations", ()))),
            ambiguous_entities=tuple(map(str, value.get("ambiguous_entities", ()))),
            unresolved_entities=tuple(map(str, value.get("unresolved_entities", ()))),
            resolution_state=str(value.get("resolution_state", "ACCEPTED")).upper(),
            require_exact_revision=exact_revision,
            temporal_scope=temporal_scope,
            trigger_reasons=tuple(map(str, value.get("trigger_reasons", ()))),
        )


@dataclass(frozen=True, slots=True)
class MemoryUseAttribution:
    """Model semantic attribution bound to observable Delivery/action facts."""

    delivery_id: str | None = None
    entity_refs: tuple[str, ...] = ()
    evidence_handles: tuple[str, ...] = ()
    usage: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MemoryUseAttribution":
        raw_delivery = value.get("delivery_id")
        delivery = (
            None
            if raw_delivery is None or not str(raw_delivery).strip()
            else str(raw_delivery).strip()
        )
        return cls(
            delivery_id=delivery,
            entity_refs=tuple(
                dict.fromkeys(map(str, value.get("entity_refs", value.get("entities", ()))))
            ),
            evidence_handles=tuple(
                dict.fromkeys(map(str, value.get("evidence_handles", value.get("evidence", ()))))
            ),
            usage=" ".join(str(value.get("usage", value.get("effect", ""))).split()),
        )


@dataclass(frozen=True, slots=True)
class AgentAction:
    action_id: str
    action_type: str
    content: str
    facts: tuple[EvidenceDraft, ...] = ()
    entity_refs: tuple[str, ...] = ()
    milestone_canonical_id: str | None = None
    execution_phase: str = "execution"
    semantic_boundary: bool = True
    memory_need: MemoryNeed | None = None
    memory_use: tuple[MemoryUseAttribution, ...] = ()
    modified_files: tuple[str, ...] = ()
    accessed_files: tuple[str, ...] = ()
    failed_tests: tuple[str, ...] = ()
    failure_signatures: tuple[str, ...] = ()
    recent_symbols: tuple[str, ...] = ()
    command: str = ""
    tool_succeeded: bool | None = None
    side_effect: Mapping[str, Any] | None = None
    physical_context_failure: str | None = None
    plan_update: PlanSpec | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], ordinal: int) -> "AgentAction":
        action_id = str(value.get("action_id") or f"action-{ordinal}")
        need_value = value.get("memory_need")
        return cls(
            action_id=action_id,
            action_type=str(value.get("action_type", "agent_action")),
            content=str(value.get("content", "")),
            facts=tuple(EvidenceDraft.from_dict(item) for item in value.get("facts", ())),
            entity_refs=tuple(map(str, value.get("entity_refs", ()))),
            milestone_canonical_id=(
                str(value["milestone_id"]) if value.get("milestone_id") else None
            ),
            execution_phase=str(value.get("execution_phase", "execution")),
            semantic_boundary=bool(value.get("semantic_boundary", True)),
            memory_need=(
                MemoryNeed.from_mapping(need_value) if isinstance(need_value, Mapping) else None
            ),
            memory_use=tuple(
                MemoryUseAttribution.from_mapping(item)
                for item in value.get("memory_use", ())
                if isinstance(item, Mapping)
            ),
            modified_files=tuple(map(str, value.get("modified_files", ()))),
            accessed_files=tuple(map(str, value.get("accessed_files", ()))),
            failed_tests=tuple(map(str, value.get("failed_tests", ()))),
            failure_signatures=tuple(map(str, value.get("failure_signatures", ()))),
            recent_symbols=tuple(map(str, value.get("recent_symbols", ()))),
            command=str(value.get("command", "")),
            tool_succeeded=(
                bool(value["tool_succeeded"]) if value.get("tool_succeeded") is not None else None
            ),
            side_effect=(
                dict(value["side_effect"])
                if isinstance(value.get("side_effect"), Mapping)
                else None
            ),
            physical_context_failure=(
                str(value["physical_context_failure"])
                if value.get("physical_context_failure")
                else None
            ),
            plan_update=(
                PlanSpec.from_dict(value["plan_update"])
                if isinstance(value.get("plan_update"), Mapping)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ExecutionResumeDirective:
    """Typed Batch-to-runtime request to reflect inside the existing Attempt.

    Batch may observe absence of durable semantic progress, but it has no
    authority to choose the next coding action. The model must decide whether
    to continue the current Step or append a corrective Step.
    """

    directive_id: str
    cause: str
    same_attempt: bool
    decision_authority: str
    allowed_step_decisions: tuple[str, ...]
    semantic_progress: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.directive_id.startswith("resume_"):
            raise ValueError("resume directive has an invalid ID")
        if self.cause not in {"NO_SEMANTIC_PROGRESS", "MODEL_CONFIRMED_BLOCKER"}:
            raise ValueError("unsupported resume directive cause")
        if not self.same_attempt:
            raise ValueError("stall recovery must preserve the current Attempt")
        if self.decision_authority != "MODEL":
            raise ValueError("Batch cannot own the semantic recovery decision")
        allowed = tuple(dict.fromkeys(self.allowed_step_decisions))
        if not allowed or set(allowed).difference({"CONTINUE", "CORRECT", "BLOCKED"}):
            raise ValueError("resume directive contains unsupported Step decisions")
        object.__setattr__(self, "allowed_step_decisions", allowed)
        object.__setattr__(self, "semantic_progress", dict(self.semantic_progress))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExecutionResumeDirective":
        progress = value.get("semantic_progress", {})
        if not isinstance(progress, Mapping):
            raise ValueError("resume directive semantic progress is not an object")
        return cls(
            directive_id=str(value["directive_id"]),
            cause=str(value["cause"]),
            same_attempt=bool(value.get("same_attempt", False)),
            decision_authority=str(value.get("decision_authority", "")),
            allowed_step_decisions=tuple(map(str, value.get("allowed_step_decisions", ()))),
            semantic_progress=dict(progress),
        )


@dataclass(frozen=True, slots=True)
class RunRequest:
    repository_path: Path
    repository_id: str
    run_id: str
    branch_id: str
    revision_id: str
    user_task: str
    plan: PlanSpec | None
    actions: tuple[AgentAction, ...]
    run_root: Path
    harness_thread_id: str | None = None
    workspace_receipt: WorkspaceSnapshotReceipt | None = None
    resume_directive: ExecutionResumeDirective | None = None
    # Internal runner boundary only.  Offline/scenario input deliberately
    # cannot populate this field: a durable benchmark preflight is the sole
    # producer and RunCoordinator validates it again before projection.
    trusted_pre_execution_evidence: tuple[EvidenceDraft, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        repository_path: Path,
        run_root: Path,
    ) -> "RunRequest":
        """Build a request from the explicit offline/test scenario adapter."""

        resolved_repository = Path(repository_path).expanduser().resolve()
        task = str(value["task"])
        repository_id = str(
            value.get("repository_id") or stable_id("repo_", str(resolved_repository))
        )
        branch_id = str(value.get("branch_id", "main"))
        revision_id = str(value.get("revision_id", "workspace-current"))
        run_id = str(
            value.get("run_id")
            or stable_id(
                "run_",
                {
                    "repository_id": repository_id,
                    "task_digest": digest(task),
                    "revision_id": revision_id,
                },
            )
        )
        plan_value = value.get("plan")
        if not isinstance(plan_value, Mapping):
            raise ValueError("scenario requires a plan object")
        actions_value = value.get("actions")
        if not isinstance(actions_value, list) or not actions_value:
            raise ValueError("scenario requires at least one Agent action")
        return cls(
            repository_path=resolved_repository,
            repository_id=repository_id,
            run_id=run_id,
            branch_id=branch_id,
            revision_id=revision_id,
            user_task=task,
            plan=PlanSpec.from_dict(plan_value),
            actions=tuple(
                AgentAction.from_mapping(item, ordinal)
                for ordinal, item in enumerate(actions_value, start=1)
            ),
            run_root=Path(run_root).expanduser().resolve(),
            harness_thread_id=(
                str(value["harness_thread_id"]) if value.get("harness_thread_id") else None
            ),
            workspace_receipt=None,
            resume_directive=(
                ExecutionResumeDirective.from_mapping(value["resume_directive"])
                if isinstance(value.get("resume_directive"), Mapping)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class TraceEvent:
    sequence: int
    name: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    repository_id: str
    branch_id: str
    revision_id: str
    thread_id: str
    epoch_id: str
    plan_version_id: str
    current_milestone_id: str
    page_ids: tuple[str, ...]
    context_image_digest: str
    context_tokens: int
    trace: tuple[TraceEvent, ...]
    metrics: Mapping[str, Any]
    metric_samples: Mapping[str, Any]
    build_identity: Mapping[str, object]
    result_path: str
    task_status: str = ""
    milestone_statuses: Mapping[str, str] = field(default_factory=dict)
    unmet_completion_criteria: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    unresolved_questions: tuple[str, ...] = ()
    failed_tests: tuple[str, ...] = ()
    provider_pressure: str | None = None
    final_review_disposition: str = "CORRECT"
    completion_verdict: str = "INCOMPLETE"
    # Task-final contract diagnostics: which final criteria stayed unmet, which
    # semantic ones were disclosed as UNVERIFIED, and any terminal route stall.
    unmet_final_criteria: tuple[str, ...] = ()
    unverified_final_criteria: tuple[str, ...] = ()
    route_stalls: tuple[Mapping[str, object], ...] = ()
    # Adaptive engagement: initial level, final level and every escalation.
    engagement: Mapping[str, object] | None = None
