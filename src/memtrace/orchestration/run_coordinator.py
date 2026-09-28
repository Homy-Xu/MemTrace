from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace

from ..build_identity import current_build_identity
from ..config import V2RuntimeConfig
from ..context_runtime import (
    Compactor,
    ContextAdmission,
    ContextImageBuilder,
    ContextLifecycle,
    DeliveryJournal,
    EpochAdmission,
    NativeCompactionAdapter,
    PressurePolicy,
    SideEffectLedger,
    ThreadLifecycle,
    WorkingSetTracker,
    artifact_from_content,
)
from ..contracts import (
    Event,
    EventGroup,
    EvidenceDraft,
    EvidenceKey,
    FactType,
    MilestoneStatus,
    PlanSpec,
    PlanStepStatus,
    Representation,
    TaskStatus,
    digest,
    primitive,
    stable_id,
)
from ..database import StateDatabase
from ..durability import SecretRedactor, atomic_write_once
from ..harness.contracts import HarnessEvent, HarnessEventType
from ..harness.dynamic_tools import DynamicToolJournal
from ..harness.events import RawHarnessEventLedger
from ..observability import MetricRecorder
from ..page_store import PageStore, TailReason
from ..planning import PlanRegistry
from ..planning.requirements import extract_task_requirements
from ..recall import BoundedLocalRepositorySearch, FallbackChain, RecallService
from ..references import ReferenceDirectory
from ..rich_graph import RichGraphScheduler
from ..semantic_memory import SemanticStore
from .acceptance_progress import AcceptanceBudgets
from .background_jobs import BackgroundJobs
from .engagement import EngagementState, engagement_from_plan
from .execution_coordinator import ExecutionCoordinator
from .memory_need import HarnessEvidenceExtractor
from .models import AgentAction, RunRequest, RunResult
from .planning_coordinator import PlanningCoordinator, SuppliedPlanProvider
from .pre_execution_evidence import PreExecutionEvidenceCoordinator
from .trace import TraceRecorder
from .verification_coordinator import VerificationCoordinator


class RunCoordinator:
    """The formal V2 entry; it never delegates to any legacy Runner."""

    def __init__(
        self,
        config: V2RuntimeConfig,
        *,
        rich_processor: object | None = None,
        reference_directory_factory: Callable[..., ReferenceDirectory] | None = None,
        native_compaction_adapter: NativeCompactionAdapter | None = None,
        fault_injector: Callable[[str], None] | None = None,
        repository_stream: object | None = None,
    ) -> None:
        self.config = config
        self.rich_processor = rich_processor
        # The default remains the frozen Python ReferenceDirectory.  An
        # explicitly multilingual benchmark may provide an additive subclass
        # that can recover committed code surfaces from its immutable baseline.
        self.reference_directory_factory = (
            reference_directory_factory or ReferenceDirectory
        )
        self.native_compaction_adapter = native_compaction_adapter
        self.fault_injector = fault_injector
        self.repository_stream = repository_stream

    def _inject_fault(self, point: str) -> None:
        if self.fault_injector is not None:
            self.fault_injector(point)

    @staticmethod
    def _planning_event_group(request: RunRequest, event: HarnessEvent) -> EventGroup:
        facts = list(HarnessEvidenceExtractor.extract(event))
        revision_constraint = f"revision:{event.revision_id}"
        if event.event_type in {
            HarnessEventType.THREAD_STARTED,
            HarnessEventType.THREAD_RESUMED,
        }:
            facts.append(
                EvidenceDraft(
                    EvidenceKey(
                        FactType.USER_CONSTRAINT,
                        f"task:{request.run_id}",
                        "original_user_request",
                        revision_constraint,
                        request.branch_id,
                    ),
                    {
                        "user_task": request.user_task,
                        "authority": "USER",
                        "source_event_id": event.source_event_id,
                        "thread_id": event.thread_id,
                    },
                    must_preserve=True,
                )
            )
        if event.event_type in {
            HarnessEventType.MILESTONE_MANIFEST,
            HarnessEventType.PLAN_PROJECTION_DECISION,
        }:
            facts.append(
                EvidenceDraft(
                    EvidenceKey(
                        FactType.PLAN_DECISION,
                        f"plan:{request.run_id}",
                        "stage_projection",
                        revision_constraint,
                        request.branch_id,
                    ),
                    {
                        "decision": primitive(event.payload),
                        "authority": "HARNESS_OBSERVED",
                        "source_event_id": event.source_event_id,
                        "provider_method": event.provider_method,
                    },
                )
            )
        durable_event = Event(
            event_id=event.source_event_id,
            event_type=event.event_type.value,
            payload={
                "harness_event": {
                    "harness_event_id": event.harness_event_id,
                    "event_type": event.event_type.value,
                    "thread_id": event.thread_id,
                    "turn_id": event.turn_id,
                    "sequence": event.sequence,
                    "provider_time_ms": event.provider_time_ms,
                    "run_id": event.run_id,
                    "branch_id": event.branch_id,
                    "revision_id": event.revision_id,
                    "source_event_id": event.source_event_id,
                    "provider_method": event.provider_method,
                    "payload": primitive(event.payload),
                }
            },
            facts=tuple(facts),
            entity_refs=tuple(
                dict.fromkeys(
                    fact.key.canonical_entity_id
                    for fact in facts
                    if fact.key.canonical_entity_id.startswith(
                        ("file:", "symbol:", "test:", "observation:")
                    )
                )
            ),
            execution_phase="harness_planning",
            revision_id=event.revision_id,
        )
        boundary = event.event_type in {
            HarnessEventType.PLAN_UPDATED,
            HarnessEventType.MILESTONE_MANIFEST,
            HarnessEventType.PLAN_PROJECTION_DECISION,
            HarnessEventType.CANONICAL_PLAN_INJECTION,
            HarnessEventType.PLANNING_VALIDATED,
            HarnessEventType.TURN_COMPLETED,
        } and not bool(event.payload.get("partial", False))
        return EventGroup(
            group_id=stable_id("group_", {"harness_event": event.harness_event_id}),
            group_type="HARNESS_PLANNING",
            run_id=event.run_id,
            branch_id=event.branch_id,
            revision_id=event.revision_id,
            events=(durable_event,),
            milestone_id=None,
            semantic_boundary=boundary,
        )

    @staticmethod
    def _is_semantic_planning_event(event: HarnessEvent) -> bool:
        if event.event_type is HarnessEventType.TOOL_RESULT:
            # Completed read-only Planning observations are reusable semantic
            # facts. Output deltas and tool intents remain Raw-Ledger telemetry
            # so the Page stream receives one complete observation, not every
            # provider chunk.
            return not bool(event.payload.get("partial", False))
        return event.event_type in {
            HarnessEventType.THREAD_STARTED,
            HarnessEventType.THREAD_RESUMED,
            HarnessEventType.PLAN_PROPOSED,
            HarnessEventType.PLAN_UPDATED,
            HarnessEventType.MILESTONE_MANIFEST,
            HarnessEventType.PLAN_PROJECTION_DECISION,
            HarnessEventType.CANONICAL_PLAN_INJECTION,
            HarnessEventType.PLANNING_VALIDATED,
            HarnessEventType.TURN_COMPLETED,
        }

    @staticmethod
    def _initial_plan_facts(request: RunRequest, plan: PlanSpec) -> tuple[EvidenceDraft, ...]:
        revision_constraint = f"revision:{request.revision_id}"
        return (
            EvidenceDraft(
                EvidenceKey(
                    FactType.USER_CONSTRAINT,
                    f"task:{request.run_id}",
                    "original_user_request",
                    revision_constraint,
                    request.branch_id,
                ),
                {"user_task": request.user_task, "authority": "USER"},
                must_preserve=True,
            ),
            EvidenceDraft(
                EvidenceKey(
                    FactType.PLAN_DECISION,
                    f"plan:{request.run_id}",
                    "offline_plan_snapshot",
                    revision_constraint,
                    request.branch_id,
                ),
                {"plan": primitive(plan), "authority": "OFFLINE_SCENARIO"},
            ),
        )

    @staticmethod
    def _transport_plan_projection(
        plan: PlanSpec,
        registry: PlanRegistry,
        run_id: str,
    ) -> PlanSpec:
        """Render Registry status into the Harness view without rewriting Plan contracts."""

        milestone_statuses = registry.milestone_statuses(run_id)
        projected_milestones = []
        for milestone in plan.milestones:
            step_statuses = {
                str(item["step_id"]): PlanStepStatus(str(item["status"]))
                for item in registry.milestone_steps(run_id, milestone.canonical_id)
            }
            projected_milestones.append(
                replace(
                    milestone,
                    status=milestone_statuses.get(milestone.canonical_id, milestone.status),
                    steps=tuple(
                        replace(step, status=step_statuses.get(step.step_id, step.status))
                        for step in milestone.steps
                    ),
                )
            )
        return replace(plan, milestones=tuple(projected_milestones))

    @staticmethod
    def _recover_planning_result(
        page_store: PageStore,
    ) -> tuple[PlanSpec, str, str, bool] | None:
        manifest_event: Event | None = None
        injected = False
        for group in page_store.durable_groups():
            for event in group.events:
                payload_value = page_store.resolve_event_payload(event)
                wrapper = payload_value.get("harness_event")
                if not isinstance(wrapper, Mapping):
                    continue
                event_type = str(wrapper.get("event_type", ""))
                payload = wrapper.get("payload", {})
                if event_type == HarnessEventType.MILESTONE_MANIFEST.value:
                    manifest_event = event
                elif event_type == HarnessEventType.CANONICAL_PLAN_INJECTION.value:
                    if isinstance(payload, Mapping) and str(payload.get("state", "")) == (
                        "TRANSPORT_ACCEPTED"
                    ):
                        injected = True
        if manifest_event is None:
            return None
        wrapper = page_store.resolve_event_payload(manifest_event).get("harness_event")
        assert isinstance(wrapper, Mapping)
        payload = wrapper.get("payload")
        if not isinstance(payload, Mapping) or not isinstance(payload.get("plan"), Mapping):
            raise RuntimeError("durable MilestoneManifest has no recoverable Plan")
        thread_id = str(wrapper.get("thread_id", ""))
        if not thread_id:
            raise RuntimeError("durable MilestoneManifest has no Harness Thread ID")
        return PlanSpec.from_dict(payload["plan"]), thread_id, manifest_event.event_id, injected

    def run(self, request: RunRequest, **kwargs) -> RunResult:
        # Stream lock spans planning, execution, checkpoint and receipt publish.
        # Default/Python callers outside SWE-Milestone retain the original path.
        if self.repository_stream is None:
            return self._run(request, **kwargs)
        try:
            self.repository_stream.open(request)
            return self._run(request, **kwargs)
        finally:
            self.repository_stream.close()

    def _run(
        self,
        request: RunRequest,
        *,
        action_source: Iterable[AgentAction] | None = None,
        harness_driver: object | None = None,
        harness_adapter: object | None = None,
        trusted_verifier: Callable[[], Mapping[str, object]] | None = None,
    ) -> RunResult:
        """Run the one V2 production chain over a live or recorded action stream.

        ``RunRequest.actions`` is the CLI/offline adapter. A provider integration
        may pass a lazy ``action_source``; both paths enter the same serial WAL,
        Semantic, Recall and Context lifecycle below.
        """

        run_started_ns = time.perf_counter_ns()
        run_started_monotonic = time.monotonic()
        recovered_plan = None
        build_identity = current_build_identity()
        if sum(item is not None for item in (harness_driver, harness_adapter, action_source)) > 1:
            raise ValueError("harness_adapter, harness_driver and action_source are exclusive")
        # This is deliberately the first stateful boundary.
        self.config.validate()
        if not request.repository_path.is_dir():
            raise ValueError(f"repository path is not a directory: {request.repository_path}")
        request.run_root.mkdir(parents=True, exist_ok=True)
        result_path = (
            self.repository_stream.invocation / "result.json"
            if self.repository_stream is not None else request.run_root / "result.json"
        )
        if result_path.exists():
            raise FileExistsError(
                f"immutable run result already exists; use a new run root: {result_path}"
            )

        metrics = MetricRecorder()
        trace = TraceRecorder()
        trace.record(
            "CONFIG_VALIDATED_BEFORE_PLANNING",
            schema_version=self.config.schema_version,
            five_stage_chain=True,
            rich_enabled=self.config.stages.rich_graph,
            build_identity=build_identity.as_mapping(),
        )
        database = StateDatabase(request.run_root / "v2-state.sqlite3")
        raw_events = RawHarnessEventLedger(database)
        dynamic_tool_journal = DynamicToolJournal(
            database,
            run_id=request.run_id,
            branch_id=request.branch_id,
        )
        reference_directory = self.reference_directory_factory(
            database,
            repository_id=request.repository_id,
            repository_path=request.repository_path,
        )
        secret_names = tuple(
            dict.fromkeys(
                (
                    *self.config.redaction_env_vars,
                    *(
                        (self.config.provider.api_key_env,)
                        if self.config.provider.api_key_env is not None
                        else ()
                    ),
                )
            )
        )
        secrets = tuple(value for name in secret_names if (value := os.environ.get(name)))
        redactor = SecretRedactor(secrets)
        page_store = PageStore(
            request.run_root / "page-store",
            database,
            request.run_id,
            request.branch_id,
            policy=self.config.page_policy,
            redactor=redactor,
            metrics=metrics,
        )
        trace.record(
            "PAGE_STORE_INITIALIZED_BEFORE_PLANNING",
            wal=str(page_store.wal_path),
        )
        semantic = SemanticStore(database)
        # Recovery may seal Planning WAL before a Task/Plan scope exists. Page
        # durability therefore precedes Semantic projection; projection is
        # repaired idempotently immediately after Registry initialization.
        recovery = page_store.recover()
        if recovery.scanned_groups or recovery.recovered_groups or recovery.sealed_pages:
            trace.record(
                "PAGE_STORE_RECOVERED_BEFORE_PLANNING",
                scanned_groups=recovery.scanned_groups,
                recovered_groups=recovery.recovered_groups,
                sealed_page_ids=list(recovery.sealed_pages),
            )

        background: BackgroundJobs | None = None
        try:
            live_driver = harness_driver
            planning_source_event_id: str | None = None
            canonical_plan_already_injected = False

            def persist_planning_event(event: HarnessEvent) -> None:
                nonlocal planning_source_event_id, canonical_plan_already_injected
                raw_events.append(event, phase="PLANNING")
                if event.event_type is HarnessEventType.MILESTONE_MANIFEST:
                    planning_source_event_id = event.source_event_id
                if (
                    event.event_type is HarnessEventType.CANONICAL_PLAN_INJECTION
                    and str(event.payload.get("state", "")) == "TRANSPORT_ACCEPTED"
                ):
                    canonical_plan_already_injected = True
                trace.record(
                    "RAW_PLANNING_PROVIDER_EVENT_OBSERVED",
                    event_type=event.event_type.value,
                    source_event_id=event.source_event_id,
                    provider_method=event.provider_method,
                    provider_sequence=event.sequence,
                    payload_keys=sorted(map(str, event.payload.keys())),
                    raw_summary_digest=digest(primitive(event.raw_provider_summary)),
                )
                if not self._is_semantic_planning_event(event):
                    trace.record(
                        "RAW_PLANNING_EVENT_EXCLUDED_FROM_SEMANTIC_PAGE_STREAM",
                        event_type=event.event_type.value,
                        source_event_id=event.source_event_id,
                    )
                    return
                if page_store.has_durable_event(event.source_event_id):
                    trace.record(
                        "PLANNING_EVENT_RECOVERED_FROM_WAL",
                        event_type=event.event_type.value,
                        source_event_id=event.source_event_id,
                        thread_id=event.thread_id,
                        turn_id=event.turn_id,
                    )
                    return
                group = self._planning_event_group(request, event)
                page_store.append_group(group, defer_seal=not group.semantic_boundary)
                trace.record(
                    "PLANNING_EVENT_WAL_COMMITTED",
                    event_type=event.event_type.value,
                    source_event_id=event.source_event_id,
                    thread_id=event.thread_id,
                    turn_id=event.turn_id,
                )

            if request.plan is None:
                if harness_adapter is None:
                    raise ValueError("live run without a supplied Plan requires a Harness adapter")
                recovered_plan = self._recover_planning_result(page_store)
                if recovered_plan is None:
                    trusted_acceptance = tuple(PreExecutionEvidenceCoordinator.validated(request))
                    bind_acceptance = getattr(
                        harness_adapter,
                        "bind_trusted_verification_contract",
                        None,
                    )
                    if trusted_acceptance and PreExecutionEvidenceCoordinator.host_managed(request):
                        if not callable(bind_acceptance):
                            raise RuntimeError(
                                "live Harness cannot bind the trusted benchmark acceptance contract"
                            )
                        bind_acceptance(trusted_acceptance)
                        trace.record(
                            "TRUSTED_VERIFICATION_CONTRACT_BOUND_BEFORE_PLANNING",
                            selector_count=len(trusted_acceptance),
                        )
                    elif trusted_acceptance:
                        trace.record(
                            "TRUSTED_VERIFICATION_CONTRACT_RESERVED_FOR_OFFICIAL_EVALUATOR",
                            selector_count=len(trusted_acceptance),
                            model_visible=False,
                            route_acceptance=False,
                        )
                    with metrics.timer("planning_provider_ms"):
                        planning = harness_adapter.plan(
                            user_task=request.user_task,
                            planning_context=PreExecutionEvidenceCoordinator.planning_context(
                                request
                            ) + ("\n" + self.repository_stream.planning_context()
                                 if self.repository_stream is not None else ""),
                            resume_thread_id=request.harness_thread_id,
                            run_id=request.run_id,
                            branch_id=request.branch_id,
                            revision_id=request.revision_id,
                            on_event=persist_planning_event,
                            inject=False,
                            workspace_receipt=request.workspace_receipt,
                        )
                    plan = planning.plan
                    thread_id = planning.thread_id
                    if planning_source_event_id is None:
                        raise RuntimeError("MilestoneManifest was not committed to WAL")
                else:
                    plan, thread_id, planning_source_event_id, canonical_plan_already_injected = (
                        recovered_plan
                    )
                    if self.repository_stream is not None:
                        persisted_registry = PlanRegistry(database, semantic)
                        if database.connection.execute(
                            "SELECT 1 FROM v2_tasks WHERE run_id=?", (request.run_id,)
                        ).fetchone():
                            # The initial WAL manifest is not the current route:
                            # native plan updates and released work may extend it.
                            plan = persisted_registry.active_plan(request.run_id)
                    context_table = database.connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='context_image_state_v2'"
                    ).fetchone()
                    if context_table is not None:
                        active_context = database.connection.execute(
                            "SELECT thread_id FROM context_image_state_v2 "
                            "WHERE run_id=? AND branch_id=? "
                            "ORDER BY updated_at DESC LIMIT 1",
                            (request.run_id, request.branch_id),
                        ).fetchone()
                        if active_context is not None:
                            thread_id = str(active_context["thread_id"])
                    harness_adapter.resume_planned_thread(
                        thread_id=thread_id,
                        run_id=request.run_id,
                        branch_id=request.branch_id,
                        revision_id=request.revision_id,
                        on_event=persist_planning_event,
                    )
                    trace.record(
                        "PLANNING_RECOVERED_FROM_WAL_WITHOUT_MODEL_REPLAY",
                        thread_id=thread_id,
                        source_event_id=planning_source_event_id,
                    )
                if live_driver is None:
                    from ..harness import CodexHarnessDriver

                    live_driver = CodexHarnessDriver(
                        harness_adapter,
                        native_compaction_timeout_seconds=(
                            self.config.native_compaction_timeout_seconds
                        ),
                        native_compaction_enabled=(self.config.provider.native_compaction_enabled),
                    )
                protocol_schema = getattr(harness_adapter, "protocol_schema", None)
                if protocol_schema is not None:
                    trace.record(
                        "CODEX_INSTALLED_PROTOCOL_SCHEMA_VERIFIED",
                        schema_digest=protocol_schema.schema_digest,
                        turn_start_schema=protocol_schema.turn_start_schema,
                        collaboration_mode_field=protocol_schema.collaboration_mode_field,
                        sandbox_policy_field=protocol_schema.sandbox_policy_field,
                        plan_mode_value=protocol_schema.plan_mode_value,
                        supports_context_injection=protocol_schema.supports_context_injection,
                        supports_native_compaction=protocol_schema.supports_native_compaction,
                        supports_context_compaction_events=(
                            protocol_schema.supports_context_compaction_events
                        ),
                        supports_turn_effort=protocol_schema.supports_turn_effort,
                        supports_output_schema=protocol_schema.supports_output_schema,
                        supports_thread_fork=protocol_schema.supports_thread_fork,
                        supports_dynamic_tools=protocol_schema.supports_dynamic_tools,
                        supports_turn_interrupt=protocol_schema.supports_turn_interrupt,
                    )
            else:
                planner = PlanningCoordinator(metrics)
                with metrics.timer("planning_provider_ms"):
                    plan = planner.generate_read_only(
                        repository_path=request.repository_path,
                        run_root=request.run_root,
                        user_task=request.user_task,
                        revision_id=request.revision_id,
                        provider=SuppliedPlanProvider(request.plan),
                        workspace_receipt=request.workspace_receipt,
                    )
                thread_id = request.harness_thread_id or stable_id(
                    "thread_", {"run": request.run_id, "epoch": 0}
                )
            # Engagement depth is decided once, from Planning-time size signals,
            # before the Plan becomes the Registry's frozen contract.  The same
            # deterministic fold applies on recovery, so the recovered manifest
            # reproduces the Registry's Milestones.
            engagement_decision, plan = engagement_from_plan(
                self.config.engagement,
                plan=plan,
                user_task=request.user_task,
                requirement_count=len(extract_task_requirements(request.run_id, request.user_task)),
            )
            engagement = EngagementState(
                initial=engagement_decision.level,
                reason=engagement_decision.reason,
                signals=engagement_decision.signals,
            )
            if self.repository_stream is not None:
                plan, repository_alignment = self.repository_stream.align_plan(plan)
                trace.record(
                    "REPOSITORY_PLAN_ALIGNED_TO_OFFICIAL_DAG",
                    **repository_alignment,
                )
            trace.record(
                "ENGAGEMENT_LEVEL_DECIDED",
                mode=self.config.engagement.mode,
                level=engagement.level.value,
                reason=engagement_decision.reason,
                milestone_groups=engagement_decision.milestone_groups,
                projected_milestone_count=len(plan.milestones),
                **engagement_decision.signals.as_mapping(),
            )
            safe_task_value, _ = redactor.redact(request.user_task)
            safe_plan_value, plan_redacted = redactor.redact(primitive(plan))
            if not isinstance(safe_task_value, str) or not isinstance(safe_plan_value, dict):
                raise RuntimeError("redaction changed Task/Plan value types")
            safe_task = safe_task_value
            # Preserve the provider's canonical Plan object exactly unless a
            # configured secret genuinely requires a redacted replacement.
            if plan_redacted:
                plan = PlanSpec.from_dict(safe_plan_value)
            trace.record(
                "PLAN_GENERATED_AND_READ_ONLY_VALIDATED",
                goal_digest=digest(plan.goal),
                milestone_count=len(plan.milestones),
            )

            if planning_source_event_id is None:
                plan_event_id = stable_id(
                    "event_", {"run": request.run_id, "kind": "PLAN_VALIDATED"}
                )
                plan_group = EventGroup(
                    group_id=stable_id("group_", {"run": request.run_id, "kind": "INITIAL_PLAN"}),
                    group_type="PLAN_SNAPSHOT",
                    run_id=request.run_id,
                    branch_id=request.branch_id,
                    revision_id=request.revision_id,
                    events=(
                        Event(
                            event_id=plan_event_id,
                            event_type="PLAN_SNAPSHOT",
                            payload={"task": safe_task, "plan": primitive(plan)},
                            facts=self._initial_plan_facts(request, plan),
                            execution_phase="planning",
                            revision_id=request.revision_id,
                        ),
                    ),
                    milestone_id=None,
                    semantic_boundary=True,
                )
                if not page_store.has_durable_event(plan_event_id):
                    page_store.append_group(plan_group, defer_seal=False)
                    trace.record(
                        "INITIAL_PLAN_WAL_COMMITTED_BEFORE_REGISTRY",
                        event_id=plan_event_id,
                        group_id=plan_group.group_id,
                    )
                else:
                    trace.record(
                        "INITIAL_PLAN_RECOVERED_FROM_WAL",
                        event_id=plan_event_id,
                        group_id=plan_group.group_id,
                    )
            else:
                plan_event_id = planning_source_event_id
                if not page_store.has_durable_event(plan_event_id):
                    raise RuntimeError("Planning projection source is not durable in WAL")
            resume_event_id: str | None = None
            if request.resume_directive is not None:
                # batch-control.jsonl is the external hand-off receipt, while
                # the Page WAL is the runtime's causal authority.  Persist the
                # directive here before reopening Task/Milestone state; using
                # the old Plan event as provenance would make recovery appear
                # to happen without a new cause.
                resume_event_id = stable_id(
                    "event_",
                    {
                        "run": request.run_id,
                        "resume_directive": request.resume_directive.directive_id,
                    },
                )
                resume_group = EventGroup(
                    group_id=stable_id(
                        "group_",
                        {
                            "run": request.run_id,
                            "resume_directive": request.resume_directive.directive_id,
                        },
                    ),
                    group_type="EXECUTION_RESUME_DIRECTIVE",
                    run_id=request.run_id,
                    branch_id=request.branch_id,
                    revision_id=request.revision_id,
                    events=(
                        Event(
                            event_id=resume_event_id,
                            event_type="EXECUTION_RESUME_DIRECTIVE",
                            payload=primitive(request.resume_directive),
                            entity_refs=("context:execution_route",),
                            execution_phase="recovery",
                            revision_id=request.revision_id,
                        ),
                    ),
                    semantic_boundary=True,
                )
                if not page_store.has_durable_event(resume_event_id):
                    page_store.append_group(resume_group, defer_seal=True)
                trace.record(
                    "EXECUTION_RESUME_DIRECTIVE_WAL_COMMITTED",
                    directive_id=request.resume_directive.directive_id,
                    source_event_id=resume_event_id,
                    cause=request.resume_directive.cause,
                    same_attempt=True,
                )
            self._inject_fault("AFTER_PLANNING_WAL_BEFORE_REGISTRY")
            registry = PlanRegistry(
                database,
                semantic,
                durable_event_guard=lambda source_event_id: (
                    page_store.has_durable_event(source_event_id)
                    or raw_events.has_durable_source_event(source_event_id)
                ),
            )
            if not engagement.full:
                registry.retain_predecessor_milestones = (
                    self.config.engagement.retain_predecessor_milestones
                )
            if self.repository_stream is not None and database.connection.execute(
                "SELECT 1 FROM v2_tasks WHERE run_id=?", (request.run_id,)
            ).fetchone():
                # Also covers recorded-action recovery: a caller's original
                # seed Plan cannot replace the subsequently extended route.
                plan = registry.active_plan(request.run_id)
            application = registry.initialize_task(
                repository_id=request.repository_id,
                run_id=request.run_id,
                branch_id=request.branch_id,
                revision_id=request.revision_id,
                user_request=safe_task,
                plan=plan,
                source_event_id=plan_event_id,
            )
            page_store.projector = semantic.project_page
            repaired_pages = page_store.project_existing_pages()
            if repaired_pages:
                trace.record(
                    "RECOVERED_PAGES_PROJECTED_AFTER_REGISTRY",
                    page_ids=list(repaired_pages),
                )
            task_status = registry.task_status(request.run_id)
            if task_status is TaskStatus.CREATED:
                registry.record_task_state(
                    run_id=request.run_id,
                    status=TaskStatus.PLANNING,
                    revision_id=request.revision_id,
                    source_event_id=plan_event_id,
                )
                registry.record_task_state(
                    run_id=request.run_id,
                    status=TaskStatus.EXECUTING,
                    revision_id=request.revision_id,
                    source_event_id=plan_event_id,
                )
            elif task_status is TaskStatus.VERIFYING:
                registry.record_task_state(
                    run_id=request.run_id,
                    status=TaskStatus.EXECUTING,
                    revision_id=request.revision_id,
                    source_event_id=plan_event_id,
                )
            elif task_status is TaskStatus.BLOCKED and request.resume_directive is not None:
                blocked_milestone = registry.current(request.run_id)
                if blocked_milestone.status != MilestoneStatus.BLOCKED.value:
                    raise RuntimeError("blocked Task has no blocked current Milestone")
                registry.record_milestone_state(
                    run_id=request.run_id,
                    canonical_id=blocked_milestone.canonical_id,
                    status=MilestoneStatus.IN_PROGRESS,
                    revision_id=request.revision_id,
                    source_event_id=resume_event_id or plan_event_id,
                )
                registry.record_task_state(
                    run_id=request.run_id,
                    status=TaskStatus.EXECUTING,
                    revision_id=request.revision_id,
                    source_event_id=resume_event_id or plan_event_id,
                )
                trace.record(
                    "BLOCKED_ATTEMPT_REOPENED_ON_EXPLICIT_RESUME",
                    directive_id=request.resume_directive.directive_id,
                    cause=request.resume_directive.cause,
                    current_milestone_id=blocked_milestone.identity_id,
                    same_attempt=True,
                    same_thread=True,
                )
            elif (
                task_status is TaskStatus.FAILED
                and self.repository_stream is not None
                and request.resume_directive is not None
                and resume_event_id is not None
            ):
                registry.reopen_repository_stream_task(
                    run_id=request.run_id,
                    revision_id=request.revision_id,
                    source_event_id=resume_event_id,
                )
                trace.record(
                    "FAILED_REPOSITORY_STREAM_INVOCATION_REOPENED",
                    directive_id=request.resume_directive.directive_id,
                    cause=request.resume_directive.cause,
                    same_repository_task=True,
                    official_pass_inferred=False,
                )
            elif task_status is TaskStatus.VERIFYING and self.repository_stream is not None:
                registry.record_task_state(run_id=request.run_id, status=TaskStatus.EXECUTING,
                    revision_id=request.revision_id, source_event_id=plan_event_id)
            elif task_status is not TaskStatus.EXECUTING:
                raise RuntimeError(
                    f"persisted Task cannot resume from terminal state {task_status.value}"
                )
            current = registry.current(request.run_id)
            roots = registry.working_set_roots(request.run_id)
            trace.record(
                "REGISTRY_AND_SEMANTIC_SEED_COMMITTED",
                task_id=application.task_id,
                goal_id=application.goal_id,
                plan_version_id=current.plan_version_id,
                current_milestone_id=current.identity_id,
                task_status=registry.task_status(request.run_id).value,
                working_roots=[
                    {"identity_id": item.identity_id, "state": item.state} for item in roots
                ],
            )
            pre_execution_summary = PreExecutionEvidenceCoordinator(
                request,
                registry,
                page_store,
                trace,
            ).commit(plan)
            # A trusted baseline is historical diagnostic context only. It
            # cannot satisfy navigation or current-code acceptance.
            current = registry.current(request.run_id)

            builder = ContextImageBuilder()
            task_artifact = artifact_from_content(
                content=json.dumps(
                    {
                        "kind": "TASK_GOAL",
                        "user_task": safe_task,
                        "goal": plan.goal,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                representation=Representation.FULL,
                must_preserve=True,
                verified=True,
                identity_seed=application.task_id,
            )
            plan_artifact = artifact_from_content(
                content=json.dumps(
                    {"kind": "PLAN_VERSION", "plan": primitive(plan)},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                representation=Representation.FULL,
                milestone_ids=application.milestone_identity_ids,
                entity_refs=("context:plan_version",),
                must_preserve=False,
                verified=True,
                identity_seed=application.plan_version_id,
            )
            milestone_artifact = artifact_from_content(
                content=json.dumps(
                    {
                        "kind": "CURRENT_MILESTONE",
                        "identity_id": current.identity_id,
                        "canonical_id": current.canonical_id,
                        "title": current.title,
                        "status": current.status,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                representation=Representation.FULL,
                milestone_ids=(current.identity_id,),
                entity_refs=("context:milestone_identity",),
                must_preserve=True,
                current_milestone=True,
                verified=True,
                identity_seed=current.identity_id,
            )
            initial_artifacts = [task_artifact, plan_artifact, milestone_artifact]
            if pre_execution_summary is not None:
                initial_artifacts.append(
                    artifact_from_content(
                        content=json.dumps(
                            pre_execution_summary,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        representation=Representation.VERIFIED_SUMMARY,
                        milestone_ids=(current.identity_id,),
                        entity_refs=("context:trusted_pre_execution_evidence",),
                        must_preserve=False,
                        verified=True,
                        identity_seed=str(pre_execution_summary["binding_digest"]),
                    )
                )
            initial_image = builder.build(
                thread_id=thread_id,
                artifacts=tuple(initial_artifacts),
                current_milestone_id=current.identity_id,
                revision_id=request.revision_id,
            )
            compactor = Compactor(metrics, self.native_compaction_adapter)
            native_capability = compactor.native_capability()
            admission = ContextAdmission(
                PressurePolicy(self.config.context_budget),
                compactor,
                builder,
                metrics,
            )
            from ..harness.provider_context import ProviderContextLedger

            provider_context = ProviderContextLedger(database, metrics, admission.policy)
            lifecycle = ContextLifecycle(
                run_id=request.run_id,
                branch_id=request.branch_id,
                admission=admission,
                delivery=DeliveryJournal(database),
                builder=builder,
                metrics=metrics,
                initial_image=initial_image,
            )
            side_effects = SideEffectLedger(database)
            if self.repository_stream is not None:
                orphan_source_event_id = stable_id(
                    "event_",
                    {
                        "run": request.run_id,
                        "kind": "SIDE_EFFECT_ORPHAN_RECONCILIATION",
                        "plan_event": plan_event_id,
                    },
                )
                orphaned = side_effects.reconcile_orphaned(
                    request.run_id,
                    source_event_id=orphan_source_event_id,
                )
                if orphaned:
                    trace.record(
                        "ORPHANED_SIDE_EFFECTS_RECONCILED",
                        source_event_id=orphan_source_event_id,
                        effect_ids=list(orphaned),
                        result_known=False,
                        safe_to_resume=True,
                    )
            working_set = WorkingSetTracker(database, request.run_id)
            root_ids = tuple(item.identity_id for item in roots)
            working_set.update(
                revision_id=request.revision_id,
                source_event_id=plan_event_id,
                current_milestone_id=current.identity_id,
                dependency_milestone_ids=tuple(
                    identity_id for identity_id in root_ids if identity_id != current.identity_id
                ),
            )
            if request.plan is None and not canonical_plan_already_injected:
                assert harness_adapter is not None
                harness_adapter.inject_normalized_plan(
                    self._transport_plan_projection(plan, registry, request.run_id),
                    on_event=persist_planning_event,
                )
                canonical_plan_already_injected = True
            thread_lifecycle = ThreadLifecycle(database, metrics)
            # ContextLifecycle may have restored a newer persisted image (for
            # example after a file-change revision and a process crash). Epoch
            # continuity must bind to that authoritative digest, not to the
            # freshly constructed bootstrap candidate.
            epoch_id = thread_lifecycle.initialize(request.run_id, lifecycle.image.image_digest)
            trace.record(
                "CONTINUOUS_THREAD_INITIALIZED",
                thread_id=thread_id,
                harness_thread=request.plan is None or bool(request.harness_thread_id),
                epoch_id=epoch_id,
                context_tokens=initial_image.total_tokens,
            )
            trace.record(
                "NATIVE_COMPACTION_CAPABILITY",
                state=native_capability.state.value,
                adapter_name=native_capability.adapter_name,
                verification_supported=native_capability.verification_supported,
                reason=native_capability.reason,
            )

            rich = (
                RichGraphScheduler(
                    request.repository_path,
                    request.repository_id,
                    request.run_root / "rich-graph.sqlite3",
                    metrics=metrics,
                    processor=self.rich_processor,
                )
                if self.config.stages.rich_graph
                else None
            )
            background = BackgroundJobs(rich, trace, metrics=metrics)
            fallback = FallbackChain(
                semantic,
                local_search=BoundedLocalRepositorySearch(request.repository_path),
            )
            recall = RecallService(
                semantic_index=semantic,
                page_reader=page_store,
                metrics=metrics,
                fallback_chain=fallback,
                rich_hints=rich,
            )
            recall.max_pages = self.config.recall_max_pages
            recall.max_tokens = self.config.recall_max_tokens
            recall.max_page_bytes_read = self.config.recall_max_page_bytes_read
            recall.max_blob_bytes_read = self.config.recall_max_blob_bytes_read
            recall.max_slice_tokens = self.config.recall_max_slice_tokens
            recall.max_recovered_block_tokens = self.config.recall_max_recovered_block_tokens
            recall.max_context_admission_tokens = self.config.recall_max_context_admission_tokens
            context_transport = (
                live_driver.context_transport()
                if live_driver is not None
                and callable(getattr(live_driver, "context_transport", None))
                else None
            )
            if context_transport is not None:
                for pending, delivery_state in lifecycle.recover_pending_recall_deliveries():
                    operation_state = dynamic_tool_journal.state_for_delivery(pending.delivery_id)
                    if operation_state is not None:
                        # A synchronous tool result belongs to its original
                        # Provider call. Reinjecting it as a new Context item
                        # would duplicate delivery after a crash. The durable
                        # command is replayed/observed through its call_id.
                        trace.record(
                            "PENDING_DYNAMIC_TOOL_DELIVERY_LEFT_FOR_IDEMPOTENT_RESUME",
                            delivery_id=pending.delivery_id,
                            operation_state=operation_state,
                            delivery_state=delivery_state.value,
                        )
                        continue
                    if delivery_state.value == "PREPARED":
                        context_transport.submit_recovered(
                            delivery_id=pending.delivery_id,
                            context_digest=pending.block.content_digest,
                            rendered_content=pending.block.rendered_content,
                            max_provider_tokens=self.config.recall_max_context_admission_tokens,
                        )
                        lifecycle.transport_accepted(pending)
                        recovered_state = "TRANSPORT_ACCEPTED"
                    else:
                        context_transport.recover_delivery(
                            delivery_id=pending.delivery_id,
                            thread_id=lifecycle.image.thread_id,
                            context_digest=pending.block.content_digest,
                            state=delivery_state,
                        )
                        recovered_state = delivery_state.value
                    context_transport.request_task_continuation(
                        "Resume the same task and observe the recovered Context delivery."
                    )
                    trace.record(
                        "PENDING_CONTEXT_DELIVERY_RECOVERED",
                        delivery_id=pending.delivery_id,
                        state=recovered_state,
                        same_thread=True,
                    )
            execution_ready_ns = time.perf_counter_ns()
            execution = ExecutionCoordinator(
                request=request,
                registry=registry,
                page_store=page_store,
                recall=recall,
                semantic=semantic,
                context=lifecycle,
                side_effects=side_effects,
                thread_lifecycle=thread_lifecycle,
                epoch_admission=EpochAdmission(),
                background=background,
                metrics=metrics,
                trace=trace,
                initial_epoch_id=epoch_id,
                plan_version_id=current.plan_version_id,
                active_plan=plan,
                rich_prefetch_budget=self.config.rich_prefetch_budget,
                task_goal_digest=digest({"task": safe_task, "goal": plan.goal}),
                task_text=safe_task,
                redactor=redactor,
                raw_events=raw_events,
                reference_directory=reference_directory,
                dynamic_tool_journal=dynamic_tool_journal,
                working_set=working_set,
                provider_context=provider_context,
                context_transport=context_transport,
                fault_injector=self.fault_injector,
                trusted_verifier=trusted_verifier,
                repository_stream=self.repository_stream,
                acceptance_budgets=AcceptanceBudgets.from_mapping(self.config.acceptance_budgets),
                engagement=engagement,
                engagement_config=self.config.engagement,
                max_execution_turns=self.config.run_budget.max_execution_turns,
                run_deadline_monotonic=(
                    None
                    if self.config.run_budget.wall_clock_seconds is None
                    else run_started_monotonic + self.config.run_budget.wall_clock_seconds
                ),
            )
            if self.repository_stream is not None:
                self.repository_stream.refresh_route(execution, continue_turn=False)
                current = registry.current(request.run_id)
                plan = registry.active_plan(request.run_id)
            if request.resume_directive is not None:
                if live_driver is None or context_transport is None:
                    raise RuntimeError(
                        "an execution resume directive requires a live same-Thread Harness"
                    )
                resume_step = registry.current_step(request.run_id)
                recovery_guidance = (
                    "The prior Turn durably established a repository-external blocker from "
                    "repeated unchanged trusted verification. External state may now have changed. "
                    "Re-run the immutable verifier once before deciding: CONTINUE when it is now "
                    "viable, CORRECT only for a concrete repository change, or BLOCKED again only "
                    "if the same runtime-validated external blocker persists."
                    if request.resume_directive.cause == "MODEL_CONFIRMED_BLOCKER"
                    else (
                        "Batch observed no durable semantic progress; this is a reflection request, "
                        "not proof that the task is blocked. Continue the current Milestone from its "
                        "live workspace and route focus. If a concrete result fails, diagnose it in "
                        "normal execution; the Milestone boundary will retain the failure and open "
                        "one bounded correction review."
                    )
                )
                resume_prompt = (
                    "Resume the SAME Task, Attempt, Codex Thread, Milestone, workspace, "
                    "WAL, Page Store and Semantic Graph route. This is not permission to restart "
                    "from the base repository. Inspect the current route, workspace state, existing "
                    "tool/test evidence and failure signals. "
                    + recovery_guidance
                    + " Preserve requirement risks and current-revision verification; Step is only a "
                    "navigation focus and requires no review call or local Evidence contract. Do not "
                    "redesign future Milestones here.\n\n"
                    "Current route focus:\n"
                    + json.dumps(
                        {
                            "step_id": (resume_step["step_id"] if resume_step else None),
                            "title": (resume_step["title"] if resume_step else None),
                            "status": (resume_step["status"] if resume_step else None),
                            "expected_outcome": (
                                resume_step["expected_outcome"] if resume_step else None
                            ),
                            "risk_checklist": (
                                list(resume_step["failure_signals"]) if resume_step else []
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                context_transport.request_task_continuation(resume_prompt)
                trace.record(
                    "SAME_ATTEMPT_STALL_REFLECTION_QUEUED",
                    directive_id=request.resume_directive.directive_id,
                    cause=request.resume_directive.cause,
                    current_milestone_id=registry.current(request.run_id).canonical_id,
                    current_step_id=(resume_step["step_id"] if resume_step else None),
                    decision_authority=request.resume_directive.decision_authority,
                    allowed_step_decisions=list(request.resume_directive.allowed_step_decisions),
                    same_attempt=True,
                    same_thread=True,
                )
            if live_driver is not None:
                bind_memory_tools = getattr(live_driver, "bind_memory_tool_handler", None)
                if not callable(bind_memory_tools):
                    raise RuntimeError(
                        "live Codex Harness lacks the required dynamic memory-tool bridge"
                    )
                bind_memory_tools(
                    execution.handle_dynamic_memory_tool,
                    execution.dynamic_memory_tool_response_written,
                )
                execution.recover_pending_epoch()
                execution.recover_provider_compaction_refreshes()
            if live_driver is None:
                receipt = execution.execute(
                    request.actions if action_source is None else action_source,
                    run_started_ns=run_started_ns,
                    execution_ready_ns=execution_ready_ns,
                )
            else:
                from ..harness.revision import WorkspaceRevisionTracker

                revision_tracker = WorkspaceRevisionTracker(
                    repository_path=request.repository_path,
                    run_root=request.run_root,
                    database=database,
                    run_id=request.run_id,
                    branch_id=request.branch_id,
                    reference_directory=reference_directory,
                )
                initial_revision = revision_tracker.capture(
                    source_event_id=plan_event_id,
                    initial_receipt=request.workspace_receipt,
                ).revision_id
                if initial_revision != request.revision_id:
                    if self.repository_stream is None or recovered_plan is None:
                        raise RuntimeError("workspace changed between planning and execution initialization")
                    # A resumed long-horizon run may arrive with the revision
                    # captured before a prior Epoch/Turn committed its patch.
                    # The revision tracker is the authoritative WAL-backed
                    # observation at execution start; rejecting it here loses
                    # the very workspace continuity that Page/TPG recovery is
                    # meant to preserve. Reconcile the execution cursor and
                    # retain both IDs in the trace instead of restarting the
                    # task from the baseline.
                    trace.record(
                        "WORKSPACE_REVISION_RECONCILED",
                        requested_revision_id=request.revision_id,
                        observed_revision_id=initial_revision,
                        resume_directive=(
                            request.resume_directive.directive_id
                            if request.resume_directive is not None
                            else None
                        ),
                        same_thread=bool(request.harness_thread_id),
                        source_event_id=plan_event_id,
                    )
                    execution.revision_id = initial_revision
                if self.repository_stream is not None:
                    self.repository_stream.refresh_route(execution, continue_turn=False)
                    current = registry.current(request.run_id)
                    plan = registry.active_plan(request.run_id)
                capabilities = live_driver.capabilities()
                trace.record(
                    "HARNESS_EXECUTION_DRIVER_STARTED",
                    harness_name=capabilities.harness_name,
                    model_name=capabilities.model_name,
                    thread_id=lifecycle.image.thread_id,
                )
                trace.record(
                    "CODEX_NATIVE_COMPACTION_CAPABILITY",
                    supported=capabilities.supports_native_compaction,
                    policy_enabled=self.config.provider.native_compaction_enabled,
                    state=(
                        "AVAILABLE_UNVERIFIED"
                        if capabilities.supports_native_compaction
                        else "DISABLED_BY_RUNTIME_POLICY"
                        if not self.config.provider.native_compaction_enabled
                        else "UNSUPPORTED"
                    ),
                    success_requires_context_compaction_event=True,
                    physical_reduction_requires_later_token_usage=True,
                )
                initial_route = dict(execution.initial_semantic_route())
                if pre_execution_summary is not None:
                    initial_route["trusted_pre_execution_evidence"] = pre_execution_summary
                transport_plan = self._transport_plan_projection(
                    plan,
                    registry,
                    request.run_id,
                )
                execution_task = safe_task
                if (
                    self.repository_stream is not None
                    and getattr(self.repository_stream, "resumed_invocation", False)
                ):
                    execution_task = (
                        "Resume the SAME repository Task and Codex Thread from the authoritative "
                        "repository_execution route card, current workspace revision, TPG, Pages "
                        "and recovered code surfaces below. The original Task remains in Thread "
                        "history; do not restart from its first requirement or re-audit unchanged "
                        "files. Revalidate a recalled fact only when its anchored file or symbol "
                        "changed at the current revision."
                    )
                receipt = execution.execute_harness(
                    live_driver.events(
                        user_task=execution_task,
                        run_id=request.run_id,
                        branch_id=request.branch_id,
                        revision_tracker=revision_tracker,
                        initial_milestone=next(
                            item
                            for item in transport_plan.milestones
                            if item.canonical_id == current.canonical_id
                        ),
                        initial_semantic_route=initial_route,
                    ),
                    run_started_ns=run_started_ns,
                    execution_ready_ns=execution_ready_ns,
                )
            # SWE-Milestone may finish the official project stream without a
            # model-authored Milestone claim or submission tag. In that case
            # the normal boundary hook has no opportunity to call the host
            # verifier and older campaigns published ``attempt_count=0``.
            # Run one deterministic terminal observation only when no earlier
            # attempt exists. It writes the ordinary verifier receipt, adds no
            # model Turn, and never changes route/acceptance state.
            if (
                trusted_verifier is not None
                and getattr(trusted_verifier, "terminal_observation_enabled", False)
                is True
                and int(getattr(trusted_verifier, "attempt", 0)) == 0
            ):
                trace.record(
                    "TERMINAL_TRUSTED_VERIFIER_STARTED",
                    revision_id=receipt.revision_id,
                    model_protocol_required=False,
                )
                try:
                    terminal_verifier_result = trusted_verifier()
                except Exception as exc:
                    trace.record(
                        "TERMINAL_TRUSTED_VERIFIER_UNAVAILABLE",
                        revision_id=receipt.revision_id,
                        error_type=type(exc).__name__,
                        model_protocol_required=False,
                    )
                else:
                    trace.record(
                        "TERMINAL_TRUSTED_VERIFIER_RECORDED",
                        revision_id=receipt.revision_id,
                        passed=(
                            terminal_verifier_result.get("passed")
                            if isinstance(terminal_verifier_result, Mapping)
                            else None
                        ),
                        receipt_path=(
                            terminal_verifier_result.get("receipt_path")
                            if isinstance(terminal_verifier_result, Mapping)
                            else None
                        ),
                        model_protocol_required=False,
                    )
            background.close()
            background = None
            verification = VerificationCoordinator(database, registry, trace).verify_run(
                request.run_id, finalize_task=self.repository_stream is None
            )
            snapshot = metrics.snapshot()
            current = registry.current(request.run_id)
            milestone_statuses = registry.milestone_statuses(request.run_id)
            unresolved_questions = tuple(
                str(row["canonical_entity_id"])
                for row in database.connection.execute(
                    "SELECT DISTINCT canonical_entity_id FROM v2_semantic_evidence "
                    "WHERE run_id=? AND evidence_type='UNRESOLVED_QUESTION' "
                    "AND valid_to_cursor IS NULL ORDER BY canonical_entity_id",
                    (request.run_id,),
                ).fetchall()
            )
            latest_provider = provider_context.latest(lifecycle.image.thread_id)
            final_task_status = registry.task_status(request.run_id)
            persisted_result_path = result_path
            if final_task_status is TaskStatus.BLOCKED:
                state_row = database.connection.execute(
                    "SELECT created_cursor FROM v2_task_state_events WHERE run_id=? "
                    "ORDER BY created_cursor DESC LIMIT 1",
                    (request.run_id,),
                ).fetchone()
                if state_row is None:
                    raise RuntimeError("blocked Task has no durable state event")
                persisted_result_path = request.run_root / (
                    f"suspension-{int(state_row['created_cursor']):08d}.json"
                )
            result = RunResult(
                run_id=request.run_id,
                repository_id=request.repository_id,
                branch_id=request.branch_id,
                revision_id=receipt.revision_id,
                thread_id=lifecycle.image.thread_id,
                epoch_id=receipt.epoch_id,
                plan_version_id=current.plan_version_id,
                current_milestone_id=current.identity_id,
                page_ids=receipt.page_ids,
                context_image_digest=lifecycle.image.image_digest,
                context_tokens=lifecycle.image.total_tokens,
                trace=trace.snapshot(),
                metrics=snapshot.values,
                metric_samples=snapshot.samples,
                result_path=str(persisted_result_path),
                task_status=final_task_status.value,
                milestone_statuses=milestone_statuses,
                unmet_completion_criteria=verification.unmet_criteria,
                unresolved_questions=unresolved_questions,
                failed_tests=receipt.failed_tests,
                provider_pressure=(
                    latest_provider.pressure.value
                    if latest_provider is not None and latest_provider.pressure is not None
                    else None
                ),
                final_review_disposition=verification.final_review_disposition,
                completion_verdict=verification.completion_verdict,
                build_identity=build_identity.as_mapping(),
                unmet_final_criteria=verification.unmet_final_criteria,
                unverified_final_criteria=verification.unverified_final_criteria,
                route_stalls=registry.route_stalls(request.run_id),
                engagement=engagement.as_mapping(),
            )
            atomic_write_once(
                persisted_result_path,
                json.dumps(
                    {
                        **primitive(result),
                        "metric_semantics": snapshot.semantics,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ).encode("utf-8"),
            )
            if self.repository_stream is not None:
                self.repository_stream.publish(result)
            return result
        except BaseException as exc:
            # Normal protocol/provider failures must leave the same diagnostic
            # evidence as successful runs. This is best-effort and must never
            # mask the original exception or weaken Page WAL recovery.
            try:
                metric_snapshot = metrics.snapshot()
                failure_payload, _ = redactor.redact(
                    {
                        "schema": "codex-longterm-v2/runtime-failure@1",
                        "run_id": request.run_id,
                        "branch_id": request.branch_id,
                        "revision_id": request.revision_id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:4000],
                        "trace": primitive(trace.snapshot()),
                        "metrics": primitive(metric_snapshot.values),
                        "metric_samples": primitive(metric_snapshot.samples),
                        "metric_semantics": metric_snapshot.semantics,
                        "build_identity": build_identity.as_mapping(),
                    }
                )
                failure_path = result_path.parent / "failure.json"
                if not failure_path.exists():
                    atomic_write_once(
                        failure_path,
                        json.dumps(
                            failure_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            indent=2,
                        ).encode("utf-8"),
                    )
            except Exception:
                pass
            # Preserve any complete groups as an explicit recovery Tail without
            # hiding the original failure.
            try:
                page_store.close(TailReason.CRASH_RECOVERY)
            except Exception:
                pass
            raise
        finally:
            if background is not None:
                background.close()
            database.close()
