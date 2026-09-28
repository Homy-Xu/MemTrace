from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from memtrace.context_runtime import (
    Compactor,
    ContextAdmission,
    ContextBudget,
    ContextImageBuilder,
    ContextLifecycle,
    DeliveryJournal,
    EpochAdmission,
    NativeCompactionCapabilityReceipt,
    NativeCompactionState,
    PressurePolicy,
    ThreadLifecycle,
    artifact_from_content,
)
from memtrace.contracts import (
    ContextHandle,
    ContextImage,
    CoverageReceipt,
    CoverageState,
    DeliveryState,
    EpochReason,
    FallbackStage,
    PageSlice,
    PressureLevel,
    RecoveredContextBlock,
    Representation,
    RichGraphState,
    digest,
    stable_id,
)
from memtrace.database import StateDatabase
from memtrace.observability import MetricRecorder


@dataclass
class Runtime:
    database: StateDatabase
    metrics: MetricRecorder
    builder: ContextImageBuilder
    delivery: DeliveryJournal
    lifecycle: ContextLifecycle
    threads: ThreadLifecycle


def _runtime(
    root: Path,
    *,
    model_limit: int,
    artifacts: tuple = (),
    run_id: str = "run-0",
    branch_id: str = "main",
    thread_id: str = "thread-0",
    milestone_id: str = "milestone-current",
    revision_id: str = "revision-1",
    native_adapter: object | None = None,
) -> Runtime:
    database = StateDatabase(root / "state.sqlite3")
    metrics = MetricRecorder()
    builder = ContextImageBuilder()
    admission = ContextAdmission(
        PressurePolicy(
            ContextBudget(
                model_limit=model_limit,
                system_overhead=0,
                tool_schema_tokens=0,
                output_reserve=0,
                safety_margin=0,
            )
        ),
        Compactor(metrics, native_adapter),  # type: ignore[arg-type]
        builder,
        metrics,
    )
    initial = builder.build(
        thread_id=thread_id,
        artifacts=artifacts,
        current_milestone_id=milestone_id,
        revision_id=revision_id,
    )
    delivery = DeliveryJournal(database)
    lifecycle = ContextLifecycle(
        run_id=run_id,
        branch_id=branch_id,
        admission=admission,
        delivery=delivery,
        builder=builder,
        metrics=metrics,
        initial_image=initial,
    )
    return Runtime(
        database=database,
        metrics=metrics,
        builder=builder,
        delivery=delivery,
        lifecycle=lifecycle,
        threads=ThreadLifecycle(database, metrics),
    )


def _handle(seed: str = "one") -> ContextHandle:
    return ContextHandle(
        page_id=f"page-{seed}",
        event_range=(10, 20),
        blob_handle=None,
        blob_range=None,
        content_digest=digest({"page": seed}),
        revision_id="revision-1",
    )


def test_provider_compaction_refresh_uses_the_durable_delivery_journal(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, model_limit=4_096)

    prepared = runtime.delivery.prepare_provider_compaction_refresh(
        thread_id="thread-0",
        source_event_id="provider-compaction-1",
        rendered_content="bounded Working Set refresh",
    )
    duplicate = runtime.delivery.prepare_provider_compaction_refresh(
        thread_id="thread-0",
        source_event_id="provider-compaction-1",
        rendered_content="bounded Working Set refresh",
    )

    assert duplicate.delivery_id == prepared.delivery_id
    assert prepared.state is DeliveryState.PREPARED
    runtime.delivery.advance_to(
        prepared.delivery_id,
        DeliveryState.CONTEXT_COMMITTED,
        context_digest=prepared.context_digest,
    )
    recovered = runtime.delivery.pending_provider_compaction_refreshes(thread_id="thread-0")
    assert len(recovered) == 1
    assert recovered[0].state is DeliveryState.CONTEXT_COMMITTED
    assert recovered[0].rendered_content == "bounded Working Set refresh"
    runtime.database.close()


def _recovered_block() -> RecoveredContextBlock:
    page_slice = PageSlice(
        slice_id="slice-1",
        page_id="page-recalled",
        level="L2",
        event_range=(5, 9),
        event_ids=("event-5", "event-9"),
        event_group_ids=("group-1",),
        evidence_ids=("evidence-1",),
        evidence_keys=("key-1",),
        revision_id="revision-1",
        page_digest=digest({"page": "recalled"}),
        content={"events": ["historical fact"]},
        token_count=12,
        content_digest=digest({"slice": "recalled"}),
    )
    coverage = CoverageReceipt(
        state=CoverageState.COMPLETE,
        required_keys=("key-1",),
        covered_keys=("key-1",),
        missing_keys=(),
        validated_page_ids=("page-recalled",),
        fallback_stage=FallbackStage.SEMANTIC_EXACT,
        freshness="EXACT_REVISION",
        rich_capability_state=RichGraphState.NOT_STARTED,
    )
    rendered = (
        "<UNTRUSTED_RECOVERED_DATA>\n"
        + "historical evidence " * 24
        + "\n</UNTRUSTED_RECOVERED_DATA>"
    )
    content_digest = digest({"rendered_content": rendered})
    return RecoveredContextBlock(
        block_id=stable_id("recovered_", content_digest),
        recall_id="recall-1",
        slices=(page_slice,),
        coverage=coverage,
        current_milestone_id="milestone-current",
        revision_id="revision-1",
        rendered_content=rendered,
        token_count=max(1, (len(rendered.encode("utf-8")) + 2) // 3),
        content_digest=content_digest,
        untrusted_data=True,
    )


class _VerifiedNativeAdapter:
    def __init__(self) -> None:
        self.builder = ContextImageBuilder()
        self.verify_called = False

    def capability(self) -> NativeCompactionCapabilityReceipt:
        return NativeCompactionCapabilityReceipt(
            state=NativeCompactionState.AVAILABLE,
            adapter_name="test-verified-native",
            verification_supported=True,
            reason=None,
        )

    def compact(self, image: ContextImage, *, target_tokens: int) -> ContextImage | None:
        original = image.artifacts[0]
        compacted = artifact_from_content(
            content='{"n":1}',
            representation=Representation.VERIFIED_SUMMARY,
            milestone_ids=original.milestone_ids,
            entity_refs=original.entity_refs,
            source_handles=original.source_handles,
            verified=True,
            derived_from=tuple(dict.fromkeys((*original.derived_from, original.artifact_id))),
            identity_seed="native-verified",
        )
        assert compacted.token_count <= target_tokens
        return self.builder.build(
            thread_id=image.thread_id,
            artifacts=(compacted,),
            current_milestone_id=image.current_milestone_id,
            revision_id=image.revision_id,
        )

    def verify(self, source: ContextImage, candidate: ContextImage) -> bool:
        self.verify_called = True
        return (
            candidate.artifacts[0].content == '{"n":1}'
            and source.artifacts[0].artifact_id in candidate.artifacts[0].derived_from
        )


class _FailingNativeAdapter(_VerifiedNativeAdapter):
    def compact(self, image: ContextImage, *, target_tokens: int) -> ContextImage | None:
        raise RuntimeError("provider native compaction unavailable")


def test_simple_task_keeps_full_content_and_restores_persisted_image(
    tmp_path: Path,
) -> None:
    core = artifact_from_content(
        content="task, goal, plan and current milestone",
        representation=Representation.FULL,
        milestone_ids=("milestone-current",),
        must_preserve=True,
        current_milestone=True,
        identity_seed="core",
    )
    runtime = _runtime(tmp_path, model_limit=4_000, artifacts=(core,))
    action = artifact_from_content(
        content="small complete coding action",
        representation=Representation.FULL,
        milestone_ids=("milestone-current",),
        must_preserve=True,
        identity_seed="action",
    )

    outcome = runtime.lifecycle.admit_artifacts(
        (action,), working_set_milestones=("milestone-current",)
    )

    assert outcome.pressure == PressureLevel.NORMAL
    assert outcome.final_pressure == PressureLevel.NORMAL
    assert [item.representation for item in outcome.image.artifacts] == [
        Representation.FULL,
        Representation.FULL,
    ]
    assert runtime.metrics.snapshot().values["compression_count"] == 0
    assert runtime.metrics.snapshot().values["epoch_count"] == 0

    # Reconstructing the formal lifecycle from the same DB restores its
    # authoritative image instead of overwriting it with the seed image.
    restored = ContextLifecycle(
        run_id="run-0",
        branch_id="main",
        admission=runtime.lifecycle.admission,
        delivery=runtime.delivery,
        builder=runtime.builder,
        metrics=runtime.metrics,
        initial_image=runtime.builder.build(
            thread_id="thread-0",
            artifacts=(core,),
            current_milestone_id="milestone-current",
            revision_id="revision-1",
        ),
    )
    assert restored.image.image_digest == outcome.image.image_digest
    assert {item.artifact_id for item in restored.image.artifacts} == {
        core.artifact_id,
        action.artifact_id,
    }


def test_pressure_accounts_for_entire_image_and_all_fixed_reserves(
    tmp_path: Path,
) -> None:
    database = StateDatabase(tmp_path / "budget.sqlite3")
    metrics = MetricRecorder()
    builder = ContextImageBuilder()
    current_artifact = artifact_from_content(
        content="x" * 900,
        representation=Representation.FULL,
        must_preserve=True,
        identity_seed="already-resident",
    )
    current = builder.build(
        thread_id="thread-budget",
        artifacts=(current_artifact,),
        current_milestone_id="milestone-current",
        revision_id="revision-1",
    )
    budget = ContextBudget(
        model_limit=1_000,
        system_overhead=100,
        tool_schema_tokens=100,
        output_reserve=100,
        safety_margin=100,
    )
    admission = ContextAdmission(PressurePolicy(budget), Compactor(metrics), builder, metrics)
    lifecycle = ContextLifecycle(
        run_id="run-budget",
        branch_id="main",
        admission=admission,
        delivery=DeliveryJournal(database),
        builder=builder,
        metrics=metrics,
        initial_image=current,
    )
    incoming = artifact_from_content(
        content="y" * 210,
        representation=Representation.FULL,
        must_preserve=True,
        identity_seed="incoming",
    )

    outcome = lifecycle.admit_artifacts(
        (incoming,),
        working_set_milestones=("milestone-current",),
    )

    assert budget.effective_limit == 600
    assert incoming.token_count < budget.soft_ratio * budget.effective_limit
    assert outcome.projected_tokens == current.total_tokens + incoming.token_count
    assert outcome.pressure == PressureLevel.SOFT
    assert outcome.image.artifacts == (current_artifact, incoming)
    database.close()


def test_long_pressure_real_chain_evicts_outside_working_set_first(
    tmp_path: Path,
) -> None:
    core = artifact_from_content(
        content="core continuity",
        representation=Representation.FULL,
        milestone_ids=("milestone-current",),
        must_preserve=True,
        current_milestone=True,
        identity_seed="core",
    )
    dependency = artifact_from_content(
        content="explicit dependency fact " * 2,
        representation=Representation.FULL,
        milestone_ids=("milestone-dependency",),
        identity_seed="dependency",
    )
    historical = artifact_from_content(
        content="\n".join(f"historical line {index} " + ("x" * 500) for index in range(50)),
        representation=Representation.FULL,
        milestone_ids=("milestone-unrelated",),
        source_handles=(_handle("historical"),),
        identity_seed="historical",
    )
    runtime = _runtime(
        tmp_path,
        model_limit=150,
        artifacts=(core, dependency, historical),
    )

    outcome = runtime.lifecycle.admit_artifacts(
        (),
        working_set_milestones=(
            "milestone-current",
            "milestone-dependency",
        ),
    )

    by_milestone = {item.milestone_ids[0]: item for item in outcome.image.artifacts}
    evicted = by_milestone["milestone-unrelated"]
    assert evicted.representation in {Representation.HANDLE, Representation.NONRESIDENT}
    assert evicted.source_handles
    assert evicted.source_handles == (_handle("historical"),)
    assert evicted.memory_ref == historical.memory_ref
    assert evicted.memory_ref is not None
    if evicted.representation == Representation.HANDLE:
        handle = json.loads(evicted.content)
        assert handle["memory_ref"].startswith("memoryref_")
        assert handle["recall_required_before_use"] is True
        assert "page_id" not in handle
        assert "event_range" not in handle
        assert evicted.token_count > 0
        assert len(evicted.derived_from) == 3  # FULL→SLICE→SUMMARY→HANDLE
    else:
        assert evicted.content == ""
        assert evicted.token_count == 0
        assert len(evicted.derived_from) == 4  # ...→HANDLE→NONRESIDENT
    assert by_milestone["milestone-dependency"].representation == Representation.FULL
    assert runtime.metrics.snapshot().values["compression_count"] == len(evicted.derived_from)
    assert outcome.final_pressure in {PressureLevel.NORMAL, PressureLevel.SOFT}


def test_virtual_section_has_one_current_residency_representation() -> None:
    handle = ContextHandle(
        page_id="page-worker",
        event_range=(10, 20),
        blob_handle=None,
        blob_range=None,
        content_digest=digest({"page": "worker"}),
        revision_id="revision-1",
    )
    memory_ref = "memoryref_worker_section"
    summary = artifact_from_content(
        content=json.dumps({"memory_ref": memory_ref, "summary": "old summary"}),
        representation=Representation.VERIFIED_SUMMARY,
        entity_refs=("symbol:src/worker.py:Worker.run",),
        source_handles=(handle,),
        memory_ref=memory_ref,
        identity_seed="worker-summary",
    )
    recovered = artifact_from_content(
        content="exact Worker.run implementation body",
        representation=Representation.SEMANTIC_SLICE,
        entity_refs=("context:recalled_slice", "symbol:src/worker.py:Worker.run"),
        source_handles=(handle,),
        memory_ref=memory_ref,
        identity_seed="worker-recovered",
    )

    resident = ContextAdmission._deduplicate((summary, recovered))

    assert len(resident) == 1
    assert resident[0].representation is Representation.SEMANTIC_SLICE
    assert resident[0].content == recovered.content
    assert resident[0].memory_ref == memory_ref
    assert summary.artifact_id in resident[0].derived_from


def test_unlocatable_artifact_reaches_safe_fixed_point_not_illegal_handle(
    tmp_path: Path,
) -> None:
    unlocatable = artifact_from_content(
        content="\n".join(f"unsealed line {index} " + ("x" * 500) for index in range(30)),
        representation=Representation.FULL,
        milestone_ids=("milestone-old",),
        identity_seed="open-wal-fact",
    )
    runtime = _runtime(tmp_path, model_limit=40, artifacts=(unlocatable,))

    outcome = runtime.lifecycle.admit_artifacts((), working_set_milestones=("milestone-current",))

    assert outcome.fixed_point is True
    assert outcome.fixed_point_reason == "NO_SAFE_REPRESENTATION_DEMOTION"
    assert outcome.image.artifacts[0].representation == Representation.VERIFIED_SUMMARY
    assert outcome.image.artifacts[0].source_handles == ()
    assert runtime.metrics.snapshot().values["compression_count"] == 2


def test_native_capability_unavailable_and_failure_keeps_safe_fallback(
    tmp_path: Path,
) -> None:
    no_adapter_metrics = MetricRecorder()
    unavailable = Compactor(no_adapter_metrics).native_capability()
    assert unavailable.state == NativeCompactionState.UNAVAILABLE
    assert unavailable.reason == "NO_NATIVE_COMPACTION_ADAPTER"
    assert no_adapter_metrics.snapshot().values["native_compaction_count"] == 0

    unlocatable = artifact_from_content(
        content="\n".join(f"native failure line {index} " + ("x" * 500) for index in range(30)),
        representation=Representation.FULL,
        milestone_ids=("milestone-old",),
        identity_seed="native-failure-source",
    )
    runtime = _runtime(
        tmp_path,
        model_limit=40,
        artifacts=(unlocatable,),
        native_adapter=_FailingNativeAdapter(),
    )
    outcome = runtime.lifecycle.admit_artifacts((), working_set_milestones=("milestone-current",))

    assert outcome.fixed_point is True
    assert outcome.image.artifacts[0].representation == Representation.VERIFIED_SUMMARY
    assert runtime.metrics.snapshot().values["compression_count"] == 2
    assert runtime.metrics.snapshot().values["native_compaction_count"] == 0
    failure = runtime.lifecycle.admission.compactor.native_capability()
    assert failure.state == NativeCompactionState.FAILED
    assert failure.reason == "NATIVE_COMPACTION_FAILED:RuntimeError"


def test_verified_native_compaction_changes_content_and_counts_once(
    tmp_path: Path,
) -> None:
    source = artifact_from_content(
        content="\n".join(f"native source line {index} " + ("x" * 500) for index in range(30)),
        representation=Representation.FULL,
        milestone_ids=("milestone-old",),
        identity_seed="native-success-source",
    )
    adapter = _VerifiedNativeAdapter()
    runtime = _runtime(
        tmp_path,
        model_limit=40,
        artifacts=(source,),
        native_adapter=adapter,
    )

    outcome = runtime.lifecycle.admit_artifacts((), working_set_milestones=("milestone-current",))

    assert adapter.verify_called is True
    assert outcome.fixed_point is False
    assert outcome.final_pressure == PressureLevel.NORMAL
    assert outcome.image.artifacts[0].content == '{"n":1}'
    assert source.artifact_id in outcome.image.artifacts[0].derived_from
    assert runtime.metrics.snapshot().values["compression_count"] == 2
    assert runtime.metrics.snapshot().values["native_compaction_count"] == 1
    capability = runtime.lifecycle.admission.compactor.native_capability()
    assert capability.state == NativeCompactionState.AVAILABLE


def test_page_fault_delivery_pin_and_non_epoch_operations(
    tmp_path: Path,
) -> None:
    core = artifact_from_content(
        content="core",
        representation=Representation.FULL,
        must_preserve=True,
        current_milestone=True,
        identity_seed="core",
    )
    runtime = _runtime(tmp_path, model_limit=220, artifacts=(core,))
    initial_epoch = runtime.threads.initialize(
        "run-page-fault", runtime.lifecycle.image.image_digest
    )
    block = _recovered_block()

    pending = runtime.lifecycle.prepare_recovered(block)
    assert pending is not None
    assert runtime.delivery.state(pending.delivery_id) == DeliveryState.PREPARED
    assert len(runtime.lifecycle.image.artifacts) == 1

    runtime.lifecycle.transport_accepted(pending)
    assert len(runtime.lifecycle.image.artifacts) == 1
    runtime.lifecycle.context_committed(pending)
    assert len(runtime.lifecycle.image.artifacts) == 1
    observed = runtime.lifecycle.model_observed(
        pending, working_set_milestones=("milestone-current",)
    )
    recalled = next(
        item
        for item in observed.image.artifacts
        if item.representation == Representation.SEMANTIC_SLICE
    )
    assert runtime.delivery.state(pending.delivery_id) == DeliveryState.MODEL_OBSERVED
    assert recalled.soft_pin_boundaries == 2
    assert runtime.lifecycle.prepare_recovered(block) is None

    runtime.lifecycle.advance_boundary()
    runtime.lifecycle.admit_artifacts((), working_set_milestones=("milestone-current",))
    recalled = next(item for item in runtime.lifecycle.image.artifacts if item.source_handles)
    assert recalled.soft_pin_boundaries == 1
    assert recalled.representation == Representation.SEMANTIC_SLICE

    runtime.lifecycle.advance_boundary()
    after_pin = runtime.lifecycle.admit_artifacts((), working_set_milestones=("milestone-current",))
    recalled = next(item for item in after_pin.image.artifacts if item.source_handles)
    assert recalled.soft_pin_boundaries == 0
    assert recalled.representation != Representation.SEMANTIC_SLICE

    # Milestone switch and an ordinary repair admission remain in the Thread.
    runtime.lifecycle.switch_scope(current_milestone_id="milestone-next", revision_id="revision-2")
    repair = artifact_from_content(
        content="repair result",
        representation=Representation.FULL,
        must_preserve=True,
        identity_seed="repair",
    )
    runtime.lifecycle.admit_artifacts((repair,), working_set_milestones=("milestone-next",))
    assert runtime.lifecycle.image.thread_id == "thread-0"
    assert runtime.threads.active_epoch("run-page-fault") == initial_epoch
    assert runtime.metrics.snapshot().values["epoch_count"] == 0


def test_memoryref_exact_section_coalesces_pending_and_resident_deliveries(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, model_limit=4_096)
    block = replace(_recovered_block(), source_memory_ref="memoryref_exact")
    differently_rendered = replace(
        block,
        block_id="block-same-address-new-purpose",
        recall_id="recall-same-address-new-purpose",
        rendered_content=block.rendered_content + "\nUse for a different stated purpose.",
        content_digest=digest(
            {"rendered_content": block.rendered_content + "\nUse for a different stated purpose."}
        ),
    )

    pending = runtime.lifecycle.prepare_recovered(block)
    assert pending is not None
    assert runtime.lifecycle.pending_recovered(differently_rendered) == pending
    assert runtime.lifecycle.prepare_recovered(differently_rendered) == pending

    runtime.lifecycle.transport_accepted(pending)
    runtime.lifecycle.context_committed(pending)
    runtime.lifecycle.model_observed(pending, working_set_milestones=("milestone-current",))

    resident = runtime.lifecycle.resident_recovered(differently_rendered)
    assert resident is not None
    assert resident.representation is Representation.SEMANTIC_SLICE
    assert runtime.lifecycle.prepare_recovered(differently_rendered) is None


def test_memoryref_continuation_chunk_is_not_coalesced_with_resident_prefix(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, model_limit=4_096)
    base = _recovered_block()
    prefix_slice = replace(
        base.slices[0],
        slice_id="slice-prefix",
        section_handle="section_" + ("1" * 32),
        content={"__direct_section_chunk__": {"content_json_excerpt": "prefix"}},
        content_digest=digest({"chunk": "prefix"}),
        continuation={
            "continuation_token": "sectioncontinuation_a_" + ("2" * 24),
            "section_content_incomplete": True,
        },
    )
    prefix = replace(
        base,
        block_id="block-prefix",
        source_memory_ref="memoryref_continued",
        slices=(prefix_slice,),
        rendered_content="prefix",
        token_count=2,
        content_digest=digest({"rendered_content": "prefix"}),
    )
    pending = runtime.lifecycle.prepare_recovered(prefix)
    assert pending is not None
    runtime.lifecycle.transport_accepted(pending)
    runtime.lifecycle.context_committed(pending)
    runtime.lifecycle.model_observed(pending, working_set_milestones=("milestone-current",))

    continuation_slice = replace(
        prefix_slice,
        slice_id="slice-continuation",
        content={"__direct_section_chunk__": {"content_json_excerpt": "continuation"}},
        content_digest=digest({"chunk": "continuation"}),
        continuation={},
    )
    continuation = replace(
        prefix,
        block_id="block-continuation",
        recall_id="recall-continuation",
        slices=(continuation_slice,),
        rendered_content="continuation",
        token_count=4,
        content_digest=digest({"rendered_content": "continuation"}),
    )

    assert runtime.lifecycle.resident_recovered(continuation) is None
    assert runtime.lifecycle.prepare_recovered(continuation) is not None


def test_provisional_replacement_is_controlled_and_persistent(tmp_path: Path) -> None:
    provisional = artifact_from_content(
        content="open WAL action body",
        representation=Representation.FULL,
        milestone_ids=("milestone-current",),
        must_preserve=True,
        identity_seed="provisional",
    )
    runtime = _runtime(tmp_path, model_limit=2_000, artifacts=(provisional,))
    durable = artifact_from_content(
        content=provisional.content,
        representation=Representation.FULL,
        milestone_ids=("milestone-current",),
        source_handles=(_handle("sealed"),),
        identity_seed="sealed-page",
    )

    outcome = runtime.lifecycle.replace_artifacts(
        (provisional.artifact_id,),
        (durable,),
        working_set_milestones=("milestone-current",),
    )
    assert {item.artifact_id for item in outcome.image.artifacts} == {durable.artifact_id}
    assert outcome.image.artifacts[0].source_handles == (_handle("sealed"),)
    old_thread = outcome.image.thread_id
    runtime.lifecycle.switch_scope(current_milestone_id="milestone-next", revision_id="revision-2")
    assert runtime.lifecycle.image.thread_id == old_thread
    assert runtime.metrics.snapshot().values["epoch_count"] == 0


def test_page_promotion_replaces_a_compacted_provisional_descendant(tmp_path: Path) -> None:
    provisional = artifact_from_content(
        content="open semantic fact " * 200,
        representation=Representation.FULL,
        milestone_ids=("milestone-current",),
        identity_seed="open-provisional",
    )
    descendant = artifact_from_content(
        content="bounded semantic summary",
        representation=Representation.VERIFIED_SUMMARY,
        milestone_ids=("milestone-current",),
        derived_from=(provisional.artifact_id,),
        identity_seed="compacted-provisional",
    )
    runtime = _runtime(tmp_path, model_limit=2_000, artifacts=(descendant,))
    durable = artifact_from_content(
        content="page-backed summary",
        representation=Representation.VERIFIED_SUMMARY,
        milestone_ids=("milestone-current",),
        source_handles=(_handle("sealed-descendant"),),
        identity_seed="sealed-descendant",
    )

    outcome = runtime.lifecycle.replace_artifacts(
        (provisional.artifact_id,),
        (durable,),
        working_set_milestones=("milestone-current",),
    )

    assert {item.artifact_id for item in outcome.image.artifacts} == {durable.artifact_id}
    assert all(provisional.artifact_id not in item.derived_from for item in outcome.image.artifacts)


def test_epoch_full_admission_delivery_and_atomic_fence(tmp_path: Path) -> None:
    # No recoverable Page handle exists yet, so logical compaction reaches a
    # real fixed point at a verified resident Summary.
    open_fact = artifact_from_content(
        content="\n".join(f"open fact {index} " + ("z" * 500) for index in range(30)),
        representation=Representation.FULL,
        milestone_ids=("milestone-old",),
        identity_seed="open-fact",
    )
    runtime = _runtime(
        tmp_path,
        model_limit=40,
        artifacts=(open_fact,),
        run_id="run-epoch",
    )
    pressure = runtime.lifecycle.admit_artifacts((), working_set_milestones=("milestone-current",))
    assert pressure.fixed_point

    checkpoint = runtime.lifecycle.build_continuity_checkpoint(
        task_goal_digest=digest({"task": "goal"}),
        plan_version_id="plan-version-1",
        milestone_state_digest=digest({"milestone": "state"}),
        workspace_revision_id="revision-1",
        uncommitted_changes=("src/open.py",),
        unresolved_questions=("provider continuation?",),
        failing_tests=("test_open",),
        pending_side_effects=(),
        safe_action_boundary=True,
    )
    admission = EpochAdmission()
    denied_page_fault = admission.evaluate(
        reason="PAGE_FAULT",  # type: ignore[arg-type]
        checkpoint=checkpoint,
        compression_fixed_point=True,
        physical_context_cannot_continue=True,
        candidate_context_tokens=runtime.lifecycle.image.total_tokens,
        safe_budget_tokens=2_000,
        candidate_context_digest=runtime.lifecycle.image.image_digest,
    )
    assert denied_page_fault.admitted is False
    assert denied_page_fault.reason == "REASON_NOT_ALLOWED"
    denied_without_physical_proof = admission.evaluate(
        reason=EpochReason.COMPRESSION_FIXED_POINT,
        checkpoint=checkpoint,
        compression_fixed_point=pressure.fixed_point,
        physical_context_cannot_continue=False,
        candidate_context_tokens=runtime.lifecycle.image.total_tokens,
        safe_budget_tokens=2_000,
        candidate_context_digest=runtime.lifecycle.image.image_digest,
    )
    assert denied_without_physical_proof.admitted is False

    decision = admission.evaluate(
        reason=EpochReason.COMPRESSION_FIXED_POINT,
        checkpoint=checkpoint,
        compression_fixed_point=pressure.fixed_point,
        physical_context_cannot_continue=True,
        candidate_context_tokens=runtime.lifecycle.image.total_tokens,
        safe_budget_tokens=2_000,
        candidate_context_digest=runtime.lifecycle.image.image_digest,
    )
    assert decision.admitted is True

    run_id = "run-epoch"
    predecessor = runtime.threads.initialize(run_id, runtime.lifecycle.image.image_digest)
    epoch_id = runtime.threads.create_pending(
        run_id=run_id,
        reason=EpochReason.COMPRESSION_FIXED_POINT,
        context_digest=runtime.lifecycle.image.image_digest,
        decision=decision,
    )
    candidate, delivery_id = runtime.lifecycle.prepare_thread_replacement(
        new_thread_id=epoch_id, epoch_id=epoch_id
    )
    assert candidate.image_digest == checkpoint.context_digest
    runtime.threads.bind_delivery(epoch_id, delivery_id)

    with pytest.raises(RuntimeError, match="not MODEL_OBSERVED"):
        runtime.lifecycle.replace_thread_after_model_observed(
            epoch_id, epoch_id=epoch_id, thread_lifecycle=runtime.threads
        )
    assert runtime.threads.state(predecessor) == "ACTIVE"
    assert runtime.threads.state(epoch_id) == "PENDING"

    runtime.delivery.advance(
        delivery_id,
        DeliveryState.TRANSPORT_ACCEPTED,
        context_digest=candidate.image_digest,
    )
    runtime.delivery.advance(
        delivery_id,
        DeliveryState.CONTEXT_COMMITTED,
        context_digest=candidate.image_digest,
    )
    with pytest.raises(RuntimeError, match="MODEL_OBSERVED"):
        runtime.threads.activate_after_model_observed(epoch_id)
    assert runtime.threads.state(predecessor) == "ACTIVE"

    runtime.delivery.advance(
        delivery_id,
        DeliveryState.MODEL_OBSERVED,
        context_digest=candidate.image_digest,
    )
    runtime.lifecycle.replace_thread_after_model_observed(
        epoch_id, epoch_id=epoch_id, thread_lifecycle=runtime.threads
    )

    assert runtime.lifecycle.image.thread_id == epoch_id
    assert runtime.lifecycle.image.image_digest == checkpoint.context_digest
    assert runtime.threads.state(predecessor) == "FENCED"
    assert runtime.threads.state(epoch_id) == "ACTIVE"
    assert runtime.threads.active_epoch(run_id) == epoch_id
    assert runtime.metrics.snapshot().values["epoch_count"] == 1

    # The stable runtime row remains writable after the carrier Thread changes.
    runtime.lifecycle.advance_boundary()
    post_epoch = artifact_from_content(
        content="first action in replacement thread",
        representation=Representation.FULL,
        must_preserve=True,
        identity_seed="post-epoch",
    )
    runtime.lifecycle.admit_artifacts((post_epoch,), working_set_milestones=("milestone-current",))
    assert post_epoch.artifact_id in {
        item.artifact_id for item in runtime.lifecycle.image.artifacts
    }
    assert runtime.lifecycle.image.thread_id == epoch_id

    # Crash reconstruction begins from the stable initial thread identity and
    # still restores the replacement carrier and its next action.
    restored = ContextLifecycle(
        run_id="run-epoch",
        branch_id="main",
        admission=runtime.lifecycle.admission,
        delivery=runtime.delivery,
        builder=runtime.builder,
        metrics=runtime.metrics,
        initial_image=runtime.builder.build(
            thread_id="thread-0",
            artifacts=(open_fact,),
            current_milestone_id="milestone-current",
            revision_id="revision-1",
        ),
    )
    assert restored.image.thread_id == epoch_id
    assert post_epoch.artifact_id in {item.artifact_id for item in restored.image.artifacts}
    restored_from_active_alias = ContextLifecycle(
        run_id="run-epoch",
        branch_id="main",
        admission=runtime.lifecycle.admission,
        delivery=runtime.delivery,
        builder=runtime.builder,
        metrics=runtime.metrics,
        initial_image=runtime.builder.build(
            thread_id=epoch_id,
            artifacts=(open_fact,),
            current_milestone_id="milestone-current",
            revision_id="revision-1",
        ),
    )
    assert restored_from_active_alias.image.image_digest == restored.image.image_digest
    recovered_threads = ThreadLifecycle(runtime.database, runtime.metrics)
    assert recovered_threads.initialize(run_id, checkpoint.context_digest) == epoch_id


def test_context_image_isolated_by_run_even_when_thread_id_is_reused(tmp_path: Path) -> None:
    first = artifact_from_content(
        content="run A private context",
        representation=Representation.FULL,
        identity_seed="run-a-private",
    )
    run_a = _runtime(
        tmp_path,
        model_limit=4_000,
        artifacts=(first,),
        run_id="run-A",
        thread_id="shared-thread",
    )
    retained = artifact_from_content(
        content="run A retained detail",
        representation=Representation.FULL,
        identity_seed="run-a-retained",
    )
    run_a.lifecycle.admit_artifacts((retained,), working_set_milestones=())

    second = artifact_from_content(
        content="run B clean context",
        representation=Representation.FULL,
        identity_seed="run-b-clean",
    )
    run_b = _runtime(
        tmp_path,
        model_limit=4_000,
        artifacts=(second,),
        run_id="run-B",
        thread_id="shared-thread",
    )

    assert {item.artifact_id for item in run_b.lifecycle.image.artifacts} == {second.artifact_id}
    assert first.artifact_id not in {item.artifact_id for item in run_b.lifecycle.image.artifacts}
    rows = run_b.database.connection.execute(
        "SELECT run_id,thread_id FROM context_image_state_v2 ORDER BY run_id"
    ).fetchall()
    assert [(str(row["run_id"]), str(row["thread_id"])) for row in rows] == [
        ("run-A", "shared-thread"),
        ("run-B", "shared-thread"),
    ]


def test_consumed_transition_handoff_is_compressed_and_unpinned(tmp_path: Path) -> None:
    handoff = artifact_from_content(
        content='{"kind":"MILESTONE_HANDOFF","detail":"verified predecessor state"}',
        representation=Representation.SEMANTIC_SLICE,
        milestone_ids=("milestone-1", "milestone-2"),
        entity_refs=("context:milestone_handoff", "file:worker.py"),
        source_handles=(_handle("handoff"),),
        soft_pin_boundaries=16,
        identity_seed="handoff",
    )
    runtime = _runtime(
        tmp_path,
        model_limit=4_000,
        artifacts=(handoff,),
        milestone_id="milestone-2",
    )

    # This is a semantic lease, not a timer: generic compaction cannot demote
    # it after the old soft-pin counter reaches zero.
    for _ in range(20):
        runtime.lifecycle.advance_boundary()
    compacted = runtime.lifecycle.compact_to_fixed_point(working_set_milestones=("milestone-2",))
    leased = next(
        item for item in compacted.image.artifacts if item.artifact_id == handoff.artifact_id
    )
    assert leased.representation is Representation.SEMANTIC_SLICE
    assert leased.soft_pin_boundaries == 0

    released = runtime.lifecycle.release_transition_handoffs(
        (handoff.artifact_id,), focus_terms=("worker.py",)
    )

    assert released == (handoff.artifact_id,)
    retained = runtime.lifecycle.image.artifacts[0]
    assert retained.representation is Representation.VERIFIED_SUMMARY
    assert retained.soft_pin_boundaries == 0
    assert "context:milestone_handoff" not in retained.entity_refs
    assert "context:consumed_milestone_handoff" in retained.entity_refs
    assert runtime.metrics.snapshot().values["compression_count"] == 1


def test_used_recalled_page_stays_resident_for_step_then_releases(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, model_limit=4_000)
    block = _recovered_block()
    pending = runtime.lifecycle.prepare_recovered(block)
    assert pending is not None
    runtime.lifecycle.transport_accepted(pending)
    runtime.lifecycle.context_committed(pending)
    runtime.lifecycle.model_observed(
        pending,
        working_set_milestones=("milestone-current",),
    )

    page_ids = tuple(
        dict.fromkeys(
            handle.page_id for handle in pending.artifact.source_handles if handle.page_id
        )
    )
    pinned = runtime.lifecycle.pin_recalled_pages_to_step(
        page_ids,
        step_id="M001.S001",
    )
    assert pinned
    for _ in range(4):
        runtime.lifecycle.advance_boundary()
    replaced, released = runtime.lifecycle.demote_expired_recovered(active_step_id="M001.S001")
    assert replaced == ()
    assert released == ()
    recalled = next(item for item in runtime.lifecycle.image.artifacts if item.source_handles)
    assert recalled.representation is Representation.SEMANTIC_SLICE

    replaced, released = runtime.lifecycle.demote_expired_recovered(active_step_id="M001.S002")
    assert replaced
    assert set(released) == set(page_ids)
    recalled = next(item for item in runtime.lifecycle.image.artifacts if item.source_handles)
    assert recalled.representation is not Representation.SEMANTIC_SLICE


def test_step_locality_lease_does_not_block_pressure_fixed_point(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, model_limit=4_000)
    block = _recovered_block()
    pending = runtime.lifecycle.prepare_recovered(block)
    assert pending is not None
    runtime.lifecycle.transport_accepted(pending)
    runtime.lifecycle.context_committed(pending)
    runtime.lifecycle.model_observed(pending, working_set_milestones=())
    page_ids = tuple(
        dict.fromkeys(
            handle.page_id for handle in pending.artifact.source_handles if handle.page_id
        )
    )
    runtime.lifecycle.pin_recalled_pages_to_step(page_ids, step_id="M001.S001")
    for _ in range(3):
        runtime.lifecycle.advance_boundary()

    outcome = runtime.lifecycle.compact_to_fixed_point(working_set_milestones=())

    recalled = next(item for item in outcome.image.artifacts if item.source_handles)
    assert recalled.representation is not Representation.SEMANTIC_SLICE


def test_page_seal_identity_is_independent_from_pressure_eviction(tmp_path: Path) -> None:
    memory_ref = "memoryref_sealed-page"
    page = artifact_from_content(
        content=json.dumps(
            {
                "schema": "codex-longterm-v2/resident-page@1",
                "memory_ref": memory_ref,
                "detail_state": "RESIDENT_FULL_PAGE",
                "recall_required_before_use": False,
                "synopsis": {"summary": "implemented the bounded behavior"},
                "event_groups": [{"detail": "x" * 2_000}],
            }
        ),
        representation=Representation.FULL,
        milestone_ids=("milestone-current",),
        entity_refs=("file:src/runtime.py",),
        source_handles=(_handle("sealed-page"),),
        memory_ref=memory_ref,
        identity_seed="sealed-page",
    )
    roomy = _runtime(tmp_path / "roomy", model_limit=8_000)

    admitted = roomy.lifecycle.replace_artifacts(
        (),
        (page,),
        working_set_milestones=("milestone-current",),
    )

    assert admitted.pressure is PressureLevel.NORMAL
    resident = roomy.lifecycle.image.artifacts[0]
    assert resident.representation is Representation.FULL
    assert resident.memory_ref == memory_ref

    pressured = _runtime(tmp_path / "pressured", model_limit=450)
    demoted = pressured.lifecycle.replace_artifacts(
        (),
        (page,),
        working_set_milestones=(),
    )
    compacted = demoted.image.artifacts[0]

    assert compacted.representation is not Representation.FULL
    assert compacted.memory_ref == memory_ref
    assert compacted.source_handles == page.source_handles
    assert memory_ref in compacted.content or compacted.representation is Representation.NONRESIDENT
