from __future__ import annotations

from pathlib import Path

from memtrace.context_runtime import SideEffectLedger
from memtrace.contracts import (
    Event,
    EventGroup,
    EvidenceDraft,
    EvidenceKey,
    FactType,
    MilestoneSpec,
    PlanSpec,
    RecallIntent,
)
from memtrace.database import StateDatabase
from memtrace.harness import CodexEventMapper, HarnessEventType
from memtrace.page_store import PageStore, TailReason
from memtrace.planning import PlanRegistry
from memtrace.semantic_memory import SemanticStore


def test_codex_event_mapper_maps_tool_file_plan_token_error_and_deduplicates() -> None:
    mapper = CodexEventMapper(
        run_id="run",
        branch_id="main",
        thread_id="thread",
        revision_id="rev-1",
    )
    plan = {
        "method": "turn/plan/updated",
        "params": {
            "threadId": "thread",
            "turnId": "turn",
            "plan": [{"step": "M001: Implement", "status": "inProgress"}],
        },
    }
    assert [item.event_type for item in mapper.map_message(plan)] == [HarnessEventType.PLAN_UPDATED]
    assert mapper.map_message(plan) == ()

    started = {
        "method": "item/started",
        "params": {
            "threadId": "thread",
            "turnId": "turn",
            "startedAtMs": 10,
            "item": {
                "id": "tool-1",
                "type": "commandExecution",
                "command": "pytest",
                "status": "inProgress",
            },
        },
    }
    assert [item.event_type for item in mapper.map_message(started)] == [
        HarnessEventType.ITEM_STARTED,
        HarnessEventType.TOOL_INTENT,
    ]
    completed_file = {
        "method": "item/completed",
        "params": {
            "threadId": "thread",
            "turnId": "turn",
            "completedAtMs": 11,
            "item": {
                "id": "file-1",
                "type": "fileChange",
                "status": "completed",
                "changes": [{"path": "module.py", "diff": "+x", "kind": {"type": "update"}}],
            },
        },
    }
    file_events = mapper.map_message(completed_file)
    assert [item.event_type for item in file_events] == [
        HarnessEventType.ITEM_COMPLETED,
        HarnessEventType.TOOL_RESULT,
        HarnessEventType.FILE_CHANGED,
    ]
    assert file_events[-1].payload["paths"] == ["module.py"]
    token = mapper.map_message(
        {
            "method": "thread/tokenUsage/updated",
            "params": {
                "threadId": "thread",
                "turnId": "turn",
                "tokenUsage": {"total": {"totalTokens": 100}},
            },
        }
    )
    assert token[0].event_type is HarnessEventType.TOKEN_USAGE_UPDATED
    failure = mapper.map_message(
        {
            "method": "error",
            "params": {
                "threadId": "thread",
                "turnId": "turn",
                "error": {"codexErrorInfo": "contextWindowExceeded"},
                "willRetry": False,
            },
        }
    )
    assert failure[0].event_type is HarnessEventType.PHYSICAL_CONTEXT_FAILURE
    session_lost = mapper.map_message(
        {
            "method": "error",
            "params": {
                "threadId": "thread",
                "turnId": "turn-2",
                "error": {"codexErrorInfo": "session_not_found"},
                "willRetry": False,
            },
        }
    )
    assert session_lost[0].event_type is HarnessEventType.SESSION_LOST
    thread_lost = mapper.map_message(
        {
            "method": "error",
            "params": {
                "threadId": "thread",
                "turnId": "turn-3",
                "error": {"codexErrorInfo": "threadUnrecoverable"},
                "willRetry": False,
            },
        }
    )
    assert thread_lost[0].event_type is HarnessEventType.THREAD_UNRECOVERABLE


def test_side_effect_ledger_requires_real_lifecycle_and_records_failure(
    tmp_path: Path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite")
    ledger = SideEffectLedger(database)
    effect = ledger.record_intent("run", "tool", {"command": "false"}, source_event_id="started")
    assert ledger.state(effect) == "INTENT_RECORDED"
    assert ledger.pending("run") == (effect,)
    ledger.execution_started(effect, source_event_id="started")
    ledger.result_observed(
        effect,
        success=False,
        detail={"exitCode": 1},
        source_event_id="completed",
    )
    ledger.resolve(effect, success=False, source_event_id="completed")
    assert ledger.state(effect) == "FAILED"
    states = [
        row[0]
        for row in database.connection.execute(
            "SELECT state FROM v2_side_effect_state_events WHERE effect_id=? ORDER BY rowid",
            (effect,),
        )
    ]
    assert states == [
        "INTENT_RECORDED",
        "EXECUTION_STARTED",
        "RESULT_OBSERVED",
        "FAILED",
    ]
    database.close()


def test_workspace_revision_advance_invalidates_old_current_evidence(tmp_path: Path) -> None:
    database = StateDatabase(tmp_path / "revision.sqlite")
    semantic = SemanticStore(database)
    registry = PlanRegistry(database, semantic)
    application = registry.initialize_task(
        repository_id="repo",
        run_id="run",
        branch_id="main",
        revision_id="rev-old",
        user_request="change module",
        plan=PlanSpec(
            "change module",
            (MilestoneSpec("M001", "Change", "edit", ("done",), ("test",)),),
        ),
        source_event_id="plan-event",
    )
    key = EvidenceKey(
        FactType.CODE_OBSERVATION,
        "file:module.py",
        "CURRENT_INTERFACE",
        "rev-old",
        "main",
    )
    store = PageStore(
        tmp_path / "pages",
        database,
        "run",
        "main",
        projector=semantic.project_page,
    )
    store.append_group(
        EventGroup(
            group_id="old-group",
            group_type="TOOL_RESULT",
            run_id="run",
            branch_id="main",
            revision_id="rev-old",
            milestone_id=application.current_milestone_id,
            events=(
                Event(
                    event_id="old-event",
                    event_type="TOOL_RESULT",
                    payload={"observation": "old"},
                    facts=(EvidenceDraft(key, {"value": "old"}),),
                    entity_refs=("file:module.py",),
                    milestone_id=application.current_milestone_id,
                    revision_id="rev-old",
                ),
            ),
        )
    )
    store.checkpoint(TailReason.USER_CHECKPOINT)
    old_intent = RecallIntent(
        recall_id="old",
        repository_id="repo",
        run_id="run",
        branch_id="main",
        revision_id="rev-old",
        required_evidence=(key,),
        current_milestone_id=application.current_milestone_id,
        question="old interface",
    )
    assert semantic.locate_exact(old_intent).complete
    assert (
        semantic.advance_workspace_revision(
            run_id="run",
            branch_id="main",
            new_revision_id="rev-new",
            source_event_id="file-change-event",
        )
        == 1
    )
    after = semantic.locate_exact(old_intent)
    assert after.hits == ()
    assert after.missing_key_digests == (key.key_digest,)
    database.close()


def test_revision_invalidation_is_selective_and_matches_symbols_to_changed_file(
    tmp_path: Path,
) -> None:
    database = StateDatabase(tmp_path / "selective-revision.sqlite")
    semantic = SemanticStore(database)
    registry = PlanRegistry(database, semantic)
    application = registry.initialize_task(
        repository_id="repo",
        run_id="run",
        branch_id="main",
        revision_id="rev-old",
        user_request="change one module",
        plan=PlanSpec(
            "change one module",
            (MilestoneSpec("M001", "Change", "edit", ("done",), ("test",)),),
        ),
        source_event_id="plan-event",
    )
    facts = (
        EvidenceDraft(
            EvidenceKey(
                FactType.CODE_OBSERVATION,
                "symbol:module.py:operation",
                "CURRENT_INTERFACE",
                "rev-old",
                "main",
            ),
            {"summary": "changed symbol observation"},
        ),
        EvidenceDraft(
            EvidenceKey(
                FactType.IMPLEMENTATION_DECISION,
                "symbol:module.py:operation",
                "IMPLEMENTATION_CHOICE",
                "rev-old",
                "main",
            ),
            {"summary": "decision bound to changed symbol"},
        ),
        EvidenceDraft(
            EvidenceKey(
                FactType.CODE_OBSERVATION,
                "file:unrelated.py",
                "CURRENT_INTERFACE",
                "rev-old",
                "main",
            ),
            {"summary": "unrelated observation"},
        ),
        EvidenceDraft(
            EvidenceKey(
                FactType.CODE_CHANGE,
                "file:module.py",
                "workspace_change",
                "rev-old",
                "main",
            ),
            {"summary": "change to the affected module"},
        ),
        EvidenceDraft(
            EvidenceKey(
                FactType.CODE_CHANGE,
                "file:unrelated.py",
                "workspace_change",
                "rev-old",
                "main",
            ),
            {"summary": "still-current change to an unrelated module"},
        ),
        EvidenceDraft(
            EvidenceKey(
                FactType.USER_CONSTRAINT,
                "task:run",
                "ORIGINAL_REQUEST",
                "rev-old",
                "main",
            ),
            {"summary": "preserve the user constraint"},
            must_preserve=True,
        ),
        EvidenceDraft(
            EvidenceKey(
                FactType.TEST_RESULT,
                "test:tests/test_module.py::test_operation",
                "EXECUTION_RESULT",
                "rev-old",
                "main",
            ),
            {"success": True},
        ),
    )
    store = PageStore(
        tmp_path / "selective-pages",
        database,
        "run",
        "main",
        projector=semantic.project_page,
    )
    store.append_group(
        EventGroup(
            group_id="facts",
            group_type="OBSERVATION",
            run_id="run",
            branch_id="main",
            revision_id="rev-old",
            milestone_id=application.current_milestone_id,
            events=(
                Event(
                    event_id="facts-event",
                    event_type="OBSERVATION",
                    payload={},
                    facts=facts,
                    entity_refs=("file:module.py", "file:unrelated.py"),
                    milestone_id=application.current_milestone_id,
                    revision_id="rev-old",
                ),
            ),
        )
    )
    store.checkpoint(TailReason.USER_CHECKPOINT)

    assert (
        semantic.advance_workspace_revision(
            run_id="run",
            branch_id="main",
            new_revision_id="rev-new",
            source_event_id="change-event",
            changed_entities=("file:module.py",),
        )
        == 3
    )
    rows = {
        (str(row["evidence_type"]), str(row["canonical_entity_id"])): row["valid_to_revision"]
        for row in database.connection.execute(
            "SELECT evidence_type,canonical_entity_id,valid_to_revision FROM v2_semantic_evidence"
        )
    }
    assert rows[("CODE_OBSERVATION", "symbol:module.py:operation")] == "rev-new"
    assert rows[("IMPLEMENTATION_DECISION", "symbol:module.py:operation")] is None
    assert rows[("TEST_RESULT", "test:tests/test_module.py::test_operation")] == "rev-new"
    assert rows[("CODE_CHANGE", "file:module.py")] == "rev-new"
    assert rows[("CODE_CHANGE", "file:unrelated.py")] is None
    assert rows[("CODE_OBSERVATION", "file:unrelated.py")] is None
    assert rows[("USER_CONSTRAINT", "task:run")] is None
    database.close()
