from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ..contracts import digest
from ..durability import load_json_lines
from ..semantic_memory.store import durable_reasoning_frontier
from ..workspace_progress import semantic_progress_token


@dataclass(frozen=True, slots=True)
class SemanticProgressReceipt:
    """A bounded control-plane view of meaningful benchmark progress.

    File growth and Provider chatter are deliberately absent. Generic Evidence
    and unresolved-observation counts remain diagnostic only. A bounded,
    structured conclusion, rejected hypothesis or unresolved semantic question
    does advance the reasoning frontier; repeated reads, tool results and agent
    progress narration do not.
    """

    phase: str | None = None
    phase_state: str | None = None
    run_id: str | None = None
    task_status: str | None = None
    plan_version_id: str | None = None
    current_milestone_id: str | None = None
    current_milestone_status: str | None = None
    current_step_id: str | None = None
    current_step_status: str | None = None
    workspace_revision_id: str | None = None
    workspace_progress_token: str | None = None
    criterion_evidence_coverage_count: int = 0
    verifier_receipt_digest: str | None = None
    durable_reasoning_state_count: int = 0
    durable_reasoning_state_digest: str | None = None
    distinct_evidence_count: int = 0
    step_review_count: int = 0
    milestone_review_count: int = 0
    distinct_unresolved_count: int = 0
    distinct_failure_signature_count: int = 0
    active_side_effect_count: int = 0
    database_available: bool = False

    @property
    def fingerprint(self) -> tuple[object, ...]:
        return (
            self.phase,
            self.phase_state,
            self.run_id,
            self.task_status,
            self.plan_version_id,
            self.current_milestone_id,
            self.current_milestone_status,
            self.current_step_id,
            self.current_step_status,
            self.workspace_progress_token or self.workspace_revision_id,
            self.criterion_evidence_coverage_count,
            self.verifier_receipt_digest,
            self.durable_reasoning_state_digest,
            self.database_available,
        )

    @property
    def control_activity_count(self) -> int:
        """Diagnostic protocol activity that must never extend the progress lease."""

        return self.step_review_count + self.milestone_review_count

    @property
    def diagnostic_fingerprint(self) -> tuple[object, ...]:
        """Bounded exploration frontier which is not completion progress.

        A new distinct fact, failure signature, or unresolved hypothesis can
        justify continued diagnosis and reset the control-churn counter.  It
        deliberately does not extend the authoritative semantic-progress
        lease, so endlessly generating diagnostics cannot keep a task alive.
        """

        return (
            self.distinct_evidence_count,
            self.distinct_unresolved_count,
            self.distinct_failure_signature_count,
            self.database_available,
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema": "codex-longterm-v2/semantic-progress@6",
            **asdict(self),
            "control_activity_count": self.control_activity_count,
        }


def _last_phase(attempt_root: Path) -> tuple[str | None, str | None]:
    try:
        events = load_json_lines(attempt_root / "phase-events.jsonl")
    except (OSError, ValueError, json.JSONDecodeError):
        return None, None
    if not events:
        return None, None
    latest = events[-1]
    return str(latest.get("phase") or "") or None, str(latest.get("state") or "") or None


def _has_table(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        is not None
    )


def _has_columns(connection: sqlite3.Connection, table: str, columns: set[str]) -> bool:
    available = {
        str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    return columns.issubset(available)


def _scalar(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[object, ...] = (),
    *,
    default: Any = None,
) -> Any:
    row = connection.execute(query, parameters).fetchone()
    return default if row is None or row[0] is None else row[0]


def read_semantic_progress(attempt_root: Path) -> SemanticProgressReceipt:
    """Read the active Attempt without scanning its repository or log files."""

    phase, phase_state = _last_phase(attempt_root)
    database_path = attempt_root / "runtime" / "v2-state.sqlite3"
    if not database_path.is_file():
        return SemanticProgressReceipt(phase=phase, phase_state=phase_state)
    uri = f"{database_path.as_uri()}?mode=ro"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=0.25)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        if not _has_table(connection, "v2_registry_runs"):
            return SemanticProgressReceipt(phase=phase, phase_state=phase_state)
        run_id_value = _scalar(
            connection, "SELECT run_id FROM v2_registry_runs ORDER BY rowid LIMIT 1"
        )
        run_id = str(run_id_value) if run_id_value is not None else None
        task_status = None
        if run_id is not None and _has_table(connection, "v2_task_state_events"):
            value = _scalar(
                connection,
                "SELECT status FROM v2_task_state_events WHERE run_id=? "
                "ORDER BY created_cursor DESC LIMIT 1",
                (run_id,),
            )
            task_status = str(value) if value is not None else None

        plan_version_id = None
        milestone_id = None
        milestone_status = None
        current_identity = None
        if (
            run_id is not None
            and _has_table(connection, "v2_semantic_edges")
            and _has_table(connection, "v2_milestone_identities")
        ):
            row = connection.execute(
                "SELECT edge.target_id AS identity_id,edge.plan_version_id,mi.canonical_id,"
                "(SELECT mse.status FROM v2_milestone_state_events mse "
                " WHERE mse.identity_id=edge.target_id ORDER BY mse.created_cursor DESC LIMIT 1) "
                "AS status FROM v2_semantic_edges edge "
                "JOIN v2_milestone_identities mi ON mi.identity_id=edge.target_id "
                "WHERE edge.run_id=? AND edge.edge_type='CURRENT_MILESTONE' "
                "AND edge.valid_to_cursor IS NULL LIMIT 1",
                (run_id,),
            ).fetchone()
            if row is not None:
                current_identity = str(row["identity_id"])
                plan_version_id = str(row["plan_version_id"])
                milestone_id = str(row["canonical_id"])
                milestone_status = str(row["status"])

        current_step_id = None
        current_step_status = None
        if (
            current_identity is not None
            and _has_table(connection, "v2_plan_step_identities")
            and _has_table(connection, "v2_semantic_edges")
        ):
            row = connection.execute(
                "SELECT si.canonical_step_id,"
                "COALESCE((SELECT state.status FROM v2_plan_step_state_events state "
                " WHERE state.step_identity_id=si.step_identity_id "
                " ORDER BY state.created_cursor DESC LIMIT 1),'PENDING') AS status "
                "FROM v2_semantic_edges edge "
                "JOIN v2_plan_step_identities si ON si.step_identity_id=edge.target_id "
                "WHERE edge.run_id=? AND edge.edge_type='CURRENT_STEP' "
                "AND edge.valid_to_cursor IS NULL AND si.milestone_identity_id=? LIMIT 1",
                (run_id, current_identity),
            ).fetchone()
            if row is not None:
                current_step_id = str(row["canonical_step_id"])
                current_step_status = str(row["status"])

        criterion_evidence_coverage = 0
        if (
            run_id is not None
            and current_identity is not None
            and _has_table(connection, "v2_semantic_evidence")
            and _has_table(connection, "v2_completion_criteria")
        ):
            criterion_evidence_coverage = int(
                _scalar(
                    connection,
                    "SELECT COUNT(*) FROM ("
                    " SELECT edge.target_id,evidence.evidence_type "
                    " FROM v2_semantic_edges edge "
                    " JOIN v2_semantic_evidence evidence ON evidence.evidence_id=edge.source_id "
                    " JOIN v2_completion_criteria criterion "
                    " ON criterion.criterion_identity_id=edge.target_id "
                    " WHERE edge.run_id=? AND edge.edge_type='SATISFIES' "
                    " AND edge.valid_to_cursor IS NULL AND evidence.valid_to_cursor IS NULL "
                    " AND criterion.milestone_identity_id=? "
                    " GROUP BY edge.target_id,evidence.evidence_type"
                    ")",
                    (run_id, current_identity),
                    default=0,
                )
            )

        workspace_revision = None
        if run_id is not None and _has_table(connection, "v2_current_workspace_revision"):
            value = _scalar(
                connection,
                "SELECT revision_id FROM v2_current_workspace_revision WHERE run_id=? LIMIT 1",
                (run_id,),
            )
            workspace_revision = str(value) if value is not None else None

        verifier_receipt_digest = None
        if (
            run_id is not None
            and workspace_revision is not None
            and _has_table(connection, "v2_semantic_evidence")
            and _has_columns(
                connection,
                "v2_semantic_evidence",
                {"semantic_role", "revision_id", "content_digest"},
            )
        ):
            value = _scalar(
                connection,
                "SELECT content_digest FROM v2_semantic_evidence "
                "WHERE run_id=? AND revision_id=? AND valid_to_cursor IS NULL "
                "AND semantic_role='trusted_external_verification' "
                "AND evidence_type IN "
                "('VERIFIER_RESULT','TEST_RESULT','TEST_FAILURE','TOOL_RESULT') "
                "ORDER BY rowid DESC LIMIT 1",
                (run_id, workspace_revision),
            )
            verifier_receipt_digest = str(value) if value is not None else None

        durable_reasoning_state: tuple[tuple[str, str, str, str], ...] = ()
        if (
            run_id is not None
            and current_step_id is not None
            and _has_table(connection, "v2_semantic_evidence")
            and _has_columns(
                connection,
                "v2_semantic_evidence",
                {
                    "evidence_type",
                    "semantic_role",
                    "content_digest",
                    "revision_id",
                    "content_json",
                    "valid_to_cursor",
                },
            )
        ):
            durable_reasoning_state = durable_reasoning_frontier(
                connection,
                run_id=run_id,
                step_id=current_step_id,
            )

        distinct_evidence = (
            int(
                _scalar(
                    connection,
                    "SELECT COUNT(DISTINCT key_digest || ':' || content_digest) "
                    "FROM v2_semantic_evidence WHERE run_id=?",
                    (run_id,),
                    default=0,
                )
            )
            if run_id is not None and _has_table(connection, "v2_semantic_evidence")
            else 0
        )
        step_reviews = (
            int(
                _scalar(
                    connection,
                    "SELECT COUNT(*) FROM v2_plan_step_review_events WHERE run_id=?",
                    (run_id,),
                    default=0,
                )
            )
            if run_id is not None and _has_table(connection, "v2_plan_step_review_events")
            else 0
        )
        milestone_reviews = (
            int(
                _scalar(
                    connection,
                    "SELECT COUNT(*) FROM v2_milestone_review_events WHERE run_id=?",
                    (run_id,),
                    default=0,
                )
            )
            if run_id is not None and _has_table(connection, "v2_milestone_review_events")
            else 0
        )
        unresolved = (
            int(
                _scalar(
                    connection,
                    "SELECT COUNT(DISTINCT title || ':' || reason || ':' || candidates_json) "
                    "FROM v2_unresolved_milestone_observations WHERE run_id=?",
                    (run_id,),
                    default=0,
                )
            )
            if run_id is not None and _has_table(connection, "v2_unresolved_milestone_observations")
            else 0
        )
        distinct_failures = (
            int(
                _scalar(
                    connection,
                    "SELECT COUNT(DISTINCT content_digest) FROM v2_semantic_evidence "
                    "WHERE run_id=? AND evidence_type='TEST_FAILURE'",
                    (run_id,),
                    default=0,
                )
            )
            if run_id is not None and _has_table(connection, "v2_semantic_evidence")
            else 0
        )
        active_side_effects = (
            int(
                _scalar(
                    connection,
                    "SELECT COUNT(*) FROM v2_side_effects WHERE run_id=? "
                    "AND state IN ('INTENT_RECORDED','EXECUTION_STARTED','RESULT_OBSERVED')",
                    (run_id,),
                    default=0,
                )
            )
            if run_id is not None and _has_table(connection, "v2_side_effects")
            else 0
        )
        workspace_progress = semantic_progress_token(connection, run_id, workspace_revision)
    except sqlite3.Error:
        return SemanticProgressReceipt(phase=phase, phase_state=phase_state)
    finally:
        if connection is not None:
            connection.close()
    return SemanticProgressReceipt(
        phase=phase,
        phase_state=phase_state,
        run_id=run_id,
        task_status=task_status,
        plan_version_id=plan_version_id,
        current_milestone_id=milestone_id,
        current_milestone_status=milestone_status,
        current_step_id=current_step_id,
        current_step_status=current_step_status,
        workspace_revision_id=workspace_revision,
        workspace_progress_token=workspace_progress,
        criterion_evidence_coverage_count=criterion_evidence_coverage,
        verifier_receipt_digest=verifier_receipt_digest,
        durable_reasoning_state_count=len(durable_reasoning_state),
        durable_reasoning_state_digest=(
            digest(durable_reasoning_state) if durable_reasoning_state else None
        ),
        distinct_evidence_count=distinct_evidence,
        step_review_count=step_reviews,
        milestone_review_count=milestone_reviews,
        distinct_unresolved_count=unresolved,
        distinct_failure_signature_count=distinct_failures,
        active_side_effect_count=active_side_effects,
        database_available=True,
    )


def latest_stall_directive(attempt_root: Path) -> Mapping[str, object] | None:
    """Return the latest durable same-Attempt resume directive."""

    try:
        events = load_json_lines(attempt_root / "batch-control.jsonl")
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    for event in reversed(events):
        if event.get("state") in {"SUSPEND_STALLED", "SUSPEND_BLOCKED"}:
            return dict(event)
    return None
