from __future__ import annotations

from dataclasses import dataclass

from .database import StateDatabase


@dataclass(frozen=True, slots=True)
class RecoveryAuditReceipt:
    active_epoch_id: str
    current_milestone_id: str
    workspace_revision_id: str | None
    page_count: int
    projected_page_count: int
    pending_delivery_ids: tuple[str, ...]
    pending_side_effect_ids: tuple[str, ...]


class RecoveryAuditor:
    """Read-only cross-subsystem recovery fence for one persisted Run."""

    def __init__(self, database: StateDatabase) -> None:
        self.database = database

    def audit(self, run_id: str, branch_id: str) -> RecoveryAuditReceipt:
        connection = self.database.connection
        active = connection.execute(
            "SELECT epoch_id FROM context_epochs WHERE run_id=? AND state='ACTIVE'",
            (run_id,),
        ).fetchall()
        if len(active) != 1:
            raise RuntimeError(f"recovery requires exactly one ACTIVE Epoch, got {len(active)}")
        current = connection.execute(
            """SELECT target_id AS identity_id FROM v2_semantic_edges
               WHERE run_id=? AND branch_id=? AND edge_type='CURRENT_MILESTONE'
               AND valid_to_cursor IS NULL""",
            (run_id, branch_id),
        ).fetchall()
        if len(current) != 1:
            raise RuntimeError(
                f"recovery requires exactly one current Milestone, got {len(current)}"
            )
        images = connection.execute(
            "SELECT COUNT(*) FROM context_image_state_v2 WHERE run_id=? AND branch_id=?",
            (run_id, branch_id),
        ).fetchone()[0]
        if int(images) != 1:
            raise RuntimeError(f"recovery requires one ContextImage row, got {images}")
        page_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM v2_pages WHERE run_id=? AND branch_id=?",
                (run_id, branch_id),
            ).fetchone()[0]
        )
        projected = int(
            connection.execute(
                "SELECT COUNT(*) FROM v2_semantic_pages WHERE run_id=? AND branch_id=?",
                (run_id, branch_id),
            ).fetchone()[0]
        )
        if page_count != projected:
            raise RuntimeError("recovery found Page/Semantic projection divergence")
        revision_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='v2_current_workspace_revision'"
        ).fetchone()
        revision = None
        if revision_table is not None:
            row = connection.execute(
                """SELECT revision_id FROM v2_current_workspace_revision
                   WHERE run_id=? AND branch_id=?""",
                (run_id, branch_id),
            ).fetchone()
            revision = None if row is None else str(row["revision_id"])
        pending_deliveries = tuple(
            str(row[0])
            for row in connection.execute(
                """SELECT delivery_id FROM context_deliveries
                   WHERE state<>'MODEL_OBSERVED' ORDER BY prepared_at,delivery_id"""
            )
        )
        pending_effects = tuple(
            str(row[0])
            for row in connection.execute(
                """SELECT effect_id FROM v2_side_effects
                   WHERE run_id=? AND state NOT IN ('CONFIRMED','FAILED')
                   ORDER BY created_at,effect_id""",
                (run_id,),
            )
        )
        return RecoveryAuditReceipt(
            active_epoch_id=str(active[0]["epoch_id"]),
            current_milestone_id=str(current[0]["identity_id"]),
            workspace_revision_id=revision,
            page_count=page_count,
            projected_page_count=projected,
            pending_delivery_ids=pending_deliveries,
            pending_side_effect_ids=pending_effects,
        )
