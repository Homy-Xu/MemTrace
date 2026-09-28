from __future__ import annotations

import sqlite3
from typing import Protocol

from ..contracts import PlanStepSpec
from .contracts import PlanProjectionInput


class PlanProjection(Protocol):
    """Write-only projection hook executed inside the Registry transaction."""

    def project_plan(self, conn: sqlite3.Connection, update: PlanProjectionInput) -> None: ...

    def project_current_milestone(
        self,
        conn: sqlite3.Connection,
        *,
        repository_id: str,
        run_id: str,
        branch_id: str,
        revision_id: str,
        plan_version_id: str,
        milestone_identity_id: str,
        source_event_id: str,
        cursor: int,
    ) -> None: ...

    def project_plan_step(
        self,
        conn: sqlite3.Connection,
        *,
        repository_id: str,
        run_id: str,
        branch_id: str,
        revision_id: str,
        plan_version_id: str,
        milestone_identity_id: str,
        step_identity_id: str,
        step: PlanStepSpec,
        source_event_id: str,
        cursor: int,
    ) -> None: ...

    def project_current_step(
        self,
        conn: sqlite3.Connection,
        *,
        repository_id: str,
        run_id: str,
        branch_id: str,
        revision_id: str,
        plan_version_id: str,
        milestone_identity_id: str,
        step_identity_id: str | None,
        source_event_id: str,
        cursor: int,
    ) -> None: ...

    def project_step_correction(
        self,
        conn: sqlite3.Connection,
        *,
        repository_id: str,
        run_id: str,
        branch_id: str,
        revision_id: str,
        plan_version_id: str,
        failed_step_identity_id: str,
        corrective_step_identity_id: str,
        source_event_id: str,
        cursor: int,
    ) -> None: ...

    def project_step_contract_revision(
        self,
        conn: sqlite3.Connection,
        *,
        repository_id: str,
        run_id: str,
        branch_id: str,
        revision_id: str,
        plan_version_id: str,
        step_identity_id: str,
        source_event_id: str,
        cursor: int,
    ) -> None: ...
