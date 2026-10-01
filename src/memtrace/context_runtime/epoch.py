from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ..contracts import (
    ContextHandle,
    ContinuityCheckpoint,
    DeliveryState,
    EpochReason,
    stable_id,
    utc_now,
)
from ..database import StateDatabase
from ..observability import CounterName, MetricRecorder


@dataclass(frozen=True, slots=True)
class EpochDecision:
    admitted: bool
    reason: str


class EpochAdmission:
    def evaluate(
        self,
        *,
        reason: EpochReason,
        checkpoint: ContinuityCheckpoint,
        compression_fixed_point: bool,
        physical_context_cannot_continue: bool,
        candidate_context_tokens: int,
        safe_budget_tokens: int,
        candidate_context_digest: str,
    ) -> EpochDecision:
        if not isinstance(reason, EpochReason):
            return EpochDecision(False, "REASON_NOT_ALLOWED")
        required_identity = (
            checkpoint.checkpoint_id,
            checkpoint.task_goal_digest,
            checkpoint.plan_version_id,
            checkpoint.current_milestone_id,
            checkpoint.milestone_state_digest,
            checkpoint.workspace_revision_id,
        )
        if any(not value for value in required_identity):
            return EpochDecision(False, "INCOMPLETE_CONTINUITY_IDENTITY")
        if not checkpoint.safe_action_boundary:
            return EpochDecision(False, "NO_SAFE_ACTION_BOUNDARY")
        if checkpoint.pending_side_effects:
            return EpochDecision(False, "UNKNOWN_SIDE_EFFECTS")
        if checkpoint.delivery_state != DeliveryState.MODEL_OBSERVED:
            return EpochDecision(False, "LATEST_DELIVERY_NOT_MODEL_OBSERVED")
        if checkpoint.context_digest != checkpoint.resident_context_image.image_digest:
            return EpochDecision(False, "CHECKPOINT_IMAGE_DIGEST_MISMATCH")
        if candidate_context_tokens != checkpoint.resident_context_image.total_tokens:
            return EpochDecision(False, "CANDIDATE_TOKEN_MEASUREMENT_MISMATCH")
        if candidate_context_tokens < 0 or safe_budget_tokens <= 0:
            return EpochDecision(False, "INVALID_CONTEXT_BUDGET_EVIDENCE")
        if candidate_context_tokens >= safe_budget_tokens:
            return EpochDecision(False, "CANDIDATE_IMAGE_EXCEEDS_SAFE_BUDGET")
        if candidate_context_digest != checkpoint.context_digest:
            return EpochDecision(False, "CANDIDATE_CONTEXT_DIGEST_MISMATCH")
        if not self._handles_are_recoverable(checkpoint.nonresident_handles):
            return EpochDecision(False, "NONRESIDENT_HANDLE_INVALID")
        represented_handles = tuple(
            handle
            for artifact in checkpoint.resident_context_image.artifacts
            if artifact.representation.value in {"HANDLE", "NONRESIDENT"}
            for handle in artifact.source_handles
        )
        if not self._handles_are_recoverable(represented_handles):
            return EpochDecision(False, "CONTEXT_HANDLE_INVALID")
        projected = {
            handle
            for artifact in checkpoint.resident_context_image.artifacts
            if artifact.representation.value == "NONRESIDENT"
            for handle in artifact.source_handles
        }
        if not projected.issubset(set(checkpoint.nonresident_handles)):
            return EpochDecision(False, "NONRESIDENT_HANDLE_OMITTED")
        if reason == EpochReason.COMPRESSION_FIXED_POINT and not (
            compression_fixed_point and physical_context_cannot_continue
        ):
            return EpochDecision(False, "COMPRESSION_OR_PHYSICAL_PROOF_MISSING")
        if (
            reason
            in {
                EpochReason.PROVIDER_CONTEXT_LIMIT,
                EpochReason.THREAD_UNRECOVERABLE,
                EpochReason.SESSION_LOST,
                EpochReason.CRASH_RESUME_UNAVAILABLE,
            }
            and not physical_context_cannot_continue
        ):
            return EpochDecision(False, "PHYSICAL_FAILURE_PROOF_MISSING")
        return EpochDecision(True, reason.value)

    @staticmethod
    def _handles_are_recoverable(handles: tuple[ContextHandle, ...]) -> bool:
        return all(
            handle.page_id
            and handle.revision_id
            and handle.content_digest.startswith("sha256:")
            and len(handle.event_range) == 2
            and 0 <= handle.event_range[0] <= handle.event_range[1]
            and (
                (handle.blob_range is None and handle.blob_handle is None)
                or (
                    handle.blob_range is not None
                    and handle.blob_handle is not None
                    and len(handle.blob_range) == 2
                    and 0 <= handle.blob_range[0] <= handle.blob_range[1]
                )
            )
            for handle in handles
        )


class ThreadLifecycle:
    """The only V2 component allowed to allocate or fence execution Epochs."""

    def __init__(self, database: StateDatabase, metrics: MetricRecorder) -> None:
        self.database = database
        self.metrics = metrics
        self.database.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS context_epochs (
                epoch_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('ACTIVE','PENDING','FENCED')),
                reason TEXT NOT NULL,
                context_digest TEXT NOT NULL,
                predecessor_epoch_id TEXT,
                delivery_id TEXT,
                created_at TEXT NOT NULL,
                activated_at TEXT,
                fenced_at TEXT,
                UNIQUE(run_id,ordinal)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ux_context_epoch_active
            ON context_epochs(run_id) WHERE state='ACTIVE';
            """
        )

    def initialize(self, run_id: str, context_digest: str) -> str:
        if not run_id or not context_digest:
            raise ValueError("initial Epoch identity must be non-empty")
        epoch_id = stable_id("epoch_", {"run_id": run_id, "ordinal": 0})
        with self.database.transaction() as connection:
            existing = connection.execute(
                """SELECT epoch_id,context_digest FROM context_epochs
                   WHERE run_id=? AND ordinal=0""",
                (run_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["epoch_id"]) != epoch_id:
                    raise RuntimeError("run already has a different initial Epoch")
                # The ordinal-0 digest records the image at Epoch creation. It
                # must not be compared with the current ContextImage during a
                # process resume: normal actions, revision advances and Memory Loading
                # legitimately evolve that image without creating a new Epoch.
                active = connection.execute(
                    """SELECT epoch_id FROM context_epochs
                       WHERE run_id=? AND state='ACTIVE'""",
                    (run_id,),
                ).fetchone()
                if active is None:
                    raise RuntimeError("run has no active Epoch during recovery")
                return str(active["epoch_id"])
            connection.execute(
                """INSERT OR IGNORE INTO context_epochs
                   (epoch_id,run_id,ordinal,state,reason,context_digest,created_at,activated_at)
                   VALUES(?,?,0,'ACTIVE','INITIAL',?,?,?)""",
                (epoch_id, run_id, context_digest, utc_now(), utc_now()),
            )
        return epoch_id

    def create_pending(
        self,
        *,
        run_id: str,
        reason: EpochReason,
        context_digest: str,
        decision: EpochDecision,
    ) -> str:
        if not isinstance(reason, EpochReason):
            raise ValueError("Epoch reason is not allowed")
        if not decision.admitted or decision.reason != reason.value:
            raise RuntimeError(f"Epoch denied: {decision.reason}")
        if not context_digest.startswith("sha256:"):
            raise ValueError("Epoch Context digest is invalid")
        with self.database.transaction() as connection:
            active = connection.execute(
                "SELECT epoch_id,ordinal FROM context_epochs WHERE run_id=? AND state='ACTIVE'",
                (run_id,),
            ).fetchone()
            if active is None:
                raise RuntimeError("run has no active Epoch")
            if (
                connection.execute(
                    "SELECT 1 FROM context_epochs WHERE run_id=? AND state='PENDING'",
                    (run_id,),
                ).fetchone()
                is not None
            ):
                raise RuntimeError("run already has a pending Epoch")
            # An abandoned candidate keeps its ordinal, so the next candidate
            # follows the highest ordinal of the run, not the ACTIVE one.
            ordinal = (
                int(
                    connection.execute(
                        "SELECT MAX(ordinal) FROM context_epochs WHERE run_id=?", (run_id,)
                    ).fetchone()[0]
                )
                + 1
            )
            epoch_id = stable_id(
                "epoch_", {"run_id": run_id, "ordinal": ordinal, "reason": reason.value}
            )
            connection.execute(
                """INSERT INTO context_epochs
                   (epoch_id,run_id,ordinal,state,reason,context_digest,
                    predecessor_epoch_id,created_at)
                   VALUES(?,?,?,'PENDING',?,?,?,?)""",
                (
                    epoch_id,
                    run_id,
                    ordinal,
                    reason.value,
                    context_digest,
                    str(active["epoch_id"]),
                    utc_now(),
                ),
            )
        return epoch_id

    def bind_delivery(self, epoch_id: str, delivery_id: str) -> None:
        """Bind the new-thread injection journal before activation."""

        with self.database.transaction() as connection:
            pending = connection.execute(
                "SELECT context_digest,delivery_id FROM context_epochs WHERE epoch_id=? AND state='PENDING'",
                (epoch_id,),
            ).fetchone()
            if pending is None:
                raise RuntimeError("Epoch is not pending")
            if pending["delivery_id"] is not None:
                if str(pending["delivery_id"]) != delivery_id:
                    raise RuntimeError("Epoch already bound to another delivery")
                return
            delivery = connection.execute(
                """SELECT context_digest,recall_id FROM context_deliveries
                   WHERE delivery_id=?""",
                (delivery_id,),
            ).fetchone()
            if delivery is None:
                raise KeyError(delivery_id)
            if str(delivery["context_digest"]) != str(pending["context_digest"]):
                raise ValueError("Epoch delivery Context digest mismatch")
            if str(delivery["recall_id"]) != f"epoch:{epoch_id}":
                raise ValueError("Epoch must bind its own candidate-image delivery")
            connection.execute(
                "UPDATE context_epochs SET delivery_id=? WHERE epoch_id=?",
                (delivery_id, epoch_id),
            )

    def activate_after_model_observed(self, epoch_id: str) -> None:
        with self.database.transaction() as connection:
            self.activate_in_transaction(connection, epoch_id)
        self.metrics.increment(CounterName.EPOCH)

    def abandon_pending(self, epoch_id: str) -> None:
        """Fence a PENDING Epoch whose candidate image the model never observed.

        The predecessor stays ACTIVE.  A later physical failure allocates a
        fresh candidate from the current image; the abandoned row keeps its
        ordinal so the ordinal sequence remains a faithful history.
        """

        with self.database.transaction() as connection:
            pending = connection.execute(
                "SELECT delivery_id FROM context_epochs WHERE epoch_id=? AND state='PENDING'",
                (epoch_id,),
            ).fetchone()
            if pending is None:
                raise RuntimeError("Epoch is not pending")
            delivery_id = pending["delivery_id"]
            if delivery_id is not None:
                delivery = connection.execute(
                    "SELECT state FROM context_deliveries WHERE delivery_id=?",
                    (str(delivery_id),),
                ).fetchone()
                if (
                    delivery is not None
                    and DeliveryState(str(delivery["state"])) is DeliveryState.MODEL_OBSERVED
                ):
                    raise RuntimeError("an observed Epoch delivery cannot be abandoned")
            connection.execute(
                "UPDATE context_epochs SET state='FENCED',fenced_at=? WHERE epoch_id=?",
                (utc_now(), epoch_id),
            )

    def state(self, epoch_id: str) -> str:
        row = self.database.connection.execute(
            "SELECT state FROM context_epochs WHERE epoch_id=?", (epoch_id,)
        ).fetchone()
        if row is None:
            raise KeyError(epoch_id)
        return str(row[0])

    def active_epoch(self, run_id: str) -> str:
        row = self.database.connection.execute(
            "SELECT epoch_id FROM context_epochs WHERE run_id=? AND state='ACTIVE'",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return str(row[0])

    @staticmethod
    def activate_in_transaction(
        connection: sqlite3.Connection,
        epoch_id: str,
        *,
        expected_thread_id: str | None = None,
    ) -> None:
        pending = connection.execute(
            "SELECT * FROM context_epochs WHERE epoch_id=? AND state='PENDING'",
            (epoch_id,),
        ).fetchone()
        if pending is None:
            raise RuntimeError("Epoch is not pending")
        delivery_id = pending["delivery_id"]
        if delivery_id is None:
            raise RuntimeError("Epoch has no candidate-image delivery")
        delivery = connection.execute(
            """SELECT state,context_digest,thread_id FROM context_deliveries
               WHERE delivery_id=?""",
            (str(delivery_id),),
        ).fetchone()
        if delivery is None:
            raise RuntimeError("Epoch delivery is missing")
        if str(delivery["context_digest"]) != str(pending["context_digest"]):
            raise RuntimeError("Epoch delivery digest changed")
        if expected_thread_id is not None and str(delivery["thread_id"]) != expected_thread_id:
            raise RuntimeError("Epoch delivery targets a different Thread")
        if DeliveryState(str(delivery["state"])) != DeliveryState.MODEL_OBSERVED:
            raise RuntimeError("Epoch delivery has not reached MODEL_OBSERVED")
        predecessor = str(pending["predecessor_epoch_id"])
        connection.execute(
            "UPDATE context_epochs SET state='FENCED',fenced_at=? WHERE epoch_id=?",
            (utc_now(), predecessor),
        )
        connection.execute(
            "UPDATE context_epochs SET state='ACTIVE',activated_at=? WHERE epoch_id=?",
            (utc_now(), epoch_id),
        )
