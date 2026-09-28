from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..contracts import (
    ContextArtifact,
    ContextHandle,
    ContextImage,
    ContinuityCheckpoint,
    DeliveryState,
    PressureLevel,
    RecoveredContextBlock,
    Representation,
    primitive,
    stable_id,
    utc_now,
)
from ..database import StateDatabase
from ..observability import CounterName, MetricRecorder
from .admission import AdmissionOutcome, ContextAdmission
from .delivery import DeliveryJournal
from .image import ContextImageBuilder, artifact_from_content

if TYPE_CHECKING:
    from .epoch import ThreadLifecycle


class SideEffectLedger:
    def __init__(self, database: StateDatabase) -> None:
        self.database = database
        self.database.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS v2_side_effects (
                effect_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                action_id TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN (
                    'INTENT_RECORDED','EXECUTION_STARTED','RESULT_OBSERVED','CONFIRMED','FAILED'
                )),
                detail_json TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(run_id, action_id)
            );
            CREATE INDEX IF NOT EXISTS v2_side_effect_pending
            ON v2_side_effects(run_id,state);
            CREATE TABLE IF NOT EXISTS v2_side_effect_state_events (
                transition_id TEXT PRIMARY KEY,
                effect_id TEXT NOT NULL REFERENCES v2_side_effects(effect_id),
                previous_state TEXT,
                state TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                detail_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(effect_id, state, source_event_id)
            );
            CREATE TRIGGER IF NOT EXISTS v2_side_effect_events_no_update
            BEFORE UPDATE ON v2_side_effect_state_events
            BEGIN SELECT RAISE(ABORT, 'SideEffect state events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS v2_side_effect_events_no_delete
            BEFORE DELETE ON v2_side_effect_state_events
            BEGIN SELECT RAISE(ABORT, 'SideEffect state events are append-only'); END;
            CREATE TABLE IF NOT EXISTS v2_side_effect_reconciliations (
                effect_id TEXT PRIMARY KEY REFERENCES v2_side_effects(effect_id),
                run_id TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('ORPHANED','REOBSERVED')),
                source_event_id TEXT NOT NULL,
                detail_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS v2_side_effect_reconciliation_run
            ON v2_side_effect_reconciliations(run_id,state);
            """
        )

    def record_intent(
        self,
        run_id: str,
        action_id: str,
        detail: dict[str, object],
        *,
        source_event_id: str | None = None,
    ) -> str:
        effect_id = stable_id(
            "effect_", {"run_id": run_id, "action_id": action_id, "detail": detail}
        )
        source = source_event_id or action_id
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO v2_side_effects
                   VALUES(?,?,?,'INTENT_RECORDED',?,NULL,?,?)""",
                (
                    effect_id,
                    run_id,
                    action_id,
                    json.dumps(detail, ensure_ascii=False, sort_keys=True),
                    utc_now(),
                    utc_now(),
                ),
            )
            row = connection.execute(
                "SELECT effect_id, detail_json FROM v2_side_effects WHERE run_id=? AND action_id=?",
                (run_id, action_id),
            ).fetchone()
            if row is None or str(row["effect_id"]) != effect_id:
                raise ValueError("Side Effect action identity collision")
            self._append_transition(
                connection,
                effect_id=effect_id,
                previous_state=None,
                state="INTENT_RECORDED",
                source_event_id=source,
                detail=detail,
            )
        return effect_id

    @staticmethod
    def _append_transition(
        connection: sqlite3.Connection,
        *,
        effect_id: str,
        previous_state: str | None,
        state: str,
        source_event_id: str,
        detail: Mapping[str, object],
    ) -> None:
        transition_id = stable_id(
            "effectstate_",
            {"effect": effect_id, "state": state, "source": source_event_id},
        )
        connection.execute(
            "INSERT OR IGNORE INTO v2_side_effect_state_events VALUES(?,?,?,?,?,?,?)",
            (
                transition_id,
                effect_id,
                previous_state,
                state,
                source_event_id,
                json.dumps(detail, ensure_ascii=False, sort_keys=True),
                utc_now(),
            ),
        )

    def _transition(
        self,
        effect_id: str,
        *,
        expected: str,
        state: str,
        source_event_id: str,
        detail: Mapping[str, object] | None = None,
        result_json: str | None = None,
    ) -> None:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT state FROM v2_side_effects WHERE effect_id=?", (effect_id,)
            ).fetchone()
            if row is None:
                raise KeyError(effect_id)
            current = str(row["state"])
            if current == state:
                return
            if current != expected:
                raise ValueError(f"illegal Side Effect transition: {current} -> {state}")
            connection.execute(
                "UPDATE v2_side_effects SET state=?,result_json=COALESCE(?,result_json),"
                "updated_at=? WHERE effect_id=?",
                (state, result_json, utc_now(), effect_id),
            )
            self._append_transition(
                connection,
                effect_id=effect_id,
                previous_state=current,
                state=state,
                source_event_id=source_event_id,
                detail=detail or {},
            )
            connection.execute(
                "UPDATE v2_side_effect_reconciliations SET state='REOBSERVED',updated_at=? "
                "WHERE effect_id=? AND state='ORPHANED'",
                (utc_now(), effect_id),
            )

    def reconcile_orphaned(
        self,
        run_id: str,
        *,
        source_event_id: str,
        reason: str = "PROCESS_RESTART_WITHOUT_RESULT",
    ) -> tuple[str, ...]:
        '''Reconcile actions left open by an owned process interruption.

        A repository-stream writer holds the task lock before this method is
        called, so another invocation cannot still be mutating the same WAL.
        We do not claim that the command succeeded or failed. Instead we
        record a durable ORPHANED marker which lets Epoch recovery proceed
        while retaining the original action and requiring any later result to
        be observed through the normal idempotent event path.
        '''

        if not run_id.strip() or not source_event_id.strip():
            raise ValueError("orphan reconciliation requires run and source identities")
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT effect_id,action_id,state,detail_json FROM v2_side_effects "
                "WHERE run_id=? AND state IN ('INTENT_RECORDED','EXECUTION_STARTED') "
                "AND NOT EXISTS (SELECT 1 FROM v2_side_effect_reconciliations r "
                "WHERE r.effect_id=v2_side_effects.effect_id AND r.state='ORPHANED')",
                (run_id,),
            ).fetchall()
            recovered: list[str] = []
            for row in rows:
                effect_id = str(row["effect_id"])
                detail = {
                    "reason": reason,
                    "action_id": str(row["action_id"]),
                    "previous_state": str(row["state"]),
                    "result_known": False,
                    "safe_to_resume": True,
                }
                connection.execute(
                    "INSERT OR IGNORE INTO v2_side_effect_reconciliations "
                    "VALUES(?,?,?,?,?,?,?)",
                    (
                        effect_id,
                        run_id,
                        "ORPHANED",
                        source_event_id,
                        json.dumps(detail, ensure_ascii=False, sort_keys=True),
                        utc_now(),
                        utc_now(),
                    ),
                )
                recovered.append(effect_id)
            return tuple(recovered)

    def execution_started(self, effect_id: str, *, source_event_id: str) -> None:
        self._transition(
            effect_id,
            expected="INTENT_RECORDED",
            state="EXECUTION_STARTED",
            source_event_id=source_event_id,
        )

    def result_observed(
        self,
        effect_id: str,
        *,
        success: bool,
        detail: Mapping[str, object],
        source_event_id: str,
    ) -> None:
        result = {"success": success, **dict(detail)}
        self._transition(
            effect_id,
            expected="EXECUTION_STARTED",
            state="RESULT_OBSERVED",
            source_event_id=source_event_id,
            detail=result,
            result_json=json.dumps(result, ensure_ascii=False, sort_keys=True),
        )

    def resolve(
        self,
        effect_id: str,
        *,
        success: bool,
        source_event_id: str | None = None,
    ) -> None:
        self._transition(
            effect_id,
            expected="RESULT_OBSERVED",
            state="CONFIRMED" if success else "FAILED",
            source_event_id=source_event_id or effect_id,
        )

    def effect_for_action(self, run_id: str, action_id: str) -> str | None:
        row = self.database.connection.execute(
            "SELECT effect_id FROM v2_side_effects WHERE run_id=? AND action_id=?",
            (run_id, action_id),
        ).fetchone()
        return str(row["effect_id"]) if row is not None else None

    def state(self, effect_id: str) -> str:
        row = self.database.connection.execute(
            "SELECT state FROM v2_side_effects WHERE effect_id=?", (effect_id,)
        ).fetchone()
        if row is None:
            raise KeyError(effect_id)
        return str(row["state"])

    def pending(self, run_id: str) -> tuple[str, ...]:
        return tuple(
            str(row[0])
            for row in self.database.connection.execute(
                "SELECT effect_id FROM v2_side_effects WHERE run_id=? "
                "AND state NOT IN ('CONFIRMED','FAILED') "
                "AND NOT EXISTS (SELECT 1 FROM v2_side_effect_reconciliations r "
                "WHERE r.effect_id=v2_side_effects.effect_id AND r.state='ORPHANED')",
                (run_id,),
            )
        )


@dataclass(frozen=True, slots=True)
class PendingDelivery:
    delivery_id: str
    block: RecoveredContextBlock
    artifact: ContextArtifact


class ContextLifecycle:
    def __init__(
        self,
        *,
        run_id: str,
        branch_id: str,
        admission: ContextAdmission,
        delivery: DeliveryJournal,
        builder: ContextImageBuilder,
        metrics: MetricRecorder,
        initial_image: ContextImage,
    ) -> None:
        if not run_id or not branch_id:
            raise ValueError("ContextLifecycle requires a Run and Branch identity")
        self.run_id = run_id
        self.branch_id = branch_id
        self.admission = admission
        self.delivery = delivery
        self.builder = builder
        self.metrics = metrics
        self._runtime_id = stable_id(
            "context_runtime_",
            {
                "run_id": run_id,
                "branch_id": branch_id,
                "bootstrap_thread_id": initial_image.thread_id,
            },
        )
        self._initialize_image_state(initial_image)
        self._pending: dict[str, PendingDelivery] = {}

    @property
    def image(self) -> ContextImage:
        """The persisted authoritative ContextImage (read-only to callers)."""

        return self._image

    def _initialize_image_state(self, initial_image: ContextImage) -> None:
        self._validate_image(initial_image)
        self.delivery.database.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS context_image_state_v2 (
                runtime_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                bootstrap_thread_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                image_id TEXT NOT NULL,
                image_digest TEXT NOT NULL,
                image_json TEXT NOT NULL,
                generation INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(run_id, branch_id, bootstrap_thread_id)
            );
            CREATE INDEX IF NOT EXISTS context_image_state_v2_scope
            ON context_image_state_v2(run_id,branch_id,updated_at);
            CREATE TABLE IF NOT EXISTS context_image_thread_alias_v2 (
                run_id TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                runtime_id TEXT NOT NULL,
                PRIMARY KEY(run_id,branch_id,thread_id),
                FOREIGN KEY(runtime_id) REFERENCES context_image_state_v2(runtime_id)
            );
            """
        )
        with self.delivery.database.transaction() as connection:
            alias = connection.execute(
                """SELECT runtime_id FROM context_image_thread_alias_v2
                   WHERE run_id=? AND branch_id=? AND thread_id=?""",
                (self.run_id, self.branch_id, initial_image.thread_id),
            ).fetchone()
            if alias is not None:
                self._runtime_id = str(alias["runtime_id"])
            row = connection.execute(
                "SELECT * FROM context_image_state_v2 WHERE runtime_id=?",
                (self._runtime_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """INSERT INTO context_image_state_v2
                       (runtime_id,run_id,branch_id,bootstrap_thread_id,thread_id,
                        image_id,image_digest,image_json,generation,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,0,?)""",
                    (
                        self._runtime_id,
                        self.run_id,
                        self.branch_id,
                        initial_image.thread_id,
                        initial_image.thread_id,
                        initial_image.image_id,
                        initial_image.image_digest,
                        self._image_json(initial_image),
                        utc_now(),
                    ),
                )
                self._image = initial_image
                self._generation = 0
            else:
                if str(row["run_id"]) != self.run_id or str(row["branch_id"]) != self.branch_id:
                    raise RuntimeError("persisted ContextImage Run/Branch identity mismatch")
                restored = self._decode_image(str(row["image_json"]))
                if (
                    restored.image_id != str(row["image_id"])
                    or restored.image_digest != str(row["image_digest"])
                    or restored.thread_id != str(row["thread_id"])
                ):
                    raise RuntimeError("persisted ContextImage metadata mismatch")
                self._validate_image(restored)
                self._image = restored
                self._generation = int(row["generation"])
            connection.execute(
                """INSERT OR IGNORE INTO context_image_thread_alias_v2
                   (run_id,branch_id,thread_id,runtime_id) VALUES(?,?,?,?)""",
                (self.run_id, self.branch_id, initial_image.thread_id, self._runtime_id),
            )

    def _validate_image(self, image: ContextImage) -> None:
        rebuilt = self.builder.build(
            thread_id=image.thread_id,
            artifacts=image.artifacts,
            current_milestone_id=image.current_milestone_id,
            revision_id=image.revision_id,
        )
        if rebuilt != image:
            raise ValueError("ContextImage is not canonical")

    @staticmethod
    def _image_json(image: ContextImage) -> str:
        return json.dumps(
            primitive(image), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def _decode_image(self, payload: str) -> ContextImage:
        value = json.loads(payload)
        artifacts: list[ContextArtifact] = []
        for item in value["artifacts"]:
            handles = tuple(
                ContextHandle(
                    page_id=str(handle["page_id"]),
                    event_range=tuple(int(part) for part in handle["event_range"]),
                    blob_handle=(
                        None if handle["blob_handle"] is None else str(handle["blob_handle"])
                    ),
                    blob_range=(
                        None
                        if handle["blob_range"] is None
                        else tuple(int(part) for part in handle["blob_range"])
                    ),
                    content_digest=str(handle["content_digest"]),
                    revision_id=str(handle["revision_id"]),
                )
                for handle in item["source_handles"]
            )
            artifacts.append(
                ContextArtifact(
                    artifact_id=str(item["artifact_id"]),
                    representation=Representation(str(item["representation"])),
                    content=str(item["content"]),
                    token_count=int(item["token_count"]),
                    content_digest=str(item["content_digest"]),
                    milestone_ids=tuple(str(part) for part in item["milestone_ids"]),
                    entity_refs=tuple(str(part) for part in item["entity_refs"]),
                    source_handles=handles,
                    must_preserve=bool(item["must_preserve"]),
                    current_milestone=bool(item["current_milestone"]),
                    verified=bool(item["verified"]),
                    soft_pin_boundaries=int(item["soft_pin_boundaries"]),
                    derived_from=tuple(str(part) for part in item["derived_from"]),
                    memory_ref=(
                        str(item["memory_ref"]) if item.get("memory_ref") is not None else None
                    ),
                )
            )
        return ContextImage(
            image_id=str(value["image_id"]),
            thread_id=str(value["thread_id"]),
            artifacts=tuple(artifacts),
            total_tokens=int(value["total_tokens"]),
            image_digest=str(value["image_digest"]),
            current_milestone_id=str(value["current_milestone_id"]),
            revision_id=str(value["revision_id"]),
        )

    def _write_image(self, connection: sqlite3.Connection, candidate: ContextImage) -> None:
        self._validate_image(candidate)
        cursor = connection.execute(
            """UPDATE context_image_state_v2
               SET thread_id=?,image_id=?,image_digest=?,image_json=?,
                   generation=generation+1,updated_at=?
               WHERE runtime_id=? AND run_id=? AND branch_id=?
                 AND generation=? AND image_digest=?""",
            (
                candidate.thread_id,
                candidate.image_id,
                candidate.image_digest,
                self._image_json(candidate),
                utc_now(),
                self._runtime_id,
                self.run_id,
                self.branch_id,
                self._generation,
                self._image.image_digest,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("concurrent ContextImage update detected")
        connection.execute(
            """INSERT OR IGNORE INTO context_image_thread_alias_v2
               (run_id,branch_id,thread_id,runtime_id) VALUES(?,?,?,?)""",
            (self.run_id, self.branch_id, candidate.thread_id, self._runtime_id),
        )
        alias = connection.execute(
            """SELECT runtime_id FROM context_image_thread_alias_v2
               WHERE run_id=? AND branch_id=? AND thread_id=?""",
            (self.run_id, self.branch_id, candidate.thread_id),
        ).fetchone()
        if alias is None or str(alias["runtime_id"]) != self._runtime_id:
            raise RuntimeError("Thread is already owned by another Context runtime")

    def _commit_image(self, candidate: ContextImage) -> None:
        with self.delivery.database.transaction() as connection:
            self._write_image(connection, candidate)
        self._image = candidate
        self._generation += 1

    @staticmethod
    def _virtual_section_address(block: RecoveredContextBlock) -> str | None:
        """Return the stable virtual address of one exact recovered section set."""

        if block.source_memory_ref is None or not block.slices:
            return None
        return stable_id(
            "virtualsection_",
            {
                "memory_ref": block.source_memory_ref,
                "slices": tuple(item.slice_id for item in block.slices),
            },
        )

    def pending_recovered(self, block: RecoveredContextBlock) -> PendingDelivery | None:
        """Find an in-flight delivery for the same immutable semantic address."""

        address = self._virtual_section_address(block)
        for pending in self._pending.values():
            if pending.block.content_digest == block.content_digest:
                return pending
            if address is not None and self._virtual_section_address(pending.block) == address:
                return pending
        return None

    def resident_recovered(self, block: RecoveredContextBlock) -> ContextArtifact | None:
        """Find the exact Page section while it is still resident as full slice content."""

        address = self._virtual_section_address(block)
        for artifact in self.image.artifacts:
            if artifact.representation is not Representation.SEMANTIC_SLICE:
                continue
            if block.content_digest in artifact.derived_from:
                return artifact
            if address is not None and address in artifact.derived_from:
                return artifact
        return None

    def prepare_recovered(
        self,
        block: RecoveredContextBlock,
        *,
        working_set_milestones: tuple[str, ...] = (),
        focus_terms: tuple[str, ...] = (),
        max_admission_tokens: int | None = None,
    ) -> PendingDelivery | None:
        if not block.slices or block.coverage.state.value == "EMPTY":
            return None
        if max_admission_tokens is not None and block.token_count > max_admission_tokens:
            return None
        pending = self.pending_recovered(block)
        if pending is not None:
            return pending
        if self.resident_recovered(block) is not None or self.delivery.is_resident(
            block.content_digest,
            thread_id=self.image.thread_id,
        ):
            return None
        handles: list[ContextHandle] = []
        requested_entities = tuple(
            dict.fromkeys(
                entity
                for entity in block.requested_entity_refs
                if entity and not entity.startswith("context:")
            )
        )
        recovered_entities: list[str] = []
        for item in block.slices:
            handles.append(
                ContextHandle(
                    page_id=item.page_id,
                    event_range=item.event_range,
                    blob_handle=None,
                    blob_range=None,
                    content_digest=item.page_digest,
                    revision_id=item.revision_id,
                )
            )
            events = item.content.get("events", ())
            if not isinstance(events, list):
                continue
            for event in events:
                if not isinstance(event, Mapping):
                    continue
                facts = event.get("facts", ())
                if isinstance(facts, list):
                    for fact in facts:
                        if not isinstance(fact, Mapping):
                            continue
                        key = fact.get("key")
                        if not isinstance(key, Mapping):
                            continue
                        entity = str(key.get("canonical_entity_id", "")).strip()
                        if (
                            entity
                            and entity not in recovered_entities
                            and (not requested_entities or entity in requested_entities)
                        ):
                            recovered_entities.append(entity)
                payload = event.get("payload")
                if not isinstance(payload, Mapping):
                    continue
                external = payload.get("external_payload")
                if not isinstance(external, Mapping):
                    continue
                byte_range = external.get("byte_range")
                event_range = event.get("event_position")
                if (
                    not isinstance(byte_range, (list, tuple))
                    or len(byte_range) != 2
                    or not isinstance(event_range, (list, tuple))
                    or len(event_range) != 2
                ):
                    raise ValueError("Recovered PageSlice contains an invalid Blob locator")
                handles.append(
                    ContextHandle(
                        page_id=item.page_id,
                        event_range=(int(event_range[0]), int(event_range[1])),
                        blob_handle=str(external["blob_handle"]),
                        blob_range=(int(byte_range[0]), int(byte_range[1])),
                        content_digest=str(external.get("content_digest", external["blob_handle"])),
                        revision_id=item.revision_id,
                    )
                )
        unique_handles = tuple(dict.fromkeys(handles))
        section_address = self._virtual_section_address(block)
        artifact = artifact_from_content(
            content=block.rendered_content,
            representation=Representation.SEMANTIC_SLICE,
            milestone_ids=(block.current_milestone_id,),
            entity_refs=tuple(
                dict.fromkeys(
                    (
                        "context:recalled_slice",
                        *requested_entities[:8],
                        *recovered_entities[:8],
                    )
                )
            ),
            source_handles=unique_handles,
            verified=block.coverage.state.value == "COMPLETE",
            soft_pin_boundaries=2,
            derived_from=tuple(
                dict.fromkeys(
                    (
                        block.content_digest,
                        *((section_address,) if section_address is not None else ()),
                    )
                )
            ),
            memory_ref=block.source_memory_ref,
            identity_seed=block.block_id,
        )
        # Admission is evaluated before any Provider transport. The candidate is
        # committed only after a real MODEL_OBSERVED signal, but an over-budget
        # block can never reach thread/inject_items.
        preview = self.admission.admit(
            current=self.image,
            incoming=(artifact,),
            working_set_milestones=working_set_milestones,
            focus_terms=focus_terms,
        )
        if artifact.artifact_id not in preview.admitted_artifact_ids:
            return None
        if max_admission_tokens is not None and artifact.token_count > max_admission_tokens:
            return None
        if preview.final_pressure is PressureLevel.HARD and preview.fixed_point:
            return None
        delivery_id = self.delivery.prepare(block, thread_id=self.image.thread_id)
        pending = PendingDelivery(delivery_id, block, artifact)
        self._pending[delivery_id] = pending
        return pending

    def _require_pending(self, pending: PendingDelivery) -> None:
        owned = self._pending.get(pending.delivery_id)
        if owned is None or owned != pending:
            raise ValueError("delivery is not pending in this ContextLifecycle")

    def pending_delivery(self, delivery_id: str) -> PendingDelivery | None:
        """Return a delivery prepared by this runtime without advancing it."""

        return self._pending.get(delivery_id)

    def pending_recall_deliveries(self) -> tuple[PendingDelivery, ...]:
        """Expose in-flight Page-In bodies for deterministic duplicate-Fault suppression."""

        return tuple(self._pending.values())

    def recover_pending_recall_deliveries(
        self,
    ) -> tuple[tuple[PendingDelivery, DeliveryState], ...]:
        """Rebuild in-memory delivery state from the append-only journal."""

        rows = self.delivery.database.connection.execute(
            "SELECT delivery_id,state,block_json FROM context_deliveries "
            "WHERE thread_id=? AND recall_id NOT LIKE 'epoch:%' "
            "AND recall_id NOT LIKE 'control:%' "
            "AND state<>'MODEL_OBSERVED' ORDER BY prepared_at,delivery_id",
            (self.image.thread_id,),
        ).fetchall()
        recovered: list[tuple[PendingDelivery, DeliveryState]] = []
        for row in rows:
            block_value = json.loads(str(row["block_json"]))
            if not isinstance(block_value, dict):
                raise RuntimeError("persisted RecoveredContextBlock is invalid")
            block = RecoveredContextBlock.from_dict(block_value)
            pending = self.prepare_recovered(block)
            if pending is None:
                pending = self._pending.get(str(row["delivery_id"]))
            if pending is None or pending.delivery_id != str(row["delivery_id"]):
                raise RuntimeError("could not reconstruct pending Context delivery")
            recovered.append((pending, DeliveryState(str(row["state"]))))
        return tuple(recovered)

    def transport_accepted(self, pending: PendingDelivery) -> None:
        self._require_pending(pending)
        self.delivery.advance(
            pending.delivery_id,
            DeliveryState.TRANSPORT_ACCEPTED,
            context_digest=pending.block.content_digest,
        )

    def context_committed(self, pending: PendingDelivery) -> None:
        self._require_pending(pending)
        self.delivery.advance(
            pending.delivery_id,
            DeliveryState.CONTEXT_COMMITTED,
            context_digest=pending.block.content_digest,
        )

    def model_observed(
        self,
        pending: PendingDelivery,
        *,
        working_set_milestones: tuple[str, ...],
        focus_terms: tuple[str, ...] = (),
    ) -> AdmissionOutcome:
        self._require_pending(pending)
        outcome = self.admission.admit(
            current=self.image,
            incoming=(pending.artifact,),
            working_set_milestones=working_set_milestones,
            focus_terms=focus_terms,
        )
        # Residency and the authoritative image advance in one SQLite
        # transaction.  A crash cannot leave MODEL_OBSERVED with an old image.
        with self.delivery.database.transaction() as connection:
            self.delivery.advance(
                pending.delivery_id,
                DeliveryState.MODEL_OBSERVED,
                context_digest=pending.block.content_digest,
                connection=connection,
            )
            self._write_image(connection, outcome.image)
        self._image = outcome.image
        self._generation += 1
        self.metrics.increment(
            CounterName.RECOVERED_CONTEXT_TOKENS,
            pending.artifact.token_count,
        )
        self._pending.pop(pending.delivery_id, None)
        return outcome

    def admit_artifacts(
        self,
        incoming: tuple[ContextArtifact, ...],
        *,
        working_set_milestones: tuple[str, ...],
        focus_terms: tuple[str, ...] = (),
    ) -> AdmissionOutcome:
        outcome = self.admission.admit(
            current=self.image,
            incoming=incoming,
            working_set_milestones=working_set_milestones,
            focus_terms=focus_terms,
        )
        self._commit_image(outcome.image)
        return outcome

    def replace_artifacts(
        self,
        remove_artifact_ids: tuple[str, ...],
        incoming: tuple[ContextArtifact, ...],
        *,
        working_set_milestones: tuple[str, ...],
        focus_terms: tuple[str, ...] = (),
    ) -> AdmissionOutcome:
        """Atomically replace provisional facts, including their compacted descendants."""

        requested = set(remove_artifact_ids)
        matched = {
            requested_id
            for requested_id in requested
            if any(
                item.artifact_id == requested_id or requested_id in item.derived_from
                for item in self.image.artifacts
            )
        }
        missing = requested - matched
        if missing:
            raise KeyError(f"unknown artifact ids for replacement: {sorted(missing)}")
        remaining = tuple(
            item
            for item in self.image.artifacts
            if item.artifact_id not in requested and not requested.intersection(item.derived_from)
        )
        base = self.builder.build(
            thread_id=self.image.thread_id,
            artifacts=remaining,
            current_milestone_id=self.image.current_milestone_id,
            revision_id=self.image.revision_id,
        )
        outcome = self.admission.admit(
            current=base,
            incoming=incoming,
            working_set_milestones=working_set_milestones,
            focus_terms=focus_terms,
        )
        self._commit_image(outcome.image)
        return outcome

    def switch_scope(
        self,
        *,
        current_milestone_id: str,
        revision_id: str,
        current_milestone_artifact: ContextArtifact | None = None,
    ) -> ContextImage:
        """Switch Milestone/Revision inside the same Thread; never creates Epoch."""

        retained = tuple(
            item
            for item in self.image.artifacts
            if current_milestone_artifact is None
            or "context:milestone_identity" not in item.entity_refs
        )
        artifacts = tuple(
            replace(
                item,
                current_milestone=(
                    current_milestone_id in item.milestone_ids
                    and "context:milestone_identity" in item.entity_refs
                ),
                must_preserve=(
                    current_milestone_id in item.milestone_ids
                    if "context:milestone_identity" in item.entity_refs
                    else item.must_preserve
                ),
            )
            for item in retained
        )
        if current_milestone_artifact is not None:
            artifacts = (*artifacts, current_milestone_artifact)
        candidate = self.builder.build(
            thread_id=self.image.thread_id,
            artifacts=artifacts,
            current_milestone_id=current_milestone_id,
            revision_id=revision_id,
        )
        self._commit_image(candidate)
        return candidate

    def release_transition_handoffs(
        self,
        artifact_ids: tuple[str, ...] | None = None,
        *,
        focus_terms: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Compress consumed transition handoffs back to durable summaries."""

        requested = None if artifact_ids is None else set(artifact_ids)
        released: list[str] = []
        artifacts: list[ContextArtifact] = []
        for artifact in self.image.artifacts:
            is_handoff = "context:milestone_handoff" in artifact.entity_refs
            if not is_handoff or (requested is not None and artifact.artifact_id not in requested):
                artifacts.append(artifact)
                continue
            unpinned = replace(
                artifact,
                entity_refs=tuple(
                    dict.fromkeys(
                        (
                            *(
                                ref
                                for ref in artifact.entity_refs
                                if ref != "context:milestone_handoff"
                            ),
                            "context:consumed_milestone_handoff",
                        )
                    )
                ),
                soft_pin_boundaries=0,
                current_milestone=False,
            )
            compacted = self.admission.compactor.demote(unpinned, focus_terms=focus_terms)
            artifacts.append(compacted or unpinned)
            released.append(artifact.artifact_id)
        if requested is not None and requested.difference(released):
            raise KeyError(
                f"unknown transition handoff artifacts: {sorted(requested.difference(released))}"
            )
        if released:
            candidate = self.builder.build(
                thread_id=self.image.thread_id,
                artifacts=tuple(artifacts),
                current_milestone_id=self.image.current_milestone_id,
                revision_id=self.image.revision_id,
            )
            self._commit_image(candidate)
        return tuple(released)

    def replace_recovered_working_set_for_fault(
        self,
        block: RecoveredContextBlock,
        *,
        focus_terms: tuple[str, ...] = (),
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Page out older recovered bodies so a new exact section can enter.

        Recovered slices are a bounded working set, not immortal copies of the
        Page Store. If a different exact section cannot be admitted, demote the
        older bodies back to their MemoryRef summaries and retry on the same
        Thread. The immutable Pages and addresses remain authoritative.
        """

        incoming_address = self._virtual_section_address(block)
        replacements: dict[str, ContextArtifact] = {}
        released_pages: set[str] = set()
        for artifact in self.image.artifacts:
            if (
                artifact.representation is not Representation.SEMANTIC_SLICE
                or "context:recalled_slice" not in artifact.entity_refs
                or (
                    incoming_address is not None
                    and incoming_address in artifact.derived_from
                )
            ):
                continue
            unpinned = replace(
                artifact,
                entity_refs=tuple(
                    entity
                    for entity in artifact.entity_refs
                    if not entity.startswith("context:active_step_lease:")
                ),
                soft_pin_boundaries=0,
                current_milestone=False,
            )
            compacted = self.admission.compactor.demote(
                unpinned,
                focus_terms=focus_terms,
            )
            if compacted is None:
                continue
            replacements[artifact.artifact_id] = compacted
            released_pages.update(
                handle.page_id for handle in artifact.source_handles if handle.page_id
            )
        if not replacements:
            return (), ()
        candidate = self.builder.build(
            thread_id=self.image.thread_id,
            artifacts=tuple(
                replacements.get(artifact.artifact_id, artifact)
                for artifact in self.image.artifacts
            ),
            current_milestone_id=self.image.current_milestone_id,
            revision_id=self.image.revision_id,
        )
        self._commit_image(candidate)
        return tuple(replacements), tuple(sorted(released_pages))

    def compact_to_fixed_point(
        self,
        *,
        working_set_milestones: tuple[str, ...],
        focus_terms: tuple[str, ...] = (),
    ) -> AdmissionOutcome:
        outcome = self.admission.compact_to_fixed_point(
            current=self.image,
            working_set_milestones=working_set_milestones,
            focus_terms=focus_terms,
        )
        self._commit_image(outcome.image)
        return outcome

    def prepare_thread_replacement(
        self, *, new_thread_id: str, epoch_id: str
    ) -> tuple[ContextImage, str]:
        """Prepare (but do not activate) a candidate Epoch Context delivery."""

        candidate = self.builder.build(
            thread_id=new_thread_id,
            artifacts=self.image.artifacts,
            current_milestone_id=self.image.current_milestone_id,
            revision_id=self.image.revision_id,
        )
        delivery_id = self.delivery.prepare_context_image(
            thread_id=new_thread_id,
            epoch_id=epoch_id,
            image=candidate,
        )
        return candidate, delivery_id

    def replace_thread_after_model_observed(
        self,
        new_thread_id: str,
        *,
        epoch_id: str,
        thread_lifecycle: ThreadLifecycle,
        candidate_image: ContextImage | None = None,
    ) -> ContextImage:
        """Install a replacement Thread only after its image was observed.

        ``ThreadLifecycle.activate_after_model_observed`` independently checks
        the same delivery before fencing the predecessor Epoch.
        """

        candidate = candidate_image or self.builder.build(
            thread_id=new_thread_id,
            artifacts=self.image.artifacts,
            current_milestone_id=self.image.current_milestone_id,
            revision_id=self.image.revision_id,
        )
        self._validate_image(candidate)
        if candidate.thread_id != new_thread_id:
            raise ValueError("candidate ContextImage targets a different Thread")
        if not self.delivery.is_resident(candidate.image_digest, thread_id=new_thread_id):
            raise RuntimeError("candidate Thread Context is not MODEL_OBSERVED")
        if thread_lifecycle.database is not self.delivery.database:
            raise ValueError("Epoch and Context lifecycles must share one StateDatabase")
        # The Context carrier switch and predecessor fence share a transaction.
        with self.delivery.database.transaction() as connection:
            thread_lifecycle.activate_in_transaction(
                connection, epoch_id, expected_thread_id=new_thread_id
            )
            self._write_image(connection, candidate)
        self._image = candidate
        self._generation += 1
        thread_lifecycle.metrics.increment(CounterName.EPOCH)
        return candidate

    def advance_boundary(self) -> ContextImage:
        candidate = self.builder.advance_boundary(self.image)
        self._commit_image(candidate)
        return candidate

    @staticmethod
    def _step_lease(step_id: str) -> str:
        return f"context:active_step_lease:{step_id}"

    def pin_recalled_pages_to_step(
        self,
        page_ids: tuple[str, ...],
        *,
        step_id: str,
    ) -> tuple[str, ...]:
        """Keep actually used recalled Pages resident for the active Step.

        Observation alone retains the Slice for two semantic boundaries.  A
        validated causal use upgrades that short lease to Step locality.  The
        marker is advisory: urgent/hard Context Pressure may still demote the
        artifact, so this cannot turn recalled history into an immortal pin.
        """

        selected = {value for value in page_ids if value}
        normalized_step = step_id.strip()
        if not selected or not normalized_step:
            return ()
        marker = self._step_lease(normalized_step)
        replacements: dict[str, ContextArtifact] = {}
        for artifact in self.image.artifacts:
            artifact_pages = {
                handle.page_id for handle in artifact.source_handles if handle.page_id
            }
            if (
                artifact.representation is not Representation.SEMANTIC_SLICE
                or "context:recalled_slice" not in artifact.entity_refs
                or not selected.intersection(artifact_pages)
            ):
                continue
            entity_refs = tuple(
                dict.fromkeys(
                    (
                        *(
                            entity
                            for entity in artifact.entity_refs
                            if not entity.startswith("context:active_step_lease:")
                        ),
                        marker,
                    )
                )
            )
            replacements[artifact.artifact_id] = replace(
                artifact,
                entity_refs=entity_refs,
                soft_pin_boundaries=max(artifact.soft_pin_boundaries, 2),
            )
        if not replacements:
            return ()
        candidate = self.builder.build(
            thread_id=self.image.thread_id,
            artifacts=tuple(
                replacements.get(artifact.artifact_id, artifact)
                for artifact in self.image.artifacts
            ),
            current_milestone_id=self.image.current_milestone_id,
            revision_id=self.image.revision_id,
        )
        self._commit_image(candidate)
        return tuple(replacements)

    def demote_expired_recovered(
        self,
        *,
        focus_terms: tuple[str, ...] = (),
        active_step_id: str | None = None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return unpinned Page-In bodies to a compact summary/handle state."""

        replacements: dict[str, ContextArtifact] = {}
        released_pages: set[str] = set()
        active_lease = self._step_lease(active_step_id) if active_step_id else None
        for artifact in self.image.artifacts:
            if (
                artifact.representation is not Representation.SEMANTIC_SLICE
                or artifact.soft_pin_boundaries > 0
                or "context:recalled_slice" not in artifact.entity_refs
                or (active_lease is not None and active_lease in artifact.entity_refs)
            ):
                continue
            compacted = self.admission.compactor.demote(
                artifact,
                focus_terms=focus_terms,
            )
            if compacted is None:
                continue
            replacements[artifact.artifact_id] = compacted
            released_pages.update(
                handle.page_id for handle in artifact.source_handles if handle.page_id
            )
        if not replacements:
            return (), ()
        candidate = self.builder.build(
            thread_id=self.image.thread_id,
            artifacts=tuple(
                replacements.get(artifact.artifact_id, artifact)
                for artifact in self.image.artifacts
            ),
            current_milestone_id=self.image.current_milestone_id,
            revision_id=self.image.revision_id,
        )
        self._commit_image(candidate)
        return tuple(replacements), tuple(sorted(released_pages))

    def release_inactive_recalled_step_leases(
        self,
        *,
        active_step_id: str | None,
        focus_terms: tuple[str, ...] = (),
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Compress used recalled Pages when their owning Step has ended."""

        active_lease = self._step_lease(active_step_id) if active_step_id else None
        replacements: dict[str, ContextArtifact] = {}
        released_pages: set[str] = set()
        for artifact in self.image.artifacts:
            leases = tuple(
                entity
                for entity in artifact.entity_refs
                if entity.startswith("context:active_step_lease:")
            )
            if (
                artifact.representation is not Representation.SEMANTIC_SLICE
                or "context:recalled_slice" not in artifact.entity_refs
                or not leases
                or (active_lease is not None and active_lease in leases)
            ):
                continue
            unleased = replace(
                artifact,
                entity_refs=tuple(
                    entity
                    for entity in artifact.entity_refs
                    if not entity.startswith("context:active_step_lease:")
                ),
                soft_pin_boundaries=0,
            )
            compacted = self.admission.compactor.demote(
                unleased,
                focus_terms=focus_terms,
            )
            if compacted is None:
                continue
            replacements[artifact.artifact_id] = compacted
            released_pages.update(
                handle.page_id for handle in artifact.source_handles if handle.page_id
            )
        if not replacements:
            return (), ()
        candidate = self.builder.build(
            thread_id=self.image.thread_id,
            artifacts=tuple(
                replacements.get(artifact.artifact_id, artifact)
                for artifact in self.image.artifacts
            ),
            current_milestone_id=self.image.current_milestone_id,
            revision_id=self.image.revision_id,
        )
        self._commit_image(candidate)
        return tuple(replacements), tuple(sorted(released_pages))

    def build_continuity_checkpoint(
        self,
        *,
        task_goal_digest: str,
        plan_version_id: str,
        milestone_state_digest: str,
        workspace_revision_id: str,
        uncommitted_changes: tuple[str, ...] = (),
        unresolved_questions: tuple[str, ...] = (),
        failing_tests: tuple[str, ...] = (),
        pending_side_effects: tuple[str, ...] = (),
        latest_recovery_receipt_id: str | None = None,
        safe_action_boundary: bool,
        execution_handoff: Mapping[str, object] | None = None,
    ) -> ContinuityCheckpoint:
        """Capture the complete persisted continuity evidence for EpochAdmission."""

        nonresident_handles = tuple(
            dict.fromkeys(
                handle
                for artifact in self.image.artifacts
                if artifact.representation == Representation.NONRESIDENT
                for handle in artifact.source_handles
            )
        )
        latest_delivery = self.delivery.latest_state(thread_id=self.image.thread_id)
        delivery_state = latest_delivery or DeliveryState.MODEL_OBSERVED
        values = {
            "task_goal_digest": task_goal_digest,
            "plan_version_id": plan_version_id,
            "current_milestone_id": self.image.current_milestone_id,
            "milestone_state_digest": milestone_state_digest,
            "workspace_revision_id": workspace_revision_id,
            "uncommitted_changes": uncommitted_changes,
            "unresolved_questions": unresolved_questions,
            "failing_tests": failing_tests,
            "pending_side_effects": pending_side_effects,
            "resident_context_image": self.image,
            "nonresident_handles": nonresident_handles,
            "latest_recovery_receipt_id": latest_recovery_receipt_id,
            "delivery_state": delivery_state,
            "context_digest": self.image.image_digest,
            "safe_action_boundary": safe_action_boundary,
            "execution_handoff": dict(execution_handoff or {}),
        }
        checkpoint = ContinuityCheckpoint(
            checkpoint_id=stable_id("continuity_", values),
            **values,
        )
        self.delivery.database.connection.execute(
            """CREATE TABLE IF NOT EXISTS v2_continuity_checkpoints (
                   checkpoint_id TEXT PRIMARY KEY,
                   thread_id TEXT NOT NULL,
                   context_digest TEXT NOT NULL,
                   workspace_revision_id TEXT NOT NULL,
                   checkpoint_json TEXT NOT NULL,
                   created_at TEXT NOT NULL
               )"""
        )
        payload = json.dumps(
            primitive(checkpoint), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        with self.delivery.database.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO v2_continuity_checkpoints
                   VALUES(?,?,?,?,?,?)""",
                (
                    checkpoint.checkpoint_id,
                    self.image.thread_id,
                    checkpoint.context_digest,
                    checkpoint.workspace_revision_id,
                    payload,
                    utc_now(),
                ),
            )
            row = connection.execute(
                "SELECT checkpoint_json FROM v2_continuity_checkpoints WHERE checkpoint_id=?",
                (checkpoint.checkpoint_id,),
            ).fetchone()
            if row is None or str(row["checkpoint_json"]) != payload:
                raise RuntimeError("ContinuityCheckpoint identity collision")
        return checkpoint
