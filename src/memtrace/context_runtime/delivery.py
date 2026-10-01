from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from ..contracts import (
    ContextImage,
    DeliveryState,
    RecoveredContextBlock,
    digest,
    primitive,
    stable_id,
    utc_now,
)
from ..database import StateDatabase

_ORDER = {
    DeliveryState.PREPARED: 0,
    DeliveryState.TRANSPORT_ACCEPTED: 1,
    DeliveryState.CONTEXT_COMMITTED: 2,
    DeliveryState.MODEL_OBSERVED: 3,
}

_PROVIDER_COMPACTION_RECALL_PREFIX = "control:provider-compaction:"


@dataclass(frozen=True, slots=True)
class ProviderCompactionRefreshDelivery:
    """Durable control-plane delivery restoring state after Provider compaction."""

    delivery_id: str
    thread_id: str
    source_event_id: str
    context_digest: str
    rendered_content: str
    state: DeliveryState


class DeliveryJournal:
    def __init__(self, database: StateDatabase) -> None:
        self.database = database
        self.database.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS context_deliveries (
                delivery_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                recall_id TEXT NOT NULL,
                block_id TEXT NOT NULL,
                context_digest TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN
                    ('PREPARED','TRANSPORT_ACCEPTED','CONTEXT_COMMITTED','MODEL_OBSERVED')),
                block_json TEXT NOT NULL,
                prepared_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                model_observed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_context_deliveries_digest_state
            ON context_deliveries(thread_id,context_digest,state);
            CREATE TABLE IF NOT EXISTS context_delivery_state_events (
                transition_id TEXT PRIMARY KEY,
                delivery_id TEXT NOT NULL REFERENCES context_deliveries(delivery_id),
                previous_state TEXT,
                state TEXT NOT NULL CHECK(state IN
                    ('PREPARED','TRANSPORT_ACCEPTED','CONTEXT_COMMITTED','MODEL_OBSERVED')),
                context_digest TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(delivery_id,state)
            );
            CREATE TRIGGER IF NOT EXISTS context_delivery_events_no_update
            BEFORE UPDATE ON context_delivery_state_events
            BEGIN SELECT RAISE(ABORT, 'Context delivery state events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS context_delivery_events_no_delete
            BEFORE DELETE ON context_delivery_state_events
            BEGIN SELECT RAISE(ABORT, 'Context delivery state events are append-only'); END;
            """
        )
        # Older V2 databases had only the authoritative current-state row. Keep
        # them readable and add an explicit migration snapshot without
        # pretending the unavailable historical transitions can be recreated.
        with self.database.transaction() as connection:
            for row in connection.execute(
                "SELECT delivery_id,state,context_digest FROM context_deliveries"
            ).fetchall():
                self._append_state_event(
                    connection,
                    delivery_id=str(row["delivery_id"]),
                    previous_state=None,
                    state=DeliveryState(str(row["state"])),
                    context_digest=str(row["context_digest"]),
                    source_event_id="SCHEMA_MIGRATION_CURRENT_STATE",
                )

    @staticmethod
    def _append_state_event(
        connection: sqlite3.Connection,
        *,
        delivery_id: str,
        previous_state: DeliveryState | None,
        state: DeliveryState,
        context_digest: str,
        source_event_id: str,
    ) -> None:
        connection.execute(
            """INSERT OR IGNORE INTO context_delivery_state_events
               (transition_id,delivery_id,previous_state,state,context_digest,
                source_event_id,created_at) VALUES(?,?,?,?,?,?,?)""",
            (
                stable_id("deliverystate_", {"delivery": delivery_id, "state": state.value}),
                delivery_id,
                None if previous_state is None else previous_state.value,
                state.value,
                context_digest,
                source_event_id,
                utc_now(),
            ),
        )

    def prepare(self, block: RecoveredContextBlock, *, thread_id: str) -> str:
        if not thread_id:
            raise ValueError("delivery thread_id must be non-empty")
        if block.content_digest != digest({"rendered_content": block.rendered_content}):
            raise ValueError("RecoveredContextBlock content digest mismatch")
        if block.token_count != max(1, (len(block.rendered_content.encode("utf-8")) + 2) // 3):
            raise ValueError("RecoveredContextBlock token count mismatch")
        return self._prepare_payload(
            thread_id=thread_id,
            recall_id=block.recall_id,
            block_id=block.block_id,
            context_digest=block.content_digest,
            payload=primitive(block),
        )

    def prepare_context_image(
        self,
        *,
        thread_id: str,
        epoch_id: str,
        image: ContextImage,
    ) -> str:
        """Prepare the candidate-image injection for a replacement Epoch."""

        if not epoch_id:
            raise ValueError("epoch_id must be non-empty")
        expected_digest = digest(
            {
                "artifacts": [primitive(item) for item in image.artifacts],
                "total_tokens": image.total_tokens,
                "current_milestone_id": image.current_milestone_id,
                "revision_id": image.revision_id,
            }
        )
        if (
            not thread_id
            or thread_id != image.thread_id
            or image.total_tokens != sum(item.token_count for item in image.artifacts)
            or image.image_digest != expected_digest
        ):
            raise ValueError("invalid candidate ContextImage")
        return self._prepare_payload(
            thread_id=thread_id,
            recall_id=f"epoch:{epoch_id}",
            block_id=image.image_id,
            context_digest=image.image_digest,
            payload=primitive(image),
        )

    def prepare_provider_compaction_refresh(
        self,
        *,
        thread_id: str,
        source_event_id: str,
        rendered_content: str,
    ) -> ProviderCompactionRefreshDelivery:
        """Prepare one idempotent Working Memory refresh after Provider compaction.

        This deliberately reuses the Context delivery journal: Memory Loading, Epoch
        continuity and Provider-compaction recovery have the same transport
        truth states even though only Memory Loading becomes semantic memory.
        """

        content = rendered_content.strip()
        if not thread_id or not source_event_id or not content:
            raise ValueError("Provider compaction refresh fields must be non-empty")
        payload = {
            "kind": "PROVIDER_COMPACTION_WORKING_SET_REFRESH",
            "source_event_id": source_event_id,
            "rendered_content": content,
        }
        context_digest = digest(payload)
        delivery_id = self._prepare_payload(
            thread_id=thread_id,
            recall_id=_PROVIDER_COMPACTION_RECALL_PREFIX + source_event_id,
            block_id=stable_id(
                "control_",
                {"thread_id": thread_id, "source_event_id": source_event_id},
            ),
            context_digest=context_digest,
            payload=payload,
        )
        record = self.provider_compaction_refresh(delivery_id)
        if record is None:
            raise RuntimeError("prepared Provider compaction refresh is not readable")
        return record

    def provider_compaction_refresh(
        self, delivery_id: str
    ) -> ProviderCompactionRefreshDelivery | None:
        row = self.database.connection.execute(
            """SELECT delivery_id,thread_id,recall_id,context_digest,state,block_json
               FROM context_deliveries WHERE delivery_id=?""",
            (delivery_id,),
        ).fetchone()
        if row is None or not str(row["recall_id"]).startswith(
            _PROVIDER_COMPACTION_RECALL_PREFIX
        ):
            return None
        value = json.loads(str(row["block_json"]))
        if (
            not isinstance(value, dict)
            or value.get("kind") != "PROVIDER_COMPACTION_WORKING_SET_REFRESH"
        ):
            raise RuntimeError("persisted Provider compaction refresh is invalid")
        source_event_id = str(value.get("source_event_id", ""))
        rendered_content = str(value.get("rendered_content", ""))
        if (
            not source_event_id
            or not rendered_content
            or str(row["recall_id"])
            != _PROVIDER_COMPACTION_RECALL_PREFIX + source_event_id
            or str(row["context_digest"]) != digest(value)
        ):
            raise RuntimeError("persisted Provider compaction refresh identity changed")
        return ProviderCompactionRefreshDelivery(
            delivery_id=str(row["delivery_id"]),
            thread_id=str(row["thread_id"]),
            source_event_id=source_event_id,
            context_digest=str(row["context_digest"]),
            rendered_content=rendered_content,
            state=DeliveryState(str(row["state"])),
        )

    def pending_provider_compaction_refreshes(
        self, *, thread_id: str
    ) -> tuple[ProviderCompactionRefreshDelivery, ...]:
        rows = self.database.connection.execute(
            """SELECT delivery_id FROM context_deliveries
               WHERE thread_id=? AND recall_id LIKE ? AND state<>'MODEL_OBSERVED'
               ORDER BY prepared_at,delivery_id""",
            (thread_id, _PROVIDER_COMPACTION_RECALL_PREFIX + "%"),
        ).fetchall()
        result: list[ProviderCompactionRefreshDelivery] = []
        for row in rows:
            record = self.provider_compaction_refresh(str(row["delivery_id"]))
            if record is None:
                raise RuntimeError("Provider compaction refresh disappeared during recovery")
            result.append(record)
        return tuple(result)

    def provider_compaction_refresh_for_source(
        self, *, thread_id: str, source_event_id: str
    ) -> ProviderCompactionRefreshDelivery | None:
        rows = self.database.connection.execute(
            """SELECT delivery_id FROM context_deliveries
               WHERE thread_id=? AND recall_id=? ORDER BY prepared_at,delivery_id""",
            (thread_id, _PROVIDER_COMPACTION_RECALL_PREFIX + source_event_id),
        ).fetchall()
        if len(rows) > 1:
            raise RuntimeError("one Provider compaction event has multiple refresh deliveries")
        if not rows:
            return None
        return self.provider_compaction_refresh(str(rows[0]["delivery_id"]))

    def _prepare_payload(
        self,
        *,
        thread_id: str,
        recall_id: str,
        block_id: str,
        context_digest: str,
        payload: Any,
    ) -> str:
        delivery_id = stable_id(
            "delivery_",
            {
                "thread_id": thread_id,
                "recall_id": recall_id,
                "digest": context_digest,
            },
        )
        with self.database.transaction() as connection:
            row = connection.execute(
                """SELECT thread_id,recall_id,block_id,context_digest,block_json
                   FROM context_deliveries WHERE delivery_id=?""",
                (delivery_id,),
            ).fetchone()
            payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if row is not None:
                observed = (
                    str(row["thread_id"]),
                    str(row["recall_id"]),
                    str(row["block_id"]),
                    str(row["context_digest"]),
                    str(row["block_json"]),
                )
                expected = (
                    thread_id,
                    recall_id,
                    block_id,
                    context_digest,
                    payload_json,
                )
                if observed != expected:
                    raise RuntimeError("delivery identity collision")
            connection.execute(
                """INSERT OR IGNORE INTO context_deliveries
                   (delivery_id,thread_id,recall_id,block_id,context_digest,state,block_json,
                    prepared_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    delivery_id,
                    thread_id,
                    recall_id,
                    block_id,
                    context_digest,
                    DeliveryState.PREPARED.value,
                    payload_json,
                    utc_now(),
                    utc_now(),
                ),
            )
            if row is None:
                self._append_state_event(
                    connection,
                    delivery_id=delivery_id,
                    previous_state=None,
                    state=DeliveryState.PREPARED,
                    context_digest=context_digest,
                    source_event_id=f"PREPARE:{recall_id}",
                )
        return delivery_id

    def advance(
        self,
        delivery_id: str,
        state: DeliveryState,
        *,
        context_digest: str,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if connection is None:
            with self.database.transaction() as owned_connection:
                self._advance(
                    owned_connection,
                    delivery_id,
                    state,
                    context_digest=context_digest,
                )
            return
        self._advance(connection, delivery_id, state, context_digest=context_digest)

    def advance_to(
        self,
        delivery_id: str,
        state: DeliveryState,
        *,
        context_digest: str,
    ) -> None:
        """Advance through implied transport states, idempotently.

        A stronger Provider fact (for example, an accepted same-Turn steer)
        proves the intermediate states as well. This helper is intentionally
        used by control-plane recovery; ordinary Memory Loading keeps its stricter
        step-by-step lifecycle API.
        """

        while True:
            current = self.state(delivery_id)
            if _ORDER[current] >= _ORDER[state]:
                return
            next_state = next(
                candidate
                for candidate, rank in _ORDER.items()
                if rank == _ORDER[current] + 1
            )
            self.advance(
                delivery_id,
                next_state,
                context_digest=context_digest,
            )

    @staticmethod
    def _advance(
        connection: sqlite3.Connection,
        delivery_id: str,
        state: DeliveryState,
        *,
        context_digest: str,
    ) -> None:
        row = connection.execute(
            "SELECT context_digest,state FROM context_deliveries WHERE delivery_id=?",
            (delivery_id,),
        ).fetchone()
        if row is None:
            raise KeyError(delivery_id)
        if str(row["context_digest"]) != context_digest:
            raise ValueError("delivery digest mismatch")
        current = DeliveryState(str(row["state"]))
        if _ORDER[state] != _ORDER[current] + 1:
            raise ValueError(f"invalid delivery transition {current}->{state}")
        now = utc_now()
        connection.execute(
            """UPDATE context_deliveries
               SET state=?,updated_at=?,model_observed_at=? WHERE delivery_id=?""",
            (
                state.value,
                now,
                now if state == DeliveryState.MODEL_OBSERVED else None,
                delivery_id,
            ),
        )
        DeliveryJournal._append_state_event(
            connection,
            delivery_id=delivery_id,
            previous_state=current,
            state=state,
            context_digest=context_digest,
            source_event_id=f"DELIVERY_TRANSITION:{state.value}",
        )

    def is_resident(self, context_digest: str, *, thread_id: str) -> bool:
        return (
            self.database.connection.execute(
                """SELECT 1 FROM context_deliveries
                   WHERE thread_id=? AND context_digest=? AND state=? LIMIT 1""",
                (thread_id, context_digest, DeliveryState.MODEL_OBSERVED.value),
            ).fetchone()
            is not None
        )

    def latest_state(self, *, thread_id: str) -> DeliveryState | None:
        row = self.database.connection.execute(
            """SELECT state FROM context_deliveries WHERE thread_id=?
               ORDER BY updated_at DESC, delivery_id DESC LIMIT 1""",
            (thread_id,),
        ).fetchone()
        return None if row is None else DeliveryState(str(row[0]))

    def assert_model_observed(
        self,
        delivery_id: str,
        *,
        context_digest: str,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        source = connection or self.database.connection
        row = source.execute(
            "SELECT state,context_digest FROM context_deliveries WHERE delivery_id=?",
            (delivery_id,),
        ).fetchone()
        if row is None:
            raise KeyError(delivery_id)
        if str(row["context_digest"]) != context_digest:
            raise ValueError("delivery digest mismatch")
        if DeliveryState(str(row["state"])) != DeliveryState.MODEL_OBSERVED:
            raise RuntimeError("delivery has not reached MODEL_OBSERVED")

    def state(self, delivery_id: str) -> DeliveryState:
        row = self.database.connection.execute(
            "SELECT state FROM context_deliveries WHERE delivery_id=?",
            (delivery_id,),
        ).fetchone()
        if row is None:
            raise KeyError(delivery_id)
        return DeliveryState(str(row[0]))
