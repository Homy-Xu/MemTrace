from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..contracts import digest, primitive, stable_id, utc_now
from ..database import StateDatabase


@dataclass(frozen=True, slots=True)
class DynamicToolInvocation:
    """One synchronous App Server dynamic-tool request.

    The model owns only the semantic arguments. Internal Page addresses remain
    behind the Harness boundary and are never copied into this request.
    """

    request_id: str | int
    call_id: str
    tool: str
    arguments: Mapping[str, Any]
    thread_id: str
    turn_id: str


@dataclass(frozen=True, slots=True)
class DynamicToolResult:
    """Model-visible result plus runtime-only delivery correlation metadata."""

    success: bool
    text: str
    delivery_id: str | None = None
    entity_refs: tuple[str, ...] = ()
    evidence_handles: tuple[str, ...] = ()
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)


_OPERATION_ORDER = {
    "REQUEST_DURABLE": 0,
    "RECALL_PREPARED": 1,
    "RESPONSE_WRITTEN": 2,
    "MODEL_OBSERVED": 3,
}


class DynamicToolJournal:
    """Durable inbox/outbox for synchronous Provider tool commands."""

    def __init__(
        self,
        database: StateDatabase,
        *,
        run_id: str,
        branch_id: str,
    ) -> None:
        self.database = database
        self.run_id = run_id
        self.branch_id = branch_id
        self.database.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS v2_dynamic_tool_operations (
                operation_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                call_id TEXT NOT NULL,
                tool TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                request_json TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN
                    ('REQUEST_DURABLE','RECALL_PREPARED','RESPONSE_WRITTEN','MODEL_OBSERVED')),
                result_digest TEXT,
                result_json TEXT,
                delivery_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(thread_id,turn_id,call_id)
            );
            CREATE TABLE IF NOT EXISTS v2_dynamic_tool_state_events (
                transition_id TEXT PRIMARY KEY,
                operation_id TEXT NOT NULL REFERENCES v2_dynamic_tool_operations(operation_id),
                previous_state TEXT,
                state TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(operation_id,state)
            );
            CREATE INDEX IF NOT EXISTS ix_v2_dynamic_tool_pending
            ON v2_dynamic_tool_operations(thread_id,turn_id,state);
            CREATE TRIGGER IF NOT EXISTS v2_dynamic_tool_state_events_no_update
            BEFORE UPDATE ON v2_dynamic_tool_state_events
            BEGIN SELECT RAISE(ABORT, 'Dynamic tool state events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS v2_dynamic_tool_state_events_no_delete
            BEFORE DELETE ON v2_dynamic_tool_state_events
            BEGIN SELECT RAISE(ABORT, 'Dynamic tool state events are append-only'); END;
            """
        )

    def _operation_id(self, invocation: DynamicToolInvocation) -> str:
        return stable_id(
            "memoryop_",
            {
                "thread": invocation.thread_id,
                "turn": invocation.turn_id,
                "call": invocation.call_id,
            },
        )

    @staticmethod
    def _result_payload(result: DynamicToolResult) -> Mapping[str, object]:
        return {
            "success": result.success,
            "text": result.text,
            "delivery_id": result.delivery_id,
            "entity_refs": list(result.entity_refs),
            "evidence_handles": list(result.evidence_handles),
            "runtime_metadata": primitive(result.runtime_metadata),
        }

    @staticmethod
    def _result_from_json(value: str) -> DynamicToolResult:
        payload = json.loads(value)
        return DynamicToolResult(
            success=bool(payload["success"]),
            text=str(payload["text"]),
            delivery_id=(str(payload["delivery_id"]) if payload.get("delivery_id") else None),
            entity_refs=tuple(map(str, payload.get("entity_refs", ()))),
            evidence_handles=tuple(map(str, payload.get("evidence_handles", ()))),
            runtime_metadata=dict(payload.get("runtime_metadata", {})),
        )

    def request_durable(self, invocation: DynamicToolInvocation) -> DynamicToolResult | None:
        """Persist the external command before any recall or Context mutation."""

        operation_id = self._operation_id(invocation)
        request = {
            "request_id": invocation.request_id,
            "call_id": invocation.call_id,
            "tool": invocation.tool,
            "arguments": primitive(invocation.arguments),
            "thread_id": invocation.thread_id,
            "turn_id": invocation.turn_id,
        }
        request_digest = digest(request)
        request_json = json.dumps(request, ensure_ascii=False, sort_keys=True)
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT request_digest,state,result_json FROM v2_dynamic_tool_operations "
                "WHERE thread_id=? AND turn_id=? AND call_id=?",
                (invocation.thread_id, invocation.turn_id, invocation.call_id),
            ).fetchone()
            if existing is not None:
                if str(existing["request_digest"]) != request_digest:
                    raise ValueError("dynamic tool call identity was reused with different facts")
                if existing["result_json"] is not None:
                    return self._result_from_json(str(existing["result_json"]))
                return None
            now = utc_now()
            connection.execute(
                "INSERT INTO v2_dynamic_tool_operations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    operation_id,
                    self.run_id,
                    self.branch_id,
                    invocation.thread_id,
                    invocation.turn_id,
                    invocation.call_id,
                    invocation.tool,
                    request_digest,
                    request_json,
                    "REQUEST_DURABLE",
                    None,
                    None,
                    None,
                    now,
                    now,
                ),
            )
            self._append_transition(
                connection,
                operation_id=operation_id,
                previous_state=None,
                state="REQUEST_DURABLE",
                source_event_id=f"DYNAMIC_REQUEST:{invocation.call_id}",
            )
        return None

    @staticmethod
    def _append_transition(
        connection,
        *,
        operation_id: str,
        previous_state: str | None,
        state: str,
        source_event_id: str,
    ) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO v2_dynamic_tool_state_events VALUES(?,?,?,?,?,?)",
            (
                stable_id("memoryopstate_", {"operation": operation_id, "state": state}),
                operation_id,
                previous_state,
                state,
                source_event_id,
                utc_now(),
            ),
        )

    def result_prepared(
        self,
        invocation: DynamicToolInvocation,
        result: DynamicToolResult,
    ) -> None:
        payload = self._result_payload(result)
        result_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._advance(
            invocation,
            "RECALL_PREPARED",
            source_event_id=f"DYNAMIC_RESULT:{invocation.call_id}",
            result_digest=digest(payload),
            result_json=result_json,
            delivery_id=result.delivery_id,
        )

    def response_written(
        self,
        invocation: DynamicToolInvocation,
        result: DynamicToolResult,
    ) -> None:
        self._advance(
            invocation,
            "RESPONSE_WRITTEN",
            source_event_id=f"DYNAMIC_RESPONSE:{invocation.call_id}",
            result_digest=digest(self._result_payload(result)),
            result_json=json.dumps(
                self._result_payload(result), ensure_ascii=False, sort_keys=True
            ),
            delivery_id=result.delivery_id,
        )

    def model_observed(self, operation_id: str, *, source_event_id: str) -> None:
        row = self.database.connection.execute(
            "SELECT thread_id,turn_id,call_id FROM v2_dynamic_tool_operations WHERE operation_id=?",
            (operation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(operation_id)
        invocation = DynamicToolInvocation(
            request_id="recovered",
            call_id=str(row["call_id"]),
            tool="recovered",
            arguments={},
            thread_id=str(row["thread_id"]),
            turn_id=str(row["turn_id"]),
        )
        self._advance(
            invocation,
            "MODEL_OBSERVED",
            source_event_id=source_event_id,
            operation_id=operation_id,
        )

    def _advance(
        self,
        invocation: DynamicToolInvocation,
        target: str,
        *,
        source_event_id: str,
        result_digest: str | None = None,
        result_json: str | None = None,
        delivery_id: str | None = None,
        operation_id: str | None = None,
    ) -> None:
        selected_id = operation_id or self._operation_id(invocation)
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT state,result_digest,result_json,delivery_id "
                "FROM v2_dynamic_tool_operations WHERE operation_id=?",
                (selected_id,),
            ).fetchone()
            if row is None:
                raise KeyError(selected_id)
            current = str(row["state"])
            if _OPERATION_ORDER[current] >= _OPERATION_ORDER[target]:
                if result_digest is not None and row["result_digest"] not in (
                    None,
                    result_digest,
                ):
                    raise ValueError("dynamic tool result changed during idempotent replay")
                return
            if _OPERATION_ORDER[target] != _OPERATION_ORDER[current] + 1:
                raise ValueError(f"invalid dynamic tool transition {current}->{target}")
            connection.execute(
                "UPDATE v2_dynamic_tool_operations SET state=?,result_digest=COALESCE(?,result_digest),"
                "result_json=COALESCE(?,result_json),delivery_id=COALESCE(?,delivery_id),updated_at=? "
                "WHERE operation_id=?",
                (
                    target,
                    result_digest,
                    result_json,
                    delivery_id,
                    utc_now(),
                    selected_id,
                ),
            )
            self._append_transition(
                connection,
                operation_id=selected_id,
                previous_state=current,
                state=target,
                source_event_id=source_event_id,
            )

    def pending_observations(
        self,
        *,
        thread_id: str,
        turn_id: str | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        query = (
            "SELECT operation_id,turn_id,call_id,tool,delivery_id "
            "FROM v2_dynamic_tool_operations WHERE thread_id=? "
        )
        parameters: tuple[object, ...] = (thread_id,)
        if turn_id is not None:
            query += "AND turn_id=? "
            parameters += (turn_id,)
        rows = self.database.connection.execute(
            query + "AND state='RESPONSE_WRITTEN' ORDER BY created_at,operation_id",
            parameters,
        ).fetchall()
        return tuple(
            {
                "operation_id": str(row["operation_id"]),
                "turn_id": str(row["turn_id"]),
                "call_id": str(row["call_id"]),
                "tool": str(row["tool"]),
                "delivery_id": (
                    str(row["delivery_id"]) if row["delivery_id"] is not None else None
                ),
            }
            for row in rows
        )

    def state_for_delivery(self, delivery_id: str) -> str | None:
        row = self.database.connection.execute(
            "SELECT state FROM v2_dynamic_tool_operations WHERE delivery_id=? "
            "ORDER BY updated_at DESC LIMIT 1",
            (delivery_id,),
        ).fetchone()
        return str(row["state"]) if row is not None else None
