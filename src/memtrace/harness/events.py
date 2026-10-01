from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..contracts import digest, primitive, stable_id, utc_now
from ..database import StateDatabase
from .contracts import SIDE_EFFECT_ITEM_TYPES, HarnessEvent, HarnessEventType


class RawHarnessEventLedger:
    """Durable provider telemetry kept outside the Memory Trace stream."""

    def __init__(self, database: StateDatabase) -> None:
        self.database = database
        self.database.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS v2_raw_harness_events (
                harness_event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                turn_id TEXT,
                phase TEXT NOT NULL,
                event_type TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                provider_method TEXT NOT NULL,
                provider_sequence INTEGER NOT NULL,
                provider_time_ms INTEGER,
                payload_digest TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                observed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_v2_raw_harness_scope
            ON v2_raw_harness_events(run_id,branch_id,phase,provider_sequence);
            CREATE TRIGGER IF NOT EXISTS v2_raw_harness_events_no_update
            BEFORE UPDATE ON v2_raw_harness_events
            BEGIN SELECT RAISE(ABORT, 'Raw Harness events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS v2_raw_harness_events_no_delete
            BEFORE DELETE ON v2_raw_harness_events
            BEGIN SELECT RAISE(ABORT, 'Raw Harness events are append-only'); END;
            """
        )

    def append(self, event: HarnessEvent, *, phase: str) -> None:
        item = event.payload.get("item")
        item = item if isinstance(item, Mapping) else {}
        item_error = item.get("error")
        summary = {
            "payload_keys": sorted(map(str, event.payload.keys())),
            "partial": bool(event.payload.get("partial", False)),
            "item": {
                "id": item.get("id"),
                "type": item.get("type"),
                "status": item.get("status"),
                "exit_code": item.get("exitCode"),
                "tool": item.get("tool", item.get("name")),
                "error_type": (
                    item_error.get("type", item_error.get("code"))
                    if isinstance(item_error, Mapping)
                    else type(item_error).__name__
                    if item_error is not None
                    else None
                ),
                "error_digest": digest(primitive(item_error)) if item_error is not None else None,
            },
            "raw_provider_summary_keys": sorted(map(str, event.raw_provider_summary.keys())),
            "raw_provider_summary_digest": digest(primitive(event.raw_provider_summary)),
        }
        payload_digest = digest(primitive(event.payload))
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT payload_digest,phase FROM v2_raw_harness_events WHERE harness_event_id=?",
                (event.harness_event_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["payload_digest"]) != payload_digest
                    or str(existing["phase"]) != phase
                ):
                    raise ValueError("Harness event identity was replayed with different raw facts")
                return
            connection.execute(
                "INSERT INTO v2_raw_harness_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event.harness_event_id,
                    event.run_id,
                    event.branch_id,
                    event.thread_id,
                    event.turn_id,
                    phase,
                    event.event_type.value,
                    event.source_event_id,
                    event.provider_method,
                    event.sequence,
                    event.provider_time_ms,
                    payload_digest,
                    json.dumps(summary, ensure_ascii=False, sort_keys=True),
                    utc_now(),
                ),
            )

    def has_durable_source_event(self, source_event_id: str) -> bool:
        """Whether a provider/control fact is durable outside the Page WAL."""

        row = self.database.connection.execute(
            "SELECT 1 FROM v2_raw_harness_events WHERE source_event_id=? LIMIT 1",
            (source_event_id,),
        ).fetchone()
        return row is not None


class CodexEventMapper:
    """Map native App Server messages to deduplicated provider-neutral events."""

    def __init__(
        self,
        *,
        run_id: str,
        branch_id: str,
        thread_id: str,
        revision_id: str,
    ) -> None:
        self.run_id = run_id
        self.branch_id = branch_id
        self.thread_id = thread_id
        self.revision_id = revision_id
        self._sequence = 0
        self._seen: set[str] = set()

    def update_revision(self, revision_id: str) -> None:
        self.revision_id = revision_id

    def update_thread(self, thread_id: str) -> None:
        """Move subsequent provider facts to a confirmed replacement Thread."""

        normalized = thread_id.strip()
        if not normalized:
            raise ValueError("replacement Thread ID must be non-empty")
        self.thread_id = normalized

    @staticmethod
    def _normalize_observed_path(value: object, cwd: object) -> str | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        path = Path(raw)
        base = Path(str(cwd)).resolve() if str(cwd or "").strip() else None
        if path.is_absolute():
            if base is None:
                return None
            try:
                path = path.resolve().relative_to(base)
            except (OSError, ValueError):
                # External reads are not addresses in the repository's
                # Memory Index.
                return None
        normalized = path.as_posix().removeprefix("./")
        if not normalized or normalized == "." or normalized.startswith("../"):
            return None
        return normalized

    @classmethod
    def _accessed_paths(cls, item: Mapping[str, Any]) -> tuple[str, ...]:
        """Extract repository reads from native commandActions.

        Codex already classifies ordinary shell commands as read/search/list
        actions.  Preserving their paths at the Harness boundary gives the
        Memory Index stable ``file:...`` addresses instead of opaque
        tool call IDs.
        """

        actions = item.get("commandActions", ())
        if not isinstance(actions, (list, tuple)):
            return ()
        cwd = item.get("cwd")
        paths: list[str] = []
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            if str(action.get("type", "")).casefold() != "read":
                continue
            candidate = cls._normalize_observed_path(action.get("path"), cwd)
            if candidate is not None and candidate not in paths:
                paths.append(candidate)
        return tuple(paths)

    @staticmethod
    def _provider_time(params: Mapping[str, Any]) -> int | None:
        for key in ("completedAtMs", "startedAtMs", "timestampMs"):
            value = params.get(key)
            if isinstance(value, int):
                return value
        return None

    def _make(
        self,
        *,
        event_type: HarnessEventType,
        method: str,
        params: Mapping[str, Any],
        payload: Mapping[str, Any],
        discriminator: str = "",
    ) -> HarnessEvent | None:
        turn = params.get("turn")
        turn_id = params.get("turnId")
        if turn_id is None and isinstance(turn, Mapping):
            turn_id = turn.get("id")
        event_key = digest(
            {
                "method": method,
                "event_type": event_type.value,
                "thread": self.thread_id,
                "turn": turn_id,
                "discriminator": discriminator,
                "params": params,
            }
        )
        if event_key in self._seen:
            return None
        self._seen.add(event_key)
        self._sequence += 1
        harness_event_id = stable_id(
            "harness_",
            {"run": self.run_id, "event_key": event_key},
        )
        return HarnessEvent(
            harness_event_id=harness_event_id,
            event_type=event_type,
            thread_id=self.thread_id,
            turn_id=str(turn_id) if turn_id is not None else None,
            sequence=self._sequence,
            provider_time_ms=self._provider_time(params),
            run_id=self.run_id,
            branch_id=self.branch_id,
            revision_id=self.revision_id,
            source_event_id=stable_id("event_", {"harness_event": harness_event_id}),
            provider_method=method,
            payload=primitive(payload),
            raw_provider_summary={
                "method": method,
                "digest": digest(params),
                "thread_id": self.thread_id,
                "turn_id": str(turn_id) if turn_id is not None else None,
            },
        )

    def turn_started(self, turn_id: str) -> HarnessEvent:
        event = self._make(
            event_type=HarnessEventType.TURN_STARTED,
            method="turn/started",
            params={"threadId": self.thread_id, "turnId": turn_id},
            payload={"turn_id": turn_id, "synthetic_from_response": True},
        )
        assert event is not None
        return event

    def local_event(
        self,
        event_type: HarnessEventType,
        *,
        provider_method: str,
        payload: Mapping[str, Any],
        turn_id: str | None = None,
        discriminator: str = "",
    ) -> HarnessEvent:
        params: dict[str, Any] = {"threadId": self.thread_id, **dict(payload)}
        if turn_id is not None:
            params["turnId"] = turn_id
        event = self._make(
            event_type=event_type,
            method=provider_method,
            params=params,
            payload=payload,
            discriminator=discriminator,
        )
        if event is None:
            raise ValueError("local HarnessEvent duplicates an earlier provider fact")
        return event

    def map_message(self, message: Mapping[str, Any]) -> tuple[HarnessEvent, ...]:
        method = str(message.get("method", ""))
        params_value = message.get("params", {})
        if not isinstance(params_value, Mapping):
            return ()
        params: Mapping[str, Any] = params_value
        if params.get("threadId") not in (None, self.thread_id):
            return ()
        outputs: list[HarnessEvent | None] = []
        if method == "thread/started":
            outputs.append(
                self._make(
                    event_type=HarnessEventType.THREAD_STARTED,
                    method=method,
                    params=params,
                    payload=params,
                )
            )
        elif method == "turn/started":
            turn = params.get("turn")
            turn_id = str(turn.get("id")) if isinstance(turn, Mapping) else ""
            # A turn/start response is already durable; discard the matching notification.
            response_key = digest(
                {
                    "method": method,
                    "event_type": HarnessEventType.TURN_STARTED.value,
                    "thread": self.thread_id,
                    "turn": turn_id,
                    "discriminator": "",
                    "params": {
                        "threadId": self.thread_id,
                        "turnId": turn_id,
                    },
                }
            )
            if response_key not in self._seen:
                outputs.append(
                    self._make(
                        event_type=HarnessEventType.TURN_STARTED,
                        method=method,
                        params=params,
                        payload=params,
                    )
                )
        elif method == "turn/plan/updated":
            outputs.append(
                self._make(
                    event_type=HarnessEventType.PLAN_UPDATED,
                    method=method,
                    params=params,
                    payload={
                        "plan": params.get("plan", ()),
                        "explanation": params.get("explanation"),
                    },
                )
            )
        elif method in {"item/started", "item/completed"}:
            item = params.get("item")
            if not isinstance(item, Mapping):
                return ()
            item_type = str(item.get("type", "unknown"))
            item_id = str(item.get("id", ""))
            accessed_paths = self._accessed_paths(item)
            item_payload = {
                "item": item,
                **({"accessed_paths": accessed_paths} if accessed_paths else {}),
            }
            lifecycle_type = (
                HarnessEventType.ITEM_STARTED
                if method == "item/started"
                else HarnessEventType.ITEM_COMPLETED
            )
            outputs.append(
                self._make(
                    event_type=lifecycle_type,
                    method=method,
                    params=params,
                    payload=item_payload,
                    discriminator=f"{item_id}:lifecycle",
                )
            )
            if item_type in SIDE_EFFECT_ITEM_TYPES:
                outputs.append(
                    self._make(
                        event_type=(
                            HarnessEventType.TOOL_INTENT
                            if method == "item/started"
                            else HarnessEventType.TOOL_RESULT
                        ),
                        method=method,
                        params=params,
                        payload={**item_payload, "partial": False},
                        discriminator=f"{item_id}:side-effect",
                    )
                )
            if method == "item/completed" and item_type == "fileChange":
                changes = item.get("changes", ())
                paths = [
                    str(change.get("path"))
                    for change in changes
                    if isinstance(change, Mapping) and change.get("path")
                ]
                outputs.append(
                    self._make(
                        event_type=HarnessEventType.FILE_CHANGED,
                        method=method,
                        params=params,
                        payload={"item": item, "paths": paths, "provisional": False},
                        discriminator=f"{item_id}:revision",
                    )
                )
            if method == "item/completed" and item_type == "plan":
                outputs.append(
                    self._make(
                        event_type=HarnessEventType.PLAN_PROPOSED,
                        method=method,
                        params=params,
                        payload={"text": item.get("text", ""), "item_id": item_id},
                        discriminator=f"{item_id}:plan",
                    )
                )
            if method == "item/completed" and item_type == "contextCompaction":
                outputs.append(
                    self._make(
                        event_type=HarnessEventType.CONTEXT_COMPACTED,
                        method=method,
                        params=params,
                        payload={"item": item},
                        discriminator=f"{item_id}:compaction",
                    )
                )
        elif method == "item/commandExecution/outputDelta":
            outputs.append(
                self._make(
                    event_type=HarnessEventType.TOOL_RESULT,
                    method=method,
                    params=params,
                    payload={**dict(params), "partial": True},
                )
            )
        elif method in {"turn/diff/updated", "item/fileChange/patchUpdated"}:
            outputs.append(
                self._make(
                    event_type=HarnessEventType.FILE_CHANGED,
                    method=method,
                    params=params,
                    payload={**dict(params), "provisional": True},
                )
            )
        elif method == "thread/tokenUsage/updated":
            outputs.append(
                self._make(
                    event_type=HarnessEventType.TOKEN_USAGE_UPDATED,
                    method=method,
                    params=params,
                    payload={"token_usage": params.get("tokenUsage", {})},
                )
            )
        elif method in {"thread/compacted", "contextCompaction"}:
            outputs.append(
                self._make(
                    event_type=HarnessEventType.CONTEXT_COMPACTED,
                    method=method,
                    params=params,
                    payload=params,
                )
            )
        elif method == "error":
            error = params.get("error", {})
            encoded = str(error).casefold()
            if "contextwindowexceeded" in encoded:
                event_type = HarnessEventType.PHYSICAL_CONTEXT_FAILURE
            elif any(
                marker in encoded
                for marker in (
                    "threadunrecoverable",
                    "thread unrecoverable",
                    "thread_not_found",
                )
            ):
                event_type = HarnessEventType.THREAD_UNRECOVERABLE
            elif any(
                marker in encoded for marker in ("sessionlost", "session lost", "session_not_found")
            ):
                event_type = HarnessEventType.SESSION_LOST
            else:
                event_type = HarnessEventType.ITEM_COMPLETED
            outputs.append(
                self._make(
                    event_type=event_type,
                    method=method,
                    params=params,
                    payload={"error": error, "will_retry": params.get("willRetry", False)},
                )
            )
        elif method == "turn/completed":
            outputs.append(
                self._make(
                    event_type=HarnessEventType.TURN_COMPLETED,
                    method=method,
                    params=params,
                    payload={"turn": params.get("turn", {})},
                )
            )
        return tuple(item for item in outputs if item is not None)
