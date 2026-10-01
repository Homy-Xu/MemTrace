from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from ..contracts import stable_id, utc_now
from ..database import StateDatabase


class WorkingSetHeat(StrEnum):
    HOT = "HOT"
    COOLING = "COOLING"
    COLD = "COLD"


@dataclass(frozen=True, slots=True)
class WorkingSetEntry:
    kind: str
    value: str
    heat: WorkingSetHeat
    boundaries_remaining: int
    revision_id: str


class WorkingSetTracker:
    """Persisted hot/cooling frontier; Trace Store, not this cache, retains history."""

    _TTL = {
        "CURRENT_MILESTONE": 3,
        "DEPENDENCY_MILESTONE": 2,
        "MODIFIED_FILE": 4,
        "ACCESSED_FILE": 2,
        "RECENT_SYMBOL": 2,
        "FAILED_TEST": 4,
        "FAILURE_SIGNATURE": 4,
        "UNRESOLVED_QUESTION": 4,
    }
    # Entries whose heat may follow structural distance instead of recency.
    _STRUCTURAL_KINDS = frozenset({"MODIFIED_FILE", "ACCESSED_FILE", "RECENT_SYMBOL"})

    def __init__(self, database: StateDatabase, run_id: str) -> None:
        self.database = database
        self.run_id = run_id
        self.database.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS v2_working_set (
                run_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                value TEXT NOT NULL,
                heat TEXT NOT NULL CHECK(heat IN ('HOT','COOLING','COLD')),
                boundaries_remaining INTEGER NOT NULL,
                revision_id TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id,kind,value)
            );
            CREATE TABLE IF NOT EXISTS v2_working_set_events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS v2_working_set_events_no_update
            BEFORE UPDATE ON v2_working_set_events
            BEGIN SELECT RAISE(ABORT, 'Working Memory events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS v2_working_set_events_no_delete
            BEFORE DELETE ON v2_working_set_events
            BEGIN SELECT RAISE(ABORT, 'Working Memory events are append-only'); END;
            """
        )

    def update(
        self,
        *,
        revision_id: str,
        source_event_id: str,
        current_milestone_id: str,
        dependency_milestone_ids: tuple[str, ...] = (),
        modified_files: tuple[str, ...] = (),
        accessed_files: tuple[str, ...] = (),
        recent_symbols: tuple[str, ...] = (),
        failed_tests: tuple[str, ...] = (),
        failure_signatures: tuple[str, ...] = (),
        unresolved_questions: tuple[str, ...] = (),
        structural_scope: frozenset[str] = frozenset(),
    ) -> tuple[WorkingSetEntry, ...]:
        """Advance the frontier by one observed action.

        ``structural_scope`` is the Rich-derived neighbourhood of the current
        Milestone (files, symbols, covering tests).  A cached file or symbol
        inside that scope stays HOT even when this action did not touch it,
        because the model will need it again soon; one outside the scope cools
        twice as fast.  Logical eviction therefore follows structural distance
        rather than pure recency.  An empty scope keeps the recency-only rule.
        """

        desired = {
            ("CURRENT_MILESTONE", current_milestone_id),
            *(("DEPENDENCY_MILESTONE", value) for value in dependency_milestone_ids),
            *(("MODIFIED_FILE", value) for value in modified_files),
            *(("ACCESSED_FILE", value) for value in accessed_files),
            *(("RECENT_SYMBOL", value) for value in recent_symbols),
            *(("FAILED_TEST", value) for value in failed_tests),
            *(("FAILURE_SIGNATURE", value) for value in failure_signatures),
            *(("UNRESOLVED_QUESTION", value) for value in unresolved_questions),
        }
        desired = {(kind, value) for kind, value in desired if value}
        with self.database.transaction() as connection:
            previous_snapshot = self._snapshot(connection)
            rows = connection.execute(
                "SELECT kind,value,boundaries_remaining FROM v2_working_set WHERE run_id=?",
                (self.run_id,),
            ).fetchall()
            existing = {(str(row["kind"]), str(row["value"])): int(row[2]) for row in rows}
            for key, remaining in existing.items():
                if key in desired:
                    continue
                kind, value = key
                structural_kind = kind in self._STRUCTURAL_KINDS
                if structural_scope and structural_kind and value in structural_scope:
                    # Structurally adjacent to the current focus: keep warm.
                    connection.execute(
                        """UPDATE v2_working_set SET heat=?,boundaries_remaining=?,updated_at=?
                           WHERE run_id=? AND kind=? AND value=?""",
                        (
                            WorkingSetHeat.HOT.value,
                            max(remaining, self._TTL[kind]),
                            utc_now(),
                            self.run_id,
                            *key,
                        ),
                    )
                    continue
                decay = 2 if structural_scope and structural_kind else 1
                next_remaining = max(0, remaining - decay)
                if next_remaining == 0:
                    connection.execute(
                        "DELETE FROM v2_working_set WHERE run_id=? AND kind=? AND value=?",
                        (self.run_id, *key),
                    )
                    continue
                connection.execute(
                    """UPDATE v2_working_set SET heat=?,boundaries_remaining=?,updated_at=?
                       WHERE run_id=? AND kind=? AND value=?""",
                    (
                        WorkingSetHeat.COOLING.value,
                        next_remaining,
                        utc_now(),
                        self.run_id,
                        *key,
                    ),
                )
            for kind, value in desired:
                connection.execute(
                    """INSERT INTO v2_working_set
                       VALUES(?,?,?,?,?,?,?)
                       ON CONFLICT(run_id,kind,value) DO UPDATE SET
                         heat=excluded.heat,
                         boundaries_remaining=excluded.boundaries_remaining,
                         revision_id=excluded.revision_id,
                         updated_at=excluded.updated_at""",
                    (
                        self.run_id,
                        kind,
                        value,
                        WorkingSetHeat.HOT.value,
                        self._TTL[kind],
                        revision_id,
                        utc_now(),
                    ),
                )
            snapshot = self._snapshot(connection)
            payload = json.dumps(
                [
                    {
                        "kind": item.kind,
                        "value": item.value,
                        "heat": item.heat.value,
                        "boundaries_remaining": item.boundaries_remaining,
                        "revision_id": item.revision_id,
                    }
                    for item in snapshot
                ],
                ensure_ascii=False,
                sort_keys=True,
            )
            if snapshot != previous_snapshot:
                event_id = stable_id(
                    "working_",
                    {"run": self.run_id, "source": source_event_id, "snapshot": payload},
                )
                connection.execute(
                    "INSERT OR IGNORE INTO v2_working_set_events VALUES(?,?,?,?,?)",
                    (event_id, self.run_id, source_event_id, payload, utc_now()),
                )
        return snapshot

    def snapshot(self) -> tuple[WorkingSetEntry, ...]:
        return self._snapshot(self.database.connection)

    def _snapshot(self, connection) -> tuple[WorkingSetEntry, ...]:
        rows = connection.execute(
            """SELECT kind,value,heat,boundaries_remaining,revision_id
               FROM v2_working_set WHERE run_id=?
               ORDER BY CASE heat WHEN 'HOT' THEN 0 WHEN 'COOLING' THEN 1 ELSE 2 END,
                        kind,value""",
            (self.run_id,),
        ).fetchall()
        return tuple(
            WorkingSetEntry(
                kind=str(row["kind"]),
                value=str(row["value"]),
                heat=WorkingSetHeat(str(row["heat"])),
                boundaries_remaining=int(row["boundaries_remaining"]),
                revision_id=str(row["revision_id"]),
            )
            for row in rows
        )
