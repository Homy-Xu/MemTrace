from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from ..durability import append_synced, json_line, load_json_lines


class BenchmarkPhaseJournal:
    """Append-only phase evidence for one benchmark attempt."""

    def __init__(self, run_root: Path) -> None:
        self.path = run_root / "phase-events.jsonl"

    def record(self, phase: str, state: str, **payload: object) -> None:
        append_synced(
            self.path,
            json_line(
                {
                    "schema": "codex-longterm-v2/swe-evo-phase@1",
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "phase": phase,
                    "state": state,
                    **payload,
                }
            ),
        )

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter_ns()
        self.record(name, "STARTED")
        try:
            yield
        except BaseException as exc:
            self.record(
                name,
                "FAILED",
                duration_ms=(time.perf_counter_ns() - started) / 1_000_000,
                error_type=type(exc).__name__,
                error_message=str(exc)[:4000],
            )
            raise
        else:
            self.record(
                name,
                "COMPLETED",
                duration_ms=(time.perf_counter_ns() - started) / 1_000_000,
            )

    def durations_ms(self) -> dict[str, float]:
        return {
            str(event["phase"]): float(event["duration_ms"])
            for event in load_json_lines(self.path)
            if event.get("state") == "COMPLETED" and event.get("duration_ms") is not None
        }


def last_phase(run_root: Path) -> str | None:
    events = load_json_lines(run_root / "phase-events.jsonl")
    return str(events[-1]["phase"]) if events else None
