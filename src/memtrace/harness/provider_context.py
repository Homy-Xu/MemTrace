from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from ..context_runtime.pressure import PressurePolicy
from ..contracts import PressureLevel, digest, utc_now
from ..database import StateDatabase
from ..observability import CounterName, MetricRecorder
from .contracts import HarnessEvent, HarnessEventType


class ProviderCompactionState(StrEnum):
    NOT_OBSERVED = "NOT_OBSERVED"
    EVENT_OBSERVED_AWAITING_USAGE = "EVENT_OBSERVED_AWAITING_USAGE"
    VERIFIED_PHYSICAL_REDUCTION = "VERIFIED_PHYSICAL_REDUCTION"
    NO_PHYSICAL_REDUCTION_OBSERVED = "NO_PHYSICAL_REDUCTION_OBSERVED"


@dataclass(frozen=True, slots=True)
class ProviderContextObservation:
    event_id: str
    thread_id: str
    turn_id: str | None
    physical_context_tokens: int | None
    cumulative_tokens: int | None
    model_context_window: int | None
    pressure: PressureLevel | None
    compaction_state: ProviderCompactionState


class ProviderContextLedger:
    """Append-only observations of Codex's physical context carrier.

    The local ContextImage estimate and Provider usage intentionally remain
    independent. Codex's cumulative ``total`` is billing/activity evidence;
    only ``last.totalTokens`` is treated as the observable current Turn size.
    """

    def __init__(
        self,
        database: StateDatabase,
        metrics: MetricRecorder,
        policy: PressurePolicy,
    ) -> None:
        self.database = database
        self.metrics = metrics
        self.policy = policy
        self.database.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS v2_provider_context_observations (
                event_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                turn_id TEXT,
                event_type TEXT NOT NULL CHECK(event_type IN
                    ('TOKEN_USAGE','CONTEXT_COMPACTION')),
                physical_context_tokens INTEGER,
                cumulative_tokens INTEGER,
                model_context_window INTEGER,
                pressure TEXT,
                compaction_state TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                observed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_v2_provider_context_thread
            ON v2_provider_context_observations(thread_id, observed_at, event_id);
            CREATE TRIGGER IF NOT EXISTS v2_provider_context_no_update
            BEFORE UPDATE ON v2_provider_context_observations
            BEGIN SELECT RAISE(ABORT, 'Provider context observations are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS v2_provider_context_no_delete
            BEFORE DELETE ON v2_provider_context_observations
            BEGIN SELECT RAISE(ABORT, 'Provider context observations are append-only'); END;
            """
        )

    @staticmethod
    def _nonnegative_int(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    @classmethod
    def _nested_total(cls, value: object) -> int | None:
        if not isinstance(value, Mapping):
            return None
        for key in ("totalTokens", "total_tokens"):
            parsed = cls._nonnegative_int(value.get(key))
            if parsed is not None:
                return parsed
        return None

    def _pressure(
        self, physical_tokens: int | None, model_context_window: int | None
    ) -> PressureLevel | None:
        if physical_tokens is None:
            return None
        budget = self.policy.budget
        if model_context_window is None:
            # Without a Provider observation, the configured budget must
            # reserve system/tool/output capacity itself.
            return self.policy.level(physical_tokens)
        # App Server's modelContextWindow is the authoritative physical
        # carrier limit for the reported ``last.totalTokens`` value.  The
        # Provider/Codex runtime has already applied its own usable-window
        # policy (and may report a value below the configured model limit).
        # Subtracting the local logical-image reserves again would double
        # reserve headroom and trigger disruptive compaction far too early.
        if model_context_window <= 0:
            return PressureLevel.HARD
        ratio = physical_tokens / model_context_window
        if ratio < budget.soft_ratio:
            return PressureLevel.NORMAL
        if ratio < budget.urgent_ratio:
            return PressureLevel.SOFT
        if ratio < budget.hard_ratio:
            return PressureLevel.URGENT
        return PressureLevel.HARD

    def observe(self, event: HarnessEvent) -> ProviderContextObservation:
        if event.event_type not in {
            HarnessEventType.TOKEN_USAGE_UPDATED,
            HarnessEventType.CONTEXT_COMPACTED,
        }:
            raise ValueError("event is not a Provider context observation")
        previous = self.latest(event.thread_id)
        if event.event_type is HarnessEventType.TOKEN_USAGE_UPDATED:
            usage = event.payload.get("token_usage", {})
            if not isinstance(usage, Mapping):
                usage = {}
            physical = self._nested_total(usage.get("last"))
            cumulative = self._nested_total(usage.get("total"))
            window = self._nonnegative_int(
                usage.get("modelContextWindow", usage.get("model_context_window"))
            )
            pressure = self._pressure(physical, window)
            state = ProviderCompactionState.NOT_OBSERVED
            if previous is not None and previous.compaction_state == (
                ProviderCompactionState.EVENT_OBSERVED_AWAITING_USAGE
            ):
                before = previous.physical_context_tokens
                if before is not None and physical is not None and physical < before:
                    state = ProviderCompactionState.VERIFIED_PHYSICAL_REDUCTION
                    self.metrics.increment(CounterName.PROVIDER_NATIVE_COMPACTION)
                else:
                    state = ProviderCompactionState.NO_PHYSICAL_REDUCTION_OBSERVED
        else:
            # Codex can emit the reduced TOKEN_USAGE observation immediately
            # before CONTEXT_COMPACTED.  Compare the two latest physical
            # observations first; otherwise retain the event-before-usage path.
            physical = previous.physical_context_tokens if previous is not None else None
            cumulative = previous.cumulative_tokens if previous is not None else None
            window = previous.model_context_window if previous is not None else None
            pressure = previous.pressure if previous is not None else None
            token_rows = self.database.connection.execute(
                "SELECT physical_context_tokens FROM v2_provider_context_observations "
                "WHERE thread_id=? AND event_type='TOKEN_USAGE' "
                "AND physical_context_tokens IS NOT NULL ORDER BY rowid DESC LIMIT 2",
                (event.thread_id,),
            ).fetchall()
            reduced_before_event = len(token_rows) == 2 and int(
                token_rows[0]["physical_context_tokens"]
            ) < int(token_rows[1]["physical_context_tokens"])
            if reduced_before_event:
                state = ProviderCompactionState.VERIFIED_PHYSICAL_REDUCTION
                self.metrics.increment(CounterName.PROVIDER_NATIVE_COMPACTION)
            else:
                state = ProviderCompactionState.EVENT_OBSERVED_AWAITING_USAGE

        observation = ProviderContextObservation(
            event_id=event.harness_event_id,
            thread_id=event.thread_id,
            turn_id=event.turn_id,
            physical_context_tokens=physical,
            cumulative_tokens=cumulative,
            model_context_window=window,
            pressure=pressure,
            compaction_state=state,
        )
        payload_json = json.dumps(event.payload, ensure_ascii=False, sort_keys=True)
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO v2_provider_context_observations
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    observation.event_id,
                    observation.thread_id,
                    observation.turn_id,
                    (
                        "TOKEN_USAGE"
                        if event.event_type is HarnessEventType.TOKEN_USAGE_UPDATED
                        else "CONTEXT_COMPACTION"
                    ),
                    observation.physical_context_tokens,
                    observation.cumulative_tokens,
                    observation.model_context_window,
                    observation.pressure.value if observation.pressure is not None else None,
                    observation.compaction_state.value,
                    event.source_event_id,
                    digest(event.payload),
                    payload_json,
                    utc_now(),
                ),
            )
        self.metrics.set("provider_context_tokens", physical)
        self.metrics.set("provider_context_limit", window)
        self.metrics.set(
            "provider_context_pressure", pressure.value if pressure is not None else None
        )
        self.metrics.set("provider_compaction_state", state.value)
        return observation

    def latest(self, thread_id: str) -> ProviderContextObservation | None:
        row = self.database.connection.execute(
            """SELECT * FROM v2_provider_context_observations
               WHERE thread_id=? ORDER BY rowid DESC LIMIT 1""",
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        return ProviderContextObservation(
            event_id=str(row["event_id"]),
            thread_id=str(row["thread_id"]),
            turn_id=None if row["turn_id"] is None else str(row["turn_id"]),
            physical_context_tokens=(
                None
                if row["physical_context_tokens"] is None
                else int(row["physical_context_tokens"])
            ),
            cumulative_tokens=(
                None if row["cumulative_tokens"] is None else int(row["cumulative_tokens"])
            ),
            model_context_window=(
                None if row["model_context_window"] is None else int(row["model_context_window"])
            ),
            pressure=(None if row["pressure"] is None else PressureLevel(str(row["pressure"]))),
            compaction_state=ProviderCompactionState(str(row["compaction_state"])),
        )

    def recall_admission_limit(
        self,
        thread_id: str,
        *,
        configured_limit: int,
        unaccounted_provider_tokens: int = 0,
        minimum_useful_tokens: int = 256,
    ) -> int:
        """Bound Memory Loading against observable Provider headroom.

        Normal and soft contexts may admit only up to the urgent boundary.
        Urgent contexts use a small slice of the remaining hard-boundary
        headroom.  Hard contexts defer recall until compaction.  The caller's
        unaccounted delivery debt closes the gap between transport acceptance
        and the next Provider token observation.
        """

        if configured_limit <= 0 or unaccounted_provider_tokens < 0:
            raise ValueError("Provider recall admission inputs must be valid")
        if minimum_useful_tokens <= 0:
            raise ValueError("minimum useful recall tokens must be positive")
        observation = self.latest(thread_id)
        if observation is None or observation.physical_context_tokens is None:
            return configured_limit
        budget = self.policy.budget
        effective_limit = (
            observation.model_context_window
            if observation.model_context_window is not None
            else budget.effective_limit
        )
        if effective_limit <= 0:
            return 0
        accounted = observation.physical_context_tokens + unaccounted_provider_tokens
        urgent_boundary = int(effective_limit * budget.urgent_ratio)
        hard_boundary = int(effective_limit * budget.hard_ratio)
        if accounted >= hard_boundary:
            return 0
        if accounted < urgent_boundary:
            headroom = urgent_boundary - accounted
        else:
            headroom = min(
                hard_boundary - accounted,
                max(minimum_useful_tokens, configured_limit // 4),
            )
        admitted = min(configured_limit, max(0, headroom))
        return admitted if admitted >= minimum_useful_tokens else 0

    def observe_logical_image(self, *, tokens: int, pressure: PressureLevel) -> None:
        self.metrics.set("logical_context_tokens", tokens)
        self.metrics.set("logical_context_pressure", pressure.value)


def provider_context_payload(observation: ProviderContextObservation) -> dict[str, Any]:
    return {
        "physical_context_tokens": observation.physical_context_tokens,
        "cumulative_tokens": observation.cumulative_tokens,
        "model_context_window": observation.model_context_window,
        "pressure": observation.pressure.value if observation.pressure is not None else None,
        "compaction_state": observation.compaction_state.value,
    }
