from __future__ import annotations

from collections.abc import Sequence

from ..contracts import RichGraphCapabilityReceipt
from ..observability.metrics import CounterName, MetricRecorder
from ..rich_graph import MilestoneFrontier, RichGraphScheduler
from .trace import TraceRecorder


class BackgroundJobs:
    """Production ownership boundary for optional, never-awaited Rich work.

    Every read of the Rich Graph goes through this object so that the
    consumption funnel (query -> hit -> used) is counted in one place.  The
    graph is a hint source: an empty answer never blocks a route, and an
    answer that is never used is reported as such instead of being assumed
    valuable.
    """

    def __init__(
        self,
        scheduler: RichGraphScheduler | None,
        trace: TraceRecorder,
        *,
        metrics: MetricRecorder | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.trace = trace
        self.metrics = metrics
        self._started = False

    def _count(self, name: CounterName, amount: int = 1) -> None:
        if self.metrics is not None and amount:
            self.metrics.increment(name, amount)

    def after_first_agent_action(
        self, frontier: MilestoneFrontier
    ) -> RichGraphCapabilityReceipt | None:
        if self.scheduler is None or self._started:
            return self.scheduler.capability() if self.scheduler is not None else None
        receipt = self.scheduler.start_after_first_action(frontier)
        self._started = True
        self.trace.record(
            "RICH_GRAPH_STARTED_AFTER_FIRST_ACTION",
            state=receipt.state.value,
            freshness_lag=receipt.freshness_lag,
            frontier=list(receipt.frontier),
        )
        return receipt

    def after_agent_action(self, frontier: MilestoneFrontier) -> RichGraphCapabilityReceipt | None:
        """Start on the first semantic action, then advance the live frontier.

        Raw Harness events and semantic Agent actions have different ordinals.
        In particular, Codex emits a synthetic ``TURN_STARTED`` before the
        first Page-worthy action.  Background lifecycle must therefore be
        keyed to this object's own durable start state, not to a Provider event
        position supplied by the caller.
        """

        if self.scheduler is None:
            return None
        if not self._started:
            return self.after_first_agent_action(frontier)
        self.submit(frontier)
        return self.scheduler.capability()

    def submit(self, frontier: MilestoneFrontier) -> int:
        if self.scheduler is None or not self._started:
            return 0
        count = self.scheduler.submit_frontier(frontier)
        self.trace.record("RICH_FRONTIER_SUBMITTED", queued=count)
        return count

    def resolve_addresses(
        self,
        tokens: tuple[str, ...],
        seeds: tuple[str, ...],
        *,
        revision_id: str | None = None,
        purpose: str = "CONTRACT_FREEZE",
    ) -> tuple[str, ...]:
        """Ask the Rich Graph for address candidates; empty when it knows nothing.

        The result is consumed as a ranked hint by the contract freeze.  The
        trace row is the funnel metric that shows whether the graph actually
        contributed addresses instead of only consuming CPU.
        """

        if self.scheduler is None:
            return ()
        self._count(CounterName.RICH_ADDRESS_QUERY)
        try:
            resolution = self.scheduler.resolve_addresses(
                tokens,
                seeds,
                revision_id=revision_id,
            )
        except Exception as exc:  # pragma: no cover - hint sources never block routes
            self.trace.record(
                "RICH_ADDRESS_RESOLUTION_FAILED",
                purpose=purpose,
                error=f"{type(exc).__name__}: {exc}",
            )
            return ()
        if resolution.candidates:
            self._count(CounterName.RICH_ADDRESS_HIT)
        self.trace.record(
            "RICH_ADDRESS_RESOLUTION",
            purpose=purpose,
            state=resolution.capability.state.value,
            tokens=list(tokens)[:16],
            seeds=list(seeds)[:8],
            by_name=resolution.by_name,
            by_structure=resolution.by_structure,
            seeds_queued=resolution.seeds_queued,
            candidates=len(resolution.candidates),
            outcome="HIT" if resolution.candidates else "MISS",
        )
        return resolution.candidates

    def related_observers(
        self,
        entity_ids: tuple[str, ...],
        *,
        revision_id: str | None = None,
        purpose: str = "RESULT_BINDING",
    ) -> tuple[str, ...]:
        """Tests/files structurally known to exercise ``entity_ids`` (hint only)."""

        if self.scheduler is None or not entity_ids:
            return ()
        self._count(CounterName.RICH_OBSERVER_QUERY)
        try:
            observers = self.scheduler.related_observers(entity_ids, revision_id=revision_id)
        except Exception as exc:  # pragma: no cover - hint sources never block routes
            self.trace.record(
                "RICH_RELATED_OBSERVERS_FAILED",
                purpose=purpose,
                error=f"{type(exc).__name__}: {exc}",
            )
            return ()
        if observers:
            self._count(CounterName.RICH_OBSERVER_HIT)
        self.trace.record(
            "RICH_RELATED_OBSERVERS",
            purpose=purpose,
            entities=list(entity_ids)[:12],
            observers=len(observers),
            outcome="HIT" if observers else "MISS",
        )
        return observers

    def changed_symbols(
        self,
        paths: Sequence[str],
        *,
        revision_id: str | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Symbols the Rich Graph knows inside ``paths`` (for Page finalization).

        The Semantic Graph stores facts at file granularity; attaching the
        structural symbols of a changed file gives later MemoryRefs a
        symbol-level address without re-parsing the Page body.
        """

        if self.scheduler is None or not paths:
            return {}
        try:
            symbols = self.scheduler.symbols_in_paths(tuple(paths), revision_id=revision_id)
        except Exception as exc:  # pragma: no cover - hint sources never block routes
            self.trace.record(
                "RICH_CHANGED_SYMBOLS_FAILED",
                error=f"{type(exc).__name__}: {exc}",
            )
            return {}
        attached = sum(len(values) for values in symbols.values())
        if attached:
            self._count(CounterName.RICH_CHANGED_SYMBOLS_ATTACHED, attached)
        return symbols

    def code_map(
        self,
        scope_entities: Sequence[str],
        *,
        revision_id: str | None = None,
    ) -> dict[str, object] | None:
        """Structural map of a Milestone scope (files, symbols, callers, coverage).

        A ``None`` result means the Rich Graph is disabled or unavailable; the
        route card then carries no map.  The map is a DERIVED hint and never
        enters the Evidence tables.
        """

        if self.scheduler is None or not scope_entities:
            return None
        self._count(CounterName.RICH_CODE_MAP_QUERY)
        try:
            code_map = self.scheduler.code_map(
                tuple(dict.fromkeys(scope_entities)),
                revision_id=revision_id,
            )
        except Exception as exc:  # pragma: no cover - hint sources never block routes
            self.trace.record(
                "RICH_CODE_MAP_FAILED",
                error=f"{type(exc).__name__}: {exc}",
            )
            return None
        if int(code_map.get("projected_files", 0)) > 0:
            self._count(CounterName.RICH_CODE_MAP_HIT)
        self.trace.record(
            "RICH_CODE_MAP_RESOLVED",
            revision_id=code_map.get("revision_id"),
            capability=code_map.get("capability"),
            projected_files=code_map.get("projected_files"),
            unprojected_files=code_map.get("unprojected_files"),
        )
        return code_map

    def search_code_graph(
        self,
        query: str,
        *,
        scope_entities: Sequence[str] = (),
        relation_kinds: Sequence[str] = (),
        revision_id: str | None = None,
        limit: int = 8,
        purpose: str = "navigation",
    ) -> dict[str, object]:
        """Read a bounded Rich Graph search answer without touching the route."""

        self._count(CounterName.RICH_CODE_SEARCH_QUERY)
        if self.scheduler is None:
            self._count(CounterName.RICH_CODE_SEARCH_FAILED)
            return {
                "schema": "homy/rich-code-search@1",
                "status": "UNAVAILABLE",
                "revision_id": revision_id,
                "matches": [],
                "truncated": False,
            }
        try:
            result = self.scheduler.search_code(
                str(query),
                scope_entities=tuple(map(str, scope_entities)),
                relation_kinds=tuple(map(str, relation_kinds)),
                revision_id=revision_id,
                limit=limit,
            )
        except Exception as exc:  # pragma: no cover - hint sources never block routes
            self._count(CounterName.RICH_CODE_SEARCH_FAILED)
            self.trace.record(
                "RICH_CODE_SEARCH_FAILED",
                purpose=purpose,
                error=f"{type(exc).__name__}: {exc}",
            )
            return {
                "schema": "homy/rich-code-search@1",
                "status": "UNAVAILABLE",
                "revision_id": revision_id,
                "matches": [],
                "truncated": False,
            }
        if result.get("matches"):
            self._count(CounterName.RICH_CODE_SEARCH_HIT)
        self.trace.record(
            "RICH_CODE_SEARCH_RESOLVED",
            purpose=purpose,
            query=str(query)[:200],
            revision_id=result.get("revision_id"),
            status=result.get("status"),
            matches=len(result.get("matches", ())),
            truncated=bool(result.get("truncated")),
            queued_files=int(result.get("queued_files", 0)),
        )
        return result

    def record_search_used(self, *, addresses: int = 1) -> None:
        """Count graph search hits that the model subsequently used as addresses."""

        self._count(CounterName.RICH_CODE_SEARCH_USED, max(0, int(addresses)))

    def record_used(self, *, addresses: int = 0, observers: int = 0) -> None:
        """Count hint candidates that entered a durable decision."""

        self._count(CounterName.RICH_ADDRESS_USED, addresses)
        self._count(CounterName.RICH_OBSERVER_USED, observers)

    def close(self) -> None:
        if self.scheduler is None:
            return
        # close() stops the worker; it is not a Recall-path wait and cannot
        # affect the completed Agent action latency.
        self.scheduler.close()
        self.trace.record("RICH_GRAPH_CLOSED", started=self._started)
