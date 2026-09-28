from __future__ import annotations

import copy
import re
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..contracts import RichGraphCapabilityReceipt, RichGraphState
from ..observability.metrics import MetricRecorder
from .models import (
    FileProjection,
    FrontierTask,
    MilestoneFrontier,
    RichGraphHintReceipt,
    is_structural_relation,
)
from .processor import process_frontier_file
from .store import RichGraphStore
from ..language_adapters import language_for_path
from .project_resolution import dependency_paths

FrontierProcessor = Callable[[FrontierTask], FileProjection | None]


@dataclass(order=True, slots=True)
class _QueueItem:
    priority: int
    sequence: int
    task: FrontierTask | None = None


@dataclass(frozen=True, slots=True)
class RichAddressResolution:
    """Structural address candidates for one requirement (hint only)."""

    candidates: tuple[str, ...]
    by_name: int
    by_structure: int
    seeds_queued: int
    capability: RichGraphCapabilityReceipt


_DEFAULT_NEIGHBOUR_BUDGET = 256


def _neighbour_budget_from_environment() -> int | None:
    """``HOMY_RICH_GRAPH_NEIGHBOUR_BUDGET``: per-generation cap; ``0`` disables the cap."""

    import os

    raw = os.environ.get("HOMY_RICH_GRAPH_NEIGHBOUR_BUDGET", "").strip()
    if not raw:
        return _DEFAULT_NEIGHBOUR_BUDGET
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_NEIGHBOUR_BUDGET
    return None if value <= 0 else value


_PRIORITIES: tuple[tuple[str, int, str], ...] = (
    ("current_milestone_files", 10, "current_milestone"),
    ("modified_files", 20, "modified"),
    ("failed_tests", 30, "failed_test"),
    ("recent_symbols", 40, "recent_symbol"),
    ("dependency_files", 50, "milestone_dependency"),
    ("accessed_files", 60, "accessed"),
    ("prefetch_files", 90, "next_milestone_prefetch"),
)


class RichGraphScheduler:
    """Single-worker, non-blocking, Milestone-frontier Rich Graph scheduler.

    The class has no repository enumeration path. Every projection is rooted in
    an explicit frontier signal or a structural hint miss, and the one daemon
    worker processes exactly one file per queue item.
    """

    def __init__(
        self,
        repository_path: Path,
        repository_id: str,
        database_path: Path | None = None,
        *,
        metrics: MetricRecorder | None = None,
        processor: FrontierProcessor | object | None = None,
    ) -> None:
        self.repository_path = Path(repository_path).expanduser().resolve()
        if not self.repository_path.is_dir():
            raise ValueError(f"repository path is not a directory: {repository_path}")
        if not repository_id:
            raise ValueError("repository_id is required")
        self.repository_id = repository_id
        self.database_path = (
            Path(database_path).expanduser().resolve()
            if database_path is not None
            else self.repository_path / ".codex-v2" / "rich-graph.sqlite3"
        )
        self.metrics = metrics or MetricRecorder()
        self._processor = processor
        self._store = RichGraphStore(self.database_path)
        self._queue: queue.PriorityQueue[_QueueItem] = queue.PriorityQueue()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._pending: dict[tuple[str, str], tuple[int, int, FrontierTask]] = {}
        self._sequence = 0
        self._thread: threading.Thread | None = None
        self._stop = False
        self._closed = False
        self._store_closed = False
        self._inflight: FrontierTask | None = None
        self._state = RichGraphState.NOT_STARTED
        self._revision_id = "unknown"
        self._generation_id: str | None = None
        self._supported_relations: set[str] = set()
        self._failure_reason: str | None = None
        self._last_cache_hit = False
        self._search_cache: dict[tuple[object, ...], dict[str, object]] = {}
        # The default/Python path keeps its established queue behaviour.
        # Explicit non-Python frontiers opt into stale-generation eviction and
        # a bounded pending queue.
        self._bounded_multilang_navigation = False
        self._multilang_pending_limit = 64
        # Import-neighbour fan-out budget per generation.  Frontier signals
        # (files the model touched, failing tests, prefetch) are never budgeted;
        # only the background neighbour expansion is, so a large TypeScript or
        # Go monorepo cannot queue thousands of files behind one edit.
        self._neighbour_budget = _neighbour_budget_from_environment()
        self._neighbour_enqueued: dict[str, int] = {}

    @property
    def generation_id(self) -> str | None:
        with self._lock:
            return self._generation_id

    @staticmethod
    def needs_structural_hint(
        required_relations: tuple[str, ...] = (),
        ambiguous_entities: tuple[str, ...] = (),
    ) -> bool:
        return bool(ambiguous_entities) or any(
            is_structural_relation(item) for item in required_relations
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RichGraphScheduler is closed")

    def _activate_revision(self, revision_id: str) -> None:
        if not revision_id:
            raise ValueError("workspace revision is required")
        if revision_id == self._revision_id and self._generation_id is not None:
            return
        previous = self._generation_id
        self._revision_id = revision_id
        self._generation_id = self._store.ensure_generation(self.repository_id, revision_id)
        self._supported_relations = set(self._store.supported_relations(self._generation_id))
        self._failure_reason = None
        self._last_cache_hit = False
        self._search_cache.clear()
        if previous is not None and self._bounded_multilang_navigation:
            # PriorityQueue items cannot be removed cheaply. Dropping their
            # authoritative pending records makes the worker discard stale
            # generations without spending parser time on them.
            self._pending = {
                key: value
                for key, value in self._pending.items()
                if key[0] == self._generation_id
            }
        if previous is not None and self._state != RichGraphState.NOT_STARTED:
            self._state = RichGraphState.LAGGED
        self._persist_state()

    def start_after_first_action(
        self,
        frontier: MilestoneFrontier | None = None,
        *,
        revision_id: str | None = None,
        current_milestone_id: str = "current",
        **frontier_fields: tuple[str, ...],
    ) -> RichGraphCapabilityReceipt:
        """Start only when the production runner reports its first Agent action."""

        with self._lock:
            self._ensure_open()
            if frontier is None:
                selected_revision = revision_id or self._revision_id
                if selected_revision == "unknown":
                    raise ValueError("revision_id or frontier is required on first start")
                frontier = MilestoneFrontier(
                    workspace_revision_id=selected_revision,
                    current_milestone_id=current_milestone_id,
                    **frontier_fields,
                )
            self._observe_frontier_language(frontier.language)
            self._activate_revision(frontier.workspace_revision_id)
            enqueued = self._enqueue_frontier_locked(frontier)
            if self._thread is None:
                self._stop = False
                self._thread = threading.Thread(
                    target=self._worker,
                    name=f"rich-graph-{self.repository_id[:12]}",
                    daemon=True,
                )
                self._state = RichGraphState.BUILDING if enqueued else RichGraphState.READY_CURRENT
                self._persist_state()
                self._thread.start()
            elif enqueued and self._state in {
                RichGraphState.READY_CURRENT,
                RichGraphState.READY_PARTIAL,
                RichGraphState.LAGGED,
                RichGraphState.FAILED,
            }:
                self._state = RichGraphState.BUILDING
                self._failure_reason = None
                self._persist_state()
            return self._capability_locked()

    def submit_frontier(
        self,
        frontier: MilestoneFrontier | None = None,
        *,
        revision_id: str | None = None,
        current_milestone_id: str = "current",
        **frontier_fields: tuple[str, ...],
    ) -> int:
        """Merge bounded signals into the queue, promoting duplicate priorities."""

        with self._lock:
            self._ensure_open()
            if frontier is None:
                selected_revision = revision_id or self._revision_id
                if selected_revision == "unknown":
                    raise ValueError("revision_id or frontier is required")
                frontier = MilestoneFrontier(
                    workspace_revision_id=selected_revision,
                    current_milestone_id=current_milestone_id,
                    **frontier_fields,
                )
            self._observe_frontier_language(frontier.language)
            self._activate_revision(frontier.workspace_revision_id)
            count = self._enqueue_frontier_locked(frontier)
            if self._thread is not None and count:
                self._state = RichGraphState.BUILDING
                self._failure_reason = None
                self._persist_state()
            return count

    def _signal_path(self, value: str, *, symbol: bool = False) -> str | None:
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.startswith("file:"):
            candidate = candidate[5:]
        if candidate.startswith("symbol:"):
            candidate = candidate[7:]
            symbol = True
        candidate = candidate.split("::", 1)[0]
        if symbol:
            parts = candidate.split(":")
            if len(parts) > 1:
                candidate = parts[0]
        candidate_path = Path(candidate)
        if candidate_path.is_absolute():
            resolved = candidate_path.expanduser().resolve()
        else:
            resolved = (self.repository_path / candidate_path).resolve()
        try:
            relative = resolved.relative_to(self.repository_path).as_posix()
        except ValueError:
            return None
        if self._bounded_multilang_navigation and any(
            part.casefold()
            in {
                ".git",
                "node_modules",
                "target",
                "dist",
                "build",
                "coverage",
                "__pycache__",
            }
            for part in Path(relative).parts
        ):
            return None
        if not resolved.is_file():
            return None
        return relative

    def _observe_frontier_language(self, language: str | None) -> None:
        normalized = str(language or "").strip().casefold()
        if normalized and normalized not in {"python", "py", "python3"}:
            self._bounded_multilang_navigation = True

    def _enqueue_frontier_locked(self, frontier: MilestoneFrontier) -> int:
        by_path: dict[str, tuple[int, set[str]]] = {}
        for field_name, priority, reason in _PRIORITIES:
            values = getattr(frontier, field_name)
            if field_name == "prefetch_files":
                values = values[: frontier.prefetch_budget]
            for value in values:
                path = self._signal_path(value, symbol=field_name == "recent_symbols")
                if path is None:
                    continue
                current = by_path.get(path)
                if current is None:
                    by_path[path] = (priority, {reason})
                else:
                    by_path[path] = (min(priority, current[0]), current[1] | {reason})
        for signature in frontier.failure_signatures:
            path = self._signal_path(signature.split(":", 1)[0])
            if path is None:
                continue
            current = by_path.get(path)
            if current is None:
                by_path[path] = (25, {"failure_signature"})
            else:
                by_path[path] = (min(25, current[0]), current[1] | {"failure_signature"})
        enqueued = 0
        assert self._generation_id is not None
        for path, (priority, reasons) in sorted(by_path.items()):
            if self._enqueue_path_locked(
                path,
                priority,
                tuple(sorted(reasons)),
                frontier.failure_signatures,
                language=frontier.language,
            ):
                enqueued += 1
        return enqueued

    # Vendored, generated, built and snapshot paths never carry an editable
    # surface the model needs, yet in earlier element-web/navidrome runs
    # they dominated the single-worker queue (hours of ``rich_graph_queue_ms``)
    # through import-neighbour fan-out.  They are excluded from background
    # projection; an explicit frontier signal (a file the model touched) is
    # still honoured.
    _EXCLUDED_SEGMENTS = frozenset(
        {
            "node_modules",
            "__snapshots__",
            "dist",
            "build",
            "target",
            "vendor",
            ".git",
            "coverage",
            ".next",
            ".cache",
        }
    )
    _EXCLUDED_SUFFIXES = (
        ".min.js",
        ".d.ts",
        ".snap",
        ".pb.go",
        "_pb.go",
        ".pb.gw.go",
        ".generated.ts",
        ".lock",
    )

    @classmethod
    def _background_path_admitted(cls, path: str, reasons: tuple[str, ...]) -> bool:
        if reasons == ("configured_generated_neighbour",):
            return True
        if reasons != ("import_neighbour",):
            return True
        parts = path.split("/")
        if any(part in cls._EXCLUDED_SEGMENTS for part in parts[:-1]):
            return False
        return not path.endswith(cls._EXCLUDED_SUFFIXES)

    def _neighbour_budget_exhausted_locked(self) -> bool:
        limit = self._neighbour_budget
        if limit is None:
            return False
        return self._neighbour_enqueued.get(self._generation_id or "", 0) >= limit

    def _enqueue_path_locked(
        self,
        path: str,
        priority: int,
        reasons: tuple[str, ...],
        failure_signatures: tuple[str, ...] = (),
        *,
        language: str | None = None,
    ) -> bool:
        assert self._generation_id is not None
        key = (self._generation_id, path)
        if self._store.has_projection(self._generation_id, path):
            return False
        if reasons in {("import_neighbour",), ("configured_generated_neighbour",)}:
            if not self._background_path_admitted(path, reasons):
                return False
            if self._neighbour_budget_exhausted_locked():
                return False
            self._neighbour_enqueued[self._generation_id] = (
                self._neighbour_enqueued.get(self._generation_id, 0) + 1
            )
        current = self._pending.get(key)
        if current is not None and current[0] <= priority:
            return False
        if (
            self._bounded_multilang_navigation
            and len(self._pending) >= self._multilang_pending_limit
        ):
            # Preserve the highest-value frontier paths. The stale queue item
            # left behind is ignored by its sequence/pending check.
            worst_key, worst = max(
                self._pending.items(),
                key=lambda item: (item[1][0], item[1][1]),
            )
            if worst[0] <= priority:
                return False
            self._pending.pop(worst_key, None)
        self._sequence += 1
        task = FrontierTask(
            generation_id=self._generation_id,
            repository_id=self.repository_id,
            workspace_revision_id=self._revision_id,
            repository_relative_path=path,
            absolute_path=(self.repository_path / path).resolve(),
            priority=priority,
            reasons=reasons,
            failure_signatures=failure_signatures,
            enqueued_ns=time.perf_counter_ns(),
            language=language,
        )
        self._pending[key] = (priority, self._sequence, task)
        self._queue.put(_QueueItem(priority, self._sequence, task))
        self._condition.notify_all()
        return True

    def _invoke_processor(self, task: FrontierTask) -> FileProjection:
        if self._processor is None:
            return process_frontier_file(task)
        target = self._processor
        if not callable(target):
            target = getattr(target, "process")
        result = target(task)  # type: ignore[operator]
        if result is not None and not isinstance(result, FileProjection):
            raise TypeError("Rich processor must return FileProjection or None")
        return result if result is not None else process_frontier_file(task)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            task = item.task
            try:
                if task is None:
                    self._close_store_once()
                    return
                key = (task.generation_id, task.repository_relative_path)
                with self._lock:
                    current = self._pending.get(key)
                    if current is None or current[1] != item.sequence:
                        continue
                    self._pending.pop(key, None)
                    if self._stop or (
                        self._bounded_multilang_navigation
                        and task.generation_id != self._generation_id
                    ):
                        continue
                    self._inflight = task
                    if task.generation_id == self._generation_id:
                        self._state = RichGraphState.BUILDING
                        self._persist_state()
                self.metrics.observe_ms(
                    "rich_graph_queue_ms",
                    (time.perf_counter_ns() - task.enqueued_ns) / 1_000_000,
                )
                with self.metrics.timer("rich_graph_projection_ms"):
                    projection = self._invoke_processor(task)
                    self._store.commit_projection(task.generation_id, projection)
                with self._lock:
                    if task.generation_id == self._generation_id:
                        self._search_cache.clear()
                        if not {
                            "import_neighbour",
                            "configured_generated_neighbour",
                        }.intersection(task.reasons):
                            for dependency in dependency_paths(
                                self.repository_path, task.repository_relative_path,
                                projection.metadata.get("navigation_index", {}),
                            ):
                                reason = ("import_neighbour",)
                                if "/target/" in f"/{dependency}":
                                    from memtrace.swe_milestone.maven_reactor import (
                                        is_configured_generated_path,
                                    )

                                    if is_configured_generated_path(
                                        self.repository_path, dependency
                                    ):
                                        reason = ("configured_generated_neighbour",)
                                self._enqueue_path_locked(dependency, 55, reason)
                        self._supported_relations.update(
                            self._store.supported_relations(task.generation_id)
                        )
                        self._failure_reason = None
                        outstanding = self._outstanding_locked()
                        self._state = (
                            RichGraphState.READY_PARTIAL
                            if outstanding > 1
                            else RichGraphState.READY_CURRENT
                        )
                        self._persist_state()
            except BaseException as exc:
                with self._lock:
                    if task is not None and task.generation_id == self._generation_id:
                        self._failure_reason = f"{type(exc).__name__}: {exc}"
                        self._state = RichGraphState.FAILED
                        self._persist_state()
            finally:
                with self._lock:
                    if task is not None and self._inflight is task:
                        self._inflight = None
                    self._condition.notify_all()
                self._queue.task_done()

    def _outstanding_locked(self) -> int:
        current_pending = sum(
            1 for generation, _ in self._pending if generation == self._generation_id
        )
        current_inflight = int(
            self._inflight is not None and self._inflight.generation_id == self._generation_id
        )
        return current_pending + current_inflight

    def _frontier_locked(self) -> tuple[str, ...]:
        tasks = [
            value
            for (generation, _), value in self._pending.items()
            if generation == self._generation_id
        ]
        tasks.sort(key=lambda value: (value[0], value[1]))
        paths = [item[2].repository_relative_path for item in tasks]
        if self._inflight is not None and self._inflight.generation_id == self._generation_id:
            paths.insert(0, self._inflight.repository_relative_path)
        return tuple(dict.fromkeys(paths))

    def _persist_state(self) -> None:
        if self._generation_id is not None:
            self._store.set_generation_state(
                self._generation_id,
                self._state,
                self._supported_relations,
                self._failure_reason,
            )

    def _capability_locked(self) -> RichGraphCapabilityReceipt:
        lag = self._outstanding_locked()
        if self._state == RichGraphState.LAGGED:
            lag = max(1, lag)
        self.metrics.set("graph_freshness_lag", lag)
        return RichGraphCapabilityReceipt(
            state=self._state,
            repository_id=self.repository_id,
            workspace_revision_id=self._revision_id,
            supported_relations=tuple(sorted(self._supported_relations)),
            frontier=self._frontier_locked(),
            freshness_lag=lag,
            failure_reason=self._failure_reason,
            cache_hit=self._last_cache_hit,
        )

    def capability(self) -> RichGraphCapabilityReceipt:
        with self._lock:
            self._ensure_open()
            return self._capability_locked()

    def request_hint(
        self,
        entity_id: str,
        relation: str,
        revision_id: str | None = None,
        *,
        file_path: str | None = None,
    ) -> RichGraphHintReceipt:
        """Return cached/current structure immediately; enqueue a miss and never wait.

        ``rich_graph_blocking_ms`` measures this method's real synchronous
        occupancy (capability/cache/index work). It never includes waiting for
        the background processor.
        """

        request_started_ns = time.perf_counter_ns()
        relation = relation.strip().upper()
        with self._lock:
            self._ensure_open()
            selected_revision = revision_id or self._revision_id
            eligible = is_structural_relation(relation)
            if not eligible or self._generation_id is None or selected_revision == "unknown":
                self._last_cache_hit = False
                receipt = RichGraphHintReceipt(
                    repository_id=self.repository_id,
                    workspace_revision_id=selected_revision,
                    entity_id=entity_id,
                    relation=relation,
                    eligible=eligible,
                    hints=(),
                    queued=False,
                    cache_hit=False,
                    capability=self._capability_locked(),
                )
                self._observe_hint_occupancy(request_started_ns)
                return receipt
            # A request against a newer revision creates a generation but does not
            # start or wait for a worker. Production still controls start timing.
            self._activate_revision(selected_revision)
            generation_id = self._generation_id

        cached = self._store.get_cached_hints(
            self.repository_id, selected_revision, entity_id, relation
        )
        if cached is None:
            hints = self._store.query_hints(
                generation_id,
                self.repository_id,
                selected_revision,
                entity_id,
                relation,
            )
            cache_hit = False
        else:
            hints = cached
            cache_hit = True

        queued = False
        with self._lock:
            self._last_cache_hit = cache_hit
            if not hints and self._thread is not None:
                selected_path = file_path or self._store.lookup_path(self.repository_id, entity_id)
                if selected_path is None:
                    selected_path = self._signal_path(
                        entity_id, symbol=entity_id.startswith("symbol:")
                    )
                else:
                    selected_path = self._signal_path(selected_path)
                if selected_path is not None:
                    queued = self._enqueue_path_locked(selected_path, 0, (f"hint:{relation}",))
                    if queued:
                        self._state = RichGraphState.BUILDING
                        self._failure_reason = None
                        self._persist_state()
            capability = self._capability_locked()
        receipt = RichGraphHintReceipt(
            repository_id=self.repository_id,
            workspace_revision_id=selected_revision,
            entity_id=entity_id,
            relation=relation,
            eligible=True,
            hints=hints,
            queued=queued,
            cache_hit=cache_hit,
            capability=capability,
        )
        self._observe_hint_occupancy(request_started_ns)
        return receipt

    def resolve_addresses(
        self,
        tokens: tuple[str, ...],
        seeds: tuple[str, ...],
        *,
        revision_id: str | None = None,
        limit: int = 48,
    ) -> RichAddressResolution:
        """Suggest structural addresses for a requirement without waiting.

        Two hint sources are consulted synchronously against already-committed
        projections: reference names that mention a requirement token, and the
        structural neighbourhood (``CALLED_BY``/``CALLS``/``IMPORTS``/``DEFINES``)
        of seed entities the model already touched.  A miss enqueues the seed
        file for projection and returns what is known now; nothing here blocks
        the route.
        """

        request_started_ns = time.perf_counter_ns()
        with self._lock:
            self._ensure_open()
            selected_revision = revision_id or self._revision_id
            if selected_revision == "unknown" or self._generation_id is None:
                self._observe_hint_occupancy(request_started_ns)
                return RichAddressResolution(
                    candidates=(),
                    by_name=0,
                    by_structure=0,
                    seeds_queued=0,
                    capability=self._capability_locked(),
                )
            self._activate_revision(selected_revision)
            generation_id = self._generation_id
        by_name = self._store.search_references(self.repository_id, tokens, limit=limit)
        structural: dict[str, None] = {}
        seeds_queued = 0
        for seed in tuple(dict.fromkeys(seeds))[:8]:
            for relation in ("CALLED_BY", "CALLS", "IMPORTS", "DEFINES"):
                cached = self._store.get_cached_hints(
                    self.repository_id, selected_revision, seed, relation
                )
                hints = (
                    cached
                    if cached is not None
                    else self._store.query_hints(
                        generation_id,
                        self.repository_id,
                        selected_revision,
                        seed,
                        relation,
                    )
                )
                for hint in hints:
                    for reference_id in (hint.target_reference_id, hint.source_reference_id):
                        entity = self._store.reference_entity(self.repository_id, reference_id)
                        if entity is not None and entity != seed:
                            structural.setdefault(entity, None)
            if not any(
                self._store.get_cached_hints(self.repository_id, selected_revision, seed, rel)
                for rel in ("CALLED_BY", "DEFINES")
            ):
                with self._lock:
                    if self._thread is not None:
                        path = self._store.lookup_path(self.repository_id, seed)
                        selected_path = (
                            self._signal_path(path)
                            if path is not None
                            else self._signal_path(seed, symbol=seed.startswith("symbol:"))
                        )
                        if selected_path is not None and self._enqueue_path_locked(
                            selected_path, 0, ("hint:ADDRESS_RESOLUTION",)
                        ):
                            seeds_queued += 1
                            self._state = RichGraphState.BUILDING
                            self._failure_reason = None
                            self._persist_state()
        with self._lock:
            capability = self._capability_locked()
        self._observe_hint_occupancy(request_started_ns)
        return RichAddressResolution(
            candidates=tuple(dict.fromkeys((*by_name, *structural)))[:limit],
            by_name=len(by_name),
            by_structure=len(structural),
            seeds_queued=seeds_queued,
            capability=capability,
        )

    def search_code(
        self,
        query: str,
        *,
        scope_entities: tuple[str, ...] = (),
        relation_kinds: tuple[str, ...] = (),
        revision_id: str | None = None,
        limit: int = 8,
    ) -> dict[str, object]:
        """Return a bounded, model-safe structural search result.

        Search is deliberately a read-only hint over committed projections.
        It never waits for the worker, creates Evidence, or changes TPG state.
        An unprojected file is reported as BUILDING/NO_MATCH and may be queued
        for the existing background worker.
        """

        started_ns = time.perf_counter_ns()
        text = str(query or "").strip()
        bounded_limit = max(1, min(int(limit), 8))
        tokens = tuple(
            dict.fromkeys(
                token.casefold()
                for token in re.findall(r"[A-Za-z0-9_./:-]+", text)
                if len(token) >= 2
            )
        )
        relation_map = {
            "CALLS": "callees",
            "CALLED_BY": "callers",
            "COVERED_BY": "covered_by",
            "IMPORTS": "imports",
            "IMPORTED_BY": "imported_by",
            "DEFINES": "defines",
        }
        relations = tuple(
            dict.fromkeys(
                relation.strip().upper()
                for relation in relation_kinds
                if relation.strip().upper() in relation_map
            )
        ) or ("CALLS", "CALLED_BY", "COVERED_BY")
        normalized_scope = tuple(
            dict.fromkeys(str(item).strip() for item in scope_entities if str(item).strip())
        )
        if not text:
            return {
                "schema": "homy/rich-code-search@1",
                "status": "NO_MATCH",
                "revision_id": revision_id or self._revision_id,
                "matches": [],
                "truncated": False,
                "query_tokens": [],
            }

        with self._lock:
            self._ensure_open()
            selected_revision = revision_id or self._revision_id
            cache_key = (
                selected_revision,
                text.casefold(),
                normalized_scope,
                relations,
                bounded_limit,
            )
            cached = self._search_cache.get(cache_key)
            if cached is not None:
                self.metrics.observe_ms(
                    "rich_code_search_ms",
                    (time.perf_counter_ns() - started_ns) / 1_000_000,
                )
                return copy.deepcopy(cached)
            if selected_revision == "unknown" or self._generation_id is None:
                capability = self._capability_locked()
                self._observe_hint_occupancy(started_ns)
                return {
                    "schema": "homy/rich-code-search@1",
                    "status": "UNAVAILABLE",
                    "revision_id": selected_revision,
                    "graph_state": capability.state.value,
                    "matches": [],
                    "truncated": False,
                    "query_tokens": list(tokens),
                }
            self._activate_revision(selected_revision)
            generation_id = self._generation_id
            capability = self._capability_locked()

        rows = self._store.search_reference_rows(
            self.repository_id,
            tokens,
            generation_id=generation_id,
            limit=bounded_limit + 1,
        )
        scope_paths: set[str] = set()
        scope_symbols: set[str] = set()
        for raw in scope_entities:
            value = str(raw).strip()
            if value.startswith("file:"):
                scope_paths.add(value.removeprefix("file:").lstrip("./"))
            elif value.startswith("symbol:"):
                scope_symbols.add(value)
                scope_paths.add(value.removeprefix("symbol:").rsplit(":", 1)[0])
            elif "/" in value or value.endswith(".py"):
                scope_paths.add(value.lstrip("./"))
        if scope_paths or scope_symbols:
            rows = tuple(
                row
                for row in rows
                if (
                    str(row["repository_relative_path"]) in scope_paths
                    or str(row["canonical_entity_id"]) in scope_symbols
                )
            )

        query_fold = text.casefold()
        def rank(row: dict[str, object]) -> tuple[object, ...]:
            name = str(row.get("qualified_name") or "").casefold()
            path = str(row.get("repository_relative_path") or "").casefold()
            canonical = str(row.get("canonical_entity_id") or "")
            exact = 0 if query_fold in {name, path, canonical.casefold()} else 1
            token_hits = sum(int(token in name or token in path) for token in tokens)
            kind_rank = {
                "SymbolReference": 0,
                "TestReference": 1,
                "FileReference": 2,
            }.get(str(row.get("reference_kind")), 3)
            return (exact, -token_hits, kind_rank, len(canonical), canonical)

        ranked = sorted(rows, key=rank)
        truncated = len(ranked) > bounded_limit
        matches: list[dict[str, object]] = []
        for row in ranked[:bounded_limit]:
            canonical = str(row["canonical_entity_id"])
            reference_id = str(row["reference_id"])
            payload = self._store.version_payload(generation_id, reference_id)
            path = str(row["repository_relative_path"])
            detected_language = language_for_path(path)
            item: dict[str, object] = {
                "address": canonical,
                "file": path,
                "qualified_name": (
                    str(row["qualified_name"])
                    if row.get("qualified_name") is not None
                    else payload.get("qualified_name")
                ),
                "symbol_kind": payload.get("symbol_kind"),
                "signature": payload.get("signature"),
                "line_start": payload.get("line"),
                "line_end": payload.get("end_line"),
                "language": payload.get("language") or detected_language,
                "parser_backend": payload.get("parser_backend")
                or ("python-ast" if detected_language == "python" else None),
                "parser_confidence": (
                    payload.get("parser_confidence")
                    if payload.get("parser_confidence") is not None
                    else (1.0 if detected_language == "python" else 0.0)
                ),
                "callers": [],
                "callees": [],
                "covered_by": [],
                "imports": [],
                "imported_by": [],
                "defines": [],
                "memory_refs": [],
            }
            preview = str(payload.get("code") or payload.get("code_surface_preview") or "")
            if preview:
                item["code_surface_preview"] = preview[:1200]
            for relation in relations:
                hints = self._store.get_cached_hints(
                    self.repository_id, selected_revision, canonical, relation
                )
                if hints is None:
                    hints = self._store.query_hints(
                        generation_id,
                        self.repository_id,
                        selected_revision,
                        canonical,
                        relation,
                    )
                values: list[str] = []
                for hint in hints:
                    target = self._store.reference_entity(
                        self.repository_id, hint.target_reference_id
                    )
                    if target is not None and target != canonical and target not in values:
                        values.append(target)
                    if len(values) >= 6:
                        break
                if values:
                    item[relation_map[relation]] = values
            matches.append(item)

        queued = 0
        if not matches and scope_paths:
            with self._lock:
                if self._thread is not None:
                    for path in sorted(scope_paths)[:4]:
                        selected_path = self._signal_path(path)
                        if selected_path is not None and self._enqueue_path_locked(
                            selected_path, 0, ("hint:CODE_SEARCH",)
                        ):
                            queued += 1
                    if queued:
                        self._state = RichGraphState.BUILDING
                        self._failure_reason = None
                        self._persist_state()
                capability = self._capability_locked()
        status = (
            "OK"
            if matches
            else "BUILDING"
            if queued or capability.state in {
                RichGraphState.BUILDING,
                RichGraphState.READY_PARTIAL,
                RichGraphState.LAGGED,
            }
            else "UNAVAILABLE"
            if capability.state in {RichGraphState.FAILED, RichGraphState.NOT_STARTED}
            else "NO_MATCH"
        )
        self._observe_hint_occupancy(started_ns)
        self.metrics.observe_ms(
            "rich_code_search_ms",
            (time.perf_counter_ns() - started_ns) / 1_000_000,
        )
        result = {
            "schema": "homy/rich-code-search@1",
            "status": status,
            "revision_id": selected_revision,
            "graph_state": capability.state.value,
            "matches": matches,
            "truncated": truncated,
            "query_tokens": list(tokens),
            "queued_files": queued,
        }
        if queued == 0 and status in {"OK", "NO_MATCH"}:
            with self._lock:
                if selected_revision == self._revision_id:
                    if len(self._search_cache) >= 128:
                        self._search_cache.pop(next(iter(self._search_cache)))
                    self._search_cache[cache_key] = copy.deepcopy(result)
        return result


    def related_observers(
        self,
        entity_ids: tuple[str, ...],
        *,
        revision_id: str | None = None,
        limit: int = 64,
    ) -> tuple[str, ...]:
        """Return tests/files known to exercise the given implementation entities.

        One structural hop over ``COVERED_BY``, ``CALLED_BY`` and ``IMPORTED_BY``
        answered from committed projections only.  Used to decide whether an
        observed test result is *about* a requirement's addresses; an empty
        answer means "unknown", never "unrelated".
        """

        request_started_ns = time.perf_counter_ns()
        with self._lock:
            self._ensure_open()
            selected_revision = revision_id or self._revision_id
            if selected_revision == "unknown" or self._generation_id is None:
                self._observe_hint_occupancy(request_started_ns)
                return ()
            self._activate_revision(selected_revision)
            generation_id = self._generation_id
        related: dict[str, None] = {}
        for entity in tuple(dict.fromkeys(entity_ids))[:12]:
            aliases = [entity]
            if entity.startswith("symbol:"):
                # The file that defines the symbol is also a valid observer
                # anchor: a test importing the module exercises it.
                path = entity.removeprefix("symbol:").rsplit(":", 1)[0]
                aliases.append(f"file:{path}")
            for alias in aliases:
                for relation in ("COVERED_BY", "CALLED_BY", "IMPORTED_BY"):
                    cached = self._store.get_cached_hints(
                        self.repository_id, selected_revision, alias, relation
                    )
                    hints = (
                        cached
                        if cached is not None
                        else self._store.query_hints(
                            generation_id,
                            self.repository_id,
                            selected_revision,
                            alias,
                            relation,
                        )
                    )
                    for hint in hints:
                        observer = self._store.reference_entity(
                            self.repository_id, hint.target_reference_id
                        )
                        if observer is not None and observer != alias:
                            related.setdefault(observer, None)
                            path = self._store.lookup_path(self.repository_id, observer)
                            if path is not None:
                                related.setdefault(f"file:{path}", None)
        self._observe_hint_occupancy(request_started_ns)
        return tuple(related)[:limit]

    def symbols_in_paths(
        self,
        paths: tuple[str, ...],
        *,
        revision_id: str | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Symbols structurally declared in ``paths`` (Page finalization hint).

        Answered from the reference table only, so it is a bounded read with no
        parsing; a file the background worker has not projected yet yields no
        symbols, which the caller records as "unknown".
        """

        request_started_ns = time.perf_counter_ns()
        with self._lock:
            self._ensure_open()
            if self._generation_id is None:
                self._observe_hint_occupancy(request_started_ns)
                return {}
            if revision_id is not None and revision_id != "unknown":
                self._activate_revision(revision_id)
            generation_id = self._generation_id
        symbols = self._store.symbols_in_paths(self.repository_id, paths, generation_id=generation_id)
        self._observe_hint_occupancy(request_started_ns)
        return symbols

    def code_map(
        self,
        scope_entities: tuple[str, ...],
        *,
        revision_id: str | None = None,
        max_files: int = 16,
        max_symbols_per_file: int = 12,
        max_neighbours: int = 6,
    ) -> dict[str, object]:
        """Compress the structural neighbourhood of a Milestone scope.

        The map is a *hint* card: for each in-scope file the symbols it
        defines, and for each in-scope symbol its callers and the tests that
        cover it, all answered from committed projections without parsing or
        waiting.  Unprojected scope stays listed with ``projected=False`` so the
        model knows the runtime has no structure for it yet; the worker is
        nudged to project those files in the background.  Never evidence.
        """

        request_started_ns = time.perf_counter_ns()
        with self._lock:
            self._ensure_open()
            selected_revision = revision_id or self._revision_id
            if selected_revision == "unknown" or self._generation_id is None:
                self._observe_hint_occupancy(request_started_ns)
                return {
                    "revision_id": selected_revision,
                    "capability": self._capability_locked().state.value,
                    "files": [],
                    "projected_files": 0,
                    "unprojected_files": 0,
                }
            self._activate_revision(selected_revision)
            generation_id = self._generation_id
            capability = self._capability_locked()
        files: dict[str, dict[str, object]] = {}
        symbol_seeds: dict[str, None] = {}
        for entity in tuple(dict.fromkeys(scope_entities)):
            if entity.startswith("file:"):
                path = entity.removeprefix("file:")
            elif entity.startswith("symbol:"):
                path = entity.removeprefix("symbol:").rsplit(":", 1)[0]
                symbol_seeds.setdefault(entity, None)
            else:
                continue
            if path and path not in files and len(files) < max_files:
                files[path] = {"path": path, "symbols": [], "projected": False}
        if not files:
            self._observe_hint_occupancy(request_started_ns)
            return {
                "revision_id": selected_revision,
                "capability": capability.state.value,
                "files": [],
                "projected_files": 0,
                "unprojected_files": 0,
            }
        declared = self._store.symbols_in_paths(self.repository_id, tuple(files), generation_id=generation_id)
        for path, card in files.items():
            symbols = tuple(declared.get(path, ()))
            card["projected"] = bool(symbols)
            enriched_symbols: list[dict[str, object]] = []
            for symbol in symbols[:max_symbols_per_file]:
                symbol_seeds.setdefault(symbol, None)
                payload = self._store.version_payload_for_entity(
                    generation_id,
                    self.repository_id,
                    symbol,
                )
                language = payload.get("language") or language_for_path(path)
                entry: dict[str, object] = {
                    "address": symbol,
                    "entity": symbol,
                    "file": path,
                    "name": symbol.rsplit(":", 1)[-1],
                    "qualified_name": payload.get("qualified_name")
                    or symbol.rsplit(":", 1)[-1],
                    "symbol_kind": payload.get("symbol_kind"),
                    "signature": payload.get("signature"),
                    "line_start": payload.get("line"),
                    "line_end": payload.get("end_line"),
                    "language": language,
                    "parser_backend": payload.get("parser_backend"),
                    "parser_confidence": payload.get("parser_confidence"),
                }
                code = str(payload.get("code") or "")
                if code:
                    entry["code_surface_preview"] = code[:1200]
                enriched_symbols.append(entry)
            card["symbols"] = enriched_symbols
        unprojected = [path for path, card in files.items() if not card["projected"]]
        for path in unprojected[:4]:
            with self._lock:
                if self._thread is not None:
                    selected_path = self._signal_path(path)
                    if selected_path is not None and self._enqueue_path_locked(
                        selected_path, 0, ("hint:CODE_MAP",)
                    ):
                        self._state = RichGraphState.BUILDING
                        self._failure_reason = None
                        self._persist_state()
        neighbourhood: dict[str, dict[str, list[str]]] = {}
        for symbol in tuple(symbol_seeds)[: max_files * max_symbols_per_file]:
            relations: dict[str, list[str]] = {}
            for relation, key in (("CALLED_BY", "callers"), ("COVERED_BY", "covered_by")):
                cached = self._store.get_cached_hints(
                    self.repository_id, selected_revision, symbol, relation
                )
                hints = (
                    cached
                    if cached is not None
                    else self._store.query_hints(
                        generation_id,
                        self.repository_id,
                        selected_revision,
                        symbol,
                        relation,
                    )
                )
                targets: dict[str, None] = {}
                for hint in hints:
                    target = self._store.reference_entity(
                        self.repository_id, hint.target_reference_id
                    )
                    if target is not None and target != symbol:
                        targets.setdefault(target, None)
                    if len(targets) >= max_neighbours:
                        break
                if targets:
                    relations[key] = list(targets)
            if relations:
                neighbourhood[symbol] = relations
        for card in files.values():
            for entry in card["symbols"]:
                relations = neighbourhood.get(str(entry["entity"]))
                if relations:
                    entry.update(relations)
        self._observe_hint_occupancy(request_started_ns)
        return {
            "revision_id": selected_revision,
            "capability": capability.state.value,
            "files": list(files.values()),
            "projected_files": len(files) - len(unprojected),
            "unprojected_files": len(unprojected),
        }

    def _observe_hint_occupancy(self, started_ns: int) -> None:
        self.metrics.observe_ms(
            "rich_graph_blocking_ms",
            max(0, time.perf_counter_ns() - started_ns) / 1_000_000,
        )

    def wait_until_idle(self, timeout: float = 5.0) -> bool:
        """Operational/test drain primitive; production recall never calls this."""

        deadline = time.monotonic() + timeout
        with self._condition:
            while self._outstanding_locked():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def projected_paths(self) -> tuple[str, ...]:
        with self._lock:
            generation_id = self._generation_id
        return self._store.projected_paths(generation_id) if generation_id else ()

    def close(self, timeout: float = 2.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop = True
            thread = self._thread
            if thread is not None:
                self._sequence += 1
                self._queue.put(_QueueItem(10**9, self._sequence, None))
        if thread is not None:
            thread.join(timeout=max(0.0, timeout))
        if thread is None or not thread.is_alive():
            self._close_store_once()

    def _close_store_once(self) -> None:
        with self._lock:
            if self._store_closed:
                return
            self._store_closed = True
        self._store.close()

    def __enter__(self) -> "RichGraphScheduler":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
