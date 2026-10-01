from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..contracts import (
    Event,
    EventGroup,
    EvidenceDraft,
    EvidenceKey,
    FactType,
    PageKind,
    PageManifest,
    PageState,
    canonical_bytes,
    digest,
    primitive,
    stable_id,
)
from ..database import StateDatabase
from ..durability import (
    SecretRedactor,
    SyncHook,
    append_synced,
    atomic_write_once,
    json_line,
    load_json_lines,
)
from ..observability.metrics import MetricRecorder
from .policy import LEGAL_TAIL_REASONS, PagePolicy, TailReason

Projector = Callable[
    [sqlite3.Connection, PageManifest, tuple[EventGroup, ...], Mapping[str, tuple[int, int]]],
    None,
]


class PageStoreError(RuntimeError):
    pass


class PageIntegrityError(PageStoreError):
    pass


class PageBoundaryError(PageStoreError):
    pass


class BranchVisibilityError(PageStoreError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    scanned_groups: int
    recovered_groups: int
    sealed_pages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BlobReference:
    handle: str
    content_digest: str
    byte_range: tuple[int, int]
    byte_count: int
    relative_path: str


@dataclass(frozen=True, slots=True)
class _PreparedGroup:
    group: EventGroup
    group_digest: str
    redaction_proof: str
    blobs: tuple[BlobReference, ...]


@dataclass(frozen=True, slots=True)
class _PageSetDirectoryEntry:
    """Bounded trace-routing metadata for one physical segment."""

    segment_index: int
    title: str
    summary: str
    semantic_kinds: tuple[str, ...]
    entity_refs: tuple[str, ...]
    event_ids: tuple[str, ...]
    logical_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PreparedPageSet:
    """One logical EventGroup represented by multiple bounded physical groups."""

    page_set_id: str
    logical_group_id: str
    logical_group_digest: str
    logical_group_blob: BlobReference
    segments: tuple[_PreparedGroup, ...]
    blobs: tuple[BlobReference, ...]
    synopsis: str
    semantic_kinds: tuple[str, ...]
    entity_refs: tuple[str, ...]
    directory: tuple[_PageSetDirectoryEntry, ...]

    @property
    def directory_digest(self) -> str:
        return digest(primitive(self.directory))


_SCHEMA_VERSION = 1
_WAL_SCHEMA_VERSION = 1
_PAGE_SCHEMA_VERSION = 1


def _safe_component(value: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")[:32] or "scope"
    suffix = digest({"scope": value}).removeprefix("sha256:")[:16]
    return f"{prefix}-{suffix}"


def _page_manifest_from_dict(value: Mapping[str, Any]) -> PageManifest:
    return PageManifest(
        page_id=str(value["page_id"]),
        run_id=str(value["run_id"]),
        branch_id=str(value["branch_id"]),
        page_seq=int(value["page_seq"]),
        revision_ids=tuple(map(str, value["revision_ids"])),
        milestone_ids=tuple(map(str, value["milestone_ids"])),
        execution_phases=tuple(map(str, value["execution_phases"])),
        event_range=(int(value["event_range"][0]), int(value["event_range"][1])),
        event_group_ids=tuple(map(str, value["event_group_ids"])),
        evidence_key_digests=tuple(map(str, value["evidence_key_digests"])),
        entity_refs=tuple(map(str, value["entity_refs"])),
        token_count=int(value["token_count"]),
        byte_count=int(value["byte_count"]),
        payload_digest=str(value["payload_digest"]),
        redaction_proof=str(value["redaction_proof"]),
        previous_page_id=(
            str(value["previous_page_id"]) if value.get("previous_page_id") else None
        ),
        seal_reason=str(value["seal_reason"]),
        page_kind=PageKind(str(value["page_kind"])),
        tail=bool(value["tail"]),
    )


class PageStore:
    """Authoritative V2 event ledger and immutable Trace Store.

    Every public append accepts one *complete logical* EventGroup. The group is
    recursively redacted before any persistent write and is represented by
    exactly one synced ledger record. A logical group whose bounded metadata
    does not fit one trace is deterministically represented as an atomic
    Memory Episode of complete physical event groups; ordinary groups retain
    the one-group path.
    """

    def __init__(
        self,
        root: Path,
        database: StateDatabase,
        run_id: str,
        branch_id: str,
        *,
        policy: PagePolicy | None = None,
        redactor: SecretRedactor | None = None,
        metrics: MetricRecorder | None = None,
        projector: Projector | None = None,
        sync_hook: SyncHook | None = None,
        parent_branch_id: str | None = None,
        parent_page_id: str | None = None,
    ) -> None:
        if not run_id or not branch_id:
            raise ValueError("run_id and branch_id are required")
        if (parent_branch_id is None) != (parent_page_id is None):
            raise ValueError("parent_branch_id and parent_page_id must be provided together")
        self.root = Path(root).expanduser().resolve()
        self.database = database
        self.run_id = run_id
        self.branch_id = branch_id
        self.policy = policy or PagePolicy()
        self.redactor = redactor or SecretRedactor()
        self.metrics = metrics or MetricRecorder()
        self.projector = projector
        self.sync_hook = sync_hook
        self._lock = threading.RLock()
        self._closed = False
        self._last_sealed: tuple[PageManifest, ...] = ()

        self.pages_dir = self.root / "pages"
        self.blobs_dir = self.root / "blobs"
        self.wal_dir = self.root / "wal"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.wal_path = self.wal_dir / (
            f"{_safe_component(run_id)}.{_safe_component(branch_id)}.jsonl"
        )
        self._create_schema(parent_branch_id, parent_page_id)

    @property
    def last_sealed(self) -> tuple[PageManifest, ...]:
        """All pages sealed by the most recent append/checkpoint operation."""

        return self._last_sealed

    def has_durable_event(self, event_id: str) -> bool:
        """Return whether an Event is in a synced WAL record and its recovered map.

        The WAL append is fsynced before ``v2_page_wal_groups`` is projected.
        Registry writers use this check as a fail-closed ordering fence.
        """

        row = self.database.connection.execute(
            "SELECT 1 FROM v2_page_wal_events WHERE run_id=? AND branch_id=? AND event_id=?",
            (self.run_id, self.branch_id, event_id),
        ).fetchone()
        return row is not None

    def durable_groups(self) -> tuple[EventGroup, ...]:
        """Return this branch's synced WAL groups in durable cursor order.

        This is the recovery boundary for Planning and other projections: a
        caller may rebuild derived state without asking the Harness to repeat
        an already-observed turn. Physical segments are reassembled to their
        original redacted logical EventGroup at this boundary, so storage
        sizing never leaks into Planning or Registry semantics.
        """

        rows = self.database.connection.execute(
            "SELECT g.group_json,s.page_set_id,s.segment_index,p.logical_group_blob_handle "
            "FROM v2_page_wal_groups g "
            "LEFT JOIN v2_page_set_segments s ON s.group_id=g.group_id "
            "LEFT JOIN v2_page_sets p ON p.page_set_id=s.page_set_id "
            "WHERE g.run_id=? AND g.branch_id=? ORDER BY g.event_start,g.group_id",
            (self.run_id, self.branch_id),
        ).fetchall()
        result: list[EventGroup] = []
        seen_sets: set[str] = set()
        for row in rows:
            page_set_id = str(row["page_set_id"] or "")
            if not page_set_id:
                result.append(EventGroup.from_dict(json.loads(str(row["group_json"]))))
                continue
            if page_set_id in seen_sets:
                continue
            seen_sets.add(page_set_id)
            handle = str(row["logical_group_blob_handle"] or "")
            if not handle:
                raise PageIntegrityError(f"PageSet logical body is missing: {page_set_id}")
            decoded = json.loads(self.open_blob(handle))
            if not isinstance(decoded, dict):
                raise PageIntegrityError(f"PageSet logical body is invalid: {page_set_id}")
            result.append(EventGroup.from_dict(decoded))
        return tuple(result)

    def open_groups(self) -> tuple[EventGroup, ...]:
        """Return only synced groups still forming the next Memory Episode.

        Local Step verification uses this authoritative view so a short Step
        does not have to violate the minimum Page-size policy merely to prove
        its acceptance contract.
        """

        rows = self.database.connection.execute(
            "SELECT group_json FROM v2_page_wal_groups "
            "WHERE run_id=? AND branch_id=? AND page_id IS NULL "
            "ORDER BY event_start,group_id",
            (self.run_id, self.branch_id),
        ).fetchall()
        return tuple(EventGroup.from_dict(json.loads(str(row["group_json"]))) for row in rows)

    def has_open_evidence(self, key_digests: Sequence[str]) -> bool:
        """Return whether an unsealed EventGroup contains one of the requested exact keys."""

        requested = set(key_digests)
        if not requested:
            return False
        rows = self.database.connection.execute(
            "SELECT group_json FROM v2_page_wal_groups "
            "WHERE run_id=? AND branch_id=? AND page_id IS NULL ORDER BY event_start",
            (self.run_id, self.branch_id),
        ).fetchall()
        for row in rows:
            group = EventGroup.from_dict(json.loads(str(row["group_json"])))
            if any(
                fact.key.key_digest in requested for event in group.events for fact in event.facts
            ):
                return True
        return False

    def open_evidence_keys(self) -> tuple[EvidenceKey, ...]:
        """Return exact keys still in the open WAL segment.

        This is a fault-time address probe, not a Page read. It lets the
        semantic resolver seal only an open segment that actually contains the
        requested evidence instead of checkpointing every MemoryNeed.
        """

        result: dict[str, EvidenceKey] = {}
        rows = self.database.connection.execute(
            "SELECT group_json FROM v2_page_wal_groups "
            "WHERE run_id=? AND branch_id=? AND page_id IS NULL ORDER BY event_start",
            (self.run_id, self.branch_id),
        ).fetchall()
        for row in rows:
            group = EventGroup.from_dict(json.loads(str(row["group_json"])))
            for event in group.events:
                for fact in event.facts:
                    result.setdefault(fact.key.key_digest, fact.key)
        return tuple(result.values())

    def resolve_event_payload(self, event: Event) -> Mapping[str, Any]:
        """Return an Event payload, reopening its immutable Blob when externalized."""

        external = event.payload.get("external_payload")
        if not isinstance(external, Mapping):
            return event.payload
        handle = str(external.get("blob_handle", ""))
        byte_range = external.get("byte_range", ())
        if not handle or not isinstance(byte_range, (list, tuple)) or len(byte_range) != 2:
            raise PageIntegrityError("invalid external Event payload reference")
        decoded = json.loads(self.open_blob(handle, (int(byte_range[0]), int(byte_range[1]))))
        if not isinstance(decoded, dict):
            raise PageIntegrityError("external Event payload is not an object")
        return decoded

    def resolve_fact_content(self, fact: EvidenceDraft) -> Mapping[str, Any]:
        """Resolve one ledger fact without sealing its open Memory Episode."""

        external = fact.content.get("external_fact")
        if not isinstance(external, Mapping):
            return fact.content
        handle = str(external.get("blob_handle", ""))
        byte_range = external.get("byte_range", ())
        if not handle or not isinstance(byte_range, (list, tuple)) or len(byte_range) != 2:
            raise PageIntegrityError("invalid external Evidence content reference")
        decoded = json.loads(self.open_blob(handle, (int(byte_range[0]), int(byte_range[1]))))
        if not isinstance(decoded, dict):
            raise PageIntegrityError("external Evidence content is not an object")
        return decoded

    def _create_schema(self, parent_branch_id: str | None, parent_page_id: str | None) -> None:
        with self.database.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS v2_page_store_schema (
                    component TEXT PRIMARY KEY,
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS v2_page_branches (
                    run_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    parent_branch_id TEXT,
                    parent_page_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, branch_id)
                );
                CREATE TABLE IF NOT EXISTS v2_page_wal_groups (
                    run_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    group_digest TEXT NOT NULL,
                    group_json TEXT NOT NULL,
                    redaction_proof TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    milestone_id TEXT,
                    event_start INTEGER NOT NULL,
                    event_end INTEGER NOT NULL,
                    wal_start INTEGER NOT NULL,
                    wal_end INTEGER NOT NULL,
                    page_id TEXT,
                    committed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, branch_id, group_id),
                    CHECK (event_start >= 0 AND event_end > event_start)
                );
                CREATE INDEX IF NOT EXISTS v2_page_wal_cursor_idx
                    ON v2_page_wal_groups(run_id, branch_id, event_start);
                CREATE INDEX IF NOT EXISTS v2_page_wal_page_idx
                    ON v2_page_wal_groups(page_id);
                CREATE TABLE IF NOT EXISTS v2_page_wal_events (
                    run_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    event_position INTEGER NOT NULL,
                    PRIMARY KEY (run_id, branch_id, event_id),
                    UNIQUE (run_id, branch_id, event_position),
                    FOREIGN KEY (run_id, branch_id, group_id)
                        REFERENCES v2_page_wal_groups(run_id, branch_id, group_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS v2_page_wal_events_group_idx
                    ON v2_page_wal_events(run_id, branch_id, group_id);
                CREATE TABLE IF NOT EXISTS v2_pages (
                    page_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    page_seq INTEGER NOT NULL,
                    event_start INTEGER NOT NULL,
                    event_end INTEGER NOT NULL,
                    payload_digest TEXT NOT NULL,
                    storage_digest TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    previous_page_id TEXT,
                    tail INTEGER NOT NULL CHECK (tail IN (0, 1)),
                    seal_reason TEXT NOT NULL,
                    sealed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (run_id, branch_id, page_seq)
                );
                CREATE INDEX IF NOT EXISTS v2_pages_scope_cursor_idx
                    ON v2_pages(run_id, branch_id, event_start, event_end);
                CREATE TABLE IF NOT EXISTS v2_page_blobs (
                    handle TEXT PRIMARY KEY,
                    content_digest TEXT NOT NULL,
                    byte_count INTEGER NOT NULL,
                    storage_path TEXT NOT NULL,
                    redaction_proof TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS v2_page_sets (
                    page_set_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    logical_group_id TEXT NOT NULL,
                    logical_group_digest TEXT NOT NULL,
                    logical_group_blob_handle TEXT NOT NULL REFERENCES v2_page_blobs(handle),
                    segment_count INTEGER NOT NULL CHECK(segment_count > 1),
                    wal_start INTEGER NOT NULL,
                    wal_end INTEGER NOT NULL,
                    committed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(run_id, branch_id, logical_group_id)
                );
                CREATE TABLE IF NOT EXISTS v2_page_set_segments (
                    page_set_id TEXT NOT NULL REFERENCES v2_page_sets(page_set_id),
                    segment_index INTEGER NOT NULL CHECK(segment_index >= 0),
                    group_id TEXT NOT NULL,
                    page_id TEXT REFERENCES v2_pages(page_id),
                    PRIMARY KEY(page_set_id, segment_index),
                    UNIQUE(group_id)
                );
                CREATE INDEX IF NOT EXISTS v2_page_set_segments_page_idx
                    ON v2_page_set_segments(page_id, page_set_id, segment_index);
                CREATE TABLE IF NOT EXISTS v2_page_set_synopses (
                    page_set_id TEXT PRIMARY KEY REFERENCES v2_page_sets(page_set_id),
                    synopsis TEXT NOT NULL,
                    semantic_kinds_json TEXT NOT NULL,
                    entity_refs_json TEXT NOT NULL,
                    directory_digest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS v2_page_set_segment_directory (
                    page_set_id TEXT NOT NULL REFERENCES v2_page_sets(page_set_id),
                    segment_index INTEGER NOT NULL CHECK(segment_index >= 0),
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    semantic_kinds_json TEXT NOT NULL,
                    entity_refs_json TEXT NOT NULL,
                    event_ids_json TEXT NOT NULL,
                    logical_event_ids_json TEXT NOT NULL,
                    PRIMARY KEY(page_set_id, segment_index),
                    FOREIGN KEY(page_set_id,segment_index)
                        REFERENCES v2_page_set_segments(page_set_id,segment_index)
                );
                """
            )
            # One-time migration for older stores. Runtime durability fences
            # are O(1) after this deterministic index is populated.
            for row in conn.execute(
                "SELECT run_id,branch_id,group_id,group_json,event_start "
                "FROM v2_page_wal_groups g WHERE NOT EXISTS ("
                "SELECT 1 FROM v2_page_wal_events e "
                "WHERE e.run_id=g.run_id AND e.branch_id=g.branch_id AND e.group_id=g.group_id)"
            ).fetchall():
                group = EventGroup.from_dict(json.loads(str(row["group_json"])))
                conn.executemany(
                    "INSERT OR IGNORE INTO v2_page_wal_events VALUES(?,?,?,?,?)",
                    (
                        (
                            str(row["run_id"]),
                            str(row["branch_id"]),
                            event.event_id,
                            str(row["group_id"]),
                            int(row["event_start"]) + offset,
                        )
                        for offset, event in enumerate(group.events)
                    ),
                )
            # A directory is a deterministic projection of immutable PageSet
            # segments. Backfill older runs without rewriting their WAL.
            self._backfill_page_set_directories(conn)
            current = conn.execute(
                "SELECT version FROM v2_page_store_schema WHERE component='page_store'"
            ).fetchone()
            if current is not None and int(current["version"]) != _SCHEMA_VERSION:
                raise PageStoreError(f"unsupported Page Store schema {current['version']}")
            conn.execute(
                "INSERT OR IGNORE INTO v2_page_store_schema(component, version) VALUES('page_store', ?)",
                (_SCHEMA_VERSION,),
            )
            existing = conn.execute(
                "SELECT parent_branch_id, parent_page_id FROM v2_page_branches "
                "WHERE run_id=? AND branch_id=?",
                (self.run_id, self.branch_id),
            ).fetchone()
            if existing is None:
                if parent_branch_id is not None:
                    parent = conn.execute(
                        "SELECT run_id, branch_id FROM v2_pages WHERE page_id=?",
                        (parent_page_id,),
                    ).fetchone()
                    if (
                        parent is None
                        or str(parent["run_id"]) != self.run_id
                        or str(parent["branch_id"]) != parent_branch_id
                    ):
                        raise BranchVisibilityError("branch parent Page does not exist")
                conn.execute(
                    "INSERT INTO v2_page_branches(run_id, branch_id, parent_branch_id, parent_page_id) "
                    "VALUES(?, ?, ?, ?)",
                    (self.run_id, self.branch_id, parent_branch_id, parent_page_id),
                )
            elif (
                existing["parent_branch_id"] != parent_branch_id
                or existing["parent_page_id"] != parent_page_id
            ):
                # Reopening an established branch does not require callers to
                # repeat its immutable ancestry, but contradictory ancestry is rejected.
                if parent_branch_id is not None or parent_page_id is not None:
                    raise BranchVisibilityError("branch ancestry is immutable")

    def _backfill_page_set_directories(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT p.* FROM v2_page_sets p "
            "LEFT JOIN v2_page_set_synopses s ON s.page_set_id=p.page_set_id "
            "WHERE s.page_set_id IS NULL ORDER BY p.committed_at,p.page_set_id"
        ).fetchall()
        for row in rows:
            blob = conn.execute(
                "SELECT storage_path FROM v2_page_blobs WHERE handle=?",
                (row["logical_group_blob_handle"],),
            ).fetchone()
            if blob is None:
                raise PageIntegrityError("PageSet logical Blob is missing during directory repair")
            logical_group = EventGroup.from_dict(
                json.loads((self.root / str(blob["storage_path"])).read_bytes())
            )
            group_rows = conn.execute(
                "SELECT s.segment_index,g.group_json FROM v2_page_set_segments s "
                "JOIN v2_page_wal_groups g ON g.group_id=s.group_id "
                "AND g.run_id=? AND g.branch_id=? "
                "WHERE s.page_set_id=? ORDER BY s.segment_index",
                (self.run_id, self.branch_id, row["page_set_id"]),
            ).fetchall()
            segments = tuple(
                _PreparedGroup(
                    group=EventGroup.from_dict(json.loads(str(item["group_json"]))),
                    group_digest="",
                    redaction_proof="",
                    blobs=(),
                )
                for item in group_rows
            )
            if len(segments) != int(row["segment_count"]):
                raise PageIntegrityError("PageSet segment map is incomplete during directory repair")
            directory = self._build_page_set_directory(logical_group, segments)
            semantic_kinds = tuple(
                sorted({kind for entry in directory for kind in entry.semantic_kinds})
            )
            entity_refs = tuple(
                dict.fromkeys(entity for entry in directory for entity in entry.entity_refs)
            )[:64]
            synopsis_parts = [
                logical_group.group_type,
                "/".join(semantic_kinds) if semantic_kinds else "EXECUTE",
                f"{len(directory)} semantic segments",
            ]
            if logical_group.milestone_id:
                synopsis_parts.append(f"milestone {logical_group.milestone_id}")
            if entity_refs:
                synopsis_parts.append("entities " + ", ".join(entity_refs[:8]))
            conn.execute(
                "INSERT INTO v2_page_set_synopses VALUES(?,?,?,?,?)",
                (
                    row["page_set_id"],
                    "; ".join(synopsis_parts)[:1600],
                    json.dumps(semantic_kinds, separators=(",", ":")),
                    json.dumps(entity_refs, separators=(",", ":")),
                    digest(primitive(directory)),
                ),
            )
            conn.executemany(
                "INSERT INTO v2_page_set_segment_directory VALUES(?,?,?,?,?,?,?,?)",
                (
                    (
                        row["page_set_id"],
                        entry.segment_index,
                        entry.title,
                        entry.summary,
                        json.dumps(entry.semantic_kinds, separators=(",", ":")),
                        json.dumps(entry.entity_refs, separators=(",", ":")),
                        json.dumps(entry.event_ids, separators=(",", ":")),
                        json.dumps(entry.logical_event_ids, separators=(",", ":")),
                    )
                    for entry in directory
                ),
            )

    def page_state(self) -> PageState:
        with self._lock:
            token_count = self._open_token_count()
            if token_count < self.policy.min_tokens:
                return PageState.OPEN_BELOW_MIN
            if token_count < self.policy.target_tokens:
                return PageState.OPEN_NORMAL
            # CLOSE_PENDING is transient: append_group immediately seals at
            # the completed group boundary while holding this lock.
            return PageState.CLOSE_PENDING

    def append_group(self, group: EventGroup, *, defer_seal: bool = False) -> PageManifest | None:
        with self.metrics.timer("event_group_commit_ms"):
            with self._lock:
                self._ensure_open()
                self._last_sealed = ()
                self._validate_scope(group)
                existing_set = self._page_set_row(group.group_id)
                if existing_set is not None:
                    redacted = self._redact_group(group)
                    if str(existing_set["logical_group_digest"]) != digest(redacted):
                        raise PageIntegrityError(f"EventGroup ID collision: {group.group_id}")
                    manifests = self._manifests_for_page_set(str(existing_set["page_set_id"]))
                    self._last_sealed = manifests
                    return manifests[-1] if manifests else None
                existing = self._group_row(group.group_id)
                if existing is not None:
                    redacted = self._redact_group(group)
                    if str(existing["group_digest"]) != digest(redacted):
                        raise PageIntegrityError(f"EventGroup ID collision: {group.group_id}")
                    return self._manifest_for_group(group.group_id)

                prepared = self._prepare_group(group)
                if isinstance(prepared, _PreparedPageSet):
                    return self._append_page_set(prepared)
                # Identifiers are redacted too. A retry using the original
                # in-memory group must therefore deduplicate against the
                # already-persisted redacted identifier.
                existing = self._group_row(prepared.group.group_id)
                if existing is not None:
                    if str(existing["group_digest"]) != prepared.group_digest:
                        raise PageIntegrityError(
                            f"EventGroup ID collision: {prepared.group.group_id}"
                        )
                    return self._manifest_for_group(prepared.group.group_id)
                sealed: list[PageManifest] = []
                current_tokens = self._open_token_count()
                if (
                    current_tokens
                    and (
                        current_tokens + prepared.group.token_count
                        > self.policy.absolute_max_tokens
                    )
                ):
                    # A complete EventGroup is the indivisible WAL unit. Seal
                    # before admitting the next complete group whenever the
                    # physical Page would cross the hard bound, even if a
                    # higher-level semantic unit is still active. This keeps
                    # every physical Page bounded while preserving the logical
                    # unit through Page relations; a single oversized group is
                    # handled separately by the PageSet path above.
                    sealed.append(
                        self._seal_open_page(
                            TailReason.ABSOLUTE_MAX_SAFETY.value,
                            force_tail=current_tokens < self.policy.min_tokens,
                        )
                    )

                wal_record = {
                    "schema_version": _WAL_SCHEMA_VERSION,
                    "record_type": "EVENT_GROUP_COMMIT",
                    "run_id": self.run_id,
                    "branch_id": self.branch_id,
                    "group": primitive(prepared.group),
                    "group_digest": prepared.group_digest,
                    "redaction_proof": prepared.redaction_proof,
                    "blobs": [primitive(item) for item in prepared.blobs],
                }
                wal_record["record_digest"] = digest(wal_record)
                wal_start, wal_end = append_synced(
                    self.wal_path,
                    json_line(wal_record),
                    sync_hook=self.sync_hook,
                )
                self._commit_prepared(prepared, wal_start, wal_end)

                current_tokens = self._open_token_count()
                # TARGET enters CLOSE_PENDING. The Page closes only after the
                # provider-neutral semantic unit reports its real boundary;
                # notification/output deltas are never boundaries by themselves.
                should_seal = (
                    current_tokens >= self.policy.target_tokens and prepared.group.semantic_boundary
                )
                if should_seal and not defer_seal:
                    reason = (
                        "SEMANTIC_BOUNDARY_OVERSHOOT"
                        if current_tokens > self.policy.absolute_max_tokens
                        else "TARGET_REACHED"
                    )
                    sealed.append(self._seal_open_page(reason, force_tail=False))
                self._last_sealed = tuple(sealed)
                return sealed[-1] if sealed else None

    def _append_page_set(self, prepared: _PreparedPageSet) -> PageManifest:
        """Commit and consolidate one oversized group as an ordered episode."""

        existing = self._page_set_row(prepared.logical_group_id)
        if existing is not None:
            if str(existing["logical_group_digest"]) != prepared.logical_group_digest:
                raise PageIntegrityError(f"EventGroup ID collision: {prepared.logical_group_id}")
            manifests = self._manifests_for_page_set(str(existing["page_set_id"]))
            if not manifests:
                raise PageIntegrityError("durable PageSet has no sealed Pages")
            self._last_sealed = manifests
            return manifests[-1]

        sealed: list[PageManifest] = []
        if self._open_group_rows():
            sealed.append(
                self._seal_open_page(
                    TailReason.ABSOLUTE_MAX_SAFETY.value,
                    force_tail=self._open_token_count() < self.policy.min_tokens,
                )
            )
        wal_record = {
            "schema_version": _WAL_SCHEMA_VERSION,
            "record_type": "EVENT_GROUP_PAGE_SET_COMMIT",
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "page_set": {
                "page_set_id": prepared.page_set_id,
                "logical_group_id": prepared.logical_group_id,
                "logical_group_digest": prepared.logical_group_digest,
                "logical_group_blob_handle": prepared.logical_group_blob.handle,
                "segment_count": len(prepared.segments),
                "synopsis": prepared.synopsis,
                "semantic_kinds": list(prepared.semantic_kinds),
                "entity_refs": list(prepared.entity_refs),
                "directory_digest": prepared.directory_digest,
            },
            "segments": [
                {
                    "group": primitive(item.group),
                    "group_digest": item.group_digest,
                    "redaction_proof": item.redaction_proof,
                    "directory": primitive(prepared.directory[index]),
                }
                for index, item in enumerate(prepared.segments)
            ],
            "blobs": [primitive(item) for item in prepared.blobs],
        }
        wal_record["record_digest"] = digest(wal_record)
        wal_start, wal_end = append_synced(
            self.wal_path,
            json_line(wal_record),
            sync_hook=self.sync_hook,
        )
        self._commit_page_set(prepared, wal_start, wal_end)
        for segment in prepared.segments:
            sealed.append(
                self._seal_open_page(
                    TailReason.PAGE_SET_SEGMENT.value,
                    force_tail=segment.group.token_count < self.policy.min_tokens,
                    through_group_id=segment.group.group_id,
                )
            )
        page_set_manifests = tuple(
            item for item in sealed if item.seal_reason == TailReason.PAGE_SET_SEGMENT.value
        )
        if len(page_set_manifests) != len(prepared.segments):
            raise PageIntegrityError("PageSet sealing did not produce one Page per segment")
        self._last_sealed = tuple(sealed)
        return page_set_manifests[-1]

    def checkpoint(self, reason: TailReason | str) -> PageManifest | None:
        with self._lock:
            self._ensure_open()
            self._last_sealed = ()
            if not self._open_group_rows():
                return None
            reason_value = reason.value if isinstance(reason, TailReason) else str(reason)
            tokens = self._open_token_count()
            tail = tokens < self.policy.min_tokens
            if tail:
                try:
                    normalized = TailReason(reason_value)
                except ValueError as exc:
                    raise PageBoundaryError(
                        f"below-MIN Page requires a legal Tail reason, got {reason_value!r}"
                    ) from exc
                if normalized not in LEGAL_TAIL_REASONS:
                    raise PageBoundaryError("illegal Tail reason")
            manifest = self._seal_open_page(reason_value, force_tail=tail)
            self._last_sealed = (manifest,)
            return manifest

    def close(self, reason: TailReason | str = TailReason.RUN_END) -> PageManifest | None:
        with self._lock:
            if self._closed:
                return None
            manifest = self.checkpoint(reason)
            self._closed = True
            return manifest

    def recover(self, *, seal_tail: bool = True) -> RecoveryReport:
        """Replay durable WAL records missing from the Page Map.

        Recovery is idempotent. By default, remaining recovered staging facts
        become an explicit CRASH_RECOVERY Tail Page, providing a durable safety
        checkpoint without pretending it met the normal MIN threshold.
        """

        with self._lock:
            self._ensure_open()
            records = load_json_lines(self.wal_path)
            recovered = 0
            sealed: list[str] = []
            wal_offset = 0
            for record in records:
                encoded_record = json_line(record)
                wal_start = wal_offset
                wal_offset += len(encoded_record)
                prepared = self._validate_wal_record(record)
                if isinstance(prepared, _PreparedPageSet):
                    existing_set = self._page_set_row(prepared.logical_group_id)
                    if existing_set is not None:
                        if (
                            str(existing_set["logical_group_digest"])
                            != prepared.logical_group_digest
                        ):
                            raise PageIntegrityError(
                                f"WAL/Map digest conflict for {prepared.logical_group_id}"
                            )
                        for segment in prepared.segments:
                            row = self._group_row(segment.group.group_id)
                            if row is not None and row["page_id"] is None:
                                manifest = self._seal_open_page(
                                    TailReason.PAGE_SET_SEGMENT.value,
                                    force_tail=(segment.group.token_count < self.policy.min_tokens),
                                    through_group_id=segment.group.group_id,
                                )
                                sealed.append(manifest.page_id)
                        continue
                    if self._open_group_rows():
                        manifest = self._seal_open_page(
                            TailReason.CRASH_RECOVERY.value,
                            force_tail=self._open_token_count() < self.policy.min_tokens,
                        )
                        sealed.append(manifest.page_id)
                    self._commit_page_set(prepared, wal_start, wal_offset)
                    recovered += 1
                    for segment in prepared.segments:
                        manifest = self._seal_open_page(
                            TailReason.PAGE_SET_SEGMENT.value,
                            force_tail=segment.group.token_count < self.policy.min_tokens,
                            through_group_id=segment.group.group_id,
                        )
                        sealed.append(manifest.page_id)
                    continue
                existing = self._group_row(prepared.group.group_id)
                if existing is not None:
                    if str(existing["group_digest"]) != prepared.group_digest:
                        raise PageIntegrityError(
                            f"WAL/Map digest conflict for {prepared.group.group_id}"
                        )
                    continue
                # The physical offsets are useful for diagnostics, but replay
                # correctness depends on the synced record and its digest.
                self._commit_prepared(prepared, wal_start, wal_offset)
                recovered += 1
                tokens = self._open_token_count()
                if tokens >= self.policy.target_tokens and prepared.group.semantic_boundary:
                    manifest = self._seal_open_page(
                        "RECOVERED_TARGET",
                        force_tail=False,
                    )
                    sealed.append(manifest.page_id)
            if seal_tail and self._open_group_rows():
                manifest = self._seal_open_page(
                    TailReason.CRASH_RECOVERY.value,
                    force_tail=self._open_token_count() < self.policy.min_tokens,
                )
                sealed.append(manifest.page_id)
            return RecoveryReport(
                scanned_groups=len(records),
                recovered_groups=recovered,
                sealed_pages=tuple(sealed),
            )

    def open_page(self, page_id: str) -> tuple[EventGroup, ...]:
        with self.metrics.timer("page_open_ms"):
            with self._lock:
                row = self.database.connection.execute(
                    "SELECT * FROM v2_pages WHERE page_id=?", (page_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(page_id)
                self._assert_page_visible(row)
                path = self.root / str(row["storage_path"])
                try:
                    payload = path.read_bytes()
                except FileNotFoundError as exc:
                    raise PageIntegrityError(f"Page body missing: {page_id}") from exc
                if digest({"bytes": payload.hex()}) != str(row["storage_digest"]):
                    raise PageIntegrityError(f"Page storage digest mismatch: {page_id}")
                try:
                    document = json.loads(payload)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise PageIntegrityError(f"invalid Page document: {page_id}") from exc
                self._validate_page_document(row, document)
                groups = tuple(EventGroup.from_dict(item) for item in document["groups"])
                for group in groups:
                    self._validate_blob_references(group)
                return groups

    def open_blob(
        self,
        handle: str,
        byte_range: tuple[int, int] | None = None,
    ) -> bytes:
        row = self.database.connection.execute(
            "SELECT * FROM v2_page_blobs WHERE handle=?", (handle,)
        ).fetchone()
        if row is None:
            raise KeyError(handle)
        path = self.root / str(row["storage_path"])
        try:
            payload = path.read_bytes()
        except FileNotFoundError as exc:
            raise PageIntegrityError(f"Blob missing: {handle}") from exc
        if digest({"blob": payload.hex()}) != str(row["content_digest"]):
            raise PageIntegrityError(f"Blob digest mismatch: {handle}")
        if len(payload) != int(row["byte_count"]):
            raise PageIntegrityError(f"Blob size mismatch: {handle}")
        start, end = byte_range or (0, len(payload))
        if start < 0 or end < start or end > len(payload):
            raise PageIntegrityError(f"invalid Blob range for {handle}")
        return payload[start:end]

    def list_manifests(self, *, include_inherited: bool = False) -> tuple[PageManifest, ...]:
        rows = self.database.connection.execute(
            "SELECT manifest_json FROM v2_pages WHERE run_id=? AND branch_id=? ORDER BY page_seq",
            (self.run_id, self.branch_id),
        ).fetchall()
        result = [_page_manifest_from_dict(json.loads(row["manifest_json"])) for row in rows]
        if include_inherited:
            ancestry = self._visible_parent_pages()
            parent_rows = []
            for visible_id in ancestry:
                row = self.database.connection.execute(
                    "SELECT manifest_json FROM v2_pages WHERE page_id=?", (visible_id,)
                ).fetchone()
                if row is not None:
                    parent_rows.append(_page_manifest_from_dict(json.loads(row["manifest_json"])))
            result = parent_rows + result
        return tuple(result)

    def project_existing_pages(self) -> tuple[str, ...]:
        """Idempotently repair Semantic projections after WAL/Page recovery.

        Planning can crash after a Page is sealed but before the Task/Plan
        registry exists. In that ordering the Page is authoritative first and
        its projection must be deferred until the Registry scope is durable.
        """

        if self.projector is None:
            raise PageStoreError("Page projection repair requires a projector")
        projected: list[str] = []
        rows = self.database.connection.execute(
            "SELECT * FROM v2_pages WHERE run_id=? AND branch_id=? ORDER BY page_seq",
            (self.run_id, self.branch_id),
        ).fetchall()
        for row in rows:
            path = self.root / str(row["storage_path"])
            try:
                payload = path.read_bytes()
                document = json.loads(payload)
            except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PageIntegrityError(
                    f"cannot repair Page projection for {row['page_id']}"
                ) from exc
            self._validate_page_document(row, document)
            groups = tuple(EventGroup.from_dict(item) for item in document["groups"])
            positions = {
                str(event_id): (int(value[0]), int(value[1]))
                for event_id, value in document["event_positions"].items()
            }
            with self.database.transaction() as conn:
                with self.metrics.timer("semantic_projection_ms"):
                    self.projector(
                        conn,
                        _page_manifest_from_dict(document["manifest"]),
                        groups,
                        positions,
                    )
            projected.append(str(row["page_id"]))
        return tuple(projected)

    def _ensure_open(self) -> None:
        if self._closed:
            raise PageStoreError("PageStore is closed")

    def _validate_scope(self, group: EventGroup) -> None:
        if group.run_id != self.run_id or group.branch_id != self.branch_id:
            raise BranchVisibilityError("EventGroup run/branch does not match this PageStore")
        if not group.complete:
            raise PageBoundaryError("incomplete EventGroup cannot be appended")

    def _redact_group(self, group: EventGroup) -> EventGroup:
        redacted_value, _ = self.redactor.redact(primitive(group))
        return EventGroup.from_dict(redacted_value)

    def _prepare_group(self, group: EventGroup) -> _PreparedGroup | _PreparedPageSet:
        redacted_group = self._redact_group(group)
        blobs: list[BlobReference] = []
        events: list[Event] = []
        for event in redacted_group.events:
            facts: list[EvidenceDraft] = []
            for fact in event.facts:
                fact_tokens = max(1, (len(canonical_bytes(fact.content)) + 2) // 3)
                if fact_tokens <= self.policy.nominal_max_tokens:
                    facts.append(fact)
                    continue
                blob = self._store_blob(fact.content)
                blobs.append(blob)
                facts.append(self._externalized_fact(fact, blob))
            event = self._replace_event_facts(event, facts)
            payload_tokens = max(1, (len(canonical_bytes(event.payload)) + 2) // 3)
            if payload_tokens <= self.policy.nominal_max_tokens:
                events.append(event)
                continue
            blob = self._store_blob(event.payload)
            blobs.append(blob)
            events.append(self._externalized_event(event, blob))
        prepared_group = self._replace_group_events(redacted_group, events)
        # Several individually moderate payloads may collectively overflow a
        # Page. Externalize the largest remaining body until the *whole* group
        # fits; this preserves one WAL record and one semantic EventGroup.
        while prepared_group.token_count > self.policy.absolute_max_tokens:
            candidates: list[tuple[int, str, int, int | None]] = []
            for index, event in enumerate(events):
                if "external_payload" not in event.payload:
                    candidates.append((len(canonical_bytes(event.payload)), "payload", index, None))
                candidates.extend(
                    (
                        len(canonical_bytes(fact.content)),
                        "fact",
                        index,
                        fact_index,
                    )
                    for fact_index, fact in enumerate(event.facts)
                    if "external_fact" not in fact.content
                )
            if not candidates:
                break
            _, kind, index, fact_index = max(candidates, key=lambda item: item[0])
            event = events[index]
            if kind == "payload":
                blob = self._store_blob(event.payload)
                events[index] = self._externalized_event(event, blob)
            else:
                assert fact_index is not None
                fact = event.facts[fact_index]
                blob = self._store_blob(fact.content)
                facts = list(event.facts)
                facts[fact_index] = self._externalized_fact(fact, blob)
                events[index] = self._replace_event_facts(event, facts)
            blobs.append(blob)
            prepared_group = self._replace_group_events(redacted_group, events)
        if prepared_group.token_count > self.policy.absolute_max_tokens:
            return self._prepare_page_set(redacted_group, events, blobs)
        return _PreparedGroup(
            group=prepared_group,
            group_digest=digest(prepared_group),
            redaction_proof=self.redactor.proof(primitive(prepared_group)),
            blobs=tuple(blobs),
        )

    def _prepare_page_set(
        self,
        logical_group: EventGroup,
        externalized_events: Sequence[Event],
        existing_blobs: Sequence[BlobReference],
    ) -> _PreparedPageSet:
        """Split only physical representation; preserve the logical group in CAS.

        Event bodies have already been externalized. Remaining size therefore
        comes from bounded structural collections (Events, Evidence drafts and
        entity references). They are packed into deterministic continuation
        Events and then into Page-sized physical groups. A single indivisible
        structural atom is itself externalized as an opaque fragment rather
        than turning Page overflow into a fatal runtime error.
        """

        logical_digest = digest(logical_group)
        logical_blob = self._store_blob(primitive(logical_group))
        blobs: list[BlobReference] = [*existing_blobs, logical_blob]
        atoms: list[Event] = []
        for event_index, event in enumerate(externalized_events):
            atoms.extend(
                self._split_event_metadata(
                    logical_group,
                    event,
                    event_index=event_index,
                    blobs=blobs,
                )
            )

        event_bins: list[list[Event]] = []
        current: list[Event] = []
        for atom in atoms:
            candidate = [*current, atom]
            probe = self._physical_page_set_group(
                logical_group,
                candidate,
                segment_index=len(event_bins),
                logical_digest=logical_digest,
                semantic_boundary=False,
            )
            if current and probe.token_count > self.policy.target_tokens:
                event_bins.append(current)
                current = [atom]
            else:
                current = candidate
            single_probe = self._physical_page_set_group(
                logical_group,
                current,
                segment_index=len(event_bins),
                logical_digest=logical_digest,
                semantic_boundary=False,
            )
            if single_probe.token_count > self.policy.absolute_max_tokens:
                opaque = self._opaque_page_set_atom(
                    logical_group,
                    atom,
                    event_index=len(atoms) + len(event_bins),
                    blobs=blobs,
                )
                current = [opaque]
                single_probe = self._physical_page_set_group(
                    logical_group,
                    current,
                    segment_index=len(event_bins),
                    logical_digest=logical_digest,
                    semantic_boundary=False,
                )
                if single_probe.token_count > self.policy.absolute_max_tokens:
                    raise PageBoundaryError("PageSet structural locator exceeds ABSOLUTE_MAX")
        if current:
            event_bins.append(current)
        if len(event_bins) > 1:
            tail_probe = self._physical_page_set_group(
                logical_group,
                event_bins[-1],
                segment_index=len(event_bins) - 1,
                logical_digest=logical_digest,
                semantic_boundary=True,
            )
            merged_probe = self._physical_page_set_group(
                logical_group,
                [*event_bins[-2], *event_bins[-1]],
                segment_index=len(event_bins) - 2,
                logical_digest=logical_digest,
                semantic_boundary=True,
            )
            if (
                tail_probe.token_count < self.policy.min_tokens
                and merged_probe.token_count <= self.policy.absolute_max_tokens
            ):
                event_bins[-2] = [*event_bins[-2], *event_bins[-1]]
                event_bins.pop()
        if len(event_bins) < 2:
            # This branch is defensive: the caller enters only after the
            # fully externalized logical group exceeded ABSOLUTE_MAX.
            raise PageBoundaryError("oversized EventGroup did not produce a multi-Page PageSet")

        page_set_id = stable_id(
            "page_set_",
            {
                "run_id": logical_group.run_id,
                "branch_id": logical_group.branch_id,
                "logical_group_id": logical_group.group_id,
                "logical_group_digest": logical_digest,
            },
        )
        segments: list[_PreparedGroup] = []
        for index, events in enumerate(event_bins):
            physical = self._physical_page_set_group(
                logical_group,
                events,
                segment_index=index,
                logical_digest=logical_digest,
                semantic_boundary=index == len(event_bins) - 1,
            )
            if physical.token_count > self.policy.absolute_max_tokens:
                raise PageBoundaryError("PageSet segment exceeds ABSOLUTE_MAX")
            segments.append(
                _PreparedGroup(
                    group=physical,
                    group_digest=digest(physical),
                    redaction_proof=self.redactor.proof(primitive(physical)),
                    blobs=(),
                )
            )
        directory = self._build_page_set_directory(logical_group, tuple(segments))
        semantic_kinds = tuple(
            sorted({kind for entry in directory for kind in entry.semantic_kinds})
        )
        entity_refs = tuple(
            dict.fromkeys(entity for entry in directory for entity in entry.entity_refs)
        )[:64]
        synopsis_parts = [
            logical_group.group_type,
            "/".join(semantic_kinds) if semantic_kinds else "EXECUTE",
            f"{len(directory)} semantic segments",
        ]
        if logical_group.milestone_id:
            synopsis_parts.append(f"milestone {logical_group.milestone_id}")
        if entity_refs:
            synopsis_parts.append("entities " + ", ".join(entity_refs[:8]))
        unique_blobs = {item.handle: item for item in blobs}
        return _PreparedPageSet(
            page_set_id=page_set_id,
            logical_group_id=logical_group.group_id,
            logical_group_digest=logical_digest,
            logical_group_blob=logical_blob,
            segments=tuple(segments),
            blobs=tuple(unique_blobs[key] for key in sorted(unique_blobs)),
            synopsis="; ".join(synopsis_parts)[:1600],
            semantic_kinds=semantic_kinds,
            entity_refs=entity_refs,
            directory=directory,
        )

    @staticmethod
    def _page_set_semantic_kinds(event: Event) -> tuple[str, ...]:
        """Classify an already structured event without a model-side summary call."""

        kinds: set[str] = set()
        fact_kinds = {
            FactType.USER_CONSTRAINT: "PLAN",
            FactType.PLAN_DECISION: "REPLAN",
            FactType.MILESTONE_STATE: "MILESTONE",
            FactType.CODE_OBSERVATION: "DIAGNOSE",
            FactType.IMPLEMENTATION_DECISION: "DECIDE",
            FactType.CODE_CHANGE: "IMPLEMENT",
            FactType.TOOL_RESULT: "EXECUTE",
            FactType.TEST_RESULT: "VERIFY",
            FactType.TEST_FAILURE: "DIAGNOSE",
            FactType.VERIFIER_RESULT: "VERIFY",
            FactType.UNRESOLVED_QUESTION: "QUESTION",
        }
        for fact in event.facts:
            kinds.add(fact_kinds[fact.key.evidence_type])
            if fact.key.evidence_type is FactType.TEST_FAILURE:
                kinds.add("VERIFY")
        event_type = event.event_type.upper()
        lexical_kinds = (
            (("PLAN", "MILESTONE", "CRITERION"), "PLAN"),
            (("TEST", "VERIFY", "VALIDAT"), "VERIFY"),
            (("FILE", "PATCH", "CHANGE", "IMPLEMENT"), "IMPLEMENT"),
            (("FAIL", "ERROR", "DIAGNOS"), "DIAGNOSE"),
            (("DEPEND",), "DEPENDENCY"),
            (("DECISION", "REVIEW"), "DECIDE"),
            (("MEMORY", "RECALL", "CONTEXT"), "MEMORY"),
            (("TOOL", "COMMAND"), "EXECUTE"),
        )
        for needles, kind in lexical_kinds:
            if any(needle in event_type for needle in needles):
                kinds.add(kind)
        return tuple(sorted(kinds or {"EXECUTE"}))

    @staticmethod
    def _logical_event_id(event: Event) -> str:
        for key in ("page_set_continuation", "external_page_set_fragment"):
            value = event.payload.get(key)
            if isinstance(value, Mapping) and value.get("logical_event_id"):
                return str(value["logical_event_id"])
        return event.event_id

    @classmethod
    def _build_page_set_directory(
        cls,
        logical_group: EventGroup,
        segments: Sequence[_PreparedGroup],
    ) -> tuple[_PageSetDirectoryEntry, ...]:
        entries: list[_PageSetDirectoryEntry] = []
        for segment_index, prepared in enumerate(segments):
            events = prepared.group.events
            semantic_kinds = tuple(
                sorted({kind for event in events for kind in cls._page_set_semantic_kinds(event)})
            )
            entities = tuple(
                dict.fromkeys(
                    entity
                    for event in events
                    for entity in (
                        *event.entity_refs,
                        *(fact.key.canonical_entity_id for fact in event.facts),
                    )
                )
            )
            event_types = tuple(dict.fromkeys(event.event_type for event in events))
            fact_count = sum(len(event.facts) for event in events)
            primary = semantic_kinds[0] if semantic_kinds else "EXECUTE"
            summary_parts = [
                "/".join(event_types[:4]),
                "/".join(semantic_kinds),
                f"{len(events)} events and {fact_count} evidence units",
            ]
            if entities:
                summary_parts.append("entities " + ", ".join(entities[:8]))
            entries.append(
                _PageSetDirectoryEntry(
                    segment_index=segment_index,
                    title=(
                        f"{logical_group.group_type}: {primary} "
                        f"({segment_index + 1}/{len(segments)})"
                    )[:240],
                    summary="; ".join(summary_parts)[:1200],
                    semantic_kinds=semantic_kinds,
                    entity_refs=entities,
                    event_ids=tuple(event.event_id for event in events),
                    logical_event_ids=tuple(
                        dict.fromkeys(cls._logical_event_id(event) for event in events)
                    ),
                )
            )
        return tuple(entries)

    def _split_event_metadata(
        self,
        logical_group: EventGroup,
        event: Event,
        *,
        event_index: int,
        blobs: list[BlobReference],
    ) -> tuple[Event, ...]:
        items: list[tuple[str, object]] = [
            *(("fact", fact) for fact in event.facts),
            *(("entity", ref) for ref in event.entity_refs),
        ]
        chunks: list[Event] = []
        facts: list[EvidenceDraft] = []
        refs: list[str] = []
        first = True

        def make_chunk() -> Event:
            chunk_index = len(chunks)
            return Event(
                event_id=(
                    event.event_id
                    if chunk_index == 0
                    else stable_id(
                        "event_segment_",
                        {
                            "logical_event_id": event.event_id,
                            "event_index": event_index,
                            "segment_index": chunk_index,
                        },
                    )
                ),
                event_type=event.event_type,
                payload=(
                    event.payload
                    if first
                    else {
                        "page_set_continuation": {
                            "logical_event_id": event.event_id,
                            "event_index": event_index,
                            "segment_index": chunk_index,
                        }
                    }
                ),
                facts=tuple(facts),
                entity_refs=tuple(refs),
                milestone_id=event.milestone_id,
                execution_phase=event.execution_phase,
                revision_id=event.revision_id,
                observed_at=event.observed_at,
            )

        if not items:
            return (make_chunk(),)
        for kind, value in items:
            if kind == "fact":
                facts.append(value)  # type: ignore[arg-type]
            else:
                refs.append(str(value))
            candidate = make_chunk()
            probe = self._physical_page_set_group(
                logical_group,
                (candidate,),
                segment_index=0,
                logical_digest=digest(logical_group),
                semantic_boundary=False,
            )
            if probe.token_count <= self.policy.target_tokens:
                continue
            if (len(facts) + len(refs)) > 1:
                if kind == "fact":
                    facts.pop()
                else:
                    refs.pop()
                chunks.append(make_chunk())
                first = False
                facts = [value] if kind == "fact" else []  # type: ignore[list-item]
                refs = [str(value)] if kind == "entity" else []
        chunks.append(make_chunk())
        result: list[Event] = []
        for ordinal, chunk in enumerate(chunks):
            probe = self._physical_page_set_group(
                logical_group,
                (chunk,),
                segment_index=0,
                logical_digest=digest(logical_group),
                semantic_boundary=False,
            )
            result.append(
                chunk
                if probe.token_count <= self.policy.absolute_max_tokens
                else self._opaque_page_set_atom(
                    logical_group,
                    chunk,
                    event_index=event_index * 1_000_000 + ordinal,
                    blobs=blobs,
                )
            )
        return tuple(result)

    def _opaque_page_set_atom(
        self,
        logical_group: EventGroup,
        event: Event,
        *,
        event_index: int,
        blobs: list[BlobReference],
    ) -> Event:
        blob = self._store_blob({"event": primitive(event)})
        blobs.append(blob)
        return Event(
            event_id=event.event_id,
            event_type="PAGE_SET_OPAQUE_FRAGMENT",
            payload={
                "external_page_set_fragment": {
                    "blob_handle": blob.handle,
                    "byte_range": list(blob.byte_range),
                    "byte_count": blob.byte_count,
                    "logical_event_id": event.event_id,
                    "fragment_index": event_index,
                }
            },
            revision_id=logical_group.revision_id,
            execution_phase="page_set",
            observed_at=event.observed_at,
        )

    @staticmethod
    def _physical_page_set_group(
        logical_group: EventGroup,
        events: Sequence[Event],
        *,
        segment_index: int,
        logical_digest: str,
        semantic_boundary: bool,
    ) -> EventGroup:
        group_id = (
            logical_group.group_id
            if segment_index == 0
            else stable_id(
                "event_group_segment_",
                {
                    "logical_group_id": logical_group.group_id,
                    "logical_group_digest": logical_digest,
                    "segment_index": segment_index,
                },
            )
        )
        return EventGroup(
            group_id=group_id,
            group_type=logical_group.group_type,
            run_id=logical_group.run_id,
            branch_id=logical_group.branch_id,
            revision_id=logical_group.revision_id,
            events=tuple(events),
            milestone_id=logical_group.milestone_id,
            semantic_boundary=semantic_boundary,
            complete=True,
            created_at=logical_group.created_at,
        )

    @staticmethod
    def _replace_group_events(group: EventGroup, events: Sequence[Event]) -> EventGroup:
        return EventGroup(
            group_id=group.group_id,
            group_type=group.group_type,
            run_id=group.run_id,
            branch_id=group.branch_id,
            revision_id=group.revision_id,
            events=tuple(events),
            milestone_id=group.milestone_id,
            semantic_boundary=group.semantic_boundary,
            complete=True,
            created_at=group.created_at,
        )

    @staticmethod
    def _externalized_event(event: Event, blob: BlobReference) -> Event:
        return Event(
            event_id=event.event_id,
            event_type=event.event_type,
            payload={
                "external_payload": {
                    "blob_handle": blob.handle,
                    "byte_range": list(blob.byte_range),
                    "byte_count": blob.byte_count,
                }
            },
            facts=event.facts,
            entity_refs=event.entity_refs,
            milestone_id=event.milestone_id,
            execution_phase=event.execution_phase,
            revision_id=event.revision_id,
            observed_at=event.observed_at,
        )

    @staticmethod
    def _replace_event_facts(event: Event, facts: Sequence[EvidenceDraft]) -> Event:
        return Event(
            event_id=event.event_id,
            event_type=event.event_type,
            payload=event.payload,
            facts=tuple(facts),
            entity_refs=event.entity_refs,
            milestone_id=event.milestone_id,
            execution_phase=event.execution_phase,
            revision_id=event.revision_id,
            observed_at=event.observed_at,
        )

    @staticmethod
    def _externalized_fact(fact: EvidenceDraft, blob: BlobReference) -> EvidenceDraft:
        return EvidenceDraft(
            key=fact.key,
            content={
                "external_fact": {
                    "blob_handle": blob.handle,
                    "byte_range": list(blob.byte_range),
                    "byte_count": blob.byte_count,
                }
            },
            authority=fact.authority,
            confidence=fact.confidence,
            must_preserve=fact.must_preserve,
        )

    def _store_blob(self, payload_value: Mapping[str, Any]) -> BlobReference:
        payload = canonical_bytes(payload_value)
        content_digest = digest({"blob": payload.hex()})
        hexadecimal = content_digest.removeprefix("sha256:")
        relative = Path("blobs") / hexadecimal[:2] / f"{hexadecimal}.blob"
        target = self.root / relative
        atomic_write_once(target, payload, sync_hook=self.sync_hook)
        return BlobReference(
            handle=content_digest,
            content_digest=content_digest,
            byte_range=(0, len(payload)),
            byte_count=len(payload),
            relative_path=relative.as_posix(),
        )

    def _commit_prepared(self, prepared: _PreparedGroup, wal_start: int, wal_end: int) -> None:
        event_start = self._next_event_cursor()
        event_end = event_start + len(prepared.group.events)
        group_json = canonical_bytes(prepared.group).decode("utf-8")
        with self.database.transaction() as conn:
            for blob in prepared.blobs:
                conn.execute(
                    "INSERT OR IGNORE INTO v2_page_blobs(handle, content_digest, byte_count, storage_path, redaction_proof) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (
                        blob.handle,
                        blob.content_digest,
                        blob.byte_count,
                        blob.relative_path,
                        self.redactor.proof({"handle": blob.handle, "byte_count": blob.byte_count}),
                    ),
                )
            conn.execute(
                "INSERT INTO v2_page_wal_groups("
                "run_id, branch_id, group_id, group_digest, group_json, redaction_proof, "
                "revision_id, milestone_id, event_start, event_end, wal_start, wal_end"
                ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.run_id,
                    self.branch_id,
                    prepared.group.group_id,
                    prepared.group_digest,
                    group_json,
                    prepared.redaction_proof,
                    prepared.group.revision_id,
                    prepared.group.milestone_id,
                    event_start,
                    event_end,
                    wal_start,
                    wal_end,
                ),
            )
            conn.executemany(
                "INSERT INTO v2_page_wal_events VALUES(?,?,?,?,?)",
                (
                    (
                        self.run_id,
                        self.branch_id,
                        event.event_id,
                        prepared.group.group_id,
                        event_start + offset,
                    )
                    for offset, event in enumerate(prepared.group.events)
                ),
            )

    def _commit_page_set(
        self,
        prepared: _PreparedPageSet,
        wal_start: int,
        wal_end: int,
    ) -> None:
        """Project every physical segment in one Page Map transaction."""

        event_cursor = self._next_event_cursor()
        with self.database.transaction() as conn:
            for blob in prepared.blobs:
                conn.execute(
                    "INSERT OR IGNORE INTO v2_page_blobs(handle, content_digest, byte_count, storage_path, redaction_proof) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (
                        blob.handle,
                        blob.content_digest,
                        blob.byte_count,
                        blob.relative_path,
                        self.redactor.proof({"handle": blob.handle, "byte_count": blob.byte_count}),
                    ),
                )
            conn.execute(
                "INSERT INTO v2_page_sets("
                "page_set_id,run_id,branch_id,logical_group_id,logical_group_digest,"
                "logical_group_blob_handle,segment_count,wal_start,wal_end"
                ") VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    prepared.page_set_id,
                    self.run_id,
                    self.branch_id,
                    prepared.logical_group_id,
                    prepared.logical_group_digest,
                    prepared.logical_group_blob.handle,
                    len(prepared.segments),
                    wal_start,
                    wal_end,
                ),
            )
            conn.execute(
                "INSERT INTO v2_page_set_synopses("
                "page_set_id,synopsis,semantic_kinds_json,entity_refs_json,directory_digest"
                ") VALUES(?,?,?,?,?)",
                (
                    prepared.page_set_id,
                    prepared.synopsis,
                    json.dumps(prepared.semantic_kinds, separators=(",", ":")),
                    json.dumps(prepared.entity_refs, separators=(",", ":")),
                    prepared.directory_digest,
                ),
            )
            for index, segment in enumerate(prepared.segments):
                event_start = event_cursor
                event_end = event_start + len(segment.group.events)
                conn.execute(
                    "INSERT INTO v2_page_wal_groups("
                    "run_id, branch_id, group_id, group_digest, group_json, redaction_proof, "
                    "revision_id, milestone_id, event_start, event_end, wal_start, wal_end"
                    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        self.run_id,
                        self.branch_id,
                        segment.group.group_id,
                        segment.group_digest,
                        canonical_bytes(segment.group).decode("utf-8"),
                        segment.redaction_proof,
                        segment.group.revision_id,
                        segment.group.milestone_id,
                        event_start,
                        event_end,
                        wal_start,
                        wal_end,
                    ),
                )
                conn.executemany(
                    "INSERT INTO v2_page_wal_events VALUES(?,?,?,?,?)",
                    (
                        (
                            self.run_id,
                            self.branch_id,
                            event.event_id,
                            segment.group.group_id,
                            event_start + offset,
                        )
                        for offset, event in enumerate(segment.group.events)
                    ),
                )
                conn.execute(
                    "INSERT INTO v2_page_set_segments(page_set_id,segment_index,group_id) "
                    "VALUES(?,?,?)",
                    (prepared.page_set_id, index, segment.group.group_id),
                )
                directory = prepared.directory[index]
                conn.execute(
                    "INSERT INTO v2_page_set_segment_directory("
                    "page_set_id,segment_index,title,summary,semantic_kinds_json,"
                    "entity_refs_json,event_ids_json,logical_event_ids_json"
                    ") VALUES(?,?,?,?,?,?,?,?)",
                    (
                        prepared.page_set_id,
                        index,
                        directory.title,
                        directory.summary,
                        json.dumps(directory.semantic_kinds, separators=(",", ":")),
                        json.dumps(directory.entity_refs, separators=(",", ":")),
                        json.dumps(directory.event_ids, separators=(",", ":")),
                        json.dumps(directory.logical_event_ids, separators=(",", ":")),
                    ),
                )
                event_cursor = event_end

    def _seal_open_page(
        self,
        reason: str,
        *,
        force_tail: bool,
        through_group_id: str | None = None,
    ) -> PageManifest:
        rows = self._open_group_rows(through_group_id=through_group_id)
        if not rows:
            raise PageBoundaryError("cannot seal an empty Page")
        groups = tuple(EventGroup.from_dict(json.loads(row["group_json"])) for row in rows)
        tokens = sum(group.token_count for group in groups)
        if tokens > self.policy.absolute_max_tokens and not (
            groups[-1].semantic_boundary or reason in {item.value for item in LEGAL_TAIL_REASONS}
        ):
            raise PageBoundaryError(
                "Page exceeds ABSOLUTE_MAX without a completed or forced semantic boundary"
            )
        tail = force_tail or tokens < self.policy.min_tokens
        if tail:
            try:
                tail_reason = TailReason(reason)
            except ValueError as exc:
                raise PageBoundaryError(f"illegal Tail reason: {reason}") from exc
            if tail_reason not in LEGAL_TAIL_REASONS:
                raise PageBoundaryError(f"illegal Tail reason: {reason}")

        page_seq = self._next_page_seq()
        previous_page_id = self._previous_page_id()
        payload_digest = digest({"groups": primitive(groups)})
        page_id = stable_id(
            "page_",
            {
                "run_id": self.run_id,
                "branch_id": self.branch_id,
                "page_seq": page_seq,
                "payload_digest": payload_digest,
                "seal_reason": reason,
                "tail": tail,
            },
        )
        event_positions: dict[str, tuple[int, int]] = {}
        for row, group in zip(rows, groups, strict=True):
            cursor = int(row["event_start"])
            for event in group.events:
                event_positions[event.event_id] = (cursor, cursor + 1)
                cursor += 1
        evidence_keys = sorted(
            {
                fact.key.key_digest
                for group in groups
                for event in group.events
                for fact in event.facts
            }
        )
        entity_refs = sorted(
            {ref for group in groups for event in group.events for ref in event.entity_refs}
        )
        milestone_ids = sorted(
            {
                milestone
                for group in groups
                for milestone in (
                    (group.milestone_id,) + tuple(event.milestone_id for event in group.events)
                )
                if milestone
            }
        )
        phases = sorted({event.execution_phase for group in groups for event in group.events})
        revisions = sorted({group.revision_id for group in groups})
        redaction_proof = digest(
            {
                "redaction_version": self.redactor.version,
                "group_proofs": [str(row["redaction_proof"]) for row in rows],
            }
        )
        page_kind = (
            PageKind.SEMANTIC if evidence_keys or entity_refs or milestone_ids else PageKind.AUDIT
        )
        # byte_count describes canonical groups, not the self-describing Page
        # envelope (whose exact storage size is verified separately).
        group_bytes = canonical_bytes(groups)
        manifest = PageManifest(
            page_id=page_id,
            run_id=self.run_id,
            branch_id=self.branch_id,
            page_seq=page_seq,
            revision_ids=tuple(revisions),
            milestone_ids=tuple(milestone_ids),
            execution_phases=tuple(phases),
            event_range=(int(rows[0]["event_start"]), int(rows[-1]["event_end"])),
            event_group_ids=tuple(group.group_id for group in groups),
            evidence_key_digests=tuple(evidence_keys),
            entity_refs=tuple(entity_refs),
            token_count=tokens,
            byte_count=len(group_bytes),
            payload_digest=payload_digest,
            redaction_proof=redaction_proof,
            previous_page_id=previous_page_id,
            seal_reason=reason,
            page_kind=page_kind,
            tail=tail,
        )
        document = {
            "schema_version": _PAGE_SCHEMA_VERSION,
            "manifest": primitive(manifest),
            "groups": primitive(groups),
            "event_positions": {key: list(value) for key, value in sorted(event_positions.items())},
        }
        page_bytes = canonical_bytes(document)
        storage_digest = digest({"bytes": page_bytes.hex()})
        relative = (
            Path("pages")
            / _safe_component(self.run_id)
            / _safe_component(self.branch_id)
            / f"{page_seq:08d}-{page_id}.page.json"
        )
        with self.metrics.timer("page_seal_ms"):
            atomic_write_once(
                self.root / relative,
                page_bytes,
                sync_hook=self.sync_hook,
            )
            manifest_json = canonical_bytes(manifest).decode("utf-8")
            with self.database.transaction() as conn:
                conn.execute(
                    "INSERT INTO v2_pages("
                    "page_id, run_id, branch_id, page_seq, event_start, event_end, "
                    "payload_digest, storage_digest, storage_path, manifest_json, "
                    "previous_page_id, tail, seal_reason"
                    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        page_id,
                        self.run_id,
                        self.branch_id,
                        page_seq,
                        manifest.event_range[0],
                        manifest.event_range[1],
                        payload_digest,
                        storage_digest,
                        relative.as_posix(),
                        manifest_json,
                        previous_page_id,
                        int(tail),
                        reason,
                    ),
                )
                conn.executemany(
                    "UPDATE v2_page_wal_groups SET page_id=? "
                    "WHERE run_id=? AND branch_id=? AND group_id=? AND page_id IS NULL",
                    [(page_id, self.run_id, self.branch_id, group.group_id) for group in groups],
                )
                conn.executemany(
                    "UPDATE v2_page_set_segments SET page_id=? "
                    "WHERE group_id=? AND page_id IS NULL",
                    [(page_id, group.group_id) for group in groups],
                )
                if self.projector is not None:
                    with self.metrics.timer("semantic_projection_ms"):
                        self.projector(conn, manifest, groups, event_positions)
        return manifest

    def _open_group_rows(self, *, through_group_id: str | None = None) -> list[sqlite3.Row]:
        through_event_end: int | None = None
        if through_group_id is not None:
            row = self._group_row(through_group_id)
            if row is None or row["page_id"] is not None:
                raise PageBoundaryError(f"PageSet segment is not open: {through_group_id}")
            through_event_end = int(row["event_end"])
        return list(
            self.database.connection.execute(
                "SELECT * FROM v2_page_wal_groups "
                "WHERE run_id=? AND branch_id=? AND page_id IS NULL "
                + ("AND event_end<=? " if through_event_end is not None else "")
                + "ORDER BY event_start",
                (
                    (self.run_id, self.branch_id, through_event_end)
                    if through_event_end is not None
                    else (self.run_id, self.branch_id)
                ),
            ).fetchall()
        )

    def _open_token_count(self) -> int:
        return sum(
            EventGroup.from_dict(json.loads(row["group_json"])).token_count
            for row in self._open_group_rows()
        )

    def _group_row(self, group_id: str) -> sqlite3.Row | None:
        return self.database.connection.execute(
            "SELECT * FROM v2_page_wal_groups WHERE run_id=? AND branch_id=? AND group_id=?",
            (self.run_id, self.branch_id, group_id),
        ).fetchone()

    def _page_set_row(self, logical_group_id: str) -> sqlite3.Row | None:
        return self.database.connection.execute(
            "SELECT * FROM v2_page_sets WHERE run_id=? AND branch_id=? AND logical_group_id=?",
            (self.run_id, self.branch_id, logical_group_id),
        ).fetchone()

    def _manifests_for_page_set(self, page_set_id: str) -> tuple[PageManifest, ...]:
        rows = self.database.connection.execute(
            "SELECT p.manifest_json FROM v2_page_set_segments s "
            "JOIN v2_pages p ON p.page_id=s.page_id "
            "WHERE s.page_set_id=? ORDER BY s.segment_index",
            (page_set_id,),
        ).fetchall()
        return tuple(_page_manifest_from_dict(json.loads(row["manifest_json"])) for row in rows)

    def _manifest_for_group(self, group_id: str) -> PageManifest | None:
        row = self._group_row(group_id)
        if row is None or row["page_id"] is None:
            return None
        page = self.database.connection.execute(
            "SELECT manifest_json FROM v2_pages WHERE page_id=?", (row["page_id"],)
        ).fetchone()
        return _page_manifest_from_dict(json.loads(page["manifest_json"])) if page else None

    def _next_event_cursor(self) -> int:
        row = self.database.connection.execute(
            "SELECT MAX(event_end) AS cursor FROM v2_page_wal_groups WHERE run_id=? AND branch_id=?",
            (self.run_id, self.branch_id),
        ).fetchone()
        return int(row["cursor"] or 0)

    def _next_page_seq(self) -> int:
        row = self.database.connection.execute(
            "SELECT MAX(page_seq) AS seq FROM v2_pages WHERE run_id=? AND branch_id=?",
            (self.run_id, self.branch_id),
        ).fetchone()
        return int(row["seq"] if row["seq"] is not None else -1) + 1

    def _previous_page_id(self) -> str | None:
        row = self.database.connection.execute(
            "SELECT page_id FROM v2_pages WHERE run_id=? AND branch_id=? ORDER BY page_seq DESC LIMIT 1",
            (self.run_id, self.branch_id),
        ).fetchone()
        return str(row["page_id"]) if row else None

    def _validate_wal_record(
        self,
        record: Mapping[str, Any],
    ) -> _PreparedGroup | _PreparedPageSet:
        if int(record.get("schema_version", -1)) != _WAL_SCHEMA_VERSION:
            raise PageIntegrityError("unsupported WAL schema")
        record_type = str(record.get("record_type", ""))
        if record_type not in {"EVENT_GROUP_COMMIT", "EVENT_GROUP_PAGE_SET_COMMIT"}:
            raise PageIntegrityError("unsupported WAL record")
        if record.get("run_id") != self.run_id or record.get("branch_id") != self.branch_id:
            raise BranchVisibilityError("foreign scope record in branch WAL")
        claimed = str(record.get("record_digest", ""))
        unsigned = dict(record)
        unsigned.pop("record_digest", None)
        if digest(unsigned) != claimed:
            raise PageIntegrityError("WAL record digest mismatch")
        blobs = self._validated_wal_blobs(record)
        if record_type == "EVENT_GROUP_PAGE_SET_COMMIT":
            raw_set = record.get("page_set")
            raw_segments = record.get("segments")
            if not isinstance(raw_set, Mapping) or not isinstance(raw_segments, Sequence):
                raise PageIntegrityError("invalid PageSet WAL structure")
            segment_count = int(raw_set.get("segment_count", 0))
            if segment_count <= 1 or len(raw_segments) != segment_count:
                raise PageIntegrityError("PageSet WAL segment count mismatch")
            segments: list[_PreparedGroup] = []
            for raw_segment in raw_segments:
                if not isinstance(raw_segment, Mapping):
                    raise PageIntegrityError("invalid PageSet segment")
                group = EventGroup.from_dict(raw_segment["group"])
                self._validate_scope(group)
                group_digest = digest(group)
                if group_digest != raw_segment.get("group_digest"):
                    raise PageIntegrityError("WAL PageSet segment digest mismatch")
                proof = str(raw_segment.get("redaction_proof", ""))
                if proof != self.redactor.proof(primitive(group)):
                    raise PageIntegrityError("WAL PageSet redaction proof mismatch")
                if group.token_count > self.policy.absolute_max_tokens:
                    raise PageIntegrityError("WAL PageSet segment exceeds ABSOLUTE_MAX")
                segments.append(_PreparedGroup(group, group_digest, proof, ()))
            logical_group_id = str(raw_set.get("logical_group_id", ""))
            logical_digest = str(raw_set.get("logical_group_digest", ""))
            logical_handle = str(raw_set.get("logical_group_blob_handle", ""))
            logical_blob = next((item for item in blobs if item.handle == logical_handle), None)
            if logical_blob is None:
                raise PageIntegrityError("WAL PageSet logical Blob is missing")
            decoded = json.loads((self.root / logical_blob.relative_path).read_bytes())
            if not isinstance(decoded, dict):
                raise PageIntegrityError("WAL PageSet logical Blob is invalid")
            logical_group = EventGroup.from_dict(decoded)
            self._validate_scope(logical_group)
            if (
                logical_group.group_id != logical_group_id
                or digest(logical_group) != logical_digest
            ):
                raise PageIntegrityError("WAL PageSet logical group digest mismatch")
            page_set_id = str(raw_set.get("page_set_id", ""))
            expected_set_id = stable_id(
                "page_set_",
                {
                    "run_id": self.run_id,
                    "branch_id": self.branch_id,
                    "logical_group_id": logical_group_id,
                    "logical_group_digest": logical_digest,
                },
            )
            if page_set_id != expected_set_id:
                raise PageIntegrityError("WAL PageSet identity mismatch")
            directory = self._build_page_set_directory(logical_group, tuple(segments))
            for raw_segment, expected_entry in zip(raw_segments, directory, strict=True):
                raw_directory = raw_segment.get("directory")
                if raw_directory is not None and (
                    not isinstance(raw_directory, Mapping)
                    or dict(raw_directory) != primitive(expected_entry)
                ):
                    raise PageIntegrityError("WAL PageSet semantic directory mismatch")
            semantic_kinds = tuple(
                sorted({kind for entry in directory for kind in entry.semantic_kinds})
            )
            entity_refs = tuple(
                dict.fromkeys(entity for entry in directory for entity in entry.entity_refs)
            )[:64]
            synopsis_parts = [
                logical_group.group_type,
                "/".join(semantic_kinds) if semantic_kinds else "EXECUTE",
                f"{len(directory)} semantic segments",
            ]
            if logical_group.milestone_id:
                synopsis_parts.append(f"milestone {logical_group.milestone_id}")
            if entity_refs:
                synopsis_parts.append("entities " + ", ".join(entity_refs[:8]))
            synopsis = "; ".join(synopsis_parts)[:1600]
            supplied_directory = any(
                key in raw_set
                for key in ("synopsis", "semantic_kinds", "entity_refs", "directory_digest")
            )
            if supplied_directory and (
                raw_set.get("synopsis") != synopsis
                or tuple(map(str, raw_set.get("semantic_kinds", ()))) != semantic_kinds
                or tuple(map(str, raw_set.get("entity_refs", ()))) != entity_refs
                or raw_set.get("directory_digest") != digest(primitive(directory))
            ):
                raise PageIntegrityError("WAL PageSet synopsis or directory digest mismatch")
            return _PreparedPageSet(
                page_set_id=page_set_id,
                logical_group_id=logical_group_id,
                logical_group_digest=logical_digest,
                logical_group_blob=logical_blob,
                segments=tuple(segments),
                blobs=blobs,
                synopsis=synopsis,
                semantic_kinds=semantic_kinds,
                entity_refs=entity_refs,
                directory=directory,
            )

        group = EventGroup.from_dict(record["group"])
        self._validate_scope(group)
        group_digest = digest(group)
        if group_digest != record.get("group_digest"):
            raise PageIntegrityError("WAL EventGroup digest mismatch")
        proof = str(record.get("redaction_proof", ""))
        if proof != self.redactor.proof(primitive(group)):
            raise PageIntegrityError("WAL redaction proof mismatch")
        return _PreparedGroup(group, group_digest, proof, blobs)

    def _validated_wal_blobs(
        self,
        record: Mapping[str, Any],
    ) -> tuple[BlobReference, ...]:
        blobs = tuple(
            BlobReference(
                handle=str(item["handle"]),
                content_digest=str(item["content_digest"]),
                byte_range=(int(item["byte_range"][0]), int(item["byte_range"][1])),
                byte_count=int(item["byte_count"]),
                relative_path=str(item["relative_path"]),
            )
            for item in record.get("blobs", ())
        )
        for blob in blobs:
            path = self.root / blob.relative_path
            try:
                payload = path.read_bytes()
            except FileNotFoundError as exc:
                raise PageIntegrityError(f"WAL Blob missing: {blob.handle}") from exc
            if digest({"blob": payload.hex()}) != blob.content_digest:
                raise PageIntegrityError(f"WAL Blob digest mismatch: {blob.handle}")
        return blobs

    def _validate_page_document(self, row: sqlite3.Row, document: Mapping[str, Any]) -> None:
        if int(document.get("schema_version", -1)) != _PAGE_SCHEMA_VERSION:
            raise PageIntegrityError("unsupported Page schema")
        manifest = _page_manifest_from_dict(document["manifest"])
        database_manifest = _page_manifest_from_dict(json.loads(row["manifest_json"]))
        if manifest != database_manifest:
            raise PageIntegrityError(f"Page manifest mismatch: {row['page_id']}")
        groups = tuple(EventGroup.from_dict(item) for item in document["groups"])
        if digest({"groups": primitive(groups)}) != manifest.payload_digest:
            raise PageIntegrityError(f"Page payload mismatch: {manifest.page_id}")
        if tuple(group.group_id for group in groups) != manifest.event_group_ids:
            raise PageIntegrityError(f"Page group map mismatch: {manifest.page_id}")
        if sum(group.token_count for group in groups) != manifest.token_count:
            raise PageIntegrityError(f"Page token count mismatch: {manifest.page_id}")
        if manifest.token_count > self.policy.absolute_max_tokens and not (
            groups[-1].semantic_boundary
            or manifest.seal_reason in {item.value for item in LEGAL_TAIL_REASONS}
        ):
            raise PageIntegrityError(
                f"Page exceeds ABSOLUTE_MAX without a semantic boundary: {manifest.page_id}"
            )
        positions = document.get("event_positions", {})
        expected_event_ids = {event.event_id for group in groups for event in group.events}
        if set(positions) != expected_event_ids:
            raise PageIntegrityError(f"Page Event map mismatch: {manifest.page_id}")

    def _validate_blob_references(self, group: EventGroup) -> None:
        for event in group.events:
            references = [
                event.payload.get("external_payload"),
                event.payload.get("external_page_set_fragment"),
            ]
            references.extend(fact.content.get("external_fact") for fact in event.facts)
            for external in references:
                if not isinstance(external, Mapping):
                    continue
                handle = str(external.get("blob_handle", ""))
                raw_range = external.get("byte_range", ())
                if len(raw_range) != 2:
                    raise PageIntegrityError("invalid Blob reference")
                payload = self.open_blob(handle, (int(raw_range[0]), int(raw_range[1])))
                expected_digest = external.get("content_digest", handle)
                if digest({"blob": payload.hex()}) != expected_digest:
                    raise PageIntegrityError("Page/Blob reference digest mismatch")

    def _assert_page_visible(self, row: sqlite3.Row) -> None:
        if str(row["run_id"]) != self.run_id:
            raise BranchVisibilityError("Page belongs to a different Run")
        page_branch = str(row["branch_id"])
        if page_branch == self.branch_id:
            return
        if str(row["page_id"]) not in self._visible_parent_pages():
            raise BranchVisibilityError("Page is not visible from this branch")

    def _visible_parent_pages(self) -> tuple[str, ...]:
        visible: list[str] = []
        branch_id = self.branch_id
        seen: set[str] = set()
        while branch_id not in seen:
            seen.add(branch_id)
            branch = self.database.connection.execute(
                "SELECT parent_branch_id, parent_page_id FROM v2_page_branches "
                "WHERE run_id=? AND branch_id=?",
                (self.run_id, branch_id),
            ).fetchone()
            if branch is None or branch["parent_branch_id"] is None:
                break
            parent_branch = str(branch["parent_branch_id"])
            parent_page = str(branch["parent_page_id"])
            rows = self.database.connection.execute(
                "SELECT page_id FROM v2_pages WHERE run_id=? AND branch_id=? "
                "AND page_seq <= (SELECT page_seq FROM v2_pages WHERE page_id=?) "
                "ORDER BY page_seq",
                (self.run_id, parent_branch, parent_page),
            ).fetchall()
            visible = [str(row["page_id"]) for row in rows] + visible
            branch_id = parent_branch
        return tuple(visible)
