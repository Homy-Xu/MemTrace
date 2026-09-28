from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from ..contracts import (
    ENTITY_REVISION_FACT_TYPES,
    GLOBAL_REVISION_FACT_TYPES,
    SEMANTIC_PAGE_QUERY_RELATIONS,
    Authority,
    Event,
    EventGroup,
    EvidenceDraft,
    EvidenceKey,
    EvidenceUnit,
    FactType,
    FallbackStage,
    NodeType,
    PageCandidate,
    PageManifest,
    PlanSpec,
    PlanStepSpec,
    RecallIntent,
    SemanticAnchor,
    SemanticEdge,
    digest,
    primitive,
    stable_id,
)
from ..database import StateDatabase
from ..page_store.descriptor import describe_page, descriptor_payload
from ..page_store.synopsis import memory_ref_for_page
from ..planning.contracts import PlanProjectionInput
from ..references import ReferenceIdentityFactory
from .contracts import (
    EntityResolutionResult,
    ExactEvidenceHit,
    ExactLookupResult,
    FallbackQueryResult,
    PageProjectionReceipt,
)
from .ontology import EdgeRegistry, default_edge_registry

_ROUTE_MILESTONE_DETAIL_LIMIT = 24
_ROUTE_STEP_DETAIL_LIMIT = 12


def durable_reasoning_frontier(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    step_id: str,
    branch_id: str | None = None,
) -> tuple[tuple[str, str, str, str], ...]:
    """Return model-owned semantic deltas that advance the reasoning frontier.

    This is shared by runtime continuity and Batch supervision. Provider
    messages, tool activity, Page faults and generic progress narration are
    deliberately absent.
    """

    branch_clause = " AND branch_id=?" if branch_id is not None else ""
    parameters: tuple[object, ...] = (
        (run_id, branch_id, step_id) if branch_id is not None else (run_id, step_id)
    )
    rows = connection.execute(
        "SELECT evidence_type,semantic_role,content_digest,revision_id "
        "FROM v2_semantic_evidence WHERE run_id=?"
        f"{branch_clause} AND valid_to_cursor IS NULL "
        "AND json_extract(content_json,'$.plan_step_id')=? "
        "AND (evidence_type IN "
        "('IMPLEMENTATION_DECISION','UNRESOLVED_QUESTION','TEST_FAILURE') "
        "OR (evidence_type='CODE_OBSERVATION' AND semantic_role IN "
        "('agent_observation','rejected_hypothesis'))) "
        "ORDER BY evidence_type,semantic_role,content_digest,revision_id",
        parameters,
    ).fetchall()
    return tuple(
        (
            str(row["evidence_type"]),
            str(row["semantic_role"]),
            str(row["content_digest"]),
            str(row["revision_id"]),
        )
        for row in rows
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS v2_semantic_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO v2_semantic_meta(key, value) VALUES('schema_version', '2');

CREATE TABLE IF NOT EXISTS v2_semantic_clock (
    run_id TEXT PRIMARY KEY,
    next_cursor INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS v2_semantic_branches (
    repository_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    parent_branch_id TEXT,
    fork_cursor INTEGER,
    fork_revision_id TEXT,
    PRIMARY KEY(repository_id, run_id, branch_id),
    CHECK((parent_branch_id IS NULL) = (fork_cursor IS NULL)),
    CHECK(parent_branch_id IS NULL OR parent_branch_id <> branch_id)
);

CREATE TABLE IF NOT EXISTS v2_semantic_nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    repository_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    revision_id TEXT,
    authority TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    source_digest TEXT NOT NULL,
    created_cursor INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS v2_nodes_scope_type
ON v2_semantic_nodes(repository_id, run_id, branch_id, node_type, revision_id);

CREATE TABLE IF NOT EXISTS v2_semantic_edges (
    edge_id TEXT PRIMARY KEY,
    edge_type TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES v2_semantic_nodes(node_id),
    source_type TEXT NOT NULL,
    target_id TEXT NOT NULL REFERENCES v2_semantic_nodes(node_id),
    target_type TEXT NOT NULL,
    authority TEXT NOT NULL,
    repository_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    plan_version_id TEXT,
    valid_from_revision TEXT,
    valid_to_revision TEXT,
    valid_from_cursor INTEGER NOT NULL,
    valid_to_cursor INTEGER,
    provenance_json TEXT NOT NULL,
    supporting_event_ids_json TEXT NOT NULL,
    properties_json TEXT NOT NULL DEFAULT '{}',
    CHECK((valid_to_revision IS NULL) = (valid_to_cursor IS NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS v2_semantic_one_current
ON v2_semantic_edges(run_id, edge_type)
WHERE edge_type='CURRENT_MILESTONE' AND valid_to_cursor IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS v2_semantic_one_current_step
ON v2_semantic_edges(run_id, edge_type)
WHERE edge_type='CURRENT_STEP' AND valid_to_cursor IS NULL;
CREATE INDEX IF NOT EXISTS v2_edges_outgoing
ON v2_semantic_edges(repository_id, run_id, branch_id, source_id, edge_type, valid_to_cursor);
CREATE INDEX IF NOT EXISTS v2_edges_incoming
ON v2_semantic_edges(repository_id, run_id, branch_id, target_id, edge_type, valid_to_cursor);
CREATE INDEX IF NOT EXISTS v2_edges_plan_validity
ON v2_semantic_edges(run_id, plan_version_id, edge_type, valid_to_cursor, valid_from_cursor);

CREATE TABLE IF NOT EXISTS v2_semantic_pages (
    page_id TEXT PRIMARY KEY,
    repository_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    page_seq INTEGER NOT NULL,
    event_start INTEGER NOT NULL,
    event_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    byte_count INTEGER NOT NULL,
    payload_digest TEXT NOT NULL,
    redaction_proof TEXT NOT NULL,
    previous_page_id TEXT,
    seal_reason TEXT NOT NULL,
    page_kind TEXT NOT NULL,
    tail INTEGER NOT NULL,
    freshness_cursor INTEGER NOT NULL,
    source_digest TEXT NOT NULL,
    UNIQUE(run_id, branch_id, page_seq),
    CHECK(event_start <= event_end),
    CHECK(length(payload_digest) > 0),
    CHECK(length(redaction_proof) > 0)
);
CREATE INDEX IF NOT EXISTS v2_pages_scope_freshness
ON v2_semantic_pages(repository_id, run_id, branch_id, freshness_cursor DESC);
CREATE INDEX IF NOT EXISTS v2_pages_event_range
ON v2_semantic_pages(run_id, branch_id, event_start, event_end);

CREATE TABLE IF NOT EXISTS v2_semantic_page_descriptors (
    page_id TEXT PRIMARY KEY REFERENCES v2_semantic_pages(page_id),
    delta_summary TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK(outcome IN ('SUCCESS','PARTIAL','FAILED')),
    delta_kinds_json TEXT NOT NULL,
    changed_files_json TEXT NOT NULL,
    changed_symbols_json TEXT NOT NULL,
    test_refs_json TEXT NOT NULL,
    decision_refs_json TEXT NOT NULL,
    unresolved_refs_json TEXT NOT NULL,
    supporting_event_ids_json TEXT NOT NULL,
    relation_targets_json TEXT NOT NULL,
    descriptor_digest TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS v2_page_descriptor_outcome
ON v2_semantic_page_descriptors(outcome, page_id);

CREATE TABLE IF NOT EXISTS v2_semantic_page_revisions (
    page_id TEXT NOT NULL REFERENCES v2_semantic_pages(page_id),
    revision_id TEXT NOT NULL,
    PRIMARY KEY(page_id, revision_id)
);
CREATE INDEX IF NOT EXISTS v2_page_revision_lookup
ON v2_semantic_page_revisions(revision_id, page_id);

CREATE TABLE IF NOT EXISTS v2_semantic_page_milestones (
    page_id TEXT NOT NULL REFERENCES v2_semantic_pages(page_id),
    milestone_identity_id TEXT NOT NULL,
    PRIMARY KEY(page_id, milestone_identity_id)
);
CREATE INDEX IF NOT EXISTS v2_page_milestone_lookup
ON v2_semantic_page_milestones(milestone_identity_id, page_id);

CREATE TABLE IF NOT EXISTS v2_semantic_page_entities (
    page_id TEXT NOT NULL REFERENCES v2_semantic_pages(page_id),
    canonical_entity_id TEXT NOT NULL,
    reference_node_id TEXT NOT NULL,
    PRIMARY KEY(page_id, canonical_entity_id)
);
CREATE INDEX IF NOT EXISTS v2_page_entity_lookup
ON v2_semantic_page_entities(canonical_entity_id, page_id);

CREATE TABLE IF NOT EXISTS v2_semantic_entity_aliases (
    repository_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    canonical_entity_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    reference_node_id TEXT,
    PRIMARY KEY(repository_id, run_id, branch_id, canonical_entity_id, alias)
);
CREATE INDEX IF NOT EXISTS v2_entity_alias_lookup
ON v2_semantic_entity_aliases(run_id, branch_id, alias, canonical_entity_id);

CREATE TABLE IF NOT EXISTS v2_semantic_event_groups (
    group_id TEXT PRIMARY KEY,
    page_id TEXT NOT NULL REFERENCES v2_semantic_pages(page_id),
    repository_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    group_type TEXT NOT NULL,
    milestone_identity_id TEXT,
    event_start INTEGER NOT NULL,
    event_end INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    source_digest TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS v2_groups_page_range
ON v2_semantic_event_groups(page_id, event_start, event_end);

CREATE TABLE IF NOT EXISTS v2_semantic_events (
    event_id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL REFERENCES v2_semantic_event_groups(group_id),
    page_id TEXT NOT NULL REFERENCES v2_semantic_pages(page_id),
    repository_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    event_position INTEGER NOT NULL,
    event_end INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    milestone_identity_id TEXT,
    execution_phase TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    UNIQUE(run_id, branch_id, event_position)
);
CREATE INDEX IF NOT EXISTS v2_events_page_position
ON v2_semantic_events(page_id, event_position, event_end, revision_id);

CREATE TABLE IF NOT EXISTS v2_semantic_evidence (
    evidence_id TEXT PRIMARY KEY,
    key_digest TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    canonical_entity_id TEXT NOT NULL,
    semantic_role TEXT NOT NULL,
    revision_constraint TEXT NOT NULL,
    key_branch_scope TEXT NOT NULL,
    validity_requirement TEXT NOT NULL,
    content_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    authority TEXT NOT NULL,
    confidence REAL NOT NULL,
    must_preserve INTEGER NOT NULL,
    event_id TEXT NOT NULL REFERENCES v2_semantic_events(event_id),
    event_group_id TEXT NOT NULL REFERENCES v2_semantic_event_groups(group_id),
    page_id TEXT NOT NULL REFERENCES v2_semantic_pages(page_id),
    repository_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    valid_from_revision TEXT NOT NULL,
    valid_to_revision TEXT,
    valid_from_cursor INTEGER NOT NULL,
    valid_to_cursor INTEGER,
    CHECK(authority IN ('ASSERTED', 'DERIVED', 'INFERRED')),
    CHECK(confidence >= 0.0 AND confidence <= 1.0),
    CHECK((valid_to_revision IS NULL) = (valid_to_cursor IS NULL))
);
CREATE INDEX IF NOT EXISTS v2_evidence_exact_lookup
ON v2_semantic_evidence(
    repository_id, run_id, evidence_type, canonical_entity_id, semantic_role,
    revision_constraint, validity_requirement, branch_id, revision_id,
    valid_to_cursor, page_id
);
CREATE INDEX IF NOT EXISTS v2_evidence_key_lookup
ON v2_semantic_evidence(repository_id, run_id, key_digest, branch_id, revision_id, valid_to_cursor);

CREATE TABLE IF NOT EXISTS v2_semantic_anchors (
    anchor_id TEXT PRIMARY KEY,
    evidence_id TEXT NOT NULL UNIQUE REFERENCES v2_semantic_evidence(evidence_id),
    page_id TEXT NOT NULL REFERENCES v2_semantic_pages(page_id),
    page_digest TEXT NOT NULL,
    event_id TEXT NOT NULL REFERENCES v2_semantic_events(event_id),
    event_group_id TEXT NOT NULL REFERENCES v2_semantic_event_groups(group_id),
    event_start INTEGER NOT NULL,
    event_end INTEGER NOT NULL,
    blob_handle TEXT,
    blob_start INTEGER,
    blob_end INTEGER,
    revision_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    CHECK(event_start <= event_end),
    CHECK((blob_handle IS NULL) = (blob_start IS NULL)),
    CHECK((blob_handle IS NULL) = (blob_end IS NULL))
);
CREATE INDEX IF NOT EXISTS v2_anchor_page_range
ON v2_semantic_anchors(page_id, event_start, event_end, revision_id);

-- Logical PageSet: exactly one per Milestone terminal state.  Physical reasons
-- (WAL size, Epoch fences) only ever produce Pages/segments below this layer;
-- the PageSet boundary is the Milestone boundary.  Repairs of the same
-- Milestone identity produce a new PageSet that SUPERSEDES the previous one.
CREATE TABLE IF NOT EXISTS v2_semantic_milestone_page_sets (
    page_set_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    milestone_identity_id TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    plan_version_id TEXT NOT NULL,
    terminal_state TEXT NOT NULL CHECK(terminal_state IN (
        'COMPLETED_VERIFIED','VERIFICATION_FAILED','ROUTE_STALLED','TASK_TERMINATED'
    )),
    supersedes_page_set_id TEXT,
    wal_start INTEGER NOT NULL,
    wal_end INTEGER NOT NULL,
    page_ids_json TEXT NOT NULL,
    memory_refs_json TEXT NOT NULL,
    synopsis_json TEXT NOT NULL,
    synopsis_digest TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, milestone_identity_id, terminal_state, source_event_id),
    CHECK(wal_start <= wal_end)
);
CREATE INDEX IF NOT EXISTS v2_milestone_page_set_lookup
ON v2_semantic_milestone_page_sets(run_id, milestone_identity_id, created_cursor DESC);
"""


_REFERENCE_TYPES: dict[str, NodeType] = {
    "file": NodeType.FILE_REFERENCE,
    "symbol": NodeType.SYMBOL_REFERENCE,
    "test": NodeType.TEST_REFERENCE,
    "failure": NodeType.FAILURE_REFERENCE,
    "change": NodeType.CHANGE_REFERENCE,
}


class SemanticStore:
    """Indexed semantic address translation and deterministic graph projection."""

    def __init__(
        self,
        database: StateDatabase,
        edge_registry: EdgeRegistry | None = None,
    ) -> None:
        self.database = database
        self.edge_registry = edge_registry or default_edge_registry()
        self.fts_enabled = False
        with self.database.transaction() as conn:
            conn.executescript(_SCHEMA)
            edge_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(v2_semantic_edges)").fetchall()
            }
            if "properties_json" not in edge_columns:
                conn.execute(
                    "ALTER TABLE v2_semantic_edges "
                    "ADD COLUMN properties_json TEXT NOT NULL DEFAULT '{}'"
                )
            version = conn.execute(
                "SELECT value FROM v2_semantic_meta WHERE key='schema_version'"
            ).fetchone()
            if version is None or int(version["value"]) != 2:
                raise RuntimeError("unsupported Semantic Store schema")
            alias_backfill = conn.execute(
                "SELECT value FROM v2_semantic_meta WHERE key='entity_alias_backfill_v1'"
            ).fetchone()
            if alias_backfill is None:
                for row in conn.execute(
                    "SELECT DISTINCT repository_id,run_id,branch_id,canonical_entity_id "
                    "FROM v2_semantic_evidence"
                ).fetchall():
                    canonical = str(row["canonical_entity_id"])
                    parsed = self._reference(canonical)
                    try:
                        reference_node_id = (
                            self._reference_node_id(str(row["repository_id"]), canonical)
                            if parsed is not None
                            else None
                        )
                    except ValueError:
                        reference_node_id = None
                    for alias in self._entity_aliases(canonical):
                        conn.execute(
                            "INSERT OR IGNORE INTO v2_semantic_entity_aliases "
                            "(repository_id,run_id,branch_id,canonical_entity_id,alias,"
                            "reference_node_id) VALUES(?,?,?,?,?,?)",
                            (
                                str(row["repository_id"]),
                                str(row["run_id"]),
                                str(row["branch_id"]),
                                canonical,
                                alias,
                                reference_node_id,
                            ),
                        )
                conn.execute(
                    "INSERT INTO v2_semantic_meta(key,value) VALUES('entity_alias_backfill_v1','1')"
                )
            fts_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(v2_semantic_page_fts)").fetchall()
            }
            rebuild_fts = not fts_columns or "semantic_summary" not in fts_columns
            if fts_columns and "semantic_summary" not in fts_columns:
                conn.execute("DROP TABLE v2_semantic_page_fts")
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS v2_semantic_page_fts USING fts5("
                    "page_id UNINDEXED, repository_id UNINDEXED, run_id UNINDEXED, "
                    "branch_id UNINDEXED, revision_ids, milestone_ids, entity_refs, "
                    "evidence_keys, execution_phases, semantic_summary)"
                )
            except sqlite3.OperationalError:
                self.fts_enabled = False
            else:
                self.fts_enabled = True
                # Rebuild after an FTS schema migration so existing Page
                # addresses remain searchable. This reads indexed metadata;
                # it never opens authoritative Page bodies on the hot path.
                if rebuild_fts:
                    conn.execute("DELETE FROM v2_semantic_page_fts")
                    conn.execute(
                        "INSERT INTO v2_semantic_page_fts "
                        "SELECT p.page_id,p.repository_id,p.run_id,p.branch_id,"
                        "COALESCE((SELECT group_concat(revision_id,' ') "
                        "FROM v2_semantic_page_revisions r WHERE r.page_id=p.page_id),''),"
                        "COALESCE((SELECT group_concat(milestone_identity_id,' ') "
                        "FROM v2_semantic_page_milestones m WHERE m.page_id=p.page_id),''),"
                        "COALESCE((SELECT group_concat(canonical_entity_id,' ') "
                        "FROM v2_semantic_page_entities e WHERE e.page_id=p.page_id),''),"
                        "COALESCE((SELECT group_concat(key_digest,' ') "
                        "FROM v2_semantic_evidence e WHERE e.page_id=p.page_id),''),"
                        "COALESCE((SELECT group_concat(execution_phase,' ') "
                        "FROM v2_semantic_events e WHERE e.page_id=p.page_id),''),"
                        "COALESCE((SELECT delta_summary FROM v2_semantic_page_descriptors d "
                        "WHERE d.page_id=p.page_id),'') FROM v2_semantic_pages p"
                    )

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        """Use a WAL reader isolated from background writers' transaction state."""

        connection = sqlite3.connect(
            self.database.path,
            isolation_level=None,
            timeout=10.0,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _advance_clock(
        conn: sqlite3.Connection, run_id: str, external_cursor: int | None = None
    ) -> int:
        row = conn.execute(
            "SELECT next_cursor FROM v2_semantic_clock WHERE run_id=?", (run_id,)
        ).fetchone()
        prior = 0 if row is None else int(row["next_cursor"])
        cursor = max(prior + 1, external_cursor or 0)
        conn.execute(
            "INSERT INTO v2_semantic_clock(run_id, next_cursor) VALUES(?,?) "
            "ON CONFLICT(run_id) DO UPDATE SET next_cursor=excluded.next_cursor",
            (run_id, cursor),
        )
        return cursor

    @staticmethod
    def _put_node(
        conn: sqlite3.Connection,
        *,
        node_id: str,
        node_type: NodeType,
        repository_id: str,
        run_id: str,
        branch_id: str,
        revision_id: str | None,
        authority: Authority,
        payload: Mapping[str, Any],
        source_digest: str,
        cursor: int,
    ) -> bool:
        payload_json = json.dumps(
            primitive(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        payload_digest = digest(payload)
        inserted = conn.execute(
            "INSERT OR IGNORE INTO v2_semantic_nodes VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                node_id,
                node_type.value,
                repository_id,
                run_id,
                branch_id,
                revision_id,
                authority.value,
                payload_json,
                payload_digest,
                source_digest,
                cursor,
            ),
        ).rowcount
        if not inserted:
            row = conn.execute(
                "SELECT node_type, repository_id, run_id, branch_id, payload_digest "
                "FROM v2_semantic_nodes WHERE node_id=?",
                (node_id,),
            ).fetchone()
            expected = (
                node_type.value,
                repository_id,
                run_id,
                branch_id,
                payload_digest,
            )
            actual = tuple(row) if row is not None else None
            if actual != expected:
                raise ValueError(f"immutable semantic node collision: {node_id}")
        return bool(inserted)

    def _put_edge(
        self,
        conn: sqlite3.Connection,
        *,
        edge_type: str,
        source_id: str,
        source_type: NodeType,
        target_id: str,
        target_type: NodeType,
        authority: Authority,
        repository_id: str,
        run_id: str,
        branch_id: str,
        plan_version_id: str | None,
        revision_id: str | None,
        cursor: int,
        provenance: tuple[str, ...] = (),
        supporting_event_ids: tuple[str, ...] = (),
        properties: Mapping[str, Any] | None = None,
        critical_path: bool = False,
    ) -> bool:
        edge_id = stable_id(
            "edge_",
            {
                "type": edge_type,
                "source": source_id,
                "target": target_id,
                "run": run_id,
                "branch": branch_id,
                "plan": plan_version_id,
                "cursor": cursor,
                "properties": properties or {},
            },
        )
        edge = SemanticEdge(
            edge_id=edge_id,
            edge_type=edge_type,
            source_id=source_id,
            source_type=source_type,
            target_id=target_id,
            target_type=target_type,
            authority=authority,
            run_id=run_id,
            branch_id=branch_id,
            plan_version_id=plan_version_id,
            valid_from_revision=revision_id,
            valid_to_revision=None,
            valid_from_cursor=cursor,
            valid_to_cursor=None,
            provenance=provenance,
            supporting_event_ids=supporting_event_ids,
            properties=dict(properties or {}),
        )
        self.edge_registry.validate(edge, critical_path=critical_path)
        source = conn.execute(
            "SELECT node_type FROM v2_semantic_nodes WHERE node_id=?", (source_id,)
        ).fetchone()
        target = conn.execute(
            "SELECT node_type FROM v2_semantic_nodes WHERE node_id=?", (target_id,)
        ).fetchone()
        if source is None or source["node_type"] != source_type.value:
            raise ValueError(f"missing or mistyped edge source: {source_id}")
        if target is None or target["node_type"] != target_type.value:
            raise ValueError(f"missing or mistyped edge target: {target_id}")
        return bool(
            conn.execute(
                "INSERT OR IGNORE INTO v2_semantic_edges("
                "edge_id,edge_type,source_id,source_type,target_id,target_type,authority,"
                "repository_id,run_id,branch_id,plan_version_id,valid_from_revision,"
                "valid_to_revision,valid_from_cursor,valid_to_cursor,provenance_json,"
                "supporting_event_ids_json,properties_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    edge.edge_id,
                    edge.edge_type,
                    edge.source_id,
                    edge.source_type.value,
                    edge.target_id,
                    edge.target_type.value,
                    edge.authority.value,
                    repository_id,
                    edge.run_id,
                    edge.branch_id,
                    edge.plan_version_id,
                    edge.valid_from_revision,
                    edge.valid_to_revision,
                    edge.valid_from_cursor,
                    edge.valid_to_cursor,
                    json.dumps(edge.provenance),
                    json.dumps(edge.supporting_event_ids),
                    json.dumps(primitive(edge.properties), sort_keys=True),
                ),
            ).rowcount
        )

    @staticmethod
    def _fact_is_positive_criterion_evidence(
        fact: EvidenceDraft,
        required_evidence_types: tuple[str, ...],
    ) -> bool:
        """Keep the TPG factual edge consistent with acceptance truth.

        ``criterion_ids`` is an address binding, not by itself a successful
        verdict.  In particular, a failed test or a green-looking pipeline
        whose producer exit status is unknown must never be projected as
        ``SATISFIES``.  Milestone semantic review remains a later route-boundary
        decision; this predicate records only positive factual contribution.
        """

        evidence_type = fact.key.evidence_type.value
        if evidence_type not in set(required_evidence_types):
            return False
        if evidence_type == FactType.TEST_FAILURE.value:
            return True
        if evidence_type in {
            FactType.TEST_RESULT.value,
            FactType.VERIFIER_RESULT.value,
            FactType.TOOL_RESULT.value,
            FactType.REQUIREMENT_REVIEW.value,
        }:
            return fact.content.get("success") is True and (
                fact.content.get("success_exit_status_reliable") is not False
            )
        return True

    @staticmethod
    def _revision_node_id(repository_id: str, revision_id: str) -> str:
        return stable_id("rev_", {"repository": repository_id, "revision": revision_id})

    @staticmethod
    def _reference(canonical_entity_id: str) -> tuple[NodeType, str] | None:
        prefix, separator, _ = canonical_entity_id.partition(":")
        if not separator or prefix not in _REFERENCE_TYPES:
            return None
        return _REFERENCE_TYPES[prefix], prefix

    @staticmethod
    def _reference_node_id(repository_id: str, canonical_entity_id: str) -> str:
        return ReferenceIdentityFactory(repository_id).id_for(canonical_entity_id)

    @staticmethod
    def _entity_aliases(canonical_entity_id: str) -> tuple[str, ...]:
        """Return deterministic model-facing aliases for one canonical entity."""

        value = canonical_entity_id.strip()
        if not value:
            return ()
        aliases = [value]
        prefix, separator, suffix = value.partition(":")
        if not separator:
            return (value,)
        if prefix == "file":
            try:
                path = ReferenceIdentityFactory.normalize_path(suffix)
            except ValueError:
                return (value,)
            basename = path.rsplit("/", 1)[-1]
            aliases.extend((path, f"file:{basename}", basename))
        elif prefix == "symbol":
            path, symbol_separator, qualified_name = suffix.partition(":")
            if symbol_separator and qualified_name:
                try:
                    path = ReferenceIdentityFactory.normalize_path(path)
                except ValueError:
                    return (value,)
                basename = path.rsplit("/", 1)[-1]
                aliases.extend(
                    (
                        f"symbol:{qualified_name}",
                        qualified_name,
                        f"symbol:{basename}:{qualified_name}",
                    )
                )
        elif prefix == "test":
            aliases.append(suffix)
        return tuple(dict.fromkeys(alias for alias in aliases if alias))

    @staticmethod
    def _request_aliases(entity: str) -> tuple[str, ...]:
        """Normalize transport spellings without changing code-symbol case."""

        value = entity.strip().replace("\\", "/")
        if not value:
            return ()
        aliases = [value]
        prefix, separator, suffix = value.partition(":")
        try:
            if prefix == "file" and separator:
                normalized = ReferenceIdentityFactory.normalize_path(suffix)
                aliases.extend((normalized, f"file:{normalized}"))
            elif prefix == "symbol" and separator and ":" in suffix:
                path, qualified_name = suffix.split(":", 1)
                normalized = ReferenceIdentityFactory.normalize_path(path)
                aliases.append(f"symbol:{normalized}:{qualified_name}")
            elif not separator and "/" in value:
                aliases.append(ReferenceIdentityFactory.normalize_path(value))
        except ValueError:
            pass
        return tuple(dict.fromkeys(aliases))

    @staticmethod
    def _file_address_equivalence(entity: str) -> str | None:
        """Collapse only canonical and legacy spellings of the same full path."""

        value = entity.strip().replace("\\", "/")
        prefix, separator, suffix = value.partition(":")
        if separator and prefix == "file":
            path = suffix
        elif not separator and "/" in value:
            path = value
        else:
            return None
        try:
            return ReferenceIdentityFactory.normalize_path(path)
        except ValueError:
            return None

    @staticmethod
    def _ensure_branch(
        conn: sqlite3.Connection,
        *,
        repository_id: str,
        run_id: str,
        branch_id: str,
    ) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO v2_semantic_branches"
            "(repository_id, run_id, branch_id, parent_branch_id, fork_cursor, fork_revision_id) "
            "VALUES(?,?,?,NULL,NULL,NULL)",
            (repository_id, run_id, branch_id),
        )

    def register_branch(
        self,
        *,
        repository_id: str,
        run_id: str,
        branch_id: str,
        parent_branch_id: str,
        fork_cursor: int,
        fork_revision_id: str,
    ) -> None:
        if fork_cursor < 0:
            raise ValueError("fork_cursor cannot be negative")
        with self.database.transaction() as conn:
            existing = conn.execute(
                "SELECT parent_branch_id, fork_cursor, fork_revision_id "
                "FROM v2_semantic_branches "
                "WHERE repository_id=? AND run_id=? AND branch_id=?",
                (repository_id, run_id, branch_id),
            ).fetchone()
            if existing is not None:
                expected = (parent_branch_id, fork_cursor, fork_revision_id)
                actual = (
                    existing["parent_branch_id"],
                    existing["fork_cursor"],
                    existing["fork_revision_id"],
                )
                if actual != expected:
                    raise ValueError("branch ancestry is immutable")
                return
            parent = conn.execute(
                "SELECT 1 FROM v2_semantic_branches "
                "WHERE repository_id=? AND run_id=? AND branch_id=?",
                (repository_id, run_id, parent_branch_id),
            ).fetchone()
            if parent is None:
                raise KeyError("parent branch is not registered")
            conn.execute(
                "INSERT INTO v2_semantic_branches VALUES(?,?,?,?,?,?)",
                (
                    repository_id,
                    run_id,
                    branch_id,
                    parent_branch_id,
                    fork_cursor,
                    fork_revision_id,
                ),
            )

    def advance_workspace_revision(
        self,
        *,
        run_id: str,
        branch_id: str,
        new_revision_id: str,
        source_event_id: str,
        changed_entities: Sequence[str] = (),
    ) -> int:
        """Invalidate only revision-sensitive CURRENT evidence after a real change."""

        with self.database.transaction() as conn:
            task = conn.execute(
                "SELECT repository_id FROM v2_tasks WHERE run_id=? AND branch_id=?",
                (run_id, branch_id),
            ).fetchone()
            if task is None:
                raise KeyError(f"unknown semantic run/branch: {run_id}/{branch_id}")
            repository_id = str(task["repository_id"])
            cursor = self._advance_clock(conn, run_id)
            self._ensure_branch(
                conn,
                repository_id=repository_id,
                run_id=run_id,
                branch_id=branch_id,
            )
            self._put_node(
                conn,
                node_id=source_event_id,
                node_type=NodeType.EVENT,
                repository_id=repository_id,
                run_id=run_id,
                branch_id=branch_id,
                revision_id=new_revision_id,
                authority=Authority.ASSERTED,
                # The same durable Event is later projected from its sealed
                # Page. Keep the immutable node payload byte-identical so the
                # early revision invalidation and Page projection converge.
                payload={"event_id": source_event_id},
                source_digest=digest({"event": source_event_id, "revision": new_revision_id}),
                cursor=cursor,
            )
            revision_node = self._revision_node_id(repository_id, new_revision_id)
            self._put_node(
                conn,
                node_id=revision_node,
                node_type=NodeType.WORKSPACE_REVISION,
                repository_id=repository_id,
                run_id="*",
                branch_id="*",
                revision_id=new_revision_id,
                authority=Authority.ASSERTED,
                payload={"revision_id": new_revision_id},
                source_digest=digest({"revision": new_revision_id}),
                cursor=cursor,
            )
            self._put_edge(
                conn,
                edge_type="UPDATES",
                source_id=source_event_id,
                source_type=NodeType.EVENT,
                target_id=revision_node,
                target_type=NodeType.WORKSPACE_REVISION,
                authority=Authority.ASSERTED,
                repository_id=repository_id,
                run_id=run_id,
                branch_id=branch_id,
                plan_version_id=None,
                revision_id=new_revision_id,
                cursor=cursor,
                supporting_event_ids=(source_event_id,),
            )
            always_revision_sensitive = tuple(
                sorted(item.value for item in GLOBAL_REVISION_FACT_TYPES)
            )
            entity_sensitive = tuple(
                sorted(item.value for item in ENTITY_REVISION_FACT_TYPES)
            )
            entity_ids = tuple(dict.fromkeys(map(str, changed_entities)))
            selected_always = (
                always_revision_sensitive
                if entity_ids
                else (*always_revision_sensitive, *entity_sensitive)
            )
            placeholders = ",".join("?" for _ in selected_always)
            stale_rows = list(
                conn.execute(
                    "SELECT evidence_id,canonical_entity_id FROM v2_semantic_evidence "
                    "WHERE repository_id=? "
                    "AND run_id=? AND branch_id=? AND revision_id<>? "
                    "AND validity_requirement='CURRENT' AND valid_to_cursor IS NULL "
                    f"AND evidence_type IN ({placeholders})",
                    (
                        repository_id,
                        run_id,
                        branch_id,
                        new_revision_id,
                        *selected_always,
                    ),
                ).fetchall()
            )
            if entity_ids:
                candidates = conn.execute(
                    "SELECT evidence_id,canonical_entity_id FROM v2_semantic_evidence "
                    "WHERE repository_id=? "
                    "AND run_id=? AND branch_id=? AND revision_id<>? "
                    "AND validity_requirement='CURRENT' AND valid_to_cursor IS NULL "
                    "AND evidence_type IN ("
                    + ",".join("?" for _ in entity_sensitive)
                    + ")",
                    (
                        repository_id,
                        run_id,
                        branch_id,
                        new_revision_id,
                        *entity_sensitive,
                    ),
                ).fetchall()
                stale_rows.extend(
                    row
                    for row in candidates
                    if self._entity_affected(str(row["canonical_entity_id"]), entity_ids)
                )
            stale = {str(row["evidence_id"]): row for row in stale_rows}
            for evidence_id in stale:
                conn.execute(
                    "UPDATE v2_semantic_evidence SET valid_to_revision=?,valid_to_cursor=? "
                    "WHERE evidence_id=? AND valid_to_cursor IS NULL",
                    (new_revision_id, cursor, evidence_id),
                )
                conn.execute(
                    "UPDATE v2_semantic_edges SET valid_to_revision=?,valid_to_cursor=? "
                    "WHERE source_id=? AND run_id=? AND branch_id=? AND valid_to_cursor IS NULL",
                    (new_revision_id, cursor, evidence_id, run_id, branch_id),
                )
                self._put_edge(
                    conn,
                    edge_type="INVALIDATED_BY",
                    source_id=evidence_id,
                    source_type=NodeType.EVIDENCE_UNIT,
                    target_id=source_event_id,
                    target_type=NodeType.EVENT,
                    authority=Authority.ASSERTED,
                    repository_id=repository_id,
                    run_id=run_id,
                    branch_id=branch_id,
                    plan_version_id=None,
                    revision_id=new_revision_id,
                    cursor=cursor,
                    supporting_event_ids=(source_event_id,),
                )
            return len(stale)

    @staticmethod
    def _entity_affected(
        canonical_entity_id: str,
        changed_entities: Sequence[str],
    ) -> bool:
        """Match a changed FileReference to its shared Symbol/Test identities."""

        for changed in changed_entities:
            if canonical_entity_id == changed:
                return True
            if not changed.startswith("file:"):
                continue
            path = changed.removeprefix("file:")
            if canonical_entity_id.startswith(f"symbol:{path}:"):
                return True
            if canonical_entity_id.startswith(f"test:{path}"):
                return True
        return False

    def project_plan(self, conn: sqlite3.Connection, update: PlanProjectionInput) -> None:
        """Deterministic Plan projection; caller owns the Registry transaction."""

        self._ensure_branch(
            conn,
            repository_id=update.repository_id,
            run_id=update.run_id,
            branch_id=update.branch_id,
        )
        cursor = self._advance_clock(conn, update.run_id, update.cursor)
        source = digest(
            {
                "plan_version_id": update.plan_version_id,
                "event": update.source_event_id,
                "revision": update.revision_id,
            }
        )
        common = {
            "repository_id": update.repository_id,
            "run_id": update.run_id,
            "branch_id": update.branch_id,
            "revision_id": update.revision_id,
            "cursor": cursor,
        }
        run_node_id = stable_id("run_", {"run": update.run_id})
        self._put_node(
            conn,
            node_id=run_node_id,
            node_type=NodeType.RUN,
            authority=Authority.ASSERTED,
            payload={"run_id": update.run_id},
            source_digest=source,
            **common,
        )
        task_row = conn.execute(
            "SELECT user_request FROM v2_tasks WHERE task_id=?", (update.task_id,)
        ).fetchone()
        self._put_node(
            conn,
            node_id=update.task_id,
            node_type=NodeType.TASK,
            authority=Authority.ASSERTED,
            payload={"user_request": task_row["user_request"] if task_row else ""},
            source_digest=source,
            **common,
        )
        self._put_node(
            conn,
            node_id=update.goal_id,
            node_type=NodeType.GOAL,
            authority=Authority.ASSERTED,
            payload={"goal": update.plan.goal},
            source_digest=source,
            **common,
        )
        self._put_node(
            conn,
            node_id=update.plan_version_id,
            node_type=NodeType.PLAN_VERSION,
            authority=Authority.ASSERTED,
            payload={
                "version_number": update.plan_version_number,
                "plan_digest": digest(update.plan),
            },
            source_digest=source,
            **common,
        )
        self._put_node(
            conn,
            node_id=update.source_event_id,
            node_type=NodeType.EVENT,
            authority=Authority.ASSERTED,
            payload={"event_id": update.source_event_id},
            source_digest=digest({"event_id": update.source_event_id}),
            **common,
        )
        revision_node = self._revision_node_id(update.repository_id, update.revision_id)
        self._put_node(
            conn,
            node_id=revision_node,
            node_type=NodeType.WORKSPACE_REVISION,
            repository_id=update.repository_id,
            run_id="*",
            branch_id="*",
            revision_id=update.revision_id,
            cursor=cursor,
            authority=Authority.ASSERTED,
            payload={"revision_id": update.revision_id},
            source_digest=digest({"revision": update.revision_id}),
        )

        asserted_edges = (
            ("HAS_GOAL", update.task_id, NodeType.TASK, update.goal_id, NodeType.GOAL),
            (
                "HAS_PLAN",
                update.task_id,
                NodeType.TASK,
                update.plan_version_id,
                NodeType.PLAN_VERSION,
            ),
        )
        for edge_type, source_id, source_type, target_id, target_type in asserted_edges:
            self._put_edge(
                conn,
                edge_type=edge_type,
                source_id=source_id,
                source_type=source_type,
                target_id=target_id,
                target_type=target_type,
                authority=Authority.ASSERTED,
                plan_version_id=update.plan_version_id,
                supporting_event_ids=(update.source_event_id,),
                **common,
            )
        if update.previous_plan_version_id:
            self._put_edge(
                conn,
                edge_type="SUPERSEDES",
                source_id=update.plan_version_id,
                source_type=NodeType.PLAN_VERSION,
                target_id=update.previous_plan_version_id,
                target_type=NodeType.PLAN_VERSION,
                authority=Authority.ASSERTED,
                plan_version_id=update.plan_version_id,
                supporting_event_ids=(update.source_event_id,),
                **common,
            )

        by_canonical = {item.spec.canonical_id: item for item in update.milestones}
        native_node_ids: dict[str, str] = {}
        if update.plan.native_plan is not None:
            for native_item in update.plan.native_plan.items:
                native_node_id = stable_id(
                    "nativeplan_",
                    {"run": update.run_id, "source_step_id": native_item.source_step_id},
                )
                native_node_ids[native_item.source_step_id] = native_node_id
                self._put_node(
                    conn,
                    node_id=native_node_id,
                    node_type=NodeType.NATIVE_PLAN_ITEM,
                    authority=Authority.ASSERTED,
                    payload=primitive(native_item),
                    source_digest=digest(native_item),
                    **common,
                )
                self._put_edge(
                    conn,
                    edge_type="HAS_NATIVE_ITEM",
                    source_id=update.plan_version_id,
                    source_type=NodeType.PLAN_VERSION,
                    target_id=native_node_id,
                    target_type=NodeType.NATIVE_PLAN_ITEM,
                    authority=Authority.ASSERTED,
                    plan_version_id=update.plan_version_id,
                    supporting_event_ids=(update.source_event_id,),
                    **common,
                )
        for record in update.milestones:
            self._put_node(
                conn,
                node_id=record.identity_id,
                node_type=NodeType.MILESTONE_IDENTITY,
                authority=Authority.ASSERTED,
                payload={"canonical_id": record.spec.canonical_id},
                source_digest=digest({"run": update.run_id, "canonical": record.spec.canonical_id}),
                **common,
            )
            if record.created_new_version:
                self._put_node(
                    conn,
                    node_id=record.version_id,
                    node_type=NodeType.MILESTONE_VERSION,
                    authority=Authority.ASSERTED,
                    payload={
                        "canonical_id": record.spec.canonical_id,
                        "version_number": record.version_number,
                        "spec": primitive(record.spec),
                    },
                    source_digest=digest(record.spec),
                    **common,
                )
            else:
                # A PlanVersion may reuse an immutable MilestoneVersion while
                # the separate execution state has advanced. Re-emitting a
                # node from the current status would turn state into versioned
                # structure and collide with its original immutable payload.
                existing_version = conn.execute(
                    "SELECT node_type FROM v2_semantic_nodes WHERE node_id=?",
                    (record.version_id,),
                ).fetchone()
                if (
                    existing_version is None
                    or str(existing_version["node_type"]) != NodeType.MILESTONE_VERSION.value
                ):
                    raise RuntimeError("reused MilestoneVersion is absent from Semantic Graph")
            for edge_type, target_id, target_type in (
                ("CONTAINS", record.identity_id, NodeType.MILESTONE_IDENTITY),
                ("HAS_VERSION", record.version_id, NodeType.MILESTONE_VERSION),
            ):
                source_id = (
                    update.plan_version_id if edge_type == "CONTAINS" else record.identity_id
                )
                source_type = (
                    NodeType.PLAN_VERSION
                    if edge_type == "CONTAINS"
                    else NodeType.MILESTONE_IDENTITY
                )
                self._put_edge(
                    conn,
                    edge_type=edge_type,
                    source_id=source_id,
                    source_type=source_type,
                    target_id=target_id,
                    target_type=target_type,
                    authority=Authority.ASSERTED,
                    plan_version_id=update.plan_version_id,
                    supporting_event_ids=(update.source_event_id,),
                    **common,
                )
            for source_plan_item_id in record.spec.source_plan_item_ids:
                native_node_id = native_node_ids.get(source_plan_item_id)
                if native_node_id is None:
                    raise RuntimeError(
                        "Milestone projection lost its native Plan source item"
                    )
                self._put_edge(
                    conn,
                    edge_type="PROJECTS_TO",
                    source_id=native_node_id,
                    source_type=NodeType.NATIVE_PLAN_ITEM,
                    target_id=record.identity_id,
                    target_type=NodeType.MILESTONE_IDENTITY,
                    authority=Authority.ASSERTED,
                    plan_version_id=update.plan_version_id,
                    supporting_event_ids=(update.source_event_id,),
                    **common,
                )
            if record.created_new_version:
                if record.previous_version_id:
                    self._put_edge(
                        conn,
                        edge_type="SUPERSEDES",
                        source_id=record.version_id,
                        source_type=NodeType.MILESTONE_VERSION,
                        target_id=record.previous_version_id,
                        target_type=NodeType.MILESTONE_VERSION,
                        authority=Authority.ASSERTED,
                        plan_version_id=update.plan_version_id,
                        supporting_event_ids=(update.source_event_id,),
                        **common,
                    )
                self._put_edge(
                    conn,
                    edge_type="UPDATES",
                    source_id=update.source_event_id,
                    source_type=NodeType.EVENT,
                    target_id=record.version_id,
                    target_type=NodeType.MILESTONE_VERSION,
                    authority=Authority.ASSERTED,
                    plan_version_id=update.plan_version_id,
                    supporting_event_ids=(update.source_event_id,),
                    **common,
                )
            for step in record.spec.steps:
                step_identity_id = stable_id(
                    "step_",
                    {"run": update.run_id, "canonical_step_id": step.step_id},
                )
                self._put_node(
                    conn,
                    node_id=step_identity_id,
                    node_type=NodeType.PLAN_STEP,
                    authority=Authority.ASSERTED,
                    payload={
                        "step_id": step.step_id,
                        "title": step.title,
                        "corrective": step.corrective,
                        "criterion_ids": step.criterion_ids,
                        "entity_refs": step.entity_refs,
                        "expected_outcome": step.expected_outcome,
                        "minimum_acceptance": primitive(step.minimum_acceptance),
                        "failure_signals": step.failure_signals,
                        "source_plan_item_ids": step.source_plan_item_ids,
                    },
                    source_digest=digest(step),
                    **common,
                )
                self._put_edge(
                    conn,
                    edge_type="HAS_STEP",
                    source_id=record.identity_id,
                    source_type=NodeType.MILESTONE_IDENTITY,
                    target_id=step_identity_id,
                    target_type=NodeType.PLAN_STEP,
                    authority=Authority.ASSERTED,
                    plan_version_id=update.plan_version_id,
                    supporting_event_ids=(update.source_event_id,),
                    **common,
                )
                for source_plan_item_id in step.source_plan_item_ids:
                    native_node_id = native_node_ids.get(source_plan_item_id)
                    if native_node_id is None:
                        raise RuntimeError("PlanStep projection lost its native Plan source item")
                    self._put_edge(
                        conn,
                        edge_type="PROJECTS_TO",
                        source_id=native_node_id,
                        source_type=NodeType.NATIVE_PLAN_ITEM,
                        target_id=step_identity_id,
                        target_type=NodeType.PLAN_STEP,
                        authority=Authority.ASSERTED,
                        plan_version_id=update.plan_version_id,
                        supporting_event_ids=(update.source_event_id,),
                        **common,
                    )
            for criterion in record.spec.criteria:
                criterion_identity_id = stable_id(
                    "criterion_",
                    {
                        "milestone_version": record.version_id,
                        "criterion": criterion.criterion_id,
                    },
                )
                self._put_node(
                    conn,
                    node_id=criterion_identity_id,
                    node_type=NodeType.COMPLETION_CRITERION,
                    authority=Authority.ASSERTED,
                    payload=primitive(criterion),
                    source_digest=digest(criterion),
                    **common,
                )
                self._put_edge(
                    conn,
                    edge_type="HAS_CRITERION",
                    source_id=record.version_id,
                    source_type=NodeType.MILESTONE_VERSION,
                    target_id=criterion_identity_id,
                    target_type=NodeType.COMPLETION_CRITERION,
                    authority=Authority.ASSERTED,
                    plan_version_id=update.plan_version_id,
                    supporting_event_ids=(update.source_event_id,),
                    **common,
                )

        conn.execute(
            "UPDATE v2_semantic_edges SET valid_to_revision=?, valid_to_cursor=? "
            "WHERE run_id=? AND edge_type IN ('DEPENDS_ON','PRECEDES') "
            "AND valid_to_cursor IS NULL AND plan_version_id<>?",
            (update.revision_id, cursor, update.run_id, update.plan_version_id),
        )
        for index, record in enumerate(update.milestones):
            for dependency in record.spec.depends_on:
                self._put_edge(
                    conn,
                    edge_type="DEPENDS_ON",
                    source_id=record.identity_id,
                    source_type=NodeType.MILESTONE_IDENTITY,
                    target_id=by_canonical[dependency].identity_id,
                    target_type=NodeType.MILESTONE_IDENTITY,
                    authority=Authority.ASSERTED,
                    plan_version_id=update.plan_version_id,
                    supporting_event_ids=(update.source_event_id,),
                    **common,
                )
            if index + 1 < len(update.milestones):
                self._put_edge(
                    conn,
                    edge_type="PRECEDES",
                    source_id=record.identity_id,
                    source_type=NodeType.MILESTONE_IDENTITY,
                    target_id=update.milestones[index + 1].identity_id,
                    target_type=NodeType.MILESTONE_IDENTITY,
                    authority=Authority.DERIVED,
                    plan_version_id=update.plan_version_id,
                    provenance=("plan_milestone_ordinal", update.source_event_id),
                    supporting_event_ids=(update.source_event_id,),
                    **common,
                )
        self._put_edge(
            conn,
            edge_type="UPDATES",
            source_id=update.source_event_id,
            source_type=NodeType.EVENT,
            target_id=revision_node,
            target_type=NodeType.WORKSPACE_REVISION,
            authority=Authority.ASSERTED,
            plan_version_id=update.plan_version_id,
            supporting_event_ids=(update.source_event_id,),
            **common,
        )

    def project_current_milestone(
        self,
        conn: sqlite3.Connection,
        *,
        repository_id: str,
        run_id: str,
        branch_id: str,
        revision_id: str,
        plan_version_id: str,
        milestone_identity_id: str,
        source_event_id: str,
        cursor: int,
    ) -> None:
        semantic_cursor = self._advance_clock(conn, run_id, cursor)
        self._put_node(
            conn,
            node_id=source_event_id,
            node_type=NodeType.EVENT,
            repository_id=repository_id,
            run_id=run_id,
            branch_id=branch_id,
            revision_id=revision_id,
            authority=Authority.ASSERTED,
            payload={"event_id": source_event_id},
            source_digest=digest({"event_id": source_event_id}),
            cursor=semantic_cursor,
        )
        conn.execute(
            "UPDATE v2_semantic_edges SET valid_to_revision=?, valid_to_cursor=? "
            "WHERE run_id=? AND edge_type='CURRENT_MILESTONE' AND valid_to_cursor IS NULL",
            (revision_id, semantic_cursor, run_id),
        )
        run_node_id = stable_id("run_", {"run": run_id})
        self._put_edge(
            conn,
            edge_type="CURRENT_MILESTONE",
            source_id=run_node_id,
            source_type=NodeType.RUN,
            target_id=milestone_identity_id,
            target_type=NodeType.MILESTONE_IDENTITY,
            authority=Authority.ASSERTED,
            repository_id=repository_id,
            run_id=run_id,
            branch_id=branch_id,
            plan_version_id=plan_version_id,
            revision_id=revision_id,
            cursor=semantic_cursor,
            supporting_event_ids=(source_event_id,),
        )
        version = conn.execute(
            "SELECT milestone_version_id FROM v2_plan_milestones "
            "WHERE plan_version_id=? AND identity_id=?",
            (plan_version_id, milestone_identity_id),
        ).fetchone()
        if version is None:
            raise ValueError("current Milestone has no version in PlanVersion")
        target_version_id = str(version["milestone_version_id"])
        existing_update = conn.execute(
            "SELECT 1 FROM v2_semantic_edges WHERE edge_type='UPDATES' "
            "AND source_id=? AND target_id=? AND run_id=? LIMIT 1",
            (source_event_id, target_version_id, run_id),
        ).fetchone()
        if existing_update is None:
            self._put_edge(
                conn,
                edge_type="UPDATES",
                source_id=source_event_id,
                source_type=NodeType.EVENT,
                target_id=target_version_id,
                target_type=NodeType.MILESTONE_VERSION,
                authority=Authority.ASSERTED,
                repository_id=repository_id,
                run_id=run_id,
                branch_id=branch_id,
                plan_version_id=plan_version_id,
                revision_id=revision_id,
                cursor=semantic_cursor,
                supporting_event_ids=(source_event_id,),
            )

    def project_plan_step(
        self,
        conn: sqlite3.Connection,
        *,
        repository_id: str,
        run_id: str,
        branch_id: str,
        revision_id: str,
        plan_version_id: str,
        milestone_identity_id: str,
        step_identity_id: str,
        step: PlanStepSpec,
        source_event_id: str,
        cursor: int,
    ) -> None:
        """Project a runtime-added local Step without creating a PlanVersion."""

        semantic_cursor = self._advance_clock(conn, run_id, cursor)
        common = {
            "repository_id": repository_id,
            "run_id": run_id,
            "branch_id": branch_id,
            "revision_id": revision_id,
            "cursor": semantic_cursor,
        }
        self._put_node(
            conn,
            node_id=source_event_id,
            node_type=NodeType.EVENT,
            authority=Authority.ASSERTED,
            payload={"event_id": source_event_id},
            source_digest=digest({"event_id": source_event_id}),
            **common,
        )
        self._put_node(
            conn,
            node_id=step_identity_id,
            node_type=NodeType.PLAN_STEP,
            authority=Authority.ASSERTED,
            payload={
                "step_id": step.step_id,
                "title": step.title,
                "corrective": step.corrective,
                "criterion_ids": step.criterion_ids,
                "entity_refs": step.entity_refs,
                "expected_outcome": step.expected_outcome,
                "minimum_acceptance": primitive(step.minimum_acceptance),
                "failure_signals": step.failure_signals,
                "source_plan_item_ids": step.source_plan_item_ids,
            },
            source_digest=digest(step),
            **common,
        )
        self._put_edge(
            conn,
            edge_type="HAS_STEP",
            source_id=milestone_identity_id,
            source_type=NodeType.MILESTONE_IDENTITY,
            target_id=step_identity_id,
            target_type=NodeType.PLAN_STEP,
            authority=Authority.ASSERTED,
            repository_id=repository_id,
            run_id=run_id,
            branch_id=branch_id,
            plan_version_id=plan_version_id,
            revision_id=revision_id,
            cursor=semantic_cursor,
            supporting_event_ids=(source_event_id,),
        )
        for source_plan_item_id in step.source_plan_item_ids:
            native_node_id = stable_id(
                "nativeplan_",
                {"run": run_id, "source_step_id": source_plan_item_id},
            )
            existing = conn.execute(
                "SELECT 1 FROM v2_semantic_nodes WHERE node_id=? AND node_type=?",
                (native_node_id, NodeType.NATIVE_PLAN_ITEM.value),
            ).fetchone()
            if existing is None:
                raise RuntimeError("runtime PlanStep lost its native Plan source item")
            self._put_edge(
                conn,
                edge_type="PROJECTS_TO",
                source_id=native_node_id,
                source_type=NodeType.NATIVE_PLAN_ITEM,
                target_id=step_identity_id,
                target_type=NodeType.PLAN_STEP,
                authority=Authority.ASSERTED,
                repository_id=repository_id,
                run_id=run_id,
                branch_id=branch_id,
                plan_version_id=plan_version_id,
                revision_id=revision_id,
                cursor=semantic_cursor,
                supporting_event_ids=(source_event_id,),
            )

    def project_current_step(
        self,
        conn: sqlite3.Connection,
        *,
        repository_id: str,
        run_id: str,
        branch_id: str,
        revision_id: str,
        plan_version_id: str,
        milestone_identity_id: str,
        step_identity_id: str | None,
        source_event_id: str,
        cursor: int,
    ) -> None:
        semantic_cursor = self._advance_clock(conn, run_id, cursor)
        conn.execute(
            "UPDATE v2_semantic_edges SET valid_to_revision=?,valid_to_cursor=? "
            "WHERE run_id=? AND edge_type='CURRENT_STEP' AND valid_to_cursor IS NULL",
            (revision_id, semantic_cursor, run_id),
        )
        if step_identity_id is None:
            return
        self._put_edge(
            conn,
            edge_type="CURRENT_STEP",
            source_id=milestone_identity_id,
            source_type=NodeType.MILESTONE_IDENTITY,
            target_id=step_identity_id,
            target_type=NodeType.PLAN_STEP,
            authority=Authority.ASSERTED,
            repository_id=repository_id,
            run_id=run_id,
            branch_id=branch_id,
            plan_version_id=plan_version_id,
            revision_id=revision_id,
            cursor=semantic_cursor,
            supporting_event_ids=(source_event_id,),
            critical_path=True,
        )

    def project_step_correction(
        self,
        conn: sqlite3.Connection,
        *,
        repository_id: str,
        run_id: str,
        branch_id: str,
        revision_id: str,
        plan_version_id: str,
        failed_step_identity_id: str,
        corrective_step_identity_id: str,
        source_event_id: str,
        cursor: int,
    ) -> None:
        semantic_cursor = self._advance_clock(conn, run_id, cursor)
        self._put_edge(
            conn,
            edge_type="HAS_CORRECTIVE_STEP",
            source_id=failed_step_identity_id,
            source_type=NodeType.PLAN_STEP,
            target_id=corrective_step_identity_id,
            target_type=NodeType.PLAN_STEP,
            authority=Authority.ASSERTED,
            repository_id=repository_id,
            run_id=run_id,
            branch_id=branch_id,
            plan_version_id=plan_version_id,
            revision_id=revision_id,
            cursor=semantic_cursor,
            supporting_event_ids=(source_event_id,),
        )

    def project_step_contract_revision(
        self,
        conn: sqlite3.Connection,
        *,
        repository_id: str,
        run_id: str,
        branch_id: str,
        revision_id: str,
        plan_version_id: str,
        step_identity_id: str,
        source_event_id: str,
        cursor: int,
    ) -> None:
        """Project one confirmed local-contract revision onto the same Step node."""

        semantic_cursor = self._advance_clock(conn, run_id, cursor)
        self._put_node(
            conn,
            node_id=source_event_id,
            node_type=NodeType.EVENT,
            authority=Authority.ASSERTED,
            payload={"event_id": source_event_id, "kind": "STEP_ACCEPTANCE_REVISED"},
            source_digest=digest(
                {"event_id": source_event_id, "kind": "STEP_ACCEPTANCE_REVISED"}
            ),
            repository_id=repository_id,
            run_id=run_id,
            branch_id=branch_id,
            revision_id=revision_id,
            cursor=semantic_cursor,
        )
        self._put_edge(
            conn,
            edge_type="UPDATES",
            source_id=source_event_id,
            source_type=NodeType.EVENT,
            target_id=step_identity_id,
            target_type=NodeType.PLAN_STEP,
            authority=Authority.ASSERTED,
            repository_id=repository_id,
            run_id=run_id,
            branch_id=branch_id,
            plan_version_id=plan_version_id,
            revision_id=revision_id,
            cursor=semantic_cursor,
            supporting_event_ids=(source_event_id,),
        )

    @staticmethod
    def _event_range(
        event_id: str,
        event_positions: Mapping[str, tuple[int, int] | int],
    ) -> tuple[int, int]:
        if event_id not in event_positions:
            raise ValueError(f"event_positions is missing {event_id}")
        raw = event_positions[event_id]
        if isinstance(raw, int):
            result = (raw, raw)
        else:
            if len(raw) != 2:
                raise ValueError("event range must contain start and end")
            result = (int(raw[0]), int(raw[1]))
        if result[0] < 0 or result[0] > result[1]:
            raise ValueError(f"invalid event range for {event_id}: {result}")
        return result

    @staticmethod
    def _milestone_identity(
        conn: sqlite3.Connection, run_id: str, milestone_id: str | None
    ) -> str | None:
        if not milestone_id:
            return None
        row = conn.execute(
            "SELECT identity_id FROM v2_milestone_identities "
            "WHERE run_id=? AND (canonical_id=? OR identity_id=?)",
            (run_id, milestone_id, milestone_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"Page references unknown Milestone: {milestone_id}")
        return str(row["identity_id"])

    def _ensure_reference_node(
        self,
        conn: sqlite3.Connection,
        *,
        repository_id: str,
        run_id: str,
        branch_id: str,
        revision_id: str,
        canonical_entity_id: str,
        cursor: int,
    ) -> tuple[NodeType, str] | None:
        parsed = self._reference(canonical_entity_id)
        if parsed is None:
            return None
        node_type, _ = parsed
        try:
            node_id = self._reference_node_id(repository_id, canonical_entity_id)
        except ValueError:
            # A model-authored address outside the repository (for example
            # ``file:/e2e_workspace/TASK_QUEUE.md`` in the SWE-Milestone
            # orchestrator prompt) is not a shared Reference.  It stays an
            # opaque entity on the Page instead of aborting Page projection.
            return None
        self._put_node(
            conn,
            node_id=node_id,
            node_type=node_type,
            repository_id=repository_id,
            run_id="*",
            branch_id="*",
            revision_id=None,
            authority=Authority.ASSERTED,
            payload={"canonical_entity_id": canonical_entity_id},
            source_digest=digest({"repository": repository_id, "entity": canonical_entity_id}),
            cursor=cursor,
        )
        return node_type, node_id

    @staticmethod
    def _blob_location(
        event: Event, fact: EvidenceDraft | None = None
    ) -> tuple[str | None, tuple[int, int] | None]:
        payload: Mapping[str, Any] = fact.content if fact is not None else event.payload
        nested = (
            payload.get("external_fact")
            or payload.get("external_payload")
            or payload.get("payload_handle")
        )
        if isinstance(nested, Mapping):
            payload = nested
        handle = payload.get("blob_handle") or payload.get("handle")
        raw_range = payload.get("blob_range") or payload.get("byte_range")
        if handle is None:
            return None, None
        if not isinstance(raw_range, Sequence) or isinstance(raw_range, (str, bytes)):
            return str(handle), (0, 0)
        if len(raw_range) != 2:
            raise ValueError("Blob range must contain start and end")
        blob_range = (int(raw_range[0]), int(raw_range[1]))
        if blob_range[0] < 0 or blob_range[1] < blob_range[0]:
            raise ValueError("invalid Blob byte range")
        return str(handle), blob_range

    def project_page(
        self,
        conn: sqlite3.Connection,
        manifest: PageManifest,
        groups: tuple[EventGroup, ...],
        event_positions: Mapping[str, tuple[int, int] | int],
    ) -> PageProjectionReceipt:
        """Project a durable Page inside the Page Store's short commit transaction.

        The method never commits and never reads a Page body. Any validation error
        propagates so the Page Map transaction can roll back and recovery can retry.
        """

        if not groups:
            raise ValueError("Semantic Page projection requires complete EventGroups")
        if tuple(group.group_id for group in groups) != manifest.event_group_ids:
            raise ValueError("Manifest/EventGroup order mismatch")
        if any(
            group.run_id != manifest.run_id
            or group.branch_id != manifest.branch_id
            or not group.complete
            for group in groups
        ):
            raise ValueError("Page contains an incomplete or out-of-scope EventGroup")
        if not manifest.payload_digest or not manifest.redaction_proof:
            raise ValueError("Page digest and redaction proof are mandatory")
        task = conn.execute(
            "SELECT repository_id FROM v2_tasks WHERE run_id=?",
            (manifest.run_id,),
        ).fetchone()
        if task is None:
            raise ValueError("Page projection requires a validated Task/Plan scope")
        repository_id = str(task["repository_id"])

        existing = conn.execute(
            "SELECT freshness_cursor, payload_digest, source_digest "
            "FROM v2_semantic_pages WHERE page_id=?",
            (manifest.page_id,),
        ).fetchone()
        source_digest = digest(manifest)
        if existing is not None:
            if (
                existing["payload_digest"] != manifest.payload_digest
                or existing["source_digest"] != source_digest
            ):
                raise ValueError(f"immutable Page projection collision: {manifest.page_id}")
            descriptor = describe_page(manifest, groups)
            conn.execute(
                "INSERT OR IGNORE INTO v2_semantic_page_descriptors "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    manifest.page_id,
                    descriptor.delta_summary,
                    descriptor.outcome,
                    json.dumps(descriptor.delta_kinds),
                    json.dumps(descriptor.changed_files),
                    json.dumps(descriptor.changed_symbols),
                    json.dumps(descriptor.test_refs),
                    json.dumps(descriptor.decision_refs),
                    json.dumps(descriptor.unresolved_refs),
                    json.dumps(descriptor.supporting_event_ids),
                    json.dumps(primitive(descriptor.relation_targets), sort_keys=True),
                    digest(descriptor),
                ),
            )
            if self.fts_enabled:
                conn.execute(
                    "UPDATE v2_semantic_page_fts SET semantic_summary=? WHERE page_id=?",
                    (descriptor.delta_summary, manifest.page_id),
                )
            counts = conn.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM v2_semantic_nodes n "
                " WHERE n.node_id=? OR n.node_id IN "
                " (SELECT evidence_id FROM v2_semantic_evidence WHERE page_id=?)) AS nodes, "
                "(SELECT COUNT(*) FROM v2_semantic_edges e "
                " WHERE e.source_id=? OR e.target_id=?) AS edges, "
                "(SELECT COUNT(*) FROM v2_semantic_evidence WHERE page_id=?) AS evidence, "
                "(SELECT COUNT(*) FROM v2_semantic_anchors WHERE page_id=?) AS anchors",
                (
                    manifest.page_id,
                    manifest.page_id,
                    manifest.page_id,
                    manifest.page_id,
                    manifest.page_id,
                    manifest.page_id,
                ),
            ).fetchone()
            return PageProjectionReceipt(
                page_id=manifest.page_id,
                cursor=int(existing["freshness_cursor"]),
                node_count=int(counts["nodes"]),
                edge_count=int(counts["edges"]),
                evidence_count=int(counts["evidence"]),
                anchor_count=int(counts["anchors"]),
                idempotent_replay=True,
            )

        all_events = [event for group in groups for event in group.events]
        descriptor = describe_page(manifest, groups)
        ranges = {
            event.event_id: self._event_range(event.event_id, event_positions)
            for event in all_events
        }
        if len(ranges) != len(all_events):
            raise ValueError("Event IDs must be unique across a Page")
        if (
            min(value[0] for value in ranges.values()) < manifest.event_range[0]
            or max(value[1] for value in ranges.values()) > manifest.event_range[1]
        ):
            raise ValueError("Event positions escape the Manifest event range")
        observed_revisions = {group.revision_id for group in groups}
        if observed_revisions != set(manifest.revision_ids):
            raise ValueError("Manifest revision index does not match Page facts")
        actual_evidence_keys = {fact.key.key_digest for event in all_events for fact in event.facts}
        if actual_evidence_keys != set(manifest.evidence_key_digests):
            raise ValueError("Manifest EvidenceKey index does not match Page facts")
        actual_entity_refs = {entity for event in all_events for entity in event.entity_refs}
        if actual_entity_refs != set(manifest.entity_refs):
            raise ValueError("Manifest entity index does not match Page facts")
        actual_milestones = {
            milestone
            for group in groups
            for milestone in (
                group.milestone_id,
                *(event.milestone_id for event in group.events),
            )
            if milestone
        }
        if actual_milestones != set(manifest.milestone_ids):
            raise ValueError("Manifest Milestone index does not match Page facts")
        actual_phases = {event.execution_phase for event in all_events}
        if actual_phases != set(manifest.execution_phases):
            raise ValueError("Manifest execution-phase index does not match Page facts")

        branch = conn.execute(
            "SELECT 1 FROM v2_semantic_branches WHERE repository_id=? AND run_id=? AND branch_id=?",
            (repository_id, manifest.run_id, manifest.branch_id),
        ).fetchone()
        if branch is None:
            physical_branch = conn.execute(
                "SELECT parent_branch_id, parent_page_id FROM v2_page_branches "
                "WHERE run_id=? AND branch_id=?",
                (manifest.run_id, manifest.branch_id),
            ).fetchone()
            if physical_branch is None:
                raise ValueError("Page branch must be registered before projection")
            if physical_branch["parent_branch_id"] is None:
                self._ensure_branch(
                    conn,
                    repository_id=repository_id,
                    run_id=manifest.run_id,
                    branch_id=manifest.branch_id,
                )
            else:
                parent = conn.execute(
                    "SELECT freshness_cursor FROM v2_semantic_pages WHERE page_id=?",
                    (physical_branch["parent_page_id"],),
                ).fetchone()
                if parent is None:
                    raise ValueError("Semantic branch parent Page is not projected")
                parent_revision = conn.execute(
                    "SELECT revision_id FROM v2_semantic_page_revisions "
                    "WHERE page_id=? ORDER BY revision_id DESC LIMIT 1",
                    (physical_branch["parent_page_id"],),
                ).fetchone()
                conn.execute(
                    "INSERT INTO v2_semantic_branches VALUES(?,?,?,?,?,?)",
                    (
                        repository_id,
                        manifest.run_id,
                        manifest.branch_id,
                        str(physical_branch["parent_branch_id"]),
                        int(parent["freshness_cursor"]),
                        str(parent_revision["revision_id"]),
                    ),
                )
        cursor = self._advance_clock(conn, manifest.run_id)
        current_row = conn.execute(
            "SELECT plan_version_id FROM v2_semantic_edges "
            "WHERE run_id=? AND edge_type='CURRENT_MILESTONE' AND valid_to_cursor IS NULL",
            (manifest.run_id,),
        ).fetchone()
        plan_version_id = str(current_row["plan_version_id"]) if current_row is not None else None
        page_set_segment = None
        page_set_tables_exist = conn.execute(
            "SELECT COUNT(*) AS count FROM sqlite_master WHERE type='table' "
            "AND name IN ('v2_page_sets','v2_page_set_segments')"
        ).fetchone()
        if manifest.event_group_ids and int(page_set_tables_exist["count"]) == 2:
            placeholders = ",".join("?" for _ in manifest.event_group_ids)
            page_set_segment = conn.execute(
                "SELECT s.page_set_id,s.segment_index,p.segment_count,prior.page_id AS prior_page_id "
                "FROM v2_page_set_segments s "
                "JOIN v2_page_sets p ON p.page_set_id=s.page_set_id "
                "LEFT JOIN v2_page_set_segments prior "
                "ON prior.page_set_id=s.page_set_id AND prior.segment_index=s.segment_index-1 "
                f"WHERE s.group_id IN ({placeholders}) ORDER BY s.segment_index LIMIT 1",
                tuple(manifest.event_group_ids),
            ).fetchone()
        inserted_nodes = 0
        inserted_edges = 0
        inserted_nodes += self._put_node(
            conn,
            node_id=manifest.page_id,
            node_type=NodeType.PAGE,
            repository_id=repository_id,
            run_id=manifest.run_id,
            branch_id=manifest.branch_id,
            revision_id=(manifest.revision_ids[0] if len(manifest.revision_ids) == 1 else None),
            authority=Authority.ASSERTED,
            payload={
                "page_seq": manifest.page_seq,
                "event_range": manifest.event_range,
                "payload_digest": manifest.payload_digest,
                "page_kind": manifest.page_kind.value,
                "semantic_descriptor": descriptor_payload(descriptor),
                "page_set": (
                    {
                        "page_set_id": str(page_set_segment["page_set_id"]),
                        "segment_index": int(page_set_segment["segment_index"]),
                        "segment_count": int(page_set_segment["segment_count"]),
                    }
                    if page_set_segment is not None
                    else None
                ),
            },
            source_digest=source_digest,
            cursor=cursor,
        )
        conn.execute(
            "INSERT INTO v2_semantic_pages VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                manifest.page_id,
                repository_id,
                manifest.run_id,
                manifest.branch_id,
                manifest.page_seq,
                manifest.event_range[0],
                manifest.event_range[1],
                manifest.token_count,
                manifest.byte_count,
                manifest.payload_digest,
                manifest.redaction_proof,
                manifest.previous_page_id,
                manifest.seal_reason,
                manifest.page_kind.value,
                int(manifest.tail),
                cursor,
                source_digest,
            ),
        )
        conn.execute(
            "INSERT INTO v2_semantic_page_descriptors VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                manifest.page_id,
                descriptor.delta_summary,
                descriptor.outcome,
                json.dumps(descriptor.delta_kinds),
                json.dumps(descriptor.changed_files),
                json.dumps(descriptor.changed_symbols),
                json.dumps(descriptor.test_refs),
                json.dumps(descriptor.decision_refs),
                json.dumps(descriptor.unresolved_refs),
                json.dumps(descriptor.supporting_event_ids),
                json.dumps(primitive(descriptor.relation_targets), sort_keys=True),
                digest(descriptor),
            ),
        )

        revision_nodes: dict[str, str] = {}
        for revision_id in manifest.revision_ids:
            revision_node = self._revision_node_id(repository_id, revision_id)
            revision_nodes[revision_id] = revision_node
            inserted_nodes += self._put_node(
                conn,
                node_id=revision_node,
                node_type=NodeType.WORKSPACE_REVISION,
                repository_id=repository_id,
                run_id="*",
                branch_id="*",
                revision_id=revision_id,
                authority=Authority.ASSERTED,
                payload={"revision_id": revision_id},
                source_digest=digest({"revision": revision_id}),
                cursor=cursor,
            )
            conn.execute(
                "INSERT INTO v2_semantic_page_revisions VALUES(?,?)",
                (manifest.page_id, revision_id),
            )
            inserted_edges += self._put_edge(
                conn,
                edge_type="VALID_AT",
                source_id=manifest.page_id,
                source_type=NodeType.PAGE,
                target_id=revision_node,
                target_type=NodeType.WORKSPACE_REVISION,
                authority=Authority.DERIVED,
                repository_id=repository_id,
                run_id=manifest.run_id,
                branch_id=manifest.branch_id,
                plan_version_id=plan_version_id,
                revision_id=revision_id,
                cursor=cursor,
                provenance=("page_manifest.revision_ids", source_digest),
                critical_path=True,
            )

        milestone_ids: set[str] = set()
        for raw_milestone_id in manifest.milestone_ids:
            identity_id = self._milestone_identity(conn, manifest.run_id, raw_milestone_id)
            if identity_id is None or identity_id in milestone_ids:
                continue
            milestone_ids.add(identity_id)
            conn.execute(
                "INSERT INTO v2_semantic_page_milestones VALUES(?,?)",
                (manifest.page_id, identity_id),
            )
            inserted_edges += self._put_edge(
                conn,
                edge_type="FOCUSES_ON",
                source_id=manifest.page_id,
                source_type=NodeType.PAGE,
                target_id=identity_id,
                target_type=NodeType.MILESTONE_IDENTITY,
                authority=Authority.ASSERTED,
                repository_id=repository_id,
                run_id=manifest.run_id,
                branch_id=manifest.branch_id,
                plan_version_id=plan_version_id,
                revision_id=(manifest.revision_ids[0] if manifest.revision_ids else None),
                cursor=cursor,
                supporting_event_ids=tuple(event.event_id for event in all_events),
                critical_path=True,
            )
            inserted_edges += self._put_edge(
                conn,
                edge_type="CONTAINS_PAGE",
                source_id=identity_id,
                source_type=NodeType.MILESTONE_IDENTITY,
                target_id=manifest.page_id,
                target_type=NodeType.PAGE,
                authority=Authority.DERIVED,
                repository_id=repository_id,
                run_id=manifest.run_id,
                branch_id=manifest.branch_id,
                plan_version_id=plan_version_id,
                revision_id=(manifest.revision_ids[0] if manifest.revision_ids else None),
                cursor=cursor,
                provenance=("page_manifest.milestone_ids", source_digest),
                supporting_event_ids=descriptor.supporting_event_ids,
                critical_path=True,
            )

        if manifest.previous_page_id:
            previous = conn.execute(
                "SELECT run_id, branch_id FROM v2_semantic_pages WHERE page_id=?",
                (manifest.previous_page_id,),
            ).fetchone()
            if (
                previous is None
                or previous["run_id"] != manifest.run_id
                or previous["branch_id"] != manifest.branch_id
            ):
                raise ValueError("previous Page is missing or outside branch scope")
            inserted_edges += self._put_edge(
                conn,
                edge_type="ADVANCES_TO",
                source_id=manifest.previous_page_id,
                source_type=NodeType.PAGE,
                target_id=manifest.page_id,
                target_type=NodeType.PAGE,
                authority=Authority.DERIVED,
                repository_id=repository_id,
                run_id=manifest.run_id,
                branch_id=manifest.branch_id,
                plan_version_id=plan_version_id,
                revision_id=(manifest.revision_ids[0] if manifest.revision_ids else None),
                cursor=cursor,
                provenance=("page_manifest.previous_page_id", source_digest),
                supporting_event_ids=descriptor.supporting_event_ids,
                properties={
                    "delta_kinds": descriptor.delta_kinds,
                    "delta_summary": descriptor.delta_summary,
                    "outcome": descriptor.outcome,
                    "changed_files": descriptor.changed_files,
                    "changed_symbols": descriptor.changed_symbols,
                    "test_refs": descriptor.test_refs,
                    "decision_refs": descriptor.decision_refs,
                    "unresolved_refs": descriptor.unresolved_refs,
                },
            )
        else:
            branch_parent = conn.execute(
                "SELECT parent_branch_id,fork_cursor FROM v2_semantic_branches "
                "WHERE repository_id=? AND run_id=? AND branch_id=?",
                (repository_id, manifest.run_id, manifest.branch_id),
            ).fetchone()
            parent_page = None
            if branch_parent is not None and branch_parent["parent_branch_id"] is not None:
                parent_page = conn.execute(
                    "SELECT page_id FROM v2_semantic_pages "
                    "WHERE repository_id=? AND run_id=? AND branch_id=? "
                    "AND freshness_cursor<=? ORDER BY freshness_cursor DESC,page_seq DESC LIMIT 1",
                    (
                        repository_id,
                        manifest.run_id,
                        str(branch_parent["parent_branch_id"]),
                        int(branch_parent["fork_cursor"]),
                    ),
                ).fetchone()
            parent_page_id = str(parent_page["page_id"]) if parent_page is not None else None
            if parent_page_id is not None:
                inserted_edges += self._put_edge(
                    conn,
                    edge_type="BRANCHES_TO",
                    source_id=parent_page_id,
                    source_type=NodeType.PAGE,
                    target_id=manifest.page_id,
                    target_type=NodeType.PAGE,
                    authority=Authority.DERIVED,
                    repository_id=repository_id,
                    run_id=manifest.run_id,
                    branch_id=manifest.branch_id,
                    plan_version_id=plan_version_id,
                    revision_id=(manifest.revision_ids[0] if manifest.revision_ids else None),
                    cursor=cursor,
                    provenance=("semantic_branch.fork_cursor", source_digest),
                    supporting_event_ids=descriptor.supporting_event_ids,
                    properties={"delta_summary": descriptor.delta_summary},
                )

        if page_set_segment is not None and int(page_set_segment["segment_index"]) > 0:
            prior_page_id = str(page_set_segment["prior_page_id"] or "")
            if not prior_page_id:
                raise ValueError("PageSet continuation has no prior physical Page")
            inserted_edges += self._put_edge(
                conn,
                edge_type="CONTINUES_WITH",
                source_id=prior_page_id,
                source_type=NodeType.PAGE,
                target_id=manifest.page_id,
                target_type=NodeType.PAGE,
                authority=Authority.DERIVED,
                repository_id=repository_id,
                run_id=manifest.run_id,
                branch_id=manifest.branch_id,
                plan_version_id=plan_version_id,
                revision_id=(manifest.revision_ids[0] if manifest.revision_ids else None),
                cursor=cursor,
                provenance=("page_set.segment_index", source_digest),
                supporting_event_ids=descriptor.supporting_event_ids,
                properties={
                    "page_set_id": str(page_set_segment["page_set_id"]),
                    "segment_index": int(page_set_segment["segment_index"]),
                    "segment_count": int(page_set_segment["segment_count"]),
                    "physical_relation": True,
                },
                critical_path=True,
            )

        for edge_type, target_pages in descriptor.relation_targets.items():
            for source_page_id in target_pages:
                source_page = conn.execute(
                    "SELECT run_id,branch_id FROM v2_semantic_pages WHERE page_id=?",
                    (source_page_id,),
                ).fetchone()
                if (
                    source_page is None
                    or str(source_page["run_id"]) != manifest.run_id
                    or str(source_page["branch_id"]) != manifest.branch_id
                ):
                    raise ValueError("explicit Page relation target is absent or out of scope")
                inserted_edges += self._put_edge(
                    conn,
                    edge_type=edge_type,
                    source_id=source_page_id,
                    source_type=NodeType.PAGE,
                    target_id=manifest.page_id,
                    target_type=NodeType.PAGE,
                    authority=Authority.ASSERTED,
                    repository_id=repository_id,
                    run_id=manifest.run_id,
                    branch_id=manifest.branch_id,
                    plan_version_id=plan_version_id,
                    revision_id=(manifest.revision_ids[0] if manifest.revision_ids else None),
                    cursor=cursor,
                    provenance=("page_fact.explicit_relation", source_digest),
                    supporting_event_ids=descriptor.supporting_event_ids,
                    properties={
                        "declared_by_page": manifest.page_id,
                        "delta_summary": descriptor.delta_summary,
                        "outcome": descriptor.outcome,
                        "provenance_records": descriptor.relation_provenance.get(
                            edge_type,
                            {},
                        ).get(source_page_id, ()),
                    },
                )

        reference_nodes: dict[str, tuple[NodeType, str]] = {}
        entity_ids = set(manifest.entity_refs)
        entity_ids.update(entity for event in all_events for entity in event.entity_refs)
        entity_ids.update(
            fact.key.canonical_entity_id for event in all_events for fact in event.facts
        )
        for entity_id in sorted(entity_ids):
            reference = self._ensure_reference_node(
                conn,
                repository_id=repository_id,
                run_id=manifest.run_id,
                branch_id=manifest.branch_id,
                revision_id=(manifest.revision_ids[0] if manifest.revision_ids else "unknown"),
                canonical_entity_id=entity_id,
                cursor=cursor,
            )
            if reference is None:
                continue
            reference_nodes[entity_id] = reference
            conn.execute(
                "INSERT OR IGNORE INTO v2_semantic_page_entities VALUES(?,?,?)",
                (manifest.page_id, entity_id, reference[1]),
            )
            for alias in self._entity_aliases(entity_id):
                conn.execute(
                    "INSERT OR IGNORE INTO v2_semantic_entity_aliases "
                    "(repository_id,run_id,branch_id,canonical_entity_id,alias,"
                    "reference_node_id) VALUES(?,?,?,?,?,?)",
                    (
                        repository_id,
                        manifest.run_id,
                        manifest.branch_id,
                        entity_id,
                        alias,
                        reference[1],
                    ),
                )

        evidence_count = 0
        anchor_count = 0
        projected_evidence_ids: set[str] = set()
        for group in groups:
            group_ranges = [ranges[event.event_id] for event in group.events]
            group_range = (
                min(value[0] for value in group_ranges),
                max(value[1] for value in group_ranges),
            )
            group_milestone = self._milestone_identity(conn, manifest.run_id, group.milestone_id)
            inserted_nodes += self._put_node(
                conn,
                node_id=group.group_id,
                node_type=NodeType.EVENT_GROUP,
                repository_id=repository_id,
                run_id=manifest.run_id,
                branch_id=manifest.branch_id,
                revision_id=group.revision_id,
                authority=Authority.ASSERTED,
                payload={"group_id": group.group_id, "group_type": group.group_type},
                source_digest=digest(group),
                cursor=cursor,
            )
            conn.execute(
                "INSERT INTO v2_semantic_event_groups VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    group.group_id,
                    manifest.page_id,
                    repository_id,
                    manifest.run_id,
                    manifest.branch_id,
                    group.revision_id,
                    group.group_type,
                    group_milestone,
                    group_range[0],
                    group_range[1],
                    group.created_at,
                    digest(group),
                ),
            )
            if group_milestone:
                inserted_edges += self._put_edge(
                    conn,
                    edge_type="EXECUTED_UNDER",
                    source_id=group.group_id,
                    source_type=NodeType.EVENT_GROUP,
                    target_id=group_milestone,
                    target_type=NodeType.MILESTONE_IDENTITY,
                    authority=Authority.ASSERTED,
                    repository_id=repository_id,
                    run_id=manifest.run_id,
                    branch_id=manifest.branch_id,
                    plan_version_id=plan_version_id,
                    revision_id=group.revision_id,
                    cursor=cursor,
                    supporting_event_ids=tuple(event.event_id for event in group.events),
                    critical_path=True,
                )

            for event in group.events:
                event_range = ranges[event.event_id]
                event_milestone = self._milestone_identity(
                    conn,
                    manifest.run_id,
                    event.milestone_id or group.milestone_id,
                )
                inserted_nodes += self._put_node(
                    conn,
                    node_id=event.event_id,
                    node_type=NodeType.EVENT,
                    repository_id=repository_id,
                    run_id=manifest.run_id,
                    branch_id=manifest.branch_id,
                    revision_id=event.revision_id,
                    authority=Authority.ASSERTED,
                    payload={"event_id": event.event_id},
                    source_digest=digest({"event_id": event.event_id}),
                    cursor=cursor,
                )
                conn.execute(
                    "INSERT INTO v2_semantic_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        event.event_id,
                        group.group_id,
                        manifest.page_id,
                        repository_id,
                        manifest.run_id,
                        manifest.branch_id,
                        event.revision_id,
                        event_range[0],
                        event_range[1],
                        event.event_type,
                        event_milestone,
                        event.execution_phase,
                        digest(event.payload),
                        event.observed_at,
                    ),
                )
                if event_milestone:
                    inserted_edges += self._put_edge(
                        conn,
                        edge_type="EXECUTED_UNDER",
                        source_id=event.event_id,
                        source_type=NodeType.EVENT,
                        target_id=event_milestone,
                        target_type=NodeType.MILESTONE_IDENTITY,
                        authority=Authority.ASSERTED,
                        repository_id=repository_id,
                        run_id=manifest.run_id,
                        branch_id=manifest.branch_id,
                        plan_version_id=plan_version_id,
                        revision_id=event.revision_id,
                        cursor=cursor,
                        supporting_event_ids=(event.event_id,),
                        critical_path=True,
                    )

                event_refs = {
                    entity: reference_nodes[entity]
                    for entity in event.entity_refs
                    if entity in reference_nodes
                }
                lowered_type = event.event_type.lower()
                modifies = any(
                    marker in lowered_type
                    for marker in ("write", "edit", "patch", "modify", "change")
                ) or any(fact.key.evidence_type == FactType.CODE_CHANGE for fact in event.facts)
                reads = "read" in lowered_type or "inspect" in lowered_type
                runs_test = "test" in lowered_type or any(
                    fact.key.evidence_type in (FactType.TEST_RESULT, FactType.TEST_FAILURE)
                    for fact in event.facts
                )
                observes_failure = "fail" in lowered_type or any(
                    fact.key.evidence_type == FactType.TEST_FAILURE for fact in event.facts
                )
                for entity_id, (reference_type, reference_id) in event_refs.items():
                    edge_type: str | None = None
                    if reference_type == NodeType.FILE_REFERENCE and modifies:
                        edge_type = "MODIFIES"
                    elif reference_type == NodeType.SYMBOL_REFERENCE and modifies:
                        edge_type = "MODIFIES"
                    elif reference_type == NodeType.FILE_REFERENCE and reads:
                        edge_type = "READS"
                    elif reference_type == NodeType.TEST_REFERENCE and runs_test:
                        edge_type = "RUNS_TEST"
                    elif reference_type == NodeType.FAILURE_REFERENCE and observes_failure:
                        edge_type = "OBSERVES_FAILURE"
                    if edge_type:
                        inserted_edges += self._put_edge(
                            conn,
                            edge_type=edge_type,
                            source_id=event.event_id,
                            source_type=NodeType.EVENT,
                            target_id=reference_id,
                            target_type=reference_type,
                            authority=Authority.ASSERTED,
                            repository_id=repository_id,
                            run_id=manifest.run_id,
                            branch_id=manifest.branch_id,
                            plan_version_id=plan_version_id,
                            revision_id=event.revision_id,
                            cursor=cursor,
                            supporting_event_ids=(event.event_id,),
                        )
                test_refs = [
                    value for value in event_refs.values() if value[0] == NodeType.TEST_REFERENCE
                ]
                failure_refs = [
                    value for value in event_refs.values() if value[0] == NodeType.FAILURE_REFERENCE
                ]
                if observes_failure:
                    for test_ref in test_refs:
                        for failure_ref in failure_refs:
                            inserted_edges += self._put_edge(
                                conn,
                                edge_type="OBSERVES_FAILURE",
                                source_id=test_ref[1],
                                source_type=NodeType.TEST_REFERENCE,
                                target_id=failure_ref[1],
                                target_type=NodeType.FAILURE_REFERENCE,
                                authority=Authority.ASSERTED,
                                repository_id=repository_id,
                                run_id=manifest.run_id,
                                branch_id=manifest.branch_id,
                                plan_version_id=plan_version_id,
                                revision_id=event.revision_id,
                                cursor=cursor,
                                supporting_event_ids=(event.event_id,),
                            )
                if modifies:
                    inserted_edges += self._put_edge(
                        conn,
                        edge_type="UPDATES",
                        source_id=event.event_id,
                        source_type=NodeType.EVENT,
                        target_id=revision_nodes[event.revision_id],
                        target_type=NodeType.WORKSPACE_REVISION,
                        authority=Authority.ASSERTED,
                        repository_id=repository_id,
                        run_id=manifest.run_id,
                        branch_id=manifest.branch_id,
                        plan_version_id=plan_version_id,
                        revision_id=event.revision_id,
                        cursor=cursor,
                        supporting_event_ids=(event.event_id,),
                    )

                raw_invalidations = event.payload.get("invalidates_evidence_ids", ())
                if isinstance(raw_invalidations, str):
                    raw_invalidations = (raw_invalidations,)
                if isinstance(raw_invalidations, Sequence):
                    for invalidated_id in dict.fromkeys(map(str, raw_invalidations)):
                        invalidated = conn.execute(
                            "SELECT evidence_id FROM v2_semantic_evidence "
                            "WHERE evidence_id=? AND repository_id=? AND run_id=? "
                            "AND branch_id=? AND valid_to_cursor IS NULL",
                            (
                                invalidated_id,
                                repository_id,
                                manifest.run_id,
                                manifest.branch_id,
                            ),
                        ).fetchone()
                        if invalidated is None:
                            raise ValueError(
                                f"invalidated Evidence is absent or out of scope: {invalidated_id}"
                            )
                        conn.execute(
                            "UPDATE v2_semantic_evidence "
                            "SET valid_to_revision=?, valid_to_cursor=? "
                            "WHERE evidence_id=? AND valid_to_cursor IS NULL",
                            (event.revision_id, cursor, invalidated_id),
                        )
                        conn.execute(
                            "UPDATE v2_semantic_edges "
                            "SET valid_to_revision=?, valid_to_cursor=? "
                            "WHERE source_id=? AND run_id=? AND branch_id=? "
                            "AND valid_to_cursor IS NULL",
                            (
                                event.revision_id,
                                cursor,
                                invalidated_id,
                                manifest.run_id,
                                manifest.branch_id,
                            ),
                        )
                        inserted_edges += self._put_edge(
                            conn,
                            edge_type="INVALIDATED_BY",
                            source_id=invalidated_id,
                            source_type=NodeType.EVIDENCE_UNIT,
                            target_id=event.event_id,
                            target_type=NodeType.EVENT,
                            authority=Authority.ASSERTED,
                            repository_id=repository_id,
                            run_id=manifest.run_id,
                            branch_id=manifest.branch_id,
                            plan_version_id=plan_version_id,
                            revision_id=event.revision_id,
                            cursor=cursor,
                            supporting_event_ids=(event.event_id,),
                        )

                for fact in event.facts:
                    blob_handle, blob_range = self._blob_location(event, fact)
                    if blob_handle is None:
                        blob_handle, blob_range = self._blob_location(event)
                    content_digest = digest(fact.content)
                    fact_source_digest = digest(primitive(fact))
                    evidence_id = stable_id(
                        "evidence_",
                        {
                            "page": manifest.page_id,
                            "event": event.event_id,
                            "fact": fact_source_digest,
                        },
                    )
                    if evidence_id in projected_evidence_ids:
                        # Event facts are semantic set members. Provider
                        # snapshots can repeat an indistinguishable item; one
                        # Evidence Unit and Anchor preserve the complete fact
                        # without polluting the graph or violating identity.
                        continue
                    projected_evidence_ids.add(evidence_id)
                    anchor_id = stable_id(
                        "anchor_", {"evidence": evidence_id, "page": manifest.page_id}
                    )
                    evidence_unit = EvidenceUnit(
                        evidence_id=evidence_id,
                        key=fact.key,
                        content=fact.content,
                        authority=fact.authority,
                        confidence=fact.confidence,
                        must_preserve=fact.must_preserve,
                        content_digest=content_digest,
                        event_ids=(event.event_id,),
                        event_group_id=group.group_id,
                        page_id=manifest.page_id,
                        run_id=manifest.run_id,
                        branch_id=manifest.branch_id,
                        revision_id=event.revision_id,
                        valid_from_revision=event.revision_id,
                    )
                    semantic_anchor = SemanticAnchor(
                        anchor_id=anchor_id,
                        evidence_id=evidence_id,
                        page_id=manifest.page_id,
                        page_digest=manifest.payload_digest,
                        event_ids=(event.event_id,),
                        event_range=event_range,
                        event_group_id=group.group_id,
                        blob_handle=blob_handle,
                        blob_range=blob_range,
                        revision_id=event.revision_id,
                        branch_id=manifest.branch_id,
                    )
                    inserted_nodes += self._put_node(
                        conn,
                        node_id=evidence_id,
                        node_type=NodeType.EVIDENCE_UNIT,
                        repository_id=repository_id,
                        run_id=manifest.run_id,
                        branch_id=manifest.branch_id,
                        revision_id=event.revision_id,
                        authority=fact.authority,
                        payload={
                            "key_digest": fact.key.key_digest,
                            "content_digest": content_digest,
                        },
                        source_digest=digest({"event": event.event_id, "fact": fact_source_digest}),
                        cursor=cursor,
                    )
                    inserted_nodes += self._put_node(
                        conn,
                        node_id=anchor_id,
                        node_type=NodeType.SEMANTIC_ANCHOR,
                        repository_id=repository_id,
                        run_id=manifest.run_id,
                        branch_id=manifest.branch_id,
                        revision_id=event.revision_id,
                        authority=Authority.DERIVED,
                        payload={
                            "page_id": manifest.page_id,
                            "page_digest": manifest.payload_digest,
                            "event_range": event_range,
                        },
                        source_digest=digest(
                            {
                                "page": manifest.payload_digest,
                                "event": event.event_id,
                                "range": event_range,
                            }
                        ),
                        cursor=cursor,
                    )
                    conn.execute(
                        "INSERT INTO v2_semantic_evidence VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            evidence_unit.evidence_id,
                            evidence_unit.key.key_digest,
                            evidence_unit.key.evidence_type.value,
                            evidence_unit.key.canonical_entity_id,
                            evidence_unit.key.semantic_role,
                            evidence_unit.key.revision_constraint,
                            evidence_unit.key.branch_scope,
                            evidence_unit.key.validity_requirement,
                            json.dumps(
                                primitive(fact.content),
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            evidence_unit.content_digest,
                            evidence_unit.authority.value,
                            evidence_unit.confidence,
                            int(evidence_unit.must_preserve),
                            event.event_id,
                            group.group_id,
                            manifest.page_id,
                            repository_id,
                            manifest.run_id,
                            manifest.branch_id,
                            evidence_unit.revision_id,
                            evidence_unit.valid_from_revision,
                            None,
                            cursor,
                            None,
                        ),
                    )
                    conn.execute(
                        "INSERT INTO v2_semantic_anchors VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            semantic_anchor.anchor_id,
                            semantic_anchor.evidence_id,
                            semantic_anchor.page_id,
                            semantic_anchor.page_digest,
                            semantic_anchor.event_ids[0],
                            semantic_anchor.event_group_id,
                            semantic_anchor.event_range[0],
                            semantic_anchor.event_range[1],
                            semantic_anchor.blob_handle,
                            semantic_anchor.blob_range[0] if semantic_anchor.blob_range else None,
                            semantic_anchor.blob_range[1] if semantic_anchor.blob_range else None,
                            semantic_anchor.revision_id,
                            semantic_anchor.branch_id,
                        ),
                    )
                    evidence_count += 1
                    anchor_count += 1
                    for target_id, target_type in (
                        (event.event_id, NodeType.EVENT),
                        (group.group_id, NodeType.EVENT_GROUP),
                    ):
                        inserted_edges += self._put_edge(
                            conn,
                            edge_type="EVIDENCED_BY",
                            source_id=evidence_id,
                            source_type=NodeType.EVIDENCE_UNIT,
                            target_id=target_id,
                            target_type=target_type,
                            authority=Authority.DERIVED,
                            repository_id=repository_id,
                            run_id=manifest.run_id,
                            branch_id=manifest.branch_id,
                            plan_version_id=plan_version_id,
                            revision_id=event.revision_id,
                            cursor=cursor,
                            provenance=("event.facts", event.event_id),
                            supporting_event_ids=(event.event_id,),
                            critical_path=True,
                        )
                    inserted_edges += self._put_edge(
                        conn,
                        edge_type="LOCATED_AT",
                        source_id=evidence_id,
                        source_type=NodeType.EVIDENCE_UNIT,
                        target_id=anchor_id,
                        target_type=NodeType.SEMANTIC_ANCHOR,
                        authority=Authority.DERIVED,
                        repository_id=repository_id,
                        run_id=manifest.run_id,
                        branch_id=manifest.branch_id,
                        plan_version_id=plan_version_id,
                        revision_id=event.revision_id,
                        cursor=cursor,
                        provenance=("page_projection.anchor", source_digest),
                        supporting_event_ids=(event.event_id,),
                        critical_path=True,
                    )
                    inserted_edges += self._put_edge(
                        conn,
                        edge_type="STORED_IN",
                        source_id=anchor_id,
                        source_type=NodeType.SEMANTIC_ANCHOR,
                        target_id=manifest.page_id,
                        target_type=NodeType.PAGE,
                        authority=Authority.DERIVED,
                        repository_id=repository_id,
                        run_id=manifest.run_id,
                        branch_id=manifest.branch_id,
                        plan_version_id=plan_version_id,
                        revision_id=event.revision_id,
                        cursor=cursor,
                        provenance=("page_projection.manifest", source_digest),
                        supporting_event_ids=(event.event_id,),
                        critical_path=True,
                    )
                    inserted_edges += self._put_edge(
                        conn,
                        edge_type="VALID_AT",
                        source_id=evidence_id,
                        source_type=NodeType.EVIDENCE_UNIT,
                        target_id=revision_nodes[event.revision_id],
                        target_type=NodeType.WORKSPACE_REVISION,
                        authority=Authority.DERIVED,
                        repository_id=repository_id,
                        run_id=manifest.run_id,
                        branch_id=manifest.branch_id,
                        plan_version_id=plan_version_id,
                        revision_id=event.revision_id,
                        cursor=cursor,
                        provenance=("event.revision_id", event.event_id),
                        supporting_event_ids=(event.event_id,),
                        critical_path=True,
                    )
                    if event.milestone_id is not None:
                        milestone_node = conn.execute(
                            "SELECT 1 FROM v2_semantic_nodes WHERE node_id=? AND node_type=?",
                            (event.milestone_id, NodeType.MILESTONE_IDENTITY.value),
                        ).fetchone()
                        if milestone_node is not None:
                            inserted_edges += self._put_edge(
                                conn,
                                edge_type="SUPPORTS",
                                source_id=evidence_id,
                                source_type=NodeType.EVIDENCE_UNIT,
                                target_id=event.milestone_id,
                                target_type=NodeType.MILESTONE_IDENTITY,
                                authority=fact.authority,
                                repository_id=repository_id,
                                run_id=manifest.run_id,
                                branch_id=manifest.branch_id,
                                plan_version_id=plan_version_id,
                                revision_id=event.revision_id,
                                cursor=cursor,
                                provenance=("event.milestone_id", event.event_id),
                                supporting_event_ids=(event.event_id,),
                                critical_path=True,
                            )
                        plan_step_id = str(fact.content.get("plan_step_id", "")).strip()
                        if plan_step_id:
                            step = conn.execute(
                                "SELECT step_identity_id FROM v2_plan_step_identities "
                                "WHERE run_id=? AND canonical_step_id=? "
                                "AND milestone_identity_id=?",
                                (manifest.run_id, plan_step_id, event.milestone_id),
                            ).fetchone()
                            if step is not None:
                                inserted_edges += self._put_edge(
                                    conn,
                                    edge_type="SUPPORTS",
                                    source_id=evidence_id,
                                    source_type=NodeType.EVIDENCE_UNIT,
                                    target_id=str(step["step_identity_id"]),
                                    target_type=NodeType.PLAN_STEP,
                                    authority=fact.authority,
                                    repository_id=repository_id,
                                    run_id=manifest.run_id,
                                    branch_id=manifest.branch_id,
                                    plan_version_id=plan_version_id,
                                    revision_id=event.revision_id,
                                    cursor=cursor,
                                    provenance=("fact.content.plan_step_id", event.event_id),
                                    supporting_event_ids=(event.event_id,),
                                    critical_path=True,
                                )
                        criterion_value = fact.content.get(
                            "criterion_ids", fact.content.get("criterion_id", ())
                        )
                        if isinstance(criterion_value, str):
                            criterion_ids = (criterion_value,)
                        elif isinstance(criterion_value, (list, tuple)):
                            criterion_ids = tuple(map(str, criterion_value))
                        else:
                            criterion_ids = ()
                        for criterion_id in criterion_ids:
                            criterion = conn.execute(
                                "SELECT criterion_identity_id,required_evidence_types_json "
                                "FROM v2_completion_criteria "
                                "WHERE run_id=? AND milestone_identity_id=? "
                                "AND local_criterion_id=? ORDER BY created_cursor DESC LIMIT 1",
                                (manifest.run_id, event.milestone_id, criterion_id),
                            ).fetchone()
                            if criterion is None or not self._fact_is_positive_criterion_evidence(
                                fact,
                                tuple(
                                    map(
                                        str,
                                        json.loads(
                                            str(criterion["required_evidence_types_json"])
                                        ),
                                    )
                                ),
                            ):
                                continue
                            inserted_edges += self._put_edge(
                                conn,
                                edge_type="SATISFIES",
                                source_id=evidence_id,
                                source_type=NodeType.EVIDENCE_UNIT,
                                target_id=str(criterion["criterion_identity_id"]),
                                target_type=NodeType.COMPLETION_CRITERION,
                                authority=fact.authority,
                                repository_id=repository_id,
                                run_id=manifest.run_id,
                                branch_id=manifest.branch_id,
                                plan_version_id=plan_version_id,
                                revision_id=event.revision_id,
                                cursor=cursor,
                                provenance=("fact.content.criterion_ids", event.event_id),
                                supporting_event_ids=(event.event_id,),
                                properties={"acceptance_scope": "FACTUAL_EVIDENCE"},
                                critical_path=True,
                            )
                    about_ref = reference_nodes.get(fact.key.canonical_entity_id)
                    if about_ref:
                        about_edges = {
                            NodeType.FILE_REFERENCE: "ABOUT_FILE",
                            NodeType.SYMBOL_REFERENCE: "ABOUT_SYMBOL",
                            NodeType.TEST_REFERENCE: "ABOUT_TEST",
                            NodeType.FAILURE_REFERENCE: "ABOUT_FAILURE",
                            NodeType.CHANGE_REFERENCE: "ABOUT_CHANGE",
                        }
                        inserted_edges += self._put_edge(
                            conn,
                            edge_type=about_edges[about_ref[0]],
                            source_id=evidence_id,
                            source_type=NodeType.EVIDENCE_UNIT,
                            target_id=about_ref[1],
                            target_type=about_ref[0],
                            authority=(
                                fact.authority
                                if fact.authority != Authority.INFERRED
                                else Authority.DERIVED
                            ),
                            repository_id=repository_id,
                            run_id=manifest.run_id,
                            branch_id=manifest.branch_id,
                            plan_version_id=plan_version_id,
                            revision_id=event.revision_id,
                            cursor=cursor,
                            provenance=("evidence_key.canonical_entity_id", event.event_id),
                            supporting_event_ids=(event.event_id,),
                        )

        if self.fts_enabled:
            conn.execute("DELETE FROM v2_semantic_page_fts WHERE page_id=?", (manifest.page_id,))
            conn.execute(
                "INSERT INTO v2_semantic_page_fts VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    manifest.page_id,
                    repository_id,
                    manifest.run_id,
                    manifest.branch_id,
                    " ".join(manifest.revision_ids),
                    " ".join(manifest.milestone_ids),
                    " ".join(sorted(entity_ids)),
                    " ".join(manifest.evidence_key_digests),
                    " ".join(manifest.execution_phases),
                    descriptor.delta_summary,
                ),
            )
        return PageProjectionReceipt(
            page_id=manifest.page_id,
            cursor=cursor,
            node_count=inserted_nodes,
            edge_count=inserted_edges,
            evidence_count=evidence_count,
            anchor_count=anchor_count,
            idempotent_replay=False,
        )

    def page_graph_context(
        self,
        page_ids: Sequence[str],
        *,
        max_hops: int = 1,
        limit: int = 8,
        required_relations: Sequence[str] = (),
        preferred_relations: Sequence[str] = (),
        direction: str = "BOTH",
    ) -> tuple[Mapping[str, object], ...]:
        """Execute a bounded, model-intent-directed Page Graph query."""

        if max_hops < 0 or max_hops > 2:
            raise ValueError("Page graph expansion is bounded to zero, one, or two hops")
        if limit <= 0:
            return ()
        frontier = set(map(str, page_ids))
        seen_pages = set(frontier)
        seen_edges: set[str] = set()
        rows: list[sqlite3.Row] = []
        # CONTAINS_PAGE is a Milestone->Page edge, so it cannot yield a
        # Page-to-Page neighbor in this bounded traversal.
        allowed = tuple(sorted(SEMANTIC_PAGE_QUERY_RELATIONS))
        allowed_set = set(allowed)
        required = tuple(
            dict.fromkeys(str(item).strip().upper() for item in required_relations if str(item))
        )
        preferred = tuple(
            dict.fromkeys(str(item).strip().upper() for item in preferred_relations if str(item))
        )
        unknown = set((*required, *preferred)).difference(allowed_set)
        if unknown:
            raise ValueError(f"unknown Page Graph relation types: {sorted(unknown)}")
        direction = direction.upper()
        if direction not in {"INCOMING", "OUTGOING", "BOTH"}:
            raise ValueError("Page Graph direction must be INCOMING, OUTGOING, or BOTH")
        selected_relations = required or allowed
        relation_priority = required or preferred
        with self._read_connection() as conn:
            for _ in range(max_hops):
                if not frontier or len(rows) >= limit:
                    break
                placeholders = ",".join("?" for _ in frontier)
                edge_placeholders = ",".join("?" for _ in selected_relations)
                if direction == "OUTGOING":
                    endpoint_sql = f"source_id IN ({placeholders})"
                    endpoint_params: tuple[object, ...] = tuple(frontier)
                elif direction == "INCOMING":
                    endpoint_sql = f"target_id IN ({placeholders})"
                    endpoint_params = tuple(frontier)
                else:
                    endpoint_sql = (
                        f"(source_id IN ({placeholders}) OR target_id IN ({placeholders}))"
                    )
                    endpoint_params = (*frontier, *frontier)
                priority_sql = ""
                priority_params: tuple[object, ...] = ()
                if relation_priority:
                    priority_sql = (
                        "CASE edge_type "
                        + " ".join(
                            f"WHEN ? THEN {index}" for index, _ in enumerate(relation_priority)
                        )
                        + f" ELSE {len(relation_priority)} END,"
                    )
                    priority_params = tuple(relation_priority)
                batch = conn.execute(
                    "SELECT * FROM v2_semantic_edges WHERE edge_type IN ("
                    + edge_placeholders
                    + ") AND valid_to_cursor IS NULL AND "
                    + endpoint_sql
                    + " ORDER BY "
                    + priority_sql
                    + "CASE authority WHEN 'ASSERTED' THEN 0 WHEN 'DERIVED' THEN 1 ELSE 2 END,"
                    + "edge_type,valid_from_cursor DESC,edge_id LIMIT ?",
                    (
                        *selected_relations,
                        *endpoint_params,
                        *priority_params,
                        limit - len(rows),
                    ),
                ).fetchall()
                next_frontier: set[str] = set()
                for row in batch:
                    edge_id = str(row["edge_id"])
                    if edge_id in seen_edges:
                        continue
                    seen_edges.add(edge_id)
                    rows.append(row)
                    for page_id in (str(row["source_id"]), str(row["target_id"])):
                        if page_id not in seen_pages:
                            seen_pages.add(page_id)
                            next_frontier.add(page_id)
                frontier = next_frontier
            descriptor_rows = (
                conn.execute(
                    "SELECT page_id,delta_summary,outcome FROM v2_semantic_page_descriptors "
                    "WHERE page_id IN (" + ",".join("?" for _ in seen_pages) + ")",
                    tuple(seen_pages),
                ).fetchall()
                if seen_pages
                else ()
            )
        descriptors = {
            str(row["page_id"]): {
                "summary": str(row["delta_summary"])[:600],
                "outcome": str(row["outcome"]),
            }
            for row in descriptor_rows
        }
        return tuple(
            {
                "graph": "SEMANTIC_PAGE",
                "edge_type": str(row["edge_type"]),
                "source_page_id": str(row["source_id"]),
                "target_page_id": str(row["target_id"]),
                "properties": json.loads(str(row["properties_json"])),
                "source_descriptor": descriptors.get(str(row["source_id"]), {}),
                "target_descriptor": descriptors.get(str(row["target_id"]), {}),
                "authority": str(row["authority"]),
                "supporting_event_ids": json.loads(str(row["supporting_event_ids_json"])),
                "intent_match": (
                    "REQUIRED"
                    if str(row["edge_type"]) in required
                    else "PREFERRED"
                    if str(row["edge_type"]) in preferred
                    else "FALLBACK"
                ),
            }
            for row in rows[:limit]
        )

    def page_graph_candidates(
        self,
        intent: RecallIntent,
        page_ids: Sequence[str],
        *,
        max_hops: int = 1,
        limit: int = 8,
    ) -> tuple[PageCandidate, ...]:
        """Translate confirmed Page-flow neighbors into bounded read candidates."""

        if limit <= 0:
            return ()
        relationships = self.page_graph_context(
            page_ids,
            max_hops=max_hops,
            limit=limit * 2,
            required_relations=intent.required_structural_relations,
            preferred_relations=intent.preferred_structural_relations,
            direction=intent.structural_relation_direction,
        )
        starting = set(map(str, page_ids))
        ordered_ids: list[str] = []
        for relation in relationships:
            for field in ("source_page_id", "target_page_id"):
                page_id = str(relation.get(field, ""))
                if page_id and page_id not in starting and page_id not in ordered_ids:
                    ordered_ids.append(page_id)
        return self._page_candidates_by_id(intent, ordered_ids, limit=limit)

    def direct_page_candidates(
        self,
        intent: RecallIntent,
        page_ids: Sequence[str],
        *,
        limit: int = 8,
    ) -> tuple[PageCandidate, ...]:
        """Dereference a runtime-owned MemoryRef without semantic re-search.

        ``page_ids`` are immutable Page addresses recovered from the opaque
        MemoryRef.  A PageSet directory may translate that address to the one
        requested semantic segment, but neither Evidence ranking nor graph
        recency is allowed to replace the address.  The addressed Page itself
        remains the fallback when the directory has no more precise entry.
        """

        starting = tuple(dict.fromkeys(str(item) for item in page_ids if str(item)))
        if not starting or limit <= 0:
            return ()
        routed = self._page_set_page_ids(
            starting,
            intent=intent,
            limit=limit,
            include_starting=True,
            allow_nearest=False,
        )
        # No directory match means the immutable address itself is the target.
        # A semantically routed PageSet section replaces, rather than merely
        # seeds, that address for this bounded Page-in.
        ordered_ids = tuple(dict.fromkeys(routed or starting))[:limit]
        return self._page_candidates_by_id(intent, ordered_ids, limit=limit)

    def page_candidates_for_references(
        self,
        intent: RecallIntent,
        reference_ids: Sequence[str],
        *,
        limit: int = 8,
    ) -> tuple[PageCandidate, ...]:
        """Bridge Rich Graph reference identities back to Semantic Pages.

        Rich structure is address assistance only. It selects immutable Pages
        that already recorded the referenced file/symbol; the Recall service
        still validates the Page body and never treats this bridge as Evidence
        coverage.
        """

        references = tuple(
            dict.fromkeys(str(value) for value in reference_ids if str(value).strip())
        )
        if not references or limit <= 0:
            return ()
        placeholders = ",".join("?" for _ in references)
        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT pe.page_id,MAX(p.freshness_cursor) AS freshness "
                "FROM v2_semantic_page_entities pe "
                "JOIN v2_semantic_pages p ON p.page_id=pe.page_id "
                "WHERE p.repository_id=? AND p.run_id=? AND p.branch_id=? AND "
                f"(pe.reference_node_id IN ({placeholders}) OR "
                f"pe.canonical_entity_id IN ({placeholders})) "
                "GROUP BY pe.page_id ORDER BY freshness DESC LIMIT ?",
                (
                    intent.repository_id,
                    intent.run_id,
                    intent.branch_id,
                    *references,
                    *references,
                    limit,
                ),
            ).fetchall()
        return self._page_candidates_by_id(
            intent,
            tuple(str(row["page_id"]) for row in rows),
            limit=limit,
        )

    def _page_candidates_by_id(
        self,
        intent: RecallIntent,
        page_ids: Sequence[str],
        *,
        limit: int,
    ) -> tuple[PageCandidate, ...]:
        """Materialize already-selected Page IDs as validated read candidates."""

        candidates: list[PageCandidate] = []
        with self._read_connection() as connection:
            for page_id in page_ids:
                row = connection.execute(
                    "SELECT p.*,pr.revision_id FROM v2_semantic_pages p "
                    "JOIN v2_semantic_page_revisions pr ON pr.page_id=p.page_id "
                    "WHERE p.page_id=? AND p.repository_id=? AND p.run_id=? "
                    "AND p.branch_id=? AND (?=0 OR pr.revision_id=?) "
                    "ORDER BY CASE WHEN pr.revision_id=? THEN 0 ELSE 1 END LIMIT 1",
                    (
                        page_id,
                        intent.repository_id,
                        intent.run_id,
                        intent.branch_id,
                        int(intent.require_exact_revision),
                        intent.revision_id,
                        intent.revision_id,
                    ),
                ).fetchone()
                if row is None:
                    continue
                evidence = connection.execute(
                    "SELECT evidence_id,key_digest FROM v2_semantic_evidence "
                    "WHERE page_id=? AND authority IN ('ASSERTED','DERIVED') "
                    "ORDER BY evidence_id",
                    (page_id,),
                ).fetchall()
                anchors = connection.execute(
                    "SELECT anchor_id FROM v2_semantic_anchors WHERE page_id=? ORDER BY anchor_id",
                    (page_id,),
                ).fetchall()
                candidates.append(
                    PageCandidate(
                        page_id=page_id,
                        payload_digest=str(row["payload_digest"]),
                        revision_id=str(row["revision_id"]),
                        branch_id=str(row["branch_id"]),
                        event_range=(int(row["event_start"]), int(row["event_end"])),
                        anchor_ids=tuple(str(item["anchor_id"]) for item in anchors),
                        evidence_ids=tuple(str(item["evidence_id"]) for item in evidence),
                        evidence_key_digests=tuple(
                            dict.fromkeys(str(item["key_digest"]) for item in evidence)
                        ),
                        estimated_tokens=int(row["token_count"]),
                        freshness_cursor=int(row["freshness_cursor"]),
                    )
                )
                if len(candidates) >= limit:
                    break
        return tuple(candidates)

    def _page_set_page_ids(
        self,
        page_ids: Sequence[str],
        *,
        intent: RecallIntent,
        limit: int,
        include_starting: bool = False,
        allow_nearest: bool = True,
    ) -> tuple[str, ...]:
        """Route through a PageSet directory before opening physical siblings."""

        starting = tuple(dict.fromkeys(str(item) for item in page_ids if str(item)))
        if not starting or limit <= 0:
            return ()
        with self._read_connection() as connection:
            tables = connection.execute(
                "SELECT COUNT(*) AS count FROM sqlite_master WHERE type='table' "
                "AND name IN ('v2_page_sets','v2_page_set_segments',"
                "'v2_page_set_synopses','v2_page_set_segment_directory')"
            ).fetchone()
            if int(tables["count"]) < 2:
                return ()
            placeholders = ",".join("?" for _ in starting)
            if int(tables["count"]) != 4:
                if not allow_nearest:
                    return ()
                rows = connection.execute(
                    "SELECT DISTINCT sibling.page_id,sibling.segment_index "
                    "FROM v2_page_set_segments seed "
                    "JOIN v2_page_set_segments sibling ON sibling.page_set_id=seed.page_set_id "
                    f"WHERE seed.page_id IN ({placeholders}) AND sibling.page_id IS NOT NULL "
                    "ORDER BY sibling.segment_index LIMIT ?",
                    (*starting, limit + len(starting)),
                ).fetchall()
                return tuple(
                    str(row["page_id"])
                    for row in rows
                    if str(row["page_id"]) not in starting
                )[:limit]
            rows = connection.execute(
                "SELECT seed.page_id AS seed_page_id,seed.segment_index AS seed_index,"
                "sibling.page_id,sibling.segment_index,d.title,d.summary,"
                "d.semantic_kinds_json,d.entity_refs_json "
                "FROM v2_page_set_segments seed "
                "JOIN v2_page_set_segments sibling ON sibling.page_set_id=seed.page_set_id "
                "JOIN v2_page_set_segment_directory d "
                "ON d.page_set_id=sibling.page_set_id AND d.segment_index=sibling.segment_index "
                f"WHERE seed.page_id IN ({placeholders}) AND sibling.page_id IS NOT NULL "
                "ORDER BY seed.page_id,sibling.segment_index",
                starting,
            ).fetchall()
        # The entity explicitly requested at fault time is the section
        # address. Required Evidence keys describe why the MemoryRef was
        # created and are only a secondary directory hint; otherwise an
        # anchor in the PageSet header can out-rank the exact section named by
        # the model.
        primary_entities = tuple(dict.fromkeys(map(str, intent.entity_refs)))
        secondary_entities = tuple(
            entity
            for entity in dict.fromkeys(
                key.canonical_entity_id for key in intent.required_evidence
            )
            if entity not in primary_entities
        )
        primary_aliases = {
            alias.casefold()
            for entity in primary_entities
            for alias in (*self._request_aliases(entity), *self._entity_aliases(entity))
        }
        secondary_aliases = {
            alias.casefold()
            for entity in secondary_entities
            for alias in (*self._request_aliases(entity), *self._entity_aliases(entity))
        }
        primary_canonical = {entity.casefold() for entity in primary_entities}
        secondary_canonical = {entity.casefold() for entity in secondary_entities}
        purpose = " ".join(
            (intent.question, intent.purpose, intent.desired_detail)
        ).casefold()
        desired_kinds: set[str] = set()
        purpose_kinds = (
            (("test", "verify", "verification", "验收", "验证"), "VERIFY"),
            (("implement", "code", "change", "修改", "实现"), "IMPLEMENT"),
            (("fail", "error", "diagnos", "失败", "错误", "原因"), "DIAGNOSE"),
            (("plan", "milestone", "计划", "里程碑"), "PLAN"),
            (("depend", "依赖"), "DEPENDENCY"),
            (("decision", "rationale", "决策", "理由"), "DECIDE"),
        )
        for needles, kind in purpose_kinds:
            if any(needle in purpose for needle in needles):
                desired_kinds.add(kind)
        relation_kinds = {
            "VERIFIED_BY": {"VERIFY"},
            "CORRECTED_BY": {"DIAGNOSE", "IMPLEMENT"},
            "RESOLVED_BY": {"DIAGNOSE", "IMPLEMENT"},
            "DEPENDED_ON_BY": {"DEPENDENCY", "IMPLEMENT"},
            "SUPERSEDED_BY": {"IMPLEMENT"},
            "ADVANCES_TO": {"IMPLEMENT", "EXECUTE"},
        }
        for relation in (
            *intent.required_structural_relations,
            *intent.preferred_structural_relations,
        ):
            desired_kinds.update(relation_kinds.get(str(relation).upper(), set()))
        whole_group = any(
            word in purpose
            for word in ("whole", "entire", "complete event", "all segments", "完整事件", "全部分段")
        )
        query_terms = {
            term
            for term in re.findall(r"[\w./:-]+", purpose)
            if len(term) >= 4
        }
        ranked: dict[str, tuple[int, int, int]] = {}
        for row in rows:
            page_id = str(row["page_id"])
            if page_id in starting and not include_starting:
                continue
            entities = tuple(map(str, json.loads(str(row["entity_refs_json"]))))
            canonical_entities = {entity.casefold() for entity in entities}
            primary_exact_match = bool(primary_canonical.intersection(canonical_entities))
            secondary_exact_match = bool(secondary_canonical.intersection(canonical_entities))
            entry_aliases = {
                alias.casefold()
                for entity in entities
                for alias in self._entity_aliases(entity)
            }
            kinds = set(map(str, json.loads(str(row["semantic_kinds_json"]))))
            primary_entity_match = bool(primary_aliases.intersection(entry_aliases))
            secondary_entity_match = bool(secondary_aliases.intersection(entry_aliases))
            kind_matches = len(desired_kinds.intersection(kinds))
            searchable = (str(row["title"]) + " " + str(row["summary"])).casefold()
            lexical_matches = sum(term in searchable for term in query_terms)
            distance = abs(int(row["segment_index"]) - int(row["seed_index"]))
            score = (
                (
                    400
                    if primary_exact_match
                    else 300
                    if primary_entity_match
                    else 200
                    if secondary_exact_match
                    else 100
                    if secondary_entity_match
                    else 0
                )
                + 20 * kind_matches
                + min(lexical_matches, 5)
            )
            candidate = (score, -distance, -int(row["segment_index"]))
            if page_id not in ranked or candidate > ranked[page_id]:
                ranked[page_id] = candidate
        if not ranked:
            return ()
        if whole_group:
            selected = ranked
        else:
            exact = {page_id: rank for page_id, rank in ranked.items() if rank[0] >= 200}
            positive = exact or {
                page_id: rank for page_id, rank in ranked.items() if rank[0] > 0
            }
            if positive:
                best = max(positive, key=positive.__getitem__)
                selected = {best: positive[best]}
            elif allow_nearest:
                # No semantic directory entry matched. Open only the nearest
                # continuation as a bounded fallback; later graph traversal can
                # advance again if Coverage remains incomplete.
                nearest = max(ranked, key=ranked.__getitem__)
                selected = {nearest: ranked[nearest]}
            else:
                return ()
        return tuple(
            page_id
            for page_id, _ in sorted(selected.items(), key=lambda item: item[1], reverse=True)
        )[:limit]

    @staticmethod
    def _selected_keys(
        intent: RecallIntent, missing_key_digests: Sequence[str] | None
    ) -> tuple[Any, ...]:
        if missing_key_digests is None:
            return tuple(intent.required_evidence)
        requested = set(missing_key_digests)
        unknown = requested.difference(item.key_digest for item in intent.required_evidence)
        if unknown:
            raise ValueError(f"unknown missing EvidenceKey digests: {sorted(unknown)}")
        return tuple(item for item in intent.required_evidence if item.key_digest in requested)

    @staticmethod
    def _lineage_cte() -> str:
        return (
            "visible(branch_id, max_cursor) AS ("
            " SELECT ?, 9223372036854775807"
            " UNION ALL"
            " SELECT b.parent_branch_id, min(v.max_cursor, b.fork_cursor)"
            " FROM visible v JOIN v2_semantic_branches b"
            " ON b.repository_id=? AND b.run_id=? AND b.branch_id=v.branch_id"
            " WHERE b.parent_branch_id IS NOT NULL"
            ")"
        )

    def _lookup_rows(
        self,
        intent: RecallIntent,
        *,
        selected_keys: Sequence[Any],
        exact_key: bool,
    ) -> list[sqlite3.Row]:
        if not selected_keys:
            return []
        values = ",".join("(?,?,?,?,?,?,?)" for _ in selected_keys)
        required_params: list[Any] = []
        for key in selected_keys:
            required_params.extend(
                (
                    key.key_digest,
                    key.evidence_type.value,
                    key.canonical_entity_id,
                    key.semantic_role,
                    key.revision_constraint,
                    key.branch_scope,
                    key.validity_requirement,
                )
            )
        semantic_role_join = "AND e.semantic_role=r.semantic_role" if exact_key else ""
        revision_constraint_join = (
            "AND e.revision_constraint=r.revision_constraint" if exact_key else ""
        )
        branch_scope_filter = (
            "e.key_branch_scope=r.branch_scope"
            if exact_key
            else "(r.branch_scope IN ('LINEAGE', '*') OR e.key_branch_scope=r.branch_scope)"
        )
        sql = f"""
WITH RECURSIVE {self._lineage_cte()},
required(
    requested_digest, evidence_type, canonical_entity_id, semantic_role,
    revision_constraint, branch_scope, validity_requirement
) AS (VALUES {values}),
matches AS (
    SELECT DISTINCT
        r.requested_digest,
        e.key_digest AS stored_key_digest,
        e.evidence_id,
        e.authority,
        e.event_id,
        e.event_group_id,
        e.revision_id,
        a.anchor_id,
        a.event_start,
        a.event_end,
        p.page_id,
        p.payload_digest,
        p.branch_id,
        p.token_count,
        p.freshness_cursor
    FROM required r
    JOIN v2_semantic_evidence e
      ON e.evidence_type=r.evidence_type
     AND e.canonical_entity_id=r.canonical_entity_id
     {semantic_role_join}
     {revision_constraint_join}
     AND e.validity_requirement=r.validity_requirement
    JOIN visible v ON v.branch_id=e.branch_id
    JOIN v2_semantic_anchors a ON a.evidence_id=e.evidence_id
    JOIN v2_semantic_pages p ON p.page_id=a.page_id
    JOIN v2_semantic_page_revisions pr ON pr.page_id=p.page_id
    WHERE e.repository_id=?
      AND e.run_id=?
      AND p.repository_id=e.repository_id
      AND p.run_id=e.run_id
      AND p.branch_id=e.branch_id
      AND p.freshness_cursor <= v.max_cursor
      AND {branch_scope_filter}
      AND (?=0 OR r.validity_requirement<>'CURRENT' OR e.valid_to_cursor IS NULL)
      AND e.authority IN ('ASSERTED','DERIVED')
      AND a.page_digest=p.payload_digest
      AND length(p.redaction_proof)>0
      AND (?=0 OR (e.revision_id=? AND a.revision_id=? AND pr.revision_id=?))
),
ranked_pages AS (
    SELECT page_id,
           COUNT(DISTINCT requested_digest) AS marginal_gain,
           MAX(freshness_cursor) AS freshness,
           MAX(token_count) AS tokens
    FROM matches
    GROUP BY page_id
    ORDER BY (COUNT(DISTINCT requested_digest) * 1000000.0) / MAX(token_count) DESC,
             freshness DESC,
             page_id
    LIMIT ?
)
SELECT m.*
FROM matches m
JOIN ranked_pages rp ON rp.page_id=m.page_id
ORDER BY (rp.marginal_gain * 1000000.0) / rp.tokens DESC,
         rp.freshness DESC, m.page_id, m.event_start, m.evidence_id
"""
        parameters: list[Any] = [
            intent.branch_id,
            intent.repository_id,
            intent.run_id,
            *required_params,
            intent.repository_id,
            intent.run_id,
            int(intent.require_exact_revision),
            int(intent.require_exact_revision),
            intent.revision_id,
            intent.revision_id,
            intent.revision_id,
            intent.max_pages,
        ]
        with self._read_connection() as connection:
            return connection.execute(sql, parameters).fetchall()

    @staticmethod
    def _candidates_from_rows(rows: Sequence[sqlite3.Row]) -> tuple[PageCandidate, ...]:
        grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
        order: list[str] = []
        for row in rows:
            page_id = str(row["page_id"])
            if page_id not in grouped:
                order.append(page_id)
            grouped[page_id].append(row)
        candidates: list[PageCandidate] = []
        for page_id in order:
            values = grouped[page_id]
            candidates.append(
                PageCandidate(
                    page_id=page_id,
                    payload_digest=str(values[0]["payload_digest"]),
                    revision_id=str(values[0]["revision_id"]),
                    branch_id=str(values[0]["branch_id"]),
                    event_range=(
                        min(int(item["event_start"]) for item in values),
                        max(int(item["event_end"]) for item in values),
                    ),
                    anchor_ids=tuple(dict.fromkeys(str(item["anchor_id"]) for item in values)),
                    evidence_ids=tuple(dict.fromkeys(str(item["evidence_id"]) for item in values)),
                    evidence_key_digests=tuple(
                        dict.fromkeys(
                            [str(item["requested_digest"]) for item in values]
                            + [str(item["stored_key_digest"]) for item in values]
                        )
                    ),
                    estimated_tokens=int(values[0]["token_count"]),
                    freshness_cursor=int(values[0]["freshness_cursor"]),
                )
            )
        return tuple(candidates)

    def _assert_materialized_page_table(
        self,
        *,
        repository_id: str | None = None,
        run_id: str | None = None,
        page_id: str | None = None,
    ) -> None:
        clauses: list[str] = []
        parameters: list[str] = []
        if repository_id is not None:
            clauses.append("e.repository_id=?")
            parameters.append(repository_id)
        if run_id is not None:
            clauses.append("e.run_id=?")
            parameters.append(run_id)
        if page_id is not None:
            clauses.append("e.page_id=?")
            parameters.append(page_id)
        where = " AND ".join(clauses) or "1=1"
        with self._read_connection() as connection:
            broken = connection.execute(
                "SELECT e.evidence_id FROM v2_semantic_evidence e "
                "JOIN v2_semantic_anchors a ON a.evidence_id=e.evidence_id "
                "JOIN v2_semantic_pages p ON p.page_id=a.page_id "
                f"WHERE {where} AND (a.page_id<>e.page_id OR a.page_digest<>p.payload_digest "
                "OR NOT EXISTS (SELECT 1 FROM v2_semantic_edges located "
                " WHERE located.edge_type='LOCATED_AT' AND located.source_id=e.evidence_id "
                " AND located.target_id=a.anchor_id) "
                "OR NOT EXISTS (SELECT 1 FROM v2_semantic_edges stored "
                " WHERE stored.edge_type='STORED_IN' AND stored.source_id=a.anchor_id "
                " AND stored.target_id=p.page_id)) LIMIT 1",
                tuple(parameters),
            ).fetchone()
        if broken is not None:
            raise RuntimeError(
                "Semantic Graph/materialized page-table divergence for "
                f"Evidence {broken['evidence_id']}"
            )

    def keys_for_memory_need(
        self,
        *,
        run_id: str,
        branch_id: str,
        revision_id: str,
        milestone_identity_id: str | None = None,
        canonical_entities: Sequence[str] = (),
        evidence_types: Sequence[FactType] = (),
        include_historical: bool = False,
        limit: int = 8,
    ) -> tuple[EvidenceKey, ...]:
        """Compatibility wrapper returning keys from the alias-aware resolver."""

        return self.resolve_memory_need_entities(
            run_id=run_id,
            branch_id=branch_id,
            revision_id=revision_id,
            milestone_identity_id=milestone_identity_id,
            canonical_entities=canonical_entities,
            evidence_types=evidence_types,
            include_historical=include_historical,
            limit=limit,
        ).keys

    def resolve_memory_need_entities(
        self,
        *,
        run_id: str,
        branch_id: str,
        revision_id: str,
        milestone_identity_id: str | None = None,
        canonical_entities: Sequence[str] = (),
        evidence_types: Sequence[FactType] = (),
        include_historical: bool = False,
        limit: int = 8,
    ) -> EntityResolutionResult:
        """Translate model-facing names into exact EvidenceKeys without guessing.

        A unique alias is resolved directly.  Ambiguous aliases retain every
        bounded candidate so the recovered block can present the alternatives
        to the model; no path or symbol is silently selected.
        """

        if limit <= 0:
            return EntityResolutionResult((), {}, {}, tuple(map(str, canonical_entities)))
        requested = tuple(
            dict.fromkeys(str(item).strip() for item in canonical_entities if str(item).strip())
        )
        # Never turn an underspecified semantic request into a broad scan of
        # arbitrary current-Milestone evidence.  The model must name at least
        # one file/symbol/test/result reference (or use the legacy exact-key
        # transport handled above this resolver).
        if not requested:
            return EntityResolutionResult((), {}, {}, ())
        resolved: dict[str, tuple[str, ...]] = {}
        ambiguous: dict[str, tuple[str, ...]] = {}
        unresolved: list[str] = []
        candidate_entities: set[str] = set()
        for request in requested:
            request_aliases = self._request_aliases(request)
            placeholders = ",".join("?" for _ in request_aliases)
            exact = self.database.connection.execute(
                "SELECT DISTINCT canonical_entity_id FROM v2_semantic_evidence "
                f"WHERE run_id=? AND branch_id=? AND canonical_entity_id IN ({placeholders}) "
                "ORDER BY canonical_entity_id LIMIT 17",
                (run_id, branch_id, *request_aliases),
            ).fetchall()
            if exact:
                candidates = tuple(str(row["canonical_entity_id"]) for row in exact)
            else:
                alias_placeholders = ",".join("?" for _ in request_aliases)
                alias_rows = self.database.connection.execute(
                    "SELECT DISTINCT canonical_entity_id FROM v2_semantic_entity_aliases "
                    f"WHERE run_id=? AND branch_id=? AND alias IN ({alias_placeholders}) "
                    "ORDER BY canonical_entity_id LIMIT 17",
                    (run_id, branch_id, *request_aliases),
                ).fetchall()
                candidates = tuple(str(row["canonical_entity_id"]) for row in alias_rows)
            if not candidates:
                unresolved.append(request)
                continue
            resolved[request] = candidates
            equivalence = {
                self._file_address_equivalence(candidate) or f"exact:{candidate}"
                for candidate in candidates
            }
            if len(candidates) > 1 and len(equivalence) > 1:
                ambiguous[request] = candidates
                # Return ambiguity before opening any candidate Page. Reading
                # every match would be bounded, but it would not be precise
                # semantic-address translation.
                continue
            candidate_entities.update(candidates)

        type_set = {item.value for item in evidence_types}
        if not candidate_entities:
            return EntityResolutionResult((), resolved, ambiguous, tuple(unresolved))
        clauses = [
            "e.run_id=?",
            "e.branch_id=?",
            "(?=1 OR (e.revision_id=? AND e.valid_to_cursor IS NULL))",
        ]
        parameters: list[object] = [
            run_id,
            branch_id,
            int(include_historical),
            revision_id,
        ]
        entity_placeholders = ",".join("?" for _ in candidate_entities)
        clauses.append(f"e.canonical_entity_id IN ({entity_placeholders})")
        parameters.extend(sorted(candidate_entities))
        if type_set:
            type_placeholders = ",".join("?" for _ in type_set)
            clauses.append(f"e.evidence_type IN ({type_placeholders})")
            parameters.extend(sorted(type_set))
        parameters.extend((milestone_identity_id, revision_id, max(32, limit * 8)))
        rows = self.database.connection.execute(
            "WITH ranked AS (SELECT e.*, "
            "CASE WHEN ev.milestone_identity_id=? THEN 0 ELSE 1 END AS milestone_rank, "
            "CASE WHEN e.revision_id=? AND e.valid_to_cursor IS NULL "
            "THEN 0 ELSE 1 END AS revision_rank, "
            "ROW_NUMBER() OVER (PARTITION BY e.canonical_entity_id ORDER BY "
            "CASE WHEN ev.milestone_identity_id=? THEN 0 ELSE 1 END, "
            "CASE WHEN e.revision_id=? AND e.valid_to_cursor IS NULL "
            "THEN 0 ELSE 1 END,e.valid_from_cursor DESC,e.evidence_id) AS entity_rank "
            "FROM v2_semantic_evidence e "
            "JOIN v2_semantic_events ev ON ev.event_id=e.event_id WHERE "
            + " AND ".join(clauses)
            + ") SELECT * FROM ranked ORDER BY entity_rank,milestone_rank,revision_rank,"
            "valid_from_cursor DESC,canonical_entity_id,evidence_id LIMIT ?",
            (
                milestone_identity_id,
                revision_id,
                milestone_identity_id,
                revision_id,
                *parameters[:-3],
                parameters[-1],
            ),
        ).fetchall()
        result: list[EvidenceKey] = []
        seen: set[str] = set()
        for row in rows:
            if requested and str(row["canonical_entity_id"]) not in candidate_entities:
                continue
            if type_set and str(row["evidence_type"]) not in type_set:
                continue
            key = EvidenceKey(
                evidence_type=FactType(str(row["evidence_type"])),
                canonical_entity_id=str(row["canonical_entity_id"]),
                semantic_role=str(row["semantic_role"]),
                revision_constraint=str(row["revision_constraint"]),
                branch_scope=str(row["key_branch_scope"]),
                validity_requirement=str(row["validity_requirement"]),
            )
            if key.key_digest not in seen:
                seen.add(key.key_digest)
                result.append(key)
            if len(result) >= limit:
                break
        selected_entities = {item.canonical_entity_id for item in result}
        for request, candidates in tuple(resolved.items()):
            if request in ambiguous or any(
                candidate in selected_entities for candidate in candidates
            ):
                continue
            resolved.pop(request)
            unresolved.append(request)
        return EntityResolutionResult(
            tuple(result),
            resolved,
            ambiguous,
            tuple(dict.fromkeys(unresolved)),
        )

    def evidence_keys_for_pages(
        self,
        *,
        run_id: str,
        branch_id: str,
        revision_id: str,
        page_ids: Sequence[str],
        canonical_entities: Sequence[str] = (),
        evidence_types: Sequence[FactType] = (),
        include_historical: bool = True,
        limit: int = 8,
    ) -> tuple[EvidenceKey, ...]:
        """Bind a NONRESIDENT Context handle back to exact Page evidence.

        The Page IDs come from runtime-owned ContextHandles, never model text.
        Entity aliases only narrow those already-confirmed Pages; if an older
        Page predates alias production, its bounded evidence remains a safe
        address source for deterministic handle dereference.
        """

        pages = tuple(dict.fromkeys(str(item) for item in page_ids if str(item)))
        if not pages or limit <= 0:
            return ()
        requested = tuple(
            dict.fromkeys(str(item).strip() for item in canonical_entities if str(item).strip())
        )
        candidate_entities: set[str] = set()
        for value in requested:
            rows = self.database.connection.execute(
                "SELECT DISTINCT canonical_entity_id FROM v2_semantic_entity_aliases "
                "WHERE run_id=? AND branch_id=? AND alias=? ORDER BY canonical_entity_id",
                (run_id, branch_id, value),
            ).fetchall()
            candidate_entities.update(str(row["canonical_entity_id"]) for row in rows)
            candidate_entities.add(value)
        placeholders = ",".join("?" for _ in pages)
        rows = self.database.connection.execute(
            "SELECT * FROM v2_semantic_evidence WHERE run_id=? AND branch_id=? "
            f"AND page_id IN ({placeholders}) "
            "AND (?=1 OR (revision_id=? AND valid_to_cursor IS NULL)) "
            "ORDER BY CASE WHEN revision_id=? AND valid_to_cursor IS NULL THEN 0 ELSE 1 END, "
            "valid_from_cursor DESC,evidence_id",
            (
                run_id,
                branch_id,
                *pages,
                int(include_historical),
                revision_id,
                revision_id,
            ),
        ).fetchall()
        allowed_types = {item.value for item in evidence_types}
        matched_rows = [
            row
            for row in rows
            if (not allowed_types or str(row["evidence_type"]) in allowed_types)
            and (not candidate_entities or str(row["canonical_entity_id"]) in candidate_entities)
        ]
        # A confirmed handle is already a bounded semantic address.  Preserve
        # precise Page evidence when legacy aliases cannot narrow it.
        selected = matched_rows or [
            row for row in rows if not allowed_types or str(row["evidence_type"]) in allowed_types
        ]
        result: list[EvidenceKey] = []
        seen: set[str] = set()
        for row in selected:
            key = EvidenceKey(
                evidence_type=FactType(str(row["evidence_type"])),
                canonical_entity_id=str(row["canonical_entity_id"]),
                semantic_role=str(row["semantic_role"]),
                revision_constraint=str(row["revision_constraint"]),
                branch_scope=str(row["key_branch_scope"]),
                validity_requirement=str(row["validity_requirement"]),
            )
            if key.key_digest in seen:
                continue
            seen.add(key.key_digest)
            result.append(key)
            if len(result) >= limit:
                break
        return tuple(result)

    def locate_exact(self, intent: RecallIntent) -> ExactLookupResult:
        """Translate exact semantic keys without reading Page bodies.

        Candidate and hit data are address hints. Stage 4 must reopen and verify the
        Page body before it records Coverage.
        """

        self._assert_materialized_page_table(
            repository_id=intent.repository_id,
            run_id=intent.run_id,
        )
        rows = self._lookup_rows(intent, selected_keys=intent.required_evidence, exact_key=True)
        hits = tuple(
            ExactEvidenceHit(
                requested_key_digest=str(row["requested_digest"]),
                stored_key_digest=str(row["stored_key_digest"]),
                evidence_id=str(row["evidence_id"]),
                anchor_id=str(row["anchor_id"]),
                page_id=str(row["page_id"]),
                event_id=str(row["event_id"]),
                event_group_id=str(row["event_group_id"]),
                event_range=(int(row["event_start"]), int(row["event_end"])),
                revision_id=str(row["revision_id"]),
                authority=str(row["authority"]),
            )
            for row in rows
        )
        required = tuple(item.key_digest for item in intent.required_evidence)
        matched_set = {hit.requested_key_digest for hit in hits}
        matched = tuple(item for item in required if item in matched_set)
        missing = tuple(item for item in required if item not in matched_set)
        return ExactLookupResult(
            candidates=self._candidates_from_rows(rows),
            hits=hits,
            required_key_digests=required,
            matched_key_digests=matched,
            missing_key_digests=missing,
            executed_stage=FallbackStage.SEMANTIC_EXACT,
        )

    def structured_fallback(
        self,
        intent: RecallIntent,
        missing_key_digests: Sequence[str] | None = None,
    ) -> FallbackQueryResult:
        """Actually execute an indexed entity/type query with hard scope filters."""

        selected = self._selected_keys(intent, missing_key_digests)
        rows = self._lookup_rows(intent, selected_keys=selected, exact_key=False)
        return FallbackQueryResult(
            candidates=self._candidates_from_rows(rows),
            executed=True,
            executed_stage=FallbackStage.STRUCTURED_INDEX,
            reason="indexed type/entity query executed with branch/revision/validity filters",
        )

    @staticmethod
    def _fts_query_terms(intent: RecallIntent, selected_keys: Sequence[Any]) -> str:
        raw = " ".join(
            [intent.question]
            + [item.canonical_entity_id for item in selected_keys]
            + [item.semantic_role for item in selected_keys]
        )
        tokens = list(
            dict.fromkeys(token.lower() for token in re.findall(r"[A-Za-z0-9_./-]{2,}", raw))
        )[:24]
        return " OR ".join(f'"{token}"' for token in tokens)

    def metadata_fts(
        self,
        intent: RecallIntent,
        missing_key_digests: Sequence[str] | None = None,
    ) -> FallbackQueryResult:
        """Execute FTS5 only when available and when the query has safe terms."""

        selected = self._selected_keys(intent, missing_key_digests)
        query = self._fts_query_terms(intent, selected)
        if not self.fts_enabled:
            return FallbackQueryResult(
                candidates=(),
                executed=False,
                executed_stage=FallbackStage.NONE,
                reason="SQLite FTS5 is unavailable",
            )
        if not query:
            return FallbackQueryResult(
                candidates=(),
                executed=False,
                executed_stage=FallbackStage.NONE,
                reason="no safe metadata FTS terms",
            )
        sql = f"""
WITH RECURSIVE {self._lineage_cte()}
SELECT p.page_id, p.payload_digest, p.branch_id,
       p.event_start, p.event_end, p.token_count, p.freshness_cursor,
       COALESCE(MAX(pr.revision_id), '') AS revision_id
FROM v2_semantic_page_fts f
JOIN v2_semantic_pages p ON p.page_id=f.page_id
JOIN visible v ON v.branch_id=p.branch_id
JOIN v2_semantic_page_revisions pr ON pr.page_id=p.page_id
WHERE v2_semantic_page_fts MATCH ?
  AND p.repository_id=? AND p.run_id=?
  AND p.freshness_cursor <= v.max_cursor
  AND length(p.redaction_proof)>0
  AND (?=0 OR pr.revision_id=?)
GROUP BY p.page_id
ORDER BY p.freshness_cursor DESC, p.page_id
LIMIT ?
"""
        with self._read_connection() as connection:
            rows = connection.execute(
                sql,
                (
                    intent.branch_id,
                    intent.repository_id,
                    intent.run_id,
                    query,
                    intent.repository_id,
                    intent.run_id,
                    int(intent.require_exact_revision),
                    intent.revision_id,
                    intent.max_pages,
                ),
            ).fetchall()
        candidates = tuple(
            PageCandidate(
                page_id=str(row["page_id"]),
                payload_digest=str(row["payload_digest"]),
                revision_id=(
                    intent.revision_id if intent.require_exact_revision else str(row["revision_id"])
                ),
                branch_id=str(row["branch_id"]),
                event_range=(int(row["event_start"]), int(row["event_end"])),
                anchor_ids=(),
                evidence_ids=(),
                evidence_key_digests=tuple(item.key_digest for item in selected),
                estimated_tokens=int(row["token_count"]),
                freshness_cursor=int(row["freshness_cursor"]),
            )
            for row in rows
        )
        return FallbackQueryResult(
            candidates=candidates,
            executed=True,
            executed_stage=FallbackStage.PAGE_METADATA_FTS,
            reason="SQLite FTS5 metadata query executed with scope filters",
        )

    def recent_related(
        self,
        intent: RecallIntent,
        missing_key_digests: Sequence[str] | None = None,
    ) -> FallbackQueryResult:
        """Execute the bounded recent-related Page stage; never claim exact evidence."""

        selected = self._selected_keys(intent, missing_key_digests)
        entities = tuple(dict.fromkeys(item.canonical_entity_id for item in selected))
        if not entities:
            return FallbackQueryResult(
                candidates=(),
                executed=False,
                executed_stage=FallbackStage.NONE,
                reason="no canonical entities for recent-related lookup",
            )
        placeholders = ",".join("?" for _ in entities)
        sql = f"""
WITH RECURSIVE {self._lineage_cte()}
SELECT p.page_id, p.payload_digest, p.branch_id,
       p.event_start, p.event_end, p.token_count, p.freshness_cursor,
       MAX(pr.revision_id) AS revision_id
FROM v2_semantic_page_entities pe
JOIN v2_semantic_pages p ON p.page_id=pe.page_id
JOIN visible v ON v.branch_id=p.branch_id
JOIN v2_semantic_page_revisions pr ON pr.page_id=p.page_id
WHERE pe.canonical_entity_id IN ({placeholders})
  AND p.repository_id=? AND p.run_id=?
  AND p.freshness_cursor <= v.max_cursor
  AND (?=0 OR pr.revision_id=?)
GROUP BY p.page_id
ORDER BY p.freshness_cursor DESC
LIMIT ?
"""
        with self._read_connection() as connection:
            rows = connection.execute(
                sql,
                (
                    intent.branch_id,
                    intent.repository_id,
                    intent.run_id,
                    *entities,
                    intent.repository_id,
                    intent.run_id,
                    int(intent.require_exact_revision),
                    intent.revision_id,
                    intent.max_pages,
                ),
            ).fetchall()
        candidates = tuple(
            PageCandidate(
                page_id=str(row["page_id"]),
                payload_digest=str(row["payload_digest"]),
                revision_id=(
                    intent.revision_id if intent.require_exact_revision else str(row["revision_id"])
                ),
                branch_id=str(row["branch_id"]),
                event_range=(int(row["event_start"]), int(row["event_end"])),
                anchor_ids=(),
                evidence_ids=(),
                evidence_key_digests=tuple(item.key_digest for item in selected),
                estimated_tokens=int(row["token_count"]),
                freshness_cursor=int(row["freshness_cursor"]),
            )
            for row in rows
        )
        return FallbackQueryResult(
            candidates=candidates,
            executed=True,
            executed_stage=FallbackStage.RECENT_RELATED_PAGE,
            reason="bounded indexed entity-to-Page query executed",
        )

    def anchors_for_page(
        self,
        page_id: str,
        evidence_key_digests: Sequence[str],
    ) -> tuple[SemanticAnchor, ...]:
        """Return physical anchors; callers still verify the immutable Page body."""

        self._assert_materialized_page_table(page_id=page_id)
        keys = tuple(dict.fromkeys(evidence_key_digests))
        if not keys:
            return ()
        placeholders = ",".join("?" for _ in keys)
        with self._read_connection() as connection:
            rows = connection.execute(
                f"SELECT a.*, e.key_digest FROM v2_semantic_anchors a "
                "JOIN v2_semantic_evidence e ON e.evidence_id=a.evidence_id "
                f"WHERE a.page_id=? AND e.key_digest IN ({placeholders}) "
                "AND e.authority IN ('ASSERTED','DERIVED') "
                "ORDER BY a.event_start, a.anchor_id",
                (page_id, *keys),
            ).fetchall()
        return tuple(
            SemanticAnchor(
                anchor_id=str(row["anchor_id"]),
                evidence_id=str(row["evidence_id"]),
                page_id=str(row["page_id"]),
                page_digest=str(row["page_digest"]),
                event_ids=(str(row["event_id"]),),
                event_range=(int(row["event_start"]), int(row["event_end"])),
                event_group_id=str(row["event_group_id"]),
                blob_handle=(str(row["blob_handle"]) if row["blob_handle"] else None),
                blob_range=(
                    (int(row["blob_start"]), int(row["blob_end"]))
                    if row["blob_handle"] is not None
                    else None
                ),
                revision_id=str(row["revision_id"]),
                branch_id=str(row["branch_id"]),
            )
            for row in rows
        )

    def query_plan(self, run_id: str) -> tuple[sqlite3.Row, ...]:
        """Indexed diagnostic view used by startup/acceptance checks."""

        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT n.* FROM v2_semantic_nodes n "
                "WHERE n.run_id=? AND n.node_type IN "
                "('Task','Goal','PlanVersion','MilestoneIdentity','MilestoneVersion') "
                "ORDER BY n.created_cursor, n.node_type, n.node_id",
                (run_id,),
            ).fetchall()
        return tuple(rows)

    @staticmethod
    def _bounded_route_milestones(
        rows: Sequence[sqlite3.Row],
        edges: Sequence[sqlite3.Row],
        *,
        current_identity_id: str,
    ) -> tuple[sqlite3.Row, ...]:
        """Keep the complete graph external while materializing its active locality."""

        if len(rows) <= _ROUTE_MILESTONE_DETAIL_LIMIT:
            return tuple(rows)
        by_identity = {str(row["identity_id"]): row for row in rows}
        current_position = next(
            (
                index
                for index, row in enumerate(rows)
                if str(row["identity_id"]) == current_identity_id
            ),
            0,
        )
        selected: list[str] = []

        def include(identity_id: str) -> None:
            if (
                identity_id in by_identity
                and identity_id not in selected
                and len(selected) < _ROUTE_MILESTONE_DETAIL_LIMIT
            ):
                selected.append(identity_id)

        include(current_identity_id)
        # Direct prerequisites are the first routing locality around current.
        for edge in edges:
            if (
                str(edge["edge_type"]) == "DEPENDS_ON"
                and str(edge["source_id"]) == current_identity_id
            ):
                include(str(edge["target_id"]))
        # Preserve the recently completed handoff and the nearest future path.
        for row in reversed(rows[max(0, current_position - 3) : current_position]):
            include(str(row["identity_id"]))
        for row in rows[current_position + 1 : current_position + 6]:
            include(str(row["identity_id"]))
        # Keep a distant terminal target visible without expanding every middle node.
        include(str(rows[-1]["identity_id"]))
        # Repair/revalidation states are more important than inert PENDING history.
        for row in rows:
            if str(row["status"]) not in {"PENDING", "COMPLETED_VERIFIED"}:
                include(str(row["identity_id"]))
        # Fill remaining capacity by semantic distance from current, not creation age.
        for _, row in sorted(
            enumerate(rows),
            key=lambda item: (abs(item[0] - current_position), item[0]),
        ):
            include(str(row["identity_id"]))
        selected_set = set(selected)
        return tuple(row for row in rows if str(row["identity_id"]) in selected_set)

    @staticmethod
    def _bounded_route_steps(
        connection: sqlite3.Connection,
        rows: Sequence[sqlite3.Row],
        *,
        run_id: str,
        current_step_identity: str | None,
        milestone_current: bool,
        milestone_status: str,
    ) -> tuple[sqlite3.Row, ...]:
        """Select current/correction/recent/next Steps without losing current."""

        if len(rows) <= _ROUTE_STEP_DETAIL_LIMIT:
            return tuple(rows)
        by_identity = {str(row["step_identity_id"]): row for row in rows}
        selected: list[str] = []

        def include(identity_id: str) -> None:
            if (
                identity_id in by_identity
                and identity_id not in selected
                and len(selected) < _ROUTE_STEP_DETAIL_LIMIT
            ):
                selected.append(identity_id)

        current_position = next(
            (
                index
                for index, row in enumerate(rows)
                if str(row["step_identity_id"]) == current_step_identity
            ),
            None,
        )
        if milestone_current and current_step_identity is not None:
            include(current_step_identity)
            correction_rows = connection.execute(
                "SELECT failed_step_identity_id,corrective_step_identity_id,created_cursor "
                "FROM v2_plan_step_corrections WHERE run_id=? ORDER BY created_cursor",
                (run_id,),
            ).fetchall()
            adjacency: dict[str, set[str]] = defaultdict(set)
            for correction in correction_rows:
                failed = str(correction["failed_step_identity_id"])
                corrective = str(correction["corrective_step_identity_id"])
                if failed in by_identity and corrective in by_identity:
                    adjacency[failed].add(corrective)
                    adjacency[corrective].add(failed)
            chain = {current_step_identity}
            frontier = [current_step_identity]
            while frontier:
                node = frontier.pop()
                for related in adjacency.get(node, ()):
                    if related not in chain:
                        chain.add(related)
                        frontier.append(related)
            ordered_chain = [
                str(row["step_identity_id"])
                for row in rows
                if str(row["step_identity_id"]) in chain
            ]
            if ordered_chain:
                include(ordered_chain[0])
                for identity_id in reversed(ordered_chain[-6:]):
                    include(identity_id)
            if current_position is not None:
                for row in reversed(rows[max(0, current_position - 3) : current_position]):
                    include(str(row["step_identity_id"]))
                for row in rows[current_position + 1 : current_position + 5]:
                    include(str(row["step_identity_id"]))
        elif milestone_status == "COMPLETED_VERIFIED":
            for row in reversed(rows[-4:]):
                include(str(row["step_identity_id"]))
        else:
            for row in rows[:4]:
                include(str(row["step_identity_id"]))
        anchor = current_position if current_position is not None else 0
        for _, row in sorted(
            enumerate(rows),
            key=lambda item: (abs(item[0] - anchor), item[0]),
        ):
            include(str(row["step_identity_id"]))
        selected_set = set(selected)
        return tuple(row for row in rows if str(row["step_identity_id"]) in selected_set)

    def route_snapshot(
        self,
        run_id: str,
        branch_id: str,
        *,
        plan_version_id: str | None = None,
        pages_per_milestone: int = 2,
    ) -> Mapping[str, object]:
        """Render the bounded Semantic Graph route consumed at Milestone boundaries."""

        if pages_per_milestone < 0 or pages_per_milestone > 4:
            raise ValueError("pages_per_milestone must be between zero and four")
        with self._read_connection() as connection:

            def displayed_step_status(value: object) -> str:
                """Keep the legacy database value out of the navigation surface."""

                status = str(value)
                return "COMPLETED_OBSERVED" if status == "COMPLETED_VERIFIED" else status

            def has_table(name: str) -> bool:
                return (
                    connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,),
                    ).fetchone()
                    is not None
                )

            current = connection.execute(
                "SELECT target_id AS identity_id,plan_version_id FROM v2_semantic_edges "
                "WHERE run_id=? AND branch_id=? AND edge_type='CURRENT_MILESTONE' "
                "AND valid_to_cursor IS NULL",
                (run_id, branch_id),
            ).fetchone()
            if current is None:
                raise KeyError(f"Run has no current Milestone: {run_id}")
            current_step_edge = connection.execute(
                "SELECT target_id FROM v2_semantic_edges WHERE run_id=? AND branch_id=? "
                "AND edge_type='CURRENT_STEP' AND valid_to_cursor IS NULL",
                (run_id, branch_id),
            ).fetchone()
            current_step_identity = (
                str(current_step_edge["target_id"]) if current_step_edge is not None else None
            )
            current_step_row = (
                connection.execute(
                    "SELECT canonical_step_id FROM v2_plan_step_identities "
                    "WHERE step_identity_id=? AND run_id=?",
                    (current_step_identity, run_id),
                ).fetchone()
                if current_step_identity is not None
                else None
            )
            current_step_canonical_id = (
                str(current_step_row["canonical_step_id"]) if current_step_row is not None else None
            )
            active_plan = plan_version_id or str(current["plan_version_id"])
            plan = connection.execute(
                "SELECT pv.plan_version_id,pv.version_number,pv.previous_plan_version_id,"
                "pv.plan_json,g.goal_text FROM v2_plan_versions pv "
                "JOIN v2_goals g ON g.goal_id=pv.goal_id "
                "JOIN v2_tasks t ON t.task_id=pv.task_id "
                "WHERE pv.plan_version_id=? AND t.run_id=?",
                (active_plan, run_id),
            ).fetchone()
            if plan is None:
                raise KeyError(f"unknown PlanVersion for Run: {active_plan}")
            plan_spec = PlanSpec.from_dict(json.loads(str(plan["plan_json"])))
            milestone_specs = {item.canonical_id: item for item in plan_spec.milestones}
            all_milestone_rows = connection.execute(
                "SELECT pm.ordinal,mi.identity_id,mi.canonical_id,mv.title,mv.description,"
                "(SELECT mse.status FROM v2_milestone_state_events mse "
                " WHERE mse.identity_id=mi.identity_id ORDER BY mse.created_cursor DESC LIMIT 1) "
                "AS status FROM v2_plan_milestones pm "
                "JOIN v2_milestone_identities mi ON mi.identity_id=pm.identity_id "
                "JOIN v2_milestone_versions mv ON mv.version_id=pm.milestone_version_id "
                "WHERE pm.plan_version_id=? ORDER BY pm.ordinal",
                (active_plan,),
            ).fetchall()
            canonical_by_identity = {
                str(row["identity_id"]): str(row["canonical_id"]) for row in all_milestone_rows
            }
            all_edge_rows = connection.execute(
                "SELECT edge_type,source_id,target_id,authority FROM v2_semantic_edges "
                "WHERE run_id=? AND branch_id=? AND plan_version_id=? "
                "AND valid_to_cursor IS NULL AND edge_type IN ('PRECEDES','DEPENDS_ON') "
                "ORDER BY edge_type,source_id,target_id",
                (run_id, branch_id, active_plan),
            ).fetchall()
            milestone_rows = self._bounded_route_milestones(
                all_milestone_rows,
                all_edge_rows,
                current_identity_id=str(current["identity_id"]),
            )
            detailed_milestone_ids = {str(row["identity_id"]) for row in milestone_rows}
            route_edges = [
                {
                    "edge_type": str(row["edge_type"]),
                    "source": canonical_by_identity.get(
                        str(row["source_id"]), str(row["source_id"])
                    ),
                    "target": canonical_by_identity.get(
                        str(row["target_id"]), str(row["target_id"])
                    ),
                    "authority": str(row["authority"]),
                }
                for row in all_edge_rows
                if str(row["source_id"]) in detailed_milestone_ids
                and str(row["target_id"]) in detailed_milestone_ids
            ]
            route_edge_counts = {
                edge_type: sum(1 for row in all_edge_rows if str(row["edge_type"]) == edge_type)
                for edge_type in ("PRECEDES", "DEPENDS_ON")
            }
            milestone_status_counts = {
                status: sum(1 for row in all_milestone_rows if str(row["status"]) == status)
                for status in sorted({str(row["status"]) for row in all_milestone_rows})
            }
            milestones: list[dict[str, object]] = []
            for row in milestone_rows:
                identity_id = str(row["identity_id"])
                milestone_spec = milestone_specs[str(row["canonical_id"])]
                dependencies = tuple(
                    edge["target"]
                    for edge in route_edges
                    if edge["edge_type"] == "DEPENDS_ON"
                    and edge["source"] == str(row["canonical_id"])
                )
                page_rows = (
                    connection.execute(
                        "SELECT p.page_id,p.payload_digest,d.delta_summary,d.outcome,"
                        "d.delta_kinds_json,d.changed_files_json,d.changed_symbols_json,"
                        "d.test_refs_json,p.freshness_cursor,"
                        "COALESCE((SELECT json_group_array(pe.canonical_entity_id) "
                        "FROM v2_semantic_page_entities pe WHERE pe.page_id=p.page_id),'[]') "
                        "AS entity_refs_json "
                        "FROM v2_semantic_edges e "
                        "JOIN v2_semantic_pages p ON p.page_id=e.target_id "
                        "JOIN v2_semantic_page_descriptors d ON d.page_id=p.page_id "
                        "WHERE e.run_id=? AND e.branch_id=? AND e.edge_type='CONTAINS_PAGE' "
                        "AND e.source_id=? AND e.valid_to_cursor IS NULL "
                        "ORDER BY p.freshness_cursor DESC LIMIT ?",
                        (run_id, branch_id, identity_id, pages_per_milestone),
                    ).fetchall()
                    if pages_per_milestone
                    else ()
                )
                all_step_rows = connection.execute(
                    "SELECT si.step_identity_id,si.canonical_step_id,si.title,si.corrective,"
                    "si.criterion_ids_json,si.entity_refs_json,"
                    "si.historical_dependency_refs_json,si.source_plan_item_ids_json,"
                    "sc.expected_outcome,"
                    "sc.minimum_acceptance_json,sc.failure_signals_json,"
                    "sc.contract_revision_number,"
                    "si.created_cursor,"
                    "(SELECT sse.status FROM v2_plan_step_state_events sse "
                    " WHERE sse.step_identity_id=si.step_identity_id "
                    " ORDER BY sse.created_cursor DESC LIMIT 1) status "
                    "FROM v2_plan_step_identities si "
                    "JOIN v2_effective_plan_step_contracts sc "
                    "ON sc.step_identity_id=si.step_identity_id "
                    "WHERE si.run_id=? AND si.milestone_identity_id=? "
                    "ORDER BY si.created_cursor",
                    (run_id, identity_id),
                ).fetchall()
                step_rows = self._bounded_route_steps(
                    connection,
                    all_step_rows,
                    run_id=run_id,
                    current_step_identity=current_step_identity,
                    milestone_current=identity_id == str(current["identity_id"]),
                    milestone_status=str(row["status"]),
                )
                step_status_counts: dict[str, int] = {}
                for item in all_step_rows:
                    display_status = displayed_step_status(item["status"])
                    step_status_counts[display_status] = (
                        step_status_counts.get(display_status, 0) + 1
                    )
                steps: list[dict[str, object]] = []
                for step_row in step_rows:
                    review_rows = connection.execute(
                        "SELECT decision,summary,evidence_event_ids_json,created_step_ids_json "
                        "FROM v2_plan_step_review_events WHERE step_identity_id=? "
                        "ORDER BY created_cursor DESC LIMIT 2",
                        (step_row["step_identity_id"],),
                    ).fetchall()
                    steps.append(
                        {
                            "step_id": str(step_row["canonical_step_id"]),
                            "step_identity_id": str(step_row["step_identity_id"]),
                            "title": str(step_row["title"]),
                            "status": displayed_step_status(step_row["status"]),
                            "current": (str(step_row["step_identity_id"]) == current_step_identity),
                            "corrective": bool(step_row["corrective"]),
                            "criterion_ids": json.loads(str(step_row["criterion_ids_json"])),
                            "entity_refs": json.loads(str(step_row["entity_refs_json"])),
                            "historical_dependency_refs": json.loads(
                                str(step_row["historical_dependency_refs_json"])
                            ),
                            "source_plan_item_ids": json.loads(
                                str(step_row["source_plan_item_ids_json"])
                            ),
                            "expected_outcome": str(step_row["expected_outcome"]),
                            "minimum_acceptance": json.loads(
                                str(step_row["minimum_acceptance_json"])
                            ),
                            "failure_signals": json.loads(str(step_row["failure_signals_json"])),
                            "contract_revision_number": int(
                                step_row["contract_revision_number"]
                            ),
                            "latest_reviews": [
                                {
                                    "decision": str(review["decision"]),
                                    "summary": str(review["summary"]),
                                    "evidence_event_ids": json.loads(
                                        str(review["evidence_event_ids_json"])
                                    ),
                                    "created_step_ids": json.loads(
                                        str(review["created_step_ids_json"])
                                    ),
                                }
                                for review in reversed(review_rows)
                            ],
                        }
                    )
                milestones.append(
                    {
                        "canonical_id": str(row["canonical_id"]),
                        "identity_id": identity_id,
                        "ordinal": int(row["ordinal"]),
                        "title": str(row["title"]),
                        "description": str(row["description"]),
                        "target_outcome": milestone_spec.target_outcome,
                        "minimum_acceptance": primitive(milestone_spec.minimum_acceptance),
                        "downstream_assumptions": list(milestone_spec.downstream_assumptions),
                        "non_goals": list(milestone_spec.non_goals),
                        "source_plan_item_ids": list(
                            milestone_spec.source_plan_item_ids
                        ),
                        "status": str(row["status"]),
                        "current": identity_id == str(current["identity_id"]),
                        "depends_on": dependencies,
                        "steps": steps,
                        "step_window": {
                            "selection": "CURRENT_CORRECTION_RECENT_NEXT",
                            "total_count": len(all_step_rows),
                            "included_count": len(step_rows),
                            "omitted_count": len(all_step_rows) - len(step_rows),
                            "status_counts": step_status_counts,
                            "complete_history_external": True,
                        },
                        "latest_page_deltas": [
                            {
                                "memory_ref": memory_ref_for_page(
                                    str(page["page_id"]), str(page["payload_digest"])
                                ),
                                "summary": str(page["delta_summary"])[:600],
                                "outcome": str(page["outcome"]),
                                "delta_kinds": json.loads(str(page["delta_kinds_json"])),
                                "changed_files": json.loads(str(page["changed_files_json"])),
                                "changed_symbols": json.loads(
                                    str(page["changed_symbols_json"])
                                ),
                                "test_refs": json.loads(str(page["test_refs_json"])),
                                "entity_refs": json.loads(str(page["entity_refs_json"])),
                            }
                            for page in reversed(page_rows)
                        ],
                    }
                )
            workspace_row = (
                connection.execute(
                    "SELECT revision_id,source_event_id,updated_at "
                    "FROM v2_current_workspace_revision WHERE run_id=? AND branch_id=?",
                    (run_id, branch_id),
                ).fetchone()
                if has_table("v2_current_workspace_revision")
                else None
            )
            workspace_revision = (
                str(workspace_row["revision_id"]) if workspace_row is not None else None
            )
            changed_files = (
                [
                    str(row["relative_path"])
                    for row in connection.execute(
                        "SELECT relative_path FROM v2_workspace_revision_changes "
                        "WHERE run_id=? AND branch_id=? AND revision_id=? "
                        "ORDER BY relative_path LIMIT 24",
                        (run_id, branch_id, workspace_revision),
                    ).fetchall()
                ]
                if workspace_revision is not None and has_table("v2_workspace_revision_changes")
                else []
            )
            changed_symbols = (
                [
                    str(row["canonical_entity_id"])
                    for row in connection.execute(
                        "SELECT DISTINCT canonical_entity_id FROM v2_reference_bindings "
                        "WHERE repository_id=(SELECT repository_id FROM v2_tasks "
                        "WHERE run_id=? AND branch_id=?) AND observed_revision_id=? "
                        "AND change_scope='CHANGED_SYMBOL' "
                        "ORDER BY canonical_entity_id LIMIT 24",
                        (run_id, branch_id, workspace_revision),
                    ).fetchall()
                ]
                if workspace_revision is not None and has_table("v2_reference_bindings")
                else []
            )
            working_set = (
                [
                    {
                        "kind": str(row["kind"]),
                        "value": str(row["value"]),
                        "heat": str(row["heat"]),
                        "revision_id": str(row["revision_id"]),
                    }
                    for row in connection.execute(
                        "SELECT kind,value,heat,revision_id FROM v2_working_set "
                        "WHERE run_id=? ORDER BY CASE heat WHEN 'HOT' THEN 0 ELSE 1 END,kind,value "
                        "LIMIT 32",
                        (run_id,),
                    ).fetchall()
                ]
                if has_table("v2_working_set")
                else []
            )
            open_page = (
                connection.execute(
                    "SELECT COUNT(*) AS group_count,MIN(event_start) AS event_start,"
                    "MAX(event_end) AS event_end FROM v2_page_wal_groups "
                    "WHERE run_id=? AND branch_id=? AND page_id IS NULL",
                    (run_id, branch_id),
                ).fetchone()
                if has_table("v2_page_wal_groups")
                else None
            )
        return {
            "schema": "codex-longterm-v2/semantic-route-snapshot@5",
            "source": "SEMANTIC_GRAPH_MATERIALIZATION",
            "goal": str(plan["goal_text"]),
            "plan_version_id": str(plan["plan_version_id"]),
            "plan_version_number": int(plan["version_number"]),
            "previous_plan_version_id": (
                None
                if plan["previous_plan_version_id"] is None
                else str(plan["previous_plan_version_id"])
            ),
            "final_acceptance": primitive(plan_spec.final_acceptance),
            "current_milestone_id": canonical_by_identity.get(
                str(current["identity_id"]), str(current["identity_id"])
            ),
            "current_step_id": current_step_canonical_id,
            "route_edges": route_edges,
            "route_window": {
                "selection": "CURRENT_DEPENDENCIES_RECENT_NEXT_TERMINAL",
                "total_milestone_count": len(all_milestone_rows),
                "included_milestone_count": len(milestone_rows),
                "omitted_milestone_count": len(all_milestone_rows) - len(milestone_rows),
                "included_milestone_ids": [str(row["canonical_id"]) for row in milestone_rows],
                "milestone_status_counts": milestone_status_counts,
                "edge_type_counts": route_edge_counts,
                "complete_graph_external": True,
            },
            "milestones": milestones,
            "workspace_state": {
                "revision_id": workspace_revision,
                "receipt_source_event_id": (
                    str(workspace_row["source_event_id"]) if workspace_row is not None else None
                ),
                "changed_files": changed_files,
                "changed_symbols": changed_symbols,
            },
            "working_set": working_set,
            "open_page_state": {
                "group_count": int(open_page["group_count"]) if open_page is not None else 0,
                "event_range": (
                    [int(open_page["event_start"]), int(open_page["event_end"])]
                    if open_page is not None and open_page["event_start"] is not None
                    else None
                ),
                "detail_state": "FORMING_SEMANTIC_PAGE",
            },
        }

    def transition_handoff_facts(
        self,
        *,
        run_id: str,
        branch_id: str,
        current_step_id: str,
        predecessor_step_id: str | None = None,
        limit: int = 6,
        max_summary_chars: int = 4800,
    ) -> tuple[Mapping[str, object], ...]:
        """Return the bounded semantic handoff at one active Step transition.

        Provider compaction may evict physical Tokens while immutable Pages and
        the TPG remain intact.  A continuation needs the already-established
        semantic delta for the active Step, not a replay of raw Pages and not a
        fresh repository audit.  This query therefore exposes only live,
        structured conclusions, rejected hypotheses, unresolved questions,
        workspace changes and verification outcomes from the active Step and
        its immediate verified predecessor. Generic progress narration and
        intended next actions are deliberately excluded. Every returned fact
        retains the stable MemoryRef of its authoritative Page for exact-detail
        faults.
        """

        current_step = current_step_id.strip()
        predecessor_step = (predecessor_step_id or "").strip()
        step_ids = tuple(
            dict.fromkeys(item for item in (current_step, predecessor_step) if item)
        )
        if not step_ids or limit <= 0 or max_summary_chars <= 0:
            return ()
        placeholders = ",".join("?" for _ in step_ids)
        rows = self.database.connection.execute(
            "SELECT e.evidence_type,e.semantic_role,e.canonical_entity_id,e.content_json,"
            "e.content_digest,e.revision_id,e.valid_from_cursor,e.page_id,"
            "p.payload_digest,ev.event_position FROM v2_semantic_evidence e "
            "JOIN v2_semantic_pages p ON p.page_id=e.page_id "
            "JOIN v2_semantic_events ev ON ev.event_id=e.event_id "
            "WHERE e.run_id=? AND e.branch_id=? AND e.valid_to_cursor IS NULL "
            f"AND json_extract(e.content_json,'$.plan_step_id') IN ({placeholders}) "
            "AND (e.evidence_type IN "
            "('IMPLEMENTATION_DECISION','UNRESOLVED_QUESTION','TEST_RESULT',"
            "'VERIFIER_RESULT','TEST_FAILURE','CODE_CHANGE') "
            "OR (e.evidence_type='CODE_OBSERVATION' "
            "AND e.semantic_role!='agent_progress_observation')) "
            "ORDER BY CASE json_extract(e.content_json,'$.plan_step_id') "
            "WHEN ? THEN 0 ELSE 1 END,ev.event_position DESC,e.valid_from_cursor DESC",
            (run_id, branch_id, *step_ids, current_step),
        ).fetchall()
        candidates: list[dict[str, object]] = []
        seen_digests: set[str] = set()
        for row in rows:
            content_digest = str(row["content_digest"])
            if content_digest in seen_digests:
                continue
            try:
                value = json.loads(str(row["content_json"]))
            except json.JSONDecodeError:
                continue
            if not isinstance(value, Mapping):
                continue
            evidence_type = str(row["evidence_type"])
            semantic_role = str(row["semantic_role"])
            summary = self._transition_handoff_summary(
                evidence_type=evidence_type,
                semantic_role=semantic_role,
                value=value,
            )[:1200]
            if not summary:
                continue
            raw_entity_refs = value.get("entity_refs", ())
            if isinstance(raw_entity_refs, str):
                candidate_entity_refs: Sequence[object] = (raw_entity_refs,)
            elif isinstance(raw_entity_refs, Sequence):
                candidate_entity_refs = raw_entity_refs
            else:
                candidate_entity_refs = ()
            entity_refs = tuple(
                dict.fromkeys(
                    item
                    for item in (
                        *(str(ref).strip() for ref in candidate_entity_refs),
                        str(row["canonical_entity_id"]).strip(),
                    )
                    if item
                )
            )[:8]
            step_id = str(value.get("plan_step_id", "")).strip()
            candidates.append(
                {
                    "selection_index": len(candidates),
                    "step_id": step_id,
                    "step_role": (
                        "ACTIVE_STEP" if step_id == current_step else "PREDECESSOR_STEP"
                    ),
                    "evidence_type": evidence_type,
                    "semantic_role": semantic_role,
                    "handoff_lane": self._transition_handoff_lane(
                        evidence_type,
                        semantic_role,
                    ),
                    "purpose": str(value.get("purpose", ""))[:400],
                    "summary": summary,
                    "entity_refs": entity_refs,
                    "workspace_revision_id": str(row["revision_id"]),
                    "event_position": int(row["event_position"]),
                    "raw_detail_memory_ref": memory_ref_for_page(
                        str(row["page_id"]), str(row["payload_digest"])
                    ),
                }
            )
            seen_digests.add(content_digest)
        selected: list[dict[str, object]] = []
        selected_ids: set[int] = set()
        lane_order = ("verification", "workspace_change", "decision", "unresolved")
        allowed_lanes = frozenset(lane_order)
        for lane in lane_order:
            candidate = next(
                (
                    item
                    for item in candidates
                    if item["handoff_lane"] == lane
                    and int(item["selection_index"]) not in selected_ids
                ),
                None,
            )
            if candidate is None:
                continue
            selected.append(candidate)
            selected_ids.add(int(candidate["selection_index"]))
            if len(selected) >= limit:
                break
        for candidate in candidates:
            if len(selected) >= limit:
                break
            if int(candidate["selection_index"]) in selected_ids:
                continue
            if candidate["handoff_lane"] not in allowed_lanes:
                continue
            selected.append(candidate)
            selected_ids.add(int(candidate["selection_index"]))
        selected.sort(
            key=lambda item: (
                0 if item["step_role"] == "ACTIVE_STEP" else 1,
                -int(item["event_position"]),
            )
        )
        facts: list[Mapping[str, object]] = []
        summary_characters = 0
        for selected_fact in selected:
            available = max_summary_chars - summary_characters
            if available <= 0:
                break
            summary = str(selected_fact["summary"])
            if len(summary) > available:
                if available < 80:
                    break
                summary = summary[:available]
            fact = {
                key: value
                for key, value in selected_fact.items()
                if key not in {"selection_index", "event_position", "handoff_lane"}
            }
            fact["summary"] = summary
            facts.append(fact)
            summary_characters += len(summary)
        return tuple(facts)

    @staticmethod
    def _transition_handoff_lane(evidence_type: str, semantic_role: str) -> str:
        if evidence_type in {
            FactType.TEST_RESULT.value,
            FactType.VERIFIER_RESULT.value,
            FactType.TEST_FAILURE.value,
        }:
            return "verification"
        if evidence_type == FactType.CODE_CHANGE.value:
            return "workspace_change"
        if evidence_type == FactType.IMPLEMENTATION_DECISION.value or (
            evidence_type == FactType.CODE_OBSERVATION.value
            and semantic_role != "agent_progress_observation"
        ):
            return "decision"
        if evidence_type == FactType.UNRESOLVED_QUESTION.value:
            return "unresolved"
        return "excluded"

    @staticmethod
    def _transition_executable_surface(value: Mapping[str, object]) -> str:
        """Extract bounded code-facing detail when a fact already carries it."""

        fields = (
            ("path", 320),
            ("accessed_paths", 420),
            ("symbol", 320),
            ("line_range", 120),
            ("signature", 520),
            ("call_chain", 900),
            ("command", 520),
            ("code_excerpt", 1800),
            ("source_excerpt", 1800),
            # Provider read receipts already carry the exact bounded output
            # even when the model did not repeat it in record_semantic_update.
            # Carry a small prefix into the transition handoff so compaction
            # leaves a usable edit surface instead of only an opaque ref.
            ("complete_output", 2400),
            ("output_excerpt", 1200),
            ("detail", 1200),
        )
        parts: list[str] = []
        for key, limit in fields:
            raw = value.get(key)
            if raw is None:
                continue
            if isinstance(raw, (list, tuple)):
                text = " -> ".join(str(item) for item in raw)
            elif isinstance(raw, Mapping):
                text = json.dumps(raw, ensure_ascii=False, sort_keys=True)
            else:
                text = str(raw)
            text = " ".join(text.split())[:limit]
            if text:
                parts.append(f"{key}={text}")
        return "; ".join(parts)[:4200]

    @staticmethod
    def _transition_handoff_summary(
        *,
        evidence_type: str,
        semantic_role: str,
        value: Mapping[str, object],
    ) -> str:
        """Render one bounded local fact without opening the Page body."""

        def compact(raw: object, limit: int) -> str:
            return " ".join(str(raw or "").split())[:limit]

        if evidence_type in {
            FactType.TEST_RESULT.value,
            FactType.TEST_FAILURE.value,
        }:
            command = compact(value.get("logical_command") or value.get("command"), 520)
            success = bool(value.get("success", evidence_type == FactType.TEST_RESULT.value))
            reliable = value.get("success_exit_status_reliable") is not False
            outcome = "passed" if success else "failed"
            reliability = "reliable exit" if reliable else "UNRELIABLE/MASKED exit"
            output = compact(value.get("output_excerpt"), 360)
            rendered = f"Test {outcome} ({reliability})"
            if command:
                rendered += f": {command}"
            if value.get("exit_code") is not None:
                rendered += f" [exit={value['exit_code']}]"
            if output:
                rendered += f". Output: {output}"
            return rendered
        if evidence_type == FactType.VERIFIER_RESULT.value:
            summary = compact(value.get("summary"), 900)
            if summary:
                return f"Verifier result: {summary}"
            return "Verifier result: " + compact(
                value.get("verdict") or value.get("status") or value.get("success"),
                500,
            )
        if evidence_type == FactType.CODE_OBSERVATION.value:
            surface = SemanticStore._transition_executable_surface(value)
            summary = compact(value.get("summary"), 1200)
            rendered = "Code observation"
            if summary:
                rendered += f": {summary}"
            if surface:
                rendered += f". Executable surface: {surface}"
            return rendered[:3000]
        if evidence_type == FactType.CODE_CHANGE.value:
            path = compact(value.get("path"), 500)
            revision = compact(value.get("revision_id"), 160)
            kind = compact(value.get("change_kind"), 120)
            rendered = f"Workspace changed: {path or 'addressed code entity'}"
            if kind:
                rendered += f" ({kind})"
            if revision:
                rendered += f" at revision {revision}"
            return rendered
        if evidence_type == FactType.TOOL_RESULT.value:
            command = compact(value.get("command"), 520)
            output = compact(value.get("output_excerpt"), 300)
            rendered = f"Tool result: {command or semantic_role}"
            if value.get("exit_code") is not None:
                rendered += f" [exit={value['exit_code']}]"
            if output:
                rendered += f". Output: {output}"
            return rendered
        summary = compact(value.get("summary"), 1000)
        if summary:
            return summary
        return compact(
            value.get("decision")
            or value.get("message_excerpt")
            or value.get("content"),
            1000,
        )

    def latest_working_plan_observation(
        self,
        run_id: str,
        branch_id: str,
        *,
        limit: int = 12,
    ) -> Mapping[str, object] | None:
        """Return the latest native Codex working Plan as navigation only.

        Provider ``update_plan`` items are already durable Page facts.  They
        describe the model's natural local route, but they are neither TPG
        Step identities nor acceptance authority.  Exposing the latest
        snapshot here lets an Epoch continue the same line of work without
        turning those provider-local items into another control state machine.
        """

        if limit <= 0:
            return None
        rows = self.database.connection.execute(
            "SELECT content_json,valid_from_cursor,evidence_id "
            "FROM v2_semantic_evidence WHERE run_id=? AND branch_id=? "
            "AND evidence_type='PLAN_DECISION' "
            "AND semantic_role='provider_plan_observation' "
            "AND valid_to_cursor IS NULL "
            "ORDER BY valid_from_cursor DESC,evidence_id DESC LIMIT 96",
            (run_id, branch_id),
        ).fetchall()
        decoded: list[Mapping[str, object]] = []
        latest_source_event_id = ""
        for row in rows:
            try:
                value = json.loads(str(row["content_json"]))
            except json.JSONDecodeError:
                continue
            if not isinstance(value, Mapping):
                continue
            source_event_id = str(value.get("source_event_id", "")).strip()
            if not source_event_id:
                continue
            if not latest_source_event_id:
                latest_source_event_id = source_event_id
            if source_event_id == latest_source_event_id:
                decoded.append(value)
        if not decoded:
            return None
        decoded.sort(
            key=lambda value: (
                int(value.get("ordinal", 0) or 0),
                str(value.get("step", "")),
            )
        )
        explanation = next(
            (
                str(value.get("explanation", "")).strip()
                for value in decoded
                if str(value.get("explanation", "")).strip()
            ),
            "",
        )
        return {
            "source": "CODEX_NATIVE_PLAN_OBSERVATION",
            "authority": "NAVIGATION_ONLY",
            "source_event_id": latest_source_event_id,
            "explanation": explanation[:1_200],
            "items": [
                {
                    "title": str(value.get("step", ""))[:800],
                    "status": str(value.get("status", "")),
                }
                for value in decoded[:limit]
                if str(value.get("step", "")).strip()
            ],
        }

    def execution_route_card(
        self,
        run_id: str,
        branch_id: str,
        *,
        plan_version_id: str | None = None,
    ) -> Mapping[str, object]:
        """Project the full TPG into one bounded model-visible working card.

        The Semantic Graph remains complete and authoritative in external
        storage.  A continuation Turn needs only the current route node, its
        observable outcomes, nearby skeleton nodes, stable Page addresses,
        and the current workspace receipt. Internal Criterion, Evidence,
        verifier and graph identities stay in the TPG; exposing them would
        turn the page table into a second control protocol for the model.
        """

        snapshot = self.route_snapshot(
            run_id,
            branch_id,
            plan_version_id=plan_version_id,
            # Keep a small candidate set per nearby route node, then let the
            # hot-path delta choose the four addresses most relevant to the
            # current Step. A single newest Page is often only an acceptance
            # receipt while the editable code surface lives one Page earlier.
            pages_per_milestone=3,
        )
        milestone_values = snapshot.get("milestones", ())
        milestones = [
            dict(item) for item in milestone_values if isinstance(item, Mapping)
        ]
        current_id = str(snapshot.get("current_milestone_id", ""))
        current = next(
            (item for item in milestones if str(item.get("canonical_id", "")) == current_id),
            None,
        )
        if current is None:
            raise RuntimeError("TPG Route Card has no current Milestone node")
        raw_steps = current.get("steps", ())
        steps = [dict(item) for item in raw_steps if isinstance(item, Mapping)]
        current_step = next((item for item in steps if bool(item.get("current"))), None)
        current_position = steps.index(current_step) if current_step in steps else -1

        def observable_claims(values: object) -> list[dict[str, object]]:
            if not isinstance(values, (list, tuple)):
                return []
            claims: list[dict[str, object]] = []
            for value in values:
                if not isinstance(value, Mapping):
                    continue
                claims.append(
                    {
                        "requirement": str(value.get("requirement_text", ""))[:1200],
                        "observable_outcome": str(
                            value.get("observable_outcome", "")
                        )[:1200],
                        "claim_type": str(value.get("claim_type", "")),
                        "entity_refs": list(value.get("entity_refs", ()))[:8],
                        "test_selectors": list(value.get("test_selectors", ()))[:8],
                    }
                )
            return claims

        upcoming_steps = [
            {
                "step_id": str(item.get("step_id", "")),
                "title": str(item.get("title", "")),
                "status": str(item.get("status", "")),
                "expected_outcome": str(item.get("expected_outcome", ""))[:800],
                "source_plan_item_ids": list(item.get("source_plan_item_ids", ())),
                "risk_checklist": list(item.get("failure_signals", ()))[:6],
            }
            for item in (
                steps[current_position + 1 : current_position + 3]
                if current_position >= 0
                else steps[:2]
            )
        ]
        route_skeleton = [
            {
                "canonical_id": str(item.get("canonical_id", "")),
                "title": str(item.get("title", "")),
                "status": str(item.get("status", "")),
                "current": bool(item.get("current")),
                "depends_on": list(item.get("depends_on", ())),
                "page_addresses": [
                    {
                        "memory_ref": str(page.get("memory_ref", "")),
                        "summary": str(page.get("summary", ""))[:240],
                        "outcome": str(page.get("outcome", "")),
                        "delta_kinds": list(page.get("delta_kinds", ()))[:8],
                        "changed_files": list(page.get("changed_files", ()))[:12],
                        "changed_symbols": list(page.get("changed_symbols", ()))[:12],
                        "test_refs": list(page.get("test_refs", ()))[:8],
                        "entity_refs": list(page.get("entity_refs", ()))[:16],
                    }
                    for page in item.get("latest_page_deltas", ())
                    if isinstance(page, Mapping)
                ],
            }
            for item in milestones
        ]
        current_step_card = None
        if current_step is not None:
            current_step_card = {
                "step_id": current_step.get("step_id"),
                "title": current_step.get("title"),
                "status": current_step.get("status"),
                "corrective": current_step.get("corrective"),
                "entity_refs": current_step.get("entity_refs", ()),
                "historical_dependency_refs": current_step.get(
                    "historical_dependency_refs", ()
                ),
                "source_plan_item_ids": current_step.get("source_plan_item_ids", ()),
                "expected_outcome": current_step.get("expected_outcome"),
                "failure_signals": current_step.get("failure_signals", ()),
            }
        workspace = snapshot.get("workspace_state", {})
        if not isinstance(workspace, Mapping):
            workspace = {}
        return {
            "schema": "codex-longterm-v2/tpg-route-card@1",
            "source": "TPG_AUTHORITATIVE_PROJECTION",
            "goal": str(snapshot.get("goal", ""))[:4000],
            "final_outcomes": observable_claims(snapshot.get("final_acceptance", ())),
            "route_skeleton": route_skeleton,
            "current_milestone": {
                "canonical_id": current.get("canonical_id"),
                "title": current.get("title"),
                "description": str(current.get("description", ""))[:1200],
                "target_outcome": current.get("target_outcome"),
                "status": current.get("status"),
                "depends_on": current.get("depends_on", ()),
                "source_plan_item_ids": current.get("source_plan_item_ids", ()),
                "terminal_outcomes": observable_claims(
                    current.get("minimum_acceptance", ())
                ),
                "downstream_assumptions": current.get("downstream_assumptions", ()),
                "non_goals": current.get("non_goals", ()),
            },
            "current_step": current_step_card,
            "step_route": [
                {
                    "step_id": item.get("step_id"),
                    "title": item.get("title"),
                    "status": item.get("status"),
                    "current": item.get("current"),
                    "corrective": item.get("corrective"),
                    "source_plan_item_ids": item.get("source_plan_item_ids", ()),
                }
                for item in steps
            ],
            "upcoming_steps": upcoming_steps,
            "working_plan": self.latest_working_plan_observation(
                run_id,
                branch_id,
            ),
            "workspace_state": {
                "revision_id": workspace.get("revision_id"),
                "receipt_source_event_id": workspace.get("receipt_source_event_id"),
                "changed_files": list(workspace.get("changed_files", ()))[:12],
                "changed_symbols": list(workspace.get("changed_symbols", ()))[:12],
            },
        }

    # ------------------------------------------------------------------
    # Milestone PageSets (logical grouping of Pages by Milestone boundary)
    # ------------------------------------------------------------------

    _PAGE_SET_TERMINAL_STATES = frozenset(
        {"COMPLETED_VERIFIED", "VERIFICATION_FAILED", "ROUTE_STALLED", "TASK_TERMINATED"}
    )

    def attach_page_symbols(
        self,
        *,
        page_id: str,
        repository_id: str,
        run_id: str,
        branch_id: str,
        revision_id: str,
        symbols_by_path: Mapping[str, Sequence[str]],
    ) -> tuple[str, ...]:
        """Attach Rich-Graph symbol addresses to an already projected Page.

        The Page body and its descriptor stay immutable; this only widens the
        Semantic Page Table index so that a later MemoryRef or address
        resolution that names a symbol (``symbol:path:Name``) reaches the Page
        that changed the file defining it.  Returns the newly attached symbol
        entity IDs (existing rows are left alone).
        """

        with self.database.transaction() as conn:
            page = conn.execute(
                "SELECT page_id FROM v2_semantic_pages WHERE page_id=? AND run_id=? AND branch_id=?",
                (page_id, run_id, branch_id),
            ).fetchone()
            if page is None:
                return ()
            known = {
                str(row["canonical_entity_id"])
                for row in conn.execute(
                    "SELECT canonical_entity_id FROM v2_semantic_page_entities WHERE page_id=?",
                    (page_id,),
                ).fetchall()
            }
            attached: list[str] = []
            for path, symbols in symbols_by_path.items():
                file_entity = f"file:{str(path).strip().replace(chr(92), '/').lstrip('./')}"
                if file_entity not in known:
                    # Only files the Page itself changed are enriched; the
                    # graph never adds files to a Page it did not touch.
                    continue
                for symbol in symbols:
                    entity_id = str(symbol).strip()
                    if not entity_id.startswith("symbol:") or entity_id in known:
                        continue
                    cursor = self._advance_clock(conn, run_id)
                    reference = self._ensure_reference_node(
                        conn,
                        repository_id=repository_id,
                        run_id=run_id,
                        branch_id=branch_id,
                        revision_id=revision_id,
                        canonical_entity_id=entity_id,
                        cursor=cursor,
                    )
                    if reference is None:
                        continue
                    conn.execute(
                        "INSERT OR IGNORE INTO v2_semantic_page_entities VALUES(?,?,?)",
                        (page_id, entity_id, reference[1]),
                    )
                    for alias in (*self._entity_aliases(entity_id), file_entity):
                        conn.execute(
                            "INSERT OR IGNORE INTO v2_semantic_entity_aliases "
                            "(repository_id,run_id,branch_id,canonical_entity_id,alias,"
                            "reference_node_id) VALUES(?,?,?,?,?,?)",
                            (repository_id, run_id, branch_id, entity_id, alias, reference[1]),
                        )
                    known.add(entity_id)
                    attached.append(entity_id)
            return tuple(attached)

    def commit_milestone_page_set(
        self,
        *,
        run_id: str,
        branch_id: str,
        milestone_identity_id: str,
        canonical_id: str,
        plan_version_id: str,
        terminal_state: str,
        wal_start: int,
        wal_end: int,
        acceptance: Mapping[str, object] | None,
        source_event_id: str,
        revision_id: str,
        stall_reason: str | None = None,
    ) -> Mapping[str, object]:
        """Close the logical PageSet of one Milestone at a terminal boundary.

        The PageSet is the Milestone-level unit of the Semantic Page Table: it
        lists every Page sealed for the Milestone, and its synopsis carries the
        only content that stays resident in the Working Set once the Milestone
        cools down -- the acceptance receipt summary, the touched entities and
        the durable implementation decisions.  Page bodies remain evictable
        and are re-opened through the MemoryRefs recorded here.
        """

        if terminal_state not in self._PAGE_SET_TERMINAL_STATES:
            raise ValueError(f"unsupported PageSet terminal state: {terminal_state}")
        with self.database.transaction() as conn:
            existing = conn.execute(
                "SELECT page_set_id FROM v2_semantic_milestone_page_sets "
                "WHERE run_id=? AND milestone_identity_id=? AND terminal_state=? "
                "AND source_event_id=?",
                (run_id, milestone_identity_id, terminal_state, source_event_id),
            ).fetchone()
            if existing is not None:
                return self._milestone_page_set_row(conn, str(existing["page_set_id"]))
            page_rows = conn.execute(
                "SELECT p.page_id,p.payload_digest,p.event_start,p.event_end,p.page_seq,"
                "d.delta_summary,d.outcome,d.changed_files_json,d.changed_symbols_json,"
                "d.test_refs_json "
                "FROM v2_semantic_pages p "
                "JOIN v2_semantic_page_milestones m ON m.page_id=p.page_id "
                "LEFT JOIN v2_semantic_page_descriptors d ON d.page_id=p.page_id "
                "WHERE p.run_id=? AND p.branch_id=? AND m.milestone_identity_id=? "
                "ORDER BY p.page_seq",
                (run_id, branch_id, milestone_identity_id),
            ).fetchall()
            page_ids = [str(row["page_id"]) for row in page_rows]
            memory_refs = [
                memory_ref_for_page(str(row["page_id"]), str(row["payload_digest"]))
                for row in page_rows
            ]
            entities: list[str] = []
            decisions: list[dict[str, object]] = []
            if page_ids:
                placeholders = ",".join("?" for _ in page_ids)
                entities = [
                    str(row["canonical_entity_id"])
                    for row in conn.execute(
                        "SELECT DISTINCT canonical_entity_id FROM v2_semantic_page_entities "
                        f"WHERE page_id IN ({placeholders}) ORDER BY canonical_entity_id",
                        tuple(page_ids),
                    ).fetchall()
                    if str(row["canonical_entity_id"]).split(":", 1)[0]
                    in {"file", "symbol", "test", "failure", "change"}
                ][:96]
                for row in conn.execute(
                    "SELECT e.event_id,e.canonical_entity_id,e.content_json,e.revision_id,"
                    "e.page_id,e.evidence_type FROM v2_semantic_evidence e "
                    f"WHERE e.run_id=? AND e.branch_id=? AND e.page_id IN ({placeholders}) "
                    "AND e.evidence_type IN ('IMPLEMENTATION_DECISION','UNRESOLVED_QUESTION') "
                    "AND e.valid_to_cursor IS NULL "
                    "ORDER BY e.valid_from_cursor,e.evidence_id",
                    (run_id, branch_id, *page_ids),
                ).fetchall()[:32]:
                    content = json.loads(str(row["content_json"]))
                    summary = content.get("summary") or content.get("question") or ""
                    decisions.append(
                        {
                            "kind": str(row["evidence_type"]),
                            "entity": str(row["canonical_entity_id"]),
                            "summary": " ".join(str(summary).split())[:400],
                            "event_id": str(row["event_id"]),
                            "revision_id": str(row["revision_id"]),
                            "page_id": str(row["page_id"]),
                        }
                    )
            changed_files = sorted(
                {
                    str(item)
                    for row in page_rows
                    if row["changed_files_json"] is not None
                    for item in json.loads(str(row["changed_files_json"]))
                }
            )[:48]
            changed_symbols = sorted(
                {
                    str(item)
                    for row in page_rows
                    if row["changed_symbols_json"] is not None
                    for item in json.loads(str(row["changed_symbols_json"]))
                }
            )[:48]
            test_refs = sorted(
                {
                    str(item)
                    for row in page_rows
                    if row["test_refs_json"] is not None
                    for item in json.loads(str(row["test_refs_json"]))
                }
            )[:48]
            previous = conn.execute(
                "SELECT page_set_id FROM v2_semantic_milestone_page_sets "
                "WHERE run_id=? AND milestone_identity_id=? "
                "ORDER BY created_cursor DESC LIMIT 1",
                (run_id, milestone_identity_id),
            ).fetchone()
            supersedes = str(previous["page_set_id"]) if previous is not None else None
            synopsis = {
                "schema": "codex-longterm-v2/milestone-page-set-synopsis@1",
                "milestone_id": canonical_id,
                "terminal_state": terminal_state,
                "stall_reason": stall_reason,
                "acceptance": dict(acceptance) if acceptance is not None else None,
                "touched_entities": entities,
                "changed_files": changed_files,
                "changed_symbols": changed_symbols,
                "test_refs": test_refs,
                "implementation_decisions": decisions,
                "page_deltas": [
                    {
                        "memory_ref": memory_ref,
                        "summary": (
                            " ".join(str(row["delta_summary"]).split())[:240]
                            if row["delta_summary"] is not None
                            else ""
                        ),
                        "outcome": str(row["outcome"]) if row["outcome"] is not None else None,
                    }
                    for row, memory_ref in zip(page_rows, memory_refs, strict=True)
                ][-12:],
                "page_count": len(page_ids),
                "wal_start": int(wal_start),
                "wal_end": int(wal_end),
                "revision_id": revision_id,
            }
            synopsis_digest = digest(synopsis)
            page_set_id = stable_id(
                "mpageset_",
                {
                    "run": run_id,
                    "milestone": milestone_identity_id,
                    "terminal_state": terminal_state,
                    "source": source_event_id,
                },
            )
            cursor = self._advance_clock(conn, run_id)
            conn.execute(
                "INSERT INTO v2_semantic_milestone_page_sets("
                "page_set_id,run_id,branch_id,milestone_identity_id,canonical_id,"
                "plan_version_id,terminal_state,supersedes_page_set_id,wal_start,wal_end,"
                "page_ids_json,memory_refs_json,synopsis_json,synopsis_digest,"
                "source_event_id,revision_id,created_cursor) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    page_set_id,
                    run_id,
                    branch_id,
                    milestone_identity_id,
                    canonical_id,
                    plan_version_id,
                    terminal_state,
                    supersedes,
                    int(wal_start),
                    int(max(wal_start, wal_end)),
                    json.dumps(page_ids),
                    json.dumps(memory_refs),
                    json.dumps(synopsis, sort_keys=True, ensure_ascii=False),
                    synopsis_digest,
                    source_event_id,
                    revision_id,
                    cursor,
                ),
            )
            return self._milestone_page_set_row(conn, page_set_id)

    @staticmethod
    def _milestone_page_set_row(conn: sqlite3.Connection, page_set_id: str) -> dict[str, object]:
        row = conn.execute(
            "SELECT * FROM v2_semantic_milestone_page_sets WHERE page_set_id=?",
            (page_set_id,),
        ).fetchone()
        if row is None:
            raise KeyError(page_set_id)
        return {
            "page_set_id": str(row["page_set_id"]),
            "milestone_identity_id": str(row["milestone_identity_id"]),
            "milestone_id": str(row["canonical_id"]),
            "plan_version_id": str(row["plan_version_id"]),
            "terminal_state": str(row["terminal_state"]),
            "supersedes_page_set_id": row["supersedes_page_set_id"],
            "wal_start": int(row["wal_start"]),
            "wal_end": int(row["wal_end"]),
            "page_ids": tuple(json.loads(str(row["page_ids_json"]))),
            "memory_refs": tuple(json.loads(str(row["memory_refs_json"]))),
            "synopsis": json.loads(str(row["synopsis_json"])),
            "synopsis_digest": str(row["synopsis_digest"]),
            "source_event_id": str(row["source_event_id"]),
            "revision_id": str(row["revision_id"]),
            "created_cursor": int(row["created_cursor"]),
        }

    def milestone_page_sets(
        self,
        run_id: str,
        *,
        milestone_identity_id: str | None = None,
    ) -> tuple[dict[str, object], ...]:
        """Return committed Milestone PageSets, oldest first."""

        conn = self.database.connection
        if milestone_identity_id is None:
            rows = conn.execute(
                "SELECT page_set_id FROM v2_semantic_milestone_page_sets "
                "WHERE run_id=? ORDER BY created_cursor",
                (run_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT page_set_id FROM v2_semantic_milestone_page_sets "
                "WHERE run_id=? AND milestone_identity_id=? ORDER BY created_cursor",
                (run_id, milestone_identity_id),
            ).fetchall()
        return tuple(self._milestone_page_set_row(conn, str(row["page_set_id"])) for row in rows)

    def latest_milestone_page_set(
        self,
        run_id: str,
        milestone_identity_id: str,
    ) -> dict[str, object] | None:
        rows = self.milestone_page_sets(run_id, milestone_identity_id=milestone_identity_id)
        return rows[-1] if rows else None

    def execution_route_delta(
        self,
        run_id: str,
        branch_id: str,
        *,
        plan_version_id: str | None = None,
    ) -> Mapping[str, object]:
        """Return the small hot-path projection of the authoritative TPG.

        The complete route remains queryable through ``execution_route_card``
        at Milestone review and recovery boundaries.  Normal Coding Turns need
        only the current node, nearby route, unresolved obligations, and direct
        Page addresses; repeatedly serializing the whole graph turns memory
        infrastructure into model control overhead.
        """

        card = self.execution_route_card(
            run_id,
            branch_id,
            plan_version_id=plan_version_id,
        )
        skeleton_values = card.get("route_skeleton", ())
        skeleton = [dict(item) for item in skeleton_values if isinstance(item, Mapping)]
        current_value = card.get("current_milestone", {})
        current = dict(current_value) if isinstance(current_value, Mapping) else {}
        current_id = str(current.get("canonical_id", ""))
        current_index = next(
            (
                index
                for index, item in enumerate(skeleton)
                if str(item.get("canonical_id", "")) == current_id
            ),
            -1,
        )

        def route_node(item: Mapping[str, object]) -> dict[str, object]:
            return {
                "canonical_id": item.get("canonical_id"),
                "title": item.get("title"),
                "status": item.get("status"),
            }

        current_step_value = card.get("current_step", {})
        current_step = (
            dict(current_step_value) if isinstance(current_step_value, Mapping) else {}
        )
        focus_entities = {
            str(value).casefold()
            for value in (
                *current_step.get("entity_refs", ()),
                *current_step.get("historical_dependency_refs", ()),
                *(
                    entity
                    for outcome in current.get("terminal_outcomes", ())
                    if isinstance(outcome, Mapping)
                    for entity in outcome.get("entity_refs", ())
                ),
            )
            if str(value).strip()
        }
        focus_text = " ".join(
            map(
                str,
                (
                    current.get("title", ""),
                    current.get("target_outcome", ""),
                    current_step.get("title", ""),
                    current_step.get("expected_outcome", ""),
                ),
            )
        ).casefold()
        focus_terms = {
            token
            for token in re.findall(r"[a-z0-9_./:-]{3,}", focus_text)
            if token not in {"the", "and", "for", "with", "from", "that"}
        }
        scored_addresses: list[tuple[int, int, str, dict[str, object]]] = []
        for node_index, node in enumerate(skeleton):
            distance = abs(node_index - current_index) if current_index >= 0 else node_index
            for address_value in node.get("page_addresses", ()):
                if not isinstance(address_value, Mapping):
                    continue
                address = dict(address_value)
                memory_ref = str(address.get("memory_ref", ""))
                if not memory_ref:
                    continue
                address_entities = {
                    str(value).casefold()
                    for field in (
                        "entity_refs",
                        "changed_files",
                        "changed_symbols",
                        "test_refs",
                    )
                    for value in address.get(field, ())
                    if str(value).strip()
                }
                exact_entities = focus_entities.intersection(address_entities)
                searchable = " ".join(
                    (
                        str(address.get("summary", "")),
                        " ".join(sorted(address_entities)),
                        " ".join(map(str, address.get("delta_kinds", ()))),
                    )
                ).casefold()
                term_overlap = sum(1 for term in focus_terms if term in searchable)
                score = (100 * len(exact_entities)) + (4 * term_overlap)
                if node_index == current_index:
                    score += 12
                elif node_index < current_index:
                    score += max(1, 8 - distance)
                address["route_node"] = str(node.get("canonical_id", ""))
                address["route_position"] = (
                    "CURRENT"
                    if node_index == current_index
                    else "PREDECESSOR"
                    if node_index < current_index
                    else "SUCCESSOR"
                )
                address["selection_reason"] = (
                    "exact route entity dependency"
                    if exact_entities
                    else "route semantic overlap"
                    if term_overlap
                    else "bounded route recency"
                )
                scored_addresses.append((-score, distance, memory_ref, address))
        scored_addresses.sort(key=lambda item: item[:3])
        memory_refs = [item[-1] for item in scored_addresses[:4]]
        return {
            "schema": "codex-longterm-v2/tpg-route-delta@1",
            "source": "TPG_AUTHORITATIVE_HOT_PATH_PROJECTION",
            "goal": card.get("goal"),
            "current_milestone": current,
            "current_step": card.get("current_step"),
            "working_plan": card.get("working_plan"),
            "upcoming_steps": list(card.get("upcoming_steps", ()))[:2],
            "recently_completed": [
                route_node(item)
                for item in (
                    skeleton[max(0, current_index - 2) : current_index]
                    if current_index >= 0
                    else ()
                )
            ],
            "next_milestones": [
                route_node(item)
                for item in (
                    skeleton[current_index + 1 : current_index + 3]
                    if current_index >= 0
                    else skeleton[:2]
                )
            ],
            "relevant_memory_refs": memory_refs,
            "workspace_state": card.get("workspace_state", {}),
        }
