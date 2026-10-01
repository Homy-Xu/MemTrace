from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from ..contracts import RichGraphState, canonical_bytes, primitive, stable_id, utc_now
from .models import (
    FileProjection,
    RichGraphHint,
    file_reference_id,
    symbol_reference_id,
    test_reference_id,
)
from .project_resolution import resolve_project_call

# Reverse views over stored forward relations.  ``CALLED_BY`` answers "who
# calls this symbol", ``COVERED_BY`` "which tests exercise it", ``IMPORTED_BY``
# "which files depend on this module".  They are derived at query time so the
# projection stays a single forward edge list.
_REVERSE_RELATIONS: dict[str, str] = {
    "CALLED_BY": "CALLS",
    "COVERED_BY": "COVERS",
    "IMPORTED_BY": "IMPORTS",
}


class RichGraphStore:
    """Independent, revision-generation-scoped RSG storage."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.path,
            isolation_level=None,
            check_same_thread=False,
            timeout=5.0,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._migrate()

    def _migrate(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS rich_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rich_generations (
                generation_id TEXT PRIMARY KEY,
                repository_id TEXT NOT NULL,
                workspace_revision_id TEXT NOT NULL,
                state TEXT NOT NULL,
                supported_relations_json TEXT NOT NULL,
                failure_reason TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(repository_id, workspace_revision_id)
            );
            CREATE TABLE IF NOT EXISTS rich_references (
                repository_id TEXT NOT NULL,
                reference_id TEXT NOT NULL,
                reference_kind TEXT NOT NULL,
                canonical_entity_id TEXT NOT NULL,
                repository_relative_path TEXT NOT NULL,
                qualified_name TEXT,
                PRIMARY KEY(repository_id, reference_id),
                UNIQUE(repository_id, canonical_entity_id)
            );
            CREATE TABLE IF NOT EXISTS rich_file_projections (
                generation_id TEXT NOT NULL,
                file_reference_id TEXT NOT NULL,
                file_version_id TEXT NOT NULL,
                repository_relative_path TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                language TEXT NOT NULL,
                byte_count INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                projected_at TEXT NOT NULL,
                PRIMARY KEY(generation_id, file_reference_id),
                FOREIGN KEY(generation_id) REFERENCES rich_generations(generation_id)
            );
            CREATE TABLE IF NOT EXISTS rich_relations (
                generation_id TEXT NOT NULL,
                source_file_reference_id TEXT NOT NULL,
                relation_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                source_reference_id TEXT NOT NULL,
                target_reference_id TEXT NOT NULL,
                authority TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                PRIMARY KEY(generation_id, relation_id),
                FOREIGN KEY(generation_id) REFERENCES rich_generations(generation_id)
            );
            CREATE INDEX IF NOT EXISTS rich_relation_source_idx
            ON rich_relations(generation_id, relation, source_reference_id);
            CREATE INDEX IF NOT EXISTS rich_relation_target_idx
            ON rich_relations(generation_id, relation, target_reference_id);
            CREATE INDEX IF NOT EXISTS rich_reference_entity_idx
            ON rich_references(repository_id, canonical_entity_id);
            CREATE TABLE IF NOT EXISTS rich_version_nodes (
                generation_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                node_kind TEXT NOT NULL,
                reference_id TEXT,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(generation_id, node_id),
                FOREIGN KEY(generation_id) REFERENCES rich_generations(generation_id)
            );
            CREATE TABLE IF NOT EXISTS rich_hint_cache (
                repository_id TEXT NOT NULL,
                workspace_revision_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                hints_json TEXT NOT NULL,
                cached_at TEXT NOT NULL,
                PRIMARY KEY(repository_id, workspace_revision_id, entity_id, relation)
            );
            """
        )
        self._connection.execute(
            "INSERT INTO rich_meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(self.SCHEMA_VERSION),),
        )

    def ensure_generation(self, repository_id: str, revision_id: str) -> str:
        generation_id = stable_id(
            "richgen_",
            {"repository_id": repository_id, "workspace_revision_id": revision_id},
        )
        now = utc_now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO rich_generations(
                    generation_id, repository_id, workspace_revision_id, state,
                    supported_relations_json, failure_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, '[]', NULL, ?, ?)
                ON CONFLICT(generation_id) DO NOTHING
                """,
                (
                    generation_id,
                    repository_id,
                    revision_id,
                    RichGraphState.NOT_STARTED.value,
                    now,
                    now,
                ),
            )
        return generation_id

    def set_generation_state(
        self,
        generation_id: str,
        state: RichGraphState,
        supported_relations: Iterable[str],
        failure_reason: str | None,
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE rich_generations
                SET state=?, supported_relations_json=?, failure_reason=?, updated_at=?
                WHERE generation_id=?
                """,
                (
                    state.value,
                    json.dumps(sorted(set(supported_relations))),
                    failure_reason,
                    utc_now(),
                    generation_id,
                ),
            )

    def commit_projection(self, generation_id: str, projection: FileProjection) -> None:
        """Replace one file projection atomically inside its Rich generation."""

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    "DELETE FROM rich_relations WHERE generation_id=? "
                    "AND source_file_reference_id=?",
                    (generation_id, projection.file_reference_id),
                )
                self._connection.execute(
                    """
                    INSERT INTO rich_file_projections(
                        generation_id, file_reference_id, file_version_id,
                        repository_relative_path, content_digest, language,
                        byte_count, metadata_json, projected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(generation_id, file_reference_id) DO UPDATE SET
                        file_version_id=excluded.file_version_id,
                        repository_relative_path=excluded.repository_relative_path,
                        content_digest=excluded.content_digest,
                        language=excluded.language,
                        byte_count=excluded.byte_count,
                        metadata_json=excluded.metadata_json,
                        projected_at=excluded.projected_at
                    """,
                    (
                        generation_id,
                        projection.file_reference_id,
                        projection.file_version_id,
                        projection.repository_relative_path,
                        projection.content_digest,
                        projection.language,
                        projection.byte_count,
                        canonical_bytes(projection.metadata).decode("utf-8"),
                        utc_now(),
                    ),
                )
                for binding in projection.bindings:
                    self._connection.execute(
                        """
                        INSERT INTO rich_references(
                            repository_id, reference_id, reference_kind,
                            canonical_entity_id, repository_relative_path, qualified_name
                        )
                        SELECT repository_id, ?, ?, ?, ?, ?
                        FROM rich_generations WHERE generation_id=?
                        ON CONFLICT(repository_id, reference_id) DO UPDATE SET
                            reference_kind=excluded.reference_kind,
                            canonical_entity_id=excluded.canonical_entity_id,
                            repository_relative_path=excluded.repository_relative_path,
                            qualified_name=excluded.qualified_name
                        """,
                        (
                            binding.reference_id,
                            binding.reference_kind,
                            binding.canonical_entity_id,
                            binding.repository_relative_path,
                            binding.qualified_name,
                            generation_id,
                        ),
                    )
                for node in projection.version_nodes:
                    self._connection.execute(
                        """INSERT OR REPLACE INTO rich_version_nodes
                           (generation_id,node_id,node_kind,reference_id,payload_json)
                           VALUES(?,?,?,?,?)""",
                        (
                            generation_id,
                            node.node_id,
                            node.node_kind,
                            node.reference_id,
                            canonical_bytes(node.payload).decode("utf-8"),
                        ),
                    )
                for relation in projection.relations:
                    relation_id = stable_id(
                        "richrel_",
                        {
                            "generation_id": generation_id,
                            "relation": relation.relation,
                            "source": relation.source_reference_id,
                            "target": relation.target_reference_id,
                            "provenance": relation.provenance,
                        },
                    )
                    self._connection.execute(
                        """
                        INSERT INTO rich_relations(
                            generation_id, source_file_reference_id, relation_id,
                            relation, source_reference_id, target_reference_id,
                            authority, provenance_json, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            generation_id,
                            projection.file_reference_id,
                            relation_id,
                            relation.relation,
                            relation.source_reference_id,
                            relation.target_reference_id,
                            relation.authority,
                            canonical_bytes(relation.provenance).decode("utf-8"),
                            relation.confidence,
                        ),
                    )
                # Invalidate only cache keys whose entity bindings can change in
                # this projection. Cache identity remains the exact four-tuple
                # repository/revision/entity/relation.
                generation = self._connection.execute(
                    "SELECT repository_id, workspace_revision_id FROM rich_generations "
                    "WHERE generation_id=?",
                    (generation_id,),
                ).fetchone()
                if generation:
                    if projection.language != "python":
                        self._refresh_project_calls(generation_id, str(generation["repository_id"]),
                                                    str(generation["workspace_revision_id"]), projection.repository_relative_path)
                    affected = (
                        {item.reference_id for item in projection.bindings}
                        | {item.canonical_entity_id for item in projection.bindings}
                        | {
                            reference
                            for item in projection.relations
                            for reference in (
                                item.source_reference_id,
                                item.target_reference_id,
                            )
                        }
                    )
                    if affected:
                        placeholders = ",".join("?" for _ in affected)
                        aliases = self._connection.execute(
                            "SELECT canonical_entity_id FROM rich_references "
                            "WHERE repository_id=? AND reference_id IN (" + placeholders + ")",
                            (generation["repository_id"], *affected),
                        ).fetchall()
                        affected.update(str(row["canonical_entity_id"]) for row in aliases)
                        placeholders = ",".join("?" for _ in affected)
                        self._connection.execute(
                            "DELETE FROM rich_hint_cache WHERE repository_id=? "
                            "AND workspace_revision_id=? AND entity_id IN (" + placeholders + ")",
                            (
                                generation["repository_id"],
                                generation["workspace_revision_id"],
                                *affected,
                            ),
                        )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def _refresh_project_calls(self, generation_id: str, repository: str, revision: str, changed_path: str) -> None:
        """Background commit only; both arrival orders and deletions are handled.

        Index input is restricted to frontier projections in this generation.
        Never consult old-revision symbols or scan/import external packages.
        """
        rows = self._connection.execute(
            "SELECT repository_relative_path,metadata_json FROM rich_file_projections "
            "WHERE generation_id=? AND language!='python'", (generation_id,)).fetchall()
        files = {str(r["repository_relative_path"]): json.loads(r["metadata_json"]).get("navigation_index", {}) for r in rows}
        # Re-resolve the changed caller and callers potentially affected by
        # this callee. Avoid rebuilding every existing cross-file edge after
        # every frontier file. Metadata is already indexed, never repo-scanned.
        changed = files.get(changed_path, {})
        affected = {changed_path}
        incoming = self._connection.execute(
            "SELECT DISTINCT s.repository_relative_path FROM rich_relations e "
            "JOIN rich_references t ON t.reference_id=e.target_reference_id AND t.repository_id=? "
            "JOIN rich_references s ON s.reference_id=e.source_file_reference_id AND s.repository_id=? "
            "WHERE e.generation_id=? AND t.repository_relative_path=?",
            (repository, repository, generation_id, changed_path)).fetchall()
        affected.update(str(row[0]) for row in incoming)
        for path, data in files.items():
            same_module = (data.get("language") in {"go", "java"} and data.get("language") == changed.get("language")
                           and data.get("module") == changed.get("module"))
            project_relations = (
                *data.get("calls", ()),
                *data.get("references", ()),
                *data.get("covers", ()),
            )
            if same_module or any(
                resolve_project_call(path, data, relation, {changed_path: changed})
                for relation in project_relations
                if float(relation.get("confidence", 1.0)) == 1.0
            ):
                affected.add(path)
        for path in affected:
            self._connection.execute("DELETE FROM rich_relations WHERE generation_id=? AND source_file_reference_id=? "
                                     "AND relation_id LIKE 'projectrel_%'", (generation_id, file_reference_id(repository, path)))
        for path, data in files.items():
            project_relations = (
                *data.get("calls", ()),
                *data.get("references", ()),
                *data.get("covers", ()),
            )
            if path not in affected or (
                data.get("parser_confidence") != 1
                and not any(
                    float(relation.get("confidence", 0.0)) == 1.0
                    for relation in project_relations
                )
            ):
                continue
            tests = {s["qualified_name"] for s in data.get("symbols", ()) if s.get("is_test")}
            for call in project_relations:
                if float(call.get("confidence", 1.0)) != 1.0:
                    continue
                match = resolve_project_call(path, data, call, files)
                if match is None:
                    continue
                target_path, target_symbol = match
                target = symbol_reference_id(repository, target_path, target_symbol)
                source = (
                    symbol_reference_id(repository, path, call["source"])
                    if call.get("source")
                    else file_reference_id(repository, path)
                )
                relation_kind = str(call.get("relation", "CALLS"))
                if relation_kind == "COVERS":
                    if call.get("source") not in tests:
                        continue
                    source = test_reference_id(repository, f"{path}::{call['source']}")
                edges = [
                    (relation_kind, source, target),
                    ("IMPORTS", file_reference_id(repository, path), file_reference_id(repository, target_path)),
                ]
                if relation_kind == "CALLS" and call.get("source") in tests:
                    edges.append(("COVERS", test_reference_id(repository, f"{path}::{call['source']}"), target))
                for kind, start, end in edges:
                    provenance = (path, f"{path}:{call['line']}", target_path, revision, "PROJECT_IMPORT_RESOLUTION")
                    relation_id = stable_id("projectrel_", {"generation": generation_id, "kind": kind,
                                                          "source": start, "target": end, "provenance": provenance})
                    self._connection.execute(
                        "INSERT OR IGNORE INTO rich_relations VALUES(?,?,?,?,?,?,?,?,?)",
                        (generation_id, file_reference_id(repository, path), relation_id, kind, start, end,
                         "DERIVED", canonical_bytes(provenance).decode(), 1.0))
        # A newly indexed callee may change reverse queries for existing files.
        self._connection.execute("DELETE FROM rich_hint_cache WHERE repository_id=? AND workspace_revision_id=?",
                                 (repository, revision))

    def supported_relations(self, generation_id: str) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT DISTINCT relation FROM rich_relations "
                "WHERE generation_id=? ORDER BY relation",
                (generation_id,),
            ).fetchall()
        return tuple(str(row["relation"]) for row in rows)

    def search_references(
        self,
        repository_id: str,
        tokens: Iterable[str],
        *,
        kinds: Iterable[str] = ("SymbolReference", "FileReference", "TestReference"),
        limit: int = 64,
    ) -> tuple[str, ...]:
        """Return canonical entity IDs whose path or qualified name mention a token.

        This is an address *hint* source: the acceptance kernel ranks the
        result against requirement text and never treats a match as evidence.
        The scan is bounded to one repository's reference table and to the
        first ``limit`` rows per token.
        """

        normalized = tuple(
            dict.fromkeys(token.strip().casefold() for token in tokens if token and token.strip())
        )
        if not normalized:
            return ()
        kind_list = tuple(dict.fromkeys(kinds))
        if not kind_list:
            return ()
        kind_placeholders = ",".join("?" for _ in kind_list)
        results: dict[str, None] = {}
        with self._lock:
            for token in normalized:
                if len(token) < 3:
                    continue
                pattern = f"%{token}%"
                rows = self._connection.execute(
                    "SELECT canonical_entity_id FROM rich_references "
                    "WHERE repository_id=? AND reference_kind IN ("
                    + kind_placeholders
                    + ") AND (LOWER(qualified_name) LIKE ? OR LOWER(repository_relative_path) LIKE ?) "
                    "ORDER BY LENGTH(canonical_entity_id), canonical_entity_id LIMIT ?",
                    (repository_id, *kind_list, pattern, pattern, int(limit)),
                ).fetchall()
                for row in rows:
                    results.setdefault(str(row["canonical_entity_id"]), None)
        return tuple(results)

    def search_reference_rows(
        self,
        repository_id: str,
        tokens: Iterable[str],
        *,
        generation_id: str | None = None,
        kinds: Iterable[str] = ("SymbolReference", "FileReference", "TestReference"),
        limit: int = 64,
    ) -> tuple[dict[str, object], ...]:
        """Return bounded reference metadata for a model-facing graph search.

        The query is intentionally limited to the committed reference table
        for one repository and, when supplied, one graph generation.  It never
        reads the working tree and never mutates route or evidence state.
        """

        normalized = tuple(
            dict.fromkeys(
                token.strip().casefold()
                for token in tokens
                if token and token.strip()
            )
        )
        kind_list = tuple(dict.fromkeys(str(kind) for kind in kinds if str(kind)))
        if not normalized or not kind_list or limit <= 0:
            return ()
        placeholders = ",".join("?" for _ in kind_list)
        generation_filter = ""
        generation_args: tuple[object, ...] = ()
        if generation_id:
            generation_filter = (
                " AND EXISTS (SELECT 1 FROM rich_version_nodes v "
                "WHERE v.generation_id=? AND v.reference_id=r.reference_id)"
            )
            generation_args = (generation_id,)
        rows_by_id: dict[str, dict[str, object]] = {}
        with self._lock:
            for token in normalized:
                if len(token) < 2:
                    continue
                pattern = f"%{token}%"
                rows = self._connection.execute(
                    "SELECT r.reference_id, r.reference_kind, "
                    "r.canonical_entity_id, r.repository_relative_path, r.qualified_name "
                    "FROM rich_references r "
                    "WHERE r.repository_id=? AND r.reference_kind IN ("
                    + placeholders
                    + ") AND (LOWER(COALESCE(r.qualified_name,'')) LIKE ? "
                    "OR LOWER(r.repository_relative_path) LIKE ?)"
                    + generation_filter
                    + " ORDER BY LENGTH(r.canonical_entity_id), r.canonical_entity_id LIMIT ?",
                    (
                        repository_id,
                        *kind_list,
                        pattern,
                        pattern,
                        *generation_args,
                        int(limit),
                    ),
                ).fetchall()
                for row in rows:
                    reference_id = str(row["reference_id"])
                    rows_by_id.setdefault(
                        reference_id,
                        {
                            "reference_id": reference_id,
                            "reference_kind": str(row["reference_kind"]),
                            "canonical_entity_id": str(row["canonical_entity_id"]),
                            "repository_relative_path": str(row["repository_relative_path"]),
                            "qualified_name": (
                                str(row["qualified_name"])
                                if row["qualified_name"] is not None
                                else None
                            ),
                        },
                    )
                    if len(rows_by_id) >= int(limit):
                        break
                if len(rows_by_id) >= int(limit):
                    break
        return tuple(rows_by_id.values())

    def version_payload(
        self,
        generation_id: str,
        reference_id: str,
    ) -> dict[str, Any]:
        """Return the latest immutable version payload for one reference."""

        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM rich_version_nodes "
                "WHERE generation_id=? AND reference_id=? "
                "ORDER BY CASE node_kind WHEN 'SymbolVersion' THEN 0 "
                "WHEN 'Test' THEN 1 ELSE 2 END, node_id LIMIT 1",
                (generation_id, reference_id),
            ).fetchone()
        if row is None:
            return {}
        try:
            value = json.loads(str(row["payload_json"]))
        except (TypeError, ValueError):
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def version_payload_for_entity(
        self,
        generation_id: str,
        repository_id: str,
        entity_id: str,
    ) -> dict[str, Any]:
        """Return the immutable version payload addressed by a canonical entity."""

        with self._lock:
            row = self._connection.execute(
                "SELECT v.payload_json FROM rich_version_nodes v "
                "JOIN rich_references r ON r.reference_id=v.reference_id "
                "WHERE v.generation_id=? AND r.repository_id=? "
                "AND r.canonical_entity_id=? "
                "ORDER BY CASE v.node_kind WHEN 'SymbolVersion' THEN 0 "
                "WHEN 'Test' THEN 1 ELSE 2 END, v.node_id LIMIT 1",
                (generation_id, repository_id, entity_id),
            ).fetchone()
        if row is None:
            return {}
        try:
            value = json.loads(str(row["payload_json"]))
        except (TypeError, ValueError):
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def symbols_in_paths(
        self,
        repository_id: str,
        paths: Iterable[str],
        *,
        limit_per_path: int = 48,
        generation_id: str | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Canonical symbol entity IDs the graph has defined inside each path.

        Used at Page finalization to attach ``changed_symbols`` to a
        file-granular CODE_CHANGE fact.  Only structurally declared symbols
        (``SymbolReference`` rows) are returned; files without a projection
        simply yield nothing.
        """

        normalized = tuple(
            dict.fromkeys(
                path.strip().replace("\\", "/").lstrip("./")
                for path in paths
                if path and path.strip()
            )
        )
        if not normalized:
            return {}
        result: dict[str, tuple[str, ...]] = {}
        with self._lock:
            for path in normalized:
                rows = self._connection.execute(
                    "SELECT canonical_entity_id FROM rich_references "
                    "WHERE repository_id=? AND reference_kind='SymbolReference' "
                    "AND repository_relative_path=? "
                    + ("AND EXISTS (SELECT 1 FROM rich_version_nodes v WHERE v.generation_id=? "
                       "AND v.reference_id=rich_references.reference_id AND v.node_kind='SymbolVersion') " if generation_id else "") +
                    "ORDER BY LENGTH(canonical_entity_id), canonical_entity_id LIMIT ?",
                    (repository_id, path, *((generation_id,) if generation_id else ()), int(limit_per_path)),
                ).fetchall()
                if rows:
                    result[path] = tuple(str(row["canonical_entity_id"]) for row in rows)
        return result

    def reference_entity(self, repository_id: str, reference_id: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT canonical_entity_id FROM rich_references "
                "WHERE repository_id=? AND reference_id=?",
                (repository_id, reference_id),
            ).fetchone()
        return str(row["canonical_entity_id"]) if row else None

    def lookup_path(self, repository_id: str, entity_id: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT repository_relative_path FROM rich_references "
                "WHERE repository_id=? AND (canonical_entity_id=? OR reference_id=?)",
                (repository_id, entity_id, entity_id),
            ).fetchone()
        return str(row["repository_relative_path"]) if row else None

    def get_cached_hints(
        self,
        repository_id: str,
        revision_id: str,
        entity_id: str,
        relation: str,
    ) -> tuple[RichGraphHint, ...] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT hints_json FROM rich_hint_cache
                WHERE repository_id=? AND workspace_revision_id=?
                  AND entity_id=? AND relation=?
                """,
                (repository_id, revision_id, entity_id, relation),
            ).fetchone()
        if row is None:
            return None
        return tuple(
            RichGraphHint(
                relation=str(item["relation"]),
                source_reference_id=str(item["source_reference_id"]),
                target_reference_id=str(item["target_reference_id"]),
                authority=str(item["authority"]),
                provenance=tuple(map(str, item["provenance"])),
                confidence=float(item["confidence"]),
                generation_id=str(item["generation_id"]),
                workspace_revision_id=str(item["workspace_revision_id"]),
            )
            for item in json.loads(row["hints_json"])
        )

    def query_hints(
        self,
        generation_id: str,
        repository_id: str,
        revision_id: str,
        entity_id: str,
        relation: str,
    ) -> tuple[RichGraphHint, ...]:
        with self._lock:
            aliases = {entity_id}
            rows = self._connection.execute(
                "SELECT reference_id FROM rich_references "
                "WHERE repository_id=? AND canonical_entity_id=?",
                (repository_id, entity_id),
            ).fetchall()
            aliases.update(str(row["reference_id"]) for row in rows)
            placeholders = ",".join("?" for _ in aliases)
            reverse_of = _REVERSE_RELATIONS.get(relation)
            if reverse_of is not None:
                sql = (
                    "SELECT * FROM rich_relations WHERE generation_id=? "
                    "AND relation=? AND target_reference_id IN ("
                    + placeholders
                    + ") ORDER BY relation_id"
                )
                parameters = (generation_id, reverse_of, *aliases)
            else:
                sql = (
                    "SELECT * FROM rich_relations WHERE generation_id=? "
                    "AND relation=? AND source_reference_id IN ("
                    + placeholders
                    + ") ORDER BY relation_id"
                )
                parameters = (generation_id, relation, *aliases)
            result_rows = self._connection.execute(sql, parameters).fetchall()

            hints = tuple(
                RichGraphHint(
                    relation=relation,
                    source_reference_id=(
                        str(row["target_reference_id"])
                        if reverse_of is not None
                        else str(row["source_reference_id"])
                    ),
                    target_reference_id=(
                        str(row["source_reference_id"])
                        if reverse_of is not None
                        else str(row["target_reference_id"])
                    ),
                    authority=str(row["authority"]),
                    provenance=tuple(json.loads(row["provenance_json"])),
                    confidence=float(row["confidence"]),
                    generation_id=generation_id,
                    workspace_revision_id=revision_id,
                )
                for row in result_rows
            )
            self._connection.execute(
                """
                INSERT INTO rich_hint_cache(
                    repository_id, workspace_revision_id, entity_id, relation,
                    hints_json, cached_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repository_id, workspace_revision_id, entity_id, relation)
                DO UPDATE SET hints_json=excluded.hints_json, cached_at=excluded.cached_at
                """,
                (
                    repository_id,
                    revision_id,
                    entity_id,
                    relation,
                    json.dumps([primitive(item) for item in hints], sort_keys=True),
                    utc_now(),
                ),
            )
        return hints

    def projected_paths(self, generation_id: str) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT repository_relative_path FROM rich_file_projections "
                "WHERE generation_id=? ORDER BY projected_at, repository_relative_path",
                (generation_id,),
            ).fetchall()
        return tuple(str(row["repository_relative_path"]) for row in rows)

    def has_projection(self, generation_id: str, path: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM rich_file_projections "
                "WHERE generation_id=? AND repository_relative_path=?",
                (generation_id, path),
            ).fetchone()
        return row is not None

    def close(self) -> None:
        with self._lock:
            self._connection.close()
