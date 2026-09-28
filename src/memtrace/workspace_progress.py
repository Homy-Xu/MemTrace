"""Durable content progress, separate from the lossless physical revision log.

New scratch probes remain in WAL/Pages/patch receipts, but cannot renew a
semantic progress lease. Baseline files are always substantive, even when
their name resembles a probe. No language or task-specific acceptance rules.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import PurePosixPath
from typing import Mapping

from .contracts import digest


def ephemeral_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if not parts:
        return False
    return (
        any(p in {".homy-probes", "__probes__"} for p in parts)
        or parts[-1].startswith(("__probe-", "__probe_"))
        or (len(parts) <= 2 and parts[0] == "proto")
        or (len(parts) == 1 and bool(re.fullmatch(r"(?:proto|(?:run-)?check\d+)\.[a-z]+", parts[0])))
    )


def record_workspace_progress(connection: sqlite3.Connection, *, run_id: str, branch_id: str,
                              revision_id: str, source_event_id: str,
                              manifest: Mapping, changed_paths: tuple[str, ...]) -> dict:
    connection.execute("""CREATE TABLE IF NOT EXISTS v2_workspace_semantic_progress (
        run_id TEXT NOT NULL, branch_id TEXT NOT NULL, source_event_id TEXT NOT NULL,
        revision_id TEXT NOT NULL, content_digest TEXT NOT NULL, progress_token INTEGER NOT NULL,
        novel INTEGER NOT NULL, excluded_paths_json TEXT NOT NULL,
        PRIMARY KEY(run_id,branch_id,source_event_id))""")
    existing = connection.execute(
        "SELECT * FROM v2_workspace_semantic_progress WHERE run_id=? AND branch_id=? AND source_event_id=?",
        (run_id, branch_id, source_event_id),
    ).fetchone()
    if existing is not None:
        return dict(existing)
    if not changed_paths:
        previous = connection.execute(
            "SELECT * FROM v2_workspace_semantic_progress WHERE run_id=? AND branch_id=? ORDER BY rowid DESC LIMIT 1",
            (run_id, branch_id),
        ).fetchone()
        if previous is not None and previous["revision_id"] == revision_id:
            return {**dict(previous), "novel": 0}
    baseline = {str(row[0]) for row in connection.execute(
        "SELECT relative_path FROM v2_workspace_revision_files WHERE run_id=? AND branch_id=?",
        (run_id, branch_id),
    )}
    excluded = sorted(path for path in set(manifest).union(changed_paths)
                      if path not in baseline and ephemeral_path(path))
    content = digest([(path, metadata[3] & 0o170111, metadata[4])
                      for path, metadata in sorted(manifest.items()) if path not in excluded])
    previous = connection.execute(
        "SELECT progress_token FROM v2_workspace_semantic_progress WHERE run_id=? AND branch_id=? ORDER BY rowid DESC LIMIT 1",
        (run_id, branch_id),
    ).fetchone()
    seen = connection.execute(
        "SELECT 1 FROM v2_workspace_semantic_progress WHERE run_id=? AND branch_id=? AND content_digest=? LIMIT 1",
        (run_id, branch_id, content),
    ).fetchone()
    novel = previous is not None and seen is None
    token = (int(previous[0]) if previous else 0) + int(novel)
    row = dict(run_id=run_id, branch_id=branch_id, source_event_id=source_event_id,
               revision_id=revision_id, content_digest=content, progress_token=token,
               novel=int(novel), excluded_paths_json=json.dumps(excluded))
    connection.execute("INSERT INTO v2_workspace_semantic_progress VALUES(?,?,?,?,?,?,?,?)", tuple(row.values()))
    return row


def semantic_progress_token(connection: sqlite3.Connection, run_id: str,
                            revision_id: str | None, branch_id: str | None = None) -> str | None:
    """Old DBs retain their historical revision semantics until upgraded capture."""
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_workspace_semantic_progress'").fetchone():
        return revision_id
    query = "SELECT progress_token FROM v2_workspace_semantic_progress WHERE run_id=?"
    args = [run_id]
    if branch_id is not None:
        query += " AND branch_id=?"
        args.append(branch_id)
    row = connection.execute(query + " ORDER BY rowid DESC LIMIT 1", args).fetchone()
    return f"semantic:{row[0]}" if row else revision_id
