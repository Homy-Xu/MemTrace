from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..contracts import canonical_bytes, stable_id, utc_now
from ..database import StateDatabase
from ..durability import atomic_write_once
from ..orchestration.planning_coordinator import (
    WorkspaceReadOnlyGuard,
    WorkspaceSnapshotReceipt,
)
from ..references import ReferenceBindingRecord, ReferenceDirectory
from ..workspace_progress import record_workspace_progress


_MAX_PORTABLE_PATCH_BYTES = 256 * 1024 * 1024
_MAX_PORTABLE_PATCH_PATHS = 4096
_MAX_PORTABLE_PATCH_PATH_LIST_BYTES = 16 * 1024 * 1024
_GIT_OUTPUT_CHUNK_BYTES = 1024 * 1024
#: Untracked files above this size are left out of the portable patch; they
#: are build outputs or fixtures, never the agent's source edits.
_MAX_PORTABLE_UNTRACKED_FILE_BYTES = 32 * 1024 * 1024
#: In an official SWE-Milestone repository stream the process must outlive a
#: patch that exceeds its bound: the official runner counts every agent exit
#: without a new tag as "no progress" and ends the repository after three
#: in an earlier run.  The tags in ``/testbed`` remain the submission.
_REPOSITORY_STREAM_ENV = "HOMY_SWE_MILESTONE_REPOSITORY_STREAM"
PATCH_OMITTED_LIMIT = "PATCH_OMITTED_LIMIT"


class WorkspacePatchLimitError(RuntimeError):
    """A Workspace Revision patch exceeded its bounded artifact contract."""


@dataclass(frozen=True, slots=True)
class WorkspaceRevisionReceipt:
    revision_id: str
    previous_revision_id: str | None
    changed: bool
    changed_paths: tuple[str, ...]
    source_event_id: str
    symbol_bindings: tuple[ReferenceBindingRecord, ...] = ()
    patch_receipt_path: str | None = None
    patch_digest: str | None = None
    semantic_progress: Mapping[str, object] | None = None


class WorkspaceStateAuthority:
    """Single writer for workspace receipts, revisions, deltas and references."""

    def __init__(
        self,
        *,
        repository_path: Path,
        run_root: Path,
        database: StateDatabase,
        run_id: str,
        branch_id: str,
        reference_directory: ReferenceDirectory | None = None,
    ) -> None:
        self.repository_path = Path(repository_path).expanduser().resolve()
        self.run_root = Path(run_root).expanduser().resolve()
        self.database = database
        self.run_id = run_id
        self.branch_id = branch_id
        self.reference_directory = reference_directory
        with self.database.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS v2_workspace_revisions (
                    revision_event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    previous_revision_id TEXT,
                    source_event_id TEXT NOT NULL,
                    changed INTEGER NOT NULL CHECK(changed IN (0,1)),
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, branch_id, source_event_id, revision_id)
                );
                CREATE TABLE IF NOT EXISTS v2_current_workspace_revision (
                    run_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    source_event_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, branch_id)
                );
                CREATE TABLE IF NOT EXISTS v2_workspace_revision_files (
                    run_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    byte_count INTEGER NOT NULL,
                    modified_ns INTEGER NOT NULL,
                    changed_ns INTEGER NOT NULL,
                    mode INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    source_event_id TEXT NOT NULL,
                    PRIMARY KEY(run_id,branch_id,revision_id,relative_path)
                );
                CREATE TABLE IF NOT EXISTS v2_workspace_revision_changes (
                    run_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    source_event_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(run_id,branch_id,revision_id,relative_path)
                );
                CREATE TABLE IF NOT EXISTS v2_workspace_revision_file_deltas (
                    run_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    deleted INTEGER NOT NULL CHECK(deleted IN (0,1)),
                    byte_count INTEGER,
                    modified_ns INTEGER,
                    changed_ns INTEGER,
                    mode INTEGER,
                    content_digest TEXT,
                    source_event_id TEXT NOT NULL,
                    PRIMARY KEY(run_id,branch_id,revision_id,relative_path)
                );
                CREATE TABLE IF NOT EXISTS v2_workspace_snapshot_receipts (
                    run_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    manifest_digest TEXT NOT NULL,
                    vcs_fingerprint TEXT,
                    exclusion_policy_json TEXT NOT NULL,
                    source_event_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(run_id,branch_id,revision_id)
                );
                CREATE TABLE IF NOT EXISTS v2_workspace_revision_artifacts (
                    run_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    patch_path TEXT,
                    receipt_path TEXT NOT NULL,
                    patch_digest TEXT,
                    patch_bytes INTEGER NOT NULL,
                    patch_state TEXT NOT NULL,
                    source_event_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(run_id,branch_id,revision_id)
                );
                CREATE TRIGGER IF NOT EXISTS v2_workspace_revision_files_no_update
                BEFORE UPDATE ON v2_workspace_revision_files
                BEGIN SELECT RAISE(ABORT, 'Workspace revision manifests are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS v2_workspace_revision_files_no_delete
                BEFORE DELETE ON v2_workspace_revision_files
                BEGIN SELECT RAISE(ABORT, 'Workspace revision manifests are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS v2_workspace_revision_changes_no_update
                BEFORE UPDATE ON v2_workspace_revision_changes
                BEGIN SELECT RAISE(ABORT, 'Workspace revision changes are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS v2_workspace_revision_changes_no_delete
                BEFORE DELETE ON v2_workspace_revision_changes
                BEGIN SELECT RAISE(ABORT, 'Workspace revision changes are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS v2_workspace_revision_file_deltas_no_update
                BEFORE UPDATE ON v2_workspace_revision_file_deltas
                BEGIN SELECT RAISE(ABORT, 'Workspace revision deltas are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS v2_workspace_revision_file_deltas_no_delete
                BEFORE DELETE ON v2_workspace_revision_file_deltas
                BEGIN SELECT RAISE(ABORT, 'Workspace revision deltas are append-only'); END;
                """
            )

    @staticmethod
    def compute(repository_path: Path, run_root: Path) -> str:
        return WorkspaceReadOnlyGuard(repository_path, (run_root,)).capture().revision_id

    def current(self) -> str | None:
        row = self.database.connection.execute(
            "SELECT revision_id FROM v2_current_workspace_revision WHERE run_id=? AND branch_id=?",
            (self.run_id, self.branch_id),
        ).fetchone()
        return str(row["revision_id"]) if row is not None else None

    def _full_manifest(self) -> dict[str, tuple[int, int, int, int, str]]:
        return dict(WorkspaceReadOnlyGuard(self.repository_path, (self.run_root,)).snapshot())

    def _current_manifest(self, revision_id: str) -> dict[str, tuple[int, int, int, int, str]]:
        rows = self.database.connection.execute(
            "SELECT * FROM v2_workspace_revision_files "
            "WHERE run_id=? AND branch_id=? AND revision_id=? ORDER BY relative_path",
            (self.run_id, self.branch_id, revision_id),
        ).fetchall()
        full = {
            str(row["relative_path"]): (
                int(row["byte_count"]),
                int(row["modified_ns"]),
                int(row["changed_ns"]),
                int(row["mode"]),
                str(row["content_digest"]),
            )
            for row in rows
        }
        if full:
            return full

        chain: list[str] = []
        cursor: str | None = revision_id
        visited: set[str] = set()
        base: dict[str, tuple[int, int, int, int, str]] = {}
        while cursor is not None and cursor not in visited:
            visited.add(cursor)
            base_rows = self.database.connection.execute(
                "SELECT * FROM v2_workspace_revision_files "
                "WHERE run_id=? AND branch_id=? AND revision_id=? ORDER BY relative_path",
                (self.run_id, self.branch_id, cursor),
            ).fetchall()
            if base_rows:
                base = {
                    str(row["relative_path"]): (
                        int(row["byte_count"]),
                        int(row["modified_ns"]),
                        int(row["changed_ns"]),
                        int(row["mode"]),
                        str(row["content_digest"]),
                    )
                    for row in base_rows
                }
                break
            chain.append(cursor)
            parent = self.database.connection.execute(
                "SELECT previous_revision_id FROM v2_workspace_revisions "
                "WHERE run_id=? AND branch_id=? AND revision_id=? "
                "AND (previous_revision_id IS NULL OR previous_revision_id<>revision_id) "
                "ORDER BY created_at,revision_event_id LIMIT 1",
                (self.run_id, self.branch_id, cursor),
            ).fetchone()
            cursor = (
                None
                if parent is None or parent["previous_revision_id"] is None
                else str(parent["previous_revision_id"])
            )
        manifest = dict(base)
        for delta_revision in reversed(chain):
            delta_rows = self.database.connection.execute(
                "SELECT * FROM v2_workspace_revision_file_deltas "
                "WHERE run_id=? AND branch_id=? AND revision_id=? ORDER BY relative_path",
                (self.run_id, self.branch_id, delta_revision),
            ).fetchall()
            for row in delta_rows:
                relative = str(row["relative_path"])
                if bool(row["deleted"]):
                    manifest.pop(relative, None)
                    continue
                manifest[relative] = (
                    int(row["byte_count"]),
                    int(row["modified_ns"]),
                    int(row["changed_ns"]),
                    int(row["mode"]),
                    str(row["content_digest"]),
                )
        return manifest

    def _relative_changed_paths(self, changed_paths: Iterable[str]) -> tuple[str, ...]:
        result: list[str] = []
        guard = WorkspaceReadOnlyGuard(self.repository_path, (self.run_root,))
        for raw in changed_paths:
            path = Path(raw)
            candidate = Path(
                os.path.abspath(path if path.is_absolute() else self.repository_path / path)
            )
            try:
                relative = candidate.relative_to(self.repository_path).as_posix()
            except ValueError as exc:
                raise ValueError(f"changed path escapes repository: {raw}") from exc
            if guard._excluded(candidate):
                continue
            result.append(relative)
        return tuple(dict.fromkeys(result))

    def _incremental_manifest(
        self,
        previous_revision: str,
        changed_paths: Iterable[str],
    ) -> dict[str, tuple[int, int, int, int, str]]:
        manifest = self._current_manifest(previous_revision)
        if not manifest:
            return self._full_manifest()
        for relative in self._relative_changed_paths(changed_paths):
            path = self.repository_path / relative
            if not path.exists() and not path.is_symlink():
                manifest.pop(relative, None)
                continue
            if not path.is_file() and not path.is_symlink():
                continue
            stat = path.lstat()
            manifest[relative] = (
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_mode,
                WorkspaceReadOnlyGuard._content_digest(path),
            )
        return manifest

    def _git_frontier_paths(self, previous_revision: str) -> tuple[str, ...] | None:
        """Return the bounded dirty frontier for a committed Git worktree.

        A command can mutate files without a native fileChange event. For Git
        repositories with a HEAD, rehash the dirty frontier plus the previous
        frontier instead of walking every repository path after each command.
        """

        head = subprocess.run(
            ("git", "rev-parse", "--verify", "HEAD"),
            cwd=self.repository_path,
            check=False,
            capture_output=True,
        )
        if head.returncode != 0:
            return None
        guard = WorkspaceReadOnlyGuard(self.repository_path, (self.run_root,))
        scope = guard.git_pathspecs()
        commands = (
            ("diff", "--name-only", "-z", "--no-renames", "HEAD", "--", *scope),
            ("ls-files", "--others", "--exclude-standard", "-z", "--", *scope),
            ("ls-files", "--deleted", "-z", "--", *scope),
        )
        paths: list[str] = []
        for arguments in commands:
            completed = subprocess.run(
                ("git", *arguments),
                cwd=self.repository_path,
                check=False,
                capture_output=True,
            )
            if completed.returncode != 0:
                return None
            paths.extend(
                item.decode(errors="surrogateescape")
                for item in completed.stdout.split(b"\0")
                if item
            )
        paths.extend(
            str(row["relative_path"])
            for row in self.database.connection.execute(
                "SELECT relative_path FROM v2_workspace_revision_changes "
                "WHERE run_id=? AND branch_id=? AND revision_id=?",
                (self.run_id, self.branch_id, previous_revision),
            ).fetchall()
        )
        return tuple(dict.fromkeys(paths))

    def _reconciled_manifest(
        self,
        previous_revision: str,
    ) -> dict[str, tuple[int, int, int, int, str]]:
        """Detect unknown command mutations without rereading unchanged bodies."""

        previous = self._current_manifest(previous_revision)
        if not previous:
            return self._full_manifest()
        git_frontier = self._git_frontier_paths(previous_revision)
        if git_frontier is not None:
            return self._incremental_manifest(previous_revision, git_frontier)
        recently_changed = {
            str(row["relative_path"])
            for row in self.database.connection.execute(
                "SELECT relative_path FROM v2_workspace_revision_changes "
                "WHERE run_id=? AND branch_id=? AND revision_id=?",
                (self.run_id, self.branch_id, previous_revision),
            ).fetchall()
        }
        manifest: dict[str, tuple[int, int, int, int, str]] = {}
        guard = WorkspaceReadOnlyGuard(self.repository_path, (self.run_root,))
        for current, directories, files in os.walk(self.repository_path):
            current_path = Path(current)
            directories[:] = sorted(
                name for name in directories if not guard._excluded(current_path / name)
            )
            for name in sorted(files):
                path = current_path / name
                if guard._excluded(path):
                    continue
                try:
                    stat = path.lstat()
                except FileNotFoundError:
                    continue
                relative = path.relative_to(self.repository_path).as_posix()
                metadata = (
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                    stat.st_mode,
                )
                prior = previous.get(relative)
                if prior is not None and prior[:4] == metadata and relative not in recently_changed:
                    content_digest = prior[4]
                else:
                    try:
                        content_digest = WorkspaceReadOnlyGuard._content_digest(path)
                    except FileNotFoundError:
                        continue
                manifest[relative] = (*metadata, content_digest)
        return manifest

    @staticmethod
    def _revision_id(manifest: dict[str, tuple[int, int, int, int, str]]) -> str:
        value = WorkspaceReadOnlyGuard._content_manifest_digest(manifest)
        return "revision_" + value.removeprefix("sha256:")

    @staticmethod
    def _content_identity(metadata: tuple[int, int, int, int, str] | None) -> object:
        if metadata is None:
            return None
        return (metadata[3] & 0o170111, metadata[4])

    def capture(
        self,
        *,
        source_event_id: str,
        changed_paths: Iterable[str] | None = None,
        initial_receipt: WorkspaceSnapshotReceipt | None = None,
        raw_changes: Sequence[Mapping[str, object]] = (),
    ) -> WorkspaceRevisionReceipt:
        previous = self.current()
        previous_manifest = self._current_manifest(previous) if previous is not None else {}
        if previous is None:
            if initial_receipt is not None:
                manifest = dict(initial_receipt.manifest)
            else:
                manifest = self._full_manifest()
        elif changed_paths is None:
            manifest = self._reconciled_manifest(previous)
        else:
            changed = tuple(changed_paths)
            manifest = (
                self._incremental_manifest(previous, changed)
                if changed
                else self._reconciled_manifest(previous)
            )
        revision_id = self._revision_id(manifest)
        changed = previous is not None and previous != revision_id
        changed_relative_paths = (
            ()
            if previous is None
            else tuple(
                sorted(
                    path
                    for path in set(previous_manifest).union(manifest)
                    if self._content_identity(previous_manifest.get(path))
                    != self._content_identity(manifest.get(path))
                )
            )
        )
        symbol_bindings: tuple[ReferenceBindingRecord, ...] = ()
        if changed and self.reference_directory is not None:
            symbol_bindings = self.reference_directory.index_changed_files(
                revision_id=revision_id,
                source_event_id=source_event_id,
                paths=changed_relative_paths,
                changes=raw_changes,
            )
        event_id = stable_id(
            "revisionevent_",
            {
                "run": self.run_id,
                "branch": self.branch_id,
                "source": source_event_id,
                "revision": revision_id,
            },
        )
        artifact = self._persist_revision_artifact(
            revision_id=revision_id,
            previous_revision_id=previous,
            changed_paths=changed_relative_paths,
        )
        with self.database.transaction() as conn:
            if previous is None:
                conn.executemany(
                    "INSERT OR IGNORE INTO v2_workspace_revision_files VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        (
                            self.run_id,
                            self.branch_id,
                            revision_id,
                            relative,
                            metadata[0],
                            metadata[1],
                            metadata[2],
                            metadata[3],
                            metadata[4],
                            source_event_id,
                        )
                        for relative, metadata in manifest.items()
                    ),
                )
            elif changed:
                conn.executemany(
                    "INSERT OR IGNORE INTO v2_workspace_revision_file_deltas "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        (
                            self.run_id,
                            self.branch_id,
                            revision_id,
                            relative,
                            int(relative not in manifest),
                            *(manifest.get(relative, (None, None, None, None, None))),
                            source_event_id,
                        )
                        for relative in changed_relative_paths
                    ),
                )
            conn.execute(
                "INSERT OR IGNORE INTO v2_workspace_revisions VALUES(?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    self.run_id,
                    self.branch_id,
                    revision_id,
                    previous,
                    source_event_id,
                    int(changed),
                    utc_now(),
                ),
            )
            if previous is None:
                receipt_digest = (
                    initial_receipt.manifest_digest
                    if initial_receipt is not None
                    else WorkspaceReadOnlyGuard._content_manifest_digest(manifest)
                )
                conn.execute(
                    "INSERT OR IGNORE INTO v2_workspace_snapshot_receipts VALUES(?,?,?,?,?,?,?,?)",
                    (
                        self.run_id,
                        self.branch_id,
                        revision_id,
                        receipt_digest,
                        initial_receipt.vcs_fingerprint if initial_receipt is not None else None,
                        json.dumps(
                            list(initial_receipt.exclusion_policy)
                            if initial_receipt is not None
                            else sorted(WorkspaceReadOnlyGuard.DEFAULT_EXCLUDED_NAMES)
                        ),
                        source_event_id,
                        utc_now(),
                    ),
                )
            conn.execute(
                "INSERT INTO v2_current_workspace_revision VALUES(?,?,?,?,?) "
                "ON CONFLICT(run_id,branch_id) DO UPDATE SET "
                "revision_id=excluded.revision_id, source_event_id=excluded.source_event_id, "
                "updated_at=excluded.updated_at",
                (self.run_id, self.branch_id, revision_id, source_event_id, utc_now()),
            )
            conn.executemany(
                "INSERT OR IGNORE INTO v2_workspace_revision_changes VALUES(?,?,?,?,?,?)",
                (
                    (
                        self.run_id,
                        self.branch_id,
                        revision_id,
                        relative_path,
                        source_event_id,
                        utc_now(),
                    )
                    for relative_path in changed_relative_paths
                ),
            )
            conn.execute(
                "INSERT OR IGNORE INTO v2_workspace_revision_artifacts "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    self.run_id,
                    self.branch_id,
                    revision_id,
                    artifact["patch_path"],
                    artifact["receipt_path"],
                    artifact["patch_digest"],
                    artifact["patch_bytes"],
                    artifact["patch_state"],
                    source_event_id,
                    artifact["created_at"],
                ),
            )
            semantic_progress = record_workspace_progress(
                conn, run_id=self.run_id, branch_id=self.branch_id,
                revision_id=revision_id, source_event_id=source_event_id,
                manifest=manifest, changed_paths=changed_relative_paths,
            )
        return WorkspaceRevisionReceipt(
            revision_id=revision_id,
            previous_revision_id=previous,
            changed=changed,
            changed_paths=changed_relative_paths,
            source_event_id=source_event_id,
            symbol_bindings=symbol_bindings,
            semantic_progress=semantic_progress,
            patch_receipt_path=str(artifact["receipt_path"]),
            patch_digest=(
                str(artifact["patch_digest"])
                if artifact["patch_digest"] is not None
                else None
            ),
        )

    def _persist_revision_artifact(
        self,
        *,
        revision_id: str,
        previous_revision_id: str | None,
        changed_paths: Sequence[str],
    ) -> dict[str, object]:
        """Write the immutable code artifact before publishing its Revision.

        Git workspaces receive a full binary patch against their prepared HEAD,
        including untracked files without mutating the index. Non-Git
        workspaces still receive a durable manifest receipt and remain
        recoverable from the workspace/Trace Store, but cannot claim a portable
        patch representation.
        """

        existing = self.database.connection.execute(
            "SELECT patch_path,receipt_path,patch_digest,patch_bytes,patch_state,created_at "
            "FROM v2_workspace_revision_artifacts "
            "WHERE run_id=? AND branch_id=? AND revision_id=?",
            (self.run_id, self.branch_id, revision_id),
        ).fetchone()
        if existing is not None:
            return dict(existing)

        skipped: list[str] = []
        omitted_reason: str | None = None
        try:
            patch, state = self._portable_git_patch(skipped=skipped)
        except WorkspacePatchLimitError as exc:
            if os.environ.get(_REPOSITORY_STREAM_ENV) != "1":
                raise
            patch, state = None, PATCH_OMITTED_LIMIT
            omitted_reason = str(exc)
        # The patch is ``git diff HEAD`` at capture time.  Agents may commit
        # while they work, after which the same patch no longer applies to the
        # moved HEAD; the evaluation handoff therefore needs the commit the
        # patch was taken against, not whatever HEAD is at the end of the run.
        base_commit = (
            _git_head_commit(self.repository_path) if state == "DURABLE_GIT_PATCH" else None
        )
        created_at = utc_now()
        patch_digest = (
            "sha256:" + hashlib.sha256(patch).hexdigest() if patch is not None else None
        )
        artifact_root = self.run_root / "workspace-revisions"
        patch_path = artifact_root / f"{revision_id}.patch" if patch is not None else None
        if patch_path is not None:
            atomic_write_once(patch_path, patch)
        receipt_path = artifact_root / f"{revision_id}.json"
        receipt = {
            "schema": "codex-longterm-v2/workspace-revision-artifact@1",
            "artifact_id": stable_id(
                "workspace_artifact_",
                {"run": self.run_id, "branch": self.branch_id, "revision": revision_id},
            ),
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "revision_id": revision_id,
            "previous_revision_id": previous_revision_id,
            "changed_paths": list(changed_paths),
            "patch_path": str(patch_path) if patch_path is not None else None,
            "patch_digest": patch_digest,
            "patch_bytes": len(patch or b""),
            "patch_state": state,
            "base_commit": base_commit,
            "skipped_paths": skipped,
            "patch_omitted_reason": omitted_reason,
        }
        atomic_write_once(receipt_path, canonical_bytes(receipt))
        return {
            **receipt,
            "receipt_path": str(receipt_path),
            "created_at": created_at,
        }

    def _portable_git_patch(self, *, skipped: list[str] | None = None) -> tuple[bytes | None, str]:
        return _portable_git_patch_for_repository(
            repository=self.repository_path,
            run_root=self.run_root,
            skipped=skipped,
        )


# Compatibility name for scenario adapters and older callers. This is an
# alias to the authority itself, not a second tracker or independently writable
# workspace state.
WorkspaceRevisionTracker = WorkspaceStateAuthority


def materialize_current_revision_for_evaluation(
    *, repository_path: Path, run_root: Path
) -> dict[str, object]:
    """Publish the current durable Workspace Revision through Git ``HEAD``.

    Some benchmark collectors intentionally export ``base_commit..HEAD`` and
    therefore cannot see a correct but uncommitted working tree after an agent
    timeout.  The Workspace authority already owns an immutable binary patch
    for every committed Revision.  This function validates that exact receipt
    and creates a commit from a temporary index; it never infers progress from
    the live working tree and never changes file contents.
    """

    repository = Path(repository_path).expanduser().resolve()
    root = Path(run_root).expanduser().resolve()
    database_path = root / "v2-state.sqlite3"
    if not database_path.is_file():
        return {"status": "NO_DURABLE_STATE"}

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        has_current = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='v2_current_workspace_revision'"
        ).fetchone()
        has_artifact = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='v2_workspace_revision_artifacts'"
        ).fetchone()
        if has_current is None or has_artifact is None:
            return {"status": "NO_DURABLE_STATE"}
        rows = connection.execute(
            "SELECT c.run_id,c.branch_id,c.revision_id,a.patch_path,a.receipt_path,"
            "a.patch_digest,a.patch_bytes,a.patch_state "
            "FROM v2_current_workspace_revision c "
            "JOIN v2_workspace_revision_artifacts a "
            "ON a.run_id=c.run_id AND a.branch_id=c.branch_id "
            "AND a.revision_id=c.revision_id"
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        return {"status": "NO_DURABLE_REVISION"}
    if len(rows) != 1:
        raise RuntimeError("evaluation handoff found multiple current Workspace Revisions")
    row = rows[0]
    revision_id = str(row["revision_id"])
    patch_state = str(row["patch_state"])
    if patch_state != "DURABLE_GIT_PATCH":
        raise RuntimeError(
            f"current Workspace Revision has no portable Git patch: {patch_state}"
        )
    patch_path_value = row["patch_path"]
    if patch_path_value is None:
        raise RuntimeError("durable Git Workspace Revision has no patch path")
    patch_path = Path(str(patch_path_value)).expanduser().resolve()
    artifact_root = (root / "workspace-revisions").resolve()
    if not patch_path.is_relative_to(artifact_root) or not patch_path.is_file():
        raise RuntimeError("Workspace Revision patch escaped or is missing from run_root")
    patch = patch_path.read_bytes()
    if len(patch) != int(row["patch_bytes"]):
        raise RuntimeError("Workspace Revision patch byte count changed")
    actual_digest = "sha256:" + hashlib.sha256(patch).hexdigest()
    if actual_digest != str(row["patch_digest"]):
        raise RuntimeError("Workspace Revision patch digest changed")

    handoff_path = artifact_root / f"{revision_id}.evaluation-handoff.json"
    if handoff_path.is_file():
        prior = json.loads(handoff_path.read_text(encoding="utf-8"))
        current_head = _git_output(repository, "rev-parse", "HEAD")
        if (
            isinstance(prior, Mapping)
            and prior.get("revision_id") == revision_id
            and prior.get("checkpoint_commit") == current_head
        ):
            return dict(prior)
        raise RuntimeError("Workspace Revision evaluation handoff no longer matches Git HEAD")

    parent = _git_output(repository, "rev-parse", "HEAD")
    patch_for_apply = patch_path
    temporary_patch: Path | None = None
    # The durable patch is ``git diff HEAD`` as of its capture.  If the agent
    # committed afterwards (r5: M001's files became commits before M002), HEAD
    # already contains part of that diff and the patch no longer applies to
    # it.  Apply it to the commit it was taken against instead; the receipt
    # records that commit, and older receipts fall back to probing HEAD's
    # recent ancestors.
    receipt_base = _receipt_base_commit(Path(str(row["receipt_path"])))
    apply_base = parent
    # Agents are allowed to commit their work.  In that case the durable
    # receipt may still contain a pre-commit patch (the capture and commit can
    # occur in adjacent tool events), while Git HEAD already contains the
    # exact current Workspace Revision.  Replaying that stale ``new file``
    # patch into a temporary index would fail with "already exists in index".
    # The content-addressed revision remains authoritative: only when the
    # live workspace digest matches it do we accept HEAD or derive the small
    # post-commit dirty patch from the current worktree.
    current_revision = WorkspaceStateAuthority.compute(repository, root)
    if current_revision == revision_id:
        current_patch, current_state = _portable_git_patch_for_repository(
            repository=repository,
            run_root=root,
        )
        if current_state != "DURABLE_GIT_PATCH":
            raise RuntimeError(
                "current Workspace Revision is Git-backed but no portable "
                f"handoff patch is available: {current_state}"
            )
        if not current_patch:
            return {
                "status": "HEAD_ALREADY_AUTHORITATIVE",
                "revision_id": revision_id,
                "checkpoint_commit": parent,
                "patch_digest": actual_digest,
            }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".evaluation-patch-", dir=artifact_root
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(current_patch)
        temporary_patch = Path(temporary_name)
        patch_for_apply = temporary_patch

    if not patch:
        return {
            "status": "HEAD_ALREADY_AUTHORITATIVE",
            "revision_id": revision_id,
            "checkpoint_commit": parent,
            "patch_digest": actual_digest,
        }

    if patch_for_apply is patch_path:
        apply_base = _resolve_patch_base(
            repository,
            patch=patch_path,
            head=parent,
            recorded_base=receipt_base,
            artifact_root=artifact_root,
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".evaluation-index-", dir=artifact_root
    )
    os.close(descriptor)
    temporary_index = Path(temporary_name)
    temporary_index.unlink()
    environment = {**os.environ, "GIT_INDEX_FILE": str(temporary_index)}
    try:
        _run_git(repository, "read-tree", apply_base, env=environment)
        _run_git(
            repository,
            "apply",
            "--cached",
            "--binary",
            "--whitespace=nowarn",
            str(patch_for_apply),
            env=environment,
        )
        tree = _git_output(repository, "write-tree", env=environment)
        commit_environment = {
            **environment,
            "GIT_AUTHOR_NAME": "Homy Workspace Authority",
            "GIT_AUTHOR_EMAIL": "homy-workspace@example.invalid",
            "GIT_COMMITTER_NAME": "Homy Workspace Authority",
            "GIT_COMMITTER_EMAIL": "homy-workspace@example.invalid",
        }
        commit = _git_output(
            repository,
            "commit-tree",
            tree,
            "-p",
            parent,
            "-m",
            f"Homy durable Workspace Revision {revision_id}",
            env=commit_environment,
        )
        _run_git(repository, "update-ref", "HEAD", commit, parent)
        # Align the ordinary index with the newly published tree without
        # touching working-tree files. Any later, non-durable writes remain
        # visible as uncommitted changes rather than entering the artifact.
        _run_git(repository, "read-tree", commit)
    finally:
        temporary_index.unlink(missing_ok=True)
        if temporary_patch is not None:
            temporary_patch.unlink(missing_ok=True)

    handoff = {
        "schema": "codex-longterm-v2/evaluation-workspace-handoff@1",
        "status": "MATERIALIZED",
        "run_id": str(row["run_id"]),
        "branch_id": str(row["branch_id"]),
        "revision_id": revision_id,
        "parent_commit": parent,
        "patch_base_commit": apply_base,
        "checkpoint_commit": commit,
        "patch_digest": actual_digest,
        "source_receipt_path": str(row["receipt_path"]),
    }
    atomic_write_once(handoff_path, canonical_bytes(handoff))
    return handoff


def _git_head_commit(repository: Path) -> str | None:
    """HEAD's commit id, or ``None`` outside a Git worktree / before any commit."""

    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.decode("utf-8", errors="replace").strip()
    return value or None


def _receipt_base_commit(receipt_path: Path) -> str | None:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(receipt, Mapping):
        return None
    value = receipt.get("base_commit")
    return str(value) if isinstance(value, str) and value.strip() else None


# How far back from HEAD a legacy receipt (no ``base_commit``) is probed for the
# commit its patch was taken against.  Agents commit a handful of times per
# run, never hundreds.
_PATCH_BASE_PROBE_DEPTH = 32


def _patch_applies_to(
    repository: Path, *, commit: str, patch: Path, artifact_root: Path
) -> bool:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".probe-index-", dir=artifact_root)
    os.close(descriptor)
    temporary_index = Path(temporary_name)
    temporary_index.unlink()
    environment = {**os.environ, "GIT_INDEX_FILE": str(temporary_index)}
    try:
        _run_git(repository, "read-tree", commit, env=environment)
        completed = subprocess.run(
            [
                "git",
                "apply",
                "--cached",
                "--check",
                "--binary",
                "--whitespace=nowarn",
                str(patch),
            ],
            cwd=repository,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.returncode == 0
    finally:
        temporary_index.unlink(missing_ok=True)


def _resolve_patch_base(
    repository: Path,
    *,
    patch: Path,
    head: str,
    recorded_base: str | None,
    artifact_root: Path,
) -> str:
    """Commit whose tree the durable patch was diffed against.

    Prefer the commit the receipt recorded at capture time.  Receipts written
    before ``base_commit`` existed are probed against HEAD and its recent
    ancestors; the first commit the patch applies to cleanly is the base.
    """

    candidates: list[str] = []
    if recorded_base:
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{recorded_base}^{{commit}}"],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if exists.returncode == 0:
            candidates.append(recorded_base)
    ancestors = _git_output(
        repository, "rev-list", f"--max-count={_PATCH_BASE_PROBE_DEPTH}", head
    ).split()
    candidates.extend(commit for commit in ancestors if commit not in candidates)
    for commit in candidates:
        if _patch_applies_to(repository, commit=commit, patch=patch, artifact_root=artifact_root):
            return commit
    raise RuntimeError(
        "durable Workspace Revision patch applies to neither its recorded base commit "
        f"({recorded_base or 'unrecorded'}) nor the last {_PATCH_BASE_PROBE_DEPTH} commits "
        f"reachable from HEAD {head}"
    )


def _bounded_git_output(
    repository: Path,
    arguments: Sequence[str],
    *,
    max_bytes: int,
    accepted_returncodes: frozenset[int] = frozenset({0}),
    output_kind: str,
) -> bytes:
    """Read Git output incrementally and stop before an artifact can exhaust memory."""

    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            ("git", *arguments),
            cwd=repository,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        if process.stdout is None:
            process.kill()
            process.wait()
            raise RuntimeError("Git subprocess has no stdout pipe")
        chunks: list[bytes] = []
        total = 0
        try:
            while chunk := process.stdout.read(_GIT_OUTPUT_CHUNK_BYTES):
                total += len(chunk)
                if total > max_bytes:
                    process.kill()
                    process.wait()
                    raise WorkspacePatchLimitError(
                        f"Workspace Revision {output_kind} byte limit exceeded: "
                        f"observed more than {max_bytes} bytes"
                    )
                chunks.append(chunk)
            returncode = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        if returncode not in accepted_returncodes:
            stderr.seek(0)
            detail = stderr.read(4096).decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"git {arguments[0]} failed while generating Workspace Revision "
                f"{output_kind}"
                + (f": {detail}" if detail else "")
            )
    return b"".join(chunks)


def _portable_git_patch_for_repository(
    *, repository: Path, run_root: Path, skipped: list[str] | None = None
) -> tuple[bytes | None, str]:
    """Return the current worktree patch without changing Git control state.

    Untracked files above ``_MAX_PORTABLE_UNTRACKED_FILE_BYTES`` are not
    added to the patch; their paths are appended to ``skipped`` when given.
    """

    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if inside.returncode != 0 or inside.stdout.strip() != b"true":
        return None, "UNAVAILABLE_NON_GIT"
    guard = WorkspaceReadOnlyGuard(repository, (run_root,))
    scope = guard.git_pathspecs()
    tracked_paths_output = _bounded_git_output(
        repository,
        (
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            "HEAD",
            "--",
            *scope,
        ),
        max_bytes=_MAX_PORTABLE_PATCH_PATH_LIST_BYTES,
        output_kind="patch path listing",
    )
    untracked_output = _bounded_git_output(
        repository,
        (
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *scope,
        ),
        max_bytes=_MAX_PORTABLE_PATCH_PATH_LIST_BYTES,
        output_kind="patch path listing",
    )
    tracked_paths = tuple(path for path in tracked_paths_output.split(b"\0") if path)
    untracked_paths = tuple(path for path in untracked_output.split(b"\0") if path)
    path_count = len(dict.fromkeys((*tracked_paths, *untracked_paths)))
    if path_count > _MAX_PORTABLE_PATCH_PATHS:
        raise WorkspacePatchLimitError(
            "Workspace Revision patch path count exceeded: "
            f"observed {path_count}, limit {_MAX_PORTABLE_PATCH_PATHS}"
        )

    tracked = _bounded_git_output(
        repository,
        (
            "-c",
            "core.fileMode=false",
            "diff",
            "--binary",
            "HEAD",
            "--",
            *scope,
        ),
        max_bytes=_MAX_PORTABLE_PATCH_BYTES,
        output_kind="patch",
    )
    payload = bytearray(tracked)
    for raw_path in untracked_paths:
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        try:
            size = (repository / relative).lstat().st_size
        except OSError:
            size = 0
        if size > _MAX_PORTABLE_UNTRACKED_FILE_BYTES:
            if skipped is not None:
                skipped.append(relative)
            continue
        addition = _bounded_git_output(
            repository,
            ("diff", "--no-index", "--binary", "--", "/dev/null", relative),
            max_bytes=_MAX_PORTABLE_PATCH_BYTES - len(payload),
            accepted_returncodes=frozenset({0, 1}),
            output_kind="patch",
        )
        payload.extend(addition)
    return bytes(payload), "DURABLE_GIT_PATCH"


def _run_git(
    repository: Path, *arguments: str, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed during evaluation handoff: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    return completed


def _git_output(
    repository: Path, *arguments: str, env: Mapping[str, str] | None = None
) -> str:
    return _run_git(repository, *arguments, env=env).stdout.decode(
        "utf-8", errors="strict"
    ).strip()
