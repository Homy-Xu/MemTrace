"""A repository is one Task; public milestone releases are append-only inputs.

This adapter never reads evaluator reports, hidden tests, future SRS files or
reference patches. Submission tags are observations, NOT verification facts.
The ordinary trace ledger, Trace Store, MTG, revision tracking and recovery remain owners
of execution state. No extra planning call or per-milestone database is used.
"""
from __future__ import annotations

import fcntl
import json
import re
import subprocess
import uuid
import sqlite3
import tarfile
from dataclasses import replace
from pathlib import Path

from ..contracts import (
    Event,
    EventGroup,
    EvidenceDraft,
    EvidenceKey,
    FactType,
    PlanSpec,
    digest,
    utc_now,
)
from ..durability import atomic_write_once
from ..page_store.policy import TailReason
from ..page_store.synopsis import memory_ref_for_page


def _is_volatile_runtime_path(path: Path) -> bool:
    """Return whether ``path`` may legitimately disappear during a snapshot.

    SQLite creates and removes ``-wal``/``-shm`` sidecars while a connection is
    checkpointing.  The repository lock serializes our writers, but it cannot
    make SQLite's sidecar lifecycle atomic with ``tarfile``'s recursive
    ``lstat`` calls.  These files are deliberately excluded from the archive,
    so a disappearance is safe to ignore.  A missing ordinary runtime file is
    still an error and must not be silently converted into a partial snapshot.
    """

    return path.name.endswith(("-wal", "-shm", ".lock"))


def _is_verifier_scratch_path(path: Path) -> bool:
    """Return whether ``path`` is a regression-guard checkout or build cache."""

    if path.name == "worktrees" and path.parent.name == "swe-milestone-verifier":
        return True
    return path.name.startswith("cargo-target-") or path.name.endswith("-verifier-scratch")


def _add_snapshot_tree(archive: tarfile.TarFile, root: Path, include) -> None:
    """Add a runtime tree without racing SQLite sidecar removal.

    ``TarFile.add(root, recursive=True)`` performs a fresh ``lstat`` after
    walking each directory.  A SQLite ``-shm`` file can disappear in that
    small window, which used to abort the whole official scoring cleanup.  We
    enumerate the tree ourselves, skip volatile sidecars before adding them,
    and retry the same classification if a sidecar disappears between
    enumeration and ``TarFile.add``.  Stable files remain fail-fast.
    """

    pending: list[tuple[Path, str]] = [(root, "repository-runtime")]
    while pending:
        path, arcname = pending.pop()
        try:
            path.lstat()
        except FileNotFoundError:
            if _is_volatile_runtime_path(path):
                continue
            raise
        if path.is_symlink() or _is_volatile_runtime_path(path):
            continue
        if path.is_dir() and _is_verifier_scratch_path(path):
            # Persistent guard checkouts and cargo target directories are
            # rebuildable caches, not runtime state; archiving them would
            # add tens of gigabytes on every process exit.
            continue
        if path.is_dir():
            try:
                children = sorted(path.iterdir(), key=lambda item: item.name)
            except FileNotFoundError:
                if _is_volatile_runtime_path(path):
                    continue
                raise
            for child in reversed(children):
                pending.append((child, f"{arcname}/{child.name}"))
        try:
            archive.add(path, arcname=arcname, recursive=False, filter=include)
        except FileNotFoundError:
            if _is_volatile_runtime_path(path):
                continue
            raise


def export_snapshot(run_root: str, public_root: str):
    """Export a quiescent state before the official runner removes its container.

    The lock prevents overlapping invocations. No live -wal/-shm is copied.
    A completed archive is renamed atomically on the same local filesystem.
    """
    root, public = Path(run_root).resolve(), Path(public_root).resolve()
    if not root.is_relative_to(public):
        raise ValueError("runtime snapshot escaped its project workspace")
    with (root / "repository.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for database in root.glob("*.sqlite*"):
            if database.name.endswith(("-wal", "-shm")) or not database.is_file():
                continue
            with sqlite3.connect(database) as connection:
                busy, _, _ = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if busy:
                    raise RuntimeError("runtime database still has an active writer")
        target = public / "repository-runtime-snapshot.tar"
        temporary = target.with_suffix(".tar.partial")
        with tarfile.open(temporary, "w") as archive:
            def include(info):
                if info.name.endswith(("-wal", "-shm", ".lock")) or info.issym() or info.islnk():
                    return None
                return info
            _add_snapshot_tree(archive, root, include)
        temporary.chmod(0o644)
        temporary.replace(target)
        # The official host collector runs under the Slurm user while the
        # disposable container may create the bind mount as fakeroot.  Keep
        # directory contents private but allow the owner-side collector to
        # traverse the two known task directories and read the published
        # archive.  This is deliberately narrower than making the runtime
        # tree world-readable.
        for directory in (public, public / "homy-v2-runs", root):
            try:
                directory.chmod(0o751)
            except OSError as exc:
                raise RuntimeError(
                    f"runtime export directory is not host-traversable: {directory}"
                ) from exc

_AVAILABLE = re.compile(r"^- ([A-Za-z0-9_.-]+): See SRS at (/e2e_workspace/srs/[A-Za-z0-9_.-]+_SRS\.md)$")
_FR_LINE = re.compile(
    r"^(?:#{1,6}\s+|\*\*)?(FR[- ]?\d+)(?:\*\*)?\s*[:.\-–—]?\s*(.*)$",
    re.IGNORECASE,
)
FR_CURSOR_NAME = "official-fr-cursor.json"
SLICE_STATE_NAME = "repository-slice-state.json"
FOCUS_FR_COUNT = 2
FOCUS_EXCERPT_CHARS = 800
CATALOG_TITLE_CHARS = 160


def parse_public_srs_frs(text: str) -> tuple[dict[str, str], ...]:
    """Split public SRS requirements into bounded, addressable FR slices."""

    items: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    body: list[str] = []
    for raw in (text or "").splitlines():
        match = _FR_LINE.match(raw.strip())
        if match:
            if current is not None:
                current["body"] = "\n".join(body).strip()
                items.append(current)
            ident = re.sub(r"[\s-]+", "", match.group(1).upper())
            current = {"id": ident, "title": match.group(2).strip().strip("*").strip()}
            body = []
            continue
        if current is not None:
            body.append(raw)
    if current is not None:
        current["body"] = "\n".join(body).strip()
        items.append(current)
    return tuple(items)


def remaining_unsubmitted_queue(
    public_root: str | Path,
    repository: str | Path,
) -> tuple[str, ...]:
    """Return released official IDs which do not yet have an implementation tag.

    The public queue is the only release authority here. This helper never
    reads hidden tests, evaluator output, or future SRS files, and is used only
    to keep one continuous repository task alive while released work remains.
    """

    queue = Path(public_root) / "TASK_QUEUE.md"
    if not queue.is_file():
        return ()
    try:
        lines = queue.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    available: list[str] = []
    for line in lines:
        match = _AVAILABLE.fullmatch(line.strip())
        if match:
            available.append(match.group(1))
    try:
        raw = subprocess.run(
            [
                "git", "-C", str(repository), "for-each-ref",
                "--format=%(refname:short)", "refs/tags/agent-impl-*",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ()
    submitted = {
        tag.removeprefix("agent-impl-")
        for tag in raw.splitlines()
        if tag.startswith("agent-impl-")
    }
    return tuple(mid for mid in available if mid not in submitted)


class RepositoryStream:
    def __init__(self, contract_path: Path, repository: Path):
        self.contract_path = contract_path.resolve(strict=True)
        self.contract = json.loads(self.contract_path.read_text())
        if self.contract.get("schema") != "homy/repository-stream@1":
            raise ValueError("invalid repository stream contract")
        self.project = str(self.contract["project_id"])
        self.selected = tuple(self.contract["selected_ids"])
        if not self.project or not self.selected or len(set(self.selected)) != len(self.selected):
            raise ValueError("repository stream requires a unique, nonempty official selection")
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", x) for x in self.selected):
            raise ValueError("unsafe milestone identity")
        self.repository = repository.resolve(strict=True)
        self.public_root = self.contract_path.parent.parent
        self._lock = None
        self._last_digest = None
        self.new_release = False
        self.run_root: Path | None = None
        self._slice_state: dict[str, object] | None = None

    def open(self, request):
        self.run_root = request.run_root.resolve()
        request.run_root.mkdir(parents=True, exist_ok=True)
        self._lock = (request.run_root / "repository.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.close()
            raise RuntimeError("repository Task already has an active writer") from None
        identity = {"schema": "homy/repository-session@1", "project_id": self.project,
                    "contract_digest": digest(self.contract), "run_id": request.run_id,
                    "repository": str(self.repository), "user_task_digest": digest(request.user_task)}
        path = request.run_root / "repository-session.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            self.close()
            raise ValueError("repository session identity changed; refusing cross-task reuse")
        if not path.exists():
            atomic_write_once(path, json.dumps(identity, sort_keys=True).encode())
        self.invocation = request.run_root / "invocations" / uuid.uuid4().hex
        self.invocation.mkdir(parents=True)
        self._slice_state = None

    def close(self):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def execution_handoff(self, execution) -> dict[str, object]:
        """Return the bounded official queue state for an Epoch handoff.

        The repository stream is the public release authority, while the TPG
        remains the internal execution map. This projection joins the two
        authorities without turning an official release into a completion
        claim. It is deliberately compact: full SRS bodies stay behind the
        public queue and Page/MemoryRef addresses.
        """

        state = self.snapshot()
        pending = [
            item for item in state["available"]
            if item["milestone_id"] not in state["submitted"]
        ]
        current = execution.registry.current(execution.request.run_id)
        plan = execution.registry.active_plan(execution.request.run_id)
        planned_target, released = self._planned_release_target(plan, pending)
        working_plan = None
        semantic = getattr(execution, "semantic", None)
        if semantic is not None and hasattr(semantic, "latest_working_plan_observation"):
            working_plan = semantic.latest_working_plan_observation(
                execution.request.run_id,
                execution.request.branch_id,
            )
        if released is not None and isinstance(working_plan, dict):
            self._apply_working_plan_focus(released, working_plan)
        edges = [
            {
                "source_id": str(edge.get("source_id", "")),
                "target_id": str(edge.get("target_id", "")),
                "type": str(edge.get("type", "")),
            }
            for edge in self.contract.get("dependencies", ())
            if str(edge.get("source_id", "")) in state["submitted"]
            or str(edge.get("target_id", "")) in {item["milestone_id"] for item in pending}
        ]
        compact_pending = [
            {
                key: item[key]
                for key in (
                    "milestone_id",
                    "srs_path",
                    "srs_digest",
                    "focus_index",
                    "focus_frs",
                    "focus_source",
                    "current_activity",
                    "predecessors",
                )
                if key in item
            }
            for item in pending[:8]
        ]
        handoff = {
            "schema": "homy/repository-execution-handoff@1",
            "authority": "PUBLIC_TASK_QUEUE_AND_FUNC_DAG",
            "project_id": self.project,
            "same_repository_task": True,
            "current_internal_milestone": {
                "canonical_id": str(getattr(current, "canonical_id", "")),
                "identity_id": str(getattr(current, "identity_id", "")),
                "status": str(getattr(current, "status", "")),
            },
            "current_official_milestone": (
                str(released["milestone_id"]) if released is not None else None
            ),
            "current_official_target_internal_id": (
                str(planned_target.canonical_id) if planned_target is not None else None
            ),
            "current_native_working_plan": working_plan,
            "released_unsubmitted": compact_pending,
            "submitted_revisions": dict(state["submitted"]),
            "func_edges": edges[:32],
            "route_rule": (
                "The official queue selects the next release. The FUNC DAG selects only "
                "released predecessors. Internal status is navigation history, not official score."
            ),
            "resume_rule": (
                "Continue the same repository task at current_official_milestone; preserve "
                "earlier Pages and MemoryRefs, revalidate only affected symbols after revision changes."
            ),
        }
        previous = self.latest_continuity_receipt()
        if previous is not None:
            previous_slice = previous.get("slice")
            handoff["previous_invocation"] = {
                "state": previous.get("state"),
                "thread_id": previous.get("thread_id"),
                "revision_id": previous.get("revision_id"),
                "epoch_id": previous.get("epoch_id"),
                "slice": (
                    {
                        key: previous_slice.get(key)
                        for key in (
                            "kind",
                            "reason",
                            "revision_id",
                            "completed_execution_turns",
                            "resumable",
                        )
                    }
                    if isinstance(previous_slice, dict)
                    else None
                ),
                "resume_same_thread": True,
            }
        return handoff

    @staticmethod
    def _apply_working_plan_focus(
        released: dict[str, object],
        working_plan: dict[str, object],
    ) -> None:
        """Project current native work onto public FRs without claiming progress.

        Native Plan status is navigation-only.  It may select the FR window
        shown to a resumed Epoch, but it never advances the durable cursor or
        verifies a requirement.  This keeps the handoff aligned with what the
        model was actually doing while preserving the official evaluator as
        completion authority.
        """

        raw_items = working_plan.get("items")
        if not isinstance(raw_items, list):
            return
        active = [
            item
            for item in raw_items
            if isinstance(item, dict)
            and str(item.get("status", "")).casefold() not in {"completed", "done"}
        ]
        if not active:
            return
        activity = [str(item.get("title", "")).strip() for item in active]
        activity = [item for item in activity if item]
        if not activity:
            return
        requested_ids: list[str] = []
        for title in activity:
            for raw in re.findall(r"\bFR[-\s]?(\d+(?:\.\d+)?)\b", title, flags=re.IGNORECASE):
                value = "FR" + raw
                if value not in requested_ids:
                    requested_ids.append(value)
        catalog = released.get("fr_catalog")
        catalog_items = catalog if isinstance(catalog, list) else []
        by_id = {
            str(item.get("id")): item
            for item in catalog_items
            if isinstance(item, dict)
        }
        focused = [by_id[item] for item in requested_ids if item in by_id][:FOCUS_FR_COUNT]
        released["focus_frs"] = focused
        released["focus_source"] = "CODEX_NATIVE_PLAN_NAVIGATION"
        released["current_activity"] = activity[:4]

    def latest_continuity_receipt(self) -> dict[str, object] | None:
        """Return the latest completed invocation receipt, excluding this writer."""

        if self.run_root is None:
            return None
        candidates = [
            path
            for path in self.run_root.glob("invocations/*/continuity-receipt.json")
            if path.parent != getattr(self, "invocation", None)
        ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        try:
            value = json.loads(latest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return dict(value) if isinstance(value, dict) else None

    @property
    def resumed_invocation(self) -> bool:
        return self.latest_continuity_receipt() is not None

    def _fr_cursor_path(self) -> Path | None:
        return None if self.run_root is None else self.run_root / FR_CURSOR_NAME

    def _load_fr_cursor(self) -> dict[str, int]:
        path = self._fr_cursor_path()
        if path is None or not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return (
            {str(key): int(value) for key, value in raw.items() if str(key)}
            if isinstance(raw, dict)
            else {}
        )

    def advance_focus_fr(self, milestone_id: str, *, evidence_digest: str) -> int:
        """Advance an FR window only after a caller supplies positive evidence.

        Route stalls and physical slice boundaries are not progress.  Requiring
        an evidence digest prevents those boundaries from silently skipping an
        unimplemented public requirement.
        """

        mid = str(milestone_id).strip()
        path = self._fr_cursor_path()
        proof = str(evidence_digest).strip()
        if not mid or not proof or path is None:
            return 0
        cursor = self._load_fr_cursor()
        cursor[mid] = int(cursor.get(mid, 0)) + 1
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(cursor, sort_keys=True) + "\n", encoding="ascii")
        temporary.replace(path)
        return cursor[mid]

    def suspend_invocation(
        self,
        *,
        kind: str,
        reason: str,
        source_event_id: str,
        revision_id: str,
        completed_execution_turns: int,
        discarded_continuations: int,
        execution_handoff: dict[str, object],
        resumable: bool,
    ) -> dict[str, object]:
        """Persist a physical slice boundary without changing Task truth."""

        if self._slice_state is not None:
            return dict(self._slice_state)
        state = {
            "schema": "homy/repository-slice-state@1",
            "kind": str(kind),
            "reason": str(reason),
            "source_event_id": str(source_event_id),
            "revision_id": str(revision_id),
            "completed_execution_turns": int(completed_execution_turns),
            "discarded_continuations": int(discarded_continuations),
            "resumable": bool(resumable),
            "execution_handoff": dict(execution_handoff),
        }
        atomic_write_once(
            self.invocation / SLICE_STATE_NAME,
            json.dumps(state, ensure_ascii=False, sort_keys=True).encode(),
        )
        self._slice_state = state
        return dict(state)

    def _fr_planning_fields(self, milestone_id: str, content: str) -> dict[str, object]:
        frs = parse_public_srs_frs(content)
        catalog = [
            {"id": item["id"], "title": item["title"][:CATALOG_TITLE_CHARS]}
            for item in frs
        ]
        index = max(0, int(self._load_fr_cursor().get(milestone_id, 0)))
        if frs:
            index = min(index, len(frs) - 1)
            window = frs[index : index + FOCUS_FR_COUNT]
            focus = [
                {
                    "id": item["id"],
                    "title": item["title"][:CATALOG_TITLE_CHARS],
                    "excerpt": item["body"][:FOCUS_EXCERPT_CHARS],
                }
                for item in window
            ]
        else:
            focus = [
                {"id": "SRS", "title": milestone_id, "excerpt": (content or "")[:FOCUS_EXCERPT_CHARS]}
            ]
        return {"fr_catalog": catalog, "focus_frs": focus, "focus_index": index}

    def fr_catalog(self, official_id: str) -> list[dict[str, str]]:
        """Public FR catalogue of one official ID, also after it left the queue.

        A tagged ID disappears from ``TASK_QUEUE.md`` at once, but its SRS
        file stays; the FR coverage self-check runs after the tag exists.
        """

        mid = str(official_id).strip()
        if mid not in self.selected:
            return []
        path = self.public_root / "srs" / f"{mid}_SRS.md"
        if path.resolve().parent != (self.public_root / "srs").resolve():
            return []
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return []
        return [
            {"id": item["id"], "title": item["title"][:CATALOG_TITLE_CHARS]}
            for item in parse_public_srs_frs(content)
        ]

    def snapshot(self) -> dict:
        queue = self.public_root / "TASK_QUEUE.md"
        text = queue.read_text(encoding="utf-8")
        available = []
        for line in text.splitlines():
            match = _AVAILABLE.fullmatch(line.strip())
            if not match:
                continue
            mid, visible_path = match.groups()
            if mid not in self.selected:
                raise ValueError("public queue contains an unselected milestone")
            path = self.public_root / "srs" / f"{mid}_SRS.md"
            if path.resolve().parent != (self.public_root / "srs").resolve():
                raise ValueError("SRS path escapes the public stream")
            content = path.read_text(encoding="utf-8")
            available.append({
                "milestone_id": mid,
                "srs_path": visible_path,
                "srs_digest": digest(content),
                "requirements": content,
                **self._fr_planning_fields(mid, content),
            })
        raw = subprocess.run(
            ["git", "-C", str(self.repository), "for-each-ref", "--format=%(refname:short)",
             "refs/tags/agent-impl-*"], check=True, capture_output=True, text=True, timeout=10).stdout
        submitted = {}
        for tag in raw.splitlines():
            mid = tag.removeprefix("agent-impl-")
            if mid not in self.selected:
                continue
            commit = subprocess.run(["git", "-C", str(self.repository), "rev-parse", "--verify",
                                     f"refs/tags/{tag}^{{commit}}"], check=True,
                                    capture_output=True, text=True, timeout=10).stdout.strip()
            submitted[mid] = commit
        visible = {x["milestone_id"] for x in available} | set(submitted)
        for item in available:
            item["predecessors"] = [
                {"milestone_id": edge["source_id"], "relation": edge["type"],
                 "submitted_revision": submitted.get(edge["source_id"]),
                 "memory_entity": f"official-milestone:{self.project}:{edge['source_id']}",
                 "verified": False}
                for edge in self.contract.get("dependencies", ())
                if edge["target_id"] == item["milestone_id"] and edge["source_id"] in visible]
        return {"project_id": self.project, "available": available, "submitted": submitted,
                "all_submitted": set(submitted) == set(self.selected),
                "grading": "UNKNOWN_TO_AGENT"}

    def planning_context(self) -> str:
        snapshot = self.snapshot()
        return ("This is ONE continuing repository task. Official milestones are releases of "
                "requirements in that task, not independent tasks. Preserve prior implementations "
                "and use their Pages/MemoryRefs; revalidate affected behavior at the new revision. "
                "Plan actual currently released requirements, not a workflow of queue/commit/review "
                "steps. A submission is not a pass. An empty queue can mean waiting, not completion.\n"
                + json.dumps(snapshot, ensure_ascii=False))

    @staticmethod
    def _primary_official_reference(value: object, official_ids) -> str | None:
        """Return the first exact official ID named by one plan field.

        A projected plan item often says ``Verify M12.1, then rerun M06/M11``.
        The later IDs are regression scope, not ownership.  Binding every ID
        mentioned anywhere made that node ambiguous and preserved the
        normalizer's synthetic list-order dependency.  The first exact ID is
        the plan item's release label; official FUNC edges remain the sole
        authority for dependencies between releases.
        """

        text = str(value)
        found: list[tuple[int, int, str]] = []
        for official_id in official_ids:
            match = re.search(
                r"(?<![A-Za-z0-9_.])"
                + re.escape(str(official_id))
                + r"(?![A-Za-z0-9_.])",
                text,
            )
            if match is not None:
                found.append((match.start(), -len(str(official_id)), str(official_id)))
        return min(found)[2] if found else None

    @classmethod
    def _milestone_official_matches(cls, milestone, official_ids, native_plan=None):
        """Resolve release ownership without treating cross-references as edges.

        Lightweight projection keeps the authoritative release labels in the
        native source items.  Prefer those labels.  If a node genuinely groups
        source items owned by different official releases, return all owners so
        the caller records an ambiguity instead of guessing.
        """

        candidates = tuple(dict.fromkeys(str(item) for item in official_ids))
        source_ids = set(getattr(milestone, "source_plan_item_ids", ()))
        source_fields = ()
        if native_plan is not None and source_ids:
            source_fields = tuple(
                getattr(item, "title", "")
                for item in getattr(native_plan, "items", ())
                if getattr(item, "source_step_id", "") in source_ids
            )
        source_matches = tuple(
            dict.fromkeys(
                match
                for field in source_fields
                if (match := cls._primary_official_reference(field, candidates))
            )
        )
        if source_matches:
            return source_matches

        fields = (
            getattr(milestone, "canonical_id", ""),
            getattr(milestone, "title", ""),
            getattr(milestone, "description", ""),
            getattr(milestone, "objective", ""),
            getattr(milestone, "scope", ""),
            getattr(milestone, "target_outcome", ""),
            *getattr(milestone, "completion_criteria", ()),
            *getattr(milestone, "verification", ()),
        )
        return tuple(
            dict.fromkeys(
                match
                for field in fields
                if (match := cls._primary_official_reference(field, candidates))
            )
        )

    @classmethod
    def _milestone_mentions(cls, milestone, official_id: str, native_plan=None) -> bool:
        """Backward-compatible exact-reference predicate."""

        return official_id in cls._milestone_official_matches(
            milestone, (official_id,), native_plan
        )

    def align_plan(self, plan: PlanSpec) -> tuple[PlanSpec, dict[str, object]]:
        """Project official ``FUNC`` edges onto the native navigation skeleton.

        The native normalizer preserves list order as a conservative chain for
        ordinary coding tasks.  That chain is false for SWE-Milestone roots:
        the public queue is the release authority and ``dependencies.csv`` is
        the only authority for executable dependencies.  TEXT/NFR relations
        remain useful navigation metadata but never gate execution.

        Unmapped native nodes retain their model-authored dependencies.  A
        mapped official node receives only mapped official ``FUNC`` parents;
        an ambiguous node is left unchanged rather than guessed.
        """

        state = self.snapshot()
        visible = {
            *(item["milestone_id"] for item in state["available"]),
            *state["submitted"].keys(),
        }
        ordered_visible = tuple(item for item in self.selected if item in visible)
        official_by_canonical: dict[str, str] = {}
        ambiguous: dict[str, tuple[str, ...]] = {}
        for milestone in plan.milestones:
            matches = self._milestone_official_matches(
                milestone,
                ordered_visible,
                plan.native_plan,
            )
            if len(matches) == 1:
                official_by_canonical[milestone.canonical_id] = matches[0]
            elif len(matches) > 1:
                ambiguous[milestone.canonical_id] = matches

        canonical_by_official = {
            official_id: canonical_id
            for canonical_id, official_id in official_by_canonical.items()
        }
        official_dependencies: dict[str, tuple[str, ...]] = {}
        for canonical_id, official_id in official_by_canonical.items():
            official_dependencies[canonical_id] = tuple(
                dict.fromkeys(
                    canonical_by_official[str(edge["source_id"])]
                    for edge in self.contract.get("dependencies", ())
                    if str(edge.get("type", "")).upper() == "FUNC"
                    and str(edge.get("target_id", "")) == official_id
                    and str(edge.get("source_id", "")) in canonical_by_official
                )
            )

        aligned_milestones = tuple(
            replace(
                milestone,
                depends_on=official_dependencies[milestone.canonical_id],
            )
            if milestone.canonical_id in official_dependencies
            else milestone
            for milestone in plan.milestones
        )
        aligned = replace(plan, milestones=aligned_milestones)
        receipt = {
            "schema": "homy/repository-plan-alignment@1",
            "official_to_internal": {
                official_id: canonical_by_official[official_id]
                for official_id in ordered_visible
                if official_id in canonical_by_official
            },
            "ambiguous_internal_nodes": {
                key: list(value) for key, value in ambiguous.items()
            },
            "executable_edge_type": "FUNC",
            "queue_order_is_dependency": False,
            "changed_dependencies": [
                milestone.canonical_id
                for milestone, replacement in zip(plan.milestones, aligned_milestones)
                if milestone.depends_on != replacement.depends_on
            ],
        }
        return aligned, receipt

    def observe(self, execution):
        """Persist only new public facts; unchanged polling adds no WAL or model Turn."""
        state = self.snapshot()
        fingerprint = digest(state)
        if fingerprint == self._last_digest:
            return state, None
        event_id = "stream_" + digest({"run": execution.request.run_id, "state": state})
        current = execution.registry.current(execution.request.run_id)
        with execution.registry.database.transaction() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS v2_repository_public_releases "
                         "(run_id TEXT, milestone_id TEXT, srs_digest TEXT, event_id TEXT, "
                         "PRIMARY KEY(run_id,milestone_id,srs_digest))")
            seen = {(r[0], r[1]) for r in conn.execute(
                "SELECT milestone_id,srs_digest FROM v2_repository_public_releases WHERE run_id=?",
                (execution.request.run_id,)).fetchall()}
        self.new_release = bool(seen) and any(
            (x["milestone_id"], x["srs_digest"]) not in seen
            and x["milestone_id"] not in state["submitted"] for x in state["available"])
        if not execution.page_store.has_durable_event(event_id):
            facts = tuple(EvidenceDraft(
                EvidenceKey(FactType.USER_CONSTRAINT,
                            f"official-milestone:{self.project}:{item['milestone_id']}",
                            "released_requirements", "historical:" + item["srs_digest"],
                            execution.request.branch_id),
                item, must_preserve=True,
            ) for item in state["available"])
            event = Event(event_id=event_id, event_type="REPOSITORY_STREAM_OBSERVATION",
                          payload=state, facts=facts,
                          milestone_id=current.identity_id, revision_id=execution.revision_id,
                          entity_refs=tuple(f.key.canonical_entity_id for f in facts))
            group = EventGroup(group_id="group_" + event_id, group_type="REPOSITORY_STREAM",
                               run_id=execution.request.run_id, branch_id=execution.request.branch_id,
                               revision_id=execution.revision_id, events=(event,),
                               milestone_id=current.identity_id, semantic_boundary=True)
            sealed = execution.page_store.append_group(group)
            if sealed:
                execution._promote_pages((sealed,))
            # Ordinary Page projection owns section directories and MemoryRefs.
            tail = execution.page_store.checkpoint(TailReason.USER_CHECKPOINT)
            if tail is not None:
                execution._promote_pages((tail,))
        with execution.registry.database.transaction() as conn:
            conn.executemany("INSERT OR IGNORE INTO v2_repository_public_releases VALUES(?,?,?,?)",
                [(execution.request.run_id, x["milestone_id"], x["srs_digest"], event_id)
                 for x in state["available"]])
        self._last_digest = fingerprint
        execution.trace.record("REPOSITORY_STREAM_OBSERVED", source_event_id=event_id,
                               official_available=[x["milestone_id"] for x in state["available"]],
                               submitted_revisions=state["submitted"], run_id=execution.request.run_id,
                               internal_milestone_id=current.identity_id, grading="NOT_INFERRED")
        return state, event_id

    @staticmethod
    def _planned_release_target(plan, pending, official_ids=None):
        """Resolve the next public release to an existing native-plan node.

        A repository stream is one long task.  The public milestone ID may be
        embedded in a native-plan title, description, outcome or criterion.
        Matching is deterministic and bounded; a release never invents a
        new internal node or marks its predecessor verified.

        ``official_ids`` is the ID universe ownership is judged against.  The
        caller passes the whole official selection so a node that names two
        IDs keeps the same single release label whether or not the other ID
        is still pending; judged against ``pending`` alone the label would
        drift to whichever ID remained untagged.
        """
        if plan is None:
            return None, None
        milestones = tuple(getattr(plan, "milestones", ()))
        pending_ids = tuple(
            str(item.get("milestone_id", "")).strip()
            for item in pending
            if str(item.get("milestone_id", "")).strip()
        )
        universe = tuple(dict.fromkeys((*(official_ids or ()), *pending_ids)))
        owners = {
            milestone.canonical_id: RepositoryStream._milestone_official_matches(
                milestone,
                universe,
                getattr(plan, "native_plan", None),
            )
            for milestone in milestones
        }
        for released in pending:
            official_id = str(released.get("milestone_id", "")).strip()
            if not official_id:
                continue
            for milestone in milestones:
                status = str(getattr(milestone, "status", "")).upper()
                if status in {"COMPLETED", "COMPLETED_VERIFIED", "VERIFIED", "TERMINAL"}:
                    continue
                if owners[milestone.canonical_id] == (official_id,):
                    return milestone, released
        return None, None

    # ------------------------------------------------------------ parked IDs
    @staticmethod
    def _ensure_parked_table(conn) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS v2_repository_parked_ids ("
            "run_id TEXT NOT NULL, official_id TEXT NOT NULL, reason TEXT NOT NULL, "
            "source_event_id TEXT NOT NULL, parked_at TEXT NOT NULL, "
            "PRIMARY KEY(run_id, official_id))"
        )

    def parked_official_ids(self, execution) -> tuple[str, ...]:
        """Released official IDs the route left for later after a stall."""

        database = getattr(execution.registry, "database", None)
        if database is None:
            return ()
        with database.transaction() as conn:
            self._ensure_parked_table(conn)
            rows = conn.execute(
                "SELECT official_id FROM v2_repository_parked_ids WHERE run_id=? ORDER BY official_id",
                (execution.request.run_id,),
            ).fetchall()
        return tuple(str(r[0]) for r in rows)

    def park_official_ids(self, execution, official_ids, *, reason: str, source_event_id: str) -> tuple[str, ...]:
        """Park official IDs so the route prefers their released siblings.

        Parking is navigation only.  The internal node keeps its recorded
        status and history; the ID is still released and still untagged, so
        the official queue and submit gate are unchanged.
        """

        ids = tuple(dict.fromkeys(str(x).strip() for x in official_ids if str(x).strip()))
        ids = tuple(x for x in ids if x in self.selected)
        if not ids:
            return ()
        with execution.registry.database.transaction() as conn:
            self._ensure_parked_table(conn)
            conn.executemany(
                "INSERT OR REPLACE INTO v2_repository_parked_ids VALUES(?,?,?,?,?)",
                [
                    (execution.request.run_id, x, str(reason), str(source_event_id), utc_now())
                    for x in ids
                ],
            )
        return ids

    def unpark_official_ids(self, execution, official_ids=None) -> tuple[str, ...]:
        with execution.registry.database.transaction() as conn:
            self._ensure_parked_table(conn)
            if official_ids is None:
                rows = conn.execute(
                    "SELECT official_id FROM v2_repository_parked_ids WHERE run_id=?",
                    (execution.request.run_id,),
                ).fetchall()
                conn.execute(
                    "DELETE FROM v2_repository_parked_ids WHERE run_id=?",
                    (execution.request.run_id,),
                )
                return tuple(str(r[0]) for r in rows)
            ids = tuple(dict.fromkeys(str(x) for x in official_ids))
            conn.executemany(
                "DELETE FROM v2_repository_parked_ids WHERE run_id=? AND official_id=?",
                [(execution.request.run_id, x) for x in ids],
            )
        return ids

    def _eligible_pending(self, execution, pending, submitted, *, source_event_id):
        """Drop parked releases; when nothing else is left, the parked ones return."""

        parked = set(self.parked_official_ids(execution))
        if not parked:
            return pending
        # A submitted ID no longer needs parking.
        stale = sorted(parked & set(submitted))
        if stale:
            self.unpark_official_ids(execution, stale)
            parked -= set(stale)
        eligible = [x for x in pending if x["milestone_id"] not in parked]
        if pending and not eligible:
            released = self.unpark_official_ids(execution)
            execution.trace.record(
                "REPOSITORY_PARKED_IDS_RELEASED",
                source_event_id=source_event_id,
                official_ids=list(released),
                reason="NO_OTHER_RELEASED_UNSUBMITTED_ID",
            )
            return pending
        return eligible

    def refresh_route(self, execution, *, continue_turn: bool):
        state, event_id = self.observe(execution)
        pending = [x for x in state["available"] if x["milestone_id"] not in state["submitted"]]
        pending = self._eligible_pending(
            execution, pending, state["submitted"], source_event_id=event_id
        )
        current = execution.registry.current(execution.request.run_id)
        statuses = execution.registry.milestone_statuses(execution.request.run_id)
        plan = execution.registry.active_plan(execution.request.run_id)
        planned_target, released = self._planned_release_target(plan, pending, self.selected)
        # A real regression-guard failure on a tagged tree pins the route: the
        # official evaluator scores that tree as it stands, so the model must
        # repair it and move the same tag before any sibling ID is entered.
        # The official runner may already have unlocked successors (tag
        # existence, not correctness, unlocks the DAG); the runtime does not
        # follow it while the hold is active.
        held_tags = tuple(getattr(execution, "official_route_held_by_guard", lambda: ())())
        # Released IDs that no live native node uniquely owns need their own
        # node now, not only after a *new* release or full internal acceptance:
        # a first plan that bundles two released IDs into one node (nushell
        # G04+M02 in an earlier run) otherwise never gets a second official anchor.
        unowned = self._unowned_releases(plan, pending, self.selected)
        needs_split = bool(unowned) and plan is not None
        # ``observe`` is intentionally idempotent and returns no event when
        # the public queue has not changed. That must not make the queue
        # unable to repair a stale internal cursor or split unowned releases.
        # The observation Event for this exact public state is already
        # durable (it was appended when the state first appeared), so it is
        # the provenance for navigation without appending another stream Page.
        if event_id is None:
            realign = planned_target is not None and planned_target.canonical_id != current.canonical_id
            if not realign and not needs_split:
                return False
            event_id = "stream_" + digest({"run": execution.request.run_id, "state": state})
            page_store = getattr(execution, "page_store", None)
            if page_store is not None and not page_store.has_durable_event(event_id):
                return False
        if held_tags and (
            (planned_target is not None and planned_target.canonical_id != current.canonical_id)
            or needs_split
        ):
            application = None
            if needs_split:
                application = execution.registry.append_released_work_nodes(
                    run_id=execution.request.run_id, revision_id=execution.revision_id,
                    source_event_id=event_id,
                    releases=self._release_nodes(unowned),
                    switch_current=False,
                )
                if application is not None:
                    execution.plan_version_id = application.plan_version_id
                    execution._active_plan = execution.registry.active_plan(execution.request.run_id)
            execution.trace.record(
                "REPOSITORY_ROUTE_HELD_BY_GUARD",
                source_event_id=event_id,
                held_tags=list(held_tags),
                canonical_id=current.canonical_id,
                released_official_ids=[x["milestone_id"] for x in pending],
                nodes_appended=application is not None,
                cursor_moved=False,
            )
            return False
        # The official queue is the release authority for one continuous
        # repository task. If the next released official milestone is already
        # represented by a later native-plan node, realign the TPG cursor even
        # when an earlier internal review receipt is still incomplete. The
        # earlier node remains IN_PROGRESS/CLAIMED history; this is navigation,
        # not an official pass.
        if (
            planned_target is not None
            and planned_target.canonical_id != current.canonical_id
        ):
            if needs_split:
                # Sibling releases without an owner get their nodes now; the
                # cursor still follows the queue's first release below.
                application = execution.registry.append_released_work_nodes(
                    run_id=execution.request.run_id, revision_id=execution.revision_id,
                    source_event_id=event_id, releases=self._release_nodes(unowned),
                    switch_current=False,
                )
                if application is not None:
                    execution.plan_version_id = application.plan_version_id
                    execution._active_plan = execution.registry.active_plan(execution.request.run_id)
                    current = execution.registry.current(execution.request.run_id)
            predecessor = current
            current = execution.registry.switch_to_released_work(
                run_id=execution.request.run_id,
                canonical_id=planned_target.canonical_id,
                revision_id=execution.revision_id,
                source_event_id=event_id,
                plan_version_id=current.plan_version_id,
            )
            execution.context.switch_scope(
                current_milestone_id=current.identity_id,
                revision_id=execution.revision_id,
                current_milestone_artifact=execution._milestone_artifact(current),
            )
            if continue_turn and execution.context_transport is not None:
                execution.context_transport.request_task_continuation(
                    "Continue the SAME repository Task at the newly released official "
                    f"milestone {released['milestone_id']}. Preserve earlier incomplete "
                    "review risks and use existing Pages/MemoryRefs; do not restart "
                    "the repository or treat the release as already verified."
                )
            execution.trace.record(
                "REPOSITORY_ROUTE_REALIGNED_TO_RELEASED_MILESTONE",
                source_event_id=event_id,
                official_milestone_id=released["milestone_id"],
                canonical_id=current.canonical_id,
                predecessor_canonical_id=predecessor.canonical_id,
                predecessor_status=predecessor.status,
                same_task=True,
                prior_review_preserved=True,
                official_verification="NOT_INFERRED",
            )
            return True
        # The official queue, NOT completion of an internal workflow node,
        # determines when additional work is released. Keep earlier incomplete
        # nodes and failed verification as history/risks; never mark them passed.
        # ``needs_split`` also covers the first plan: released IDs without a
        # uniquely owning live node get one each immediately.
        if needs_split or (pending and planned_target is None and (
            self.new_release or all(s == "COMPLETED_VERIFIED" for s in statuses.values())
        )):
            # Advertise real previous execution addresses, not just a fresh SRS
            # synopsis. They remain directly resolvable after context eviction.
            # Temporal association is NOT verification of an official milestone.
            page_ids = set(execution._latest_milestone_pages(current.identity_id, limit=4))
            memory = [
                {"memory_ref": memory_ref_for_page(p.page_id, p.payload_digest),
                 "page_id": p.page_id, "entity_preview": list(p.entity_refs[:12]),
                 "entity_count": len(p.entity_refs), "revision_ids": list(p.revision_ids)}
                for p in execution.page_store.list_manifests() if p.page_id in page_ids]
            handoff = {"prior_route": current.canonical_id,
                       "prior_status": current.status, "recent_execution_memory": memory,
                       "memory_relation": "OBSERVED_AT_RELEASE_BOUNDARY_NOT_PROOF",
                       "public_predecessors": [d for x in pending for d in x.get("predecessors", ())]}
            # One internal node per released official ID.  A single bundled
            # node ("Continue released repository requirements: A, B, C, D")
            # has no unique owner, so _planned_release_target() returns None,
            # the submit gate loses its tag anchor and the route drifts.  The
            # queue's first release becomes current unless a live native node
            # already owns it; siblings stay PENDING and are reached by
            # queue-driven realignment after their predecessors tag.
            releases = unowned if unowned else pending
            switch_current = planned_target is None
            application = execution.registry.append_released_work_nodes(
                run_id=execution.request.run_id, revision_id=execution.revision_id,
                source_event_id=event_id,
                releases=self._release_nodes(releases),
                navigation_text="Navigation memory (revalidate changed code; no extra approval):\n"
                    + json.dumps(handoff, ensure_ascii=False),
                switch_current=switch_current,
            )
            if application is None:
                return
            execution.plan_version_id = application.plan_version_id
            execution._active_plan = execution.registry.active_plan(execution.request.run_id)
            predecessor = current
            current = execution.registry.current(execution.request.run_id)
            work_now = pending[0]["milestone_id"]
            if switch_current:
                if predecessor.status == "COMPLETED_VERIFIED":
                    execution._install_transition_handoff(predecessor, current)
                execution.context.switch_scope(current_milestone_id=current.identity_id,
                    revision_id=execution.revision_id, current_milestone_artifact=execution._milestone_artifact(current))
            if continue_turn and execution.context_transport is not None:
                execution.context_transport.request_task_continuation(
                    "Continue the SAME repository Task using the current TPG and previous Pages. "
                    f"Released official IDs {[x['milestone_id'] for x in releases]} each own "
                    f"one plan node. Work ONLY {work_now} now; sibling released "
                    "IDs wait for their own node. After its public tests pass, create "
                    f"git tag agent-impl-{work_now} — an internal review is "
                    "not an official submission. Preserve earlier implementations.\n"
                    + json.dumps({"released": pending, "handoff": handoff}, ensure_ascii=False))
            execution.trace.record("REPOSITORY_ROUTE_EXTENDED", source_event_id=event_id,
                                   canonical_id=current.canonical_id, same_task=True, same_page_store=True,
                                   released_official_ids=[x["milestone_id"] for x in releases],
                                   one_node_per_official_id=True,
                                   cursor_moved=switch_current,
                                   predecessor_status_preserved=predecessor.status,
                                   predecessor_memory_refs=[x["memory_ref"] for x in memory],
                                   addresses_used="NOT_INFERRED_FROM_ADVERTISEMENT")
            return True
        return False

    @staticmethod
    def _release_nodes(releases):
        return [
            {"official_id": x["milestone_id"],
             "title": f"Implement official milestone {x['milestone_id']}",
             "description": f"{x['milestone_id']}: {x['srs_path']}"}
            for x in releases
        ]

    @classmethod
    def _unowned_releases(cls, plan, pending, official_ids=None):
        """Released official IDs that no live native-plan node uniquely owns."""

        if plan is None or not pending:
            return []
        pending_ids = tuple(
            str(item.get("milestone_id", "")).strip()
            for item in pending
            if str(item.get("milestone_id", "")).strip()
        )
        universe = tuple(dict.fromkeys((*(official_ids or ()), *pending_ids)))
        owned: set[str] = set()
        for milestone in getattr(plan, "milestones", ()):
            status = str(getattr(milestone, "status", "")).upper()
            if status in {"COMPLETED", "COMPLETED_VERIFIED", "VERIFIED", "TERMINAL"}:
                continue
            matches = cls._milestone_official_matches(
                milestone, universe, getattr(plan, "native_plan", None)
            )
            if len(matches) == 1:
                owned.add(matches[0])
        return [item for item in pending if item["milestone_id"] not in owned]

    def publish(self, result):
        state = self.snapshot()
        slice_state = self._slice_state
        if state["all_submitted"]:
            continuity_state = "SUBMITTED_AWAITING_SCORE"
        elif slice_state is not None and slice_state.get("resumable") is True:
            continuity_state = "SLICE_SUSPENDED_RESUMABLE"
        elif slice_state is not None:
            continuity_state = "STREAM_STALLED"
        else:
            continuity_state = "STREAM_SUSPENDED"
        receipt = {"schema": "homy/repository-continuity-receipt@1", "project_id": self.project,
                   "run_id": result.run_id, "revision_id": result.revision_id,
                   "thread_id": result.thread_id, "epoch_id": result.epoch_id,
                   "page_ids": list(result.page_ids), "runtime_result": result.result_path,
                   "selected_count": len(self.selected), "submitted_revisions": state["submitted"],
                   "all_submitted": state["all_submitted"], "official_pass": None,
                   "state": continuity_state, "slice": slice_state}
        atomic_write_once(self.invocation / "continuity-receipt.json",
                          json.dumps(receipt, ensure_ascii=False, sort_keys=True).encode())


def repository_resume_decision(run_root: str | Path) -> str:
    """Return the fail-closed outer-loop action for a repository invocation."""

    root = Path(run_root)
    candidates = list(root.glob("invocations/*/continuity-receipt.json"))
    if not candidates:
        return "STOP"
    latest = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    try:
        receipt = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "STOP"
    if not isinstance(receipt, dict):
        return "STOP"
    if receipt.get("all_submitted") is True:
        return "COMPLETE"
    slice_state = receipt.get("slice")
    if (
        receipt.get("state") == "SLICE_SUSPENDED_RESUMABLE"
        and isinstance(slice_state, dict)
        and slice_state.get("resumable") is True
    ):
        # sklearn in an earlier run: every slice completed one Turn, tagged nothing
        # and suspended again on the same stall.  Two such slices in a row
        # hand control back to the official runner instead of burning the
        # remaining slices on the same loop.
        if _consecutive_idle_slices(candidates) >= 2:
            return "STOP"
        return "RESUME"
    return "STOP"


def _consecutive_idle_slices(candidates: list[Path], *, max_turns: int = 2) -> int:
    """Count, from the latest receipt backwards, slices that tagged nothing in <= ``max_turns`` Turns."""

    ordered = sorted(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)
    receipts: list[dict] = []
    for path in ordered[:4]:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            break
        if not isinstance(loaded, dict):
            break
        receipts.append(loaded)
    idle = 0
    for index, receipt in enumerate(receipts):
        if index + 1 >= len(receipts):
            break
        older = receipts[index + 1]
        slice_state = receipt.get("slice") if isinstance(receipt.get("slice"), dict) else {}
        turns = int(slice_state.get("completed_execution_turns", 0) or 0)
        same_tags = receipt.get("submitted_revisions") == older.get("submitted_revisions")
        if turns <= max_turns and same_tags:
            idle += 1
            continue
        break
    return idle
