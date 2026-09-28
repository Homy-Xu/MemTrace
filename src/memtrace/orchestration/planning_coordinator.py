from __future__ import annotations

import hashlib
import os
import stat as stat_module
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from ..contracts import PlanSpec, digest
from ..observability import MetricRecorder


class PlanProvider(Protocol):
    def generate(self, *, user_task: str, revision_id: str) -> PlanSpec: ...


@dataclass(frozen=True, slots=True)
class SuppliedPlanProvider:
    """Formal offline adapter for an already-produced provider Plan response."""

    plan: PlanSpec

    def generate(self, *, user_task: str, revision_id: str) -> PlanSpec:
        del user_task, revision_id
        return self.plan


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshotReceipt:
    """One shared, content-addressed workspace credential for a run boundary."""

    revision_id: str
    manifest_digest: str
    manifest: Mapping[str, tuple[int, int, int, int, str]]
    vcs_fingerprint: str | None
    exclusion_policy: tuple[str, ...]


class WorkspaceReadOnlyGuard:
    """Detect repository mutations made while a Planning provider is running.

    The fingerprint hashes file bytes without parsing program structure. This
    catches same-size rewrites even on coarse-timestamp filesystems and is not
    a hidden Rich Graph build.
    """

    DEFAULT_EXCLUDED_NAMES = frozenset(
        {
            ".git",
            ".hg",
            ".svn",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            ".tox",
            ".nox",
            # DeepSWE Rust tasks intentionally direct Cargo output here.  It is
            # an execution cache, never repository evidence or a model patch.
            "target-sandbox",
        }
    )

    def __init__(
        self,
        root: Path,
        excluded_roots: tuple[Path, ...] = (),
        *,
        excluded_names: tuple[str, ...] = (),
    ) -> None:
        self.root = Path(root).resolve()
        self.excluded = tuple(item.resolve() for item in excluded_roots)
        self.excluded_names = frozenset(
            (
                *self.DEFAULT_EXCLUDED_NAMES,
                *(item.strip() for item in excluded_names if item.strip()),
            )
        )
        self._baseline_vcs_fingerprint: str | None = None

    def _excluded(self, path: Path) -> bool:
        try:
            relative_parts = path.relative_to(self.root).parts
        except ValueError:
            relative_parts = (path.name,)
        if any(part in self.excluded_names for part in relative_parts):
            return True
        return any(path == item or item in path.parents for item in self.excluded)

    def git_pathspecs(self) -> tuple[str, ...]:
        """Return the same repository scope for Git and filesystem receipts."""

        pathspecs: list[str] = ["."]
        for name in sorted(self.excluded_names):
            pathspecs.extend(
                (
                    f":(exclude,glob){name}/**",
                    f":(exclude,glob)**/{name}/**",
                )
            )
        for excluded in self.excluded:
            try:
                relative = excluded.relative_to(self.root).as_posix()
            except ValueError:
                continue
            pathspecs.extend(
                (
                    f":(exclude,glob){relative}",
                    f":(exclude,glob){relative}/**",
                )
            )
        return tuple(dict.fromkeys(pathspecs))

    @staticmethod
    def _content_digest(path: Path) -> str:
        if path.is_symlink():
            return digest({"symlink": os.readlink(path)})
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
        return "sha256:" + hasher.hexdigest()

    def _vcs_fingerprint(self) -> str | None:
        """Hash Git identity plus its scoped index/worktree view.

        Planning is read-only with respect to repository control state as well
        as file bytes.  A branch checkout can leave the worktree byte-identical
        while changing the execution base, so HEAD and its symbolic ref are
        part of the shared workspace credential.
        """

        repository = subprocess.run(
            ("git", "rev-parse", "--is-inside-work-tree"),
            cwd=self.root,
            check=False,
            capture_output=True,
            timeout=30,
        )
        if repository.returncode != 0 or repository.stdout.strip() != b"true":
            return None
        scope = self.git_pathspecs()
        commands = (
            ("status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *scope),
            ("diff", "--binary", "--no-ext-diff", "--", *scope),
            ("diff", "--cached", "--binary", "--no-ext-diff", "--", *scope),
        )
        hasher = hashlib.sha256()
        status_output = b""
        head = subprocess.run(
            ("git", "rev-parse", "--verify", "HEAD"),
            cwd=self.root,
            check=False,
            capture_output=True,
            timeout=30,
        )
        symbolic = subprocess.run(
            ("git", "symbolic-ref", "--quiet", "HEAD"),
            cwd=self.root,
            check=False,
            capture_output=True,
            timeout=30,
        )
        if symbolic.returncode not in {0, 1}:
            return None
        # An initialized repository may not have its first commit yet.  Its
        # symbolic branch plus index/worktree are still a complete Planning
        # fence, so represent the unborn HEAD explicitly instead of falling
        # back to a second full file-body scan.
        if head.returncode != 0 and symbolic.returncode != 0:
            return None
        for label, completed in ((b"HEAD", head), (b"SYMBOLIC_HEAD", symbolic)):
            hasher.update(label)
            hasher.update(b"\0")
            hasher.update(str(completed.returncode).encode())
            hasher.update(b"\0")
            hasher.update(completed.stdout)
        for arguments in commands:
            completed = subprocess.run(
                ("git", *arguments),
                cwd=self.root,
                check=False,
                capture_output=True,
                timeout=30,
            )
            if completed.returncode != 0:
                return None
            hasher.update(b"\0".join(part.encode() for part in arguments))
            hasher.update(b"\0")
            hasher.update(completed.stdout)
            if arguments[0] == "status":
                status_output = completed.stdout
        # Git's porcelain state does not encode untracked file bodies. Bind
        # them explicitly so an untracked source rewrite cannot preserve the
        # same VCS fingerprint on a coarse-timestamp filesystem.
        for record in status_output.split(b"\0"):
            if not record.startswith(b"?? "):
                continue
            relative = record[3:].decode(errors="surrogateescape")
            candidate = self.root / relative
            if self._excluded(candidate) or not (candidate.is_file() or candidate.is_symlink()):
                continue
            hasher.update(relative.encode(errors="surrogateescape"))
            hasher.update(self._content_digest(candidate).encode())
        return "sha256:" + hasher.hexdigest()

    def snapshot(self) -> Mapping[str, tuple[int, int, int, int, str]]:
        records: dict[str, tuple[int, int, int, int, str]] = {}
        for current, directories, files in os.walk(self.root):
            current_path = Path(current)
            directories[:] = sorted(
                name for name in directories if not self._excluded(current_path / name)
            )
            for name in sorted(files):
                path = current_path / name
                if self._excluded(path):
                    continue
                try:
                    stat = path.lstat()
                except FileNotFoundError:
                    continue
                try:
                    content_digest = self._content_digest(path)
                except FileNotFoundError:
                    continue
                records[path.relative_to(self.root).as_posix()] = (
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                    stat.st_mode,
                    content_digest,
                )
        self._baseline_vcs_fingerprint = self._vcs_fingerprint()
        return records

    @staticmethod
    def _content_manifest_digest(
        manifest: Mapping[str, tuple[int, int, int, int, str]],
    ) -> str:
        # mtime/ctime are an incremental-read optimization, never part of the
        # semantic workspace identity. Preserve file kind, executable bits and
        # bytes/symlink target in the content revision.
        return digest(
            sorted(
                (
                    path,
                    stat_module.S_IFMT(metadata[3]),
                    metadata[3] & 0o111,
                    metadata[4],
                )
                for path, metadata in manifest.items()
            )
        )

    def capture(self) -> WorkspaceSnapshotReceipt:
        manifest = self.snapshot()
        manifest_digest = self._content_manifest_digest(manifest)
        return WorkspaceSnapshotReceipt(
            revision_id="revision_" + manifest_digest.removeprefix("sha256:"),
            manifest_digest=manifest_digest,
            manifest=manifest,
            vcs_fingerprint=self._baseline_vcs_fingerprint,
            exclusion_policy=tuple(sorted(self.excluded_names)),
        )

    def unchanged(
        self,
        before: Mapping[str, tuple[int, int, int, int, str]] | WorkspaceSnapshotReceipt,
    ) -> bool:
        """Verify the initial content manifest with a safe Git fast path.

        Clean Git files use scoped status/diff/index evidence plus the path and
        metadata fence. This avoids rereading every tracked body. Non-Git roots
        retain the byte-digest fallback because coarse timestamp filesystems can
        otherwise hide same-size rewrites.
        """

        receipt = before if isinstance(before, WorkspaceSnapshotReceipt) else None
        manifest = receipt.manifest if receipt is not None else before
        if receipt is not None:
            self._baseline_vcs_fingerprint = receipt.vcs_fingerprint
        if self._baseline_vcs_fingerprint is not None:
            current_vcs_fingerprint = self._vcs_fingerprint()
            if current_vcs_fingerprint is not None:
                return current_vcs_fingerprint == self._baseline_vcs_fingerprint
        current_paths: dict[str, Path] = {}
        for current, directories, files in os.walk(self.root):
            current_path = Path(current)
            directories[:] = sorted(
                name for name in directories if not self._excluded(current_path / name)
            )
            for name in sorted(files):
                path = current_path / name
                if not self._excluded(path):
                    current_paths[path.relative_to(self.root).as_posix()] = path
        if set(current_paths) != set(manifest):
            return False
        metadata_unchanged = True
        for relative, path in current_paths.items():
            try:
                stat = path.lstat()
            except FileNotFoundError:
                return False
            previous = manifest[relative]
            metadata = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_mode)
            if metadata != previous[:4]:
                metadata_unchanged = False
                break
        if not metadata_unchanged:
            return False
        for relative, path in current_paths.items():
            try:
                if self._content_digest(path) != manifest[relative][4]:
                    return False
            except FileNotFoundError:
                return False
        return True

    def fingerprint(self) -> str:
        return self._content_manifest_digest(self.snapshot())


class PlanningCoordinator:
    def __init__(self, metrics: MetricRecorder) -> None:
        self.metrics = metrics

    def generate_read_only(
        self,
        *,
        repository_path: Path,
        run_root: Path,
        user_task: str,
        revision_id: str,
        provider: PlanProvider,
        workspace_receipt: WorkspaceSnapshotReceipt | None = None,
    ) -> PlanSpec:
        guard = WorkspaceReadOnlyGuard(repository_path, (run_root,))
        before = workspace_receipt or guard.capture()
        if workspace_receipt is not None and before.revision_id != revision_id:
            raise RuntimeError("Workspace receipt does not match the Planning revision")
        plan = provider.generate(user_task=user_task, revision_id=revision_id)
        # ``PlanSpec`` is the canonical, immutable planning contract and has
        # already validated IDs, dependencies and acceptance criteria in its
        # constructor.  Re-serializing it here used to maintain a second,
        # incomplete schema which silently dropped newly-added contract fields
        # such as target_outcome and final_acceptance.  Arbitrary providers
        # must normalize once at their adapter boundary; this coordinator then
        # preserves that exact canonical value.
        if not isinstance(plan, PlanSpec):
            raise TypeError("Planning provider must return a canonical PlanSpec")
        if not guard.unchanged(before):
            raise RuntimeError("Planning provider modified the repository")
        return plan
