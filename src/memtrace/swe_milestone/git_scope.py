"""Repository scope for one SWE-Milestone submission.

The scope is derived only from Git facts the agent already produced: the
official submission tags (``agent-impl-<milestone>``), the recorded run
baseline revision, and the working tree.  No hidden benchmark data is read.
"""

from __future__ import annotations

import fnmatch
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .contract import ROOT_BUILD_FILES, SweMilestoneContract

_SHA1 = re.compile(r"^[0-9a-f]{40}$")


class GitScopeError(RuntimeError):
    """Raised when the repository cannot provide a trustworthy scope."""


def _git(repository: Path, *arguments: str, check: bool = True, timeout: int = 120) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        raise GitScopeError(
            f"git {' '.join(arguments)} failed ({completed.returncode}): "
            f"{completed.stderr.strip()[:400]}"
        )
    return completed.stdout


@dataclass(frozen=True, slots=True)
class SubmissionScope:
    baseline_revision: str
    target_revision: str
    target_tag: str | None
    working_tree_dirty: bool
    changed_paths: tuple[str, ...]
    in_scope_source: tuple[str, ...]
    root_manifests: tuple[str, ...]
    test_only: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    baseline_source: str
    prior_tags: tuple[str, ...] = field(default=())

    @property
    def submitted_paths(self) -> tuple[str, ...]:
        """Paths the official capture would copy: in-scope source plus root manifests.

        Test files under the source directories are excluded on purpose: the
        official protocol states that ordinary test files are not submitted.
        """
        return tuple((*self.in_scope_source, *self.root_manifests))


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    posix = PurePosixPath(path)
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern) or posix.match(pattern):
            return True
        # ``**/src/test/**`` style globs must also match a path whose first
        # segment is the wildcard directory itself.
        if pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:]):
            return True
    return False


def _under_dirs(path: str, dirs: tuple[str, ...]) -> bool:
    return any(path == item or path.startswith(item.rstrip("/") + "/") for item in dirs)


def classify_paths(
    paths: tuple[str, ...],
    contract: SweMilestoneContract,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Split changed paths into submitted source, root manifests, tests, out-of-scope."""

    source: list[str] = []
    manifests: list[str] = []
    tests: list[str] = []
    outside: list[str] = []
    for path in paths:
        if _matches_any(path, contract.exclude_patterns):
            continue
        if path in ROOT_BUILD_FILES:
            manifests.append(path)
            continue
        is_test = _matches_any(path, contract.test_dirs)
        if _under_dirs(path, contract.repo_src_dirs):
            if is_test:
                tests.append(path)
            else:
                source.append(path)
            continue
        if is_test:
            tests.append(path)
            continue
        outside.append(path)
    return tuple(source), tuple(manifests), tuple(tests), tuple(outside)


def submission_tags(repository: Path, prefix: str) -> dict[str, str]:
    """Return ``tag -> commit`` for every official submission tag."""

    output = _git(
        repository,
        "for-each-ref",
        "--format=%(refname:short) %(*objectname) %(objectname)",
        f"refs/tags/{prefix}*",
    )
    tags: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        # Annotated tags expose the peeled commit in the second column.
        commit = parts[1] if len(parts) > 1 and _SHA1.match(parts[1]) else parts[-1]
        if _SHA1.match(commit):
            tags[name] = commit
    return tags


def _is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    return completed.returncode == 0


def _distance(repository: Path, ancestor: str, descendant: str) -> int:
    return int(_git(repository, "rev-list", "--count", f"{ancestor}..{descendant}").strip() or 0)


def recorded_baseline(repository: Path, contract: SweMilestoneContract) -> str | None:
    path = repository / contract.baseline_revision_file
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if not _SHA1.match(value):
        return None
    try:
        return _git(repository, "rev-parse", f"{value}^{{commit}}").strip()
    except GitScopeError:
        return None


def resolve_scope(
    repository: Path,
    contract: SweMilestoneContract,
    *,
    target_tag: str | None = None,
    fallback_baseline: str | None = None,
    baseline_override: str | None = None,
) -> SubmissionScope:
    """Derive the baseline/target pair and the changed path partition.

    ``target_tag`` verifies a submission the agent already tagged: the target
    is that tag's commit and the baseline is the closest earlier submission
    tag on its history.  Without a tag, the target is the working tree and the
    baseline is the latest submission tag reachable from ``HEAD``.

    ``baseline_override`` pins the baseline to one commit (the run's initial
    HEAD): the official PASS_TO_PASS set is "passed at the start of the
    repository task", so regressions must be judged against that tree, not
    against the previous submission where an earlier milestone may already
    have broken the unit.  The changed-path partition then covers the whole
    accumulated diff, exactly the tree the official evaluator receives.
    """

    head = _git(repository, "rev-parse", "HEAD").strip()
    tags = submission_tags(repository, contract.submission_tag_prefix)
    if target_tag is not None:
        if target_tag not in tags:
            raise GitScopeError(f"submission tag {target_tag} does not exist")
        target = tags[target_tag]
    else:
        target = head
    candidates: list[tuple[int, str, str]] = []
    for name, commit in tags.items():
        # A tagged submission is compared with the previous submission, never
        # with a sibling tag on the same commit.  The working tree, however,
        # is compared with the latest submission even when HEAD is that tag.
        if target_tag is not None and (name == target_tag or commit == target):
            continue
        if _is_ancestor(repository, commit, target):
            candidates.append((_distance(repository, commit, target), name, commit))
    baseline_source = "previous_submission_tag"
    if baseline_override and _SHA1.match(baseline_override) and _is_ancestor(
        repository, baseline_override, target
    ):
        baseline = baseline_override
        baseline_source = "run_initial_head"
        candidates.sort()
        prior = tuple(name for _, name, _ in candidates)
    elif candidates:
        candidates.sort()
        _, _, baseline = candidates[0]
        prior = tuple(name for _, name, _ in candidates)
    else:
        prior = ()
        recorded = recorded_baseline(repository, contract)
        if recorded and _is_ancestor(repository, recorded, target):
            baseline = recorded
            baseline_source = "recorded_run_baseline"
        elif fallback_baseline and _is_ancestor(repository, fallback_baseline, target):
            baseline = fallback_baseline
            baseline_source = "verifier_initial_head"
        else:
            root = _git(repository, "rev-list", "--max-parents=0", target).split()
            if not root:
                raise GitScopeError("repository has no root commit")
            baseline = root[-1]
            baseline_source = "repository_root"
    status = _git(repository, "status", "--porcelain", "--untracked-files=all")
    dirty = bool(status.strip())
    if target_tag is None:
        # Working tree against baseline: committed and uncommitted edits.
        diff = _git(repository, "diff", "--name-only", "--diff-filter=ACMRD", baseline)
        untracked = _git(repository, "ls-files", "--others", "--exclude-standard")
        changed = tuple(dict.fromkeys((*diff.split(), *untracked.split())))
    else:
        diff = _git(repository, "diff", "--name-only", "--diff-filter=ACMRD", baseline, target)
        changed = tuple(dict.fromkeys(diff.split()))
    source, manifests, tests, outside = classify_paths(changed, contract)
    return SubmissionScope(
        baseline_revision=baseline,
        target_revision=target,
        target_tag=target_tag,
        working_tree_dirty=dirty,
        changed_paths=changed,
        in_scope_source=source,
        root_manifests=manifests,
        test_only=tests,
        out_of_scope=outside,
        baseline_source=baseline_source,
        prior_tags=prior,
    )


def overlay_submitted_paths(
    repository: Path,
    worktree: Path,
    scope: SubmissionScope,
) -> tuple[str, ...]:
    """Copy exactly the paths the official evaluator would receive onto ``worktree``.

    The evaluator applies the agent's source directories and root manifests on
    top of its own tree and keeps its own tests.  Reproducing that view here
    means locally edited or added test files never influence the verdict, and
    behaviour that depends on an out-of-scope file (``package.json``, tooling
    config) is missing exactly as it will be missing in the official run.
    """

    applied: list[str] = []
    for relative in scope.submitted_paths:
        destination = worktree / relative
        if scope.target_tag is None:
            source = repository / relative
            if source.is_file():
                data = source.read_bytes()
                executable = bool(source.stat().st_mode & 0o111)
                _write_if_changed(destination, data, executable)
                applied.append(relative)
            elif destination.exists():
                destination.unlink()
                applied.append(relative)
            continue
        completed = subprocess.run(
            ["git", "-C", str(repository), "show", f"{scope.target_revision}:{relative}"],
            capture_output=True,
            check=False,
            timeout=120,
        )
        if completed.returncode == 0:
            mode = subprocess.run(
                ["git", "-C", str(repository), "ls-tree", scope.target_revision, "--", relative],
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            ).stdout.split()
            _write_if_changed(destination, completed.stdout, bool(mode and mode[0] == "100755"))
            applied.append(relative)
        elif destination.exists():
            destination.unlink()
            applied.append(relative)
    return tuple(applied)


def _write_if_changed(destination: Path, data: bytes, executable: bool) -> bool:
    """Write ``data`` unless the file already holds it byte for byte.

    The verify worktree persists between attempts so that mtime-keyed build
    caches (cargo fingerprints) stay warm; rewriting an unchanged submitted
    file would mark its whole crate stale on every attempt.
    """

    if destination.is_file() and not destination.is_symlink():
        try:
            same_mode = bool(destination.stat().st_mode & 0o111) == executable
            if same_mode and destination.read_bytes() == data:
                return False
        except OSError:
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.is_dir():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.write_bytes(data)
    destination.chmod(0o755 if executable else 0o644)
    return True


class BaselineWorktree:
    """A detached checkout of the baseline revision that shares dependency caches.

    The checkout persists between verifier attempts.  Build caches keyed by
    file mtimes (cargo fingerprints) only stay warm when unchanged files keep
    their mtimes, so :meth:`ensure` resets a reusable checkout path by path
    instead of recreating it: files outside ``keep_paths`` return to the
    baseline content, files inside ``keep_paths`` are left for the overlay to
    reconcile byte for byte.
    """

    def __init__(self, repository: Path, revision: str, root: Path, *, label: str = "baseline") -> None:
        self.repository = repository
        self.revision = revision
        self.label = label
        self.path = root / f"{label}-{revision[:12]}"

    def ensure(self, shared_links: tuple[str, ...], *, keep_paths: tuple[str, ...] = ()) -> Path:
        if self._reusable():
            self._reset(keep_paths)
        else:
            self.remove()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _git(
                self.repository,
                "worktree",
                "add",
                "--detach",
                "--force",
                str(self.path),
                self.revision,
                timeout=600,
            )
        for relative in shared_links:
            source = self.repository / relative
            target = self.path / relative
            if source.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source, target_is_directory=source.is_dir())
        return self.path

    def _reusable(self) -> bool:
        if not (self.path / ".git").exists():
            return False
        try:
            head = _git(self.path, "rev-parse", "--verify", "HEAD^{commit}", timeout=60).strip()
            wanted = _git(
                self.repository, "rev-parse", "--verify", f"{self.revision}^{{commit}}", timeout=60
            ).strip()
        except GitScopeError:
            return False
        return bool(_SHA1.match(head)) and head == wanted

    def _reset(self, keep_paths: tuple[str, ...]) -> None:
        keep = set(keep_paths)
        listing = _git(
            self.path,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--no-renames",
            timeout=300,
        )
        restore: list[str] = []
        for entry in listing.split("\0"):
            if len(entry) < 4:
                continue
            code, relative = entry[:2], entry[3:]
            if relative in keep:
                continue
            if code == "??":
                stray = self.path / relative
                try:
                    if stray.is_dir() and not stray.is_symlink():
                        shutil.rmtree(stray)
                    else:
                        stray.unlink()
                except FileNotFoundError:
                    pass
                continue
            restore.append(relative)
        for start in range(0, len(restore), 200):
            _git(
                self.path,
                "checkout",
                "--force",
                "--",
                *restore[start : start + 200],
                timeout=300,
            )

    def remove(self) -> None:
        subprocess.run(
            ["git", "-C", str(self.repository), "worktree", "remove", "--force", str(self.path)],
            capture_output=True,
            check=False,
            timeout=300,
        )
        shutil.rmtree(self.path, ignore_errors=True)
        subprocess.run(
            ["git", "-C", str(self.repository), "worktree", "prune"],
            capture_output=True,
            check=False,
            timeout=120,
        )
