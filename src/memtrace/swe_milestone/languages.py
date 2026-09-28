"""Per-language affected-scope verification backends.

Each backend answers four questions for one submission scope:

* ``manifest_guard``: can the official offline evaluator even read/build the
  root manifests as submitted (``go.mod``/``go.sum`` consistency, ``Cargo.toml``
  syntax, POM completeness)?
* ``build_guard``: does the affected code still compile / type-check?
* ``affected_units``: which test units (files, packages, crates, modules) are
  reached by the changed source paths?
* ``run_units``: run those units and report one PASSED/FAILED outcome each.

Backends only run the repository's own tooling offline.  They never fetch
dependencies, never write outside the repository and never modify sources.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .contract import SweMilestoneContract
from .git_scope import SubmissionScope
from .maven_reactor import MavenReactor

_SECRET_ENV = re.compile(r"(API_KEY|_TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)
_OUTPUT_TAIL = 6000

#: A unit that could not run because the offline dependency cache lacks an
#: artifact.  It is neither a test failure nor a pass; the verifier decides
#: whether the submission or the environment is responsible by calibrating
#: against the untouched baseline (dubbo/junit-platform-commons in an earlier run).
ENVIRONMENT_OUTCOME = "ENVIRONMENT"

_OFFLINE_RESOLUTION = re.compile(
    r"(could not be resolved|Cannot access \S+ \([^)]*\) in offline mode|"
    r"has not been downloaded from it before|Could not resolve dependencies|"
    r"Could not transfer artifact|NonResolvableDependencyException)",
    re.IGNORECASE,
)


def is_offline_resolution_failure(text: str) -> bool:
    """Whether a build tool failed for lack of an artifact in the offline cache."""

    return bool(text) and _OFFLINE_RESOLUTION.search(text) is not None


@dataclass(frozen=True, slots=True)
class GuardFailure:
    kind: str
    message: str
    command: str = ""
    output: str = ""


@dataclass(frozen=True, slots=True)
class UnitOutcome:
    unit: str
    outcome: str
    returncode: int
    duration_ms: float
    output: str = ""


@dataclass(slots=True)
class CommandResult:
    argv: tuple[str, ...]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False

    @property
    def command(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)

    @property
    def tail(self) -> str:
        text = (self.stdout or "") + ("\n" + self.stderr if self.stderr else "")
        return text[-_OUTPUT_TAIL:]


def clean_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if _SECRET_ENV.search(key) is None}
    env.setdefault("CI", "1")
    env["HOMY_SWE_MILESTONE_VERIFIER"] = "1"
    if extra:
        env.update(extra)
    return env


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=dict(env) if env is not None else clean_environment(),
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            argv=tuple(argv),
            cwd=str(cwd),
            returncode=124,
            stdout=(exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or ""),
            stderr=f"timed out after {timeout:.0f}s",
            duration_ms=(time.monotonic() - started) * 1000.0,
            timed_out=True,
        )
    except OSError as exc:
        return CommandResult(
            argv=tuple(argv),
            cwd=str(cwd),
            returncode=127,
            stdout="",
            stderr=f"failed to start: {exc}",
            duration_ms=(time.monotonic() - started) * 1000.0,
        )
    return CommandResult(
        argv=tuple(argv),
        cwd=str(cwd),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        duration_ms=(time.monotonic() - started) * 1000.0,
    )


def _nearest_ancestor_with(path: str, root: Path, marker: str) -> str:
    """Return the repository-relative directory nearest to ``path`` containing ``marker``."""

    current = PurePosixPath(path).parent
    while True:
        candidate = root / current / marker if str(current) != "." else root / marker
        if candidate.is_file():
            return "." if str(current) == "." else str(current)
        if str(current) in {".", ""}:
            return "."
        current = current.parent


def _flags_from_build_command(build_command: str | None, names: Sequence[str]) -> list[str]:
    """Copy selected flags (with values) from the official build command."""

    if not build_command:
        return []
    try:
        tokens = shlex.split(build_command)
    except ValueError:
        return []
    picked: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if any(token == name or token.startswith(name + "=") for name in names):
            picked.append(token)
            if "=" not in token and index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                picked.append(tokens[index + 1])
                index += 1
        index += 1
    return picked


class LanguageBackend:
    name = "generic"
    #: repository-relative directories to share with the baseline worktree
    shared_links: tuple[str, ...] = ()

    def __init__(self, contract: SweMilestoneContract) -> None:
        self.contract = contract

    # --- guards -----------------------------------------------------------
    def manifest_guard(self, root: Path, scope: SubmissionScope, timeout: float) -> list[GuardFailure]:
        return []

    def build_guard(self, root: Path, scope: SubmissionScope, timeout: float) -> list[GuardFailure]:
        return []

    # --- tests ------------------------------------------------------------
    def affected_units(self, root: Path, scope: SubmissionScope, timeout: float) -> list[str]:
        return []

    def run_units(
        self,
        root: Path,
        units: Sequence[str],
        timeout: float,
    ) -> list[UnitOutcome]:
        return []

    def unit_exists(self, root: Path, unit: str) -> bool:
        return True


# ---------------------------------------------------------------------------
# TypeScript / JavaScript
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _NodePackage:
    root: str  # repository-relative package root ("." for repository root)
    runner: str  # jest | vitest | none
    manager: str  # yarn | pnpm | npm


class NodeBackend(LanguageBackend):
    name = "node"
    shared_links = ("node_modules",)

    def _package(self, root: Path, package_root: str) -> _NodePackage:
        base = root if package_root == "." else root / package_root
        runner = "none"
        try:
            manifest = json.loads((base / "package.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        deps = {
            **(manifest.get("dependencies") or {}),
            **(manifest.get("devDependencies") or {}),
        }
        scripts = manifest.get("scripts") or {}
        test_script = str(scripts.get("test", ""))
        if "vitest" in deps or "vitest" in test_script:
            runner = "vitest"
        elif "jest" in deps or "jest" in test_script or (base / "jest.config.ts").exists() or (
            base / "jest.config.js"
        ).exists() or (base / "jest.config.cjs").exists() or (base / "jest.config.mjs").exists():
            runner = "jest"
        manager = "yarn" if (base / "yarn.lock").exists() else "pnpm" if (
            base / "pnpm-lock.yaml"
        ).exists() else "npm"
        return _NodePackage(root=package_root, runner=runner, manager=manager)

    def _package_roots(self, root: Path, paths: Sequence[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for path in paths:
            if not re.search(r"\.(?:[cm]?[jt]sx?)$", path):
                continue
            package_root = _nearest_ancestor_with(path, root, "package.json")
            relative = path if package_root == "." else str(PurePosixPath(path).relative_to(package_root))
            grouped.setdefault(package_root, []).append(relative)
        return grouped

    def _bin(self, root: Path, package: _NodePackage, tool: str) -> list[str]:
        base = root if package.root == "." else root / package.root
        local = base / "node_modules" / ".bin" / tool
        if local.is_file():
            return [str(local)]
        if package.manager == "yarn":
            return ["yarn", "--silent", tool]
        if package.manager == "pnpm":
            return ["pnpm", "exec", tool]
        return ["npx", "--no-install", tool]

    @staticmethod
    def _unit(package_root: str, test_file: str) -> str:
        return test_file if package_root == "." else f"{package_root}::{test_file}"

    @staticmethod
    def _split_unit(unit: str) -> tuple[str, str]:
        if "::" in unit:
            package_root, test_file = unit.split("::", 1)
            return package_root, test_file
        return ".", unit

    def build_guard(self, root: Path, scope: SubmissionScope, timeout: float) -> list[GuardFailure]:
        failures: list[GuardFailure] = []
        for package_root in self._package_roots(root, scope.in_scope_source):
            base = root if package_root == "." else root / package_root
            if not (base / "tsconfig.json").exists():
                continue
            tsc = base / "node_modules" / ".bin" / "tsc"
            if not tsc.is_file():
                continue
            result = run_command([str(tsc), "--noEmit", "-p", str(base)], cwd=base, timeout=timeout)
            if result.timed_out:
                failures.append(
                    GuardFailure("typecheck_timeout", "tsc --noEmit did not finish", result.command, result.tail)
                )
            elif result.returncode != 0:
                failures.append(
                    GuardFailure(
                        "typecheck_failed",
                        "TypeScript type check failed; official jest suites importing these "
                        "modules would fail to load",
                        result.command,
                        result.tail,
                    )
                )
        return failures

    def affected_units(self, root: Path, scope: SubmissionScope, timeout: float) -> list[str]:
        units: list[str] = []
        for package_root, files in self._package_roots(root, scope.in_scope_source).items():
            package = self._package(root, package_root)
            base = root if package_root == "." else root / package_root
            if package.runner == "jest":
                result = run_command(
                    [*self._bin(root, package, "jest"), "--listTests", "--findRelatedTests", *files],
                    cwd=base,
                    timeout=timeout,
                    env=clean_environment(),
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        line = line.strip()
                        if not line or not line.startswith("/"):
                            continue
                        try:
                            relative = str(Path(line).resolve().relative_to(base.resolve()))
                        except ValueError:
                            continue
                        units.append(self._unit(package_root, relative))
            elif package.runner == "vitest":
                # vitest has no --listTests; run --related directly (units are
                # discovered from the JSON report at run time).
                units.append(self._unit(package_root, "__related__:" + ",".join(files)))
        return list(dict.fromkeys(units))

    def unit_exists(self, root: Path, unit: str) -> bool:
        package_root, test_file = self._split_unit(unit)
        if test_file.startswith("__related__:"):
            return True
        base = root if package_root == "." else root / package_root
        return (base / test_file).is_file()

    def run_units(self, root: Path, units: Sequence[str], timeout: float) -> list[UnitOutcome]:
        outcomes: list[UnitOutcome] = []
        by_package: dict[str, list[str]] = {}
        for unit in units:
            package_root, test_file = self._split_unit(unit)
            by_package.setdefault(package_root, []).append(test_file)
        for package_root, files in by_package.items():
            package = self._package(root, package_root)
            base = root if package_root == "." else root / package_root
            with tempfile.NamedTemporaryFile(prefix="homy-verifier-", suffix=".json", delete=False) as handle:
                report = Path(handle.name)
            try:
                if package.runner == "jest":
                    argv = [
                        *self._bin(root, package, "jest"),
                        "--ci",
                        "--json",
                        f"--outputFile={report}",
                        "--passWithNoTests",
                        *files,
                    ]
                elif package.runner == "vitest":
                    related = [
                        item
                        for spec in files
                        if spec.startswith("__related__:")
                        for item in spec.removeprefix("__related__:").split(",")
                        if item
                    ]
                    plain = [spec for spec in files if not spec.startswith("__related__:")]
                    argv = [
                        *self._bin(root, package, "vitest"),
                        "run",
                        "--reporter=json",
                        f"--outputFile={report}",
                        "--passWithNoTests",
                    ]
                    if related:
                        argv.extend(["--related", *related])
                    argv.extend(plain)
                else:
                    for test_file in files:
                        outcomes.append(
                            UnitOutcome(self._unit(package_root, test_file), "MISSING", 127, 0.0, "no test runner")
                        )
                    continue
                result = run_command(argv, cwd=base, timeout=timeout, env=clean_environment({"CI": "true"}))
                parsed = self._parse_report(report, base)
                if parsed is None:
                    outcome = "ERROR" if result.returncode != 0 else "PASSED"
                    for test_file in files:
                        outcomes.append(
                            UnitOutcome(
                                self._unit(package_root, test_file),
                                outcome if not test_file.startswith("__related__:") else outcome,
                                result.returncode,
                                result.duration_ms,
                                result.tail,
                            )
                        )
                    continue
                seen: set[str] = set()
                for test_file, status in parsed.items():
                    seen.add(test_file)
                    outcomes.append(
                        UnitOutcome(
                            self._unit(package_root, test_file),
                            status,
                            result.returncode,
                            result.duration_ms,
                            result.tail if status != "PASSED" else "",
                        )
                    )
                for test_file in files:
                    if test_file.startswith("__related__:") or test_file in seen:
                        continue
                    outcomes.append(
                        UnitOutcome(self._unit(package_root, test_file), "MISSING", result.returncode, result.duration_ms, result.tail)
                    )
            finally:
                report.unlink(missing_ok=True)
        return outcomes

    @staticmethod
    def _parse_report(report: Path, base: Path) -> dict[str, str] | None:
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        results = data.get("testResults")
        if not isinstance(results, list):
            return None
        parsed: dict[str, str] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("testFilePath") or "")
            if not name:
                continue
            try:
                relative = str(Path(name).resolve().relative_to(base.resolve()))
            except ValueError:
                relative = name
            status = str(item.get("status", "")).lower()
            if status in {"passed", "pass"}:
                outcome = "PASSED"
            elif status in {"failed", "fail"}:
                outcome = "FAILED"
            elif status in {"skipped", "pending", "todo"}:
                outcome = "SKIPPED"
            else:
                outcome = "ERROR"
            # A suite that failed to load reports zero assertions with a message.
            if outcome == "PASSED" and item.get("failureMessage"):
                outcome = "FAILED"
            parsed[relative] = outcome
        return parsed


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------


class GoBackend(LanguageBackend):
    name = "go"

    def _env(self) -> dict[str, str]:
        """Mirror the official evaluator: readonly module mode on the container's own proxy.

        The agent image already points ``GOPROXY`` at the local file cache
        (``file:///go/pkg/mod/cache/download``) and disables ``GOSUMDB``; the
        evaluator adds ``-mod=readonly``.  ``GOPROXY=off`` would be stricter
        than the evaluator and fails even the untouched baseline (an earlier run
        canary), so it is never used here.
        """
        env = clean_environment()
        flags = env.get("GOFLAGS", "").split()
        if not any(flag.startswith("-mod=") for flag in flags):
            flags.append("-mod=readonly")
        env["GOFLAGS"] = " ".join(flags)
        env.setdefault("GOSUMDB", "off")
        return env

    def _tags(self) -> list[str]:
        return _flags_from_build_command(self.contract.build_command, ("-tags",))

    def _module_roots(self, root: Path, paths: Sequence[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for path in paths:
            if not path.endswith(".go"):
                continue
            module_root = _nearest_ancestor_with(path, root, "go.mod")
            grouped.setdefault(module_root, []).append(path)
        return grouped

    def manifest_guard(self, root: Path, scope: SubmissionScope, timeout: float) -> list[GuardFailure]:
        """Only a submission that edits the module files can break the offline graph.

        The untouched baseline is not re-validated: the official image may
        legitimately lack modules that ``go list -m all`` would enumerate.  When
        ``go.mod``/``go.sum`` changed, every package and test dependency of the
        module must still resolve from the local proxy in readonly mode, which
        is exactly what the evaluator's ``go test -mod=readonly`` needs.
        """
        touched = [p for p in scope.root_manifests if p in {"go.mod", "go.sum", "go.work", "go.work.sum"}]
        if not touched:
            return []
        failures: list[GuardFailure] = []
        module_roots = set(self._module_roots(root, scope.in_scope_source)) | {"."}
        for module_root in sorted(module_roots):
            base = root if module_root == "." else root / module_root
            if not (base / "go.mod").is_file():
                continue
            result = run_command(
                ["go", "list", *self._tags(), "-deps", "-test", "./..."],
                cwd=base,
                timeout=timeout,
                env=self._env(),
            )
            if result.returncode != 0:
                failures.append(
                    GuardFailure(
                        "go_module_graph_offline",
                        f"after your edit to {', '.join(touched)} the module graph no longer resolves "
                        "in -mod=readonly mode from the local module cache (the official evaluator "
                        "has no network); revert added/upgraded dependencies or restore the original "
                        "go.mod/go.sum",
                        result.command,
                        result.tail,
                    )
                )
                continue
            verify = run_command(["go", "mod", "verify"], cwd=base, timeout=timeout, env=self._env())
            if verify.returncode != 0:
                failures.append(
                    GuardFailure("go_mod_verify_failed", "go mod verify failed offline", verify.command, verify.tail)
                )
                continue
            # go list -deps -test can succeed while the actual test compiler
            # still discovers test-only imports that require a manifest
            # update. Compile the repository test graph with no test
            # execution so the route guard catches the same go.mod/go.sum
            # failure the official evaluator would report.
            test_graph = run_command(
                ["go", "test", *self._tags(), "-run", "^$", "-count=0", "./..."],
                cwd=base,
                timeout=timeout,
                env=self._env(),
            )
            if test_graph.returncode != 0:
                failures.append(
                    GuardFailure(
                        "go_test_module_graph",
                        "offline Go test compilation failed; update only the SRS-required "
                        "go.mod/go.sum entries from the local cache before submission",
                        test_graph.command,
                        test_graph.tail,
                    )
                )
        return failures

    def build_guard(self, root: Path, scope: SubmissionScope, timeout: float) -> list[GuardFailure]:
        failures: list[GuardFailure] = []
        for module_root, files in self._module_roots(root, scope.in_scope_source).items():
            base = root if module_root == "." else root / module_root
            result = run_command(["go", "build", *self._tags(), "./..."], cwd=base, timeout=timeout, env=self._env())
            if result.returncode != 0:
                failures.append(
                    GuardFailure(
                        "go_build_failed",
                        "go build ./... failed; official P2P packages in this module cannot compile",
                        result.command,
                        result.tail,
                    )
                )
                continue
            dirs = sorted({str(PurePosixPath(path).parent) for path in files})
            targets = []
            for directory in dirs:
                relative = directory if module_root == "." else str(PurePosixPath(directory).relative_to(module_root))
                targets.append("./" + relative if relative not in {".", ""} else ".")
            vet = run_command(["go", "vet", *self._tags(), *targets], cwd=base, timeout=timeout, env=self._env())
            if vet.returncode != 0 and re.search(r"(?m)^#|\.go:\d+:\d+: (?:undefined|cannot|syntax|missing)", vet.tail):
                failures.append(GuardFailure("go_vet_failed", "go vet reported compile-level errors in changed packages", vet.command, vet.tail))
        return failures

    def affected_units(self, root: Path, scope: SubmissionScope, timeout: float) -> list[str]:
        units: list[str] = []
        for module_root, files in self._module_roots(root, scope.in_scope_source).items():
            base = root if module_root == "." else root / module_root
            dirs = sorted({str(PurePosixPath(path).parent) for path in files})
            changed_pkgs: list[str] = []
            for directory in dirs:
                relative = directory if module_root == "." else str(PurePosixPath(directory).relative_to(module_root))
                result = run_command(
                    ["go", "list", *self._tags(), "./" + relative if relative not in {".", ""} else "."],
                    cwd=base,
                    timeout=timeout,
                    env=self._env(),
                )
                if result.returncode == 0:
                    changed_pkgs.extend(line.strip() for line in result.stdout.splitlines() if line.strip())
            changed_set = set(changed_pkgs)
            dependents: list[str] = []
            if changed_set:
                listing = run_command(
                    ["go", "list", *self._tags(), "-f", "{{.ImportPath}}\t{{join .Deps \" \"}}", "./..."],
                    cwd=base,
                    timeout=timeout,
                    env=self._env(),
                )
                if listing.returncode == 0:
                    for line in listing.stdout.splitlines():
                        import_path, _, deps = line.partition("\t")
                        if import_path in changed_set:
                            continue
                        if changed_set.intersection(deps.split()):
                            dependents.append(import_path)
            for pkg in (*changed_pkgs, *sorted(dependents)):
                unit = pkg if module_root == "." else f"{module_root}::{pkg}"
                units.append(unit)
        return list(dict.fromkeys(units))

    def run_units(self, root: Path, units: Sequence[str], timeout: float) -> list[UnitOutcome]:
        outcomes: list[UnitOutcome] = []
        by_module: dict[str, list[str]] = {}
        for unit in units:
            module_root, pkg = (unit.split("::", 1) if "::" in unit else (".", unit))
            by_module.setdefault(module_root, []).append(pkg)
        for module_root, pkgs in by_module.items():
            base = root if module_root == "." else root / module_root
            # ``-p 1`` runs the package test binaries one at a time.  Several
            # go-zero packages start the same dev HTTP server on :6060 in their
            # tests; in parallel the second one fails with "address already in
            # use", which the guard then reported as a regression of a package
            # the submission never touched (an earlier run, core/stores/redis).
            result = run_command(
                ["go", "test", "-count=1", "-p", "1", "-json", *self._tags(), *pkgs],
                cwd=base,
                timeout=timeout,
                env=self._env(),
            )
            status: dict[str, str] = {}
            texts: dict[str, list[str]] = {}
            for line in result.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # Build errors are printed as plain text before any event.
                    texts.setdefault("", []).append(line)
                    continue
                if not isinstance(event, dict):
                    continue
                pkg = str(event.get("Package", ""))
                action = event.get("Action")
                if action == "output":
                    texts.setdefault(pkg, []).append(str(event.get("Output", "")))
                if event.get("Test"):
                    continue
                if action == "pass":
                    status[pkg] = "PASSED"
                elif action == "fail":
                    status[pkg] = "FAILED"
                elif action == "skip":
                    status.setdefault(pkg, "SKIPPED")
            for pkg in pkgs:
                unit = pkg if module_root == "." else f"{module_root}::{pkg}"
                outcome = status.get(pkg)
                if outcome is None:
                    outcome = "PASSED" if result.returncode == 0 else "ERROR"
                text = "".join(texts.get(pkg, ())) + "".join(texts.get("", ()))
                if result.stderr:
                    text += "\n" + result.stderr
                outcomes.append(
                    UnitOutcome(
                        unit,
                        outcome,
                        result.returncode,
                        result.duration_ms,
                        text[-_OUTPUT_TAIL:] if outcome != "PASSED" else "",
                    )
                )
        return outcomes

    def unit_exists(self, root: Path, unit: str) -> bool:
        module_root, pkg = (unit.split("::", 1) if "::" in unit else (".", unit))
        base = root if module_root == "." else root / module_root
        result = run_command(["go", "list", *self._tags(), pkg], cwd=base, timeout=120, env=self._env())
        return result.returncode == 0


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------


class CargoBackend(LanguageBackend):
    name = "cargo"
    # No shared ``target``.  Cargo names artifacts by workspace-relative
    # package path and judges freshness by mtime alone, so a target directory
    # shared between the baseline tree, the verify tree and /testbed handed
    # the verify tree a baseline rlib whose source it had changed (nushell
    # core_development.2 in an earlier run: ``PipelineData::byte_stream`` "not
    # found" while pointing at the baseline worktree).  Every tree compiles
    # into its own directory next to the worktrees.
    shared_links = ()

    #: full-workspace warm build of the baseline tree, started in the
    #: background when the verifier is created so the first guard does not
    #: pay a cold compile inside its own budget.
    warmup_timeout_seconds = 3600.0
    #: budget for a build guard whose target directory is still cold.
    cold_build_timeout_seconds = 1800.0

    @staticmethod
    def target_dir(root: Path) -> Path:
        label, _, rest = root.name.partition("-")
        if label in {"verify", "baseline"} and rest:
            return root.parent / f"cargo-target-{label}"
        return root / "target"

    def _env(self, root: Path) -> dict[str, str]:
        env = clean_environment({"CARGO_NET_OFFLINE": "true"})
        env["CARGO_TARGET_DIR"] = str(self.target_dir(root))
        return env

    def cold(self, root: Path) -> bool:
        return not (self.target_dir(root) / "debug").is_dir()

    def warm_baseline(self, root: Path, timeout: float) -> CommandResult:
        argv = ["cargo", "test", "--offline", "--no-run", "--workspace", *self._features()]
        return run_command(argv, cwd=root, timeout=timeout, env=self._env(root))

    def seed_target(self, baseline_root: Path, verify_root: Path, touch_paths: Sequence[str]) -> bool:
        """Copy the warm baseline target for the verify tree, once.

        ``cp -a`` keeps artifact mtimes, so files the verify worktree shares
        with the baseline (created before the baseline build) stay fresh
        while ``touch_paths`` — the submitted files — are bumped to *now*
        and therefore rebuilt together with their dependents.
        """

        source = self.target_dir(baseline_root)
        destination = self.target_dir(verify_root)
        if destination.exists() or not (source / "debug").is_dir():
            return False
        partial = destination.with_name(destination.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        copied = subprocess.run(
            ["cp", "-a", str(source), str(partial)],
            capture_output=True,
            check=False,
            timeout=3600,
        )
        if copied.returncode != 0:
            shutil.rmtree(partial, ignore_errors=True)
            return False
        partial.rename(destination)
        now = time.time()
        for relative in touch_paths:
            path = verify_root / relative
            if path.is_file():
                try:
                    os.utime(path, (now, now))
                except OSError:
                    pass
        return True

    def _features(self) -> list[str]:
        return _flags_from_build_command(self.contract.build_command, ("--features", "--all-features", "--no-default-features"))

    def _metadata(self, root: Path, timeout: float) -> tuple[dict[str, Any] | None, CommandResult]:
        result = run_command(
            ["cargo", "metadata", "--offline", "--format-version", "1", "--no-deps"],
            cwd=root,
            timeout=timeout,
            env=self._env(root),
        )
        if result.returncode != 0:
            return None, result
        try:
            return json.loads(result.stdout), result
        except json.JSONDecodeError:
            return None, result

    def manifest_guard(self, root: Path, scope: SubmissionScope, timeout: float) -> list[GuardFailure]:
        metadata, result = self._metadata(root, timeout)
        if metadata is None:
            return [
                GuardFailure(
                    "cargo_manifest_invalid",
                    "cargo metadata --offline failed; Cargo.toml/Cargo.lock as submitted cannot be "
                    "parsed or resolved by the offline evaluator",
                    result.command,
                    result.tail,
                )
            ]
        return []

    def _crates_for_paths(self, root: Path, paths: Sequence[str], timeout: float) -> tuple[list[str], dict[str, str]]:
        metadata, _ = self._metadata(root, timeout)
        crate_dirs: dict[str, str] = {}
        if metadata:
            for package in metadata.get("packages", ()):
                manifest = Path(str(package.get("manifest_path", "")))
                try:
                    relative = manifest.parent.resolve().relative_to(root.resolve())
                except ValueError:
                    continue
                crate_dirs[str(relative) if str(relative) != "." else "."] = str(package.get("name"))
        crates: list[str] = []
        for path in paths:
            if not path.endswith(".rs"):
                continue
            directory = _nearest_ancestor_with(path, root, "Cargo.toml")
            name = crate_dirs.get(directory)
            if name:
                crates.append(name)
        return list(dict.fromkeys(crates)), crate_dirs

    def build_guard(self, root: Path, scope: SubmissionScope, timeout: float) -> list[GuardFailure]:
        crates, _ = self._crates_for_paths(root, scope.in_scope_source, timeout)
        if not crates:
            return []
        argv = ["cargo", "test", "--offline", "--no-run"]
        for crate in crates:
            argv.extend(["-p", crate])
        result = run_command(argv, cwd=root, timeout=timeout, env=self._env(root))
        if result.timed_out:
            return [GuardFailure("cargo_build_timeout", "cargo test --no-run did not finish", result.command, result.tail)]
        if result.returncode != 0:
            return [
                GuardFailure(
                    "cargo_build_failed",
                    "affected crates (or their tests) no longer compile",
                    result.command,
                    result.tail,
                )
            ]
        return []

    def affected_units(self, root: Path, scope: SubmissionScope, timeout: float) -> list[str]:
        crates, _ = self._crates_for_paths(root, scope.in_scope_source, timeout)
        if not crates:
            return []
        # Reverse dependencies inside the workspace, from the full resolve graph.
        result = run_command(
            ["cargo", "metadata", "--offline", "--format-version", "1"],
            cwd=root,
            timeout=timeout,
            env=self._env(root),
        )
        dependents: list[str] = []
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                data = {}
            members = set(map(str, data.get("workspace_members", ())))
            names = {str(pkg["id"]): str(pkg["name"]) for pkg in data.get("packages", ()) if pkg.get("id")}
            changed_ids = {pkg_id for pkg_id, name in names.items() if name in set(crates)}
            for node in (data.get("resolve") or {}).get("nodes", ()):
                node_id = str(node.get("id", ""))
                if node_id not in members or node_id in changed_ids:
                    continue
                if changed_ids.intersection(map(str, node.get("dependencies", ()))):
                    dependents.append(names.get(node_id, node_id))
        return list(dict.fromkeys((*crates, *sorted(dependents))))

    def run_units(self, root: Path, units: Sequence[str], timeout: float) -> list[UnitOutcome]:
        outcomes: list[UnitOutcome] = []
        for crate in units:
            argv = ["cargo", "test", "--offline", "-p", crate]
            result = run_command(argv, cwd=root, timeout=timeout, env=self._env(root))
            if result.timed_out:
                outcome = "ERROR"
            elif result.returncode == 0:
                outcome = "PASSED"
            elif "test result: FAILED" in result.stdout or "failures:" in result.stdout:
                outcome = "FAILED"
            else:
                outcome = "ERROR"
            outcomes.append(UnitOutcome(crate, outcome, result.returncode, result.duration_ms, result.tail if outcome != "PASSED" else ""))
        return outcomes

    def unit_exists(self, root: Path, unit: str) -> bool:
        metadata, _ = self._metadata(root, 300)
        if not metadata:
            return False
        return any(str(pkg.get("name")) == unit for pkg in metadata.get("packages", ()))


# ---------------------------------------------------------------------------
# Java / Groovy (Maven)
# ---------------------------------------------------------------------------


class MavenBackend(LanguageBackend):
    name = "maven"

    def _skips(self) -> list[str]:
        picked = _flags_from_build_command(
            self.contract.build_command,
            ("-Pskip-spotless", "-Dcheckstyle.skip", "-Drat.skip", "-Dmaven.javadoc.skip", "-Dlicense.skip", "-Dspotless.check.skip"),
        )
        defaults = ["-Dcheckstyle.skip=true", "-Drat.skip=true", "-Dmaven.javadoc.skip=true", "-Dlicense.skip=true", "-Dspotless.check.skip=true"]
        return list(dict.fromkeys((*picked, *defaults)))

    def _env(self) -> dict[str, str]:
        env = clean_environment()
        env["MAVEN_ARGS"] = (env.get("MAVEN_ARGS", "") + " -o -B").strip()
        return env

    def _reactor(self, root: Path) -> MavenReactor:
        return MavenReactor.load(root)

    def _modules(self, root: Path, paths: Sequence[str]) -> list[str]:
        reactor = self._reactor(root)
        affected = reactor.affected_modules(
            paths,
            upstream_depth=1,
            downstream_depth=2,
            max_modules=self.contract.max_units,
        )
        runnable = [module.path for module in affected if module.packaging != "pom"]
        return runnable or [module.path for module in affected]

    def manifest_guard(self, root: Path, scope: SubmissionScope, timeout: float) -> list[GuardFailure]:
        reactor = self._reactor(root)
        if reactor.errors:
            return [
                GuardFailure(
                    "maven_pom_invalid",
                    "the declared Maven reactor contains invalid POM metadata: "
                    + "; ".join(reactor.errors[:8]),
                )
            ]
        result = run_command(["mvn", "-o", "-B", "-q", "validate", *self._skips()], cwd=root, timeout=timeout, env=self._env())
        if result.returncode != 0:
            return [
                GuardFailure(
                    "maven_pom_invalid",
                    "mvn validate failed offline; a POM as submitted cannot be read by the official evaluator "
                    "(missing dependency versions, unresolved plugins)",
                    result.command,
                    result.tail,
                )
            ]
        return self._root_pom_property_guard(root, scope)

    _POM_PROPERTY_REF = re.compile(r"\$\{([A-Za-z0-9_.-]+)\}")

    @staticmethod
    def _pom_property_names(text: str) -> set[str]:
        try:
            tree = ET.fromstring(text)
        except ET.ParseError:
            return set()
        names: set[str] = set()
        for properties in tree.iter():
            if properties.tag.rsplit("}", 1)[-1] != "properties":
                continue
            for child in properties:
                names.add(child.tag.rsplit("}", 1)[-1])
        return names

    def _root_pom_property_guard(self, root: Path, scope: SubmissionScope) -> list[GuardFailure]:
        """Properties added to the root POM and used only by agent-added module POMs.

        The official evaluator overlays the agent's root ``pom.xml`` onto its
        prepared parent with a three-way merge and keeps the parent's hunk on
        conflict.  A property the agent added next to evaluator-prepared
        edits is dropped, and a new module POM that reads ``${property}`` no
        longer resolves (dubbo M003.1, ``${mutiny.version}``).  Defining the
        version in the module POM itself survives the overlay.
        """

        if "pom.xml" not in scope.root_manifests:
            return []
        try:
            target_text = (root / "pom.xml").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        baseline = run_command(
            ["git", "show", f"{scope.baseline_revision}:pom.xml"], cwd=root, timeout=60
        )
        if baseline.returncode != 0:
            return []
        added = self._pom_property_names(target_text) - self._pom_property_names(baseline.stdout)
        if not added:
            return []
        module_poms = [
            path
            for path in scope.changed_paths
            if path != "pom.xml" and PurePosixPath(path).name == "pom.xml"
        ]
        at_risk: dict[str, list[str]] = {}
        for path in module_poms:
            exists_at_baseline = run_command(
                ["git", "cat-file", "-e", f"{scope.baseline_revision}:{path}"], cwd=root, timeout=60
            ).returncode == 0
            if exists_at_baseline:
                continue
            try:
                text = (root / path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for name in set(self._POM_PROPERTY_REF.findall(text)) & added:
                at_risk.setdefault(name, []).append(path)
        if not at_risk:
            return []
        detail = "; ".join(f"${{{name}}} used by {', '.join(sorted(paths))}" for name, paths in sorted(at_risk.items()))
        return [
            GuardFailure(
                "maven_root_pom_property_conflict_risk",
                "the root pom.xml gains properties that only your new module POMs read. The official "
                "evaluator merges your root pom.xml onto its prepared copy and keeps its own hunk on "
                "any conflict, so a new root property can vanish and the module then fails with an "
                "unresolvable ${property}. Define the version inside the module's own pom.xml "
                "(or its <dependencyManagement>) instead of the root <properties>: " + detail,
            )
        ]

    def build_guard(self, root: Path, scope: SubmissionScope, timeout: float) -> list[GuardFailure]:
        modules = self._modules(root, scope.submitted_paths)
        if not modules:
            return []
        argv = ["mvn", "-o", "-B", "-q", "-pl", ",".join(modules), "-am", "test-compile", *self._skips()]
        result = run_command(argv, cwd=root, timeout=timeout, env=self._env())
        if result.timed_out:
            return [GuardFailure("maven_compile_timeout", "mvn test-compile did not finish", result.command, result.tail)]
        if result.returncode != 0:
            return [GuardFailure("maven_compile_failed", "affected modules (or their tests) no longer compile", result.command, result.tail)]
        return []

    def affected_units(self, root: Path, scope: SubmissionScope, timeout: float) -> list[str]:
        return self._modules(root, scope.submitted_paths)

    def run_units(self, root: Path, units: Sequence[str], timeout: float) -> list[UnitOutcome]:
        if not units:
            return []
        before = self._report_snapshot(root, units)
        argv = [
            "mvn",
            "-o",
            "-B",
            "-fae",
            "-pl",
            ",".join(units),
            "-am",
            "test",
            "-DfailIfNoTests=false",
            "-Dsurefire.failIfNoSpecifiedTests=false",
            *self._skips(),
        ]
        result = run_command(argv, cwd=root, timeout=timeout, env=self._env())
        status = self._reactor_status(result.stdout)
        reactor = self._reactor(root)
        outcomes: list[UnitOutcome] = []
        offline_failure = result.returncode != 0 and is_offline_resolution_failure(result.tail)
        for module in units:
            report_outcome = self._module_report_outcome(root, module, before)
            model = reactor.module(module)
            artifact = model.coordinate.artifact_id if model is not None else self._artifact_id(
                root / module / "pom.xml"
            )
            outcome = report_outcome or status.get(artifact) or status.get(PurePosixPath(module).name)
            if outcome is None:
                outcome = "PASSED" if result.returncode == 0 else "ERROR"
            if result.timed_out and outcome != "FAILED":
                outcome = "ERROR"
            # A reactor that aborted because an artifact is absent from the
            # offline repository produced no test verdict for any module: the
            # aborted module and every SKIPPED sibling report ENVIRONMENT, so
            # the verifier does not read "all units skipped" as a pass.
            if offline_failure and report_outcome is None and outcome in {"FAILED", "ERROR", "SKIPPED"}:
                outcome = ENVIRONMENT_OUTCOME
            outcomes.append(UnitOutcome(module, outcome, result.returncode, result.duration_ms, result.tail if outcome != "PASSED" else ""))
        return outcomes

    @staticmethod
    def _report_paths(root: Path, module: str) -> tuple[Path, ...]:
        base = root if module == "." else root / module
        paths: list[Path] = []
        for directory in ("surefire-reports", "failsafe-reports"):
            report_dir = base / "target" / directory
            try:
                paths.extend(sorted(report_dir.glob("TEST-*.xml"))[:2048])
            except OSError:
                continue
        return tuple(paths)

    @classmethod
    def _report_snapshot(
        cls, root: Path, modules: Sequence[str]
    ) -> dict[Path, tuple[int, int]]:
        snapshot: dict[Path, tuple[int, int]] = {}
        for module in modules:
            for report in cls._report_paths(root, module):
                try:
                    stat = report.stat()
                except OSError:
                    continue
                snapshot[report] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

    @classmethod
    def _module_report_outcome(
        cls,
        root: Path,
        module: str,
        before: Mapping[Path, tuple[int, int]] | None = None,
    ) -> str | None:
        totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        parsed = False
        for report in cls._report_paths(root, module):
            if before is not None:
                try:
                    stat = report.stat()
                except OSError:
                    continue
                if before.get(report) == (stat.st_mtime_ns, stat.st_size):
                    continue
            try:
                suite = ET.parse(report).getroot()
                values = {
                    key: int(suite.attrib.get(key, "0") or "0")
                    for key in totals
                }
            except (OSError, ET.ParseError, TypeError, ValueError):
                return "ERROR"
            parsed = True
            for key, value in values.items():
                totals[key] += value
        if not parsed:
            return None
        if totals["failures"] or totals["errors"]:
            return "FAILED"
        if totals["tests"] and totals["skipped"] >= totals["tests"]:
            return "SKIPPED"
        return "PASSED"

    @staticmethod
    def _artifact_id(pom: Path) -> str:
        try:
            text = pom.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        # The first artifactId outside <parent> is the module's own.
        stripped = re.sub(r"<parent>.*?</parent>", "", text, flags=re.S)
        match = re.search(r"<artifactId>\s*([^<\s]+)\s*</artifactId>", stripped)
        return match.group(1) if match else ""

    @staticmethod
    def _reactor_status(output: str) -> dict[str, str]:
        status: dict[str, str] = {}
        for line in output.splitlines():
            match = re.match(r"\[INFO\]\s+(.+?)\s+\.{2,}\s+(SUCCESS|FAILURE|SKIPPED)", line)
            if match:
                name = match.group(1).strip()
                verdict = match.group(2)
                outcome = "PASSED" if verdict == "SUCCESS" else "FAILED" if verdict == "FAILURE" else "SKIPPED"
                status[name] = outcome
                # Reactor names are "Dubbo Common" style display names or artifactIds.
                status[name.replace(" ", "-").lower()] = outcome
        return status

    def unit_exists(self, root: Path, unit: str) -> bool:
        return ((root if unit == "." else root / unit) / "pom.xml").is_file()


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


_PY_TEST_FILE = re.compile(r"(?:^|/)(?:test_[^/]*\.py|[^/]*_test\.py)$")
_PY_BINARY_SUFFIXES = (".so", ".pyd", ".dylib")


class PytestBackend(LanguageBackend):
    """Affected-scope pytest verification for Python repositories.

    The official evaluator runs the project's own pytest suite on the
    submitted source tree.  This backend selects the test files that a change
    can reach (same package ``tests/`` directories and files importing the
    changed modules), runs them one file at a time so every unit gets its own
    PASSED/FAILED verdict, and treats "no tests collected" as SKIPPED.

    Repositories with in-place compiled extensions (scikit-learn) keep those
    binaries next to the sources; a disposable worktree has none.  The
    binaries are copied from the live repository into the worktree at the
    same relative paths so imports resolve; they are build products of the
    current sources, which is the closest offline approximation available.
    """

    name = "python"
    #: set by the verifier; the live checkout that owns compiled artifacts
    repository: Path | None = None

    def _python(self, root: Path) -> list[str]:
        for candidate in (
            os.environ.get("HOMY_SM_PYTHON"),
            str(root / ".venv" / "bin" / "python"),
        ):
            if candidate and Path(candidate).is_file():
                return [candidate]
        for name in ("python", "python3"):
            found = shutil.which(name)
            if found:
                return [found]
        return ["python3"]

    def _pytest(self, root: Path) -> list[str]:
        return [*self._python(root), "-m", "pytest"]

    def _env(self) -> dict[str, str]:
        env = clean_environment()
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        env.setdefault("PYTHONHASHSEED", "0")
        return env

    def _copy_binary_artifacts(self, root: Path) -> int:
        """Mirror compiled extension modules from the live repository into ``root``."""

        source_root = self.repository
        if source_root is None or Path(source_root).resolve() == root.resolve():
            return 0
        copied = 0
        for directory in self.contract.repo_src_dirs:
            base = Path(source_root) / directory
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                if not path.is_file() or path.suffix not in _PY_BINARY_SUFFIXES:
                    continue
                try:
                    relative = path.relative_to(source_root)
                except ValueError:
                    continue
                destination = root / relative
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                copied += 1
        return copied

    @staticmethod
    def _module_names(path: str) -> tuple[str, ...]:
        """Dotted module names a repository-relative ``.py`` path may be imported as."""

        pure = PurePosixPath(path)
        if pure.suffix != ".py":
            return ()
        parts = list(pure.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            return ()
        names = [".".join(parts)]
        # ``src/pkg/mod.py`` layouts are imported as ``pkg.mod``.
        if parts[0] in {"src", "lib"} and len(parts) > 1:
            names.append(".".join(parts[1:]))
        return tuple(dict.fromkeys(names))

    def _test_files(self, root: Path) -> list[str]:
        found: list[str] = []
        for path in root.rglob("*.py"):
            if any(part in {".git", "node_modules", "__pycache__", ".venv", "build", "dist"}
                   for part in path.parts):
                continue
            relative = str(path.relative_to(root).as_posix())
            if not _PY_TEST_FILE.search(relative):
                continue
            if self.contract.test_dirs and not any(
                fnmatch_path(relative, pattern) for pattern in self.contract.test_dirs
            ):
                # Only tests inside the repository's declared test scope are
                # official P2P candidates; examples and docs are not.
                continue
            found.append(relative)
        return sorted(found)

    def build_guard(self, root: Path, scope: SubmissionScope, timeout: float) -> list[GuardFailure]:
        files = [p for p in scope.in_scope_source if p.endswith(".py") and (root / p).is_file()]
        if not files:
            return []
        self._copy_binary_artifacts(root)
        result = run_command(
            [*self._python(root), "-m", "compileall", "-q", *files],
            cwd=root,
            timeout=timeout,
            env=self._env(),
        )
        if result.returncode != 0:
            return [
                GuardFailure(
                    "python_syntax_error",
                    "changed Python sources do not compile; every official test importing "
                    "them would error at collection",
                    result.command,
                    result.tail,
                )
            ]
        return []

    def affected_units(self, root: Path, scope: SubmissionScope, timeout: float) -> list[str]:
        changed = [p for p in scope.in_scope_source if p.endswith(".py")]
        if not changed:
            return []
        modules: set[str] = set()
        package_dirs: set[str] = set()
        for path in changed:
            modules.update(self._module_names(path))
            package_dirs.add(str(PurePosixPath(path).parent))
        units: list[str] = []
        needles = sorted(modules, key=len, reverse=True)
        for test_file in self._test_files(root):
            test_dir = PurePosixPath(test_file).parent
            # ``pkg/tests/test_x.py`` and ``pkg/test_x.py`` belong to ``pkg``.
            owner = str(test_dir.parent) if test_dir.name in {"tests", "test"} else str(test_dir)
            if owner in package_dirs:
                units.append(test_file)
                continue
            try:
                text = (root / test_file).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if any(
                re.search(r"(?<![\w.])" + re.escape(name) + r"(?![\w])", text) for name in needles
            ):
                units.append(test_file)
        return list(dict.fromkeys(units))

    def unit_exists(self, root: Path, unit: str) -> bool:
        return (root / unit).is_file()

    def run_units(self, root: Path, units: Sequence[str], timeout: float) -> list[UnitOutcome]:
        outcomes: list[UnitOutcome] = []
        if not units:
            return outcomes
        self._copy_binary_artifacts(root)
        for unit in units:
            result = run_command(
                [
                    *self._pytest(root),
                    "-p", "no:cacheprovider",
                    "-q",
                    "-x",
                    "--no-header",
                    "-o", "addopts=",
                    unit,
                ],
                cwd=root,
                timeout=timeout,
                env=self._env(),
            )
            if result.timed_out:
                outcome = "ERROR"
            elif result.returncode == 0:
                outcome = "PASSED"
            elif result.returncode == 5:
                outcome = "SKIPPED"
            elif result.returncode == 1:
                outcome = "FAILED"
            else:
                outcome = "ERROR"
            outcomes.append(
                UnitOutcome(
                    unit,
                    outcome,
                    result.returncode,
                    result.duration_ms,
                    result.tail if outcome not in {"PASSED", "SKIPPED"} else "",
                )
            )
        return outcomes


def fnmatch_path(path: str, pattern: str) -> bool:
    """``**``-aware glob match on a repository-relative POSIX path."""

    import fnmatch

    posix = PurePosixPath(path)
    if fnmatch.fnmatch(path, pattern) or posix.match(pattern):
        return True
    if pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:]):
        return True
    # ``pkg/tests/**`` style: every path below the directory.
    prefix = pattern.split("*", 1)[0].rstrip("/")
    return bool(prefix) and (path == prefix or path.startswith(prefix + "/"))


def backend_for(contract: SweMilestoneContract, language: str) -> LanguageBackend:
    language = language.lower()
    if language == "python":
        return PytestBackend(contract)
    if language in {"typescript", "javascript"}:
        return NodeBackend(contract)
    if language == "go":
        return GoBackend(contract)
    if language == "rust":
        return CargoBackend(contract)
    if language in {"java", "groovy"}:
        return MavenBackend(contract)
    raise ValueError(f"no SWE-Milestone verifier backend for {language}")


def backends_for_contract(contract: SweMilestoneContract) -> tuple[LanguageBackend, ...]:
    seen: dict[str, LanguageBackend] = {}
    for language in contract.languages:
        backend = backend_for(contract, language)
        seen.setdefault(backend.name, backend)
    return tuple(seen.values())


def tool_available(name: str) -> bool:
    return shutil.which(name) is not None
