from __future__ import annotations

import hashlib
import json
import shlex
import stat
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .models import SweEvoInstance

CONTROL_DIRECTORY = ".swe-evo"
FAIL_TO_PASS_TARGETS_FILE = "fail-to-pass.txt"
PASS_TO_PASS_TARGETS_FILE = "pass-to-pass.txt"
VERIFICATION_SCRIPT = "verify.sh"
GIT_CONFIG_FILE = "gitconfig"
VERIFICATION_COMMAND = f"./{CONTROL_DIRECTORY}/{VERIFICATION_SCRIPT}"
CONTAINER_TEST_PATH = (
    "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
CONTAINER_REPOSITORY_PATH = "/testbed"
CONTAINER_GIT_ENVIRONMENT = (
    f"GIT_CONFIG_GLOBAL={CONTAINER_REPOSITORY_PATH}/{CONTROL_DIRECTORY}/{GIT_CONFIG_FILE}",
)
PASS_TO_PASS_CALIBRATION_CHUNK = 512


class BenchmarkBaselineError(RuntimeError):
    """The declared benchmark cannot produce a trustworthy baseline contract."""


def _fixture_path_state(path: Path) -> Mapping[str, object]:
    if path.is_symlink():
        target = path.readlink().as_posix()
        return {
            "kind": "SYMLINK",
            "target": target,
            "sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
        }
    if not path.exists():
        return {"kind": "MISSING"}
    if not path.is_file():
        return {"kind": "UNSUPPORTED"}
    metadata = path.stat()
    return {
        "kind": "REGULAR_FILE",
        "size": metadata.st_size,
        "mode": stat.S_IMODE(metadata.st_mode),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def capture_fixture_integrity(
    workspace: Path,
    fixture_paths: Sequence[str],
) -> Mapping[str, Mapping[str, object]]:
    """Capture benchmark-owned files outside the model's authority domain."""

    root = workspace.resolve()
    result: dict[str, Mapping[str, object]] = {}
    for raw in fixture_paths:
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"invalid benchmark fixture path: {raw!r}")
        result[relative.as_posix()] = _fixture_path_state(root / relative)
    return dict(sorted(result.items()))


def fixture_integrity_violations(
    workspace: Path,
    expected: Mapping[str, object],
) -> tuple[str, ...]:
    """Return exact fixture paths changed since the trusted preparation receipt."""

    actual = capture_fixture_integrity(workspace, tuple(map(str, expected)))
    return tuple(
        path
        for path in sorted(actual)
        if not isinstance(expected.get(path), Mapping) or dict(actual[path]) != dict(expected[path])
    )


def _test_command_arguments(instance: SweEvoInstance) -> list[str]:
    """Return the benchmark-owned pytest command for the instance image.

    SWE-EVO images carry their dependencies in the ``testbed`` conda
    environment.  Executing the pytest binary by absolute path is not enough:
    tests may spawn sibling tools such as ``ruff`` and therefore also need the
    environment's bin directory on PATH.
    """

    try:
        arguments = shlex.split(instance.test_cmds)
    except ValueError as exc:
        raise ValueError("SWE-EVO test_cmds is not valid shell-style argv") from exc
    if not arguments or Path(arguments[0]).name != "pytest":
        raise ValueError("SWE-EVO test_cmds must start with pytest")
    arguments[0] = "/opt/miniconda3/envs/testbed/bin/pytest"
    return arguments


def _container_environment_arguments() -> list[str]:
    arguments: list[str] = []
    for assignment in (f"PATH={CONTAINER_TEST_PATH}", *CONTAINER_GIT_ENVIRONMENT):
        arguments.extend(("-e", assignment))
    return arguments


def run_command(
    arguments: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int = 600,
    input_text: str | None = None,
    environment: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        input=input_text,
        env=dict(environment) if environment is not None else None,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "command failed"
        raise RuntimeError(f"{arguments[0]} exited {completed.returncode}: {detail[:4000]}")
    return completed


def _repository_mirror(instance: SweEvoInstance, mirror_root: Path | None) -> Path | None:
    if mirror_root is None:
        return None
    candidate = mirror_root.expanduser().resolve() / f"{instance.repo.replace('/', '__')}.git"
    if not candidate.is_dir():
        return None
    present = run_command(
        [
            "git",
            "--git-dir",
            str(candidate),
            "cat-file",
            "-e",
            f"{instance.base_commit}^{{commit}}",
        ],
        cwd=candidate,
        check=False,
    )
    return candidate if present.returncode == 0 else None


def _install_verification_assets(instance: SweEvoInstance, workspace: Path) -> tuple[str, ...]:
    control = workspace / CONTROL_DIRECTORY
    if control.exists():
        raise RuntimeError(f"benchmark control path already exists in repository: {control}")
    if not instance.fail_to_pass:
        raise ValueError("SWE-EVO instance has no FAIL_TO_PASS tests")
    all_targets = (*instance.fail_to_pass, *instance.pass_to_pass)
    invalid = [target for target in all_targets if "\n" in target or "\r" in target]
    if invalid:
        raise ValueError("SWE-EVO test identifiers must not contain line breaks")
    control.mkdir()
    fail_targets = control / FAIL_TO_PASS_TARGETS_FILE
    pass_targets = control / PASS_TO_PASS_TARGETS_FILE
    fail_targets.write_text("\n".join(instance.fail_to_pass) + "\n", encoding="utf-8")
    pass_targets.write_text(
        ("\n".join(instance.pass_to_pass) + "\n") if instance.pass_to_pass else "",
        encoding="utf-8",
    )
    git_config = control / GIT_CONFIG_FILE
    git_config.write_text(
        f"[safe]\n\tdirectory = {CONTAINER_REPOSITORY_PATH}\n",
        encoding="utf-8",
    )
    test_command = " ".join(shlex.quote(item) for item in _test_command_arguments(instance))
    script = control / VERIFICATION_SCRIPT
    container_environment = " ".join(
        f"-e {shlex.quote(assignment)}"
        for assignment in (f"PATH={CONTAINER_TEST_PATH}", *CONTAINER_GIT_ENVIRONMENT)
    )
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"\n'
        'repo_root="$(cd -- "$script_dir/.." && pwd -P)"\n'
        'mapfile -t fail_targets < "$script_dir/' + FAIL_TO_PASS_TARGETS_FILE + '"\n'
        'mapfile -t pass_targets < "$script_dir/' + PASS_TO_PASS_TARGETS_FILE + '"\n'
        'targets=("${fail_targets[@]}" "${pass_targets[@]}")\n'
        "if (( ${#targets[@]} == 0 )); then\n"
        '  echo "SWE-EVO verification target list is empty" >&2\n'
        "  exit 2\n"
        "fi\n"
        "exec docker run --rm --network none "
        + f'-v "$repo_root:{CONTAINER_REPOSITORY_PATH}" '
        + f"-w {shlex.quote(CONTAINER_REPOSITORY_PATH)} "
        + container_environment
        + " "
        + shlex.quote(instance.image)
        + " "
        + test_command
        + ' "${targets[@]}"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return (
        fail_targets.relative_to(workspace).as_posix(),
        pass_targets.relative_to(workspace).as_posix(),
        git_config.relative_to(workspace).as_posix(),
        script.relative_to(workspace).as_posix(),
    )


def prepare_workspace(
    instance: SweEvoInstance,
    workspace: Path,
    *,
    mirror_root: Path | None = None,
) -> Mapping[str, object]:
    workspace = workspace.resolve()
    if workspace.exists() and (not workspace.is_dir() or any(workspace.iterdir())):
        raise ValueError(f"SWE-EVO workspace is not empty: {workspace}")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    clone_url = f"https://github.com/{instance.repo}.git"
    mirror = _repository_mirror(instance, mirror_root)
    if mirror is None:
        clone_arguments = [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            clone_url,
            str(workspace),
        ]
        clone_source = "network"
    else:
        clone_arguments = ["git", "clone", "--local", "--no-checkout", str(mirror), str(workspace)]
        clone_source = "shared_mirror"
    run_command(clone_arguments, cwd=workspace.parent, timeout_seconds=1200)
    if mirror is not None:
        run_command(["git", "remote", "set-url", "origin", clone_url], cwd=workspace)
    run_command(["git", "checkout", "--detach", instance.base_commit], cwd=workspace)
    observed = run_command(["git", "rev-parse", "HEAD"], cwd=workspace).stdout.strip()
    if observed != instance.base_commit:
        raise RuntimeError("SWE-EVO checkout does not match base_commit")
    has_test_patch = bool(instance.test_patch.strip())
    if has_test_patch:
        run_command(
            ["git", "apply", "--check", "-"],
            cwd=workspace,
            input_text=instance.test_patch,
        )
        run_command(["git", "apply", "-"], cwd=workspace, input_text=instance.test_patch)
    verification_assets = _install_verification_assets(instance, workspace)
    run_command(["git", "config", "user.email", "swe-evo-fixture@example.invalid"], cwd=workspace)
    run_command(["git", "config", "user.name", "SWE-EVO Fixture"], cwd=workspace)
    run_command(["git", "add", "-A"], cwd=workspace)
    fixture_paths = tuple(
        line
        for line in run_command(
            ["git", "diff", "--cached", "--name-only"], cwd=workspace
        ).stdout.splitlines()
        if line
    )
    run_command(["git", "commit", "-m", f"test fixture for {instance.instance_id}"], cwd=workspace)
    fixture_commit = run_command(["git", "rev-parse", "HEAD"], cwd=workspace).stdout.strip()
    if (
        run_command(["git", "rev-parse", "HEAD^"], cwd=workspace).stdout.strip()
        != instance.base_commit
    ):
        raise RuntimeError("SWE-EVO fixture commit has the wrong parent")
    if run_command(["git", "status", "--porcelain"], cwd=workspace).stdout.strip():
        raise RuntimeError("SWE-EVO prepared workspace is dirty")
    return {
        "clone_url": clone_url,
        "base_commit": observed,
        "fixture_commit": fixture_commit,
        "fixture_paths": fixture_paths,
        "fixture_integrity": capture_fixture_integrity(workspace, fixture_paths),
        "clone_source": clone_source,
        "mirror_path": str(mirror) if mirror is not None else None,
        "test_patch_applied": has_test_patch,
        "verification_assets": verification_assets,
        "verification_command": VERIFICATION_COMMAND,
    }


def docker_test_arguments(
    instance: SweEvoInstance,
    workspace: Path,
    *,
    junit_path: Path | None = None,
    include_pass_to_pass: bool = False,
    pass_to_pass_targets: Sequence[str] | None = None,
) -> list[str]:
    if pass_to_pass_targets is not None and not include_pass_to_pass:
        raise ValueError("calibrated PASS_TO_PASS targets require include_pass_to_pass")
    arguments = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-v",
        f"{workspace.resolve()}:{CONTAINER_REPOSITORY_PATH}",
        *_container_environment_arguments(),
    ]
    if junit_path is not None:
        report = junit_path.expanduser().resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        arguments.extend(["-v", f"{report.parent}:/homy-results"])
    arguments.extend(
        [
            "-w",
            CONTAINER_REPOSITORY_PATH,
            instance.image,
            *_test_command_arguments(instance),
        ]
    )
    if junit_path is not None:
        arguments.append(f"--junitxml=/homy-results/{junit_path.name}")
    arguments.extend(instance.fail_to_pass)
    if include_pass_to_pass:
        arguments.extend(
            instance.pass_to_pass if pass_to_pass_targets is None else tuple(pass_to_pass_targets)
        )
    return arguments


def _junit_summary(path: Path) -> Mapping[str, int] | None:
    if not path.is_file():
        return None
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failures = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    return {
        "tests": len(cases),
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "passed": len(cases) - failures - errors - skipped,
    }


def _pytest_case_key(target: str) -> tuple[str, str] | None:
    parts = target.split("::")
    if len(parts) < 2:
        return None
    module = parts[0].replace("\\", "/")
    while module.startswith("./"):
        module = module[2:]
    if module.endswith(".py"):
        module = module[:-3]
    classname = ".".join((module.replace("/", "."), *parts[1:-1]))
    return classname, parts[-1]


def _junit_target_outcomes(path: Path, targets: Sequence[str]) -> Mapping[str, str]:
    if not path.is_file():
        return {target: "MISSING" for target in targets}
    observed: dict[tuple[str, str], str] = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        key = (str(case.attrib.get("classname", "")), str(case.attrib.get("name", "")))
        if case.find("failure") is not None:
            outcome = "FAILED"
        elif case.find("error") is not None:
            outcome = "ERROR"
        elif case.find("skipped") is not None:
            outcome = "SKIPPED"
        else:
            outcome = "PASSED"
        previous = observed.get(key)
        if previous is None or previous == "PASSED":
            observed[key] = outcome
    return {
        target: observed.get(key, "MISSING") if key is not None else "MISSING"
        for target in targets
        for key in (_pytest_case_key(target),)
    }


def _require_container_started(completed: object, *, scope: str) -> None:
    """Separate Docker launch failure from a test process reporting non-passing tests."""

    returncode = int(getattr(completed, "returncode", -1))
    if returncode not in {125, 126, 127}:
        return
    stderr = str(getattr(completed, "stderr", ""))
    stdout = str(getattr(completed, "stdout", ""))
    detail = " ".join((stderr or stdout or "no Docker diagnostic").split())[-1200:]
    raise RuntimeError(
        f"SWE-EVO {scope} container did not start (docker exit {returncode}): {detail}"
    )


def calibrate_isolated_pass_to_pass(
    instance: SweEvoInstance,
    workspace: Path,
    *,
    junit_path: Path,
) -> tuple[tuple[str, ...], Mapping[str, object]]:
    """Select only regressions that pass on the untouched network-isolated baseline."""

    if not instance.pass_to_pass:
        empty_digest = hashlib.sha256(b"[]").hexdigest()
        return (), {
            "schema": "codex-longterm-v2/swe-evo-regression-calibration@1",
            "declared_count": 0,
            "selected_count": 0,
            "excluded_count": 0,
            "selected_sha256": empty_digest,
            "selected_targets": [],
            "outcomes": {},
            "state": "READY",
        }

    def selector_error(target: str) -> str | None:
        if "::" not in target or not target.split("::", 1)[0].endswith(".py"):
            return "INVALID_NODE_ID"
        raw_path = target.split("::", 1)[0].replace("\\", "/")
        while raw_path.startswith("./"):
            raw_path = raw_path[2:]
        relative = Path(raw_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            return "INVALID_NODE_ID"
        depth = 0
        for character in target:
            if character == "[":
                depth += 1
            elif character == "]":
                depth -= 1
                if depth < 0:
                    return "UNBALANCED_PARAMETER_ID"
        return "UNBALANCED_PARAMETER_ID" if depth else None

    declared_outcomes: dict[str, str] = {}
    valid: list[str] = []
    selector_files: dict[str, str] = {}
    for target in instance.pass_to_pass:
        error = selector_error(target)
        if error is None:
            valid.append(target)
            raw_path = target.split("::", 1)[0].replace("\\", "/")
            while raw_path.startswith("./"):
                raw_path = raw_path[2:]
            selector_files[target] = Path(raw_path).as_posix()
        else:
            declared_outcomes[target] = error

    collection_files: list[str] = []
    seen_collection_files: set[str] = set()
    for target in valid:
        relative_path = selector_files[target]
        if not (workspace / relative_path).is_file():
            declared_outcomes[target] = "NOT_COLLECTED"
            continue
        if relative_path not in seen_collection_files:
            seen_collection_files.add(relative_path)
            collection_files.append(relative_path)

    def normalize_node_id(target: str) -> str:
        normalized = target.strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    collected_by_normalized: dict[str, str] = {}
    collection_chunk_receipts: list[Mapping[str, object]] = []
    collection_commands: list[str] = []
    for ordinal, start in enumerate(
        range(0, len(collection_files), PASS_TO_PASS_CALIBRATION_CHUNK),
        start=1,
    ):
        file_chunk = tuple(
            collection_files[start : start + PASS_TO_PASS_CALIBRATION_CHUNK]
        )
        file_set = set(file_chunk)
        declared_count = sum(
            selector_files[target] in file_set
            for target in valid
            if target not in declared_outcomes
        )
        collection_arguments = docker_test_arguments(instance, workspace)
        if instance.fail_to_pass:
            collection_arguments = collection_arguments[: -len(instance.fail_to_pass)]
        collection_arguments.extend(
            (
                "-o",
                "addopts=",
                "--collect-only",
                "-q",
                "--continue-on-collection-errors",
                *file_chunk,
            )
        )
        collection = run_command(
            collection_arguments,
            cwd=workspace,
            timeout_seconds=1800,
            check=False,
        )
        _require_container_started(collection, scope="PASS_TO_PASS collection")
        chunk_collected: list[str] = []
        for line in collection.stdout.splitlines():
            candidate = line.strip()
            if "::" not in candidate or candidate.startswith(("<", "=")):
                continue
            normalized = normalize_node_id(candidate)
            if normalized not in collected_by_normalized:
                collected_by_normalized[normalized] = candidate
                chunk_collected.append(candidate)
        command = shlex.join(collection_arguments)
        collection_commands.append(command)
        collection_chunk_receipts.append(
            {
                "ordinal": ordinal,
                "file_count": len(file_chunk),
                "declared_count": declared_count,
                "returncode": collection.returncode,
                "collected_count": len(chunk_collected),
                "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
                "stdout_tail": collection.stdout[-2000:],
                "stderr_tail": collection.stderr[-2000:],
            }
        )
    collected = tuple(collected_by_normalized.values())
    declared_to_collected: dict[str, tuple[str, ...]] = {}
    calibration_targets: list[str] = []
    for target in valid:
        normalized = normalize_node_id(target)
        matches = tuple(
            actual
            for node_id, actual in collected_by_normalized.items()
            if node_id == normalized
            or node_id.startswith(normalized + "[")
            or node_id.startswith(normalized + "::")
        )
        if not matches:
            declared_outcomes[target] = "NOT_COLLECTED"
            continue
        declared_to_collected[target] = matches
        calibration_targets.extend(matches)
    calibration_targets = list(dict.fromkeys(calibration_targets))

    target_outcomes: dict[str, str] = {}
    chunk_receipts: list[Mapping[str, object]] = []
    for ordinal, start in enumerate(
        range(0, len(calibration_targets), PASS_TO_PASS_CALIBRATION_CHUNK),
        start=1,
    ):
        targets = tuple(
            calibration_targets[start : start + PASS_TO_PASS_CALIBRATION_CHUNK]
        )
        selected_junit = (
            junit_path
            if len(calibration_targets) <= PASS_TO_PASS_CALIBRATION_CHUNK and ordinal == 1
            else junit_path.with_name(f"{junit_path.stem}.chunk-{ordinal:04d}{junit_path.suffix}")
        )
        arguments = docker_test_arguments(
            instance,
            workspace,
            junit_path=selected_junit,
            include_pass_to_pass=True,
            pass_to_pass_targets=targets,
        )
        suffix_count = len(instance.fail_to_pass) + len(targets)
        arguments = arguments[:-suffix_count] + list(targets)
        completed = run_command(arguments, cwd=workspace, timeout_seconds=1800, check=False)
        _require_container_started(completed, scope="PASS_TO_PASS calibration")
        observed = _junit_target_outcomes(selected_junit, targets)
        if (
            completed.returncode != 0
            and all(value == "MISSING" for value in observed.values())
        ):
            detail = " ".join((completed.stderr or completed.stdout or "no diagnostic").split())
            raise BenchmarkBaselineError(
                "collected PASS_TO_PASS nodes produced no JUnit cases; refusing recursive "
                f"selector bisection (chunk {ordinal}): {detail[-1600:]}"
            )
        target_outcomes.update(observed)
        chunk_receipts.append(
            {
                "ordinal": ordinal,
                "target_count": len(targets),
                "returncode": completed.returncode,
                "junit_path": str(selected_junit),
                "outcomes": dict(sorted(Counter(observed.values()).items())),
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
        )

    selected = tuple(
        target for target in calibration_targets if target_outcomes.get(target) == "PASSED"
    )
    outcome_priority = ("ERROR", "FAILED", "MISSING", "SKIPPED", "PASSED")
    for declared, targets in declared_to_collected.items():
        values = tuple(target_outcomes.get(target, "MISSING") for target in targets)
        declared_outcomes[declared] = next(
            (outcome for outcome in outcome_priority if outcome in values),
            "MISSING",
        )
    counts = Counter(declared_outcomes.values())
    selected_payload = json.dumps(
        selected,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    receipt = {
        "schema": "codex-longterm-v2/swe-evo-regression-calibration@1",
        "returncode": max(
            (int(item["returncode"]) for item in chunk_receipts),
            default=0,
        ),
        "declared_count": len(instance.pass_to_pass),
        "selected_count": len(selected),
        "excluded_count": sum(value != "PASSED" for value in declared_outcomes.values()),
        "selected_node_count": len(selected),
        "selected_sha256": hashlib.sha256(selected_payload).hexdigest(),
        "selected_targets": list(selected),
        "outcomes": dict(sorted(counts.items())),
        "invalid_selector_count": sum(
            value in {"INVALID_NODE_ID", "UNBALANCED_PARAMETER_ID"}
            for value in declared_outcomes.values()
        ),
        "not_collected_count": sum(
            value == "NOT_COLLECTED" for value in declared_outcomes.values()
        ),
        "selector_outcomes": dict(sorted(declared_outcomes.items())),
        "declared_to_collected": {
            key: list(value) for key, value in sorted(declared_to_collected.items())
        },
        "collection": {
            "returncode": max(
                (int(item["returncode"]) for item in collection_chunk_receipts),
                default=0,
            ),
            "chunk_count": len(collection_chunk_receipts),
            "file_count": len(collection_files),
            "collected_count": len(collected),
            "collected_sha256": hashlib.sha256(
                json.dumps(collected, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "command_sha256": hashlib.sha256(
                json.dumps(
                    collection_commands,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "chunks": collection_chunk_receipts,
        },
        "calibration_chunk_size": PASS_TO_PASS_CALIBRATION_CHUNK,
        "chunks": chunk_receipts,
        "state": "READY" if selected else "NO_STABLE_SELECTORS",
    }
    return selected, receipt


def run_docker_tests(
    instance: SweEvoInstance,
    workspace: Path,
    *,
    junit_path: Path | None = None,
    include_pass_to_pass: bool = False,
    pass_to_pass_targets: Sequence[str] | None = None,
) -> Mapping[str, object]:
    arguments = docker_test_arguments(
        instance,
        workspace,
        junit_path=junit_path,
        include_pass_to_pass=include_pass_to_pass,
        pass_to_pass_targets=pass_to_pass_targets,
    )
    completed = run_command(arguments, cwd=workspace, timeout_seconds=1800, check=False)
    _require_container_started(completed, scope="test")
    junit = _junit_summary(junit_path) if junit_path is not None else None
    fail_to_pass_outcomes = (
        _junit_target_outcomes(junit_path, instance.fail_to_pass)
        if junit_path is not None
        else {target: "MISSING" for target in instance.fail_to_pass}
    )
    pass_to_pass_targets_used = (
        tuple(instance.pass_to_pass if pass_to_pass_targets is None else pass_to_pass_targets)
        if include_pass_to_pass
        else ()
    )
    pass_to_pass_outcomes = (
        _junit_target_outcomes(junit_path, pass_to_pass_targets_used)
        if junit_path is not None and include_pass_to_pass
        else {}
    )
    outcome_counts = Counter(fail_to_pass_outcomes.values())
    regression_outcome_counts = Counter(pass_to_pass_outcomes.values())
    return {
        "command": shlex.join(arguments),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-20000:],
        "passed": completed.returncode == 0,
        "junit": junit,
        "verification_scope": (
            "FAIL_TO_PASS_AND_PASS_TO_PASS" if include_pass_to_pass else "FAIL_TO_PASS"
        ),
        "fail_to_pass_count": len(instance.fail_to_pass),
        "fail_to_pass_outcomes": dict(fail_to_pass_outcomes),
        "fail_to_pass_outcome_counts": dict(sorted(outcome_counts.items())),
        "exact_failed_count": outcome_counts["FAILED"],
        "any_fail_to_pass_nonpassing": any(
            outcome != "PASSED" for outcome in fail_to_pass_outcomes.values()
        ),
        "pass_to_pass_count": (
            len(pass_to_pass_targets_used)
        ),
        "pass_to_pass_outcomes": dict(pass_to_pass_outcomes),
        "pass_to_pass_outcome_counts": dict(sorted(regression_outcome_counts.items())),
        "any_pass_to_pass_nonpassing": any(
            outcome != "PASSED" for outcome in pass_to_pass_outcomes.values()
        ),
        "regression_passed": (
            all(outcome == "PASSED" for outcome in pass_to_pass_outcomes.values())
            if include_pass_to_pass
            else None
        ),
        # Aggregate JUnit counts cannot prove which declared selector failed.
        # This exact per-target receipt is the only pre-execution fact that may
        # later satisfy a TEST_FAILURE Step criterion.
        "all_fail_to_pass_observed": bool(instance.fail_to_pass)
        and all(outcome == "FAILED" for outcome in fail_to_pass_outcomes.values()),
    }


def collect_model_patch(
    workspace: Path,
    *,
    fixture_commit: str,
    fixture_paths: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    tracked_names = tuple(
        line
        for line in run_command(
            ["git", "diff", "--name-only", fixture_commit], cwd=workspace
        ).stdout.splitlines()
        if line
    )
    untracked_names = tuple(
        line
        for line in run_command(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=workspace
        ).stdout.splitlines()
        if line
    )
    names = tuple(dict.fromkeys((*tracked_names, *untracked_names)))
    fixture_set = set(fixture_paths)
    forbidden = [
        name
        for name in names
        if name in fixture_set or name == "tests" or name.startswith(("tests/", "test/"))
    ]
    if forbidden:
        raise RuntimeError(f"agent modified benchmark fixture/test files: {forbidden}")
    if untracked_names:
        run_command(["git", "add", "-N", "--", *untracked_names], cwd=workspace)
    patch = run_command(["git", "diff", "--binary", fixture_commit], cwd=workspace).stdout
    return patch, names
