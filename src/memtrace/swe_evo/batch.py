from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

from ..build_identity import current_build_identity
from ..contracts import stable_id
from ..durability import append_synced, atomic_write_once, json_line, load_json_lines
from ..harness import resolve_codex_executable
from .models import list_instance_ids
from .progress import read_semantic_progress

_MAX_CONTROL_ACTIVITY_WITHOUT_SEMANTIC_PROGRESS = 8
_MAX_SAME_FRONTIER_RECOVERY_SEGMENTS = 3
_CANARY_RUNTIME_IDENTITY_FIELDS = (
    "schema",
    "distribution",
    "version",
    "build_id",
    "source_digest",
    "codex_cli_bin_version",
    "load_mode",
    "receipt_verified",
)


def _write_once(path: Path, value: object) -> None:
    atomic_write_once(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
    )


def _latest_attempt_failure(attempt_root: Path) -> Mapping[str, object]:
    events = load_json_lines(attempt_root / "swe_evo-failure-events.jsonl")
    if events and isinstance(events[-1], Mapping):
        return events[-1]
    failure_path = attempt_root / "swe_evo_failure.json"
    if failure_path.is_file():
        value = json.loads(failure_path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            return value
    return {}


def _repairable_infrastructure_attempt(attempt_root: Path) -> bool:
    failure = _latest_attempt_failure(attempt_root)
    return bool(
        not (attempt_root / "swe_evo_result.json").is_file()
        and failure.get("failure_class") == "EVALUATION_INFRASTRUCTURE"
        and failure.get("retryable") is True
    )


def _latest_workspace_revision_artifact(attempt_root: Path) -> Mapping[str, object] | None:
    """Return the newest durable code receipt left by the runtime process.

    This is deliberately read-only: Batch never invents route progress or
    rewrites the workspace. It only carries the runtime's already-committed
    Revision/Patch address across a timeout or non-zero process exit.
    """

    candidates = sorted(
        (attempt_root / "runtime" / "workspace-revisions").glob("revision_*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        return None
    receipt_path = candidates[-1]
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError("Workspace Revision artifact receipt is not a JSON object")
    patch_path = value.get("patch_path")
    if patch_path is not None and not Path(str(patch_path)).is_file():
        raise RuntimeError("Workspace Revision artifact points to a missing patch")
    return {**dict(value), "receipt_path": str(receipt_path)}


class _BatchJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def record(self, instance_id: str, state: str, **payload: object) -> None:
        value = {
            "schema": "codex-longterm-v2/swe-evo-batch-event@1",
            "recorded_at": datetime.now(UTC).isoformat(),
            "instance_id": instance_id,
            "state": state,
            **payload,
        }
        with self.lock:
            append_synced(self.path, json_line(value))


def _private_log(path: Path):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    return os.fdopen(descriptor, "wb")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _linux_cpu_sample() -> tuple[int, int] | None:
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    except (OSError, IndexError):
        return None
    if not fields or fields[0] != "cpu":
        return None
    values = [int(item) for item in fields[1:]]
    if len(values) < 4:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def _cpu_utilization_percent(sample_seconds: float = 0.25) -> float | None:
    before = _linux_cpu_sample()
    if before is None:
        return None
    time.sleep(sample_seconds)
    after = _linux_cpu_sample()
    if after is None:
        return None
    total_delta = after[0] - before[0]
    idle_delta = after[1] - before[1]
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (total_delta - idle_delta) / total_delta))


def _wait_for_cpu_capacity(
    threshold: float | None,
    *,
    deadline: float,
    journal: _BatchJournal,
    instance_id: str,
    pause_requested: threading.Event | None = None,
) -> bool:
    """Gate new Attempt admission; never interrupt an already running task."""

    if threshold is None:
        return pause_requested is None or not pause_requested.is_set()
    while time.monotonic() < deadline:
        if pause_requested is not None and pause_requested.is_set():
            return False
        observed = _cpu_utilization_percent()
        if observed is None or observed < threshold:
            return True
        journal.record(
            instance_id,
            "CPU_ADMISSION_DEFERRED",
            observed_percent=round(observed, 2),
            threshold_percent=threshold,
        )
        time.sleep(5)
    return False


def _attempt_directories(task_root: Path) -> list[Path]:
    return sorted(
        path
        for path in task_root.glob("attempt-*")
        if path.is_dir() and path.name.removeprefix("attempt-").isdigit()
    )


def _completed_attempt(task_root: Path) -> Path | None:
    return next(
        (
            path
            for path in reversed(_attempt_directories(task_root))
            if (path / "swe_evo_result.json").is_file()
        ),
        None,
    )


def _attempt_evaluation_state(attempt_root: Path) -> str:
    rescores = sorted(attempt_root.glob("official-rescore-*.json"))
    source = rescores[-1] if rescores else attempt_root / "swe_evo_result.json"
    if not source.is_file():
        return "MISSING"
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        return "INVALID"
    official = value.get("official_evaluation")
    if not isinstance(official, Mapping):
        return str(value.get("evaluation_state", "UNKNOWN"))
    if (official.get("score_valid") is False and official.get("rescore_eligible") is True) or (
        official.get("official_pass") is False and official.get("error_type") is not None
    ):
        return "EVALUATION_PENDING"
    return str(official.get("state", value.get("evaluation_state", "COMPLETED")))


def _validated_canary_receipt(
    path: Path,
    *,
    build_identity: Mapping[str, object],
) -> Mapping[str, object]:
    """Require one completed run from the exact executable build before full scale."""

    selected = path.expanduser().resolve()
    try:
        raw = selected.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"canary result is unreadable: {selected}") from exc
    if not isinstance(value, Mapping) or value.get("schema") != (
        "codex-longterm-v2/swe-evo-result@1"
    ):
        raise RuntimeError("canary result does not use the production SWE-EVO result schema")
    canary_identity = value.get("build_identity")
    if (
        not isinstance(canary_identity, Mapping)
        or canary_identity.get("load_mode") != "verified-installed-package"
        or canary_identity.get("receipt_verified") is not True
        or not isinstance(canary_identity.get("codex_cli_bin_version"), str)
        or not canary_identity.get("codex_cli_bin_version")
        or any(
            canary_identity.get(key) != build_identity.get(key)
            for key in _CANARY_RUNTIME_IDENTITY_FIELDS
        )
    ):
        raise RuntimeError("canary result was not produced by the exact current installed runtime")
    runtime = value.get("runtime_result")
    if not isinstance(runtime, Mapping) or not (
        runtime.get("completion_verdict") == "COMPLETED"
        and runtime.get("task_status") == "COMPLETED"
    ):
        raise RuntimeError("canary did not complete the production runtime contract")
    official = value.get("official_evaluation")
    if not isinstance(official, Mapping) or not (
        value.get("evaluation_state") == "COMPLETED"
        and official.get("state") == "COMPLETED"
        and official.get("score_valid") is True
        and official.get("rescore_eligible") is not True
        and official.get("error_type") is None
        and isinstance(official.get("official_pass"), bool)
    ):
        raise RuntimeError("canary did not complete a valid official evaluation")
    return {
        "path": str(selected),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "instance_id": (
            value.get("instance", {}).get("instance_id")
            if isinstance(value.get("instance"), Mapping)
            else None
        ),
        "build_id": canary_identity.get("build_id"),
        "runtime_completion": "COMPLETED",
        "official_evaluation": "COMPLETED",
        "official_pass": official.get("official_pass"),
        "score_valid": True,
    }


def _model_blocker_directive_count(attempt_root: Path) -> int:
    """Count durable model-owned blocker suspensions for one Attempt.

    A segment compares the count before and after its child process.  This
    avoids mistaking a directive from an earlier, explicitly resumed segment
    for a new suspension decision.
    """

    events = load_json_lines(attempt_root / "batch-control.jsonl")
    return sum(event.get("state") == "SUSPEND_BLOCKED" for event in events)


def _run_subprocess_attempt(
    *,
    instance_id: str,
    attempt_root: Path,
    task_root: Path,
    common_arguments: Sequence[str],
    timeout_seconds: int,
    temporary_root: Path,
    journal: _BatchJournal,
    resume: bool = False,
    wall_timeout_seconds: int | None = None,
    pause_requested: threading.Event | None = None,
) -> Mapping[str, object]:
    index = int(attempt_root.name.removeprefix("attempt-"))
    if resume:
        if not attempt_root.is_dir():
            raise FileNotFoundError("cannot resume a missing SWE-EVO Attempt")
    else:
        attempt_root.mkdir(parents=True, exist_ok=False)
    segment = len(tuple(task_root.glob(f"attempt-{index:03d}.segment-*.json"))) + 1
    stdout_path = task_root / f"attempt-{index:03d}.segment-{segment:03d}.stdout.log"
    stderr_path = task_root / f"attempt-{index:03d}.segment-{segment:03d}.stderr.log"
    command = [
        sys.executable,
        "-m",
        "memtrace",
        "swe-evo",
        "--instance-id",
        instance_id,
        "--run-root",
        str(attempt_root),
        *common_arguments,
    ]
    if resume:
        command.append("--resume")
    environment = dict(os.environ)
    temporary_root.mkdir(parents=True, exist_ok=True)
    environment["HOMY_SWE_EVO_TMPDIR"] = str(temporary_root)
    started = time.perf_counter_ns()
    journal.record(
        instance_id,
        "ATTEMPT_STARTED",
        attempt=index,
        segment=segment,
        resume=resume,
        run_root=str(attempt_root),
    )
    timed_out = False
    paused = False
    timeout_reason: str | None = None
    stall_directive_id: str | None = None
    blocker_directives_before = _model_blocker_directive_count(attempt_root)
    semantic_progress = read_semantic_progress(attempt_root)
    semantic_transition_count = 0
    diagnostic_transition_count = 0
    with _private_log(stdout_path) as stdout, _private_log(stderr_path) as stderr:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            start_new_session=True,
        )
        last_progress = semantic_progress.fingerprint
        last_diagnostic = semantic_progress.diagnostic_fingerprint
        control_activity_baseline = semantic_progress.control_activity_count
        idle_deadline = time.monotonic() + timeout_seconds
        wall_deadline = (
            time.monotonic() + wall_timeout_seconds if wall_timeout_seconds is not None else None
        )
        while True:
            try:
                returncode = process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                observed_progress = read_semantic_progress(attempt_root)
                if observed_progress.fingerprint != last_progress:
                    last_progress = observed_progress.fingerprint
                    last_diagnostic = observed_progress.diagnostic_fingerprint
                    control_activity_baseline = observed_progress.control_activity_count
                    semantic_transition_count += 1
                    idle_deadline = now + timeout_seconds
                elif observed_progress.diagnostic_fingerprint != last_diagnostic:
                    # Novel diagnosis protects a valid exploration branch from
                    # the review-churn fence, but does not masquerade as route
                    # progress or extend the semantic idle deadline.
                    last_diagnostic = observed_progress.diagnostic_fingerprint
                    control_activity_baseline = observed_progress.control_activity_count
                    diagnostic_transition_count += 1
                semantic_progress = observed_progress
                control_churn = max(
                    0,
                    observed_progress.control_activity_count - control_activity_baseline,
                )
                if pause_requested is not None and pause_requested.is_set():
                    paused = True
                    timeout_reason = "OPERATOR_PAUSE"
                elif wall_deadline is not None and now >= wall_deadline:
                    timed_out = True
                    timeout_reason = "BATCH_DEADLINE"
                elif (
                    observed_progress.active_side_effect_count == 0
                    and control_churn >= _MAX_CONTROL_ACTIVITY_WITHOUT_SEMANTIC_PROGRESS
                ):
                    timed_out = True
                    timeout_reason = "NO_SEMANTIC_PROGRESS_CONTROL_CHURN"
                elif now >= idle_deadline:
                    timed_out = True
                    timeout_reason = "NO_SEMANTIC_PROGRESS"
                if paused or timed_out:
                    if timeout_reason in {
                        "NO_SEMANTIC_PROGRESS",
                        "NO_SEMANTIC_PROGRESS_CONTROL_CHURN",
                    }:
                        stall_directive_id = stable_id(
                            "resume_",
                            {
                                "instance_id": instance_id,
                                "attempt": index,
                                "segment": segment,
                                "semantic_progress": semantic_progress.as_mapping(),
                            },
                        )
                        directive = {
                            "schema": "codex-longterm-v2/execution-resume-directive@1",
                            "directive_id": stall_directive_id,
                            "state": "SUSPEND_STALLED",
                            "cause": "NO_SEMANTIC_PROGRESS",
                            "trigger": timeout_reason,
                            "same_attempt": True,
                            "decision_authority": "MODEL",
                            "allowed_step_decisions": ["CONTINUE", "CORRECT"],
                            "semantic_progress": semantic_progress.as_mapping(),
                        }
                        append_synced(
                            attempt_root / "batch-control.jsonl",
                            json_line(directive),
                        )
                        journal.record(
                            instance_id,
                            "SEMANTIC_STALL_SUSPEND_REQUESTED",
                            attempt=index,
                            segment=segment,
                            directive_id=stall_directive_id,
                            trigger=timeout_reason,
                            same_attempt=True,
                        )
                    if paused:
                        pause_id = stable_id(
                            "pause_",
                            {
                                "instance_id": instance_id,
                                "attempt": index,
                                "segment": segment,
                                "semantic_progress": semantic_progress.as_mapping(),
                            },
                        )
                        append_synced(
                            attempt_root / "batch-control.jsonl",
                            json_line(
                                {
                                    "schema": "codex-longterm-v2/operator-pause@1",
                                    "directive_id": pause_id,
                                    "state": "OPERATOR_PAUSED",
                                    "cause": "OPERATOR_PAUSE",
                                    "same_attempt": True,
                                    "semantic_progress": semantic_progress.as_mapping(),
                                }
                            ),
                        )
                        journal.record(
                            instance_id,
                            "OPERATOR_PAUSE_DURABLE",
                            attempt=index,
                            segment=segment,
                            directive_id=pause_id,
                            same_attempt=True,
                        )
                    _terminate_process_group(process)
                    returncode = (
                        process.returncode if process.returncode is not None else -signal.SIGKILL
                    )
                    break
    result_exists = (attempt_root / "swe_evo_result.json").is_file()
    runtime_suspended = _model_blocker_directive_count(attempt_root) > blocker_directives_before
    failure = _latest_attempt_failure(attempt_root)
    latest_workspace_artifact = _latest_workspace_revision_artifact(attempt_root)
    failure_class = str(failure.get("failure_class", ""))
    infrastructure_repair_eligible = bool(
        not result_exists
        and not timed_out
        and not paused
        and not runtime_suspended
        and failure.get("retryable") is True
        and failure_class == "EVALUATION_INFRASTRUCTURE"
    )
    same_attempt_resume_eligible = bool(
        not result_exists
        and not timed_out
        and not paused
        and not runtime_suspended
        and failure.get("retryable") is True
        and not failure_class.startswith(("BENCHMARK_BASELINE", "EVALUATION_INFRASTRUCTURE"))
    )
    receipt = {
        "schema": "codex-longterm-v2/swe-evo-attempt-segment@3",
        "instance_id": instance_id,
        "attempt": index,
        "segment": segment,
        "resumed": resume,
        "run_root": str(attempt_root),
        "returncode": returncode,
        "timed_out": timed_out,
        "paused": paused,
        "timeout_reason": timeout_reason,
        "timeout_seconds": timeout_seconds,
        "stall_directive_id": stall_directive_id,
        "semantic_transition_count": semantic_transition_count,
        "diagnostic_transition_count": diagnostic_transition_count,
        "semantic_progress": semantic_progress.as_mapping(),
        "duration_ms": (time.perf_counter_ns() - started) / 1_000_000,
        "result_exists": result_exists,
        "runtime_suspended": runtime_suspended,
        "failure_class": failure.get("failure_class"),
        "failure_retryable": failure.get("retryable"),
        "infrastructure_repair_eligible": infrastructure_repair_eligible,
        "same_attempt_resume_eligible": same_attempt_resume_eligible,
        "latest_workspace_revision_artifact": latest_workspace_artifact,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    _write_once(task_root / f"attempt-{index:03d}.segment-{segment:03d}.json", receipt)
    attempt_summary = task_root / f"attempt-{index:03d}.json"
    if (
        not timed_out
        and not paused
        and not runtime_suspended
        and not same_attempt_resume_eligible
        and not infrastructure_repair_eligible
        and not (resume and attempt_summary.exists())
    ):
        _write_once(attempt_summary, receipt)
    journal.record(
        instance_id,
        (
            "ATTEMPT_COMPLETED"
            if result_exists
            else "ATTEMPT_SUSPENDED"
            if timed_out or paused or runtime_suspended or same_attempt_resume_eligible
            else "ATTEMPT_INFRASTRUCTURE_BLOCKED"
            if infrastructure_repair_eligible
            else "ATTEMPT_FAILED"
        ),
        **{key: value for key, value in receipt.items() if key not in {"schema", "instance_id"}},
    )
    return receipt


def run_swe_evo_batch(
    *,
    dataset_path: Path,
    swe_bench_root: Path,
    batch_root: Path,
    config_path: Path,
    release_manifest: Path | None,
    instance_ids: Sequence[str] = (),
    all_instances: bool = False,
    model: str | None = None,
    reasoning_effort: str | None = None,
    codex_bin: str | None = None,
    dataset_python: Path | None = None,
    evaluator_python: Path | None = None,
    evaluator_workers: int = 1,
    concurrency: int = 2,
    max_attempts: int = 2,
    task_timeout_seconds: int = 14_400,
    batch_timeout_seconds: int = 172_800,
    repo_mirror_root: Path | None = None,
    skip_official: bool = False,
    allow_development_build: bool = False,
    cpu_utilization_threshold: float | None = None,
    canary_result: Path | None = None,
    provider_api_key_envs: Sequence[str] = (),
) -> Mapping[str, object]:
    if concurrency < 1 or max_attempts < 1 or evaluator_workers < 1:
        raise ValueError("batch concurrency, attempts and evaluator workers must be positive")
    if task_timeout_seconds < 60 or batch_timeout_seconds < task_timeout_seconds:
        raise ValueError("batch timeouts are invalid")
    if cpu_utilization_threshold is not None and not 0 < cpu_utilization_threshold <= 100:
        raise ValueError("CPU utilization threshold must be in (0, 100]")
    provider_api_key_envs = tuple(dict.fromkeys(map(str, provider_api_key_envs)))
    if provider_api_key_envs:
        invalid_envs = [
            name
            for name in provider_api_key_envs
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
        ]
        if invalid_envs:
            raise ValueError(
                "provider API key pool must contain environment-variable names: "
                + ", ".join(invalid_envs)
            )
        missing_envs = [name for name in provider_api_key_envs if not os.environ.get(name)]
        if missing_envs:
            raise RuntimeError(
                "provider API key pool environment variables are absent: "
                + ", ".join(missing_envs)
            )
        if concurrency > len(provider_api_key_envs):
            raise ValueError(
                "provider API key pool has fewer slots than batch concurrency; "
                "reduce --concurrency or provide one environment variable per worker"
            )
    available = list_instance_ids(dataset_path, dataset_python=dataset_python)
    requested = available if all_instances else tuple(dict.fromkeys(instance_ids))
    if not requested:
        raise ValueError("batch requires --all-instances or at least one --instance-id")
    unknown = sorted(set(requested).difference(available))
    if unknown:
        raise ValueError(f"batch contains unknown SWE-EVO instance IDs: {unknown}")
    # Resolve once in the Batch parent before a manifest or Attempt exists.
    # Every child receives this exact executable, so a bad path cannot consume
    # attempts and a resumed Batch cannot silently switch Codex versions.
    resolved_codex_bin = resolve_codex_executable(codex_bin)
    root = batch_root.expanduser().resolve()
    manifest_path = root / "batch_manifest.json"
    if root.exists() and any(root.iterdir()) and not manifest_path.is_file():
        raise FileExistsError("non-empty batch root has no batch manifest")
    root.mkdir(parents=True, exist_ok=True)
    build_identity = current_build_identity(
        require_receipt=not allow_development_build
    ).as_mapping()
    if all_instances and not allow_development_build and canary_result is None:
        raise ValueError("official full SWE-EVO requires --canary-result from the exact build")
    canary = (
        _validated_canary_receipt(canary_result, build_identity=build_identity)
        if canary_result is not None
        else None
    )
    manifest = {
        "schema": "codex-longterm-v2/swe-evo-batch-manifest@1",
        "instances": list(requested),
        "dataset_path": str(dataset_path.expanduser().resolve()),
        "swe_bench_root": str(swe_bench_root.expanduser().resolve()),
        "config_path": str(config_path.expanduser().resolve()),
        "provider_api_key_envs": list(provider_api_key_envs),
        "release_manifest": (
            str(release_manifest.expanduser().resolve()) if release_manifest is not None else None
        ),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "codex_bin": resolved_codex_bin,
        "dataset_python": str(dataset_python) if dataset_python is not None else None,
        "evaluator_python": str(evaluator_python) if evaluator_python is not None else None,
        "evaluator_workers": evaluator_workers,
        "concurrency": concurrency,
        "concurrency_semantics": "MAXIMUM_WORKERS_WITH_CPU_ADMISSION_GATE",
        "max_attempts": max_attempts,
        "task_timeout_seconds": task_timeout_seconds,
        "task_timeout_semantics": "NO_SEMANTIC_PROGRESS_SAME_ATTEMPT_REFLECTION",
        "batch_timeout_seconds": batch_timeout_seconds,
        "repo_mirror_root": str(repo_mirror_root) if repo_mirror_root is not None else None,
        "skip_official": skip_official,
        "allow_development_build": allow_development_build,
        "cpu_utilization_threshold": cpu_utilization_threshold,
        "cpu_threshold_semantics": "ADMISSION_CEILING_NEW_ATTEMPTS_ONLY",
        "canary": canary,
        "build_identity": build_identity,
    }
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing_manifest, Mapping):
            raise RuntimeError("batch manifest is not a JSON object")
        immutable_keys = (
            "instances",
            "dataset_path",
            "swe_bench_root",
            "config_path",
            "provider_api_key_envs",
            "release_manifest",
            "model",
            "reasoning_effort",
            "codex_bin",
            "max_attempts",
            "repo_mirror_root",
            "skip_official",
            "allow_development_build",
            "cpu_utilization_threshold",
            "canary",
            "build_identity",
        )
        changed = []
        for key in immutable_keys:
            existing_value = existing_manifest.get(
                key,
                [] if key == "provider_api_key_envs" else None,
            )
            if existing_value != manifest.get(key):
                changed.append(key)
        if changed:
            raise RuntimeError(f"resumed batch manifest changed immutable fields: {changed}")
    else:
        _write_once(manifest_path, manifest)
    existing_result = root / "batch_result.json"
    if existing_result.is_file():
        value = json.loads(existing_result.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise RuntimeError("batch result is not a JSON object")
        return value
    common = [
        "--dataset-path",
        str(dataset_path.expanduser().resolve()),
        "--swe-bench-root",
        str(swe_bench_root.expanduser().resolve()),
        "--config",
        str(config_path.expanduser().resolve()),
        "--max-workers",
        str(evaluator_workers),
    ]
    optional_arguments = (
        ("--release-manifest", release_manifest),
        ("--model", model),
        ("--reasoning-effort", reasoning_effort),
        ("--dataset-python", dataset_python),
        ("--evaluator-python", evaluator_python),
        ("--repo-mirror-root", repo_mirror_root),
    )
    for option, value in optional_arguments:
        if value is not None:
            common.extend([option, str(value)])
    common.extend(["--codex-bin", resolved_codex_bin])
    if skip_official:
        common.append("--skip-official")
    if allow_development_build:
        common.append("--allow-development-build")
    journal = _BatchJournal(root / "batch-events.jsonl")
    deadline = time.monotonic() + batch_timeout_seconds
    temporary_root = root / "tmp"
    pause_requested = threading.Event()
    api_key_slots: queue.Queue[str] | None = None
    if provider_api_key_envs:
        api_key_slots = queue.Queue()
        for name in provider_api_key_envs:
            api_key_slots.put(name)
    cpu_admission_lock = threading.Lock()
    previous_pause_handler: object | None = None
    if threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGUSR1"):
        previous_pause_handler = signal.getsignal(signal.SIGUSR1)

        def request_pause(_signum: int, _frame: object) -> None:
            pause_requested.set()

        signal.signal(signal.SIGUSR1, request_pause)

    def admit_cpu(instance_id: str) -> bool:
        # Serialize samples so four workers cannot all observe the same idle
        # instant and stampede past the host-level admission ceiling.
        with cpu_admission_lock:
            return _wait_for_cpu_capacity(
                cpu_utilization_threshold,
                deadline=deadline,
                journal=journal,
                instance_id=instance_id,
                pause_requested=pause_requested,
            )

    def execute(instance_id: str) -> Mapping[str, object]:
        selected_key_env = api_key_slots.get() if api_key_slots is not None else None
        try:
            return _execute_one(instance_id, selected_key_env)
        finally:
            if api_key_slots is not None and selected_key_env is not None:
                api_key_slots.put(selected_key_env)

    def _execute_one(
        instance_id: str,
        selected_key_env: str | None,
    ) -> Mapping[str, object]:
        task_common = list(common)
        if selected_key_env is not None:
            task_common.extend(["--provider-api-key-env", selected_key_env])
        task_root = root / "runs" / instance_id
        task_root.mkdir(parents=True, exist_ok=True)
        if pause_requested.is_set():
            journal.record(instance_id, "QUEUED_TASK_PAUSED_BEFORE_ADMISSION")
            return {
                "instance_id": instance_id,
                "state": "SUSPENDED",
                "attempts": len(_attempt_directories(task_root)),
                "task_root": str(task_root),
                "reason": "OPERATOR_PAUSE",
            }
        completed = _completed_attempt(task_root)
        if completed is not None:
            evaluation_state = _attempt_evaluation_state(completed)
            if evaluation_state == "EVALUATION_PENDING":
                index = int(completed.name.removeprefix("attempt-"))
                remaining = int(deadline - time.monotonic())
                if remaining >= 60 and admit_cpu(instance_id):
                    _run_subprocess_attempt(
                        instance_id=instance_id,
                        attempt_root=completed,
                        task_root=task_root,
                        common_arguments=task_common,
                        timeout_seconds=task_timeout_seconds,
                        temporary_root=temporary_root / instance_id / completed.name,
                        journal=journal,
                        resume=True,
                        wall_timeout_seconds=remaining,
                        pause_requested=pause_requested,
                    )
                    evaluation_state = _attempt_evaluation_state(completed)
                journal.record(
                    instance_id,
                    (
                        "EVALUATION_RESCORED"
                        if evaluation_state != "EVALUATION_PENDING"
                        else "EVALUATION_PENDING"
                    ),
                    run_root=str(completed),
                )
                return {
                    "instance_id": instance_id,
                    "state": (
                        "EVALUATION_PENDING"
                        if evaluation_state == "EVALUATION_PENDING"
                        else "COMPLETED"
                    ),
                    "run_root": str(completed),
                    "attempt": index,
                }
            journal.record(instance_id, "SKIPPED_COMPLETED", run_root=str(completed))
            return {"instance_id": instance_id, "state": "COMPLETED", "run_root": str(completed)}
        attempts = _attempt_directories(task_root)
        recoverable_segments = 0
        semantic_stall_segments = 0
        while True:
            if pause_requested.is_set():
                journal.record(instance_id, "TASK_PAUSED_BETWEEN_SEGMENTS")
                return {
                    "instance_id": instance_id,
                    "state": "SUSPENDED",
                    "attempts": len(attempts),
                    "task_root": str(task_root),
                    "reason": "OPERATOR_PAUSE",
                }
            remaining = int(deadline - time.monotonic())
            if remaining < 60:
                journal.record(instance_id, "BATCH_DEADLINE_EXHAUSTED")
                return {
                    "instance_id": instance_id,
                    "state": "SUSPENDED",
                    "attempts": len(attempts),
                    "task_root": str(task_root),
                }
            resumable = bool(attempts) and (
                (
                    not (task_root / f"{attempts[-1].name}.json").is_file()
                    and any(task_root.glob(f"{attempts[-1].name}.segment-*.json"))
                )
                or _repairable_infrastructure_attempt(attempts[-1])
            )
            if resumable:
                attempt_root = attempts[-1]
                index = int(attempt_root.name.removeprefix("attempt-"))
            else:
                if len(attempts) >= max_attempts:
                    break
                index = len(attempts) + 1
                attempt_root = task_root / f"attempt-{index:03d}"
            if not admit_cpu(instance_id):
                return {
                    "instance_id": instance_id,
                    "state": "SUSPENDED",
                    "attempts": len(attempts),
                    "task_root": str(task_root),
                    "reason": ("OPERATOR_PAUSE" if pause_requested.is_set() else "BATCH_DEADLINE"),
                }
            receipt = _run_subprocess_attempt(
                instance_id=instance_id,
                attempt_root=attempt_root,
                task_root=task_root,
                common_arguments=task_common,
                timeout_seconds=task_timeout_seconds,
                temporary_root=temporary_root / instance_id / f"attempt-{index:03d}",
                journal=journal,
                resume=resumable,
                wall_timeout_seconds=remaining,
                pause_requested=pause_requested,
            )
            attempts = _attempt_directories(task_root)
            if bool(receipt["result_exists"]):
                evaluation_state = _attempt_evaluation_state(attempt_root)
                return {
                    "instance_id": instance_id,
                    "state": (
                        "EVALUATION_PENDING"
                        if evaluation_state == "EVALUATION_PENDING"
                        else "COMPLETED"
                    ),
                    "run_root": str(attempt_root),
                    "attempt": index,
                }
            if str(receipt.get("failure_class", "")).startswith(
                ("BENCHMARK_BASELINE", "EVALUATION_INFRASTRUCTURE")
            ):
                return {
                    "instance_id": instance_id,
                    "state": "EVALUATION_BLOCKED",
                    "attempts": len(attempts),
                    "run_root": str(attempt_root),
                    "task_root": str(task_root),
                    "reason": receipt.get("failure_class"),
                }
            if bool(receipt.get("runtime_suspended", False)):
                # A model-confirmed repository-external blocker is a durable
                # pause, not a failed Attempt and not a reason to restart from
                # the base repository.  A later Batch invocation resumes this
                # exact Attempt after the external condition can be rechecked.
                return {
                    "instance_id": instance_id,
                    "state": "SUSPENDED",
                    "attempts": len(attempts),
                    "run_root": str(attempt_root),
                    "task_root": str(task_root),
                    "reason": "MODEL_CONFIRMED_BLOCKER",
                }
            if bool(receipt.get("paused", False)):
                return {
                    "instance_id": instance_id,
                    "state": "SUSPENDED",
                    "attempts": len(attempts),
                    "run_root": str(attempt_root),
                    "task_root": str(task_root),
                    "reason": "OPERATOR_PAUSE",
                }
            if bool(receipt.get("timed_out", False)):
                if str(receipt.get("timeout_reason")) == "BATCH_DEADLINE":
                    return {
                        "instance_id": instance_id,
                        "state": "SUSPENDED",
                        "attempts": len(attempts),
                        "run_root": str(attempt_root),
                        "task_root": str(task_root),
                    }
                # A no-progress lease fences the process, but it does not erase
                # the Attempt. The next segment re-enters the same Run/WAL,
                # workspace, current Milestone and current Step.
                if int(receipt.get("semantic_transition_count", 0)) > 0:
                    semantic_stall_segments = 0
                else:
                    semantic_stall_segments += 1
                if semantic_stall_segments >= _MAX_SAME_FRONTIER_RECOVERY_SEGMENTS:
                    journal.record(
                        instance_id,
                        "SAME_ATTEMPT_SEMANTIC_STALL_PAUSED",
                        attempt=index,
                        semantic_stall_segments=semantic_stall_segments,
                        timeout_reason=receipt.get("timeout_reason"),
                        same_attempt=True,
                    )
                    return {
                        "instance_id": instance_id,
                        "state": "SUSPENDED",
                        "attempts": len(attempts),
                        "run_root": str(attempt_root),
                        "task_root": str(task_root),
                        "reason": "REPEATED_NO_SEMANTIC_PROGRESS",
                    }
                continue
            if bool(receipt.get("same_attempt_resume_eligible", False)):
                recoverable_segments += 1
                if recoverable_segments >= 3:
                    journal.record(
                        instance_id,
                        "SAME_ATTEMPT_RECOVERY_PAUSED",
                        attempt=index,
                        recoverable_segments=recoverable_segments,
                        failure_class=receipt.get("failure_class"),
                    )
                    return {
                        "instance_id": instance_id,
                        "state": "SUSPENDED",
                        "attempts": len(attempts),
                        "run_root": str(attempt_root),
                        "task_root": str(task_root),
                        "reason": "REPEATED_RETRYABLE_RUNTIME_FAILURE",
                    }
                continue
        return {
            "instance_id": instance_id,
            "state": "EXHAUSTED",
            "attempts": len(attempts),
            "task_root": str(task_root),
        }

    results: list[Mapping[str, object]] = []

    def isolated_failure(
        instance_id: str,
        exc: BaseException,
        *,
        event_name: str = "ATTEMPT_EXCEPTION_ISOLATED",
    ) -> Mapping[str, object]:
        # Never copy provider exception text into the shared Batch
        # journal/result: SDKs occasionally echo credential material.
        error_text = str(exc)
        failure = {
            "instance_id": instance_id,
            "state": "FAILED",
            "attempts": len(_attempt_directories(root / "runs" / instance_id)),
            "task_root": str(root / "runs" / instance_id),
            "error_type": type(exc).__name__,
            "error_digest": hashlib.sha256(error_text.encode("utf-8")).hexdigest(),
        }
        journal.record(
            instance_id,
            event_name,
            error_type=failure["error_type"],
            error_digest=failure["error_digest"],
        )
        return failure

    def execute_isolated(instance_id: str) -> Mapping[str, object]:
        """Keep one unexpected Attempt exception from aborting the Batch."""

        try:
            value = execute(instance_id)
            if not isinstance(value, Mapping):
                raise TypeError("Attempt result is not a mapping")
            if str(value.get("instance_id")) != instance_id:
                raise ValueError("Attempt result has the wrong instance identity")
            if not str(value.get("state", "")):
                raise ValueError("Attempt result has no state")
            return value
        except Exception as exc:
            return isolated_failure(instance_id, exc)

    try:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(requested))) as executor:
            futures = {executor.submit(execute_isolated, item): item for item in requested}
            for future in as_completed(futures):
                results.append(future.result())
    finally:
        if previous_pause_handler is not None:
            signal.signal(signal.SIGUSR1, previous_pause_handler)
    ordered = sorted(results, key=lambda item: requested.index(str(item["instance_id"])))
    complete = [item for item in ordered if item["state"] == "COMPLETED"]
    suspended = [item for item in ordered if item["state"] == "SUSPENDED"]
    evaluation_pending = [item for item in ordered if item["state"] == "EVALUATION_PENDING"]
    evaluation_blocked = [item for item in ordered if item["state"] == "EVALUATION_BLOCKED"]
    failed = [item for item in ordered if item["state"] == "FAILED"]
    official_passes = 0
    for item in complete:
        try:
            value = json.loads(
                (Path(str(item["run_root"])) / "swe_evo_result.json").read_text(encoding="utf-8")
            )
            if not isinstance(value, Mapping):
                raise TypeError("SWE-EVO result is not a JSON object")
        except Exception as exc:
            failed_item = isolated_failure(
                str(item["instance_id"]),
                exc,
                event_name="BATCH_RESULT_AGGREGATION_FAILED",
            )
            ordered[requested.index(str(item["instance_id"]))] = failed_item
            continue
        official_evaluation = value.get("official_evaluation")
        if (
            isinstance(official_evaluation, Mapping)
            and official_evaluation.get("official_pass") is True
        ):
            official_passes += 1
    complete = [item for item in ordered if item["state"] == "COMPLETED"]
    suspended = [item for item in ordered if item["state"] == "SUSPENDED"]
    evaluation_pending = [item for item in ordered if item["state"] == "EVALUATION_PENDING"]
    evaluation_blocked = [item for item in ordered if item["state"] == "EVALUATION_BLOCKED"]
    failed = [item for item in ordered if item["state"] == "FAILED"]
    result = {
        "schema": "codex-longterm-v2/swe-evo-batch-result@2",
        "requested": len(requested),
        "completed": len(complete),
        "exhausted": (
            sum(item["state"] == "EXHAUSTED" for item in ordered)
        ),
        "failed": len(failed),
        "suspended": len(suspended),
        "evaluation_pending": len(evaluation_pending),
        "evaluation_blocked": len(evaluation_blocked),
        "official_passes": official_passes,
        "results": ordered,
    }
    if not suspended and not evaluation_pending and not evaluation_blocked:
        _write_once(existing_result, result)
    return result
