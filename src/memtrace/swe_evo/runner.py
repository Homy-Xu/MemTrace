from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
from urllib.parse import urlparse

from ..build_identity import RuntimeBuildIdentity, current_build_identity
from ..config import load_config
from ..contracts import (
    Authority,
    EvidenceDraft,
    EvidenceKey,
    FactType,
    digest,
    primitive,
    stable_id,
)
from ..durability import append_synced, atomic_write_once, json_line
from ..harness import CodexHarnessAdapter, resolve_codex_executable
from ..orchestration import ExecutionResumeDirective, RunCoordinator, RunRequest
from ..orchestration.planning_coordinator import WorkspaceReadOnlyGuard
from .image_artifact import IMAGE_RECEIPT_SCHEMA, ensure_image_artifact
from .journal import BenchmarkPhaseJournal, last_phase
from .models import SweEvoInstance, build_task_text, load_instance
from .progress import latest_stall_directive, read_semantic_progress
from .workspace import (
    VERIFICATION_COMMAND,
    BenchmarkBaselineError,
    calibrate_isolated_pass_to_pass,
    collect_model_patch,
    fixture_integrity_violations,
    prepare_workspace,
    run_command,
    run_docker_tests,
)

RELEASE_MANIFEST_SCHEMA = "codex-longterm-v2/release-manifest@1"


def _write_once(path: Path, value: object) -> None:
    atomic_write_once(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _trusted_pre_execution_evidence(
    instance: SweEvoInstance,
    receipt: Mapping[str, object],
    *,
    revision_id: str,
    branch_id: str,
) -> tuple[EvidenceDraft, ...]:
    """Translate exact failed selectors from a container baseline into facts.

    The runner owns the trust boundary.  No stdout parsing, aggregate failure
    count, model assertion, or offline RunRequest field can create these facts.
    A noisy baseline may contain PASSED, ERROR, SKIPPED, or MISSING outcomes;
    those remain in the durable receipt and are never promoted to TEST_FAILURE.
    """

    raw_outcomes = receipt.get("fail_to_pass_outcomes")
    if not isinstance(raw_outcomes, Mapping):
        raise RuntimeError("SWE-EVO pre-agent receipt has no exact target outcomes")
    outcomes = {str(key): str(value) for key, value in raw_outcomes.items()}
    expected = tuple(instance.fail_to_pass)
    if set(outcomes) != set(expected):
        raise RuntimeError("SWE-EVO pre-agent receipt target set does not match FAIL_TO_PASS")
    exact_failures = tuple(target for target in expected if outcomes[target] == "FAILED")
    receipt_digest = digest(receipt)
    return tuple(
        EvidenceDraft(
            key=EvidenceKey(
                evidence_type=FactType.TEST_FAILURE,
                canonical_entity_id=f"test:{selector}",
                semantic_role="trusted_pre_execution_failure",
                revision_constraint=f"revision:{revision_id}",
                branch_scope=branch_id,
                validity_requirement="HISTORICAL_EXECUTION",
            ),
            content={
                "source_authority": "RUNTIME_TRUSTED_BENCHMARK_PREFLIGHT",
                "source_receipt": "pre_agent_test.json",
                "source_receipt_digest": receipt_digest,
                "test_selector": selector,
                "outcome": "FAILED",
                "verification_scope": "FAIL_TO_PASS",
                "logical_command": VERIFICATION_COMMAND,
            },
            # The JUnit testcase is a direct isolated observation.  Exact
            # selector binding is derived later by RunCoordinator.
            authority=Authority.ASSERTED,
            confidence=1.0,
            must_preserve=True,
        )
        for selector in exact_failures
    )


def _verified_release(
    identity: RuntimeBuildIdentity,
    manifest_path: Path | None,
    *,
    allow_development_build: bool,
) -> Mapping[str, object]:
    if not allow_development_build and identity.load_mode != "verified-installed-package":
        raise RuntimeError(
            "SWE-EVO requires execution from the freshly installed V2 wheel; "
            "development/source-tree execution is disabled"
        )
    if manifest_path is None:
        if allow_development_build:
            return {"development_override": True, **identity.as_mapping()}
        raise ValueError("SWE-EVO requires --release-manifest dist/current-build.json")
    path = manifest_path.expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != RELEASE_MANIFEST_SCHEMA:
        raise RuntimeError("release manifest schema is unsupported")
    for key in ("version", "build_id", "source_digest"):
        if str(manifest.get(key)) != str(identity.as_mapping()[key]):
            raise RuntimeError(f"release manifest {key} does not match the running package")
    wheel = path.parent / str(manifest.get("wheel", ""))
    if not wheel.is_file() or _file_sha256(wheel) != str(manifest.get("wheel_sha256")):
        raise RuntimeError("release manifest does not identify an intact exact wheel")
    return dict(manifest)


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _provider_preflight(
    config_path: Path,
    codex_bin: str | None,
    *,
    api_key_env_override: str | None = None,
) -> tuple[object, dict[str, object], str]:
    config = load_config(config_path)
    if api_key_env_override is not None:
        if _ENVIRONMENT_NAME.fullmatch(api_key_env_override) is None:
            raise ValueError("provider API key override must be an environment-variable name")
        config = replace(
            config,
            provider=replace(config.provider, api_key_env=api_key_env_override),
        )
    resolved_codex_bin = resolve_codex_executable(codex_bin)
    missing_commands = [name for name in ("git", "docker") if shutil.which(name) is None]
    if missing_commands:
        raise RuntimeError(f"SWE-EVO host commands are missing: {missing_commands}")
    key_name = config.provider.api_key_env
    if key_name and not os.environ.get(key_name):
        raise RuntimeError(f"configured Provider credential environment is absent: {key_name}")
    docker = run_command(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        cwd=config_path.parent,
        timeout_seconds=60,
        check=False,
    )
    if docker.returncode != 0:
        raise RuntimeError("Docker daemon is unavailable for SWE-EVO")
    return (
        config,
        {
            "git": shutil.which("git"),
            "docker": shutil.which("docker"),
            "codex_bin": resolved_codex_bin,
            "docker_server_version": docker.stdout.strip(),
            "provider_id": config.provider.id,
            "provider_model": config.provider.model,
            "credential_env": key_name,
            "credential_present": bool(key_name and os.environ.get(key_name)),
        },
        resolved_codex_bin,
    )


def _ensure_image(instance: SweEvoInstance, cwd: Path) -> Mapping[str, object]:
    return ensure_image_artifact(instance.image, cwd=cwd)


def _parse_official(stdout: str) -> dict[str, object]:
    resolved = re.search(r"\[Resolved rate\].*\[(\d+)/(\d+)\]", stdout)
    applied = re.search(r"\[Applied rate\].*\[(\d+)/(\d+)\]", stdout)
    fix_rate = re.search(r"\[Fix rate\]\s*=\s*([^\n]+)", stdout)
    resolved_count = int(resolved.group(1)) if resolved else None
    total = int(resolved.group(2)) if resolved else None
    return {
        "resolved": resolved_count,
        "total": total,
        "applied": int(applied.group(1)) if applied else None,
        "fix_rate": fix_rate.group(1).strip() if fix_rate else None,
        "official_pass": (
            bool(resolved_count == total)
            if total is not None and resolved_count is not None
            else None
        ),
    }


def _official_result_state(
    *,
    returncode: int,
    stdout: str,
    stderr: str,
    parsed: Mapping[str, object],
) -> Mapping[str, object]:
    """Separate a model score from evaluator/Docker infrastructure state."""

    if parsed.get("total") is not None and parsed.get("resolved") is not None:
        return {
            "state": "COMPLETED",
            "score_valid": True,
            "rescore_eligible": False,
            "error_type": None,
        }
    diagnostic = f"{stderr}\n{stdout}".lower()
    infrastructure_signatures = (
        "descriptor is neither a manifest or index",
        "cannot connect to the docker daemon",
        "docker daemon is unavailable",
        "toomanyrequests",
        "too many requests",
        "no space left on device",
        "connection reset by peer",
        "temporary failure in name resolution",
        "context deadline exceeded",
    )
    infrastructure = any(item in diagnostic for item in infrastructure_signatures)
    return {
        "state": "INFRASTRUCTURE_ERROR" if infrastructure else "EVALUATOR_ERROR",
        "score_valid": False,
        "rescore_eligible": True,
        "error_type": (
            "OfficialEvaluatorInfrastructureError" if infrastructure else "OfficialEvaluatorError"
        ),
        "evaluator_returncode": returncode,
    }


def _runtime_completion_receipt(runtime_result: object) -> dict[str, object]:
    """Expose the runtime contract as an explicit benchmark gate."""

    verdict = str(getattr(runtime_result, "completion_verdict", ""))
    task_status = str(getattr(runtime_result, "task_status", ""))
    return {
        "completion_verdict": verdict,
        "task_status": task_status,
        "accepted": verdict == "COMPLETED" and task_status == "COMPLETED",
        "unmet_completion_criteria": primitive(
            getattr(runtime_result, "unmet_completion_criteria", {})
        ),
    }


def _control_plane_score_receipt(
    runtime_completion: Mapping[str, object],
    official: Mapping[str, object],
) -> dict[str, bool]:
    """Classify control acceptance without rewriting the official patch score."""

    accepted = runtime_completion.get("accepted") is True
    return {
        "control_plane_accepted": accepted,
        "control_plane_false_negative_candidate": (
            not accepted
            and official.get("score_valid") is True
            and official.get("official_pass") is True
        ),
    }


def _configure_official_container_network(
    environment: dict[str, str],
) -> Mapping[str, object]:
    """Forward a host proxy without silently breaking Docker isolation.

    This server reaches the public package index through a loopback proxy.
    A bridge container cannot use the host's ``127.0.0.1``, so official
    scorer containers use host networking only for that explicit topology.
    Hosts with direct egress retain Docker's default bridge network.
    """

    proxy_names = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    proxy_values = tuple(
        value.strip() for name in proxy_names if (value := environment.get(name, "")).strip()
    )
    if not proxy_values:
        return {"network_mode": "bridge", "proxy_forwarded": False}
    environment["SWE_EVO_FORWARD_PROXY_ENV"] = "1"
    loopback_proxy = any(
        (urlparse(value).hostname or "").casefold() in {"127.0.0.1", "localhost", "::1"}
        for value in proxy_values
    )
    configured_mode = environment.get("SWE_EVO_CONTAINER_NETWORK_MODE", "").strip()
    if loopback_proxy and configured_mode not in {"", "host"}:
        raise RuntimeError(
            "official scorer has a loopback proxy but a non-host Docker network mode"
        )
    if loopback_proxy:
        environment["SWE_EVO_CONTAINER_NETWORK_MODE"] = "host"
    return {
        "network_mode": environment.get("SWE_EVO_CONTAINER_NETWORK_MODE", "bridge"),
        "proxy_forwarded": True,
    }


def _official_evaluation(
    instance: SweEvoInstance,
    model_patch: str,
    *,
    runtime_result_path: str,
    model: str,
    provider: str,
    reasoning_effort: str | None,
    build_identity: Mapping[str, object],
    swe_bench_root: Path,
    evaluator_python: Path,
    max_workers: int,
    artifacts_root: Path,
    image_receipt_path: Path,
) -> Mapping[str, object]:
    evaluator = swe_bench_root / "evaluate_instance.py"
    if not evaluator.is_file():
        raise FileNotFoundError(f"SWE-EVO official evaluator is missing: {evaluator}")
    root = artifacts_root.resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"official evaluator artifact root is not empty: {root}")
    output = root / "output_final"
    trajectories = root / "trajectories" / f"codex-longterm-v2-{build_identity['build_id']}"
    output.mkdir(parents=True)
    trajectories.mkdir(parents=True)
    official_row = dict(instance.official_row)
    official_row["patch"] = ""
    _write_once(output / f"{instance.instance_id}.json", official_row)
    trajectory = {
        "instance_id": instance.instance_id,
        "test_result": {"git_patch": model_patch},
        "metadata": {
            "source": "codex-longterm-v2",
            "runtime_result": runtime_result_path,
            "model": model,
            "provider": provider,
            "reasoning_effort": reasoning_effort,
            "build_identity": build_identity,
        },
    }
    atomic_write_once(
        trajectories / "output.jsonl",
        (json.dumps(trajectory, ensure_ascii=False, separators=(",", ":")) + "\n").encode(),
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(swe_bench_root), environment.get("PYTHONPATH", "")) if part
    )
    network_receipt = _configure_official_container_network(environment)
    temporary_root = (
        Path(environment.get("HOMY_SWE_EVO_TMPDIR", str(root / "tmp"))).expanduser().resolve()
    )
    temporary_root.mkdir(parents=True, exist_ok=True)
    environment["TMPDIR"] = str(temporary_root)
    completed = subprocess.run(
        [
            str(evaluator_python),
            str(evaluator),
            "--instance",
            instance.instance_id,
            "--trajectories_path",
            str(trajectories),
            "--max_workers",
            str(max_workers),
            "--scaffold",
            "OpenHands",
            "--image_receipt",
            str(image_receipt_path.resolve()),
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    atomic_write_once(root / "stdout.log", completed.stdout.encode("utf-8", errors="replace"))
    atomic_write_once(root / "stderr.log", completed.stderr.encode("utf-8", errors="replace"))
    parsed = _parse_official(completed.stdout)
    state = _official_result_state(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        parsed=parsed,
    )
    return {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-30000:],
        "stderr_tail": completed.stderr[-30000:],
        "stdout_path": str(root / "stdout.log"),
        "stderr_path": str(root / "stderr.log"),
        "artifact_root": str(root),
        "temporary_root": str(temporary_root),
        "gold_patch_written_to_run_root": False,
        "container_network": network_receipt,
        **parsed,
        **state,
    }


def _rescore_existing_result(
    *,
    existing: Mapping[str, object],
    instance_id: str,
    dataset_path: Path,
    swe_bench_root: Path,
    root: Path,
    config_path: Path,
    release_manifest: Path | None,
    model: str | None,
    reasoning_effort: str | None,
    dataset_python: Path | None,
    evaluator_python: Path | None,
    max_workers: int,
    allow_development_build: bool,
) -> Mapping[str, object]:
    """Append an official-only score receipt without rerunning the model."""

    prior = existing.get("official_evaluation")
    if not isinstance(prior, Mapping):
        return existing
    eligible = prior.get("rescore_eligible") is True or (
        prior.get("official_pass") is False and prior.get("error_type") is not None
    )
    if not eligible:
        return existing
    instance = load_instance(instance_id, dataset_path, dataset_python=dataset_python)
    identity = current_build_identity(require_receipt=not allow_development_build)
    _verified_release(
        identity,
        release_manifest,
        allow_development_build=allow_development_build,
    )
    config = load_config(config_path.expanduser().resolve())
    evaluator = (evaluator_python or Path(os.sys.executable)).expanduser().absolute()
    if not evaluator.is_file():
        raise FileNotFoundError("SWE-EVO evaluator Python is missing")
    model_patch = (root / "model.patch").read_text(encoding="utf-8")
    runtime = existing.get("runtime_result")
    if not isinstance(runtime, Mapping) or not runtime.get("result_path"):
        raise RuntimeError("existing SWE-EVO result has no runtime result path")
    ordinal = len(tuple(root.glob("official-rescore-*.json"))) + 1
    original_image_receipt = _read_json(root / "docker_image.json")
    if original_image_receipt.get("schema"):
        image_receipt = ensure_image_artifact(
            instance.image,
            cwd=root,
            expected_receipt=original_image_receipt,
        )
        image_receipt_path = root / "docker_image.json"
    else:
        # Runs produced before ImageArtifactReceipt@1 still retain the image
        # config digest. Reconstruct a full immutable receipt, but reject a
        # registry object that differs from the image used by the agent.
        image_receipt = ensure_image_artifact(instance.image, cwd=root)
        legacy_image_id = str(original_image_receipt.get("image_id", ""))
        if legacy_image_id and image_receipt.get("image_id") != legacy_image_id:
            raise RuntimeError("restored scorer image differs from the agent's legacy image ID")
        image_receipt_path = root / f"docker_image_rescore-{ordinal:03d}.json"
        _write_once(image_receipt_path, image_receipt)
    try:
        official = _official_evaluation(
            instance,
            model_patch,
            runtime_result_path=str(runtime["result_path"]),
            model=(model or str(existing.get("model", config.provider.model))),
            provider=config.provider.id,
            reasoning_effort=reasoning_effort,
            build_identity=identity.as_mapping(),
            swe_bench_root=swe_bench_root.expanduser().resolve(),
            evaluator_python=evaluator,
            max_workers=max_workers,
            artifacts_root=root / f"official-rescore-artifacts-{ordinal:03d}",
            image_receipt_path=image_receipt_path,
        )
    except Exception as exc:
        official = {
            "state": "INFRASTRUCTURE_ERROR",
            "official_pass": None,
            "score_valid": False,
            "rescore_eligible": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:4000],
        }
    receipt_path = root / f"official-rescore-{ordinal:03d}.json"
    receipt = {
        "schema": "codex-longterm-v2/swe-evo-official-rescore@1",
        "instance_id": instance_id,
        "base_result_digest": existing.get("result_digest"),
        "model_rerun": False,
        "model_patch_path": str(root / "model.patch"),
        "evaluator_build_identity": identity.as_mapping(),
        "official_evaluation": official,
        "evaluation_state": str(official.get("state", "UNKNOWN")),
    }
    _write_once(receipt_path, receipt)
    return {
        **dict(existing),
        "official_evaluation": official,
        "evaluation_state": receipt["evaluation_state"],
        "official_rescore_receipt": str(receipt_path),
    }


def _run_swe_evo_instance(
    *,
    instance_id: str,
    dataset_path: Path,
    swe_bench_root: Path,
    run_root: Path,
    config_path: Path,
    release_manifest: Path | None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    codex_bin: str | None = None,
    dataset_python: Path | None = None,
    evaluator_python: Path | None = None,
    max_workers: int = 1,
    skip_official: bool = False,
    allow_development_build: bool = False,
    repo_mirror_root: Path | None = None,
    resume: bool = False,
    provider_api_key_env: str | None = None,
) -> Mapping[str, object]:
    if max_workers < 1:
        raise ValueError("SWE-EVO max_workers must be positive")
    root = run_root.expanduser().resolve()
    existing_result = root / "swe_evo_result.json"
    if resume and existing_result.is_file():
        existing = _read_json(existing_result)
        return _rescore_existing_result(
            existing=existing,
            instance_id=instance_id,
            dataset_path=dataset_path,
            swe_bench_root=swe_bench_root,
            root=root,
            config_path=config_path,
            release_manifest=release_manifest,
            model=model,
            reasoning_effort=reasoning_effort,
            dataset_python=dataset_python,
            evaluator_python=evaluator_python,
            max_workers=max_workers,
            allow_development_build=allow_development_build,
        )
    if not resume and root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"SWE-EVO run root is not empty: {root}")
    if resume and root.exists() and not root.is_dir():
        raise FileExistsError(f"SWE-EVO run root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    journal = BenchmarkPhaseJournal(root)
    with journal.phase("PREFLIGHT"):
        if not dataset_path.expanduser().resolve().exists():
            raise FileNotFoundError(f"SWE-EVO dataset path is missing: {dataset_path}")
        resolved_swe_bench = swe_bench_root.expanduser().resolve()
        selected_evaluator_python = (
            (evaluator_python or Path(os.sys.executable)).expanduser().absolute()
        )
        if not skip_official:
            if not (resolved_swe_bench / "evaluate_instance.py").is_file():
                raise FileNotFoundError("SWE-EVO evaluate_instance.py is missing")
            if not selected_evaluator_python.is_file():
                raise FileNotFoundError("SWE-EVO evaluator Python is missing")
        identity = current_build_identity(require_receipt=not allow_development_build)
        release = _verified_release(
            identity,
            release_manifest,
            allow_development_build=allow_development_build,
        )
        resolved_config = config_path.expanduser().resolve()
        config, preflight, resolved_codex_bin = _provider_preflight(
            resolved_config,
            codex_bin,
            api_key_env_override=provider_api_key_env,
        )
        selected_model = model or config.provider.model
        selected_reasoning_effort = reasoning_effort or config.codex_reasoning_effort
        if not selected_model:
            raise ValueError("SWE-EVO requires a model")
    with journal.phase("LOAD_INSTANCE"):
        instance = load_instance(
            instance_id,
            dataset_path,
            dataset_python=dataset_python,
        )
    _write_once(root / "instance_metadata.json", instance.public_metadata())
    _write_once(root / "preflight.json", preflight)
    benchmark_manifest = {
        "schema": "codex-longterm-v2/swe-evo-run-manifest@1",
        "implementation": "memtrace",
        "instance_id": instance.instance_id,
        "model": selected_model,
        "reasoning_effort": selected_reasoning_effort,
        "codex_bin": resolved_codex_bin,
        "config_path": str(resolved_config),
        "config_digest": _file_sha256(resolved_config),
        "provider_api_key_env": getattr(config.provider, "api_key_env", None),
        "build_identity": identity.as_mapping(),
        "release": release,
        "repo_mirror_root": (
            str(repo_mirror_root.expanduser().resolve()) if repo_mirror_root is not None else None
        ),
        "gold_patch_available_to_agent": False,
        "legacy_runtime_imported": False,
    }
    _write_once(root / "benchmark_manifest.json", benchmark_manifest)
    workspace = root / "workspace"
    preparation_path = root / "workspace_preparation.json"
    if resume and preparation_path.is_file():
        preparation = _read_json(preparation_path)
        if not workspace.is_dir():
            raise RuntimeError("resumable SWE-EVO Attempt lost its prepared workspace")
        journal.record("PREPARE_WORKSPACE", "RECOVERED")
    else:
        with journal.phase("PREPARE_WORKSPACE"):
            preparation = prepare_workspace(instance, workspace, mirror_root=repo_mirror_root)
        _write_once(preparation_path, preparation)
    raw_fixture_integrity = preparation.get("fixture_integrity")
    if not isinstance(raw_fixture_integrity, Mapping):
        raise RuntimeError("SWE-EVO preparation has no immutable fixture-integrity receipt")
    image_path = root / "docker_image.json"
    active_image_path = image_path
    if resume and image_path.is_file():
        prior_image = _read_json(image_path)
        if prior_image.get("schema") == IMAGE_RECEIPT_SCHEMA:
            image = ensure_image_artifact(
                instance.image,
                cwd=root,
                expected_receipt=prior_image,
            )
            journal.record("ENSURE_IMAGE", "RECOVERED")
        else:
            # Legacy attempts recorded a mutable tag/config ID. Upgrade them
            # to the immutable artifact contract. Once Agent execution has
            # started, the exact historical image ID remains mandatory; a
            # pre-Agent infrastructure failure may safely repair it.
            recovery_paths = sorted(root.glob("docker_image_recovery-*.json"))
            if recovery_paths:
                active_image_path = recovery_paths[-1]
                image = ensure_image_artifact(
                    instance.image,
                    cwd=root,
                    expected_receipt=_read_json(active_image_path),
                )
                journal.record(
                    "ENSURE_IMAGE",
                    "LEGACY_RECEIPT_RECOVERED",
                    receipt_path=str(active_image_path),
                )
                prior_image = {}
            else:
                image = ensure_image_artifact(instance.image, cwd=root)
            legacy_image_id = str(prior_image.get("image_id", ""))
            legacy_runtime_root = root / "runtime"
            agent_execution_started = legacy_runtime_root.is_dir() and any(
                legacy_runtime_root.iterdir()
            )
            if (
                agent_execution_started
                and legacy_image_id
                and image.get("image_id") != legacy_image_id
            ):
                raise RuntimeError("resumable Agent Attempt image differs from its legacy image ID")
            if not recovery_paths:
                active_image_path = root / "docker_image_recovery-001.json"
                _write_once(active_image_path, image)
                journal.record(
                    "ENSURE_IMAGE",
                    "LEGACY_RECEIPT_REPAIRED",
                    receipt_path=str(active_image_path),
                    agent_execution_started=agent_execution_started,
                    legacy_image_id=legacy_image_id or None,
                    repaired_image_id=image.get("image_id"),
                )
    else:
        with journal.phase("ENSURE_IMAGE"):
            image = _ensure_image(instance, root)
        _write_once(image_path, image)
    before_path = root / "pre_agent_test.json"
    if resume and before_path.is_file():
        before = _read_json(before_path)
        journal.record("PRE_AGENT_TEST", "RECOVERED")
    else:
        with journal.phase("PRE_AGENT_TEST"):
            before = run_docker_tests(
                instance,
                workspace,
                junit_path=root / "pre_agent_junit.xml",
            )
            _write_once(before_path, before)
    if before["passed"]:
        raise RuntimeError("SWE-EVO FAIL_TO_PASS tests passed before the agent ran")
    raw_outcomes = before.get("fail_to_pass_outcomes", {})
    any_nonpassing = bool(before.get("any_fail_to_pass_nonpassing")) or (
        isinstance(raw_outcomes, Mapping)
        and any(str(outcome) != "PASSED" for outcome in raw_outcomes.values())
    )
    if not any_nonpassing:
        raise RuntimeError("SWE-EVO pre-agent test observed no non-passing FAIL_TO_PASS selector")
    calibration_path = root / "regression_calibration.json"
    if resume and calibration_path.is_file():
        regression_calibration = _read_json(calibration_path)
        selected_targets = regression_calibration.get("selected_targets")
        if not isinstance(selected_targets, list):
            raise RuntimeError(
                "resumable regression calibration has no exact selected_targets receipt"
            )
        calibrated_pass_to_pass = tuple(map(str, selected_targets))
        journal.record("CALIBRATE_PASS_TO_PASS", "RECOVERED")
    else:
        with journal.phase("CALIBRATE_PASS_TO_PASS"):
            calibrated_pass_to_pass, regression_calibration = calibrate_isolated_pass_to_pass(
                instance,
                workspace,
                junit_path=root / "pass_to_pass_baseline_junit.xml",
            )
        _write_once(calibration_path, regression_calibration)
    if str(regression_calibration.get("state", "READY")) != "READY":
        raise BenchmarkBaselineError(
            "no declared PASS_TO_PASS selector is stable on the untouched "
            "network-isolated baseline; see regression_calibration.json"
        )
    task = build_task_text(instance, VERIFICATION_COMMAND)
    atomic_write_once(root / "task.md", task.encode("utf-8"))
    runtime_root = root / "runtime"
    trusted_verification_root = root / "trusted-verification"
    trusted_verification_root.mkdir(exist_ok=True)
    trusted_verification_attempt = max(
        (
            int(path.stem.removeprefix("attempt-"))
            for path in trusted_verification_root.glob("attempt-*.json")
            if path.stem.removeprefix("attempt-").isdigit()
        ),
        default=0,
    )

    def trusted_verifier() -> Mapping[str, object]:
        nonlocal trusted_verification_attempt
        trusted_verification_attempt += 1
        attempt = trusted_verification_attempt
        violations = fixture_integrity_violations(workspace, raw_fixture_integrity)
        if violations:
            # The verifier contract is host-owned.  Never execute repository
            # tests after the agent changed their fixtures, since that could
            # manufacture false completion evidence.  This is a failed
            # verification observation at the current workspace revision; the
            # model can inspect its diff and restore the listed paths.
            receipt = {
                "command": VERIFICATION_COMMAND,
                "returncode": 126,
                "stdout": "",
                "stderr": (
                    "trusted verification refused because benchmark-owned fixture/test "
                    f"paths changed: {list(violations)}"
                ),
                "passed": False,
                "junit": None,
                "verification_scope": "FIXTURE_INTEGRITY_REJECTED",
                "fail_to_pass_count": len(instance.fail_to_pass),
                "pass_to_pass_count": len(calibrated_pass_to_pass),
                "fixture_integrity_violations": list(violations),
            }
        else:
            receipt = run_docker_tests(
                instance,
                workspace,
                junit_path=trusted_verification_root / f"attempt-{attempt}.xml",
                include_pass_to_pass=True,
                pass_to_pass_targets=calibrated_pass_to_pass,
            )
        _write_once(trusted_verification_root / f"attempt-{attempt}.json", receipt)
        # The model chooses no command. Expose only the immutable logical
        # verifier address; the receipt retains the exact host Docker argv.
        return {**receipt, "command": VERIFICATION_COMMAND}

    runtime_result_path = runtime_root / "result.json"
    if resume and runtime_result_path.is_file():
        runtime_result = SimpleNamespace(**_read_json(runtime_result_path))
        journal.record("AGENT_RUNTIME", "RECOVERED", result_path=str(runtime_result_path))
    else:
        with journal.phase("AGENT_RUNTIME"):
            raw_resume_directive = latest_stall_directive(root) if resume else None
            resume_directive = (
                ExecutionResumeDirective.from_mapping(raw_resume_directive)
                if raw_resume_directive is not None
                else None
            )
            adapter = CodexHarnessAdapter(
                repository_path=workspace,
                model=selected_model,
                run_root=runtime_root,
                provider=config.provider,
                executable=resolved_codex_bin,
                reasoning_effort=selected_reasoning_effort,
                sandbox_mode=config.codex_sandbox_mode,
                timeout_seconds=config.codex_timeout_seconds,
            )
            repository_id = stable_id("repo_", str(workspace))
            workspace_receipt = WorkspaceReadOnlyGuard(workspace, ()).capture()
            trusted_pre_execution_evidence = _trusted_pre_execution_evidence(
                instance,
                before,
                revision_id=workspace_receipt.revision_id,
                branch_id="main",
            )
            request = RunRequest(
                repository_path=workspace,
                repository_id=repository_id,
                run_id=stable_id(
                    "run_",
                    {
                        "benchmark": "SWE-EVO",
                        "instance": instance.instance_id,
                        "build": identity.build_id,
                        "run_root": str(root),
                    },
                ),
                branch_id="main",
                revision_id=workspace_receipt.revision_id,
                user_task=task,
                plan=None,
                actions=(),
                run_root=runtime_root,
                workspace_receipt=workspace_receipt,
                resume_directive=resume_directive,
                trusted_pre_execution_evidence=trusted_pre_execution_evidence,
            )
            try:
                runtime_result = RunCoordinator(config).run(
                    request,
                    harness_adapter=adapter,
                    trusted_verifier=trusted_verifier,
                )
            finally:
                adapter.close()
    with journal.phase("COLLECT_PATCH"):
        model_patch, changed_files = collect_model_patch(
            workspace,
            fixture_commit=str(preparation["fixture_commit"]),
            fixture_paths=tuple(str(item) for item in preparation["fixture_paths"]),
        )
    with journal.phase("RUNTIME_ACCEPTANCE"):
        runtime_completion = _runtime_completion_receipt(runtime_result)
        if runtime_completion["task_status"] == "BLOCKED":
            semantic_progress = read_semantic_progress(root).as_mapping()
            directive_id = stable_id(
                "resume_",
                {
                    "instance_id": instance.instance_id,
                    "cause": "MODEL_CONFIRMED_BLOCKER",
                    "semantic_progress": semantic_progress,
                },
            )
            directive = {
                "schema": "codex-longterm-v2/execution-resume-directive@1",
                "directive_id": directive_id,
                "state": "SUSPEND_BLOCKED",
                "cause": "MODEL_CONFIRMED_BLOCKER",
                "same_attempt": True,
                "decision_authority": "MODEL",
                "allowed_step_decisions": ["CONTINUE", "CORRECT", "BLOCKED"],
                "semantic_progress": semantic_progress,
            }
            append_synced(root / "batch-control.jsonl", json_line(directive))
            completion_path = root / f"runtime_completion-{directive_id}.json"
            patch_path = root / f"model-{directive_id}.patch"
            suspension_path = root / f"swe_evo_suspension-{directive_id}.json"
            _write_once(completion_path, runtime_completion)
            atomic_write_once(patch_path, model_patch.encode("utf-8"))
            suspension = {
                "schema": "codex-longterm-v2/swe-evo-suspension@1",
                "instance_id": instance.instance_id,
                "state": "SUSPEND_BLOCKED",
                "cause": "MODEL_CONFIRMED_BLOCKER",
                "same_attempt": True,
                "directive_id": directive_id,
                "runtime_result_path": runtime_result.result_path,
                "runtime_completion_path": str(completion_path),
                "model_patch_path": str(patch_path),
                "changed_files": list(changed_files),
                "semantic_progress": semantic_progress,
            }
            _write_once(suspension_path, suspension)
            return suspension
        _write_once(root / "runtime_completion.json", runtime_completion)
        # Runtime acceptance and benchmark patch correctness are independent
        # measurements.  A normally terminated Attempt with an unaccepted
        # control state still has a real patch that the official evaluator can
        # score.  Preserve the control-plane failure; do not convert it into a
        # model failure by skipping evaluation.
    atomic_write_once(root / "model.patch", model_patch.encode("utf-8"))
    post_path = root / "post_agent_test.json"
    if resume and post_path.is_file():
        post = _read_json(post_path)
        journal.record("POST_AGENT_TEST", "RECOVERED")
    else:
        with journal.phase("POST_AGENT_TEST"):
            post = run_docker_tests(
                instance,
                workspace,
                junit_path=root / "post_agent_junit.xml",
                include_pass_to_pass=True,
                pass_to_pass_targets=calibrated_pass_to_pass,
            )
        _write_once(post_path, post)
    official: Mapping[str, object]
    official_path = root / "official_evaluation.json"
    if resume and official_path.is_file():
        official = _read_json(official_path)
        journal.record("OFFICIAL_EVALUATION", "RECOVERED")
    elif skip_official:
        official = {
            "state": "SKIPPED",
            "official_pass": None,
            "score_valid": False,
            "rescore_eligible": False,
        }
    else:
        try:
            with journal.phase("OFFICIAL_EVALUATION"):
                official = _official_evaluation(
                    instance,
                    model_patch,
                    runtime_result_path=runtime_result.result_path,
                    model=selected_model,
                    provider=config.provider.id,
                    reasoning_effort=selected_reasoning_effort,
                    build_identity=identity.as_mapping(),
                    swe_bench_root=resolved_swe_bench,
                    evaluator_python=selected_evaluator_python,
                    max_workers=max_workers,
                    artifacts_root=root / "official-evaluator",
                    image_receipt_path=active_image_path,
                )
        except Exception as exc:
            official = {
                "state": "INFRASTRUCTURE_ERROR",
                "official_pass": None,
                "score_valid": False,
                "rescore_eligible": True,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:4000],
                "artifact_root": str(root / "official-evaluator"),
                "gold_patch_written_to_run_root": False,
            }
    _write_once(official_path, official)
    result = {
        "schema": "codex-longterm-v2/swe-evo-result@1",
        "implementation": "memtrace",
        "instance": instance.public_metadata(),
        "build_identity": identity.as_mapping(),
        "runtime_result": {
            "run_id": runtime_result.run_id,
            "result_path": runtime_result.result_path,
            "thread_id": runtime_result.thread_id,
            "completion_verdict": runtime_result.completion_verdict,
            "task_status": runtime_result.task_status,
            "completion_accepted": bool(runtime_completion["accepted"]),
            "unmet_completion_criteria": runtime_completion["unmet_completion_criteria"],
            "page_count": len(runtime_result.page_ids),
            "metrics": primitive(runtime_result.metrics),
        },
        "pre_agent_test_passed": before["passed"],
        "post_agent_test_passed": post["passed"],
        "changed_files": list(changed_files),
        "model_patch_lines": len(model_patch.splitlines()),
        "phase_durations_ms": journal.durations_ms(),
        "official_evaluation": official,
        **_control_plane_score_receipt(runtime_completion, official),
        "evaluation_state": str(official.get("state", "UNKNOWN")),
        "gold_patch_written_to_run_root": False,
        "result_digest": digest(
            {
                "runtime_result": runtime_result.context_image_digest,
                "model_patch": model_patch,
                "official": official,
            }
        ),
    }
    _write_once(root / "swe_evo_result.json", result)
    return result


def run_swe_evo_instance(
    *,
    instance_id: str,
    dataset_path: Path,
    swe_bench_root: Path,
    run_root: Path,
    config_path: Path,
    release_manifest: Path | None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    codex_bin: str | None = None,
    dataset_python: Path | None = None,
    evaluator_python: Path | None = None,
    max_workers: int = 1,
    skip_official: bool = False,
    allow_development_build: bool = False,
    repo_mirror_root: Path | None = None,
    resume: bool = False,
    provider_api_key_env: str | None = None,
) -> Mapping[str, object]:
    root = run_root.expanduser().resolve()
    try:
        return _run_swe_evo_instance(
            instance_id=instance_id,
            dataset_path=dataset_path,
            swe_bench_root=swe_bench_root,
            run_root=root,
            config_path=config_path,
            release_manifest=release_manifest,
            model=model,
            reasoning_effort=reasoning_effort,
            codex_bin=codex_bin,
            dataset_python=dataset_python,
            evaluator_python=evaluator_python,
            max_workers=max_workers,
            skip_official=skip_official,
            allow_development_build=allow_development_build,
            repo_mirror_root=repo_mirror_root,
            resume=resume,
            provider_api_key_env=provider_api_key_env,
        )
    except BaseException as exc:
        if root.is_dir():
            phase = last_phase(root)
            if isinstance(exc, BenchmarkBaselineError):
                failure_class = "BENCHMARK_BASELINE_INVALID"
                retryable = False
            elif phase in {
                "PREFLIGHT",
                "PREPARE_WORKSPACE",
                "ENSURE_IMAGE",
                "PRE_AGENT_TEST",
                "CALIBRATE_PASS_TO_PASS",
                "OFFICIAL_EVALUATION",
            }:
                failure_class = "EVALUATION_INFRASTRUCTURE"
                retryable = not isinstance(exc, (FileExistsError, ValueError))
            else:
                failure_class = "AGENT_RUNTIME"
                retryable = not isinstance(exc, (FileExistsError, ValueError))
            failure = {
                "schema": "codex-longterm-v2/swe-evo-failure@1",
                "instance_id": instance_id,
                "phase": phase,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:4000],
                "failure_class": failure_class,
                "retryable": retryable,
            }
            append_synced(root / "swe_evo-failure-events.jsonl", json_line(failure))
            failure_path = root / "swe_evo_failure.json"
            if not failure_path.exists():
                _write_once(failure_path, failure)
        raise
