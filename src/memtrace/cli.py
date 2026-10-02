from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from .build_identity import current_build_identity
from .config import ConfigurationError, load_config
from .contracts import primitive, stable_id
from .harness import CodexHarnessAdapter
from .migration import LegacyImportError, LegacyPageReader
from .orchestration import RunCoordinator, RunRequest
from .orchestration.planning_coordinator import WorkspaceReadOnlyGuard

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="homy-v2",
        description="State-consistent long-running coding runtime V2",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="run with Codex App Server or an offline scenario")
    run.add_argument("--scenario", type=Path)
    task_source = run.add_mutually_exclusive_group()
    task_source.add_argument("--task")
    task_source.add_argument(
        "--task-file",
        type=Path,
        help="read the exact UTF-8 Task from a file (useful for benchmark harnesses)",
    )
    run.add_argument("--harness", choices=("codex", "scenario"))
    run.add_argument("--repository-stream", type=Path,
                     help="SWE-Milestone repository-session contract; keep trace state and MTG relations across releases")
    run.add_argument(
        "--multilang-plan", action="store_true",
        help="opt in to non-Python native Plan input compatibility",
    )
    run.add_argument(
        "--swe-milestone-verifier",
        type=Path,
        help=(
            "SWE-Milestone verifier contract JSON; installs the runtime-owned affected-scope "
            "regression guard as HOST_MANAGED Milestone acceptance (requires --multilang-plan)"
        ),
    )
    run.add_argument(
        "--task-appendix",
        type=Path,
        help=(
            "UTF-8 file appended verbatim to the Task text (benchmark runtime contract the "
            "official prompt file must not be edited to carry)"
        ),
    )
    run.add_argument(
        "--env-file",
        type=Path,
        action="append",
        default=[],
        help=(
            "private NAME=value file to load before the Provider starts; repeat for a provider "
            "file and a proxy file (the file is never executed)"
        ),
    )
    run.add_argument(
        "--provider-api-key-env",
        help="override the configured Provider credential variable name",
    )
    run.add_argument("--model")
    run.add_argument(
        "--codex-bin",
        help=(
            "exact Codex Runtime path; otherwise use HOMY_CODEX_BIN, the pinned "
            "codex_cli_bin dependency, or PATH"
        ),
    )
    run.add_argument(
        "--reasoning-effort",
        help=(
            "Codex model effort advertised by the selected model; for current Codex App Server "
            "this may also be 'ultra' to request proactive multi-agent execution"
        ),
    )
    run.add_argument("--resume-thread-id")
    benchmark = subcommands.add_parser(
        "benchmark", help="run an offline benchmark through the shared memory runtime"
    )
    benchmark.add_argument("--scenario", type=Path, required=True)
    for command in (run, benchmark):
        command.add_argument("--repository", type=Path, required=True)
        command.add_argument("--run-root", type=Path)
        command.add_argument("--config", type=Path)
        command.add_argument("--legacy-page-root", type=Path)
        command.add_argument("--legacy-run-id")
        command.add_argument("--legacy-branch-id", default="main")

    validate = subcommands.add_parser(
        "validate-config", help="validate runtime dependencies before Planning"
    )
    validate.add_argument("--config", type=Path)
    validate.add_argument(
        "--env-file",
        type=Path,
        action="append",
        default=[],
        help="private NAME=value file to load before checking credential presence (repeatable)",
    )
    validate.add_argument("--provider-api-key-env")

    inspect = subcommands.add_parser("inspect", help="read an immutable V2 result")
    inspect.add_argument("--run-root", type=Path, required=True)
    subcommands.add_parser("runtime-info", help="print the executable V2 build identity")
    swe_evo = subcommands.add_parser(
        "swe-evo", help="run one SWE-EVO instance through the production V2 runtime"
    )
    swe_evo.add_argument("--instance-id", required=True)
    swe_evo.add_argument("--dataset-path", type=Path, required=True)
    swe_evo.add_argument("--swe-bench-root", type=Path, required=True)
    swe_evo.add_argument("--run-root", type=Path, required=True)
    swe_evo.add_argument("--config", type=Path, required=True)
    swe_evo.add_argument("--release-manifest", type=Path)
    swe_evo.add_argument("--model")
    swe_evo.add_argument("--reasoning-effort")
    swe_evo.add_argument("--codex-bin")
    swe_evo.add_argument(
        "--env-file",
        type=Path,
        action="append",
        default=[],
        help="private NAME=value file to load before the Provider starts (repeatable)",
    )
    swe_evo.add_argument("--dataset-python", type=Path)
    swe_evo.add_argument("--evaluator-python", type=Path)
    swe_evo.add_argument(
        "--provider-api-key-env",
        help="environment-variable name holding this Attempt's Provider API key (never a literal key)",
    )
    swe_evo.add_argument("--max-workers", type=int, default=1)
    swe_evo.add_argument("--repo-mirror-root", type=Path)
    swe_evo.add_argument("--skip-official", action="store_true")
    swe_evo.add_argument(
        "--resume",
        action="store_true",
        help="resume the same durable SWE-EVO Attempt, workspace, Run and Milestone",
    )
    swe_evo.add_argument(
        "--allow-development-build",
        action="store_true",
        help="test-only override; official measurements must use the exact wheel manifest",
    )
    swe_batch = subcommands.add_parser(
        "swe-evo-batch",
        help="run a resumable, bounded set of SWE-EVO instances through the production V2 runtime",
    )
    swe_batch.add_argument("--instance-id", action="append", default=[])
    swe_batch.add_argument("--all-instances", action="store_true")
    swe_batch.add_argument("--dataset-path", type=Path, required=True)
    swe_batch.add_argument("--swe-bench-root", type=Path, required=True)
    swe_batch.add_argument("--batch-root", type=Path, required=True)
    swe_batch.add_argument("--config", type=Path, required=True)
    swe_batch.add_argument("--release-manifest", type=Path)
    swe_batch.add_argument("--model")
    swe_batch.add_argument("--reasoning-effort")
    swe_batch.add_argument("--codex-bin")
    swe_batch.add_argument(
        "--env-file",
        type=Path,
        action="append",
        default=[],
        help="private NAME=value file to load before the Provider starts (repeatable)",
    )
    swe_batch.add_argument("--dataset-python", type=Path)
    swe_batch.add_argument("--evaluator-python", type=Path)
    swe_batch.add_argument(
        "--provider-api-key-env",
        action="append",
        default=[],
        help="repeatable environment-variable names used as a deterministic per-task API-key pool",
    )
    swe_batch.add_argument("--evaluator-workers", type=int, default=1)
    swe_batch.add_argument("--concurrency", type=int, default=2)
    swe_batch.add_argument(
        "--canary-result",
        type=Path,
        help="completed swe_evo_result.json from this exact build; required for official full runs",
    )
    swe_batch.add_argument("--max-attempts", type=int, default=2)
    swe_batch.add_argument(
        "--task-timeout-seconds",
        type=int,
        default=14_400,
        help=(
            "maximum interval without durable runtime/log progress; active long tasks are not "
            "stopped merely because total wall time is large"
        ),
    )
    swe_batch.add_argument("--batch-timeout-seconds", type=int, default=172_800)
    swe_batch.add_argument("--repo-mirror-root", type=Path)
    swe_batch.add_argument("--skip-official", action="store_true")
    swe_batch.add_argument("--allow-development-build", action="store_true")
    swe_batch.add_argument(
        "--cpu-utilization-threshold",
        type=float,
        help=(
            "defer starting a new Attempt while host CPU utilization is at or above this "
            "percentage; running Attempts are never interrupted"
        ),
    )
    return parser


def _load_scenario(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid scenario JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("scenario root must be an object")
    return value


def _default_run_root(repository: Path, label: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return repository.expanduser().resolve() / ".homy-v2" / "runs" / f"{label}-{timestamp}"


def _load_cli_environment_files(args: argparse.Namespace) -> None:
    paths = tuple(getattr(args, "env_file", ()) or ())
    if not paths:
        return
    from .harness.environment import apply_environment_files

    apply_environment_files(paths)


def _override_provider_key_env(config, override: str | None):
    if override is None:
        return config
    name = override.strip()
    if not name:
        raise ValueError("provider API key override must be non-empty")
    if _ENVIRONMENT_NAME.fullmatch(name) is None:
        raise ValueError("provider API key override must be an environment-variable name")
    return replace(
        config,
        provider=replace(config.provider, api_key_env=name),
        redaction_env_vars=tuple(dict.fromkeys((*config.redaction_env_vars, name))),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _load_cli_environment_files(args)
        if args.command == "validate-config":
            config = load_config(args.config)
            config = _override_provider_key_env(config, args.provider_api_key_env)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "schema_version": config.schema_version,
                        "runtime_lifecycle": "shared",
                        "rich_graph": config.stages.rich_graph,
                        "provider": {
                            "id": config.provider.id,
                            "model": config.provider.model,
                            "api_key_env": config.provider.api_key_env,
                            "api_key_present": bool(
                                config.provider.api_key_env
                                and os.environ.get(config.provider.api_key_env)
                            ),
                        },
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "inspect":
            result = args.run_root.expanduser().resolve() / "result.json"
            print(result.read_text(encoding="utf-8"))
            return 0
        if args.command == "runtime-info":
            print(json.dumps(current_build_identity().as_mapping(), sort_keys=True))
            return 0
        if args.command == "swe-evo":
            from .swe_evo import run_swe_evo_instance

            result = run_swe_evo_instance(
                instance_id=args.instance_id,
                dataset_path=args.dataset_path,
                swe_bench_root=args.swe_bench_root,
                run_root=args.run_root,
                config_path=args.config,
                release_manifest=args.release_manifest,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                codex_bin=args.codex_bin,
                dataset_python=args.dataset_python,
                evaluator_python=args.evaluator_python,
                max_workers=args.max_workers,
                repo_mirror_root=args.repo_mirror_root,
                skip_official=args.skip_official,
                allow_development_build=args.allow_development_build,
                resume=args.resume,
                provider_api_key_env=args.provider_api_key_env,
            )
            suspended = str(result.get("state", "")).startswith("SUSPEND_")
            payload = {
                "instance_id": args.instance_id,
                "run_root": str(args.run_root.expanduser().resolve()),
                "state": str(result.get("state", "COMPLETED")),
                "post_agent_test_passed": result.get("post_agent_test_passed"),
                "official_evaluation": result.get("official_evaluation"),
                "build_identity": result.get("build_identity"),
            }
            if suspended:
                payload.update(
                    {
                        "cause": result.get("cause"),
                        "same_attempt": result.get("same_attempt"),
                        "directive_id": result.get("directive_id"),
                    }
                )
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return 5 if suspended else 0
        if args.command == "swe-evo-batch":
            from .swe_evo import run_swe_evo_batch

            result = run_swe_evo_batch(
                dataset_path=args.dataset_path,
                swe_bench_root=args.swe_bench_root,
                batch_root=args.batch_root,
                config_path=args.config,
                release_manifest=args.release_manifest,
                instance_ids=args.instance_id,
                all_instances=args.all_instances,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                codex_bin=args.codex_bin,
                dataset_python=args.dataset_python,
                evaluator_python=args.evaluator_python,
                evaluator_workers=args.evaluator_workers,
                concurrency=args.concurrency,
                max_attempts=args.max_attempts,
                task_timeout_seconds=args.task_timeout_seconds,
                batch_timeout_seconds=args.batch_timeout_seconds,
                repo_mirror_root=args.repo_mirror_root,
                skip_official=args.skip_official,
                allow_development_build=args.allow_development_build,
                cpu_utilization_threshold=args.cpu_utilization_threshold,
                canary_result=args.canary_result,
                provider_api_key_envs=args.provider_api_key_env,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            if int(result.get("suspended", 0)):
                return 5
            if int(result.get("evaluation_pending", 0)) or int(result.get("evaluation_blocked", 0)):
                return 6
            return 0 if int(result["exhausted"]) == 0 else 4
        if args.command in {"run", "benchmark"}:
            # Config is parsed and cross-stage validated before RunRequest or
            # any Planning/Registry object is created.
            config = load_config(args.config)
            if args.command == "run":
                config = _override_provider_key_env(config, args.provider_api_key_env)
            repository = args.repository.expanduser().resolve()
            harness = (
                "scenario"
                if args.command == "benchmark"
                else (args.harness or ("scenario" if args.scenario else None))
            )
            if harness not in {"scenario", "codex"}:
                raise ValueError("run requires --harness codex or --scenario")
            label = args.scenario.stem if args.scenario else "codex"
            run_root = (
                args.run_root.expanduser().resolve()
                if args.run_root
                else _default_run_root(repository, label)
            )
            harness_adapter = None
            trusted_verifier = None
            coordinator_options: dict[str, object] = {}
            adapter_class = CodexHarnessAdapter
            if harness == "scenario":
                if args.scenario is None:
                    raise ValueError("scenario harness requires --scenario")
                scenario = _load_scenario(args.scenario)
                request = RunRequest.from_mapping(
                    scenario,
                    repository_path=repository,
                    run_root=run_root,
                )
            else:
                if args.scenario is not None:
                    raise ValueError("--scenario cannot be combined with --harness codex")
                user_task = args.task
                if args.task_file is not None:
                    try:
                        user_task = (
                            args.task_file.expanduser().resolve().read_text(encoding="utf-8")
                        )
                    except OSError as exc:
                        raise ValueError(f"cannot read Task file: {args.task_file}") from exc
                if args.task_appendix is not None:
                    try:
                        appendix = args.task_appendix.expanduser().resolve().read_text(
                            encoding="utf-8"
                        )
                    except OSError as exc:
                        raise ValueError(
                            f"cannot read Task appendix: {args.task_appendix}"
                        ) from exc
                    if appendix.strip():
                        user_task = (user_task or "").rstrip() + "\n\n" + appendix.strip() + "\n"
                selected_model = args.model or config.provider.model
                selected_reasoning_effort = args.reasoning_effort or config.codex_reasoning_effort
                if not user_task or not user_task.strip() or not selected_model:
                    raise ValueError(
                        "Codex harness requires --task/--task-file and a model from "
                        "--model or provider.model"
                    )
                normalizer_options = {}
                if args.swe_milestone_verifier is not None and args.multilang_plan:
                    from .harness.multilang_normalizer import SweMilestonePlanNormalizer
                    from .multilang_processor import process_frontier_file
                    from .multilang_references import MultilangReferenceDirectory
                    from .swe_milestone import (
                        SweMilestoneCodexHarnessAdapter,
                        SweMilestoneVerifier,
                        load_contract,
                    )

                    contract = load_contract(args.swe_milestone_verifier.expanduser().resolve())
                    normalizer_options["normalizer"] = SweMilestonePlanNormalizer()
                    trusted_verifier = SweMilestoneVerifier(repository, contract, run_root)
                    adapter_class = SweMilestoneCodexHarnessAdapter
                    coordinator_options = {
                        "rich_processor": process_frontier_file,
                        "reference_directory_factory": MultilangReferenceDirectory,
                    }
                elif args.swe_milestone_verifier is not None:
                    # Python repository streams keep the native planner, AST
                    # and ReferenceDirectory; only the host-managed regression
                    # guard is attached.  Outside a repository stream the guard
                    # still needs the multilang planner's plan normalizer.
                    if args.repository_stream is None:
                        raise ValueError(
                            "--swe-milestone-verifier requires --multilang-plan or --repository-stream"
                        )
                    from .swe_milestone import SweMilestoneVerifier, load_contract

                    contract = load_contract(args.swe_milestone_verifier.expanduser().resolve())
                    trusted_verifier = SweMilestoneVerifier(repository, contract, run_root)
                elif args.multilang_plan:
                    from .harness.multilang_normalizer import MultilangPlanNormalizer
                    normalizer_options["normalizer"] = MultilangPlanNormalizer()
                if args.repository_stream is not None:
                    from .swe_milestone.adapter import SweMilestoneCodexHarnessAdapter
                    from .swe_milestone.repository_stream import RepositoryStream
                    coordinator_options["repository_stream"] = RepositoryStream(
                        args.repository_stream, repository)
                    # Expose the same optional navigation tool to Python too;
                    # retain Python AST/ReferenceDirectory and normalizer.
                    adapter_class = SweMilestoneCodexHarnessAdapter
                harness_adapter = adapter_class(
                    repository_path=repository,
                    model=selected_model,
                    run_root=run_root,
                    provider=config.provider,
                    executable=args.codex_bin,
                    timeout_seconds=config.codex_timeout_seconds,
                    reasoning_effort=selected_reasoning_effort,
                    sandbox_mode=config.codex_sandbox_mode,
                    **normalizer_options,
                )
                repository_id = stable_id("repo_", str(repository))
                workspace_receipt = WorkspaceReadOnlyGuard(
                    repository,
                    (run_root,),
                ).capture()
                revision_id = workspace_receipt.revision_id
                run_id = stable_id(
                    "run_",
                    {
                        "repository": repository_id,
                        "task": user_task,
                        "model": selected_model,
                        "reasoning_effort": selected_reasoning_effort,
                        "run_root": str(run_root),
                        **({"repository_stream": coordinator_options["repository_stream"].contract}
                           if args.repository_stream is not None else {}),
                    },
                )
                request = RunRequest(
                    repository_path=repository,
                    repository_id=repository_id,
                    run_id=run_id,
                    branch_id="main",
                    revision_id=revision_id,
                    user_task=user_task,
                    plan=None,
                    actions=(),
                    run_root=run_root,
                    harness_thread_id=args.resume_thread_id,
                    workspace_receipt=workspace_receipt,
                )
            if args.legacy_page_root is not None:
                if harness != "scenario":
                    raise ValueError("legacy trace import is only available with scenario harness")
                if not args.legacy_run_id:
                    raise ValueError("--legacy-run-id is required with --legacy-page-root")
                imported = LegacyPageReader(args.legacy_page_root).read(
                    run_id=args.legacy_run_id,
                    branch_id=args.legacy_branch_id,
                )
                request = replace(
                    request,
                    actions=(*imported.as_actions(), *request.actions),
                )
            run_options = {"harness_adapter": harness_adapter}
            if trusted_verifier is not None:
                # Only the SWE-Milestone entry configures a host verifier; the
                # frozen Python entry keeps its exact coordinator call.
                run_options["trusted_verifier"] = trusted_verifier
            try:
                result = RunCoordinator(config, **coordinator_options).run(
                    request,
                    **run_options,
                )
            finally:
                if harness_adapter is not None:
                    harness_adapter.close()
            print(
                json.dumps(
                    {
                        "run_id": result.run_id,
                        "entry_kind": args.command,
                        "thread_id": result.thread_id,
                        "epoch_id": result.epoch_id,
                        "page_count": len(result.page_ids),
                        "result_path": result.result_path,
                        "metrics": primitive(result.metrics),
                        "task_status": result.task_status,
                            "final_review_disposition": getattr(
                                result, "final_review_disposition", "COMPLETE"
                            ),
                        "completion_verdict": result.completion_verdict,
                        "milestone_statuses": primitive(result.milestone_statuses),
                        "unmet_completion_criteria": primitive(result.unmet_completion_criteria),
                        "build_identity": primitive(result.build_identity),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            # A durable runtime result is a successful process handoff even
            # when the semantic task verdict is INCOMPLETE or TARGETED_VERIFY.
            # The official evaluator must be allowed to inspect the same
            # workspace/patch and distinguish model incompleteness from an
            # infrastructure failure. Returning a non-zero code here turns a
            # normal semantic outcome into Pier's infrastructure-error path.
            return 0
    except (
        ConfigurationError,
        FileExistsError,
        LegacyImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"homy-v2: {exc}", file=sys.stderr)
        return 2
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
