from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..build_identity import package_tree_digest
from ..harness.base import HarnessBackend, HarnessCheckpoint, HarnessSession
from .receipt import build_receipt, write_receipt


def classify_failure(exc: BaseException) -> str:
    """Normalize common launcher/provider errors for public receipts."""

    explicit = getattr(exc, "failure_class", None)
    if explicit:
        return str(explicit)
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    if "auth" in name or "unauthorized" in message or "invalid token" in message:
        return "PROVIDER_AUTH_FAILURE"
    if "contextwindow" in name or "context window" in message or "too many tokens" in message:
        return "CONTEXT_LIMIT_FAILURE"
    if "docker" in name or "container" in message or "image" in message:
        return "IMAGE_CONTAINER_FAILURE"
    if "timeout" in name or "time exceeded" in message:
        return "AGENT_TIMEOUT"
    if "json" in name or "serialize" in message:
        return "INFRA_OR_AGENT_FAILURE"
    if "eval" in name or "score" in message:
        return "EVALUATOR_FAILURE"
    return type(exc).__name__


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    receipt: Mapping[str, Any]
    events: int
    checkpoint: HarnessCheckpoint


class BenchmarkRunner:
    """Single-entry runner for redacted benchmark metadata.

    The official benchmark launchers still own containers and scorers.  This
    runner owns the common lifecycle and receipt format so Codex and mini runs
    cannot silently diverge in accounting or provenance.
    """

    def __init__(self, backend: HarnessBackend, *, receipt_dir: Path) -> None:
        self.backend = backend
        self.receipt_dir = Path(receipt_dir)

    def run(
        self,
        *,
        benchmark: str,
        task_id: str,
        task: str,
        wheel_sha256: str | None = None,
        official_score: float | None = None,
        f2p: Mapping[str, int] | None = None,
        p2p: Mapping[str, int] | None = None,
        provenance: Mapping[str, Any] | None = None,
        execution_kwargs: Mapping[str, Any] | None = None,
    ) -> BenchmarkRun:
        if not isinstance(benchmark, str) or not benchmark.strip():
            raise TypeError("benchmark must be a non-empty string")
        if not isinstance(task_id, str) or not task_id.strip():
            raise TypeError("task_id must be a non-empty string")
        if not isinstance(task, str) or not task.strip():
            raise TypeError("task must be a non-empty string")
        if wheel_sha256 is not None and not isinstance(wheel_sha256, str):
            raise TypeError("wheel_sha256 must be a string when supplied")
        if provenance is not None:
            for key in ("commit", "base_sha", "image", "image_reference"):
                value = provenance.get(key)
                if value is not None and not isinstance(value, str):
                    raise TypeError(f"provenance[{key!r}] must be a string")
        run_id = f"{benchmark}:{task_id}"
        branch_id = "main"
        started = time.monotonic()
        session: HarnessSession | None = None
        status = "COMPLETED"
        failure_class = None
        event_count = 0
        try:
            session = self.backend.start_session(run_id=run_id, branch_id=branch_id)
            plan = self.backend.plan(task)
            for _event in self.backend.execute(task, **dict(execution_kwargs or {})):
                event_count += 1
            checkpoint = self.backend.checkpoint(
                revision_id="post-execution",
                thread_id=session.thread_id,
                plan_digest=plan.get("task_digest", plan.get("plan_digest")),
                branch_id=branch_id,
            )
        except Exception as exc:  # noqa: BLE001 - receipt must classify all launcher failures
            status = "FAILED"
            failure_class = classify_failure(exc)
            if session is not None:
                checkpoint = self.backend.checkpoint(
                    revision_id="failure",
                    thread_id=session.thread_id,
                    branch_id=branch_id,
                    error=str(exc),
                    failure_class=failure_class,
                )
            else:
                checkpoint = HarnessCheckpoint(
                    run_id,
                    f"{run_id}:failure",
                    "failure",
                    {"branch_id": branch_id, "error": str(exc), "failure_class": failure_class},
                )
        finally:
            try:
                self.backend.close()
            except Exception as exc:  # noqa: BLE001 - cleanup failure is part of the receipt
                if status == "COMPLETED":
                    status = "FAILED"
                    failure_class = classify_failure(exc)
        elapsed = time.monotonic() - started
        usage = self.backend.usage().as_dict()
        usage["wall_time_seconds"] = elapsed
        capabilities = self.backend.capabilities()
        session_id = session.thread_id if session is not None else None
        receipt = build_receipt(
            benchmark=benchmark,
            task_id=task_id,
            harness=capabilities.harness_name,
            harness_version=(
                "2.4.6" if capabilities.harness_name == "mini-swe-agent" else "codex-app-server"
            ),
            source_digest=package_tree_digest(),
            wheel_sha256=wheel_sha256,
            official_score=official_score,
            f2p=f2p,
            p2p=p2p,
            usage=usage,
            status=status,
            failure_class=failure_class,
            provenance={"session_id": session_id, **dict(provenance or {})},
        )
        write_receipt(self.receipt_dir / f"{benchmark}-{task_id}.json", receipt)
        return BenchmarkRun(receipt, event_count, checkpoint)
