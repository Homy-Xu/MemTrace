from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..build_identity import package_tree_digest
from ..harness.base import HarnessBackend, HarnessCheckpoint
from .receipt import build_receipt, write_receipt


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
        run_id = f"{benchmark}:{task_id}"
        branch_id = "main"
        started = time.monotonic()
        session = self.backend.start_session(run_id=run_id, branch_id=branch_id)
        status = "COMPLETED"
        failure_class = None
        event_count = 0
        try:
            plan = self.backend.plan(task)
            for _event in self.backend.execute(task, **dict(execution_kwargs or {})):
                event_count += 1
            checkpoint = self.backend.checkpoint(
                revision_id="post-execution",
                thread_id=session.thread_id,
                plan_digest=plan.get("task_digest", plan.get("plan_digest")),
                branch_id=branch_id,
            )
        except Exception as exc:
            status = "FAILED"
            failure_class = type(exc).__name__
            checkpoint = self.backend.checkpoint(
                revision_id="failure",
                thread_id=session.thread_id,
                branch_id=branch_id,
                error=str(exc),
            )
        finally:
            self.backend.close()
        elapsed = time.monotonic() - started
        usage = self.backend.usage().as_dict()
        usage["wall_time_seconds"] = elapsed
        capabilities = self.backend.capabilities()
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
            provenance={"session_id": session.thread_id, **dict(provenance or {})},
        )
        write_receipt(self.receipt_dir / f"{benchmark}-{task_id}.json", receipt)
        return BenchmarkRun(receipt, event_count, checkpoint)
