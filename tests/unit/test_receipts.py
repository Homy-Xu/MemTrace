from __future__ import annotations

import json
from pathlib import Path

import pytest

from memtrace.benchmarks.receipt import ReceiptError, build_receipt, write_receipt


def test_receipt_redacts_keys_and_private_paths(tmp_path: Path) -> None:
    receipt = build_receipt(
        benchmark="smoke",
        task_id="task-1",
        harness="mini-swe-agent",
        harness_version="2.4.6",
        source_digest="source",
        wheel_sha256=None,
        official_score=None,
        f2p=None,
        p2p=None,
        usage={
            "cost": 0.25,
            "detail": "".join(("sk", "-")) + ("x" * 24) + " /home/private",
        },
        status="COMPLETED",
    )
    out = tmp_path / "receipt.json"
    write_receipt(out, receipt)
    data = json.loads(out.read_text())
    assert "sk-" + ("x" * 24) not in json.dumps(data)
    assert "/home/private" not in json.dumps(data)
    with pytest.raises(ReceiptError):
        write_receipt(out, receipt)


def test_receipt_converts_private_launcher_scalars_and_preserves_unknown_cost() -> None:
    receipt = build_receipt(
        benchmark="smoke",
        task_id="task-1",
        harness="mini-swe-agent",
        harness_version="2.4.6",
        source_digest="source",
        wheel_sha256=None,
        official_score=None,
        f2p=None,
        p2p=None,
        usage={"cost": None, "cost_available": False, "path": Path("/work/projects/secret")},
        status="FAILED",
        provenance={"launcher_path": Path("/work/projects/secret/runner.py")},
    )
    encoded = json.dumps(receipt)
    assert "PosixPath" not in encoded
    assert "<private-path>" in encoded
    assert receipt["usage"]["cost"] is None
