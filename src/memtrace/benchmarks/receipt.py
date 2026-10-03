"""Append-only, privacy-safe benchmark receipts."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "memtrace/benchmark-receipt@1"
_SECRET = re.compile(r"(?i)(?:sk-[A-Za-z0-9_-]{12,}|bearer\s+[A-Za-z0-9._-]{12,})")
_PRIVATE_PATH = re.compile(r"(?:/home/[^\s/]+|/work/projects/[^\s]+|/raid/[^\s]+)")


class ReceiptError(RuntimeError):
    pass


def redact(value: Any) -> Any:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, Mapping):
        return {str(k): redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = _SECRET.sub("<redacted>", value)
        return _PRIVATE_PATH.sub("<private-path>", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    # Receipts must remain JSON-safe even when a private launcher passes an
    # enum, UUID, or other scalar-like object in provenance.
    return str(value)


def build_receipt(
    *,
    benchmark: str,
    task_id: str,
    harness: str,
    harness_version: str,
    source_digest: str,
    wheel_sha256: str | None,
    official_score: float | None,
    f2p: Mapping[str, int] | None,
    p2p: Mapping[str, int] | None,
    usage: Mapping[str, Any],
    status: str,
    failure_class: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": RECEIPT_SCHEMA,
        "benchmark": benchmark,
        "task_id": task_id,
        "harness": harness,
        "harness_version": harness_version,
        "source_digest": source_digest,
        "wheel_sha256": wheel_sha256,
        "official_score": official_score,
        "f2p": dict(f2p) if f2p is not None else None,
        "p2p": dict(p2p) if p2p is not None else None,
        "usage": dict(usage),
        "status": status,
        "failure_class": failure_class,
        "generation": {"status": status, "failure_class": failure_class},
        "evaluation": {
            "status": "SCORED" if official_score is not None else "PENDING",
            "official": official_score is not None,
        },
        "score": {"official": official_score},
        "runtime": {"status": status, "failure_class": failure_class},
        "provenance": dict(provenance or {}),
    }
    return redact(payload)


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Write a receipt atomically and refuse replacement of an existing one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ReceiptError(f"receipt already exists: {path}")
    content = json.dumps(redact(receipt), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
