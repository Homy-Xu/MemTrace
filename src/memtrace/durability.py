from __future__ import annotations

import errno
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from .contracts import canonical_bytes, digest

SyncHook = Callable[[str, Path], None]


def sync_file(descriptor: int) -> None:
    fdatasync = getattr(os, "fdatasync", None)
    if callable(fdatasync):
        fdatasync(descriptor)
    else:
        os.fsync(descriptor)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_once(
    target: Path,
    payload: bytes,
    *,
    mode: int = 0o600,
    sync_hook: SyncHook | None = None,
) -> bool:
    """Atomically install immutable bytes without overwriting an existing path.

    A same-directory hard-link provides no-replace conflict detection on Linux
    and macOS. Existing identical content is idempotent; differing content is
    an integrity failure.
    """

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    installed = False
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            sync_file(handle.fileno())
            if sync_hook:
                sync_hook("file", temporary)
        try:
            os.link(temporary, target)
            installed = True
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            if target.read_bytes() != payload:
                raise RuntimeError(f"immutable object collision: {target}") from exc
        if installed:
            sync_directory(target.parent)
            if sync_hook:
                sync_hook("directory", target.parent)
        return installed
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def append_synced(
    path: Path,
    payload: bytes,
    *,
    sync_hook: SyncHook | None = None,
) -> tuple[int, int]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        start = os.lseek(descriptor, 0, os.SEEK_END)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        sync_file(descriptor)
        if sync_hook:
            sync_hook("wal", path)
        return start, start + len(payload)
    finally:
        os.close(descriptor)


class SecretRedactor:
    version = "v2.literal-recursive.1"

    def __init__(self, secrets: tuple[str, ...] = ()) -> None:
        self._secrets = tuple(sorted({item for item in secrets if item}, key=len, reverse=True))

    def redact(self, value: Any) -> tuple[Any, bool]:
        applied = False

        def visit(item: Any) -> Any:
            nonlocal applied
            if isinstance(item, str):
                result = item
                for secret in self._secrets:
                    if secret in result:
                        result = result.replace(secret, "[REDACTED]")
                        applied = True
                return result
            if isinstance(item, dict):
                return {str(key): visit(child) for key, child in item.items()}
            if isinstance(item, (list, tuple)):
                return [visit(child) for child in item]
            return item

        return visit(value), applied

    def proof(self, redacted_value: Any) -> str:
        return digest(
            {
                "redaction_version": self.version,
                "redacted_payload_digest": digest(redacted_value),
            }
        )


def json_line(value: Any) -> bytes:
    return canonical_bytes(value) + b"\n"


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    result = []
    with Path(path).open("rb") as handle:
        for ordinal, line in enumerate(handle, start=1):
            if not line.endswith(b"\n"):
                raise RuntimeError(f"truncated WAL line {ordinal}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"invalid WAL object at line {ordinal}")
            result.append(value)
    return result
