from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..contracts import digest
from ..orchestration.models import AgentAction


class LegacyImportError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LegacyAuditPage:
    page_id: str
    page_seq: int
    payload_digest: str
    events: tuple[Mapping[str, Any], ...]
    source_manifest: str


@dataclass(frozen=True, slots=True)
class LegacyImportReceipt:
    source_root: str
    source_fingerprint: str
    run_id: str
    branch_id: str
    pages: tuple[LegacyAuditPage, ...]

    def as_actions(self) -> tuple[AgentAction, ...]:
        """Preserve legacy facts as audit payloads without inventing V2 Evidence."""

        return tuple(
            AgentAction(
                action_id=f"legacy-import:{page.page_id}",
                action_type="LEGACY_AUDIT_IMPORT",
                content=json.dumps(
                    {
                        "schema": "codex-longterm-v2/legacy-audit-import@1",
                        "authority": "LEGACY_AUDIT_ONLY_NOT_V2_EVIDENCE",
                        "source_manifest": page.source_manifest,
                        "source_payload_digest": page.payload_digest,
                        "events": page.events,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                execution_phase="legacy_read_only_import",
                semantic_boundary=True,
            )
            for page in self.pages
        )


class LegacyPageReader:
    """Read V1 exported immutable Pages without importing or executing V1 code."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise LegacyImportError(f"legacy Page root is not a directory: {root}")

    def _fingerprint(self) -> str:
        values: list[tuple[str, int, int]] = []
        for path in sorted(self.root.rglob("*")):
            if path.is_symlink():
                raise LegacyImportError(
                    f"legacy source contains a symlink: {path.relative_to(self.root)}"
                )
            if not path.is_file():
                continue
            stat = path.stat()
            values.append((path.relative_to(self.root).as_posix(), stat.st_size, stat.st_mtime_ns))
        return digest(values)

    def read(self, *, run_id: str, branch_id: str) -> LegacyImportReceipt:
        before = self._fingerprint()
        manifest_root = (
            self.root
            / "runs"
            / self._safe_component(run_id)
            / "branches"
            / self._safe_component(branch_id)
            / "manifests"
        )
        if not manifest_root.is_dir():
            raise LegacyImportError(f"legacy manifest branch does not exist: {run_id}/{branch_id}")
        pages: list[LegacyAuditPage] = []
        expected_seq = 0
        for path in sorted(manifest_root.glob("page-*.json")):
            value = self._json_object(path.read_bytes(), path)
            page = value.get("page")
            if not isinstance(page, dict):
                raise LegacyImportError(f"legacy manifest lacks page object: {path}")
            if str(page.get("run_id")) != run_id or str(page.get("branch_id")) != branch_id:
                raise LegacyImportError(f"legacy manifest scope mismatch: {path}")
            page_seq = int(page.get("page_seq", -1))
            if page_seq != expected_seq:
                raise LegacyImportError(
                    f"legacy Page sequence gap: expected {expected_seq}, got {page_seq}"
                )
            expected_seq += 1
            payload_digest = str(page.get("payload_digest", ""))
            events = self._read_object(payload_digest, int(page.get("event_count", -1)))
            pages.append(
                LegacyAuditPage(
                    page_id=str(page["page_id"]),
                    page_seq=page_seq,
                    payload_digest=payload_digest,
                    events=events,
                    source_manifest=path.relative_to(self.root).as_posix(),
                )
            )
        after = self._fingerprint()
        if after != before:
            raise LegacyImportError("legacy source changed during read-only import")
        return LegacyImportReceipt(
            source_root=str(self.root),
            source_fingerprint=after,
            run_id=run_id,
            branch_id=branch_id,
            pages=tuple(pages),
        )

    def _read_object(
        self, payload_digest: str, expected_events: int
    ) -> tuple[Mapping[str, Any], ...]:
        algorithm, separator, hexadecimal = payload_digest.partition(":")
        if (
            algorithm != "sha256"
            or separator != ":"
            or len(hexadecimal) != 64
            or any(character not in "0123456789abcdef" for character in hexadecimal)
        ):
            raise LegacyImportError(f"invalid legacy payload digest: {payload_digest}")
        object_root = (self.root / "objects" / "sha256").resolve()
        directory = (object_root / hexadecimal[:2] / hexadecimal).resolve()
        if object_root not in directory.parents or directory.is_symlink():
            raise LegacyImportError("legacy object path escapes immutable object root")
        events_path = directory / "events.jsonl"
        descriptor_path = directory / "descriptor.json"
        if events_path.is_symlink() or descriptor_path.is_symlink():
            raise LegacyImportError("legacy immutable object contains a symlink")
        payload = events_path.read_bytes()
        if "sha256:" + hashlib.sha256(payload).hexdigest() != payload_digest:
            raise LegacyImportError("legacy immutable Page payload digest mismatch")
        descriptor = self._json_object(descriptor_path.read_bytes(), descriptor_path)
        lines = payload.splitlines()
        expected = {
            "payload_digest": payload_digest,
            "size_bytes": len(payload),
            "event_count": len(lines),
        }
        if any(descriptor.get(key) != value for key, value in expected.items()):
            raise LegacyImportError("legacy immutable object descriptor mismatch")
        if expected_events != len(lines):
            raise LegacyImportError("legacy manifest/object event count mismatch")
        events: list[Mapping[str, Any]] = []
        for ordinal, line in enumerate(lines, start=1):
            value = self._json_object(line, events_path)
            if int(value.get("schema_version", -1)) != 1:
                raise LegacyImportError(f"unsupported legacy event schema at line {ordinal}")
            events.append(value)
        return tuple(events)

    @staticmethod
    def _json_object(payload: bytes, source: Path) -> dict[str, Any]:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LegacyImportError(f"invalid legacy JSON: {source}") from exc
        if not isinstance(value, dict):
            raise LegacyImportError(f"legacy JSON root must be an object: {source}")
        return value

    @staticmethod
    def _safe_component(value: str) -> str:
        if (
            not value
            or value in {".", ".."}
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for character in value
            )
        ):
            raise LegacyImportError(f"unsafe legacy scope component: {value!r}")
        return value
