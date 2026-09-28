from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from . import __version__
from .contracts import stable_id

BUILD_RECEIPT_SCHEMA = "memtrace/build-receipt@1"
BUILD_IDENTITY_SCHEMA = "memtrace/runtime-build-identity@1"
_RECEIPT_NAME = "_build_receipt.json"


def package_tree_digest(package_root: Path | None = None) -> str:
    """Hash executable package sources; generated receipts never hash themselves."""

    root = (package_root or Path(__file__).resolve().parent).resolve()
    records: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        records.append(
            (
                path.relative_to(root).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def expected_build_receipt(package_root: Path | None = None) -> dict[str, object]:
    source_digest = package_tree_digest(package_root)
    return {
        "schema": BUILD_RECEIPT_SCHEMA,
        "distribution": "memtrace",
        "version": __version__,
        "source_digest": source_digest,
        "build_id": stable_id(
            "build_",
            {"version": __version__, "source_digest": source_digest},
        ),
    }


@dataclass(frozen=True, slots=True)
class RuntimeBuildIdentity:
    schema: str
    distribution: str
    version: str
    build_id: str
    source_digest: str
    load_mode: str
    receipt_verified: bool
    codex_cli_bin_version: str | None

    def as_mapping(self) -> Mapping[str, object]:
        return asdict(self)


def current_build_identity(*, require_receipt: bool = False) -> RuntimeBuildIdentity:
    root = Path(__file__).resolve().parent
    installed_package = any(part in {"site-packages", "dist-packages"} for part in root.parts)
    expected = expected_build_receipt(root)
    receipt_path = root / _RECEIPT_NAME
    receipt_verified = False
    if receipt_path.exists():
        try:
            actual = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("installed V2 build receipt is unreadable") from exc
        if actual != expected:
            if require_receipt or installed_package:
                raise RuntimeError("installed V2 build receipt does not match executable sources")
        else:
            receipt_verified = True
    elif require_receipt:
        raise RuntimeError(
            "benchmark execution requires an installed, verified V2 build receipt; "
            "run scripts/build_release.py and install the wheel"
        )
    try:
        codex_version = importlib.metadata.version("openai-codex-cli-bin")
    except importlib.metadata.PackageNotFoundError:
        codex_version = None
    return RuntimeBuildIdentity(
        schema=BUILD_IDENTITY_SCHEMA,
        distribution=str(expected["distribution"]),
        version=str(expected["version"]),
        build_id=str(expected["build_id"]),
        source_digest=str(expected["source_digest"]),
        load_mode=(
            "verified-installed-package"
            if receipt_verified and installed_package
            else "verified-source-tree"
            if receipt_verified
            else "development-source"
        ),
        receipt_verified=receipt_verified,
        codex_cli_bin_version=codex_version,
    )
