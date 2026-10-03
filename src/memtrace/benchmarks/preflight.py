"""Fail-closed checks shared by public benchmark launchers.

The public package does not know a cluster's scheduler or Docker layout.  It
does, however, validate the invariants that must hold before a model call is
made: a provider configuration is complete, the selected image can be
resolved, and the run/scratch roots have enough independent space.
"""
from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .image import ImageResolution, resolve_docker_image


class PreflightError(RuntimeError):
    """A required launch invariant was not satisfied."""

    failure_class = "PREFLIGHT_FAILURE"


@dataclass(frozen=True, slots=True)
class StorageCheck:
    path: str
    available_bytes: int
    required_bytes: int

    def as_dict(self) -> dict[str, int | str | bool]:
        return {
            "path": self.path,
            "available_bytes": self.available_bytes,
            "required_bytes": self.required_bytes,
            "sufficient": self.available_bytes >= self.required_bytes,
        }


def check_storage(path: Path, *, minimum_free_bytes: int = 1_073_741_824) -> StorageCheck:
    """Check free space without creating or deleting files."""

    target = Path(path).expanduser()
    probe = target if target.exists() else target.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        raise PreflightError(f"cannot inspect storage for {target}: {exc}") from exc
    result = StorageCheck(str(target), int(usage.free), int(minimum_free_bytes))
    if result.available_bytes < result.required_bytes:
        raise PreflightError(
            f"insufficient free space for {target}: "
            f"{result.available_bytes} < {result.required_bytes} bytes"
        )
    return result


def validate_provider_settings(
    *,
    endpoint: str,
    model: str,
    api_key_env: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str | bool]:
    """Validate provider metadata without exposing or contacting credentials."""

    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PreflightError("provider endpoint must be an HTTP(S) URL")
    if not model.strip():
        raise PreflightError("provider model must be non-empty")
    env = os.environ if environ is None else environ
    if not str(env.get(api_key_env, "")).strip():
        raise PreflightError(f"provider credential {api_key_env} is not available")
    return {
        "endpoint": f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}",
        "model": model,
        "credential_available": True,
    }


def resolve_image_for_run(
    reference: str,
    *,
    archive: Path | None = None,
    docker_executable: str = "docker",
    runner: Callable[..., Any] | None = None,
) -> ImageResolution:
    """Resolve/tag an image before a launcher starts an agent."""

    return resolve_docker_image(
        reference,
        archive=archive,
        docker_executable=docker_executable,
        runner=runner,
    )


def run_preflight(
    *,
    endpoint: str,
    model: str,
    api_key_env: str,
    run_root: Path,
    scratch_root: Path,
    image_reference: str | None = None,
    image_archive: Path | None = None,
    minimum_free_bytes: int = 1_073_741_824,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run local checks and return a redacted, receipt-ready summary."""

    provider = validate_provider_settings(
        endpoint=endpoint,
        model=model,
        api_key_env=api_key_env,
        environ=environ,
    )
    try:
        same_root = Path(run_root).expanduser().resolve() == Path(scratch_root).expanduser().resolve()
    except OSError as exc:
        raise PreflightError(f"cannot resolve run and scratch roots: {exc}") from exc
    if same_root:
        raise PreflightError("run_root and scratch_root must be independent")
    run_check = check_storage(run_root, minimum_free_bytes=minimum_free_bytes)
    scratch_check = check_storage(scratch_root, minimum_free_bytes=minimum_free_bytes)
    image = None
    if image_reference:
        image = resolve_image_for_run(image_reference, archive=image_archive)
    return {
        "provider": provider,
        "storage": {"run_root": run_check.as_dict(), "scratch_root": scratch_check.as_dict()},
        "image": image.as_dict() if image is not None else None,
    }
