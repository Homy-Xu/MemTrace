"""Deterministic Docker image resolution for benchmark launchers."""
from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


class ImageResolutionError(RuntimeError):
    """The requested image could not be made available under its locked tag."""

    failure_class = "IMAGE_RESOLUTION_FAILURE"


@dataclass(frozen=True, slots=True)
class ImageResolution:
    reference: str
    image_id: str
    loaded_from: str | None
    tagged_after_load: bool

    def as_dict(self) -> dict[str, str | bool | None]:
        return {
            "reference": self.reference,
            "image_id": self.image_id,
            "loaded_from": self.loaded_from,
            "tagged_after_load": self.tagged_after_load,
        }


_IMAGE_ID = re.compile(r"Loaded image ID:\s*(sha256:[0-9a-f]+)")


def _run(
    command: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
) -> subprocess.CompletedProcess[str]:
    invoke = runner or subprocess.run
    return invoke(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )


def _inspect(
    reference: str,
    *,
    docker_executable: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
) -> str | None:
    result = _run(
        [
            docker_executable,
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            reference,
        ],
        runner=runner,
    )
    if result.returncode != 0:
        return None
    image_id = result.stdout.strip()
    return image_id or None


def resolve_docker_image(
    reference: str,
    *,
    archive: Path | None = None,
    docker_executable: str = "docker",
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> ImageResolution:
    """Resolve an image and repair archive loads that expose only an ID."""

    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("image reference must be a non-empty string")
    reference = reference.strip()
    existing = _inspect(reference, docker_executable=docker_executable, runner=runner)
    if existing:
        return ImageResolution(reference, existing, None, False)
    if archive is None or not Path(archive).is_file():
        raise ImageResolutionError(f"image reference is unavailable and archive is missing: {reference}")

    load = _run(
        [docker_executable, "load", "--input", str(Path(archive))],
        runner=runner,
    )
    if load.returncode != 0:
        detail = (load.stderr or load.stdout).strip()[-1000:]
        raise ImageResolutionError(f"docker load failed for {reference}: {detail}")

    loaded_id_match = _IMAGE_ID.search(load.stdout or "")
    loaded_id = loaded_id_match.group(1) if loaded_id_match else None
    if loaded_id:
        tag = _run(
            [docker_executable, "image", "tag", loaded_id, reference],
            runner=runner,
        )
        if tag.returncode != 0:
            detail = (tag.stderr or tag.stdout).strip()[-1000:]
            raise ImageResolutionError(f"docker tag failed for {reference}: {detail}")

    resolved = _inspect(reference, docker_executable=docker_executable, runner=runner)
    if not resolved:
        raise ImageResolutionError(
            f"loaded archive did not expose expected image reference: {reference}"
        )
    return ImageResolution(reference, resolved, str(Path(archive)), bool(loaded_id))
