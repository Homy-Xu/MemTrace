from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from memtrace.benchmarks.image import ImageResolutionError, resolve_docker_image


def test_image_archive_id_is_tagged_before_agent_start(tmp_path: Path) -> None:
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"fixture")
    calls: list[list[str]] = []
    inspect_count = 0

    def runner(command, **kwargs):
        nonlocal inspect_count
        calls.append(list(command))
        if command[1:3] == ["image", "inspect"]:
            inspect_count += 1
            if inspect_count == 1:
                return subprocess.CompletedProcess(command, 1, "", "not found")
            return subprocess.CompletedProcess(command, 0, "sha256:abc\n", "")
        if command[1] == "load":
            return subprocess.CompletedProcess(command, 0, "Loaded image ID: sha256:abc\n", "")
        if command[1:3] == ["image", "tag"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)

    result = resolve_docker_image(
        "registry.example/deepswe:locked",
        archive=archive,
        runner=runner,
    )
    assert result.tagged_after_load is True
    assert any(call[1:3] == ["image", "tag"] for call in calls)


def test_image_resolution_fails_before_model_call(tmp_path: Path) -> None:
    with pytest.raises(ImageResolutionError):
        resolve_docker_image(
            "missing:locked",
            archive=tmp_path / "missing.tar",
            runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", ""),
        )
