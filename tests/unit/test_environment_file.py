from __future__ import annotations

import os
from pathlib import Path

import pytest

from memtrace.harness.environment import (
    EnvironmentFileError,
    apply_environment_files,
    read_environment_file,
)


def _private_file(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)
    return path


def test_environment_file_accepts_export_and_quotes_without_shell_expansion(tmp_path: Path) -> None:
    path = _private_file(
        tmp_path / "provider.env",
        "export MEMTENSOR_API_KEY='token-value'\nHTTPS_PROXY=\"http://proxy.invalid:8080\"\n",
    )
    assert read_environment_file(path) == {
        "MEMTENSOR_API_KEY": "token-value",
        "HTTPS_PROXY": "http://proxy.invalid:8080",
    }


def test_environment_file_rejects_non_private_files(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.env"
    path.write_text("TOKEN=value\n", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(EnvironmentFileError, match="private"):
        read_environment_file(path)


def test_apply_environment_files_updates_process_without_returning_values(
    tmp_path: Path, monkeypatch
) -> None:
    path = _private_file(tmp_path / "env", "MEMTRACE_TEST_SECRET='not-printed'\n")
    monkeypatch.delenv("MEMTRACE_TEST_SECRET", raising=False)
    assert apply_environment_files([path]) == ("MEMTRACE_TEST_SECRET",)
    assert os.environ["MEMTRACE_TEST_SECRET"] == "not-printed"
