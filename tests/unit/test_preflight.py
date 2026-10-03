from __future__ import annotations

from pathlib import Path

import pytest

from memtrace.benchmarks.preflight import PreflightError, check_storage, validate_provider_settings


def test_provider_preflight_checks_url_model_and_protected_credential() -> None:
    result = validate_provider_settings(
        endpoint="https://api.example.test/v1",
        model="deepseek-v4-flash",
        api_key_env="MEMTENSOR_API_KEY",
        environ={"MEMTENSOR_API_KEY": "protected-value"},
    )
    assert result["credential_available"] is True
    assert "protected-value" not in str(result)


def test_provider_preflight_does_not_accept_missing_credential() -> None:
    with pytest.raises(PreflightError):
        validate_provider_settings(
            endpoint="https://api.example.test/v1",
            model="deepseek-v4-flash",
            api_key_env="MEMTENSOR_API_KEY",
            environ={},
        )


def test_storage_check_accepts_existing_tmp_path(tmp_path: Path) -> None:
    result = check_storage(tmp_path, minimum_free_bytes=1)
    assert result.available_bytes >= result.required_bytes
