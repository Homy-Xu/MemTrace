"""Provider availability is an execution outcome, never task correctness.

Only structured provider errors are passed here. Do not classify task text or
test output (which may legitimately discuss HTTP errors) as an API outage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderFailure:
    kind: str
    retryable: bool
    retry_after_seconds: float | None = None


def classify_provider_error(error: object) -> ProviderFailure | None:
    text = str(error or "").lower()
    if not text:
        return None
    if any(
        term in text
        for term in (
            "insufficient_quota",
            "insufficient quota",
            "credit balance",
            "quota exhausted",
        )
    ):
        return ProviderFailure("PROVIDER_QUOTA_EXHAUSTED", False)
    if any(
        term in text
        for term in (
            "invalid_api_key",
            "invalid api key",
            "unauthorized",
            "authentication",
            "401",
            "403 forbidden",
        )
    ):
        return ProviderFailure("PROVIDER_AUTH_FAILURE", False)
    retry_after = re.search(r"retry[-_ ]after[\s\"':=]+(\d+(?:\.\d+)?)", text)
    delay = float(retry_after.group(1)) if retry_after else None
    if "429" in text or "too many requests" in text or "rate limit" in text:
        return ProviderFailure("PROVIDER_RATE_LIMITED", True, delay)
    if "at capacity" in text or "overloaded" in text:
        return ProviderFailure("PROVIDER_CAPACITY", True, delay)
    if any(
        term in text
        for term in (
            "502",
            "503",
            "504",
            "connection reset",
            "connection refused",
            "timed out",
            "timeout",
            "stream disconnected",
            "stream closed",
        )
    ):
        return ProviderFailure("PROVIDER_UNAVAILABLE", True, delay)
    return None


def provider_retry_delay(error: object, attempt: int, fallback: float) -> float:
    """Same-thread retry, bounded even when upstream supplies a bad header."""
    failure = classify_provider_error(error)
    if failure is None or not failure.retryable:
        return fallback
    if failure.kind == "PROVIDER_UNAVAILABLE" and failure.retry_after_seconds is None:
        return fallback  # retain the Python baseline's transient network policy
    schedule = (30.0, 60.0, 120.0, 240.0, 300.0)
    delay = schedule[min(max(1, attempt), len(schedule)) - 1]
    return min(600.0, max(delay, failure.retry_after_seconds or 0.0))
