"""Effective compute envelope of the sandbox the model's commands run in.

Inside a cgroup-limited container ``os.cpu_count()`` still reports the host's
CPUs.  Tools that size themselves from it (``pytest -n auto``, ``make -j``,
BLAS thread pools) then spawn a host-sized swarm inside a container that owns
two cores and a few gigabytes, and the memory cgroup kills the Provider
process along with the swarm.  The runtime cannot change what the model types,
but it owns the environment those commands inherit, so it publishes the real
budget through the conventional knobs and to the Route Card.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_CPU_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_CPU_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
_CGROUP_V2_MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1_MEMORY_LIMIT = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_UNLIMITED_MEMORY_SENTINEL = 1 << 60

# Environment variables honored by common parallel tools when they size
# themselves "automatically".  Operator-provided values are never overridden.
XDIST_AUTO_WORKERS_ENV = "PYTEST_XDIST_AUTO_NUM_WORKERS"
_THREAD_POOL_ENVS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True, slots=True)
class ResourceEnvelope:
    visible_cpus: int
    effective_cpus: int
    memory_limit_mb: int | None
    source: str

    @property
    def cpu_limited(self) -> bool:
        return self.effective_cpus < self.visible_cpus

    def as_receipt(self) -> dict[str, object]:
        return {
            "visible_cpus": self.visible_cpus,
            "effective_cpus": self.effective_cpus,
            "memory_limit_mb": self.memory_limit_mb,
            "source": self.source,
        }


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _visible_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _cgroup_cpu_quota() -> tuple[int | None, str]:
    """CPUs granted by the cgroup CFS quota, rounded up; ``None`` when unlimited."""

    v2 = _read_text(_CGROUP_V2_CPU_MAX)
    if v2:
        parts = v2.split()
        if parts and parts[0] != "max" and len(parts) >= 2:
            try:
                quota, period = int(parts[0]), int(parts[1])
            except ValueError:
                return None, "cgroup-v2-unparsable"
            if quota > 0 and period > 0:
                return max(1, -(-quota // period)), "cgroup-v2"
        return None, "cgroup-v2-unlimited"
    quota_text = _read_text(_CGROUP_V1_CPU_QUOTA)
    period_text = _read_text(_CGROUP_V1_CPU_PERIOD)
    if quota_text and period_text:
        try:
            quota, period = int(quota_text), int(period_text)
        except ValueError:
            return None, "cgroup-v1-unparsable"
        if quota > 0 and period > 0:
            return max(1, -(-quota // period)), "cgroup-v1"
        return None, "cgroup-v1-unlimited"
    return None, "no-cgroup"


def _cgroup_memory_limit_mb() -> int | None:
    for path in (_CGROUP_V2_MEMORY_MAX, _CGROUP_V1_MEMORY_LIMIT):
        text = _read_text(path)
        if not text or text == "max":
            continue
        try:
            limit = int(text)
        except ValueError:
            continue
        if 0 < limit < _UNLIMITED_MEMORY_SENTINEL:
            return limit // (1024 * 1024)
    return None


def detect_resource_envelope() -> ResourceEnvelope:
    visible = _visible_cpus()
    quota, source = _cgroup_cpu_quota()
    effective = min(visible, quota) if quota is not None else visible
    return ResourceEnvelope(
        visible_cpus=visible,
        effective_cpus=effective,
        memory_limit_mb=_cgroup_memory_limit_mb(),
        source=source,
    )


def resource_envelope_environment(
    environment: Mapping[str, str],
    envelope: ResourceEnvelope | None = None,
) -> dict[str, str]:
    """Return the variables that pin auto-sizing tools to the real CPU budget.

    Only variables absent from ``environment`` are produced, so an operator who
    deliberately sets them keeps control.  Nothing is produced when the sandbox
    is not CPU-limited: the tools' own defaults are then already correct.
    """

    envelope = envelope or detect_resource_envelope()
    if not envelope.cpu_limited:
        return {}
    budget = str(envelope.effective_cpus)
    produced: dict[str, str] = {}
    for name in (XDIST_AUTO_WORKERS_ENV, *_THREAD_POOL_ENVS):
        if not environment.get(name, "").strip():
            produced[name] = budget
    return produced
