"""Language-neutral benchmark manifests for DeepSWE and SWE-Milestone.

The benchmark runners remain responsible for containers, model calls and
official scoring.  This module only reads the pinned task/config inputs and
turns them into explicit language, image, timeout and test-framework records.
It deliberately has no Python-only assumptions and performs no repository
walk beyond the manifest files named by the caller.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from .language_adapters import normalize_language, supported_languages


@dataclass(frozen=True, slots=True)
class DeepSWETaskSpec:
    task_id: str
    language: str
    task_dir: Path
    repository_url: str
    base_commit_hash: str
    docker_image: str
    agent_timeout_sec: float
    verifier_timeout_sec: float
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    verifier: dict[str, Any] = field(default_factory=dict)
    manifest_sha256: str = ""


@dataclass(frozen=True, slots=True)
class SWEMilestoneProjectSpec:
    project_id: str
    config_path: Path
    languages: tuple[str, ...]
    repo_src_dirs: tuple[str, ...]
    test_dirs: tuple[str, ...]
    main_branch: str | None
    build_command: str | None
    test_framework: str | None
    base_image_name: str | None
    raw: dict[str, Any] = field(default_factory=dict)
    milestone_order: tuple[str, ...] = ()
    evaluator: dict[str, Any] = field(default_factory=dict)
    image_mapping: dict[str, str] = field(default_factory=dict)
    config_sha256: str = ""

    @property
    def language(self) -> str:
        return self.languages[0] if self.languages else "unknown"


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _task_id_from(data: dict[str, Any], task_dir: Path) -> str:
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    value = str(metadata.get("task_id") or task_dir.name).strip()
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def load_deepswe_tasks(
    tasks_root: str | Path, task_ids: Iterable[str] | None = None
) -> tuple[DeepSWETaskSpec, ...]:
    """Load every DeepSWE task.toml, regardless of declared language."""

    root = Path(tasks_root).expanduser().resolve()
    if root.is_file() and root.name == "task.toml":
        candidates = (root,)
    else:
        candidates = tuple(sorted(root.glob("*/task.toml")))
    if not candidates:
        raise FileNotFoundError(f"no DeepSWE task.toml files under {root}")
    selected = {str(item) for item in task_ids} if task_ids is not None else None
    specs: list[DeepSWETaskSpec] = []
    for manifest in candidates:
        with manifest.open("rb") as handle:
            data = tomllib.load(handle)
        metadata = _as_dict(data.get("metadata"))
        environment = _as_dict(data.get("environment"))
        verifier = _as_dict(data.get("verifier"))
        environment_limits = _as_dict(data.get("environment"))
        task_id = _task_id_from(data, manifest.parent)
        if (
            selected is not None
            and task_id not in selected
            and str(metadata.get("ext_id", "")) not in selected
        ):
            continue
        language = normalize_language(str(metadata.get("language") or "unknown"))
        if language not in supported_languages():
            raise ValueError(f"unsupported task language {language}: {manifest}")
        if not environment.get("docker_image"):
            raise ValueError(f"missing task image: {manifest}")
        for section in ("agent", "verifier"):
            if (_number(_as_dict(data.get(section)).get("timeout_sec"), 0) or 0) <= 0:
                raise ValueError(f"missing positive {section} timeout: {manifest}")
        specs.append(
            DeepSWETaskSpec(
                task_id=task_id,
                language=language,
                task_dir=manifest.parent,
                repository_url=str(metadata.get("repository_url", "")),
                base_commit_hash=str(metadata.get("base_commit_hash", "")),
                docker_image=str(environment.get("docker_image", "")),
                agent_timeout_sec=_number(_as_dict(data.get("agent")).get("timeout_sec"), 0.0)
                or 0.0,
                verifier_timeout_sec=_number(verifier.get("timeout_sec"), 0.0) or 0.0,
                cpus=_integer(environment_limits.get("cpus")),
                memory_mb=_integer(environment_limits.get("memory_mb")),
                storage_mb=_integer(environment_limits.get("storage_mb")),
                metadata={
                    "display_title": metadata.get("display_title"),
                    "category": metadata.get("category"),
                    "ext_id": metadata.get("ext_id"),
                    "verifier_environment": _as_dict(verifier.get("environment")),
                },
                verifier=dict(verifier),
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
        )
    specs.sort(key=lambda item: item.task_id)
    seen: set[str] = set()
    for item in specs:
        if item.task_id in seen:
            raise ValueError(f"duplicate DeepSWE task id: {item.task_id}")
        seen.add(item.task_id)
    if selected is not None:
        found = seen | {str(s.metadata.get("ext_id")) for s in specs}
        if selected - found:
            raise ValueError(f"unknown DeepSWE tasks: {sorted(selected - found)}")
    return tuple(specs)


def deepswe_language_counts(specs: Iterable[DeepSWETaskSpec]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in specs:
        counts[item.language] = counts.get(item.language, 0) + 1
    return dict(sorted(counts.items()))


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    # Full safe YAML: nested evaluator policy, folded shell commands and
    # indentless lists must survive. Never execute Python YAML object tags.
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"project config must be a mapping: {path}")
    return result


def _project_languages(project_id: str, raw: dict[str, Any]) -> tuple[str, ...]:
    declarations = raw.get("languages")
    if isinstance(declarations, list):
        return tuple(normalize_language(str(v)) for v in declarations)
    declared = str(raw.get("language") or "").strip().casefold()
    if declared:
        return (declared,)
    lowered = project_id.casefold()
    if "dubbo" in lowered:
        return ("java", "groovy")
    if "element" in lowered:
        return ("typescript", "javascript")
    if "navidrome" in lowered:
        return ("go", "typescript", "javascript")
    if "ripgrep" in lowered or "nushell" in lowered:
        return ("rust",)
    if "scikit" in lowered:
        return ("python",)
    if "go-zero" in lowered:
        return ("go",)
    return ("unknown",)


def load_swe_milestone_configs(
    config_root: str | Path,
    *,
    sequences: dict[str, list[dict[str, Any]]] | None = None,
    image_mapping: dict[str, str] | None = None,
) -> tuple[SWEMilestoneProjectSpec, ...]:
    root = Path(config_root).expanduser().resolve()
    candidates = tuple(sorted((*root.glob("*.yaml"), *root.glob("*.yml"))))
    if not candidates:
        raise FileNotFoundError(f"no SWE-Milestone YAML configs under {root}")
    specs: list[SWEMilestoneProjectSpec] = []
    for path in candidates:
        raw = _read_yaml(path)
        project_id = path.stem
        milestones = (sequences or {}).get(project_id, raw.get("milestones", []))
        if not isinstance(milestones, list):
            raise ValueError(f"milestone sequence must be ordered: {project_id}")
        order = tuple(
            str(m.get("milestone_id", m.get("id", ""))) if isinstance(m, dict) else str(m)
            for m in milestones
        )
        if len(set(order)) != len(order) or any(not x for x in order):
            raise ValueError(f"invalid milestone sequence: {project_id}")
        specs.append(
            SWEMilestoneProjectSpec(
                project_id=project_id,
                config_path=path,
                languages=_project_languages(project_id, raw),
                repo_src_dirs=tuple(
                    str(item) for item in raw.get("repo_src_dirs", ()) if item is not None
                ),
                test_dirs=tuple(str(item) for item in raw.get("test_dirs", ()) if item is not None),
                main_branch=str(raw.get("main_branch")) if raw.get("main_branch") else None,
                build_command=str(raw.get("build_command")) if raw.get("build_command") else None,
                test_framework=str(raw.get("test_framework"))
                if raw.get("test_framework")
                else None,
                base_image_name=str(raw.get("base_image_name"))
                if raw.get("base_image_name")
                else None,
                raw=raw,
                milestone_order=order,
                evaluator=dict(raw.get("evaluation", raw.get("evaluator", {}))),
                image_mapping=dict(image_mapping or {}),
                config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return tuple(specs)


def load_swe_milestone_projects(
    data_root: Path, evaluator_root: Path, digest_manifest: Path
) -> tuple[SWEMilestoneProjectSpec, ...]:
    """Read official ordered metadata and selected IDs; never sort M10 before M2.

    Host paths in upstream metadata are provenance only, NOT executable paths.
    Keep evaluator policy intact and resolve images through the pinned digest map.
    """
    image_map = {}
    for line in digest_manifest.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        alias, immutable = line.split("\t", 1)
        if "@sha256:" not in immutable or len(immutable.rsplit(":", 1)[-1]) != 64:
            raise ValueError("official image map contains an unpinned image")
        if alias in image_map and image_map[alias] != immutable:
            raise ValueError(f"ambiguous image mapping: {alias}")
        image_map[alias] = immutable
    specs = load_swe_milestone_configs(data_root / "config")
    result = []
    for spec in specs:
        project = data_root / spec.project_id
        metadata_path = project / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        order = tuple(metadata.get("topological_order", {}).get("full_order", ()))
        if not order:
            order = tuple(str(m["id"]) for m in metadata["milestones"])
        selected_path = project / "selected_milestone_ids.txt"
        if selected_path.is_file():
            selected = tuple(x.strip() for x in selected_path.read_text().splitlines() if x.strip())
            selected_source = "selected_milestone_ids.txt"
        else:
            # Some official projects (currently scikit-learn) publish no
            # separate selection file. Their complete ordered metadata is the
            # authoritative selected set; do not invent lexical ordering.
            selected = order
            selected_source = "metadata.topological_order.full_order"
        if len(set(order)) != len(order) or set(selected) - set(order):
            raise ValueError(f"invalid official milestone order: {spec.project_id}")
        with (project / "milestones.csv").open(newline="") as handle:
            known = {row["id"] for row in csv.DictReader(handle)}
        if set(order) - known:
            raise ValueError(f"metadata milestones missing from CSV: {spec.project_id}")
        framework = spec.test_framework
        # Explicit language family only as a documented fallback. Never a
        # blanket pytest assumption or a replacement for official evaluator.
        framework = framework or {
            "java": "maven",
            "rust": "cargo",
            "go": "go_test",
            "typescript": "repository-configured",
            "python": "pytest",
        }.get(spec.language)
        project_map = {
            a: v for a, v in image_map.items() if "/" + spec.project_id.lower() + "__" in a.lower()
        }
        result.append(
            replace(
                spec,
                milestone_order=order,
                test_framework=framework,
                image_mapping=project_map,
                evaluator={
                    **spec.evaluator,
                    "source_root": str(evaluator_root.resolve()),
                    "entrypoint": "harness.e2e.run_e2e",
                    "selected_ids": selected,
                    "selected_ids_source": selected_source,
                    "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                    "digest_manifest_sha256": hashlib.sha256(
                        digest_manifest.read_bytes()
                    ).hexdigest(),
                },
            )
        )
    return tuple(result)


def map_deepswe_images(
    specs: Iterable[DeepSWETaskSpec], receipt_path: Path
) -> dict[str, dict[str, Any]]:
    """Join by exact source identity; image counts need not equal task counts."""
    receipt = json.loads(receipt_path.read_text())
    index: dict[str, dict[str, Any]] = {}
    for row in receipt.get("images", ()):
        source = str(row["source"])
        if source in index and index[source] != row:
            raise ValueError(f"ambiguous image receipt: {source}")
        if not str(row.get("digest", "")).startswith("sha256:"):
            raise ValueError(f"image not pinned: {source}")
        index[source] = row
    mapped = {}
    for task in specs:
        row = index.get(task.docker_image)
        mapped[task.task_id] = {
            "task_id": task.task_id,
            "language": task.language,
            "source": task.docker_image,
            "status": "MAPPED" if row else "MISSING_IMAGE_RECEIPT",
            **(dict(row) if row else {}),
        }
    return mapped


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepswe-root", type=Path)
    parser.add_argument("--milestone-root", type=Path)
    args = parser.parse_args(argv)
    payload: dict[str, Any] = {}
    if args.deepswe_root:
        specs = load_deepswe_tasks(args.deepswe_root)
        payload["deepswe"] = {
            "count": len(specs),
            "languages": deepswe_language_counts(specs),
            "tasks": [_jsonable(asdict(item)) for item in specs],
        }
    if args.milestone_root:
        specs = load_swe_milestone_configs(args.milestone_root)
        payload["swe_milestone"] = {
            "count": len(specs),
            "projects": [_jsonable(asdict(item)) for item in specs],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
