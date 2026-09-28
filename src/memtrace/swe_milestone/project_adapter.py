"""Project-level navigation for long-horizon SWE-Milestone runs.

Language adapters describe symbols. SWE-Milestone also needs the surrounding
maintenance unit: package/module boundaries, build manifests and test entry
points. This module builds that small, deterministic map from the official
task workspace. It is navigation context only; it never decides acceptance
and never runs a command.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SCHEMA = "homy/swe-milestone-project-map@1"
_IGNORED_DIRS = {
    ".git", ".venv", "node_modules", "target", "dist", "build",
    "coverage", "vendor", "__pycache__",
}
_MANIFESTS = (
    ("go", "go.work", "go-workspace"),
    ("go", "go.mod", "go-module"),
    ("rust", "Cargo.toml", "cargo"),
    ("java", "pom.xml", "maven"),
    ("java", "build.gradle", "gradle"),
    ("java", "build.gradle.kts", "gradle-kotlin"),
    ("java", "settings.gradle", "gradle-settings"),
    ("java", "settings.gradle.kts", "gradle-settings"),
    ("typescript", "package.json", "node-package"),
)


@dataclass(frozen=True, slots=True)
class ProjectUnit:
    """One build/test boundary, expressed relative to the task root."""

    path: str
    kind: str
    manifest: str
    languages: tuple[str, ...]
    source_dirs: tuple[str, ...]
    test_dirs: tuple[str, ...]
    build_targets: tuple[str, ...]
    test_targets: tuple[str, ...]
    exclude_patterns: tuple[str, ...] = ()


def _relative(path: Path, root: Path) -> str:
    value = path.relative_to(root).as_posix()
    return "." if value == "." else value


def _walk_manifests(root: Path, limit: int = 160) -> list[tuple[Path, str, str, str]]:
    """Find bounded project manifests without indexing dependencies or outputs."""

    found: list[tuple[Path, str, str, str]] = []
    for current, directories, files in os.walk(root):
        directories[:] = sorted(name for name in directories if name not in _IGNORED_DIRS)
        base = Path(current)
        for language, name, kind in _MANIFESTS:
            if name in files:
                found.append((base / name, language, kind, name))
                if len(found) >= limit:
                    return sorted(found, key=lambda item: str(item[0]))
    return sorted(found, key=lambda item: str(item[0]))


def _dirs_for(root: Path, names: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            result.append(name)
    return tuple(result)


def _unit_for(path: Path, language: str, kind: str, manifest_name: str, root: Path) -> ProjectUnit:
    relative = _relative(path.parent, root)
    if language == "go":
        source_dirs, tests = (".",), (".",)
        build, test, languages = ("go build ./...",), ("go test ./...",), ("go",)
    elif language == "rust":
        source_dirs = _dirs_for(path.parent, ("src", "tests", "benches"))
        tests = _dirs_for(path.parent, ("tests", "src"))
        build, test, languages = ("cargo build --workspace --offline",), ("cargo test --workspace --offline",), ("rust",)
    elif kind.startswith("node"):
        source_dirs = _dirs_for(path.parent, ("src", "packages", "apps"))
        tests = _dirs_for(path.parent, ("test", "tests", "__tests__"))
        build, test, languages = ("npm run build",), ("npm test",), ("typescript", "javascript")
    else:
        source_dirs = _dirs_for(path.parent, ("src/main", "src/main/java", "src/main/groovy"))
        tests = _dirs_for(path.parent, ("src/test", "src/test/java", "src/test/groovy"))
        if kind.startswith("gradle"):
            build, test = ("./gradlew classes --offline",), ("./gradlew test --offline",)
        else:
            build, test = ("mvn -o test-compile -DskipTests",), ("mvn -o test",)
        languages = ("java", "groovy")
    return ProjectUnit(
        path=relative,
        kind=kind,
        manifest=manifest_name,
        languages=languages,
        source_dirs=source_dirs,
        test_dirs=tests,
        build_targets=build,
        test_targets=test,
    )


def build_project_map(
    root: str | Path,
    *,
    language: str,
    languages: Iterable[str] = (),
    build_command: str | None = None,
    test_framework: str | None = None,
    milestone_ids: Iterable[str] = (),
    repo_src_dirs: Iterable[str] = (),
    test_dirs: Iterable[str] = (),
    exclude_patterns: Iterable[str] = (),
) -> dict[str, object]:
    """Build a compact project map suitable for a Route Card."""

    repository = Path(root).expanduser().resolve(strict=True)
    normalized = tuple(
        dict.fromkeys(str(item).strip().lower() for item in (language, *languages) if item)
    )
    configured_sources = tuple(
        dict.fromkeys(str(item).strip().strip("/") or "." for item in repo_src_dirs if str(item).strip())
    )
    configured_tests = tuple(
        dict.fromkeys(str(item).strip().strip("/") or "." for item in test_dirs if str(item).strip())
    )
    configured_excludes = tuple(
        dict.fromkeys(str(item).strip() for item in exclude_patterns if str(item).strip())
    )
    units = [
        _unit_for(path, item_language, kind, manifest, repository)
        for path, item_language, kind, manifest in _walk_manifests(repository)
        if item_language in normalized or (item_language == "java" and "groovy" in normalized)
    ]
    # Official metadata is authoritative when the disposable workspace has
    # no manifests (or has generated/nested manifests outside the bounded
    # scan). Keep it as a navigation unit instead of presenting an empty map.
    if configured_sources or configured_tests:
        units.append(
            ProjectUnit(
                path=".",
                kind="official-config",
                manifest="",
                languages=normalized or (language,),
                source_dirs=configured_sources,
                test_dirs=configured_tests,
                build_targets=(str(build_command),) if build_command else (),
                test_targets=(str(test_framework),) if test_framework else (),
                exclude_patterns=configured_excludes,
            )
        )
    if not units:
        units = [
            ProjectUnit(
                path=".",
                kind="unmanifested",
                manifest="",
                languages=normalized or (language,),
                source_dirs=(),
                test_dirs=(),
                build_targets=(str(build_command),) if build_command else (),
                test_targets=(str(test_framework),) if test_framework else (),
                exclude_patterns=configured_excludes,
            )
        ]
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "language": str(language).lower(),
        "languages": list(normalized),
        "official_build_command": build_command,
        "official_test_framework": test_framework,
        "selected_milestones": list(dict.fromkeys(map(str, milestone_ids))),
        "configured_source_dirs": list(configured_sources),
        "configured_test_dirs": list(configured_tests),
        "exclude_patterns": list(configured_excludes),
        "units": [json.loads(json.dumps(asdict(unit))) for unit in units],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["map_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def render_project_context(project_map: dict[str, object], *, limit: int = 7000) -> str:
    """Render bounded, model-facing maintenance context without new gates."""

    units = project_map.get("units") or ()
    lines = [
        "## Project maintenance map (navigation only)",
        "This is a long-horizon SWE-Milestone project stream. Preserve earlier milestone behavior; this map is not acceptance evidence and never blocks an action.",
        f"Languages: {', '.join(map(str, project_map.get('languages') or ())) or 'unknown'}",
        f"Official build: {project_map.get('official_build_command') or 'repository configured'}",
        f"Test framework: {project_map.get('official_test_framework') or 'repository configured'}",
        "Build/test units:",
    ]
    for unit in units:
        if not isinstance(unit, dict):
            continue
        fields = (
            str(unit.get("path", ".")),
            str(unit.get("kind", "unit")),
            "src=" + ",".join(map(str, unit.get("source_dirs") or ())),
            "tests=" + ",".join(map(str, unit.get("test_dirs") or ())),
            "exclude=" + ",".join(map(str, unit.get("exclude_patterns") or ())),
            "build=" + "; ".join(map(str, unit.get("build_targets") or ())),
            "test=" + "; ".join(map(str, unit.get("test_targets") or ())),
        )
        lines.append("- " + " | ".join(fields))
    lines.extend(
        (
            "At each milestone boundary: inspect the affected unit, keep prior passing behavior, run narrow unit validation, then run the official evaluator path when available.",
            "When a relation is uncertain (dynamic Groovy dispatch, generated code, macro expansion, aliases), treat it as a navigation hint and inspect current source/test rather than guessing.",
        )
    )
    text = "\n".join(lines)
    return (
        text
        if len(text) <= limit
        else text[: limit - 80] + "\n[project map truncated; use the durable project-map receipt]\n"
    )
