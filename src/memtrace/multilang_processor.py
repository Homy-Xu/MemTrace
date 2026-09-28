"""Opt-in Rich Graph metadata for multilingual project resolution.

The ordinary processor remains the source of symbols/relations.  This wrapper
adds only bounded project metadata needed to resolve TypeScript path aliases
and Rust workspace/path-dependency crate roots; it never creates acceptance
evidence.
"""

from __future__ import annotations

import json
import posixpath
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

from .rich_graph.models import FileProjection, FrontierTask
from .rich_graph.processor import process_frontier_file as _default_processor

_MAX_CONFIG_BYTES = 262_144
_MAX_WORKSPACE_MEMBERS = 256
_MAX_PATH_DEPENDENCIES = 256


def _repository_root(task: FrontierTask) -> Path:
    root = task.absolute_path
    for _ in Path(task.repository_relative_path).parts:
        root = root.parent
    return root.resolve()


def _tsconfig_metadata(task: FrontierTask) -> dict[str, object]:
    root = _repository_root(task)
    for directory in (task.absolute_path.parent, *task.absolute_path.parents):
        if not directory.is_relative_to(root):
            break
        config = directory / "tsconfig.json"
        try:
            if not config.is_file() or config.stat().st_size > _MAX_CONFIG_BYTES:
                continue
            raw = json.loads(config.read_text(encoding="utf-8"))
            options = raw.get("compilerOptions", {}) if isinstance(raw, dict) else {}
            paths = options.get("paths", {}) if isinstance(options, dict) else {}
            aliases: dict[str, list[str]] = {}
            if isinstance(paths, dict):
                for alias, targets in list(paths.items())[:256]:
                    if isinstance(targets, list) and targets:
                        aliases[str(alias)] = [str(value) for value in targets[:4]]
            if not aliases:
                return {}
            base_url = (
                str(options.get("baseUrl", "."))
                if isinstance(options, dict)
                else "."
            )
            base_path = (directory / base_url).resolve()
            result: dict[str, object] = {"path_aliases": aliases}
            if base_path.is_relative_to(root):
                result["tsconfig_base_url"] = base_path.relative_to(root).as_posix()
            return result
        except (OSError, UnicodeError, ValueError, TypeError):
            return {}
    return {}


def _cargo_manifest(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file() or path.stat().st_size > _MAX_CONFIG_BYTES:
            return None
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _cargo_workspace_root(file_path: Path, repository_root: Path) -> Path | None:
    nearest_package: Path | None = None
    for directory in (file_path.parent, *file_path.parents):
        if not directory.is_relative_to(repository_root):
            break
        manifest = directory / "Cargo.toml"
        value = _cargo_manifest(manifest)
        if value is None:
            continue
        if nearest_package is None and isinstance(value.get("package"), dict):
            nearest_package = directory
        if isinstance(value.get("workspace"), dict):
            return directory
    return nearest_package


def _manifest_source_root(
    manifest_dir: Path,
    manifest: dict[str, Any],
    repository_root: Path,
) -> str | None:
    package = manifest.get("package")
    if not isinstance(package, dict):
        return None
    candidates: list[Path] = []
    lib = manifest.get("lib")
    if isinstance(lib, dict) and lib.get("path"):
        candidates.append(manifest_dir / str(lib["path"]))
    for key in ("bin", "example", "test", "bench"):
        items = manifest.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("path"):
                    candidates.append(manifest_dir / str(item["path"]))
    candidates.extend((manifest_dir / "src/lib.rs", manifest_dir / "src/main.rs"))
    for candidate in candidates:
        if candidate.is_file():
            source = candidate.parent.resolve()
            if source.is_relative_to(repository_root):
                return source.relative_to(repository_root).as_posix()
    fallback = (manifest_dir / "src").resolve()
    if fallback.is_relative_to(repository_root):
        return fallback.relative_to(repository_root).as_posix()
    return None


def _path_dependencies(
    manifest_dir: Path,
    manifest: dict[str, Any],
    repository_root: Path,
) -> dict[str, str]:
    result: dict[str, str] = {}
    tables = [
        manifest.get("dependencies"),
        manifest.get("dev-dependencies"),
        manifest.get("build-dependencies"),
    ]
    target = manifest.get("target")
    if isinstance(target, dict):
        for cfg in target.values():
            if isinstance(cfg, dict):
                tables.extend(
                    (
                        cfg.get("dependencies"),
                        cfg.get("dev-dependencies"),
                        cfg.get("build-dependencies"),
                    )
                )
    for table in tables:
        if not isinstance(table, dict):
            continue
        for alias, raw in list(table.items())[:_MAX_PATH_DEPENDENCIES]:
            if not isinstance(raw, dict) or not raw.get("path"):
                continue
            dependency = (manifest_dir / str(raw["path"])).resolve()
            if not dependency.is_relative_to(repository_root):
                continue
            name = str(raw.get("package") or alias)
            result[name.replace("-", "_")] = dependency.relative_to(
                repository_root
            ).as_posix()
    return result


def _cargo_metadata(task: FrontierTask) -> dict[str, object]:
    """Return bounded Cargo workspace/path-dependency navigation metadata.

    Only manifests on the current file's ancestor chain and path dependencies
    explicitly named by that package are read.  Workspace member globs are
    expanded only from the declared workspace root, bounded to 256 entries.
    """

    root = _repository_root(task)
    package_dir: Path | None = None
    package_manifest: dict[str, Any] | None = None
    for directory in (task.absolute_path.parent, *task.absolute_path.parents):
        if not directory.is_relative_to(root):
            break
        value = _cargo_manifest(directory / "Cargo.toml")
        if value is not None and isinstance(value.get("package"), dict):
            package_dir, package_manifest = directory, value
            break
    if package_dir is None or package_manifest is None:
        return {}
    workspace_dir = _cargo_workspace_root(task.absolute_path, root) or package_dir
    package = package_manifest["package"]
    crate_name = str(package.get("name") or "").replace("-", "_")
    crate_root = _manifest_source_root(package_dir, package_manifest, root)
    path_dependencies = _path_dependencies(
        package_dir,
        package_manifest,
        root,
    )
    crate_roots: dict[str, str] = {}
    if crate_name and crate_root:
        crate_roots[crate_name] = crate_root
    for dependency_name, dependency_dir in path_dependencies.items():
        dependency_path = root / dependency_dir
        dependency_manifest = _cargo_manifest(dependency_path / "Cargo.toml")
        dependency_root = (
            _manifest_source_root(dependency_path, dependency_manifest, root)
            if dependency_manifest is not None
            else None
        )
        if dependency_root:
            crate_roots[dependency_name] = dependency_root
    result: dict[str, object] = {
        "crate_name": crate_name,
        "crate_manifest_dir": package_dir.relative_to(root).as_posix()
        if package_dir != root
        else ".",
        "cargo_workspace_root": workspace_dir.relative_to(root).as_posix()
        if workspace_dir != root
        else ".",
        "path_dependencies": path_dependencies,
    }
    if crate_root:
        result["crate_source_root"] = crate_root
    workspace_manifest = _cargo_manifest(workspace_dir / "Cargo.toml") or {}
    workspace = workspace_manifest.get("workspace")
    member_roots: dict[str, str] = {}
    if isinstance(workspace, dict):
        seen = 0
        for pattern in workspace.get("members", ()) or ():
            if seen >= _MAX_WORKSPACE_MEMBERS:
                break
            normalized = posixpath.normpath(str(pattern))
            if normalized.startswith("../") or normalized.startswith("/"):
                continue
            for member in sorted(workspace_dir.glob(normalized)):
                if seen >= _MAX_WORKSPACE_MEMBERS:
                    break
                manifest = _cargo_manifest(member / "Cargo.toml")
                package_value = (
                    manifest.get("package") if isinstance(manifest, dict) else None
                )
                if not isinstance(package_value, dict):
                    continue
                name = str(package_value.get("name") or "").replace("-", "_")
                source = _manifest_source_root(member, manifest, root)
                if name and source:
                    member_roots[name] = source
                    seen += 1
    if member_roots:
        result["cargo_workspace_members"] = member_roots
        crate_roots.update(member_roots)
    if crate_roots:
        result["cargo_crate_roots"] = crate_roots
    return result


def process_frontier_file(task: FrontierTask) -> FileProjection:
    projection = _default_processor(task)
    metadata = dict(projection.metadata)
    navigation = dict(metadata.get("navigation_index", {}))
    if projection.language in {"typescript", "javascript"}:
        navigation.update(_tsconfig_metadata(task))
    elif projection.language == "rust":
        navigation.update(_cargo_metadata(task))
    if navigation:
        metadata["navigation_index"] = navigation
    return replace(projection, metadata=metadata)
