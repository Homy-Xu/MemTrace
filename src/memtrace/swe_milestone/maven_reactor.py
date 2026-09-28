"""Bounded Maven reactor metadata derived only from declared POM modules.

The index deliberately does not invoke Maven and never searches ``target/``.
It follows the root POM's ``<modules>`` declarations, parses XML with the
standard library, and exposes deterministic project-local dependency edges.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

_PROPERTY = re.compile(r"\$\{([^}]+)\}")
_PROJECT_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+\-]*$")
_MAX_POM_BYTES = 1_048_576
_KNOWN_GENERATORS: dict[str, tuple[tuple[str, str], ...]] = {
    "build-helper-maven-plugin": (("sources", "source"), ("testSources", "testSource")),
    "protobuf-maven-plugin": (
        ("outputDirectory", "source"),
        ("testOutputDirectory", "testSource"),
    ),
    "templating-maven-plugin": (("outputDirectory", "source"),),
    "mustache-maven-plugin": (
        ("outputDirectory", "source"),
        ("generatedSourcesDirectory", "source"),
    ),
    "openapi-generator-maven-plugin": (("output", "source"),),
}
_GENERATOR_DEFAULTS: dict[str, tuple[tuple[str, str], ...]] = {
    "protobuf-maven-plugin": (
        ("${project.build.directory}/generated-sources/protobuf/java", "source"),
        ("${project.build.directory}/generated-test-sources/protobuf/java", "testSource"),
    ),
    "templating-maven-plugin": (
        ("${project.build.directory}/generated-sources/templating", "source"),
    ),
    "mustache-maven-plugin": (
        ("${project.build.directory}/generated-sources/mustache", "source"),
    ),
    "openapi-generator-maven-plugin": (
        ("${project.build.directory}/generated-sources/openapi", "source"),
    ),
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    return next((item for item in element if _local(item.tag) == name), None)


def _children(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []
    return [item for item in element if _local(item.tag) == name]


def _text(element: ET.Element | None, name: str, default: str = "") -> str:
    child = _child(element, name)
    return (child.text or "").strip() if child is not None else default


def _interpolate(value: str, properties: dict[str, str]) -> str:
    current = value
    for _ in range(8):
        updated = _PROPERTY.sub(lambda match: properties.get(match.group(1), match.group(0)), current)
        if updated == current:
            return updated
        current = updated
    return current


def _safe_relative(root: Path, base: Path, value: str) -> Path | None:
    candidate = (base / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


@dataclass(frozen=True, slots=True)
class MavenCoordinate:
    group_id: str
    artifact_id: str
    version: str = ""


@dataclass(frozen=True, slots=True)
class MavenModule:
    path: str
    pom_path: str
    coordinate: MavenCoordinate
    packaging: str
    declared_modules: tuple[str, ...]
    dependencies: tuple[str, ...]
    source_roots: tuple[str, ...]
    test_source_roots: tuple[str, ...]
    generated_source_roots: tuple[str, ...]
    generated_test_roots: tuple[str, ...]

    @property
    def selector(self) -> str:
        return self.path


@dataclass(slots=True)
class _RawPom:
    path: str
    pom_path: str
    root: ET.Element
    parent_group: str
    parent_artifact: str
    parent_version: str
    parent_relative: str
    modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MavenReactor:
    root: Path
    modules: tuple[MavenModule, ...]
    errors: tuple[str, ...] = ()

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        max_modules: int = 256,
        max_pom_bytes: int = _MAX_POM_BYTES,
    ) -> "MavenReactor":
        root = root.resolve()
        queue: deque[Path] = deque([root / "pom.xml"])
        seen: set[Path] = set()
        raw: list[_RawPom] = []
        errors: list[str] = []
        while queue and len(raw) < max_modules:
            pom = queue.popleft().resolve()
            if pom in seen:
                continue
            seen.add(pom)
            try:
                relative_pom = pom.relative_to(root).as_posix()
            except ValueError:
                errors.append(f"declared module escapes reactor root: {pom}")
                continue
            try:
                if pom.stat().st_size > max_pom_bytes:
                    raise ValueError(f"POM exceeds {max_pom_bytes} bytes")
                document = ET.parse(pom)
            except (OSError, ET.ParseError, ValueError) as exc:
                errors.append(f"{relative_pom}: {exc}")
                continue
            project = document.getroot()
            if _local(project.tag) != "project":
                errors.append(f"{relative_pom}: root element is not <project>")
                continue
            module_path = pom.parent.relative_to(root).as_posix() or "."
            parent = _child(project, "parent")
            declared = tuple(
                (item.text or "").strip()
                for item in _children(_child(project, "modules"), "module")
                if (item.text or "").strip()
            )
            model = _RawPom(
                module_path,
                relative_pom,
                project,
                _text(parent, "groupId"),
                _text(parent, "artifactId"),
                _text(parent, "version"),
                _text(parent, "relativePath", "../pom.xml"),
                declared,
            )
            raw.append(model)
            for declaration in declared:
                child = _safe_relative(root, pom.parent, declaration)
                if child is None:
                    errors.append(f"{relative_pom}: module escapes reactor root: {declaration}")
                    continue
                queue.append(child if child.name == "pom.xml" else child / "pom.xml")
        if queue:
            errors.append(f"reactor exceeds module limit {max_modules}")

        by_pom = {item.pom_path: item for item in raw}
        resolved: dict[str, tuple[MavenCoordinate, dict[str, str]]] = {}
        resolving: set[str] = set()

        def resolve(model: _RawPom) -> tuple[MavenCoordinate, dict[str, str]]:
            if model.pom_path in resolved:
                return resolved[model.pom_path]
            if model.pom_path in resolving:
                errors.append(f"{model.pom_path}: cyclic parent relationship")
                return MavenCoordinate("", _text(model.root, "artifactId"), ""), {}
            resolving.add(model.pom_path)
            inherited: dict[str, str] = {}
            parent_coordinate = MavenCoordinate(
                model.parent_group, model.parent_artifact, model.parent_version
            )
            if model.parent_artifact and model.parent_relative:
                parent_path = _safe_relative(root, root / model.path, model.parent_relative)
                parent_pom = None
                if parent_path is not None:
                    parent_key = (
                        parent_path.relative_to(root).as_posix()
                        if parent_path.name == "pom.xml"
                        else (parent_path / "pom.xml").relative_to(root).as_posix()
                    )
                    parent_pom = by_pom.get(parent_key)
                if parent_pom is not None:
                    parent_coordinate, inherited = resolve(parent_pom)
            properties = dict(inherited)
            properties_element = _child(model.root, "properties")
            if properties_element is not None:
                for item in properties_element:
                    properties[_local(item.tag)] = (item.text or "").strip()
            group = _text(model.root, "groupId") or parent_coordinate.group_id
            version = _text(model.root, "version") or parent_coordinate.version
            artifact = _text(model.root, "artifactId")
            bootstrap = {
                **properties,
                "project.groupId": group,
                "pom.groupId": group,
                "project.artifactId": artifact,
                "pom.artifactId": artifact,
                "project.version": version,
                "pom.version": version,
                "project.basedir": str(root / model.path),
                "basedir": str(root / model.path),
                "project.build.directory": str(root / model.path / "target"),
            }
            group = _interpolate(group, bootstrap)
            artifact = _interpolate(artifact, bootstrap)
            version = _interpolate(version, bootstrap)
            coordinate = MavenCoordinate(group, artifact, version)
            bootstrap.update(
                {
                    "project.groupId": group,
                    "pom.groupId": group,
                    "project.artifactId": artifact,
                    "pom.artifactId": artifact,
                    "project.version": version,
                    "pom.version": version,
                }
            )
            resolving.remove(model.pom_path)
            resolved[model.pom_path] = coordinate, bootstrap
            return coordinate, bootstrap

        coordinates: dict[str, MavenCoordinate] = {}
        properties: dict[str, dict[str, str]] = {}
        for model in raw:
            coordinate, model_properties = resolve(model)
            coordinates[model.path] = coordinate
            properties[model.path] = model_properties
            if not coordinate.artifact_id:
                errors.append(f"{model.pom_path}: missing artifactId")
            for label, value in (
                ("version", coordinate.version),
                ("parent version", _interpolate(model.parent_version, model_properties)),
            ):
                if value and ("${" in value or _PROJECT_VERSION.fullmatch(value) is None):
                    errors.append(f"{model.pom_path}: invalid {label} {value!r}")

        coordinate_paths: dict[tuple[str, str], list[str]] = {}
        for path, coordinate in coordinates.items():
            coordinate_paths.setdefault((coordinate.group_id, coordinate.artifact_id), []).append(path)

        modules: list[MavenModule] = []
        for model in raw:
            props = properties[model.path]
            dependencies: list[str] = []
            parent_key = (
                _interpolate(model.parent_group, props),
                _interpolate(model.parent_artifact, props),
            )
            parent_paths = coordinate_paths.get(parent_key, [])
            if len(parent_paths) == 1 and parent_paths[0] != model.path:
                dependencies.append(parent_paths[0])
            dependency_elements = _children(_child(model.root, "dependencies"), "dependency")
            for profile in _children(_child(model.root, "profiles"), "profile"):
                dependency_elements.extend(_children(_child(profile, "dependencies"), "dependency"))
            for dependency in dependency_elements:
                key = (
                    _interpolate(_text(dependency, "groupId"), props),
                    _interpolate(_text(dependency, "artifactId"), props),
                )
                matches = coordinate_paths.get(key, [])
                if len(matches) == 1 and matches[0] != model.path:
                    dependencies.append(matches[0])

            build = _child(model.root, "build")
            module_root = root / model.path

            def relative_root(value: str, default: str) -> str:
                expanded = _interpolate(value or default, props)
                path = Path(expanded)
                absolute = path.resolve() if path.is_absolute() else (module_root / path).resolve()
                try:
                    return absolute.relative_to(root).as_posix()
                except ValueError:
                    return ""

            source_roots = [
                relative_root(_text(build, "sourceDirectory"), "src/main/java"),
                relative_root("", "src/main/groovy"),
            ]
            test_roots = [
                relative_root(_text(build, "testSourceDirectory"), "src/test/java"),
                relative_root("", "src/test/groovy"),
            ]
            generated: list[str] = []
            generated_tests: list[str] = []
            plugins = _children(_child(build, "plugins"), "plugin")
            management = _child(build, "pluginManagement")
            plugins.extend(_children(_child(management, "plugins"), "plugin"))
            for profile in _children(_child(model.root, "profiles"), "profile"):
                profile_build = _child(profile, "build")
                plugins.extend(_children(_child(profile_build, "plugins"), "plugin"))
            for plugin in plugins:
                plugin_artifact = _interpolate(_text(plugin, "artifactId"), props)
                fields = _KNOWN_GENERATORS.get(plugin_artifact)
                if fields is None:
                    continue
                for value, kind in _GENERATOR_DEFAULTS.get(plugin_artifact, ()):
                    candidate = relative_root(value, "")
                    if candidate:
                        (generated_tests if kind == "testSource" else generated).append(
                            candidate
                        )
                configurations = [_child(plugin, "configuration")]
                configurations.extend(
                    _child(execution, "configuration")
                    for execution in _children(_child(plugin, "executions"), "execution")
                )
                for configuration in configurations:
                    if configuration is None:
                        continue
                    for field, kind in fields:
                        element = _child(configuration, field)
                        values = (
                            [(child.text or "").strip() for child in element]
                            if element is not None and list(element)
                            else [((element.text or "").strip() if element is not None else "")]
                        )
                        for value in values:
                            if not value:
                                continue
                            candidate = relative_root(value, "")
                            if not candidate:
                                continue
                            absolute = (root / candidate).resolve()
                            target = (module_root / "target").resolve()
                            if not absolute.is_relative_to(target):
                                continue
                            (generated_tests if kind == "testSource" else generated).append(candidate)
            modules.append(
                MavenModule(
                    model.path,
                    model.pom_path,
                    coordinates[model.path],
                    _interpolate(_text(model.root, "packaging", "jar"), props),
                    model.modules,
                    tuple(dict.fromkeys(dependencies)),
                    tuple(filter(None, dict.fromkeys(source_roots))),
                    tuple(filter(None, dict.fromkeys(test_roots))),
                    tuple(dict.fromkeys(generated)),
                    tuple(dict.fromkeys(generated_tests)),
                )
            )
        return cls(root, tuple(modules), tuple(dict.fromkeys(errors)))

    def module(self, path: str) -> MavenModule | None:
        return next((module for module in self.modules if module.path == path), None)

    def nearest_module(self, changed_path: str) -> MavenModule | None:
        normalized = PurePosixPath(changed_path)
        candidates = [
            module
            for module in self.modules
            if module.path == "."
            or normalized == PurePosixPath(module.path)
            or PurePosixPath(module.path) in normalized.parents
        ]
        return max(candidates, key=lambda item: len(PurePosixPath(item.path).parts), default=None)

    def affected_modules(
        self,
        changed_paths: Iterable[str],
        *,
        upstream_depth: int = 1,
        downstream_depth: int = 2,
        max_modules: int = 64,
    ) -> tuple[MavenModule, ...]:
        starts = {
            module.path
            for path in changed_paths
            if (module := self.nearest_module(path)) is not None
        }
        dependencies = {module.path: set(module.dependencies) for module in self.modules}
        dependents = {module.path: set() for module in self.modules}
        for module, required in dependencies.items():
            for dependency in required:
                dependents.setdefault(dependency, set()).add(module)

        selected = set(starts)

        def expand(graph: dict[str, set[str]], depth: int) -> None:
            frontier = set(starts)
            for _ in range(max(0, depth)):
                frontier = {
                    target
                    for source in frontier
                    for target in graph.get(source, ())
                    if target not in selected
                }
                if not frontier:
                    return
                selected.update(frontier)

        expand(dependencies, upstream_depth)
        expand(dependents, downstream_depth)
        ordered = [module for module in self.modules if module.path in selected]
        if len(ordered) <= max_modules:
            return tuple(ordered)
        # A bound may trim neighbours, never the changed module itself.
        mandatory = [module for module in ordered if module.path in starts][:max_modules]
        room = max_modules - len(mandatory)
        chosen = {module.path for module in mandatory}
        for module in ordered:
            if room <= 0:
                break
            if module.path in chosen:
                continue
            chosen.add(module.path)
            room -= 1
        return tuple(module for module in ordered if module.path in chosen)

    def existing_generated_roots(self, module_path: str) -> tuple[Path, ...]:
        module = self.module(module_path)
        if module is None:
            return ()
        roots = (*module.generated_source_roots, *module.generated_test_roots)
        return tuple(self.root / path for path in roots if (self.root / path).is_dir())

    def is_configured_generated_path(self, path: str) -> bool:
        absolute = (self.root / path).resolve()
        for module in self.modules:
            for generated in (*module.generated_source_roots, *module.generated_test_roots):
                candidate = (self.root / generated).resolve()
                if candidate.is_dir() and (
                    absolute == candidate or candidate in absolute.parents
                ):
                    return True
        return False


def load_maven_reactor(root: Path) -> MavenReactor:
    return MavenReactor.load(root)


def is_configured_generated_path(root: Path, path: str) -> bool:
    return load_maven_reactor(root).is_configured_generated_path(path)
