from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..contracts import digest, stable_id
from ..language_adapters import index_source, language_for_path, resolve_local_call
from ..references import ReferenceIdentityFactory
from .models import (
    FileProjection,
    FrontierTask,
    ReferenceBinding,
    RichRelation,
    VersionNode,
    file_reference_id,
    symbol_reference_id,
    test_reference_id,
)


# Bounded preview for navigation; immutable Pages remain authoritative.
_MAX_GRAPH_CODE_SURFACE_CHARS = 16_384


def _external_reference_id(repository_id: str, kind: str, name: str) -> str:
    return stable_id(
        "extref_",
        {"repository_id": repository_id, "kind": kind, "name": name},
    )


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


@dataclass(slots=True)
class _PythonFacts:
    task: FrontierTask
    source: str
    file_ref: str
    file_version: str
    bindings: list[ReferenceBinding]
    relations: list[RichRelation]
    version_nodes: list[VersionNode]
    scope: list[str]
    test_scope: list[str | None]
    imported_symbols: dict[str, tuple[str, str]]
    imported_modules: dict[str, str]

    @property
    def path(self) -> str:
        return self.task.repository_relative_path

    def provenance(self, line: int | None = None) -> tuple[str, ...]:
        location = self.path if line is None else f"{self.path}:{line}"
        return (location, self.task.workspace_revision_id)


class _PythonVisitor(ast.NodeVisitor):
    def __init__(self, facts: _PythonFacts) -> None:
        self.facts = facts

    def _qualified(self, name: str) -> str:
        return ".".join((*self.facts.scope, name))

    def _repository_root(self) -> Path:
        root = self.facts.task.absolute_path
        for _ in Path(self.facts.path).parts:
            root = root.parent
        return root

    def _resolve_module(self, module: str, level: int = 0) -> str | None:
        current_package = list(Path(self.facts.path).parent.parts)
        if current_package == ["."]:
            current_package = []
        if level:
            remove = max(0, level - 1)
            if remove > len(current_package):
                return None
            parts = current_package[: len(current_package) - remove]
        else:
            parts = []
        parts.extend(part for part in module.split(".") if part)
        if not parts:
            return None
        root = self._repository_root()
        candidates = (
            root.joinpath(*parts).with_suffix(".py"),
            root.joinpath(*parts, "__init__.py"),
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError:
                continue
            if resolved.is_file():
                return relative
        return None

    def _bind_file(self, path: str) -> str:
        reference = file_reference_id(self.facts.task.repository_id, path)
        self.facts.bindings.append(
            ReferenceBinding(
                reference_id=reference,
                reference_kind="FileReference",
                canonical_entity_id=f"file:{path}",
                repository_relative_path=path,
            )
        )
        return reference

    def _bind_imported_symbol(self, path: str, qualified_name: str) -> str:
        reference = symbol_reference_id(self.facts.task.repository_id, path, qualified_name)
        self.facts.bindings.append(
            ReferenceBinding(
                reference_id=reference,
                reference_kind="SymbolReference",
                canonical_entity_id=f"symbol:{path}:{qualified_name}",
                repository_relative_path=path,
                qualified_name=qualified_name,
            )
        )
        return reference

    def _record_symbol(self, node: ast.AST, name: str, kind: str) -> str:
        qualified = self._qualified(name)
        reference_id = symbol_reference_id(
            self.facts.task.repository_id, self.facts.path, qualified
        )
        entity_id = f"symbol:{self.facts.path}:{qualified}"
        self.facts.bindings.append(
            ReferenceBinding(
                reference_id=reference_id,
                reference_kind="SymbolReference",
                canonical_entity_id=entity_id,
                repository_relative_path=self.facts.path,
                qualified_name=qualified,
            )
        )
        symbol_version = stable_id(
            "symver_",
            {
                "reference_id": reference_id,
                "revision_id": self.facts.task.workspace_revision_id,
                "line": getattr(node, "lineno", 0),
                "end_line": getattr(node, "end_lineno", 0),
            },
        )
        provenance = self.facts.provenance(getattr(node, "lineno", None))
        lines = self.facts.source.splitlines()
        line_start = int(getattr(node, "lineno", 1) or 1)
        line_end = int(getattr(node, "end_lineno", line_start) or line_start)
        body = getattr(node, "body", ())
        body_line = (
            int(getattr(body[0], "lineno", line_start + 1) or line_start + 1)
            if body
            else line_start + 1
        )
        signature = "\n".join(
            lines[line_start - 1 : max(line_start, body_line - 1)]
        ).strip()
        code_surface = "\n".join(lines[line_start - 1 : line_end])
        truncated = len(code_surface) > _MAX_GRAPH_CODE_SURFACE_CHARS
        if truncated:
            code_surface = code_surface[:_MAX_GRAPH_CODE_SURFACE_CHARS]
        self.facts.relations.extend(
            (
                RichRelation(
                    relation="DEFINES",
                    source_reference_id=self.facts.file_ref,
                    target_reference_id=reference_id,
                    authority="DERIVED",
                    provenance=provenance,
                ),
                RichRelation(
                    relation="RESOLVES_TO",
                    source_reference_id=reference_id,
                    target_reference_id=symbol_version,
                    authority="DERIVED",
                    provenance=provenance,
                ),
            )
        )
        self.facts.version_nodes.append(
            VersionNode(
                node_id=symbol_version,
                node_kind="SymbolVersion",
                reference_id=reference_id,
                payload={
                    "path": self.facts.path,
                    "qualified_name": qualified,
                    "symbol_kind": kind,
                    "line": line_start,
                    "end_line": line_end,
                    "signature": signature,
                    "code": code_surface,
                    "code_surface_truncated": truncated,
                    "language": "python",
                    "parser_backend": "python-ast",
                    "parser_confidence": 1.0,
                    "revision_id": self.facts.task.workspace_revision_id,
                },
            )
        )
        return reference_id

    def _visit_callable(self, node: ast.AST, name: str, kind: str) -> None:
        symbol_ref = self._record_symbol(node, name, kind)
        test_ref: str | None = None
        if name.startswith("test_"):
            qualified = self._qualified(name)
            selector = f"{self.facts.path}::{qualified}"
            test_ref = test_reference_id(self.facts.task.repository_id, selector)
            self.facts.bindings.append(
                ReferenceBinding(
                    reference_id=test_ref,
                    reference_kind="TestReference",
                    canonical_entity_id=f"test:{selector}",
                    repository_relative_path=self.facts.path,
                    qualified_name=qualified,
                )
            )
            test_node = stable_id(
                "testver_",
                {
                    "test_reference_id": test_ref,
                    "revision_id": self.facts.task.workspace_revision_id,
                    "line": getattr(node, "lineno", 0),
                },
            )
            self.facts.version_nodes.append(
                VersionNode(
                    node_id=test_node,
                    node_kind="Test",
                    reference_id=test_ref,
                    payload={"selector": selector, "symbol_reference_id": symbol_ref},
                )
            )
            self.facts.relations.extend(
                (
                    RichRelation(
                        relation="DEFINES",
                        source_reference_id=self.facts.file_ref,
                        target_reference_id=test_ref,
                        authority="DERIVED",
                        provenance=self.facts.provenance(getattr(node, "lineno", None)),
                    ),
                    RichRelation(
                        relation="SAME_TEST_AS",
                        source_reference_id=test_ref,
                        target_reference_id=test_node,
                        authority="DERIVED",
                        provenance=self.facts.provenance(getattr(node, "lineno", None)),
                    ),
                )
            )
        self.facts.scope.append(name)
        self.facts.test_scope.append(test_ref)
        self.generic_visit(node)
        self.facts.test_scope.pop()
        self.facts.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_callable(node, node.name, "Function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_callable(node, node.name, "Function")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._record_symbol(node, node.name, "Class")
        self.facts.scope.append(node.name)
        self.generic_visit(node)
        self.facts.scope.pop()

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            local_path = self._resolve_module(alias.name)
            if local_path is None:
                target = _external_reference_id(self.facts.task.repository_id, "module", alias.name)
            else:
                target = self._bind_file(local_path)
                local_name = alias.asname or alias.name.split(".", 1)[0]
                self.facts.imported_modules[local_name] = local_path
            self.facts.relations.append(
                RichRelation(
                    relation="IMPORTS",
                    source_reference_id=self.facts.file_ref,
                    target_reference_id=target,
                    authority="DERIVED",
                    provenance=self.facts.provenance(node.lineno),
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = "." * node.level + (node.module or "")
        local_path = self._resolve_module(node.module or "", node.level)
        target = (
            self._bind_file(local_path)
            if local_path is not None
            else _external_reference_id(self.facts.task.repository_id, "module", module)
        )
        self.facts.relations.append(
            RichRelation(
                relation="IMPORTS",
                source_reference_id=self.facts.file_ref,
                target_reference_id=target,
                authority="DERIVED",
                provenance=self.facts.provenance(node.lineno),
            )
        )
        if local_path is not None:
            for alias in node.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                self.facts.imported_symbols[local_name] = (local_path, alias.name)
                self._bind_imported_symbol(local_path, alias.name)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        called_name = _dotted_name(node.func)
        if called_name:
            if self.facts.scope:
                source = symbol_reference_id(
                    self.facts.task.repository_id,
                    self.facts.path,
                    ".".join(self.facts.scope),
                )
            else:
                source = self.facts.file_ref
            root, _, remainder = called_name.partition(".")
            imported_symbol = self.facts.imported_symbols.get(root)
            imported_module = self.facts.imported_modules.get(root)
            if imported_symbol is not None:
                target_path, imported_name = imported_symbol
                qualified = f"{imported_name}.{remainder}" if remainder else imported_name
                target = self._bind_imported_symbol(target_path, qualified)
            elif imported_module is not None and remainder:
                target = self._bind_imported_symbol(imported_module, remainder)
            elif "." not in called_name:
                target = symbol_reference_id(
                    self.facts.task.repository_id, self.facts.path, called_name
                )
            else:
                target = _external_reference_id(
                    self.facts.task.repository_id, "symbol", called_name
                )
            self.facts.relations.append(
                RichRelation(
                    relation="CALLS",
                    source_reference_id=source,
                    target_reference_id=target,
                    authority="DERIVED",
                    provenance=self.facts.provenance(node.lineno),
                )
            )
            current_test = next(
                (reference for reference in reversed(self.facts.test_scope) if reference),
                None,
            )
            if current_test is not None:
                self.facts.relations.append(
                    RichRelation(
                        relation="COVERS",
                        source_reference_id=current_test,
                        target_reference_id=target,
                        authority="DERIVED",
                        provenance=self.facts.provenance(node.lineno),
                    )
                )
        self.generic_visit(node)


def _unique_bindings(items: Iterable[ReferenceBinding]) -> tuple[ReferenceBinding, ...]:
    return tuple({item.reference_id: item for item in items}.values())


def _unique_relations(items: Iterable[RichRelation]) -> tuple[RichRelation, ...]:
    unique: dict[tuple[str, str, str], RichRelation] = {}
    for item in items:
        unique[(item.relation, item.source_reference_id, item.target_reference_id)] = item
    return tuple(unique.values())


def process_frontier_file(task: FrontierTask) -> FileProjection:
    """Parse exactly one frontier file without walking or scanning the repository."""

    payload = task.absolute_path.read_bytes()
    content_digest = digest({"bytes_hex": payload.hex()})
    path = task.repository_relative_path
    file_ref = file_reference_id(task.repository_id, path)
    file_version = stable_id(
        "filever_",
        {
            "file_reference_id": file_ref,
            "revision_id": task.workspace_revision_id,
            "content_digest": content_digest,
        },
    )
    file_binding = ReferenceBinding(
        reference_id=file_ref,
        reference_kind="FileReference",
        canonical_entity_id=f"file:{path}",
        repository_relative_path=path,
    )
    relations = [
        RichRelation(
            relation="RESOLVES_TO",
            source_reference_id=file_ref,
            target_reference_id=file_version,
            authority="DERIVED",
            provenance=(path, task.workspace_revision_id),
        )
    ]
    bindings = [file_binding]
    version_nodes = [
        VersionNode(
            node_id=file_version,
            node_kind="FileVersion",
            reference_id=file_ref,
            payload={
                "path": path,
                "revision_id": task.workspace_revision_id,
                "content_digest": content_digest,
            },
        )
    ]
    language = language_for_path(path, task.language)
    metadata: dict[str, object] = {
        "suffix": task.absolute_path.suffix.lower(),
        "reasons": task.reasons,
    }

    if language == "python":
        source = payload.decode("utf-8", errors="replace")
        line_count = max(1, len(source.splitlines()))
        version_nodes[0] = VersionNode(
            node_id=file_version,
            node_kind="FileVersion",
            reference_id=file_ref,
            payload={
                **version_nodes[0].payload,
                "language": "python",
                "code_surface_preview": source[:1200],
                "code_surface_bytes": len(payload),
                "code_surface_truncated": len(source) > 1200,
                "line_range": [1, line_count],
                "parser_backend": "python-ast",
                "parser_confidence": 1.0,
            },
        )
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as exc:
            metadata["parse_error"] = f"{exc.msg} at line {exc.lineno}"
            version_nodes[0] = VersionNode(
                node_id=file_version,
                node_kind="FileVersion",
                reference_id=file_ref,
                payload={
                    **version_nodes[0].payload,
                    "parser_confidence": 0.0,
                    "degraded": True,
                    "degraded_reason": "syntax_error",
                },
            )
        else:
            facts = _PythonFacts(
                task=task,
                source=source,
                file_ref=file_ref,
                file_version=file_version,
                bindings=bindings,
                relations=relations,
                version_nodes=version_nodes,
                scope=[],
                test_scope=[],
                imported_symbols={},
                imported_modules={},
            )
            _PythonVisitor(facts).visit(tree)
    else:
        source = payload.decode("utf-8", errors="replace")
        projection = index_source(source, path, language, revision=task.workspace_revision_id)
        language = projection.language
        metadata.update(
            {
                "parser_backend": projection.parser_backend,
                "parser_confidence": projection.parser_confidence,
                "symbol_count": len(projection.symbols),
                "degraded_reason": projection.degraded_reason,
                "navigation_index": {
                    "language": language, "parser_confidence": projection.parser_confidence,
                    "module": projection.module,
                    "symbols": [{"qualified_name": s.qualified_name, "is_test": s.is_test} for s in projection.symbols],
                    "imports": [asdict(r) for r in projection.relations if r.relation == "IMPORTS"],
                    "exports": {r.alias: r.target for r in projection.relations if r.relation == "EXPORTS"},
                    "calls": [asdict(r) for r in projection.relations if r.relation == "CALLS" and not resolve_local_call(projection, r)],
                    "references": [
                        asdict(r)
                        for r in projection.relations
                        if r.relation == "REFERENCES"
                    ],
                    "covers": [
                        asdict(r)
                        for r in projection.relations
                        if r.relation == "COVERS"
                    ],
                },
            }
        )
        if language == "rust":
            # Bounded ancestor lookup follows Cargo crate boundaries in a
            # monorepo (e.g. Boa core/engine), without indexing the workspace.
            root = task.absolute_path
            for _ in Path(path).parts:
                root = root.parent
            for directory in task.absolute_path.parents:
                if not directory.is_relative_to(root):
                    break
                manifest = directory / "Cargo.toml"
                try:
                    if manifest.is_file() and manifest.stat().st_size <= 262144:
                        import tomllib
                        cargo = tomllib.loads(manifest.read_text())
                        if "package" in cargo:
                            lib = (directory / cargo.get("lib", {}).get("path", "src/lib.rs")).resolve()
                            if lib.is_relative_to(root):
                                metadata["navigation_index"]["crate_source_root"] = lib.parent.relative_to(root).as_posix()
                            break
                except (OSError, UnicodeError, ValueError):
                    break
        if language == "go":
            root = task.absolute_path
            for _ in Path(path).parts:
                root = root.parent
            try:
                import re
                match = re.search(r"(?m)^module\s+([^\s]+)", (root / "go.mod").read_text())
                if match:
                    metadata["navigation_index"]["module_prefix"] = match.group(1)
            except (OSError, UnicodeError):
                pass
        if not projection.symbols:
            version_nodes[0] = VersionNode(
                node_id=file_version, node_kind="FileVersion", reference_id=file_ref,
                payload={**version_nodes[0].payload, "language": language, "code": source,
                         "line_range": [1, max(1, len(source.splitlines()))], "degraded": True,
                         "parser_confidence": projection.parser_confidence,
                         "degraded_reason": projection.degraded_reason},
            )
        symbol_refs: dict[str, str] = {}
        short_refs: dict[str, list[str]] = {}
        for symbol in projection.symbols:
            reference_id = symbol_reference_id(
                task.repository_id,
                path,
                symbol.qualified_name,
            )
            symbol_refs[symbol.qualified_name] = reference_id
            short_refs.setdefault(symbol.qualified_name.rsplit(".", 1)[-1], []).append(reference_id)
            bindings.append(
                ReferenceBinding(
                    reference_id=reference_id,
                    reference_kind="SymbolReference",
                    canonical_entity_id=f"symbol:{path}:{symbol.qualified_name}",
                    repository_relative_path=path,
                    qualified_name=symbol.qualified_name,
                )
            )
            symbol_version = stable_id(
                "symver_",
                {
                    "reference_id": reference_id,
                    "revision_id": task.workspace_revision_id,
                    "line": symbol.line_start,
                    "end_line": symbol.line_end,
                },
            )
            code_surface = "\n".join(
                source.splitlines()[max(0, symbol.line_start - 1) : symbol.line_end]
            )
            provenance = (path, f"{path}:{symbol.line_start}", task.workspace_revision_id)
            relations.extend(
                (
                    RichRelation(
                        relation="DEFINES",
                        source_reference_id=file_ref,
                        target_reference_id=reference_id,
                        authority="DERIVED",
                        provenance=provenance,
                        confidence=projection.parser_confidence,
                    ),
                    RichRelation(
                        relation="RESOLVES_TO",
                        source_reference_id=reference_id,
                        target_reference_id=symbol_version,
                        authority="DERIVED",
                        provenance=provenance,
                        confidence=projection.parser_confidence,
                    ),
                )
            )
            version_nodes.append(
                VersionNode(
                    node_id=symbol_version,
                    node_kind="SymbolVersion",
                    reference_id=reference_id,
                    payload={
                        "path": path,
                        "language": language,
                        "qualified_name": symbol.qualified_name,
                        "symbol_kind": symbol.symbol_kind,
                        "signature": symbol.signature,
                        "line": symbol.line_start,
                        "end_line": symbol.line_end,
                        "code": code_surface,
                        "revision_id": task.workspace_revision_id,
                        "parser_backend": projection.parser_backend,
                        "parser_confidence": projection.parser_confidence,
                    },
                )
            )
            if symbol.is_test:
                selector = f"{path}::{symbol.qualified_name}"
                test_ref = test_reference_id(task.repository_id, selector)
                bindings.append(
                    ReferenceBinding(
                        reference_id=test_ref,
                        reference_kind="TestReference",
                        canonical_entity_id=f"test:{selector}",
                        repository_relative_path=path,
                        qualified_name=symbol.qualified_name,
                    )
                )
                test_node = stable_id(
                    "testver_",
                    {
                        "test_reference_id": test_ref,
                        "revision_id": task.workspace_revision_id,
                        "line": symbol.line_start,
                    },
                )
                version_nodes.append(
                    VersionNode(
                        node_id=test_node,
                        node_kind="Test",
                        reference_id=test_ref,
                        payload={
                            "selector": selector,
                            "symbol_reference_id": reference_id,
                            "language": language,
                        },
                    )
                )
                relations.extend(
                    (
                        RichRelation(
                            relation="DEFINES",
                            source_reference_id=file_ref,
                            target_reference_id=test_ref,
                            authority="DERIVED",
                            provenance=provenance,
                            confidence=projection.parser_confidence,
                        ),
                        RichRelation(
                            relation="SAME_TEST_AS",
                            source_reference_id=test_ref,
                            target_reference_id=test_node,
                            authority="DERIVED",
                            provenance=provenance,
                            confidence=projection.parser_confidence,
                        ),
                    )
                )

        for relation in projection.relations:
            if relation.relation in {"EXTENDS", "IMPLEMENTS"}:
                target = symbol_refs.get(relation.target)
                relations.append(RichRelation(
                    relation=relation.relation, source_reference_id=symbol_refs.get(relation.source, file_ref),
                    target_reference_id=target or _external_reference_id(task.repository_id, "type", relation.target),
                    authority="DERIVED", provenance=(path, f"{path}:{relation.line}", task.workspace_revision_id),
                    confidence=1.0 if target else 0.2,
                ))
                continue
            if relation.relation == "IMPORTS":
                target = _external_reference_id(
                    task.repository_id,
                    "module",
                    relation.target,
                )
                bindings.append(ReferenceBinding(
                    reference_id=target, reference_kind="ExternalReference",
                    canonical_entity_id=f"external:{language}:module:{relation.target}",
                    repository_relative_path="", qualified_name=relation.target,
                ))
                relations.append(
                    RichRelation(
                        relation="IMPORTS",
                        source_reference_id=file_ref,
                        target_reference_id=target,
                        authority="DERIVED",
                        provenance=(path, f"{path}:{relation.line}", task.workspace_revision_id),
                        confidence=min(0.2, projection.parser_confidence),
                    )
                )
                continue
            if relation.relation != "CALLS":
                continue
            source_ref = file_ref
            if relation.source:
                source_ref = symbol_refs.get(relation.source, file_ref)
            resolved = resolve_local_call(projection, relation)
            candidates = symbol_refs.get(resolved) if resolved else None
            target_ref = (
                candidates
                if isinstance(candidates, str)
                else _external_reference_id(task.repository_id, "symbol", relation.target)
            )
            if not candidates:
                bindings.append(ReferenceBinding(
                    reference_id=target_ref, reference_kind="ExternalReference",
                    canonical_entity_id=f"external:{language}:symbol:{relation.target}",
                    repository_relative_path="", qualified_name=relation.target,
                ))
            relations.append(
                RichRelation(
                    relation="CALLS",
                    source_reference_id=source_ref,
                    target_reference_id=target_ref,
                    authority="DERIVED",
                    provenance=(path, f"{path}:{relation.line}", task.workspace_revision_id),
                    confidence=(
                        min(projection.parser_confidence, relation.confidence)
                        if isinstance(candidates, str)
                        else min(0.2, relation.confidence)
                    ),
                )
            )
            if candidates and any(s.is_test and s.qualified_name == relation.source for s in projection.symbols):
                relations.append(RichRelation(
                    relation="COVERS", source_reference_id=test_reference_id(task.repository_id, f"{path}::{relation.source}"),
                    target_reference_id=target_ref, authority="DERIVED",
                    provenance=(path, f"{path}:{relation.line}", task.workspace_revision_id, "syntactic_test_call_not_execution"),
                    confidence=min(projection.parser_confidence, relation.confidence),
                ))

    identity = ReferenceIdentityFactory(task.repository_id)
    if "modified" in task.reasons:
        change_entity = f"change:{task.workspace_revision_id}:{path}"
        change_ref = identity.id_for(change_entity)
        bindings.append(
            ReferenceBinding(
                reference_id=change_ref,
                reference_kind="ChangeReference",
                canonical_entity_id=change_entity,
                repository_relative_path=path,
            )
        )
        change_set = stable_id(
            "changeset_",
            {"change_reference_id": change_ref, "revision": task.workspace_revision_id},
        )
        version_nodes.append(
            VersionNode(
                node_id=change_set,
                node_kind="ChangeSet",
                reference_id=change_ref,
                payload={"path": path, "reasons": task.reasons},
            )
        )
        relations.append(
            RichRelation(
                relation="MAY_IMPACT",
                source_reference_id=change_ref,
                target_reference_id=file_ref,
                authority="DERIVED",
                provenance=(path, task.workspace_revision_id),
            )
        )

    for signature in task.failure_signatures:
        failure_entity = f"failure:{signature}"
        failure_ref = identity.id_for(failure_entity)
        bindings.append(
            ReferenceBinding(
                reference_id=failure_ref,
                reference_kind="FailureReference",
                canonical_entity_id=failure_entity,
                repository_relative_path=path,
            )
        )
        diagnostic = stable_id(
            "diagnostic_",
            {
                "failure_reference_id": failure_ref,
                "revision": task.workspace_revision_id,
                "path": path,
            },
        )
        version_nodes.append(
            VersionNode(
                node_id=diagnostic,
                node_kind="Diagnostic",
                reference_id=failure_ref,
                payload={"signature": signature, "path": path},
            )
        )
        relations.extend(
            (
                RichRelation(
                    relation="MATCHES_FAILURE",
                    source_reference_id=failure_ref,
                    target_reference_id=diagnostic,
                    authority="DERIVED",
                    provenance=(path, task.workspace_revision_id),
                ),
                RichRelation(
                    relation="MAY_IMPACT",
                    source_reference_id=failure_ref,
                    target_reference_id=file_ref,
                    authority="INFERRED",
                    provenance=(signature, path, task.workspace_revision_id),
                    confidence=0.7,
                ),
            )
        )

    return FileProjection(
        repository_relative_path=path,
        file_reference_id=file_ref,
        file_version_id=file_version,
        content_digest=content_digest,
        language=language,
        byte_count=len(payload),
        bindings=_unique_bindings(bindings),
        relations=_unique_relations(relations),
        version_nodes=tuple({item.node_id: item for item in version_nodes}.values()),
        metadata=metadata,
    )
