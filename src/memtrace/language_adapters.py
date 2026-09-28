"""Bounded, dependency-tolerant source navigation (never acceptance evidence).

Python's existing graph visitor and Reference IDs remain authoritative. Other
languages use pinned Tree-sitter grammars, not regex-generated call graphs.
Only Groovy has a lexical *symbol-only* fallback. No adapter scans a repository.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import re
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

_EXTENSIONS = {
    ".py": "python",
    ".pyi": "python",
    ".go": "go",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".java": "java",
    ".groovy": "groovy",
}
_ALIASES = {
    "py": "python",
    "python3": "python",
    "golang": "go",
    "rs": "rust",
    "ts": "typescript",
    "tsx": "typescript",
    "js": "javascript",
    "jsx": "javascript",
}
PARSER_VERSIONS = {
    "tree-sitter": "0.25.2",
    "tree-sitter-go": "0.25.0",
    "tree-sitter-rust": "0.24.2",
    "tree-sitter-typescript": "0.23.2",
    "tree-sitter-javascript": "0.25.0",
    "tree-sitter-java": "0.23.5",
}
_DECLARATIONS = {
    "go": {"function_declaration", "method_declaration", "type_spec", "method_elem"},
    "rust": {
        "function_item",
        "function_signature_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
        "type_item",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "abstract_class_declaration",
        "method_definition",
        "method_signature",
        "abstract_method_signature",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
    },
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "java": {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
        "annotation_type_declaration",
        "method_declaration",
        "constructor_declaration",
        "compact_constructor_declaration",
    },
}


@dataclass(frozen=True, slots=True)
class SymbolRecord:
    qualified_name: str
    symbol_kind: str
    line_start: int
    line_end: int
    signature: str
    is_test: bool = False


@dataclass(frozen=True, slots=True)
class RelationRecord:
    relation: str
    source: str | None
    target: str
    line: int
    alias: str | None = None
    imported_name: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class AdapterProjection:
    language: str
    symbols: tuple[SymbolRecord, ...]
    relations: tuple[RelationRecord, ...]
    parser_backend: str
    parser_confidence: float
    module: str = ""
    degraded_reason: str | None = None


class LanguageAdapter(Protocol):
    language: str

    def detect_language(self, path: str | Path, declared_language: str | None = None) -> str: ...
    def project(self, source: str, path: str | Path) -> AdapterProjection: ...
    def index_symbols(self, source: str, path: str | Path) -> tuple[SymbolRecord, ...]: ...
    def index_imports(self, source: str, path: str | Path) -> tuple[RelationRecord, ...]: ...
    def index_calls(self, source: str, path: str | Path) -> tuple[RelationRecord, ...]: ...
    def index_tests(self, source: str, path: str | Path) -> tuple[SymbolRecord, ...]: ...
    def build_line_ranges(self, source: str, path: str | Path) -> tuple[tuple[int, int], ...]: ...


def normalize_language(value: str | None, path: str | Path | None = None) -> str:
    value = str(value or "").strip().lower()
    return (
        _ALIASES.get(value, value)
        if value
        else _EXTENSIONS.get(Path(path or "").suffix.lower(), "unknown")
    )


def language_for_path(path: str | Path, declared_language: str | None = None) -> str:
    # Task metadata is a fallback, not a cast of every file in a mixed repo.
    # E.g. a TypeScript task's JS tests must use the JavaScript grammar.
    suffix = Path(path).suffix.lower()
    if suffix in _EXTENSIONS:
        return _EXTENSIONS[suffix]
    return normalize_language(declared_language) if not suffix else "unknown"


def supported_languages() -> tuple[str, ...]:
    return ("python", "go", "rust", "typescript", "javascript", "java", "groovy")


@dataclass(frozen=True, slots=True)
class GenericLanguageAdapter:
    language: str

    def detect_language(self, path: str | Path, declared_language: str | None = None) -> str:
        return language_for_path(path, declared_language)

    def project(self, source: str, path: str | Path) -> AdapterProjection:
        return index_source(source, path, self.language)

    def index_symbols(self, source: str, path: str | Path) -> tuple[SymbolRecord, ...]:
        return self.project(source, path).symbols

    def index_imports(self, source: str, path: str | Path) -> tuple[RelationRecord, ...]:
        return tuple(r for r in self.project(source, path).relations if r.relation == "IMPORTS")

    def index_calls(self, source: str, path: str | Path) -> tuple[RelationRecord, ...]:
        return tuple(r for r in self.project(source, path).relations if r.relation == "CALLS")

    def index_tests(self, source: str, path: str | Path) -> tuple[SymbolRecord, ...]:
        return tuple(s for s in self.project(source, path).symbols if s.is_test)

    def build_line_ranges(self, source: str, path: str | Path) -> tuple[tuple[int, int], ...]:
        return tuple((s.line_start, s.line_end) for s in self.project(source, path).symbols)


class PythonLanguageAdapter(GenericLanguageAdapter):
    def __init__(self) -> None:
        super().__init__("python")


def get_language_adapter(language: str | None, path: str | Path | None = None) -> LanguageAdapter:
    name = normalize_language(language, path)
    return PythonLanguageAdapter() if name == "python" else GenericLanguageAdapter(name)


_LOCAL = threading.local()
_CACHE_LOCK = threading.RLock()
_CACHE: OrderedDict[tuple[str, str, str, str], AdapterProjection] = OrderedDict()


def clear_index_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def _tree_sitter_parser(language: str, dialect: str = "") -> tuple[Any, str] | None:
    if language not in _DECLARATIONS:
        return None
    key = (language, dialect)
    parsers = getattr(_LOCAL, "parsers", {})
    if key in parsers:
        return parsers[key]
    try:
        for package in ("tree-sitter", f"tree-sitter-{language}"):
            if importlib.metadata.version(package) != PARSER_VERSIONS[package]:
                return None  # do not silently use an unpinned global grammar
        core = importlib.import_module("tree_sitter")
        grammar = importlib.import_module(f"tree_sitter_{language}")
        factory = (
            "language_tsx"
            if dialect == ".tsx"
            else "language_typescript"
            if language == "typescript"
            else "language"
        )
        result = (
            core.Parser(core.Language(getattr(grammar, factory)())),
            f"tree-sitter:{language}",
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        return None
    parsers[key] = result
    _LOCAL.parsers = parsers
    return result


def _text(node: Any) -> str:
    return node.text.decode("utf-8", errors="replace") if node is not None else ""


def _field(node: Any, name: str) -> Any:
    return node.child_by_field_name(name)


def _descendants(node: Any):
    yield node
    for child in node.named_children:
        yield from _descendants(child)


def _signature(node: Any, source: bytes) -> str:
    body = _field(node, "body")
    if body is None:
        body = next(
            (
                c
                for c in node.named_children
                if c.type in {"struct_type", "interface_type", "enum_body"}
            ),
            None,
        )
    end = body.start_byte if body is not None else node.end_byte
    return source[node.start_byte : end].decode("utf-8", errors="replace").strip().rstrip(";")


def _python_projection(source: str, path: str) -> AdapterProjection:
    try:
        root = ast.parse(source, filename=path)
    except (SyntaxError, ValueError, RecursionError):
        return AdapterProjection("python", (), (), "python-ast", 0, degraded_reason="syntax_error")
    symbols: list[SymbolRecord] = []
    relations: list[RelationRecord] = []
    lines = source.splitlines()

    def walk(node: ast.AST, scope: tuple[str, ...]) -> None:
        next_scope = scope
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = ".".join((*scope, node.name))
            body_line = node.body[0].lineno if node.body else node.lineno + 1
            signature = "\n".join(lines[node.lineno - 1 : max(node.lineno, body_line - 1)])
            if body_line == node.lineno:
                signature = (
                    lines[node.lineno - 1].encode()[: node.body[0].col_offset].decode().strip()
                )
            symbols.append(
                SymbolRecord(
                    name,
                    "Class" if isinstance(node, ast.ClassDef) else "Function",
                    node.lineno,
                    node.end_lineno or node.lineno,
                    signature,
                    node.name.startswith("test_"),
                )
            )
            next_scope = (*scope, node.name)
        if isinstance(node, ast.Import):
            relations.extend(
                RelationRecord("IMPORTS", None, a.name, node.lineno, a.asname or a.name)
                for a in node.names
            )
        if isinstance(node, ast.ImportFrom):
            relations.extend(
                RelationRecord(
                    "IMPORTS",
                    None,
                    "." * node.level + (node.module or ""),
                    node.lineno,
                    a.asname or a.name,
                    a.name,
                )
                for a in node.names
            )
        if isinstance(node, ast.Call):
            relations.append(
                RelationRecord(
                    "CALLS", ".".join(scope) or None, ast.unparse(node.func), node.lineno
                )
            )
        for child in ast.iter_child_nodes(node):
            walk(child, next_scope)

    walk(root, ())
    return _finish("python", symbols, relations, "python-ast")


def _finish(
    language: str,
    symbols: list[SymbolRecord],
    relations: list[RelationRecord],
    backend: str,
    module: str = "",
    confidence: float = 1.0,
) -> AdapterProjection:
    # Overloads share a user-facing name but must not alias one another. Calls
    # to the unsuffixed ambiguous name stay external; no guessed overload.
    counts = Counter(s.qualified_name for s in symbols)
    if language != "python":
        symbols = [
            replace(
                s,
                qualified_name=s.qualified_name
                + "#"
                + hashlib.sha256(s.signature.encode()).hexdigest()[:12],
            )
            if counts[s.qualified_name] > 1
            else s
            for s in symbols
        ]
        adjusted = []
        for r in relations:
            if r.source and counts[r.source] > 1:
                owners = [
                    s
                    for s in symbols
                    if s.qualified_name.startswith(r.source + "#")
                    and s.line_start <= r.line <= s.line_end
                ]
                r = (
                    replace(r, source=owners[0].qualified_name)
                    if len(owners) == 1
                    else replace(r, source=None)
                )
            adjusted.append(r)
        relations = adjusted
    relations.extend(
        RelationRecord("DEFINES", None, s.qualified_name, s.line_start) for s in symbols
    )
    return AdapterProjection(
        language,
        tuple(dict.fromkeys(symbols)),
        tuple(dict.fromkeys(relations)),
        backend,
        confidence,
        module,
    )


def _imports(node: Any, language: str) -> list[RelationRecord]:
    raw, line = _text(node), node.start_point.row + 1
    if language == "go" and node.type == "import_spec":
        target = _text(_field(node, "path")).strip('"`')
        alias = _text(_field(node, "name")) or target.rsplit("/", 1)[-1]
        return [RelationRecord("IMPORTS", None, target, line, alias)]
    if language in {"typescript", "javascript"} and node.type == "import_statement":
        target = _text(_field(node, "source")).strip("'\"")
        result = []
        for c in _descendants(node):
            if c.type == "import_specifier":
                name = _text(_field(c, "name"))
                result.append(
                    RelationRecord(
                        "IMPORTS", None, target, line, _text(_field(c, "alias")) or name, name
                    )
                )
            elif c.type == "namespace_import":
                result.append(
                    RelationRecord("IMPORTS", None, target, line, _text(c.named_children[-1]))
                )
            elif c.type == "import_clause":
                for ident in c.named_children:
                    if ident.type == "identifier":
                        result.append(
                            RelationRecord("IMPORTS", None, target, line, _text(ident), "default")
                        )
        return result or [RelationRecord("IMPORTS", None, target, line)]
    if language == "rust" and node.type == "use_declaration":
        # Brace trees and aliases are syntax nodes; recurse with the prefix.
        result = []

        def use(n: Any, prefix: str = "") -> None:
            if n.type == "scoped_use_list":
                p = _text(_field(n, "path"))
                for c in n.named_children:
                    if c.type == "use_list":
                        use(c, prefix + p + "::")
            elif n.type == "use_list":
                for c in n.named_children:
                    use(c, prefix)
            elif n.type == "use_as_clause":
                full = prefix + _text(_field(n, "path"))
                result.append(
                    RelationRecord(
                        "IMPORTS",
                        None,
                        full.rsplit("::", 1)[0] if "::" in full else "",
                        line,
                        _text(_field(n, "alias")),
                        full.rsplit("::", 1)[-1],
                    )
                )
            elif n.type in {"scoped_identifier", "identifier", "crate", "self", "super"}:
                full = prefix + _text(n)
                result.append(
                    RelationRecord(
                        "IMPORTS",
                        None,
                        full.rsplit("::", 1)[0] if "::" in full else "",
                        line,
                        full.rsplit("::", 1)[-1],
                        full.rsplit("::", 1)[-1],
                    )
                )

        for c in node.named_children:
            use(c)
        return result
    if language == "java" and node.type == "import_declaration":
        static = bool(re.match(r"^import\s+static\s+", raw.strip()))
        imported = re.sub(r"^import\s+(?:static\s+)?|;\s*$", "", raw).strip()
        target = imported.rsplit(".", 1)[0] if static else imported
        member = imported.rsplit(".", 1)[-1] if static else None
        return [
            RelationRecord(
                "IMPORTS",
                None,
                target,
                line,
                member if static else target.rsplit(".", 1)[-1],
                member,
            )
        ]
    return []


def _tree_projection(source: str, path: str, language: str) -> AdapterProjection:
    loaded = _tree_sitter_parser(language, Path(path).suffix)
    if loaded is None:
        return AdapterProjection(
            language, (), (), "unavailable", 0, degraded_reason="pinned_parser_unavailable"
        )
    parser, backend = loaded
    payload = source.encode()
    try:
        root = parser.parse(payload).root_node
        if root.has_error:
            return AdapterProjection(language, (), (), backend, 0, degraded_reason="syntax_error")
    except (ValueError, TypeError, RecursionError):
        return AdapterProjection(language, (), (), backend, 0, degraded_reason="parse_failed")
    symbols: list[SymbolRecord] = []
    relations: list[RelationRecord] = []
    module = ""
    shadowed: dict[str, set[str]] = {}

    def visit(node: Any, scope: tuple[str, ...], receivers: dict[str, str]) -> None:
        nonlocal module
        kind, name = node.type, _text(_field(node, "name"))
        line = node.start_point.row + 1
        next_scope, next_receivers = scope, receivers
        if kind in {"package_clause", "package_declaration"}:
            module = re.sub(r"^package\s+|;\s*$", "", _text(node)).strip()
        if language == "rust" and kind == "impl_item":
            owner = _text(_field(node, "type"))
            trait = _text(_field(node, "trait"))
            # impl is a scope, not a second definition of the struct itself.
            if owner:
                next_scope = (*scope, owner + ("@" + trait if trait else ""))
                next_receivers = {
                    **receivers,
                    "self": ".".join(next_scope),
                    "Self": ".".join(next_scope),
                }
        declaration = kind in _DECLARATIONS[language]
        declaration_node = node
        if language in {"typescript", "javascript"} and kind == "variable_declarator":
            value = _field(node, "value")
            if value is not None and value.type in {
                "arrow_function",
                "function_expression",
                "generator_function",
            }:
                declaration, declaration_node = True, value
        if declaration and name:
            symbol_kind = (
                "Class"
                if "class" in kind or kind == "record_declaration"
                else "Interface"
                if "interface" in kind or kind == "annotation_type_declaration"
                else "Struct"
                if "struct" in kind
                else "Trait"
                if "trait" in kind
                else "Enum"
                if "enum" in kind
                else "Module"
                if kind == "mod_item"
                else "Type"
                if "type" in kind
                else "Function"
            )
            owner_scope = scope
            if language == "go" and kind == "method_declaration":
                receiver = _field(node, "receiver")
                param = receiver.named_children[0] if receiver and receiver.named_children else None
                owner = _text(_field(param, "type")) if param is not None else ""
                owner = re.sub(r"^\*|\[.*\]$", "", owner)
                if owner:
                    owner_scope = (*scope, owner)
                    next_receivers = {
                        **receivers,
                        _text(_field(param, "name")): ".".join(owner_scope),
                    }
            if kind == "type_spec":
                type_node = _field(node, "type")
                symbol_kind = {"struct_type": "Struct", "interface_type": "Interface"}.get(
                    type_node.type if type_node else "", "Type"
                )
            qualified = ".".join((*owner_scope, name))
            if language in {"typescript", "javascript"}:
                parent = node.parent
                if parent is not None and parent.type in {
                    "lexical_declaration",
                    "variable_declaration",
                }:
                    parent = parent.parent
                if parent is not None and parent.type == "export_statement":
                    alias = (
                        "default" if _text(parent).lstrip().startswith("export default ") else name
                    )
                    relations.append(RelationRecord("EXPORTS", None, qualified, line, alias))
            if symbol_kind == "Function":
                params = _field(declaration_node, "parameters")
                blocked = set()
                if params is not None:
                    for p in _descendants(params):
                        n = _field(p, "name") or _field(p, "pattern")
                        if n is not None and n.type in {
                            "identifier",
                            "shorthand_property_identifier_pattern",
                        }:
                            blocked.add(_text(n))
                        elif p.type in {"identifier", "shorthand_property_identifier_pattern"}:
                            blocked.add(_text(p))
                shadowed[qualified] = blocked
            # Coverage is a navigation hint, not a claim that a verifier ran.
            # Read actual declaration attributes, never annotation-like text in
            # a method body (or a Rust helper merely named test_something).
            test = False
            if language == "rust" and symbol_kind == "Function":
                sibling = node.prev_named_sibling
                attributes = []
                while sibling is not None and sibling.type in {
                    "attribute_item",
                    "line_comment",
                    "block_comment",
                }:
                    if sibling.type == "attribute_item":
                        attributes.append(_text(sibling))
                    sibling = sibling.prev_named_sibling
                test = any(
                    re.search(r"#\[\s*(?:\w+::)*test\s*(?:\]|\()", attr) for attr in attributes
                )
            elif language == "go" and kind == "function_declaration":
                test = path.endswith("_test.go") and bool(
                    re.match(r"^(?:Test|Benchmark|Example|Fuzz)(?:[A-Z_]|$)", name)
                )
            elif language == "java" and kind == "method_declaration":
                modifiers = next((c for c in node.named_children if c.type == "modifiers"), None)
                test = modifiers is not None and any(
                    _text(_field(c, "name")).rsplit(".", 1)[-1]
                    in {"Test", "ParameterizedTest", "RepeatedTest", "TestFactory", "TestTemplate"}
                    for c in modifiers.named_children
                    if c.type in {"annotation", "marker_annotation"}
                )
            signature = _signature(declaration_node, payload)
            if declaration_node is not node:
                signature = name + " = " + signature
            symbols.append(
                SymbolRecord(qualified, symbol_kind, line, node.end_point.row + 1, signature, test)
            )
            next_scope = (*owner_scope, name)
            if symbol_kind in {"Class", "Struct", "Interface"}:
                next_receivers = {**receivers, "this": qualified}
        relations.extend(_imports(node, language))
        if language in {"typescript", "javascript"} and kind == "export_specifier":
            original = _text(_field(node, "name"))
            alias = _text(_field(node, "alias")) or original
            relations.append(RelationRecord("EXPORTS", None, original, line, alias))
        if language == "java" and kind in {
            "class_declaration",
            "interface_declaration",
            "record_declaration",
        }:
            for field, relation in (("superclass", "EXTENDS"), ("interfaces", "IMPLEMENTS")):
                clause = _field(node, field)
                if clause is not None:
                    children = clause.named_children
                    if children and children[0].type == "type_list":
                        children = children[0].named_children
                    for base in children:
                        relations.append(
                            RelationRecord(relation, ".".join(next_scope), _text(base), line)
                        )
        if language == "java" and kind in {"annotation", "marker_annotation"}:
            annotation = _text(_field(node, "name")).strip()
            if annotation:
                relations.append(
                    RelationRecord(
                        "REFERENCES",
                        ".".join(scope) or None,
                        annotation,
                        line,
                    )
                )
        if language == "java" and kind == "object_creation_expression":
            created = re.sub(r"<.*>", "", _text(_field(node, "type"))).strip()
            if created:
                simple = created.rsplit(".", 1)[-1]
                relations.append(
                    RelationRecord(
                        "CALLS",
                        ".".join(scope) or None,
                        created + "." + simple,
                        line,
                    )
                )
        if language == "java" and kind == "method_reference":
            reference = re.sub(r"\s+", "", _text(node)).replace("::", ".")
            if reference:
                relations.append(
                    RelationRecord(
                        "REFERENCES",
                        ".".join(scope) or None,
                        reference,
                        line,
                    )
                )
        if kind in {"call_expression", "method_invocation"}:
            target = _text(_field(node, "function"))
            if kind == "method_invocation":
                obj = _text(_field(node, "object"))
                target = (obj + "." if obj else "") + _text(_field(node, "name"))
            for receiver, owner in receivers.items():
                if receiver and target.startswith(receiver + "."):
                    target = owner + target[len(receiver) :]
                    break
            if target:
                # A function parameter named helper shadows a module helper.
                # Without type/scope proof it must remain an external hint.
                first = re.split(r"[.:]", target, 1)[0]
                if first in shadowed.get(".".join(scope), set()):
                    target = "dynamic:" + target
                relations.append(RelationRecord("CALLS", ".".join(scope) or None, target, line))
            # JS test/it callback gets an addressable, stable labelled Section.
            if language in {"typescript", "javascript"} and target in {
                "test",
                "it",
                "test.only",
                "it.only",
                "test.skip",
                "it.skip",
                "describe",
                "describe.only",
                "describe.skip",
            }:
                args = _field(node, "arguments")
                args = args.named_children if args is not None else []
                if (
                    args
                    and args[0].type in {"string", "template_string"}
                    and any(a.type in {"arrow_function", "function_expression"} for a in args)
                ):
                    label = _text(args[0]).strip("'\"`")
                    marker = "suite" if target.startswith("describe") else "test"
                    q = ".".join((*scope, marker + "[" + label + "]"))
                    symbols.append(
                        SymbolRecord(
                            q,
                            "Function",
                            line,
                            node.end_point.row + 1,
                            f"{target}({label!r})",
                            marker == "test",
                        )
                    )
                    next_scope = (*scope, marker + "[" + label + "]")
        if language in {"typescript", "javascript"} and kind == "assignment_expression":
            left, right = _field(node, "left"), _field(node, "right")
            name = _text(left)
            if right is not None and right.type == "identifier":
                if name.startswith("exports.") or name.startswith("module.exports."):
                    relations.append(RelationRecord("EXPORTS", None, _text(right), line, name.rsplit(".", 1)[-1]))
                elif name == "module.exports":
                    relations.append(RelationRecord("EXPORTS", None, _text(right), line, "default"))
            elif name == "module.exports" and right is not None and right.type == "object":
                for member in right.named_children:
                    if member.type == "shorthand_property_identifier":
                        relations.append(RelationRecord("EXPORTS", None, _text(member), line, _text(member)))
                    elif member.type == "pair":
                        value = _field(member, "value")
                        if value is not None and value.type == "identifier":
                            alias = _text(_field(member, "key")).strip("'\"")
                            relations.append(RelationRecord("EXPORTS", None, _text(value), line, alias))
        # CommonJS imports are structurally recognized, not arbitrary calls.
        if language == "javascript" and kind == "variable_declarator":
            value = _field(node, "value")
            if (
                value is not None
                and value.type == "call_expression"
                and _text(_field(value, "function")) == "require"
            ):
                args = _field(value, "arguments")
                if (
                    args is not None
                    and len(args.named_children) == 1
                    and args.named_children[0].type == "string"
                ):
                    target = _text(args.named_children[0]).strip("'\"")
                    binding = _field(node, "name")
                    if binding.type == "identifier":
                        relations.append(
                            RelationRecord("IMPORTS", None, target, line, _text(binding))
                        )
                    elif binding.type == "object_pattern":
                        for c in binding.named_children:
                            original = _text(_field(c, "key")) or _text(c)
                            alias = _text(_field(c, "value")) or original
                            relations.append(
                                RelationRecord("IMPORTS", None, target, line, alias, original)
                            )
        for child in node.named_children:
            visit(child, next_scope, next_receivers)

    try:
        visit(root, (), {})
    except RecursionError:
        return AdapterProjection(language, (), (), backend, 0, degraded_reason="nesting_limit")
    return _finish(language, symbols, relations, backend, module)


def _groovy_projection(source: str) -> AdapterProjection:
    # Groovy/Spock supports runtime metaprogramming. Without a fixed reliable
    # grammar expose conservative declarations only, NEVER CALLS/COVERS.
    masked = re.sub(
        r"/\*.*?\*/|//[^\n]*|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"",
        lambda m: re.sub(r"[^\n]", " ", m.group()),
        source,
        flags=re.S,
    )
    lines, mask_lines = source.splitlines(), masked.splitlines()
    symbols: list[SymbolRecord] = []
    scopes: list[tuple[str, int, int]] = []
    depth = 0
    for i, (line, mask) in enumerate(zip(lines, mask_lines), 1):
        match = re.match(
            r"\s*(?:(?:public|private|protected|static|final|abstract)\s+)*class\s+(\w+)", mask
        )
        kind = "Class"
        if match is None:
            match = re.match(
                r"\s*(?:(?:public|private|protected|static|final)\s+)*(?:def|void|boolean|int|String)\s+(\w+)\s*\(",
                mask,
            )
            kind = "Function"
        name = match.group(1) if match else None
        if not name and re.match(r"\s*def\s+", mask) and re.match(r"\s*def\s+['\"]", line):
            spock = re.match(r"\s*def\s+(['\"])(.*?)\1\s*\(", line)
            name = "feature[" + spock.group(2) + "]" if spock else None
        if name:
            q = ".".join((*[s[0] for s in scopes], name))
            symbols.append(
                SymbolRecord(
                    q,
                    kind,
                    i,
                    i,
                    line.split("{", 1)[0].strip(),
                    name.startswith(("test", "feature[")),
                )
            )
            if mask.count("{") > mask.count("}"):
                scopes.append((name, depth + 1, len(symbols) - 1))
        depth += mask.count("{") - mask.count("}")
        if depth < 0:
            return AdapterProjection(
                "groovy", (), (), "lexical-fallback", 0, degraded_reason="unbalanced_braces"
            )
        while scopes and depth < scopes[-1][1]:
            _, _, index = scopes.pop()
            symbols[index] = replace(symbols[index], line_end=i)
    if depth or scopes:
        return AdapterProjection(
            "groovy", (), (), "lexical-fallback", 0, degraded_reason="unbalanced_braces"
        )
    # Spock labels are executable regions inside a feature method.  They are
    # useful page-in addresses, but Groovy's dynamic dispatch is deliberately
    # not represented as a call graph.  Keep the names deterministic and
    # bounded by the containing feature so strings/comments cannot create
    # phantom blocks.
    features = [
        symbol for symbol in symbols
        if ".feature[" in symbol.qualified_name or symbol.qualified_name.startswith("feature[")
    ]
    block_symbols: list[SymbolRecord] = []
    for feature in features:
        blocks: list[tuple[str, int]] = []
        for line_no in range(feature.line_start, feature.line_end + 1):
            if line_no > len(mask_lines):
                break
            match = re.match(r"\s*(setup|given|when|then|expect|where)\s*:\s*", mask_lines[line_no - 1])
            if match:
                blocks.append((match.group(1), line_no))
        occurrences: Counter[str] = Counter()
        for index, (label, start) in enumerate(blocks):
            occurrences[label] += 1
            block_name = label if occurrences[label] == 1 else f"{label}[{occurrences[label]}]"
            end = (blocks[index + 1][1] - 1) if index + 1 < len(blocks) else feature.line_end
            block_symbols.append(
                SymbolRecord(
                    feature.qualified_name + "." + block_name,
                    "Block",
                    start,
                    max(start, end),
                    label + ":",
                    True,
                )
            )
    symbols.extend(block_symbols)
    imports: list[RelationRecord] = []
    explicit_types: dict[str, str] = {}
    static_members: dict[str, tuple[str, str]] = {}
    for match in re.finditer(
        r"(?m)^\s*import\s+(static\s+)?([\w.]+(?:\.\*)?)(?:\s+as\s+(\w+))?",
        masked,
    ):
        static, imported, renamed = bool(match.group(1)), match.group(2), match.group(3)
        line = source[: match.start()].count("\n") + 1
        if static:
            owner, _, member = imported.rpartition(".")
            if not owner or member == "*":
                imports.append(
                    RelationRecord("IMPORTS", None, owner or imported, line, "*", member or "*")
                )
                continue
            alias = renamed or member
            imports.append(RelationRecord("IMPORTS", None, owner, line, alias, member))
            static_members[alias] = (owner, member)
        else:
            alias = renamed or imported.rsplit(".", 1)[-1]
            imports.append(RelationRecord("IMPORTS", None, imported, line, alias))
            if alias != "*":
                explicit_types[alias] = imported

    # Dynamic receiver dispatch remains deliberately absent. Only an exact
    # static import is strong enough to create a Groovy CALLS edge.
    declared_functions = {
        symbol.qualified_name.rsplit(".", 1)[-1]
        for symbol in symbols
        if symbol.symbol_kind == "Function"
    }
    exact: list[RelationRecord] = []
    for alias, (_owner, _member) in static_members.items():
        if alias in declared_functions:
            continue
        for match in re.finditer(rf"\b{re.escape(alias)}\s*\(", masked):
            line = masked[: match.start()].count("\n") + 1
            owner = next(
                (
                    symbol
                    for symbol in symbols
                    if symbol.symbol_kind == "Function"
                    and symbol.line_start <= line <= symbol.line_end
                ),
                None,
            )
            exact.append(
                RelationRecord(
                    "CALLS",
                    owner.qualified_name if owner else None,
                    alias,
                    line,
                    confidence=1.0,
                )
            )

    # A Spock feature that names an explicitly imported Java type is a safe
    # navigation-only coverage hint. It is not runtime verification evidence.
    for feature in features:
        region = "\n".join(mask_lines[feature.line_start - 1 : feature.line_end])
        for alias in explicit_types:
            match = re.search(rf"\b{re.escape(alias)}\b", region)
            if match is None:
                continue
            line = feature.line_start + region[: match.start()].count("\n")
            exact.append(
                RelationRecord(
                    "COVERS",
                    feature.qualified_name,
                    alias,
                    line,
                    confidence=1.0,
                )
            )
    return _finish(
        "groovy",
        symbols,
        [*imports, *exact],
        "lexical-fallback",
        confidence=0.4,
    )


def index_source(
    source: str, path: str | Path, declared_language: str | None = None, *, revision: str = ""
) -> AdapterProjection:
    language = language_for_path(path, declared_language)
    # Digest+revision+path includes dialect and prevents cross-revision stale
    # text. Cache owns projections, not source strings or live parser trees.
    key = (language, hashlib.sha256(source.encode()).hexdigest(), revision, str(path))
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    try:
        if language == "python":
            result = _python_projection(source, str(path))
        elif language == "groovy":
            result = _groovy_projection(source)
        else:
            result = _tree_projection(source, str(path), language)
    except Exception as error:
        # Optional navigation must not break the synchronous revision receipt.
        # Keep the file address and surface, with an explicit failure reason.
        result = AdapterProjection(
            language,
            (),
            (),
            "adapter-error",
            0,
            degraded_reason="adapter_error:" + type(error).__name__,
        )
    with _CACHE_LOCK:
        # Do not cache missing parsers: an optional runtime can become ready.
        if result.degraded_reason != "pinned_parser_unavailable":
            _CACHE[key] = result
            while len(_CACHE) > 128:
                _CACHE.popitem(last=False)
    return result


def resolve_local_call(projection: AdapterProjection, relation: RelationRecord) -> str | None:
    """Lexical scopes only. An arbitrary obj.method never resolves by suffix."""
    if projection.parser_confidence < 1:
        return None
    names = {s.qualified_name for s in projection.symbols}
    target = relation.target.replace("::", ".")
    if target in names:
        return target
    if "." in target:
        return None
    scope = (relation.source or "").split(".")[:-1]
    while scope:
        candidate = ".".join((*scope, target))
        if candidate in names:
            return candidate
        scope.pop()
    return target if target in names else None
