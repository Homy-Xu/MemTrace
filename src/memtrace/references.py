from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .contracts import stable_id, utc_now
from .database import StateDatabase
from .language_adapters import AdapterProjection, index_source, language_for_path


@dataclass(frozen=True, slots=True)
class ReferenceIdentityFactory:
    """One stable identity namespace shared by Semantic and Rich graphs."""

    repository_id: str

    def __post_init__(self) -> None:
        if not self.repository_id:
            raise ValueError("repository_id is required")

    @staticmethod
    def normalize_path(path: str) -> str:
        value = path.replace("\\", "/").strip()
        while value.startswith("./"):
            value = value[2:]
        if not value or value.startswith("/") or any(part == ".." for part in value.split("/")):
            raise ValueError("reference path must be a repository-relative path")
        return value

    def canonical_file(self, path: str) -> str:
        return f"file:{self.normalize_path(path)}"

    def canonical_symbol(self, path: str, qualified_name: str) -> str:
        name = qualified_name.strip()
        if not name:
            raise ValueError("qualified symbol name is required")
        return f"symbol:{self.normalize_path(path)}:{name}"

    def canonical_test(self, selector: str) -> str:
        value = selector.strip().replace("\\", "/")
        if not value:
            raise ValueError("test selector is required")
        return f"test:{value.removeprefix('test:')}"

    def id_for(self, canonical_entity_id: str) -> str:
        prefix, separator, suffix = canonical_entity_id.partition(":")
        if not separator or prefix not in {"file", "symbol", "test", "failure", "change"}:
            raise ValueError(f"unsupported reference entity: {canonical_entity_id}")
        if prefix == "file":
            canonical_entity_id = self.canonical_file(suffix)
        elif prefix == "symbol":
            path, separator, qualified = suffix.partition(":")
            if not separator:
                raise ValueError("symbol reference requires path and qualified name")
            canonical_entity_id = self.canonical_symbol(path, qualified)
        elif prefix == "test":
            canonical_entity_id = self.canonical_test(suffix)
        return stable_id(
            "ref_",
            {"repository": self.repository_id, "entity": canonical_entity_id},
        )

    def file(self, path: str) -> str:
        return self.id_for(self.canonical_file(path))

    def symbol(self, path: str, qualified_name: str) -> str:
        return self.id_for(self.canonical_symbol(path, qualified_name))

    def test(self, selector: str) -> str:
        return self.id_for(self.canonical_test(selector))


@dataclass(frozen=True, slots=True)
class ReferenceBindingRecord:
    """One evidence-backed semantic address observed in a workspace revision."""

    reference_id: str
    canonical_entity_id: str
    reference_kind: str
    repository_relative_path: str
    qualified_name: str | None
    line_start: int | None
    line_end: int | None
    change_scope: str
    observed_revision_id: str
    source_event_id: str
    language: str = "unknown"
    symbol_kind: str | None = None
    parser_backend: str | None = None
    parser_confidence: float = 0.0
    # Carried into the immutable execution Page, NOT duplicated in the address
    # table. Old on-disk bindings and stable IDs therefore stay compatible.
    code_surface: Mapping[str, object] | None = None


def symbols_for_file(path: Path, language: str | None = None, *, revision: str = "") -> AdapterProjection:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return AdapterProjection(language_for_path(path, language), (), (), "unavailable", 0,
                                 degraded_reason="unreadable_source")
    return index_source(source, path, language, revision=revision)


def _code_surface(source: str, projection: AdapterProjection, *, repository: str,
                  revision: str, path: str, symbol=None,
                  ranges: tuple[tuple[int, int], ...] = ()) -> Mapping[str, object]:
    lines = source.splitlines(keepends=True)
    if symbol is not None:
        selected = ((symbol.line_start, symbol.line_end),)
    else:
        selected = ranges or ((1, max(1, len(lines))),)
    selected = tuple((max(1, start), min(max(1, len(lines)), end)) for start, end in selected)
    q = symbol.qualified_name if symbol else None
    from .language_adapters import resolve_local_call
    callers = [r.source for r in projection.relations if r.relation == "CALLS" and q is not None and resolve_local_call(projection, r) == q]
    callees = [r.target for r in projection.relations if r.relation == "CALLS" and r.source == q]
    tests = [s.qualified_name for s in projection.symbols if s.is_test and s.qualified_name in callers]
    return {
        "repository": repository, "revision": revision, "language": projection.language,
        "path": path, "symbol": q, "signature": symbol.signature if symbol else None,
        "line_range": list(selected[0]) if len(selected) == 1 else [list(r) for r in selected],
        "code_excerpt": "".join(lines[selected[0][0] - 1:selected[0][1]]) if len(selected) == 1 else None,
        "code_sections": [{"line_range": [start, end], "code": "".join(lines[start - 1:end])} for start, end in selected] if len(selected) > 1 else [],
        "callers": callers, "callees": callees, "related_tests": tests,
        "implementation_decisions": [], "decision_status": "not_inferred_from_syntax",
        "parser_backend": projection.parser_backend, "parser_confidence": projection.parser_confidence,
        "degraded": symbol is None, "degraded_reason": projection.degraded_reason,
    }


@dataclass(frozen=True, slots=True)
class ReferenceResolution:
    """A deterministic address translation result; ambiguity is never guessed."""

    requested: str
    candidates: tuple[str, ...]
    containing_files: tuple[str, ...]

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def _changed_line_ranges(diff_text: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for match in _HUNK.finditer(diff_text):
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        # A pure deletion has no new lines. Bind it to the nearest surviving
        # source position so the enclosing current symbol can still be found.
        end = start if count <= 0 else start + count - 1
        ranges.append((start, end))
    return tuple(ranges)


def _python_symbols(path: Path) -> tuple[tuple[str, int, int], ...]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError):
        return ()

    symbols: list[tuple[str, int, int]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def _record(self, node: ast.AST, name: str, *, nested: bool) -> None:
            qualified = ".".join((*self.scope, name))
            start = int(getattr(node, "lineno", 0) or 0)
            end = int(getattr(node, "end_lineno", start) or start)
            symbols.append((qualified, start, end))
            if nested:
                self.scope.append(name)
                self.generic_visit(node)
                self.scope.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            self._record(node, node.name, nested=True)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._record(node, node.name, nested=True)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            self._record(node, node.name, nested=True)

    Visitor().visit(tree)
    return tuple(sorted(set(symbols), key=lambda item: (item[1], item[2], item[0])))


class ReferenceDirectory:
    """Shared Semantic/Rich address directory, separate from Page bodies.

    The directory does not decide factual truth and does not scan a repository.
    It records only files already present in the execution frontier. Page Store
    remains the authoritative source for historical detail.
    """

    def __init__(
        self,
        database: StateDatabase,
        *,
        repository_id: str,
        repository_path: Path,
    ) -> None:
        self.database = database
        self.repository_id = repository_id
        self.repository_path = Path(repository_path).expanduser().resolve()
        self.identity = ReferenceIdentityFactory(repository_id)
        self.database.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS v2_reference_bindings (
                binding_id TEXT PRIMARY KEY,
                repository_id TEXT NOT NULL,
                reference_id TEXT NOT NULL,
                canonical_entity_id TEXT NOT NULL,
                reference_kind TEXT NOT NULL,
                repository_relative_path TEXT NOT NULL,
                qualified_name TEXT,
                line_start INTEGER,
                line_end INTEGER,
                change_scope TEXT NOT NULL,
                observed_revision_id TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT 'unknown',
                symbol_kind TEXT,
                parser_backend TEXT,
                parser_confidence REAL NOT NULL DEFAULT 0.0,
                UNIQUE(repository_id, canonical_entity_id, observed_revision_id, source_event_id)
            );
            CREATE INDEX IF NOT EXISTS ix_v2_reference_entity
            ON v2_reference_bindings(repository_id,canonical_entity_id,observed_at);
            CREATE INDEX IF NOT EXISTS ix_v2_reference_symbol
            ON v2_reference_bindings(repository_id,qualified_name,repository_relative_path);
            CREATE INDEX IF NOT EXISTS ix_v2_reference_id
            ON v2_reference_bindings(repository_id,reference_id);
            CREATE TRIGGER IF NOT EXISTS v2_reference_bindings_no_update
            BEFORE UPDATE ON v2_reference_bindings
            BEGIN SELECT RAISE(ABORT, 'Reference bindings are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS v2_reference_bindings_no_delete
            BEFORE DELETE ON v2_reference_bindings
            BEGIN SELECT RAISE(ABORT, 'Reference bindings are append-only'); END;
            """
        )
        columns = {
            str(row["name"])
            for row in self.database.connection.execute(
                "PRAGMA table_info(v2_reference_bindings)"
            ).fetchall()
        }
        for name, declaration in (
            ("language", "TEXT NOT NULL DEFAULT 'unknown'"),
            ("symbol_kind", "TEXT"),
            ("parser_backend", "TEXT"),
            ("parser_confidence", "REAL NOT NULL DEFAULT 0.0"),
        ):
            if name not in columns:
                self.database.connection.execute(
                    f"ALTER TABLE v2_reference_bindings ADD COLUMN {name} {declaration}"
                )

    def _safe_relative_path(self, raw: str) -> str | None:
        try:
            relative = self.identity.normalize_path(raw)
            resolved = (self.repository_path / relative).resolve()
            resolved.relative_to(self.repository_path)
        except (OSError, ValueError):
            return None
        return relative

    def _git_frontier_diff(self, relative: str) -> str:
        """Read only the exact touched path; never rescan the repository."""

        pieces: list[str] = []
        for staged in (False, True):
            command = [
                "git",
                "diff",
                "--no-ext-diff",
                "--no-renames",
                "--unified=0",
            ]
            if staged:
                command.append("--cached")
            command.extend(("--", relative))
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.repository_path,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except (OSError, subprocess.TimeoutExpired):
                return ""
            if completed.returncode not in (0, 1):
                return ""
            if completed.stdout:
                pieces.append(completed.stdout)
        if pieces:
            return "\n".join(pieces)
        try:
            status = subprocess.run(
                [
                    "git",
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--",
                    relative,
                ],
                cwd=self.repository_path,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if status.returncode == 0 and any(
            line.startswith("?? ") for line in status.stdout.splitlines()
        ):
            # Every definition in a new source file is a changed symbol.
            return "@@ -0,0 +1,1000000000 @@\n"
        return ""

    def index_changed_files(
        self,
        *,
        revision_id: str,
        source_event_id: str,
        paths: Sequence[str],
        changes: Sequence[Mapping[str, object]] = (),
    ) -> tuple[ReferenceBindingRecord, ...]:
        """Index changed frontier files without a repository walk or model guess."""

        change_by_path = {
            str(change.get("path")): change
            for change in changes
            if str(change.get("path", "")).strip()
        }
        records: list[ReferenceBindingRecord] = []
        for raw_path in dict.fromkeys(map(str, paths)):
            relative = self._safe_relative_path(raw_path)
            if relative is None:
                continue
            absolute = self.repository_path / relative
            file_record = ReferenceBindingRecord(
                reference_id=self.identity.file(relative),
                canonical_entity_id=self.identity.canonical_file(relative),
                reference_kind="FileReference",
                repository_relative_path=relative,
                qualified_name=None,
                line_start=None,
                line_end=None,
                change_scope="CHANGED_FILE",
                observed_revision_id=revision_id,
                source_event_id=source_event_id,
                language=language_for_path(relative),
            )
            records.append(file_record)
            if not absolute.is_file():
                continue
            raw_change = change_by_path.get(raw_path, change_by_path.get(relative, {}))
            diff_text = str(raw_change.get("diff", "")) if raw_change else ""
            if not diff_text:
                # Codex command/file events do not consistently include a
                # textual diff.  The working tree is still authoritative, so
                # derive changed line ranges deterministically for this exact
                # frontier file instead of asking the model or cataloguing
                # every symbol as if it changed.
                diff_text = self._git_frontier_diff(relative)
            ranges = _changed_line_ranges(diff_text)
            language = language_for_path(relative)
            try:
                source = absolute.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            projection = index_source(source, relative, language, revision=revision_id)
            surface_symbols = {item.qualified_name: item for item in projection.symbols}
            # A real command-based edit may have no provider diff (especially
            # a new, untracked file). Imports/top-level statements also live
            # outside symbol ranges. Always preserve the observed file surface;
            # changed symbol sections add precision when ranges are known.
            records[-1] = replace(file_record, code_surface=_code_surface(
                source, projection, repository=self.repository_id, revision=revision_id,
                path=relative, ranges=ranges))
            if language == "python":
                symbol_records = tuple(
                    (s.qualified_name, s.line_start, s.line_end, "Symbol", "python-ast", 1.0)
                    for s in sorted(projection.symbols, key=lambda item: (item.line_start, item.line_end, item.qualified_name))
                )
            else:
                symbol_records = (
                    tuple(
                        (
                            item.qualified_name,
                            item.line_start,
                            item.line_end,
                            item.symbol_kind,
                            projection.parser_backend,
                            projection.parser_confidence,
                        )
                        for item in projection.symbols
                    )
                    if projection is not None
                    else ()
                )
            for (
                qualified,
                line_start,
                line_end,
                symbol_kind,
                parser_backend,
                parser_confidence,
            ) in symbol_records:
                intersects = any(
                    line_start <= changed_end and changed_start <= line_end
                    for changed_start, changed_end in ranges
                )
                scope = (
                    "CHANGED_SYMBOL"
                    if ranges and intersects
                    else "UNCHANGED_SYMBOL_IN_CHANGED_FILE"
                    if ranges
                    else "SYMBOL_CATALOG_FROM_CHANGED_FILE"
                )
                canonical = self.identity.canonical_symbol(relative, qualified)
                records.append(
                    ReferenceBindingRecord(
                        reference_id=self.identity.id_for(canonical),
                        canonical_entity_id=canonical,
                        reference_kind="SymbolReference",
                        repository_relative_path=relative,
                        qualified_name=qualified,
                        line_start=line_start,
                        line_end=line_end,
                        change_scope=scope,
                        observed_revision_id=revision_id,
                        source_event_id=source_event_id,
                        language=language,
                        symbol_kind=symbol_kind,
                        parser_backend=parser_backend,
                        parser_confidence=parser_confidence,
                        code_surface=_code_surface(
                            source, projection, repository=self.repository_id, revision=revision_id,
                            path=relative, symbol=surface_symbols.get(qualified), ranges=ranges,
                        ) if scope == "CHANGED_SYMBOL" else None,
                    )
                )
        with self.database.transaction() as connection:
            for record in records:
                connection.execute(
                    "INSERT OR IGNORE INTO v2_reference_bindings "
                    "(binding_id,repository_id,reference_id,canonical_entity_id,"
                    "reference_kind,repository_relative_path,qualified_name,line_start,"
                    "line_end,change_scope,observed_revision_id,source_event_id,observed_at,"
                    "language,symbol_kind,parser_backend,parser_confidence) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        stable_id(
                            "refbinding_",
                            {
                                "repository": self.repository_id,
                                "entity": record.canonical_entity_id,
                                "revision": revision_id,
                                "source": source_event_id,
                            },
                        ),
                        self.repository_id,
                        record.reference_id,
                        record.canonical_entity_id,
                        record.reference_kind,
                        record.repository_relative_path,
                        record.qualified_name,
                        record.line_start,
                        record.line_end,
                        record.change_scope,
                        revision_id,
                        source_event_id,
                        utc_now(),
                        record.language,
                        record.symbol_kind,
                        record.parser_backend,
                        record.parser_confidence,
                    ),
                )
        return tuple(records)

    def resolve(self, requested: str, *, limit: int = 8) -> ReferenceResolution:
        """Resolve exact/shared/symbol-suffix addresses without selecting ambiguity."""

        value = requested.strip().replace("\\", "/")
        if not value:
            return ReferenceResolution(requested, (), ())
        clauses = ["canonical_entity_id=?", "reference_id=?"]
        parameters: list[object] = [value, value]
        symbol_value = value.removeprefix("symbol:")
        if value.startswith("symbol:") or (":" not in value and "." in value):
            if ":" in symbol_value:
                raw_path, qualified = symbol_value.split(":", 1)
                try:
                    path = self.identity.normalize_path(raw_path)
                except ValueError:
                    path = ""
                if path and qualified:
                    clauses.append("(repository_relative_path=? AND qualified_name=?)")
                    parameters.extend((path, qualified))
            elif symbol_value:
                clauses.append("(qualified_name=? OR qualified_name LIKE ?)")
                parameters.extend((symbol_value, f"%.{symbol_value}"))
        sql = (
            "SELECT canonical_entity_id,repository_relative_path,MAX(observed_at) AS latest "
            "FROM v2_reference_bindings WHERE repository_id=? AND ("
            + " OR ".join(clauses)
            + ") GROUP BY canonical_entity_id,repository_relative_path "
            "ORDER BY canonical_entity_id LIMIT ?"
        )
        rows = self.database.connection.execute(
            sql,
            (self.repository_id, *parameters, max(1, limit + 1)),
        ).fetchall()
        candidates = tuple(dict.fromkeys(str(row["canonical_entity_id"]) for row in rows))
        files = tuple(
            dict.fromkeys(
                self.identity.canonical_file(str(row["repository_relative_path"])) for row in rows
            )
        )
        return ReferenceResolution(value, candidates, files)

    def resolve_for_write(
        self,
        requested: str,
        *,
        containing_files: Sequence[str] = (),
        limit: int = 8,
    ) -> ReferenceResolution:
        """Canonicalize a model-owned semantic address before it enters the WAL.

        Fully qualified addresses are syntax-checked but need not have appeared in
        the directory yet. Natural symbol names, in contrast, must translate to one
        already observed shared Reference. Module-like prefixes may be removed one
        component at a time, and companion file addresses from the same semantic
        update narrow the candidate set. The directory never guesses ambiguity.
        """

        value = requested.strip().replace("\\", "/")
        if not value:
            return ReferenceResolution(requested, (), ())

        if value.startswith("file:"):
            try:
                canonical = self.identity.canonical_file(value.removeprefix("file:"))
            except ValueError:
                return ReferenceResolution(value, (), ())
            return ReferenceResolution(value, (canonical,), (canonical,))

        if value.startswith("test:"):
            try:
                canonical = self.identity.canonical_test(value)
            except ValueError:
                return ReferenceResolution(value, (), ())
            return ReferenceResolution(value, (canonical,), ())

        prefix, separator, suffix = value.partition(":")
        if separator and prefix in {"failure", "change"}:
            if not suffix.strip():
                return ReferenceResolution(value, (), ())
            canonical = f"{prefix}:{suffix.strip()}"
            return ReferenceResolution(value, (canonical,), ())

        if not separator:
            relative = self._safe_relative_path(value)
            if relative is not None and (self.repository_path / relative).is_file():
                canonical = self.identity.canonical_file(relative)
                return ReferenceResolution(value, (canonical,), (canonical,))

        explicit_symbol = value.startswith("symbol:")
        natural_symbol = not separator and "." in value and "/" not in value
        if not explicit_symbol and not natural_symbol:
            # Semantic update entities outside the shared file/symbol/test
            # address namespace remain opaque semantic identifiers.
            return ReferenceResolution(value, (value,), ())

        symbol_value = value.removeprefix("symbol:")
        if explicit_symbol and ":" in symbol_value:
            raw_path, qualified = symbol_value.split(":", 1)
            try:
                canonical = self.identity.canonical_symbol(raw_path, qualified)
                containing = self.identity.canonical_file(raw_path)
            except ValueError:
                return ReferenceResolution(value, (), ())
            return ReferenceResolution(value, (canonical,), (containing,))

        allowed_files: set[str] = set()
        for raw_file in containing_files:
            raw_path = str(raw_file).strip().replace("\\", "/").removeprefix("file:")
            try:
                allowed_files.add(self.identity.canonical_file(raw_path))
            except ValueError:
                continue

        parts = tuple(part for part in symbol_value.split(".") if part)
        queries = tuple(".".join(parts[index:]) for index in range(len(parts)))
        for query in queries:
            resolved = self.resolve(f"symbol:{query}", limit=limit)
            candidates = resolved.candidates
            if allowed_files:
                candidates = tuple(
                    candidate
                    for candidate in candidates
                    if self._symbol_containing_file(candidate) in allowed_files
                )
            if not candidates:
                continue
            bounded = tuple(dict.fromkeys(candidates))[: max(1, limit)]
            files = tuple(
                dict.fromkeys(
                    containing
                    for candidate in bounded
                    if (containing := self._symbol_containing_file(candidate)) is not None
                )
            )
            return ReferenceResolution(value, bounded, files)
        return ReferenceResolution(value, (), ())

    def _symbol_containing_file(self, canonical_symbol: str) -> str | None:
        suffix = canonical_symbol.removeprefix("symbol:")
        raw_path, separator, qualified = suffix.partition(":")
        if not separator or not qualified:
            return None
        try:
            return self.identity.canonical_file(raw_path)
        except ValueError:
            return None
