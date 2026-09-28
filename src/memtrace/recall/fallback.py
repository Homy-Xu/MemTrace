from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..contracts import FallbackStage, PageCandidate, RecallIntent
from ..semantic_memory.contracts import FallbackQueryResult


class SemanticFallbackIndex(Protocol):
    def structured_fallback(
        self,
        intent: RecallIntent,
        missing_key_digests: tuple[str, ...] | None = None,
    ) -> FallbackQueryResult: ...

    def metadata_fts(
        self,
        intent: RecallIntent,
        missing_key_digests: tuple[str, ...] | None = None,
    ) -> FallbackQueryResult: ...

    def recent_related(
        self,
        intent: RecallIntent,
        missing_key_digests: tuple[str, ...] | None = None,
    ) -> FallbackQueryResult: ...


@dataclass(frozen=True, slots=True)
class LocalSearchHit:
    relative_path: str
    line_number: int
    excerpt: str


@dataclass(frozen=True, slots=True)
class ExecutedFallback:
    stage: FallbackStage
    candidates: tuple[PageCandidate, ...]
    reason: str
    local_hits: tuple[LocalSearchHit, ...] = ()


class BoundedLocalRepositorySearch:
    """A final, bounded textual hint search; its hits never satisfy Coverage."""

    _SUFFIXES = {
        ".c",
        ".cc",
        ".cpp",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".jsx",
        ".md",
        ".py",
        ".rs",
        ".toml",
        ".ts",
        ".tsx",
        ".yaml",
        ".yml",
    }

    def __init__(
        self,
        repository_path: Path,
        *,
        max_files: int = 128,
        max_bytes: int = 1_000_000,
        max_hits: int = 16,
        max_entries: int = 2_048,
    ) -> None:
        if min(max_files, max_bytes, max_hits, max_entries) <= 0:
            raise ValueError("local search bounds must be positive")
        self.repository_path = Path(repository_path).expanduser().resolve()
        self.max_files = max_files
        self.max_bytes = max_bytes
        self.max_hits = max_hits
        self.max_entries = max_entries

    def available_for(self, intent: RecallIntent) -> bool:
        return self.repository_path.is_dir() and bool(self._terms(intent))

    def search(self, intent: RecallIntent) -> tuple[LocalSearchHit, ...]:
        terms = self._terms(intent)
        if not terms or not self.repository_path.is_dir():
            return ()
        hits: list[LocalSearchHit] = []
        files_seen = 0
        bytes_seen = 0
        entries_seen = 0
        excluded = {".git", ".venv", "node_modules", "__pycache__"}
        stop = False
        for root, directory_names, file_names in os.walk(self.repository_path, followlinks=False):
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name not in excluded and not (Path(root) / name).is_symlink()
            )
            for name in sorted(file_names):
                entries_seen += 1
                if (
                    entries_seen > self.max_entries
                    or files_seen >= self.max_files
                    or bytes_seen >= self.max_bytes
                ):
                    stop = True
                    break
                path = Path(root) / name
                if path.is_symlink() or path.suffix.lower() not in self._SUFFIXES:
                    continue
                try:
                    resolved = path.resolve(strict=True)
                    relative = resolved.relative_to(self.repository_path)
                    size = resolved.stat().st_size
                except (OSError, ValueError):
                    continue
                if size > self.max_bytes - bytes_seen:
                    continue
                files_seen += 1
                bytes_seen += size
                try:
                    text = resolved.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for number, line in enumerate(text.splitlines(), 1):
                    lowered = line.casefold()
                    if any(term in lowered for term in terms):
                        hits.append(
                            LocalSearchHit(
                                relative_path=relative.as_posix(),
                                line_number=number,
                                excerpt=line[:240],
                            )
                        )
                        if len(hits) >= self.max_hits:
                            return tuple(hits)
            if stop:
                break
        return tuple(hits)

    @staticmethod
    def _terms(intent: RecallIntent) -> tuple[str, ...]:
        terms: set[str] = set()
        for key in intent.required_evidence:
            entity = key.canonical_entity_id.split(":", 1)[-1]
            for item in re.findall(r"[A-Za-z0-9_./-]{3,}", entity):
                terms.add(item.casefold())
        for item in re.findall(r"[A-Za-z0-9_./-]{4,}", intent.question):
            terms.add(item.casefold())
        return tuple(sorted(terms))


class FallbackChain:
    """Execute only the named fallback stage and truthfully report execution."""

    def __init__(
        self,
        semantic_index: object,
        *,
        local_search: BoundedLocalRepositorySearch | None = None,
    ) -> None:
        self.semantic_index = semantic_index
        self.local_search = local_search

    @property
    def stages(self) -> tuple[FallbackStage, ...]:
        return (
            FallbackStage.STRUCTURED_INDEX,
            FallbackStage.PAGE_METADATA_FTS,
            FallbackStage.RECENT_RELATED_PAGE,
            FallbackStage.LOCAL_REPOSITORY_SEARCH,
        )

    def execute(self, stage: FallbackStage, intent: RecallIntent) -> ExecutedFallback | None:
        if stage is FallbackStage.LOCAL_REPOSITORY_SEARCH:
            if self.local_search is None or not self.local_search.available_for(intent):
                return None
            hits = self.local_search.search(intent)
            return ExecutedFallback(
                stage=stage,
                candidates=(),
                reason=f"bounded repository search executed; hits={len(hits)}",
                local_hits=hits,
            )
        method_name = {
            FallbackStage.STRUCTURED_INDEX: "structured_fallback",
            FallbackStage.PAGE_METADATA_FTS: "metadata_fts",
            FallbackStage.RECENT_RELATED_PAGE: "recent_related",
        }.get(stage)
        if method_name is None:
            raise ValueError(f"not a fallback stage: {stage}")
        method = getattr(self.semantic_index, method_name, None)
        if method is None:
            return None
        result = method(intent)
        if not isinstance(result, FallbackQueryResult):
            raise TypeError(f"{method_name} returned an invalid result")
        if not result.executed:
            return None
        if result.executed_stage is not stage:
            raise RuntimeError(
                f"{method_name} reported {result.executed_stage} while executing {stage}"
            )
        return ExecutedFallback(stage, result.candidates, result.reason)
