"""Opt-in multilingual reference indexing helpers.

The default ``ReferenceDirectory`` remains unchanged.  SWE-Milestone selects
this subclass explicitly for non-Python runs so committed milestone tags stay
visible to the existing symbol/index pipeline through the run's immutable base
commit.  It changes navigation only, never acceptance.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import replace

from .references import ReferenceDirectory, _code_surface, symbols_for_file


class MultilangReferenceDirectory(ReferenceDirectory):
    """ReferenceDirectory with an opt-in committed-worktree diff fallback."""

    def _git_frontier_diff(self, relative: str) -> str:
        current = super()._git_frontier_diff(relative)
        if current:
            return current
        baseline = os.environ.get("HOMY_MULTILANG_BASE_COMMIT", "")
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", baseline):
            return current
        try:
            committed = subprocess.run(
                [
                    "git",
                    "diff",
                    "--no-ext-diff",
                    "--no-renames",
                    "--unified=0",
                    baseline,
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
            return current
        return committed.stdout if committed.returncode in (0, 1) else current

    def index_changed_files(self, **kwargs):
        records = super().index_changed_files(**kwargs)
        revision_id = str(kwargs["revision_id"])
        enriched = []
        cache: dict[str, tuple[object, dict[str, object]]] = {}
        for record in records:
            language = str(record.language).casefold()
            if (
                record.reference_kind != "SymbolReference"
                or record.code_surface is not None
                or language in {"python", "py", "python3", "unknown", ""}
                or not record.qualified_name
            ):
                enriched.append(record)
                continue
            relative = record.repository_relative_path
            entry = cache.get(relative)
            if entry is None:
                path = self.repository_path / relative
                try:
                    projection = symbols_for_file(
                        path,
                        language,
                        revision=revision_id,
                    )
                except (OSError, UnicodeError):
                    projection = None
                surface_symbols = {
                    item.qualified_name: item
                    for item in (projection.symbols if projection else ())
                }
                entry = (projection, surface_symbols)
                cache[relative] = entry
            projection, surface_symbols = entry
            symbol = surface_symbols.get(record.qualified_name)
            if projection is None or symbol is None:
                enriched.append(record)
                continue
            try:
                source = (self.repository_path / relative).read_text(
                    encoding="utf-8"
                )
                surface = _code_surface(
                    source,
                    projection,
                    repository=self.repository_id,
                    revision=revision_id,
                    path=relative,
                    symbol=symbol,
                )
            except (OSError, UnicodeError):
                enriched.append(record)
                continue
            enriched.append(replace(record, code_surface=surface))
        return tuple(enriched)
