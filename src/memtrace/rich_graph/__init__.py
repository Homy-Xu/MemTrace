"""Optional, non-blocking Repository State Graph for the clean V2 runtime."""

from .models import (
    STRUCTURAL_RELATIONS,
    FileProjection,
    FrontierTask,
    MilestoneFrontier,
    ReferenceBinding,
    RichGraphHint,
    RichGraphHintReceipt,
    RichRelation,
    VersionNode,
    file_reference_id,
    is_structural_relation,
    symbol_reference_id,
    test_reference_id,
)
from .processor import process_frontier_file
from .scheduler import RichGraphScheduler

__all__ = [
    "STRUCTURAL_RELATIONS",
    "FileProjection",
    "FrontierTask",
    "MilestoneFrontier",
    "ReferenceBinding",
    "RichGraphHint",
    "RichGraphHintReceipt",
    "RichGraphScheduler",
    "RichRelation",
    "VersionNode",
    "file_reference_id",
    "is_structural_relation",
    "process_frontier_file",
    "symbol_reference_id",
    "test_reference_id",
]
