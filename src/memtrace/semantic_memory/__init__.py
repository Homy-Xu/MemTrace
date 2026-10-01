"""Synchronous Memory Index; independent from optional RSG state."""

from .contracts import (
    EntityResolutionResult,
    ExactEvidenceHit,
    ExactLookupResult,
    FallbackQueryResult,
    PageProjectionReceipt,
)
from .ontology import EdgeRegistry, default_edge_registry
from .store import SemanticStore

__all__ = [
    "EdgeRegistry",
    "EntityResolutionResult",
    "ExactEvidenceHit",
    "ExactLookupResult",
    "FallbackQueryResult",
    "PageProjectionReceipt",
    "SemanticStore",
    "default_edge_registry",
]
