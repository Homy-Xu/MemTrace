"""V2 exact-evidence Trace Recall and Context Recovery subsystem."""

from .assembler import ContextAssembler, public_evidence_handle
from .coverage import CoverageTracker, PageBodyEvidence
from .fallback import (
    BoundedLocalRepositorySearch,
    ExecutedFallback,
    FallbackChain,
    LocalSearchHit,
)
from .intent import normalize_intent
from .locator import CandidateRanker, LocatedCandidates, SemanticLocator
from .retriever import PageScanResult, PageSliceRetriever
from .service import RecallOutcome, RecallService, RecallTrace

__all__ = [
    "BoundedLocalRepositorySearch",
    "CandidateRanker",
    "ContextAssembler",
    "CoverageTracker",
    "ExecutedFallback",
    "FallbackChain",
    "LocalSearchHit",
    "LocatedCandidates",
    "PageBodyEvidence",
    "PageScanResult",
    "PageSliceRetriever",
    "RecallOutcome",
    "RecallService",
    "RecallTrace",
    "SemanticLocator",
    "normalize_intent",
    "public_evidence_handle",
]
