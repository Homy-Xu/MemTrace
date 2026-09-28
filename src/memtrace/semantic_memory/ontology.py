from __future__ import annotations

from dataclasses import dataclass

from ..contracts import Authority, EdgeTypeSpec, NodeType, SemanticEdge


@dataclass(frozen=True, slots=True)
class _RegisteredEdge:
    spec: EdgeTypeSpec
    allowed_pairs: frozenset[tuple[NodeType, NodeType]]


class EdgeRegistry:
    """Closed ontology enforced by every production edge insertion."""

    def __init__(self) -> None:
        self._types: dict[str, _RegisteredEdge] = {}

    def register(
        self,
        spec: EdgeTypeSpec,
        *,
        allowed_pairs: tuple[tuple[NodeType, NodeType], ...] | None = None,
    ) -> None:
        if spec.edge_type in self._types:
            raise ValueError(f"duplicate edge type: {spec.edge_type}")
        pairs = allowed_pairs or tuple(
            (source, target) for source in spec.source_types for target in spec.target_types
        )
        if not pairs:
            raise ValueError("edge type needs at least one domain/range pair")
        self._types[spec.edge_type] = _RegisteredEdge(spec, frozenset(pairs))

    def spec(self, edge_type: str) -> EdgeTypeSpec:
        try:
            return self._types[edge_type].spec
        except KeyError as exc:
            raise ValueError(f"unknown semantic edge type: {edge_type}") from exc

    def validate(self, edge: SemanticEdge, *, critical_path: bool = False) -> None:
        try:
            registered = self._types[edge.edge_type]
        except KeyError as exc:
            raise ValueError(f"unknown semantic edge type: {edge.edge_type}") from exc
        if (edge.source_type, edge.target_type) not in registered.allowed_pairs:
            raise ValueError(
                f"invalid {edge.edge_type} domain/range: "
                f"{edge.source_type.value}->{edge.target_type.value}"
            )
        if edge.authority not in registered.spec.allowed_authority:
            raise ValueError(f"{edge.authority.value} is not allowed for {edge.edge_type}")
        if critical_path and not registered.spec.critical_path_allowed:
            raise ValueError(f"{edge.edge_type} cannot decide a critical-path address")
        if critical_path and edge.authority == Authority.INFERRED:
            raise ValueError("INFERRED edges cannot decide Page Fault addresses")
        if edge.authority == Authority.DERIVED and not edge.provenance:
            raise ValueError("DERIVED edges require deterministic provenance")
        if not edge.run_id or not edge.branch_id:
            raise ValueError("semantic edges require run and branch scope")
        if edge.valid_to_cursor is not None:
            if edge.valid_to_cursor < edge.valid_from_cursor:
                raise ValueError("edge validity cursor is reversed")
            if edge.valid_to_revision is None:
                raise ValueError("closed cursor validity requires a closing revision")

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._types))


def _spec(
    name: str,
    sources: tuple[NodeType, ...],
    targets: tuple[NodeType, ...],
    *,
    derived: bool = False,
    inferred: bool = False,
    critical: bool = False,
) -> EdgeTypeSpec:
    authorities = [Authority.ASSERTED]
    if derived:
        authorities.append(Authority.DERIVED)
    if inferred:
        authorities.append(Authority.INFERRED)
    return EdgeTypeSpec(
        edge_type=name,
        source_types=sources,
        target_types=targets,
        allowed_authority=tuple(authorities),
        creation_rule="validated synchronous V2 projection",
        update_rule="append a temporal edge; close prior current interval explicitly",
        critical_path_allowed=critical,
    )


def default_edge_registry() -> EdgeRegistry:
    registry = EdgeRegistry()

    def add(
        name: str,
        sources: tuple[NodeType, ...],
        targets: tuple[NodeType, ...],
        *,
        derived: bool = False,
        inferred: bool = False,
        critical: bool = False,
        pairs: tuple[tuple[NodeType, NodeType], ...] | None = None,
    ) -> None:
        registry.register(
            _spec(
                name,
                sources,
                targets,
                derived=derived,
                inferred=inferred,
                critical=critical,
            ),
            allowed_pairs=pairs,
        )

    add("HAS_GOAL", (NodeType.TASK,), (NodeType.GOAL,))
    add("HAS_PLAN", (NodeType.TASK,), (NodeType.PLAN_VERSION,))
    add("HAS_NATIVE_ITEM", (NodeType.PLAN_VERSION,), (NodeType.NATIVE_PLAN_ITEM,))
    add(
        "PROJECTS_TO",
        (NodeType.NATIVE_PLAN_ITEM,),
        (NodeType.MILESTONE_IDENTITY, NodeType.PLAN_STEP),
    )
    add("CONTAINS", (NodeType.PLAN_VERSION,), (NodeType.MILESTONE_IDENTITY,))
    add(
        "SUPERSEDES",
        (NodeType.PLAN_VERSION, NodeType.MILESTONE_VERSION),
        (NodeType.PLAN_VERSION, NodeType.MILESTONE_VERSION),
        pairs=(
            (NodeType.PLAN_VERSION, NodeType.PLAN_VERSION),
            (NodeType.MILESTONE_VERSION, NodeType.MILESTONE_VERSION),
        ),
    )
    add(
        "HAS_VERSION",
        (NodeType.MILESTONE_IDENTITY,),
        (NodeType.MILESTONE_VERSION,),
    )
    add("HAS_STEP", (NodeType.MILESTONE_IDENTITY,), (NodeType.PLAN_STEP,))
    add(
        "HAS_CRITERION",
        (NodeType.MILESTONE_VERSION,),
        (NodeType.COMPLETION_CRITERION,),
    )
    add(
        "SATISFIES",
        (NodeType.EVIDENCE_UNIT, NodeType.VERIFICATION_RESULT),
        (NodeType.COMPLETION_CRITERION,),
        critical=True,
    )
    add(
        "SUPPORTS",
        (NodeType.EVIDENCE_UNIT,),
        (NodeType.MILESTONE_IDENTITY, NodeType.PLAN_STEP),
        critical=True,
    )
    add(
        "DEPENDS_ON",
        (NodeType.MILESTONE_IDENTITY,),
        (NodeType.MILESTONE_IDENTITY,),
    )
    add(
        "PRECEDES",
        (NodeType.MILESTONE_IDENTITY,),
        (NodeType.MILESTONE_IDENTITY,),
        derived=True,
    )
    add(
        "CURRENT_MILESTONE",
        (NodeType.RUN,),
        (NodeType.MILESTONE_IDENTITY,),
        critical=True,
    )
    add(
        "CURRENT_STEP",
        (NodeType.MILESTONE_IDENTITY,),
        (NodeType.PLAN_STEP,),
        critical=True,
    )
    add(
        "EXECUTED_UNDER",
        (NodeType.EVENT, NodeType.EVENT_GROUP),
        (NodeType.MILESTONE_IDENTITY,),
        critical=True,
    )
    add("FOCUSES_ON", (NodeType.PAGE,), (NodeType.MILESTONE_IDENTITY,), critical=True)
    add(
        "CONTAINS_PAGE",
        (NodeType.MILESTONE_IDENTITY,),
        (NodeType.PAGE,),
        derived=True,
        critical=True,
    )
    add(
        "UPDATES",
        (NodeType.EVENT, NodeType.MILESTONE_REVIEW),
        (NodeType.MILESTONE_VERSION, NodeType.WORKSPACE_REVISION, NodeType.PLAN_VERSION),
        pairs=(
            (NodeType.EVENT, NodeType.MILESTONE_VERSION),
            (NodeType.EVENT, NodeType.WORKSPACE_REVISION),
            (NodeType.EVENT, NodeType.PLAN_STEP),
            (NodeType.MILESTONE_REVIEW, NodeType.PLAN_VERSION),
        ),
    )
    add("READS", (NodeType.EVENT,), (NodeType.FILE_REFERENCE,))
    add(
        "MODIFIES",
        (NodeType.EVENT,),
        (NodeType.FILE_REFERENCE, NodeType.SYMBOL_REFERENCE),
    )
    add("RUNS_TEST", (NodeType.EVENT,), (NodeType.TEST_REFERENCE,))
    add(
        "OBSERVES_FAILURE",
        (NodeType.EVENT, NodeType.TEST_REFERENCE),
        (NodeType.FAILURE_REFERENCE,),
    )
    add(
        "ABOUT_FILE",
        (NodeType.EVIDENCE_UNIT,),
        (NodeType.FILE_REFERENCE,),
        derived=True,
    )
    add(
        "ABOUT_SYMBOL",
        (NodeType.EVIDENCE_UNIT,),
        (NodeType.SYMBOL_REFERENCE,),
        derived=True,
    )
    add(
        "ABOUT_TEST",
        (NodeType.EVIDENCE_UNIT,),
        (NodeType.TEST_REFERENCE,),
        derived=True,
    )
    add(
        "ABOUT_FAILURE",
        (NodeType.EVIDENCE_UNIT,),
        (NodeType.FAILURE_REFERENCE,),
        derived=True,
    )
    add(
        "ABOUT_CHANGE",
        (NodeType.EVIDENCE_UNIT,),
        (NodeType.CHANGE_REFERENCE,),
        derived=True,
    )
    add(
        "EVIDENCED_BY",
        (NodeType.EVIDENCE_UNIT,),
        (NodeType.EVENT, NodeType.EVENT_GROUP),
        derived=True,
        critical=True,
    )
    add(
        "LOCATED_AT",
        (NodeType.EVIDENCE_UNIT,),
        (NodeType.SEMANTIC_ANCHOR,),
        derived=True,
        critical=True,
    )
    add(
        "STORED_IN",
        (NodeType.SEMANTIC_ANCHOR,),
        (NodeType.PAGE,),
        derived=True,
        critical=True,
    )
    add("ADVANCES_TO", (NodeType.PAGE,), (NodeType.PAGE,), derived=True)
    add("CONTINUES_WITH", (NodeType.PAGE,), (NodeType.PAGE,), derived=True, critical=True)
    add("HAS_CORRECTIVE_STEP", (NodeType.PLAN_STEP,), (NodeType.PLAN_STEP,))
    add("CORRECTED_BY", (NodeType.PAGE,), (NodeType.PAGE,))
    add("VERIFIED_BY", (NodeType.PAGE,), (NodeType.PAGE,))
    add("SUPERSEDED_BY", (NodeType.PAGE,), (NodeType.PAGE,))
    add("DEPENDED_ON_BY", (NodeType.PAGE,), (NodeType.PAGE,))
    add("BRANCHES_TO", (NodeType.PAGE,), (NodeType.PAGE,), derived=True)
    add("RESOLVED_BY", (NodeType.PAGE,), (NodeType.PAGE,))
    add(
        "VALID_AT",
        (NodeType.EVIDENCE_UNIT, NodeType.PAGE),
        (NodeType.WORKSPACE_REVISION,),
        derived=True,
        critical=True,
    )
    add(
        "INVALIDATED_BY",
        (NodeType.EVIDENCE_UNIT,),
        (NodeType.EVENT, NodeType.CHANGE_REFERENCE),
    )
    add(
        "RESOLVES_TO",
        (NodeType.FILE_REFERENCE, NodeType.SYMBOL_REFERENCE),
        (NodeType.FILE_VERSION, NodeType.SYMBOL_VERSION),
        derived=True,
        inferred=True,
        pairs=(
            (NodeType.FILE_REFERENCE, NodeType.FILE_VERSION),
            (NodeType.SYMBOL_REFERENCE, NodeType.SYMBOL_VERSION),
        ),
    )
    add(
        "SAME_TEST_AS",
        (NodeType.TEST_REFERENCE,),
        (NodeType.TEST,),
        derived=True,
        inferred=True,
    )
    add(
        "MATCHES_FAILURE",
        (NodeType.FAILURE_REFERENCE,),
        (NodeType.DIAGNOSTIC,),
        derived=True,
        inferred=True,
    )
    return registry
