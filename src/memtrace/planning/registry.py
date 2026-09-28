from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import replace

from ..contracts import (
    ClaimType,
    CommitmentLevel,
    CompletionCriterionSpec,
    CriterionVerificationMode,
    FactType,
    MilestoneReviewDecision,
    MilestoneSpec,
    MilestoneStatus,
    PlanSpec,
    PlanStepReviewDecision,
    PlanStepSpec,
    PlanStepStatus,
    TaskStatus,
    default_verification_mode,
    digest,
    primitive,
    stable_id,
    utc_now,
)
from ..database import StateDatabase
from .contracts import (
    AcceptanceProgress,
    AcceptanceProgressClass,
    ContractFreezeReceipt,
    CurrentMilestone,
    MilestoneProjectionRecord,
    MilestoneStateEvent,
    PlanApplication,
    PlanCoverage,
    PlanObservationKind,
    PlanObservationResult,
    PlanProjectionInput,
    PlanStepProgressClaim,
    RouteTransitionKind,
    RouteTransitionReceipt,
    TaskStateEvent,
    UnresolvedMilestoneObservation,
    WorkingSetRoot,
)
from .projection import PlanProjection
from .requirements import extract_task_requirements, requirement_link_score
from .step_address import PlanStepAddress

_SCHEMA = """
CREATE TABLE IF NOT EXISTS v2_registry_runs (
    run_id TEXT PRIMARY KEY,
    repository_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    next_cursor INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS v2_tasks (
    task_id TEXT PRIMARY KEY,
    repository_id TEXT NOT NULL,
    run_id TEXT NOT NULL UNIQUE REFERENCES v2_registry_runs(run_id),
    branch_id TEXT NOT NULL,
    initial_revision_id TEXT NOT NULL,
    user_request TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS v2_goals (
    goal_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES v2_tasks(task_id),
    goal_text TEXT NOT NULL,
    goal_digest TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, goal_digest)
);

CREATE TABLE IF NOT EXISTS v2_task_requirement_sets (
    requirement_set_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL UNIQUE REFERENCES v2_tasks(task_id),
    run_id TEXT NOT NULL UNIQUE REFERENCES v2_registry_runs(run_id),
    raw_request_digest TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS v2_task_requirements (
    requirement_id TEXT PRIMARY KEY,
    requirement_set_id TEXT NOT NULL REFERENCES v2_task_requirement_sets(requirement_set_id),
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    ordinal INTEGER NOT NULL,
    requirement_text TEXT NOT NULL,
    requirement_digest TEXT NOT NULL,
    modality TEXT NOT NULL CHECK(modality IN ('MUST','SHOULD','MAY','FORBIDDEN','CONTEXT')),
    category TEXT NOT NULL,
    required INTEGER NOT NULL CHECK(required IN (0,1)),
    source_line INTEGER NOT NULL,
    source_event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(requirement_set_id, ordinal),
    UNIQUE(requirement_set_id, requirement_digest)
);

CREATE TABLE IF NOT EXISTS v2_plan_versions (
    plan_version_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES v2_tasks(task_id),
    goal_id TEXT NOT NULL REFERENCES v2_goals(goal_id),
    version_number INTEGER NOT NULL,
    previous_plan_version_id TEXT REFERENCES v2_plan_versions(plan_version_id),
    plan_digest TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, version_number),
    UNIQUE(task_id, plan_digest)
);

CREATE TABLE IF NOT EXISTS v2_milestone_identities (
    identity_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES v2_tasks(task_id),
    run_id TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    created_plan_version_id TEXT NOT NULL REFERENCES v2_plan_versions(plan_version_id),
    created_at TEXT NOT NULL,
    UNIQUE(run_id, canonical_id)
);

CREATE TABLE IF NOT EXISTS v2_milestone_versions (
    version_id TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    version_number INTEGER NOT NULL,
    plan_version_id TEXT NOT NULL REFERENCES v2_plan_versions(plan_version_id),
    previous_version_id TEXT REFERENCES v2_milestone_versions(version_id),
    spec_digest TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    completion_criteria_json TEXT NOT NULL,
    verification_json TEXT NOT NULL,
    status TEXT NOT NULL,
    entity_refs_json TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(identity_id, version_number),
    UNIQUE(identity_id, spec_digest)
);

CREATE TABLE IF NOT EXISTS v2_task_state_events (
    state_event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    task_id TEXT NOT NULL REFERENCES v2_tasks(task_id),
    previous_status TEXT,
    status TEXT NOT NULL CHECK(status IN (
        'CREATED','PLANNING','EXECUTING','VERIFYING','COMPLETED',
        'BLOCKED','FAILED','CANCELLED'
    )),
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, created_cursor)
);

CREATE TABLE IF NOT EXISTS v2_milestone_state_events (
    state_event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    previous_status TEXT,
    status TEXT NOT NULL CHECK(status IN (
        'PENDING','IN_PROGRESS','COMPLETED_CLAIMED','COMPLETED_VERIFIED',
        'VERIFICATION_FAILED','REPAIRING','REQUIRES_REVALIDATION',
        'BLOCKED','FAILED','CANCELLED'
    )),
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(identity_id, created_cursor)
);
CREATE INDEX IF NOT EXISTS v2_milestone_state_current
ON v2_milestone_state_events(identity_id, created_cursor DESC);

CREATE TABLE IF NOT EXISTS v2_completion_criteria (
    criterion_identity_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    milestone_version_id TEXT NOT NULL REFERENCES v2_milestone_versions(version_id),
    local_criterion_id TEXT NOT NULL,
    observable_outcome TEXT NOT NULL,
    required_evidence_types_json TEXT NOT NULL,
    entity_refs_json TEXT NOT NULL,
    test_selectors_json TEXT NOT NULL,
    required INTEGER NOT NULL CHECK(required IN (0,1)),
    requirement_id TEXT NOT NULL,
    requirement_text TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    commitment_level TEXT NOT NULL CHECK(commitment_level IN ('DIRECTION','MILESTONE','STEP')),
    source_event_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    verification_mode TEXT NOT NULL DEFAULT '',
    UNIQUE(milestone_version_id, local_criterion_id)
);
CREATE TABLE IF NOT EXISTS v2_milestone_contract_freezes (
    freeze_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    previous_plan_version_id TEXT NOT NULL REFERENCES v2_plan_versions(plan_version_id),
    resulting_plan_version_id TEXT NOT NULL REFERENCES v2_plan_versions(plan_version_id),
    previous_milestone_version_id TEXT NOT NULL,
    resulting_milestone_version_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    address_resolution_json TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id,milestone_identity_id,resulting_plan_version_id)
);
CREATE TABLE IF NOT EXISTS v2_acceptance_progress_observations (
    observation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    plan_version_id TEXT NOT NULL,
    boundary_kind TEXT NOT NULL,
    gap_digest TEXT NOT NULL,
    bound_evidence_digest TEXT NOT NULL,
    scoped_revision_digest TEXT NOT NULL,
    unmet_criterion_ids_json TEXT NOT NULL,
    progress_class TEXT NOT NULL CHECK(progress_class IN ('INITIAL','STRONG','WEAK','NONE')),
    weak_streak INTEGER NOT NULL,
    none_streak INTEGER NOT NULL,
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id,milestone_identity_id,source_event_id,boundary_kind)
);
CREATE TABLE IF NOT EXISTS v2_semantic_review_rounds (
    round_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    revision_id TEXT NOT NULL,
    gap_digest TEXT NOT NULL,
    criterion_ids_json TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    source_event_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id,milestone_identity_id,source_event_id)
);

CREATE TABLE IF NOT EXISTS v2_requirement_plan_links (
    link_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    requirement_id TEXT NOT NULL REFERENCES v2_task_requirements(requirement_id),
    criterion_identity_id TEXT NOT NULL REFERENCES v2_completion_criteria(criterion_identity_id),
    plan_version_id TEXT NOT NULL REFERENCES v2_plan_versions(plan_version_id),
    link_basis TEXT NOT NULL,
    link_score REAL NOT NULL,
    source_event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(requirement_id, criterion_identity_id)
);

CREATE TABLE IF NOT EXISTS v2_plan_step_identities (
    step_identity_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    canonical_step_id TEXT NOT NULL,
    title TEXT NOT NULL,
    corrective INTEGER NOT NULL CHECK(corrective IN (0,1)),
    criterion_ids_json TEXT NOT NULL,
    entity_refs_json TEXT NOT NULL,
    historical_dependency_refs_json TEXT NOT NULL,
    source_plan_item_ids_json TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, canonical_step_id)
);

CREATE TABLE IF NOT EXISTS v2_plan_step_state_events (
    state_event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    step_identity_id TEXT NOT NULL REFERENCES v2_plan_step_identities(step_identity_id),
    previous_status TEXT,
    status TEXT NOT NULL CHECK(status IN (
        'PENDING','IN_PROGRESS','COMPLETED_CLAIMED','COMPLETED_VERIFIED','FAILED','CANCELLED'
    )),
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(step_identity_id, created_cursor)
);

CREATE TABLE IF NOT EXISTS v2_plan_step_contracts (
    contract_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    step_identity_id TEXT NOT NULL UNIQUE REFERENCES v2_plan_step_identities(step_identity_id),
    expected_outcome TEXT NOT NULL,
    minimum_acceptance_json TEXT NOT NULL,
    failure_signals_json TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE VIEW IF NOT EXISTS v2_effective_plan_step_contracts AS
SELECT
    base.contract_id AS contract_id,
    base.run_id AS run_id,
    base.step_identity_id AS step_identity_id,
    base.expected_outcome AS expected_outcome,
    base.minimum_acceptance_json AS minimum_acceptance_json,
    base.failure_signals_json AS failure_signals_json,
    base.source_event_id AS source_event_id,
    base.created_cursor AS created_cursor,
    base.created_at AS created_at,
    0 AS contract_revision_number
FROM v2_plan_step_contracts base;

CREATE TABLE IF NOT EXISTS v2_plan_step_review_events (
    review_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    step_identity_id TEXT NOT NULL REFERENCES v2_plan_step_identities(step_identity_id),
    decision TEXT NOT NULL CHECK(decision IN ('SATISFIED','CONTINUE','CORRECT','BLOCKED')),
    summary TEXT NOT NULL,
    evidence_event_ids_json TEXT NOT NULL,
    created_step_ids_json TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(step_identity_id,source_event_id)
);

CREATE TABLE IF NOT EXISTS v2_plan_step_corrections (
    correction_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    failed_step_identity_id TEXT NOT NULL REFERENCES v2_plan_step_identities(step_identity_id),
    corrective_step_identity_id TEXT NOT NULL REFERENCES v2_plan_step_identities(step_identity_id),
    review_id TEXT NOT NULL REFERENCES v2_plan_step_review_events(review_id),
    source_event_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(failed_step_identity_id,corrective_step_identity_id)
);

CREATE TABLE IF NOT EXISTS v2_corrective_steps (
    corrective_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    step_identity_id TEXT NOT NULL REFERENCES v2_plan_step_identities(step_identity_id),
    criterion_identity_id TEXT REFERENCES v2_completion_criteria(criterion_identity_id),
    reason TEXT NOT NULL,
    evidence_event_ids_json TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    failure_signature TEXT,
    failure_criterion_ids_json TEXT NOT NULL DEFAULT '[]',
    baseline_revision_id TEXT,
    baseline_cursor INTEGER,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS v2_milestone_review_events (
    review_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    decision TEXT NOT NULL CHECK(decision IN (
        'CONTINUE','REPLAN_FUTURE','UPDATE_FUTURE','SPLIT_FUTURE','MERGE_FUTURE',
        'CANCEL_FUTURE','REPLACE_FUTURE'
    )),
    reason TEXT NOT NULL,
    evidence_event_ids_json TEXT NOT NULL,
    previous_plan_version_id TEXT NOT NULL REFERENCES v2_plan_versions(plan_version_id),
    resulting_plan_version_id TEXT REFERENCES v2_plan_versions(plan_version_id),
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS v2_milestone_failure_review_events (
    review_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    decision TEXT NOT NULL CHECK(decision='CORRECT_CURRENT'),
    reason TEXT NOT NULL,
    failure_signature TEXT NOT NULL,
    failure_criterion_ids_json TEXT NOT NULL,
    evidence_event_ids_json TEXT NOT NULL,
    created_step_ids_json TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    baseline_cursor INTEGER NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id,source_event_id),
    UNIQUE(milestone_identity_id,failure_signature,revision_id)
);

CREATE TABLE IF NOT EXISTS v2_milestone_acceptance_receipts (
    receipt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    revision_id TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK(verdict IN ('COMPLETED_VERIFIED','VERIFICATION_FAILED')),
    satisfied_criterion_ids_json TEXT NOT NULL,
    failed_criterion_ids_json TEXT NOT NULL,
    evidence_event_ids_json TEXT NOT NULL,
    failure_signature TEXT,
    source_event_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    unverified_criterion_ids_json TEXT NOT NULL DEFAULT '[]',
    UNIQUE(milestone_identity_id,revision_id,source_event_id,verdict)
);

CREATE TABLE IF NOT EXISTS v2_milestone_focus_events (
    focus_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    focus_kind TEXT NOT NULL CHECK(focus_kind IN ('VERIFICATION','CORRECTIVE')),
    criterion_id TEXT NOT NULL,
    step_identity_id TEXT NOT NULL REFERENCES v2_plan_step_identities(step_identity_id),
    frontier_digest TEXT NOT NULL,
    failure_signature TEXT,
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id,milestone_identity_id,focus_kind,frontier_digest)
);

CREATE TABLE IF NOT EXISTS v2_route_frontier_observations (
    observation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    frontier_digest TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id,source_event_id,frontier_digest)
);
CREATE INDEX IF NOT EXISTS v2_route_frontier_digest
ON v2_route_frontier_observations(run_id,frontier_digest,created_cursor);

CREATE TABLE IF NOT EXISTS v2_route_stall_events (
    stall_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    frontier_digest TEXT NOT NULL,
    reason TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id,frontier_digest)
);

CREATE TABLE IF NOT EXISTS v2_milestone_lineage (
    relation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    relation_type TEXT NOT NULL CHECK(relation_type IN ('SUPERSEDES','DERIVED_FROM')),
    source_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    target_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    source_event_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, relation_type, source_identity_id, target_identity_id)
);

CREATE TABLE IF NOT EXISTS v2_milestone_aliases (
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    normalized_alias TEXT NOT NULL,
    alias_text TEXT NOT NULL,
    identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    source_event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(run_id, normalized_alias)
);

CREATE TABLE IF NOT EXISTS v2_plan_observations (
    observation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    observation_kind TEXT NOT NULL CHECK(observation_kind IN ('SNAPSHOT','PATCH')),
    coverage TEXT NOT NULL CHECK(coverage IN ('COMPLETE','UNKNOWN')),
    source_event_id TEXT NOT NULL,
    observation_digest TEXT NOT NULL,
    applied_plan_version_id TEXT REFERENCES v2_plan_versions(plan_version_id),
    superseded INTEGER NOT NULL CHECK(superseded IN (0,1)),
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, source_event_id, observation_digest)
);

CREATE TABLE IF NOT EXISTS v2_plan_step_route_commits (
    commit_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    owner_step_identity_id TEXT NOT NULL REFERENCES v2_plan_step_identities(step_identity_id),
    covered_step_identity_ids_json TEXT NOT NULL,
    evidence_event_ids_json TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id,source_event_id)
);

CREATE TABLE IF NOT EXISTS v2_focus_observation_events (
    focus_event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    plan_version_id TEXT NOT NULL REFERENCES v2_plan_versions(plan_version_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    step_identity_id TEXT NOT NULL REFERENCES v2_plan_step_identities(step_identity_id),
    observation_basis TEXT NOT NULL,
    matched_entities_json TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, source_event_id)
);

CREATE TABLE IF NOT EXISTS v2_focus_span_observation_events (
    focus_span_event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    plan_version_id TEXT NOT NULL REFERENCES v2_plan_versions(plan_version_id),
    milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    covered_step_identity_ids_json TEXT NOT NULL,
    observation_basis TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, source_event_id)
);

CREATE TABLE IF NOT EXISTS v2_unresolved_milestone_observations (
    observation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    source_event_id TEXT NOT NULL,
    title TEXT NOT NULL,
    reason TEXT NOT NULL,
    candidates_json TEXT NOT NULL,
    observation_json TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS v2_plan_milestones (
    plan_version_id TEXT NOT NULL REFERENCES v2_plan_versions(plan_version_id),
    identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    milestone_version_id TEXT NOT NULL REFERENCES v2_milestone_versions(version_id),
    ordinal INTEGER NOT NULL,
    PRIMARY KEY(plan_version_id, identity_id),
    UNIQUE(plan_version_id, ordinal)
);

CREATE TABLE IF NOT EXISTS v2_milestone_dependencies (
    plan_version_id TEXT NOT NULL REFERENCES v2_plan_versions(plan_version_id),
    source_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    target_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    source_event_id TEXT NOT NULL,
    created_cursor INTEGER NOT NULL,
    PRIMARY KEY(plan_version_id, source_identity_id, target_identity_id),
    CHECK(source_identity_id <> target_identity_id)
);

CREATE TABLE IF NOT EXISTS v2_current_milestones (
    relation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    plan_version_id TEXT NOT NULL REFERENCES v2_plan_versions(plan_version_id),
    source_event_id TEXT NOT NULL,
    valid_from_revision TEXT NOT NULL,
    valid_to_revision TEXT,
    valid_from_cursor INTEGER NOT NULL,
    valid_to_cursor INTEGER,
    CHECK((valid_to_revision IS NULL) = (valid_to_cursor IS NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS v2_one_current_milestone_per_run
ON v2_current_milestones(run_id) WHERE valid_to_cursor IS NULL;

CREATE TABLE IF NOT EXISTS v2_working_set_memberships (
    membership_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
    plan_version_id TEXT NOT NULL REFERENCES v2_plan_versions(plan_version_id),
    identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
    state TEXT NOT NULL CHECK(state IN ('HOT', 'PREFETCH', 'COLD')),
    reason TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    valid_from_revision TEXT NOT NULL,
    valid_to_revision TEXT,
    valid_from_cursor INTEGER NOT NULL,
    valid_to_cursor INTEGER,
    CHECK((valid_to_revision IS NULL) = (valid_to_cursor IS NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS v2_one_active_working_state
ON v2_working_set_memberships(run_id, identity_id) WHERE valid_to_cursor IS NULL;
CREATE INDEX IF NOT EXISTS v2_working_roots_lookup
ON v2_working_set_memberships(run_id, state, valid_to_cursor, plan_version_id);

CREATE TRIGGER IF NOT EXISTS v2_plan_versions_no_update
BEFORE UPDATE ON v2_plan_versions BEGIN SELECT RAISE(ABORT, 'PlanVersion is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_versions_no_delete
BEFORE DELETE ON v2_plan_versions BEGIN SELECT RAISE(ABORT, 'PlanVersion is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_milestone_versions_no_update
BEFORE UPDATE ON v2_milestone_versions BEGIN SELECT RAISE(ABORT, 'MilestoneVersion is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_milestone_versions_no_delete
BEFORE DELETE ON v2_milestone_versions BEGIN SELECT RAISE(ABORT, 'MilestoneVersion is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_tasks_no_update
BEFORE UPDATE ON v2_tasks BEGIN SELECT RAISE(ABORT, 'Task is immutable'); END;
CREATE TRIGGER IF NOT EXISTS v2_tasks_no_delete
BEFORE DELETE ON v2_tasks BEGIN SELECT RAISE(ABORT, 'Task is immutable'); END;
CREATE TRIGGER IF NOT EXISTS v2_goals_no_update
BEFORE UPDATE ON v2_goals BEGIN SELECT RAISE(ABORT, 'Goal is immutable'); END;
CREATE TRIGGER IF NOT EXISTS v2_goals_no_delete
BEFORE DELETE ON v2_goals BEGIN SELECT RAISE(ABORT, 'Goal is immutable'); END;
CREATE TRIGGER IF NOT EXISTS v2_milestone_identities_no_update
BEFORE UPDATE ON v2_milestone_identities BEGIN SELECT RAISE(ABORT, 'MilestoneIdentity is immutable'); END;
CREATE TRIGGER IF NOT EXISTS v2_milestone_identities_no_delete
BEFORE DELETE ON v2_milestone_identities BEGIN SELECT RAISE(ABORT, 'MilestoneIdentity is immutable'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_milestones_no_update
BEFORE UPDATE ON v2_plan_milestones BEGIN SELECT RAISE(ABORT, 'Plan membership is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_milestones_no_delete
BEFORE DELETE ON v2_plan_milestones BEGIN SELECT RAISE(ABORT, 'Plan membership is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_milestone_dependencies_no_update
BEFORE UPDATE ON v2_milestone_dependencies BEGIN SELECT RAISE(ABORT, 'Milestone dependency is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_milestone_dependencies_no_delete
BEFORE DELETE ON v2_milestone_dependencies BEGIN SELECT RAISE(ABORT, 'Milestone dependency is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_task_state_events_no_update
BEFORE UPDATE ON v2_task_state_events BEGIN SELECT RAISE(ABORT, 'TaskStateEvent is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_task_state_events_no_delete
BEFORE DELETE ON v2_task_state_events BEGIN SELECT RAISE(ABORT, 'TaskStateEvent is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_milestone_state_events_no_update
BEFORE UPDATE ON v2_milestone_state_events BEGIN SELECT RAISE(ABORT, 'MilestoneStateEvent is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_milestone_state_events_no_delete
BEFORE DELETE ON v2_milestone_state_events BEGIN SELECT RAISE(ABORT, 'MilestoneStateEvent is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_milestone_aliases_no_update
BEFORE UPDATE ON v2_milestone_aliases BEGIN SELECT RAISE(ABORT, 'MilestoneAlias is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_milestone_aliases_no_delete
BEFORE DELETE ON v2_milestone_aliases BEGIN SELECT RAISE(ABORT, 'MilestoneAlias is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_observations_no_update
BEFORE UPDATE ON v2_plan_observations BEGIN SELECT RAISE(ABORT, 'PlanObservation is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_observations_no_delete
BEFORE DELETE ON v2_plan_observations BEGIN SELECT RAISE(ABORT, 'PlanObservation is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_step_route_commits_no_update
BEFORE UPDATE ON v2_plan_step_route_commits BEGIN SELECT RAISE(ABORT, 'PlanStepRouteCommit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_step_route_commits_no_delete
BEFORE DELETE ON v2_plan_step_route_commits BEGIN SELECT RAISE(ABORT, 'PlanStepRouteCommit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_focus_observations_no_update
BEFORE UPDATE ON v2_focus_observation_events BEGIN SELECT RAISE(ABORT, 'FocusObservation is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_focus_observations_no_delete
BEFORE DELETE ON v2_focus_observation_events BEGIN SELECT RAISE(ABORT, 'FocusObservation is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_focus_span_observations_no_update
BEFORE UPDATE ON v2_focus_span_observation_events BEGIN SELECT RAISE(ABORT, 'FocusSpanObservation is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_focus_span_observations_no_delete
BEFORE DELETE ON v2_focus_span_observation_events BEGIN SELECT RAISE(ABORT, 'FocusSpanObservation is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_unresolved_observations_no_update
BEFORE UPDATE ON v2_unresolved_milestone_observations BEGIN SELECT RAISE(ABORT, 'Unresolved observation is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_unresolved_observations_no_delete
BEFORE DELETE ON v2_unresolved_milestone_observations BEGIN SELECT RAISE(ABORT, 'Unresolved observation is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_completion_criteria_no_update
BEFORE UPDATE ON v2_completion_criteria BEGIN SELECT RAISE(ABORT, 'CompletionCriterion is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_completion_criteria_no_delete
BEFORE DELETE ON v2_completion_criteria BEGIN SELECT RAISE(ABORT, 'CompletionCriterion is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_step_identities_no_update
BEFORE UPDATE ON v2_plan_step_identities BEGIN SELECT RAISE(ABORT, 'PlanStepIdentity is immutable'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_step_identities_no_delete
BEFORE DELETE ON v2_plan_step_identities BEGIN SELECT RAISE(ABORT, 'PlanStepIdentity is immutable'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_step_state_events_no_update
BEFORE UPDATE ON v2_plan_step_state_events BEGIN SELECT RAISE(ABORT, 'PlanStepStateEvent is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_step_state_events_no_delete
BEFORE DELETE ON v2_plan_step_state_events BEGIN SELECT RAISE(ABORT, 'PlanStepStateEvent is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_step_contracts_no_update
BEFORE UPDATE ON v2_plan_step_contracts BEGIN SELECT RAISE(ABORT, 'PlanStepContract is immutable'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_step_contracts_no_delete
BEFORE DELETE ON v2_plan_step_contracts BEGIN SELECT RAISE(ABORT, 'PlanStepContract is immutable'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_step_review_events_no_update
BEFORE UPDATE ON v2_plan_step_review_events BEGIN SELECT RAISE(ABORT, 'PlanStepReviewEvent is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_step_review_events_no_delete
BEFORE DELETE ON v2_plan_step_review_events BEGIN SELECT RAISE(ABORT, 'PlanStepReviewEvent is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_step_corrections_no_update
BEFORE UPDATE ON v2_plan_step_corrections BEGIN SELECT RAISE(ABORT, 'PlanStepCorrection is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_plan_step_corrections_no_delete
BEFORE DELETE ON v2_plan_step_corrections BEGIN SELECT RAISE(ABORT, 'PlanStepCorrection is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_corrective_steps_no_update
BEFORE UPDATE ON v2_corrective_steps BEGIN SELECT RAISE(ABORT, 'CorrectiveStep is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_corrective_steps_no_delete
BEFORE DELETE ON v2_corrective_steps BEGIN SELECT RAISE(ABORT, 'CorrectiveStep is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_milestone_review_events_no_update
BEFORE UPDATE ON v2_milestone_review_events BEGIN SELECT RAISE(ABORT, 'MilestoneReviewEvent is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_milestone_review_events_no_delete
BEFORE DELETE ON v2_milestone_review_events BEGIN SELECT RAISE(ABORT, 'MilestoneReviewEvent is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_milestone_lineage_no_update
BEFORE UPDATE ON v2_milestone_lineage BEGIN SELECT RAISE(ABORT, 'Milestone lineage is append-only'); END;
CREATE TRIGGER IF NOT EXISTS v2_milestone_lineage_no_delete
BEFORE DELETE ON v2_milestone_lineage BEGIN SELECT RAISE(ABORT, 'Milestone lineage is append-only'); END;
"""

_MILESTONE_PREFIX = re.compile(
    r"^\s*(M\d{3,})(?:\s*:\s*|\s+-\s+|\s+)",
    re.IGNORECASE,
)

_TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset({TaskStatus.PLANNING, TaskStatus.CANCELLED, TaskStatus.FAILED}),
    TaskStatus.PLANNING: frozenset(
        {TaskStatus.EXECUTING, TaskStatus.BLOCKED, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.EXECUTING: frozenset(
        {TaskStatus.VERIFYING, TaskStatus.BLOCKED, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.VERIFYING: frozenset(
        {
            TaskStatus.COMPLETED,
            TaskStatus.EXECUTING,
            TaskStatus.BLOCKED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.BLOCKED: frozenset({TaskStatus.EXECUTING, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}

_MILESTONE_TRANSITIONS: dict[MilestoneStatus, frozenset[MilestoneStatus]] = {
    MilestoneStatus.PENDING: frozenset(
        {
            MilestoneStatus.IN_PROGRESS,
            MilestoneStatus.BLOCKED,
            MilestoneStatus.FAILED,
            MilestoneStatus.CANCELLED,
        }
    ),
    MilestoneStatus.IN_PROGRESS: frozenset(
        {
            MilestoneStatus.COMPLETED_CLAIMED,
            MilestoneStatus.BLOCKED,
            MilestoneStatus.FAILED,
            MilestoneStatus.CANCELLED,
        }
    ),
    MilestoneStatus.COMPLETED_CLAIMED: frozenset(
        {
            MilestoneStatus.COMPLETED_VERIFIED,
            MilestoneStatus.VERIFICATION_FAILED,
            MilestoneStatus.IN_PROGRESS,
            MilestoneStatus.BLOCKED,
            MilestoneStatus.FAILED,
        }
    ),
    MilestoneStatus.VERIFICATION_FAILED: frozenset(
        {
            MilestoneStatus.REPAIRING,
            MilestoneStatus.IN_PROGRESS,
            MilestoneStatus.BLOCKED,
            MilestoneStatus.FAILED,
            MilestoneStatus.CANCELLED,
        }
    ),
    MilestoneStatus.REPAIRING: frozenset(
        {
            MilestoneStatus.IN_PROGRESS,
            MilestoneStatus.COMPLETED_CLAIMED,
            MilestoneStatus.BLOCKED,
            MilestoneStatus.FAILED,
            MilestoneStatus.CANCELLED,
        }
    ),
    MilestoneStatus.REQUIRES_REVALIDATION: frozenset(
        {
            MilestoneStatus.COMPLETED_VERIFIED,
            MilestoneStatus.VERIFICATION_FAILED,
            MilestoneStatus.IN_PROGRESS,
            MilestoneStatus.COMPLETED_CLAIMED,
            MilestoneStatus.BLOCKED,
        }
    ),
    MilestoneStatus.BLOCKED: frozenset(
        {MilestoneStatus.IN_PROGRESS, MilestoneStatus.FAILED, MilestoneStatus.CANCELLED}
    ),
    # A verified Milestone is an immutable historical route node. Later
    # revisions may invalidate current-code assumptions, but that is recorded
    # by the active/final acceptance predicate rather than reopening history.
    MilestoneStatus.COMPLETED_VERIFIED: frozenset(),
    MilestoneStatus.FAILED: frozenset(),
    MilestoneStatus.CANCELLED: frozenset(),
}

_STEP_TRANSITIONS: dict[PlanStepStatus, frozenset[PlanStepStatus]] = {
    PlanStepStatus.PENDING: frozenset({PlanStepStatus.IN_PROGRESS, PlanStepStatus.CANCELLED}),
    PlanStepStatus.IN_PROGRESS: frozenset(
        {
            PlanStepStatus.COMPLETED_CLAIMED,
            PlanStepStatus.COMPLETED_VERIFIED,
            PlanStepStatus.FAILED,
            PlanStepStatus.CANCELLED,
        }
    ),
    PlanStepStatus.COMPLETED_CLAIMED: frozenset(
        {PlanStepStatus.COMPLETED_VERIFIED, PlanStepStatus.FAILED}
    ),
    PlanStepStatus.COMPLETED_VERIFIED: frozenset(),
    PlanStepStatus.FAILED: frozenset(),
    PlanStepStatus.CANCELLED: frozenset(),
}


class PlanRegistry:
    """Sole writer for Task, Goal, PlanVersion and Milestone lifecycle state."""

    def __init__(
        self,
        database: StateDatabase,
        semantic_projection: PlanProjection,
        *,
        prefetch_limit: int = 1,
        durable_event_guard: Callable[[str], bool] | None = None,
    ) -> None:
        if prefetch_limit < 0:
            raise ValueError("prefetch_limit cannot be negative")
        self.database = database
        self._semantic_projection = semantic_projection
        self.prefetch_limit = prefetch_limit
        # Predecessor Milestones kept HOT in the Working Set when a successor
        # becomes current.  Zero (FULL engagement) cools them at the boundary;
        # lighter engagement levels keep the last one hot so the model does not
        # re-read the files it just finished with.
        self.retain_predecessor_milestones = 0
        self._durable_event_guard = durable_event_guard
        self._migrate_step_review_decision_schema()
        with self.database.transaction() as conn:
            self._migrate_milestone_state_schema(conn)
            self._migrate_review_decision_schema(conn)
            conn.executescript(_SCHEMA)
            self._migrate_corrective_contract_schema(conn)
            self._migrate_acceptance_contract_schema(conn)
            self._migrate_step_dependency_schema(conn)

    @staticmethod
    def _migrate_step_dependency_schema(conn: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(v2_plan_step_identities)").fetchall()
        }
        if "historical_dependency_refs_json" not in columns:
            conn.execute(
                "ALTER TABLE v2_plan_step_identities ADD COLUMN "
                "historical_dependency_refs_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "source_plan_item_ids_json" not in columns:
            conn.execute(
                "ALTER TABLE v2_plan_step_identities ADD COLUMN "
                "source_plan_item_ids_json TEXT NOT NULL DEFAULT '[]'"
            )

    @staticmethod
    def _migrate_acceptance_contract_schema(conn: sqlite3.Connection) -> None:
        """Add lightweight requirement commitments and retire proof-era views."""

        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(v2_completion_criteria)").fetchall()
        }
        additions = {
            "requirement_id": "TEXT NOT NULL DEFAULT ''",
            "requirement_text": "TEXT NOT NULL DEFAULT ''",
            "claim_type": "TEXT NOT NULL DEFAULT 'STRUCTURAL'",
            "commitment_level": "TEXT NOT NULL DEFAULT 'STEP'",
            "verification_mode": "TEXT NOT NULL DEFAULT ''",
        }
        for name, declaration in additions.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE v2_completion_criteria ADD COLUMN {name} {declaration}")
        receipt_columns = {
            str(row["name"])
            for row in conn.execute(
                "PRAGMA table_info(v2_milestone_acceptance_receipts)"
            ).fetchall()
        }
        if receipt_columns and "unverified_criterion_ids_json" not in receipt_columns:
            conn.execute(
                "ALTER TABLE v2_milestone_acceptance_receipts ADD COLUMN "
                "unverified_criterion_ids_json TEXT NOT NULL DEFAULT '[]'"
            )
        # Older stores may retain the append-only proof-revision table.  It is
        # deliberately left as historical data, but no live route may consult
        # it: a Step contract is fixed for that Step and failed work creates a
        # meaningful successor Step instead of rewriting acceptance in place.
        conn.execute("DROP VIEW IF EXISTS v2_effective_plan_step_contracts")
        conn.execute(
            "CREATE VIEW v2_effective_plan_step_contracts AS "
            "SELECT contract_id,run_id,step_identity_id,expected_outcome,"
            "minimum_acceptance_json,failure_signals_json,source_event_id,"
            "created_cursor,created_at,0 AS contract_revision_number "
            "FROM v2_plan_step_contracts"
        )

    @staticmethod
    def _migrate_corrective_contract_schema(conn: sqlite3.Connection) -> None:
        """Extend legacy correction rows with their causal failure contract.

        A Corrective Step is not progress merely because its row exists.  The
        immutable row must also identify the failed fact and the revision from
        which the corrective work starts.  SQLite can add these nullable/defaulted
        columns without rewriting the append-only history already on disk.
        """

        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(v2_corrective_steps)").fetchall()
        }
        additions = {
            "failure_signature": "TEXT",
            "failure_criterion_ids_json": "TEXT NOT NULL DEFAULT '[]'",
            "baseline_revision_id": "TEXT",
            "baseline_cursor": "INTEGER",
        }
        for name, declaration in additions.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE v2_corrective_steps ADD COLUMN {name} {declaration}")

    def _migrate_step_review_decision_schema(self) -> None:
        """Add BLOCKED without breaking correction rows that reference reviews."""

        conn = self.database.connection
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='v2_plan_step_review_events'"
        ).fetchone()
        if row is None or "'BLOCKED'" in str(row["sql"]):
            return
        # SQLite cannot alter a CHECK constraint in place. Build the replacement
        # beside the stable parent table, then swap it into the same name. Renaming
        # the old parent first would make SQLite rewrite child-table foreign keys
        # to the temporary name and strand correction history when it is dropped.
        # This migration runs before Registry traffic begins; FK enforcement is
        # disabled only for the atomic table replacement and checked immediately
        # afterwards.
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            conn.executescript(
                """
                BEGIN IMMEDIATE;
                DROP TABLE IF EXISTS v2_plan_step_review_events_with_blocked;
                CREATE TABLE v2_plan_step_review_events_with_blocked (
                    review_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
                    milestone_identity_id TEXT NOT NULL
                        REFERENCES v2_milestone_identities(identity_id),
                    step_identity_id TEXT NOT NULL
                        REFERENCES v2_plan_step_identities(step_identity_id),
                    decision TEXT NOT NULL CHECK(decision IN (
                        'SATISFIED','CONTINUE','CORRECT','BLOCKED'
                    )),
                    summary TEXT NOT NULL,
                    evidence_event_ids_json TEXT NOT NULL,
                    created_step_ids_json TEXT NOT NULL,
                    source_event_id TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    created_cursor INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(step_identity_id,source_event_id)
                );
                INSERT INTO v2_plan_step_review_events_with_blocked
                SELECT * FROM v2_plan_step_review_events;
                DROP TABLE v2_plan_step_review_events;
                ALTER TABLE v2_plan_step_review_events_with_blocked
                RENAME TO v2_plan_step_review_events;
                COMMIT;
                """
            )
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.execute("PRAGMA foreign_keys=ON")
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError("PlanStep review migration violated existing foreign keys")

    @staticmethod
    def _migrate_milestone_state_schema(conn: sqlite3.Connection) -> None:
        """Upgrade the pre-Stage3 status CHECK without discarding history."""

        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='v2_milestone_state_events'"
        ).fetchone()
        if row is None or "REPAIRING" in str(row["sql"]):
            return
        conn.executescript(
            """
            ALTER TABLE v2_milestone_state_events
            RENAME TO v2_milestone_state_events_pre_stage3;
            CREATE TABLE v2_milestone_state_events (
                state_event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
                identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
                previous_status TEXT,
                status TEXT NOT NULL CHECK(status IN (
                    'PENDING','IN_PROGRESS','COMPLETED_CLAIMED','COMPLETED_VERIFIED',
                    'VERIFICATION_FAILED','REPAIRING','REQUIRES_REVALIDATION',
                    'BLOCKED','FAILED','CANCELLED'
                )),
                source_event_id TEXT NOT NULL,
                revision_id TEXT NOT NULL,
                created_cursor INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(identity_id, created_cursor)
            );
            INSERT INTO v2_milestone_state_events
            SELECT * FROM v2_milestone_state_events_pre_stage3;
            DROP TABLE v2_milestone_state_events_pre_stage3;
            """
        )

    @staticmethod
    def _migrate_review_decision_schema(conn: sqlite3.Connection) -> None:
        """Add the explicit future-only review decision without losing history."""

        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='v2_milestone_review_events'"
        ).fetchone()
        if row is None or "REPLAN_FUTURE" in str(row["sql"]):
            return
        conn.executescript(
            """
            ALTER TABLE v2_milestone_review_events
            RENAME TO v2_milestone_review_events_pre_replan;
            CREATE TABLE v2_milestone_review_events (
                review_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES v2_registry_runs(run_id),
                milestone_identity_id TEXT NOT NULL REFERENCES v2_milestone_identities(identity_id),
                decision TEXT NOT NULL CHECK(decision IN (
                    'CONTINUE','REPLAN_FUTURE','UPDATE_FUTURE','SPLIT_FUTURE','MERGE_FUTURE',
                    'CANCEL_FUTURE','REPLACE_FUTURE'
                )),
                reason TEXT NOT NULL,
                evidence_event_ids_json TEXT NOT NULL,
                previous_plan_version_id TEXT NOT NULL REFERENCES v2_plan_versions(plan_version_id),
                resulting_plan_version_id TEXT REFERENCES v2_plan_versions(plan_version_id),
                source_event_id TEXT NOT NULL,
                revision_id TEXT NOT NULL,
                created_cursor INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO v2_milestone_review_events
            SELECT * FROM v2_milestone_review_events_pre_replan;
            DROP TABLE v2_milestone_review_events_pre_replan;
            """
        )

    def _projection(self) -> PlanProjection:
        return self._semantic_projection

    def _require_durable_event(self, source_event_id: str) -> None:
        if self._durable_event_guard is not None and not self._durable_event_guard(source_event_id):
            raise RuntimeError(
                f"source Event is not present in the synced Event WAL: {source_event_id}"
            )

    @staticmethod
    def _milestone_structure(spec: MilestoneSpec) -> dict[str, object]:
        return {
            "canonical_id": spec.canonical_id,
            "title": spec.title,
            "description": spec.description,
            "completion_criteria": list(spec.completion_criteria),
            "verification": list(spec.verification),
            "depends_on": list(spec.depends_on),
            "entity_refs": list(spec.entity_refs),
            "objective": spec.objective,
            "scope": spec.scope,
            "target_outcome": spec.target_outcome,
            "downstream_assumptions": list(spec.downstream_assumptions),
            "non_goals": list(spec.non_goals),
            "source_plan_item_ids": list(spec.source_plan_item_ids),
            "criteria": [primitive(item) for item in spec.criteria],
        }

    @classmethod
    def _plan_structure(cls, plan: PlanSpec) -> dict[str, object]:
        return {
            "goal": plan.goal,
            "milestones": [cls._milestone_structure(item) for item in plan.milestones],
            "final_verification": list(plan.final_verification),
            "final_acceptance": [primitive(item) for item in plan.final_acceptance],
            "native_plan": primitive(plan.native_plan) if plan.native_plan is not None else None,
        }

    @classmethod
    def _resume_plan_compatible(cls, current: PlanSpec, candidate: PlanSpec) -> bool:
        """Allow durable resume metadata without rewriting the route skeleton.

        A recovery may carry a freshly serialized native Plan, final checklist,
        or additional future milestones. Those are provenance/navigation data,
        not permission to discard completed route identities. Existing
        Milestones therefore remain an ordered subsequence with the exact same
        terminal contract; new Milestones may only be appended. This is the
        smallest safe relaxation of the old byte-for-byte comparison and keeps
        all changes observable through a new PlanVersion.
        """
        previous = cls._plan_structure(current)
        incoming = cls._plan_structure(candidate)
        if previous["goal"] != incoming["goal"]:
            return False
        old_items = list(previous["milestones"])
        new_items = list(incoming["milestones"])
        if len(new_items) < len(old_items):
            return False
        for ordinal, old_item in enumerate(old_items):
            if old_item != new_items[ordinal]:
                return False
        return True

    @staticmethod
    def _coerce_milestone_status(value: str | MilestoneStatus) -> MilestoneStatus:
        if isinstance(value, MilestoneStatus):
            return value
        normalized = value.strip().replace("-", "_")
        codex = {
            "pending": MilestoneStatus.PENDING,
            "inProgress": MilestoneStatus.IN_PROGRESS,
            "in_progress": MilestoneStatus.IN_PROGRESS,
            "completed": MilestoneStatus.COMPLETED_CLAIMED,
        }
        if normalized in codex:
            return codex[normalized]
        try:
            return MilestoneStatus(normalized.upper())
        except ValueError as exc:
            raise ValueError(f"unknown Milestone status: {value!r}") from exc

    @staticmethod
    def _scope_step_id(milestone_id: str, step_id: str) -> str:
        """Give every local Provider Step one globally unique semantic address."""

        milestone = milestone_id.strip().upper()
        raw = step_id.strip()
        if not raw:
            raise ValueError(f"{milestone} PlanStep has an empty ID")
        qualified = re.fullmatch(r"(M\d{3,})[.-](.+)", raw, re.IGNORECASE)
        if qualified is not None:
            owner = qualified.group(1).upper()
            if owner != milestone:
                raise ValueError(
                    f"PlanStep {raw!r} cannot be owned by both {owner} and {milestone}"
                )
            local = qualified.group(2).strip()
        else:
            if re.match(r"M\d{3,}", raw, re.IGNORECASE):
                raise ValueError(f"PlanStep {raw!r} has an invalid qualified address")
            local = raw
        if not local:
            raise ValueError(f"{milestone} PlanStep has an empty local ID")
        return f"{milestone}.{local}"

    @staticmethod
    def _coerce_plan(plan: PlanSpec | Mapping[str, object]) -> PlanSpec:
        source = plan if isinstance(plan, PlanSpec) else PlanSpec.from_dict(plan)
        result = PlanSpec(
            goal=source.goal,
            milestones=tuple(
                MilestoneSpec(
                    canonical_id=item.canonical_id,
                    title=item.title,
                    description=item.description,
                    completion_criteria=item.completion_criteria,
                    verification=item.verification,
                    depends_on=item.depends_on,
                    status=PlanRegistry._coerce_milestone_status(item.status).value,
                    entity_refs=item.entity_refs,
                    objective=item.objective,
                    scope=item.scope,
                    target_outcome=item.target_outcome,
                    downstream_assumptions=item.downstream_assumptions,
                    non_goals=item.non_goals,
                    source_plan_item_ids=item.source_plan_item_ids,
                    criteria=item.criteria,
                    steps=tuple(
                        replace(
                            step,
                            step_id=PlanRegistry._scope_step_id(
                                item.canonical_id,
                                step.step_id,
                            ),
                        )
                        for step in item.steps
                    ),
                )
                for item in source.milestones
            ),
            final_verification=source.final_verification,
            final_acceptance=source.final_acceptance,
            native_plan=source.native_plan,
        )
        PlanRegistry._validate_step_addresses(result)
        PlanRegistry._validate_acyclic(result)
        return result

    @staticmethod
    def _validate_step_addresses(plan: PlanSpec) -> None:
        """Reject two spellings that would address the same route node."""

        for milestone in plan.milestones:
            seen: dict[PlanStepAddress, str] = {}
            for step in milestone.steps:
                address = PlanStepAddress.parse(step.step_id)
                if address is None:
                    continue
                previous = seen.get(address)
                if previous is not None:
                    raise ValueError(
                        "PlanStep address aliases cannot name two route nodes: "
                        f"{previous!r} and {step.step_id!r}"
                    )
                seen[address] = step.step_id

    @staticmethod
    def _validate_acyclic(plan: PlanSpec) -> None:
        dependencies = {item.canonical_id: set(item.depends_on) for item in plan.milestones}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError(f"Milestone dependency cycle includes {node}")
            if node in visited:
                return
            visiting.add(node)
            for dependency in dependencies[node]:
                visit(dependency)
            visiting.remove(node)
            visited.add(node)

        for milestone_id in dependencies:
            visit(milestone_id)

    @staticmethod
    def _next_cursor(conn: sqlite3.Connection, run_id: str) -> int:
        row = conn.execute(
            "SELECT next_cursor FROM v2_registry_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        cursor = int(row["next_cursor"]) + 1
        conn.execute(
            "UPDATE v2_registry_runs SET next_cursor=? WHERE run_id=?",
            (cursor, run_id),
        )
        return cursor

    @staticmethod
    def _seed_task_requirements_conn(
        conn: sqlite3.Connection,
        *,
        task_id: str,
        run_id: str,
        user_request: str,
        source_event_id: str,
    ) -> str:
        """Persist the original Task checklist independently of every Plan."""

        raw_digest = digest({"original_user_request": user_request})
        requirement_set_id = stable_id(
            "requirement_set_",
            {"task": task_id, "raw_request_digest": raw_digest},
        )
        existing = conn.execute(
            "SELECT requirement_set_id,raw_request_digest FROM v2_task_requirement_sets "
            "WHERE task_id=?",
            (task_id,),
        ).fetchone()
        if existing is not None:
            if (
                str(existing["requirement_set_id"]) != requirement_set_id
                or str(existing["raw_request_digest"]) != raw_digest
            ):
                raise ValueError("the immutable Task requirement set changed during resume")
            return requirement_set_id
        now = utc_now()
        conn.execute(
            "INSERT INTO v2_task_requirement_sets VALUES(?,?,?,?,?,?,?)",
            (
                requirement_set_id,
                task_id,
                run_id,
                raw_digest,
                "deterministic-task-clauses-v1",
                source_event_id,
                now,
            ),
        )
        for requirement in extract_task_requirements(task_id, user_request):
            conn.execute(
                "INSERT INTO v2_task_requirements VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    requirement.requirement_id,
                    requirement_set_id,
                    run_id,
                    requirement.ordinal,
                    requirement.text,
                    requirement.text_digest,
                    requirement.modality,
                    requirement.category,
                    1 if requirement.required else 0,
                    requirement.source_line,
                    source_event_id,
                    now,
                ),
            )
        return requirement_set_id

    @staticmethod
    def _link_task_requirements_conn(
        conn: sqlite3.Connection,
        *,
        run_id: str,
        plan_version_id: str,
        source_event_id: str,
    ) -> None:
        """Index immutable requirements onto the current TPG route.

        These links are address translations, not correctness claims.  A weak
        or absent lexical match remains visible for final targeted review.
        """

        requirements = conn.execute(
            "SELECT requirement_id,requirement_text FROM v2_task_requirements "
            "WHERE run_id=? ORDER BY ordinal",
            (run_id,),
        ).fetchall()
        criteria = conn.execute(
            "SELECT c.criterion_identity_id,c.requirement_text,c.observable_outcome,"
            "mv.title,mi.canonical_id FROM v2_completion_criteria c "
            "JOIN v2_plan_milestones pm ON pm.milestone_version_id=c.milestone_version_id "
            "JOIN v2_milestone_versions mv ON mv.version_id=pm.milestone_version_id "
            "JOIN v2_milestone_identities mi ON mi.identity_id=pm.identity_id "
            "WHERE c.run_id=? AND pm.plan_version_id=?",
            (run_id, plan_version_id),
        ).fetchall()
        for requirement in requirements:
            ranked: list[tuple[float, str, sqlite3.Row]] = []
            for criterion in criteria:
                target = " ".join(
                    (
                        str(criterion["requirement_text"]),
                        str(criterion["observable_outcome"]),
                        str(criterion["title"]),
                    )
                )
                score, basis = requirement_link_score(str(requirement["requirement_text"]), target)
                ranked.append((score, basis, criterion))
            if not ranked:
                continue
            best = max(score for score, _basis, _criterion in ranked)
            if best < 0.18:
                continue
            for score, basis, criterion in ranked:
                if score + 0.12 < best:
                    continue
                criterion_identity_id = str(criterion["criterion_identity_id"])
                conn.execute(
                    "INSERT OR IGNORE INTO v2_requirement_plan_links VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        stable_id(
                            "requirement_link_",
                            {
                                "requirement": requirement["requirement_id"],
                                "criterion": criterion_identity_id,
                            },
                        ),
                        run_id,
                        str(requirement["requirement_id"]),
                        criterion_identity_id,
                        plan_version_id,
                        basis,
                        float(score),
                        source_event_id,
                        utc_now(),
                    ),
                )

    def initialize_task(
        self,
        *,
        repository_id: str,
        run_id: str,
        branch_id: str,
        revision_id: str,
        user_request: str,
        plan: PlanSpec | Mapping[str, object],
        source_event_id: str,
        current_milestone_id: str | None = None,
    ) -> PlanApplication:
        """Validate first, then atomically seed Registry and Semantic Graph."""

        checked = self._coerce_plan(plan)
        if not all(
            value.strip()
            for value in (
                repository_id,
                run_id,
                branch_id,
                revision_id,
                user_request,
                source_event_id,
            )
        ):
            raise ValueError("Task identity and source fields must be non-empty")
        self._require_durable_event(source_event_id)
        projection = self._projection()
        task_id = stable_id("task_", {"repository": repository_id, "run": run_id})
        with self.database.transaction() as conn:
            existing = conn.execute(
                "SELECT task_id, repository_id, branch_id, initial_revision_id, user_request "
                "FROM v2_tasks WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["task_id"] != task_id
                    or existing["repository_id"] != repository_id
                    or existing["branch_id"] != branch_id
                    or existing["user_request"] != user_request
                ):
                    raise ValueError("run_id is already bound to a different Task scope")
                current_plan = self._current_plan_conn(conn, run_id)
                if not self._resume_plan_compatible(current_plan, checked):
                    raise ValueError(
                        "a resumed Task cannot rewrite completed Plan structure; "
                        "only native-plan metadata or appended future Milestones are "
                        "allowed during recovery"
                    )
                # ``initial_revision_id`` is immutable historical provenance,
                # not part of the Task identity. A resumed Run is expected to
                # present the newer WorkspaceRevision produced by real file
                # changes while preserving the same Task.
                self._seed_task_requirements_conn(
                    conn,
                    task_id=task_id,
                    run_id=run_id,
                    user_request=user_request,
                    source_event_id=source_event_id,
                )
                application = self._apply_plan_conn(
                    conn,
                    projection,
                    run_id=run_id,
                    revision_id=revision_id,
                    plan=checked,
                    source_event_id=source_event_id,
                    requested_current=current_milestone_id,
                )
                self._link_task_requirements_conn(
                    conn,
                    run_id=run_id,
                    plan_version_id=application.plan_version_id,
                    source_event_id=source_event_id,
                )
                return application

            now = utc_now()
            conn.execute(
                "INSERT INTO v2_registry_runs VALUES(?,?,?,?,?)",
                (run_id, repository_id, branch_id, 0, now),
            )
            conn.execute(
                "INSERT INTO v2_tasks VALUES(?,?,?,?,?,?,?,?)",
                (
                    task_id,
                    repository_id,
                    run_id,
                    branch_id,
                    revision_id,
                    user_request,
                    source_event_id,
                    now,
                ),
            )
            self._seed_task_requirements_conn(
                conn,
                task_id=task_id,
                run_id=run_id,
                user_request=user_request,
                source_event_id=source_event_id,
            )
            self._record_task_state_conn(
                conn,
                run_id=run_id,
                status=TaskStatus.CREATED,
                revision_id=revision_id,
                source_event_id=source_event_id,
            )
            application = self._apply_plan_conn(
                conn,
                projection,
                run_id=run_id,
                revision_id=revision_id,
                plan=checked,
                source_event_id=source_event_id,
                requested_current=current_milestone_id,
            )
            self._link_task_requirements_conn(
                conn,
                run_id=run_id,
                plan_version_id=application.plan_version_id,
                source_event_id=source_event_id,
            )
            return application

    def apply_plan(
        self,
        *,
        run_id: str,
        revision_id: str,
        plan: PlanSpec | Mapping[str, object],
        source_event_id: str,
        current_milestone_id: str | None = None,
    ) -> PlanApplication:
        checked = self._coerce_plan(plan)
        if not run_id.strip() or not revision_id.strip() or not source_event_id.strip():
            raise ValueError("Plan Update requires run, revision, and source Event")
        self._require_durable_event(source_event_id)
        projection = self._projection()
        with self.database.transaction() as conn:
            current_plan = self._current_plan_conn(conn, run_id)
            if digest(self._plan_structure(current_plan)) != digest(self._plan_structure(checked)):
                raise ValueError(
                    "structural Plan changes require a post-verification "
                    "MilestoneReview(REPLAN_FUTURE)"
                )
            application = self._apply_plan_conn(
                conn,
                projection,
                run_id=run_id,
                revision_id=revision_id,
                plan=checked,
                source_event_id=source_event_id,
                requested_current=current_milestone_id,
            )
            self._link_task_requirements_conn(
                conn,
                run_id=run_id,
                plan_version_id=application.plan_version_id,
                source_event_id=source_event_id,
            )
            return application

    def _apply_plan_conn(
        self,
        conn: sqlite3.Connection,
        projection: PlanProjection,
        *,
        run_id: str,
        revision_id: str,
        plan: PlanSpec,
        source_event_id: str,
        requested_current: str | None,
    ) -> PlanApplication:
        task = conn.execute("SELECT * FROM v2_tasks WHERE run_id=?", (run_id,)).fetchone()
        if task is None:
            raise KeyError(f"unknown run: {run_id}")
        plan_document = self._plan_structure(plan)
        plan_digest = digest(plan_document)
        existing = conn.execute(
            "SELECT * FROM v2_plan_versions WHERE task_id=? AND plan_digest=?",
            (task["task_id"], plan_digest),
        ).fetchone()
        if existing is not None:
            self._ensure_plan_details_conn(
                conn,
                run_id=run_id,
                plan_version_id=str(existing["plan_version_id"]),
                plan=plan,
                revision_id=revision_id,
                source_event_id=source_event_id,
            )
            state_cursor = self._ensure_initial_milestone_states_conn(
                conn,
                run_id=run_id,
                plan_version_id=str(existing["plan_version_id"]),
                revision_id=revision_id,
                source_event_id=source_event_id,
            )
            if requested_current is not None:
                current = self.current(run_id, conn=conn)
                if current.canonical_id != requested_current:
                    self._switch_current_conn(
                        conn,
                        projection,
                        run_id=run_id,
                        canonical_id=requested_current,
                        revision_id=revision_id,
                        source_event_id=source_event_id,
                        plan_version_id=existing["plan_version_id"],
                    )
            # A lightweight Milestone activation materializes Steps without
            # creating a new stage-skeleton PlanVersion. Publish the newly
            # authoritative current Step in the same Registry transaction;
            # otherwise the Registry advances while the TPG route card still
            # reports an unactivated Milestone.
            current_step = self.current_step(run_id, conn=conn)
            if current_step is not None:
                self._project_current_step_conn(
                    conn,
                    run_id=run_id,
                    revision_id=revision_id,
                    source_event_id=source_event_id,
                    cursor=self._next_cursor(conn, run_id),
                )
            current = self.current(run_id, conn=conn)
            identity_rows = conn.execute(
                "SELECT identity_id FROM v2_plan_milestones "
                "WHERE plan_version_id=? ORDER BY ordinal",
                (existing["plan_version_id"],),
            ).fetchall()
            return PlanApplication(
                task_id=task["task_id"],
                goal_id=existing["goal_id"],
                plan_version_id=existing["plan_version_id"],
                plan_version_number=int(existing["version_number"]),
                current_milestone_id=current.identity_id,
                milestone_identity_ids=tuple(row["identity_id"] for row in identity_rows),
                created_new_version=False,
                cursor=max(int(existing["created_cursor"]), state_cursor or 0),
            )

        previous = conn.execute(
            "SELECT * FROM v2_plan_versions WHERE task_id=? ORDER BY version_number DESC LIMIT 1",
            (task["task_id"],),
        ).fetchone()
        version_number = 1 if previous is None else int(previous["version_number"]) + 1
        cursor = self._next_cursor(conn, run_id)
        goal_digest = digest({"goal": plan.goal})
        goal_id = stable_id("goal_", {"task": task["task_id"], "goal_digest": goal_digest})
        conn.execute(
            "INSERT OR IGNORE INTO v2_goals VALUES(?,?,?,?,?,?)",
            (
                goal_id,
                task["task_id"],
                plan.goal,
                goal_digest,
                source_event_id,
                utc_now(),
            ),
        )
        plan_version_id = stable_id(
            "planv_",
            {
                "task": task["task_id"],
                "version": version_number,
                "digest": plan_digest,
            },
        )
        conn.execute(
            "INSERT INTO v2_plan_versions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                plan_version_id,
                task["task_id"],
                goal_id,
                version_number,
                previous["plan_version_id"] if previous else None,
                plan_digest,
                json.dumps(plan_document, sort_keys=True, separators=(",", ":")),
                source_event_id,
                revision_id,
                cursor,
                utc_now(),
            ),
        )

        records: list[MilestoneProjectionRecord] = []
        identity_by_canonical: dict[str, str] = {}
        for ordinal, spec in enumerate(plan.milestones):
            identity_id = stable_id("mil_", {"run": run_id, "canonical": spec.canonical_id})
            identity_by_canonical[spec.canonical_id] = identity_id
            conn.execute(
                "INSERT OR IGNORE INTO v2_milestone_identities VALUES(?,?,?,?,?,?)",
                (
                    identity_id,
                    task["task_id"],
                    run_id,
                    spec.canonical_id,
                    plan_version_id,
                    utc_now(),
                ),
            )
            spec_digest = digest(self._milestone_structure(spec))
            prior_version = conn.execute(
                "SELECT * FROM v2_milestone_versions WHERE identity_id=? "
                "ORDER BY version_number DESC LIMIT 1",
                (identity_id,),
            ).fetchone()
            same = conn.execute(
                "SELECT * FROM v2_milestone_versions WHERE identity_id=? AND spec_digest=?",
                (identity_id, spec_digest),
            ).fetchone()
            if same is not None:
                version_id = same["version_id"]
                milestone_version_number = int(same["version_number"])
                previous_version_id = same["previous_version_id"]
                created_new_version = False
            else:
                milestone_version_number = (
                    1 if prior_version is None else int(prior_version["version_number"]) + 1
                )
                previous_version_id = (
                    prior_version["version_id"] if prior_version is not None else None
                )
                version_id = stable_id(
                    "milv_",
                    {
                        "identity": identity_id,
                        "version": milestone_version_number,
                        "digest": spec_digest,
                    },
                )
                conn.execute(
                    "INSERT INTO v2_milestone_versions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        version_id,
                        identity_id,
                        milestone_version_number,
                        plan_version_id,
                        previous_version_id,
                        spec_digest,
                        spec.title,
                        spec.description,
                        json.dumps(spec.completion_criteria),
                        json.dumps(spec.verification),
                        self._coerce_milestone_status(spec.status).value,
                        json.dumps(spec.entity_refs),
                        source_event_id,
                        revision_id,
                        cursor,
                        utc_now(),
                    ),
                )
                created_new_version = True
            conn.execute(
                "INSERT INTO v2_plan_milestones VALUES(?,?,?,?)",
                (plan_version_id, identity_id, version_id, ordinal),
            )
            records.append(
                MilestoneProjectionRecord(
                    identity_id=identity_id,
                    version_id=version_id,
                    version_number=milestone_version_number,
                    ordinal=ordinal,
                    spec=spec,
                    previous_version_id=previous_version_id,
                    created_new_version=created_new_version,
                )
            )

        for spec in plan.milestones:
            for dependency in spec.depends_on:
                conn.execute(
                    "INSERT INTO v2_milestone_dependencies VALUES(?,?,?,?,?)",
                    (
                        plan_version_id,
                        identity_by_canonical[spec.canonical_id],
                        identity_by_canonical[dependency],
                        source_event_id,
                        cursor,
                    ),
                )

        self._cancel_superseded_pending_steps_conn(
            conn,
            run_id=run_id,
            plan_version_id=plan_version_id,
            plan=plan,
            revision_id=revision_id,
            source_event_id=source_event_id,
        )
        self._ensure_plan_details_conn(
            conn,
            run_id=run_id,
            plan_version_id=plan_version_id,
            plan=plan,
            revision_id=revision_id,
            source_event_id=source_event_id,
        )

        projection_update = PlanProjectionInput(
            repository_id=task["repository_id"],
            run_id=run_id,
            branch_id=task["branch_id"],
            revision_id=revision_id,
            task_id=task["task_id"],
            goal_id=goal_id,
            plan_version_id=plan_version_id,
            plan_version_number=version_number,
            previous_plan_version_id=(previous["plan_version_id"] if previous else None),
            source_event_id=source_event_id,
            cursor=cursor,
            plan=plan,
            milestones=tuple(records),
        )
        projection.project_plan(conn, projection_update)

        old_current = conn.execute(
            "SELECT mi.canonical_id FROM v2_semantic_edges edge "
            "JOIN v2_milestone_identities mi ON mi.identity_id=edge.target_id "
            "WHERE edge.run_id=? AND edge.edge_type='CURRENT_MILESTONE' "
            "AND edge.valid_to_cursor IS NULL",
            (run_id,),
        ).fetchone()
        canonical_current = requested_current
        if canonical_current is None and old_current is not None:
            if old_current["canonical_id"] in identity_by_canonical:
                canonical_current = old_current["canonical_id"]
        canonical_current = canonical_current or plan.milestones[0].canonical_id
        if canonical_current not in identity_by_canonical:
            raise ValueError("current Milestone must belong to the new PlanVersion")
        self._switch_current_conn(
            conn,
            projection,
            run_id=run_id,
            canonical_id=canonical_current,
            revision_id=revision_id,
            source_event_id=source_event_id,
            plan_version_id=plan_version_id,
            cursor=cursor,
        )
        state_cursor = self._ensure_initial_milestone_states_conn(
            conn,
            run_id=run_id,
            plan_version_id=plan_version_id,
            revision_id=revision_id,
            source_event_id=source_event_id,
        )
        return PlanApplication(
            task_id=task["task_id"],
            goal_id=goal_id,
            plan_version_id=plan_version_id,
            plan_version_number=version_number,
            current_milestone_id=identity_by_canonical[canonical_current],
            milestone_identity_ids=tuple(item.identity_id for item in records),
            created_new_version=True,
            cursor=max(cursor, state_cursor or 0),
        )

    def _cancel_superseded_pending_steps_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        plan_version_id: str,
        plan: PlanSpec,
        revision_id: str,
        source_event_id: str,
    ) -> None:
        """Retire only unstarted Step addresses replaced by a future replan.

        Step contracts stay immutable after allocation. A changed pending
        route therefore gets a fresh address, while the old PENDING address is
        retained as history and marked CANCELLED. Executed work is never
        rewritten by this route-version transaction.
        """

        desired = {
            milestone.canonical_id: {step.step_id for step in milestone.steps}
            for milestone in plan.milestones
        }
        for canonical_id, desired_step_ids in desired.items():
            identity = conn.execute(
                "SELECT mi.identity_id FROM v2_plan_milestones pm "
                "JOIN v2_milestone_identities mi ON mi.identity_id=pm.identity_id "
                "WHERE pm.plan_version_id=? AND mi.run_id=? AND mi.canonical_id=?",
                (plan_version_id, run_id, canonical_id),
            ).fetchone()
            if identity is None:
                continue
            identity_id = str(identity["identity_id"])
            if self._latest_milestone_status_conn(conn, identity_id) is not (
                MilestoneStatus.PENDING
            ):
                continue
            rows = conn.execute(
                "SELECT si.step_identity_id,si.canonical_step_id,"
                "COALESCE((SELECT sse.status FROM v2_plan_step_state_events sse "
                "WHERE sse.step_identity_id=si.step_identity_id "
                "ORDER BY sse.created_cursor DESC LIMIT 1),'PENDING') AS status "
                "FROM v2_plan_step_identities si "
                "WHERE si.run_id=? AND si.milestone_identity_id=?",
                (run_id, identity_id),
            ).fetchall()
            for row in rows:
                if (
                    str(row["canonical_step_id"]) in desired_step_ids
                    or PlanStepStatus(str(row["status"])) is not PlanStepStatus.PENDING
                ):
                    continue
                self._record_step_state_conn(
                    conn,
                    run_id=run_id,
                    step_identity_id=str(row["step_identity_id"]),
                    status=PlanStepStatus.CANCELLED,
                    revision_id=revision_id,
                    source_event_id=source_event_id,
                )

    def _ensure_plan_details_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        plan_version_id: str,
        plan: PlanSpec,
        revision_id: str,
        source_event_id: str,
    ) -> None:
        """Persist criteria and Step identities without inflating PlanVersion.

        Structural Milestone changes already created ``plan_version_id``. Step
        progress and corrective work are an append-only execution history under
        the stable Milestone identity and deliberately do not alter that Plan.
        """

        semantic_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_semantic_nodes'"
        ).fetchone()
        plan_already_projected = bool(
            semantic_table
            and conn.execute(
                "SELECT 1 FROM v2_semantic_nodes WHERE node_id=? AND node_type='PlanVersion'",
                (plan_version_id,),
            ).fetchone()
        )
        projection_scope: tuple[str, str] | None = None
        for spec in plan.milestones:
            row = conn.execute(
                "SELECT mi.identity_id,pm.milestone_version_id "
                "FROM v2_plan_milestones pm JOIN v2_milestone_identities mi "
                "ON mi.identity_id=pm.identity_id "
                "WHERE pm.plan_version_id=? AND mi.run_id=? AND mi.canonical_id=?",
                (plan_version_id, run_id, spec.canonical_id),
            ).fetchone()
            if row is None:
                continue
            identity_id = str(row["identity_id"])
            milestone_version_id = str(row["milestone_version_id"])
            for criterion in spec.criteria:
                criterion_identity_id = stable_id(
                    "criterion_",
                    {
                        "milestone_version": milestone_version_id,
                        "criterion": criterion.criterion_id,
                    },
                )
                cursor = self._next_cursor(conn, run_id)
                conn.execute(
                    "INSERT OR IGNORE INTO v2_completion_criteria("
                    "criterion_identity_id,run_id,milestone_identity_id,milestone_version_id,"
                    "local_criterion_id,observable_outcome,required_evidence_types_json,"
                    "entity_refs_json,test_selectors_json,required,requirement_id,"
                    "requirement_text,claim_type,commitment_level,source_event_id,"
                    "created_cursor,created_at,verification_mode) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        criterion_identity_id,
                        run_id,
                        identity_id,
                        milestone_version_id,
                        criterion.criterion_id,
                        criterion.observable_outcome,
                        json.dumps([item.value for item in criterion.required_evidence_types]),
                        json.dumps(criterion.entity_refs),
                        json.dumps(criterion.test_selectors),
                        int(criterion.required),
                        criterion.requirement_id,
                        criterion.requirement_text,
                        criterion.claim_type.value,
                        criterion.commitment_level.value,
                        source_event_id,
                        cursor,
                        utc_now(),
                        (
                            criterion.verification_mode.value
                            if criterion.verification_mode is not None
                            else ""
                        ),
                    ),
                )
            for step in spec.steps:
                step_identity_id = self._ensure_step_conn(
                    conn,
                    run_id=run_id,
                    milestone_identity_id=identity_id,
                    step=step,
                    revision_id=revision_id,
                    source_event_id=source_event_id,
                )
                if not plan_already_projected:
                    continue
                semantic_step = conn.execute(
                    "SELECT 1 FROM v2_semantic_nodes WHERE node_id=? AND node_type='PlanStep'",
                    (step_identity_id,),
                ).fetchone()
                if semantic_step is not None:
                    continue
                if projection_scope is None:
                    projection_scope = self._projection_scope_conn(conn, run_id)
                repository_id, branch_id = projection_scope
                step_cursor = self._next_cursor(conn, run_id)
                self._projection().project_plan_step(
                    conn,
                    repository_id=repository_id,
                    run_id=run_id,
                    branch_id=branch_id,
                    revision_id=revision_id,
                    plan_version_id=plan_version_id,
                    milestone_identity_id=identity_id,
                    step_identity_id=step_identity_id,
                    step=step,
                    source_event_id=source_event_id,
                    cursor=step_cursor,
                )

    def _ensure_step_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        milestone_identity_id: str,
        step: PlanStepSpec,
        revision_id: str,
        source_event_id: str,
    ) -> str:
        step_identity_id = stable_id("step_", {"run": run_id, "canonical_step_id": step.step_id})
        existing = conn.execute(
            "SELECT * FROM v2_plan_step_identities WHERE run_id=? AND canonical_step_id=?",
            (run_id, step.step_id),
        ).fetchone()
        created = existing is None
        if existing is not None:
            if str(existing["milestone_identity_id"]) != milestone_identity_id:
                raise ValueError("PlanStep identity cannot move to another Milestone")
            step_identity_id = str(existing["step_identity_id"])
        else:
            cursor = self._next_cursor(conn, run_id)
            conn.execute(
                "INSERT INTO v2_plan_step_identities("
                "step_identity_id,run_id,milestone_identity_id,canonical_step_id,title,"
                "corrective,criterion_ids_json,entity_refs_json,"
                "historical_dependency_refs_json,source_plan_item_ids_json,"
                "source_event_id,created_cursor,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    step_identity_id,
                    run_id,
                    milestone_identity_id,
                    step.step_id,
                    step.title,
                    int(step.corrective),
                    json.dumps(step.criterion_ids),
                    json.dumps(step.entity_refs),
                    json.dumps(step.historical_dependency_refs),
                    json.dumps(step.source_plan_item_ids),
                    source_event_id,
                    cursor,
                    utc_now(),
                ),
            )
        acceptance_json = json.dumps(
            primitive(step.minimum_acceptance),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        failure_json = json.dumps(
            step.failure_signals,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        contract = conn.execute(
            "SELECT expected_outcome,minimum_acceptance_json,failure_signals_json "
            "FROM v2_plan_step_contracts WHERE step_identity_id=?",
            (step_identity_id,),
        ).fetchone()
        if contract is None:
            contract_cursor = self._next_cursor(conn, run_id)
            conn.execute(
                "INSERT INTO v2_plan_step_contracts VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    stable_id("stepcontract_", {"step": step_identity_id}),
                    run_id,
                    step_identity_id,
                    step.expected_outcome,
                    acceptance_json,
                    failure_json,
                    source_event_id,
                    contract_cursor,
                    utc_now(),
                ),
            )
        elif (
            str(contract["expected_outcome"]),
            str(contract["minimum_acceptance_json"]),
            str(contract["failure_signals_json"]),
        ) != (step.expected_outcome, acceptance_json, failure_json):
            raise ValueError("PlanStep contract is immutable after its identity is allocated")
        if created:
            # Plan/Harness status fields are observations, not execution
            # authority. Every new route node starts PENDING and can advance
            # only through the WAL-backed action/evidence/acceptance chain.
            self._record_step_state_conn(
                conn,
                run_id=run_id,
                step_identity_id=step_identity_id,
                status=PlanStepStatus.PENDING,
                revision_id=revision_id,
                source_event_id=source_event_id,
            )
        return step_identity_id

    def _record_step_state_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        step_identity_id: str,
        status: PlanStepStatus,
        revision_id: str,
        source_event_id: str,
    ) -> int:
        row = conn.execute(
            "SELECT status FROM v2_plan_step_state_events WHERE step_identity_id=? "
            "ORDER BY created_cursor DESC LIMIT 1",
            (step_identity_id,),
        ).fetchone()
        previous = None if row is None else str(row["status"])
        if previous == status.value:
            existing = conn.execute(
                "SELECT created_cursor FROM v2_plan_step_state_events "
                "WHERE step_identity_id=? ORDER BY created_cursor DESC LIMIT 1",
                (step_identity_id,),
            ).fetchone()
            return int(existing["created_cursor"]) if existing is not None else 0
        if previous is not None:
            previous_status = PlanStepStatus(previous)
            if status not in _STEP_TRANSITIONS[previous_status]:
                raise ValueError(
                    f"illegal PlanStep state transition: {previous_status.value} -> {status.value}"
                )
        cursor = self._next_cursor(conn, run_id)
        state_event_id = stable_id(
            "stepstate_",
            {
                "step": step_identity_id,
                "status": status.value,
                "source": source_event_id,
                "cursor": cursor,
            },
        )
        conn.execute(
            "INSERT INTO v2_plan_step_state_events VALUES(?,?,?,?,?,?,?,?,?)",
            (
                state_event_id,
                run_id,
                step_identity_id,
                previous,
                status.value,
                source_event_id,
                revision_id,
                cursor,
                utc_now(),
            ),
        )
        return cursor

    @staticmethod
    def _projection_scope_conn(conn: sqlite3.Connection, run_id: str) -> tuple[str, str]:
        task = conn.execute(
            "SELECT repository_id,branch_id FROM v2_tasks WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if task is None:
            raise KeyError(run_id)
        return str(task["repository_id"]), str(task["branch_id"])

    def _project_current_step_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        revision_id: str,
        source_event_id: str,
        cursor: int,
    ) -> None:
        current = self.current(run_id, conn=conn)
        step = self.current_step(run_id, conn=conn)
        repository_id, branch_id = self._projection_scope_conn(conn, run_id)
        self._projection().project_current_step(
            conn,
            repository_id=repository_id,
            run_id=run_id,
            branch_id=branch_id,
            revision_id=revision_id,
            plan_version_id=current.plan_version_id,
            milestone_identity_id=current.identity_id,
            step_identity_id=(str(step["step_identity_id"]) if step is not None else None),
            source_event_id=source_event_id,
            cursor=cursor,
        )

    def _ensure_initial_milestone_states_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        plan_version_id: str,
        revision_id: str,
        source_event_id: str,
    ) -> int | None:
        """Create PENDING state for new route identities only.

        A Plan may report provider-native progress, but that report cannot
        mutate the authoritative execution route. Existing state is preserved;
        new Milestones become visible as PENDING until real Agent work enters
        the WAL-backed acceptance chain.
        """

        last_cursor: int | None = None
        identities = conn.execute(
            "SELECT mi.identity_id FROM v2_plan_milestones pm "
            "JOIN v2_milestone_identities mi ON mi.identity_id=pm.identity_id "
            "WHERE pm.plan_version_id=? AND mi.run_id=? ORDER BY pm.ordinal",
            (plan_version_id, run_id),
        ).fetchall()
        for identity in identities:
            prior = conn.execute(
                "SELECT 1 FROM v2_milestone_state_events WHERE identity_id=? LIMIT 1",
                (str(identity["identity_id"]),),
            ).fetchone()
            if prior is not None:
                continue
            event = self._record_milestone_state_conn(
                conn,
                run_id=run_id,
                identity_id=str(identity["identity_id"]),
                status=MilestoneStatus.PENDING,
                revision_id=revision_id,
                source_event_id=source_event_id,
            )
            if event is not None:
                last_cursor = event.cursor
        return last_cursor

    def _record_task_state_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        status: TaskStatus,
        revision_id: str,
        source_event_id: str,
        allow_repository_stream_reopen: bool = False,
    ) -> TaskStateEvent | None:
        task = conn.execute("SELECT task_id FROM v2_tasks WHERE run_id=?", (run_id,)).fetchone()
        if task is None:
            raise KeyError(f"unknown run: {run_id}")
        prior = conn.execute(
            "SELECT status FROM v2_task_state_events WHERE run_id=? "
            "ORDER BY created_cursor DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        previous = TaskStatus(str(prior["status"])) if prior is not None else None
        if previous == status:
            return None
        repository_stream_reopen = (
            allow_repository_stream_reopen
            and previous is TaskStatus.FAILED
            and status is TaskStatus.EXECUTING
        )
        if (
            previous is not None
            and status not in _TASK_TRANSITIONS[previous]
            and not repository_stream_reopen
        ):
            raise ValueError(f"illegal Task state transition: {previous.value} -> {status.value}")
        cursor = self._next_cursor(conn, run_id)
        state_event_id = stable_id(
            "taskstate_",
            {"run": run_id, "status": status.value, "source": source_event_id, "cursor": cursor},
        )
        conn.execute(
            "INSERT INTO v2_task_state_events VALUES(?,?,?,?,?,?,?,?,?)",
            (
                state_event_id,
                run_id,
                task["task_id"],
                previous.value if previous is not None else None,
                status.value,
                source_event_id,
                revision_id,
                cursor,
                utc_now(),
            ),
        )
        return TaskStateEvent(
            state_event_id=state_event_id,
            run_id=run_id,
            previous_status=previous,
            status=status,
            source_event_id=source_event_id,
            revision_id=revision_id,
            cursor=cursor,
        )

    def record_task_state(
        self,
        *,
        run_id: str,
        status: TaskStatus | str,
        revision_id: str,
        source_event_id: str,
    ) -> TaskStateEvent | None:
        self._require_durable_event(source_event_id)
        checked = status if isinstance(status, TaskStatus) else TaskStatus(status)
        with self.database.transaction() as conn:
            return self._record_task_state_conn(
                conn,
                run_id=run_id,
                status=checked,
                revision_id=revision_id,
                source_event_id=source_event_id,
            )

    def reopen_repository_stream_task(
        self,
        *,
        run_id: str,
        revision_id: str,
        source_event_id: str,
    ) -> tuple[MilestoneStateEvent | None, TaskStateEvent | None]:
        """Resume one failed invocation of a still-live repository stream.

        ``FAILED`` is terminal for an ordinary Task.  SWE-Milestone is a
        single externally released repository Task, however, and older builds
        could incorrectly terminal-fail it when one invocation exhausted its
        physical run budget.  An explicit, durable resume directive is the
        only authority accepted by this narrow recovery path.  The historical
        failure rows remain append-only; new state rows reopen the current
        route without claiming that the failed milestone passed.
        """

        self._require_durable_event(source_event_id)
        with self.database.transaction() as conn:
            task_row = conn.execute(
                "SELECT status FROM v2_task_state_events WHERE run_id=? "
                "ORDER BY created_cursor DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if task_row is None:
                raise KeyError(f"Run has no Task state: {run_id}")
            if TaskStatus(str(task_row["status"])) is not TaskStatus.FAILED:
                raise ValueError("repository stream reopen requires a failed Task")
            current = self.current(run_id, conn=conn)
            milestone_event = None
            if MilestoneStatus(current.status) is MilestoneStatus.FAILED:
                milestone_event = self._record_milestone_state_conn(
                    conn,
                    run_id=run_id,
                    identity_id=current.identity_id,
                    status=MilestoneStatus.IN_PROGRESS,
                    revision_id=revision_id,
                    source_event_id=source_event_id,
                    allow_repository_stream_reopen=True,
                )
            task_event = self._record_task_state_conn(
                conn,
                run_id=run_id,
                status=TaskStatus.EXECUTING,
                revision_id=revision_id,
                source_event_id=source_event_id,
                allow_repository_stream_reopen=True,
            )
            return milestone_event, task_event

    def task_status(self, run_id: str) -> TaskStatus:
        row = self.database.connection.execute(
            "SELECT status FROM v2_task_state_events WHERE run_id=? "
            "ORDER BY created_cursor DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Run has no Task state: {run_id}")
        return TaskStatus(str(row["status"]))

    def _record_milestone_state_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        identity_id: str,
        status: MilestoneStatus,
        revision_id: str,
        source_event_id: str,
        allow_repository_stream_reopen: bool = False,
    ) -> MilestoneStateEvent | None:
        identity = conn.execute(
            "SELECT identity_id FROM v2_milestone_identities WHERE run_id=? AND identity_id=?",
            (run_id, identity_id),
        ).fetchone()
        if identity is None:
            raise KeyError(f"unknown Milestone identity: {identity_id}")
        prior = conn.execute(
            "SELECT status FROM v2_milestone_state_events WHERE identity_id=? "
            "ORDER BY created_cursor DESC LIMIT 1",
            (identity_id,),
        ).fetchone()
        previous = MilestoneStatus(str(prior["status"])) if prior is not None else None
        if previous == status:
            return None
        if (
            previous is MilestoneStatus.COMPLETED_VERIFIED
            and status is MilestoneStatus.COMPLETED_CLAIMED
        ):
            # Codex continues to report its native terminal `completed`
            # status after our stronger criterion-bound verification. That is
            # an idempotent observation, not a state regression.
            return None
        if (
            previous is MilestoneStatus.VERIFICATION_FAILED
            and status is MilestoneStatus.COMPLETED_CLAIMED
        ):
            # A Provider may keep reporting its stale native `completed`
            # marker after the stronger Criterion fold rejected it.  Only a
            # durable Milestone failure review may create corrective navigation;
            # a repeated completion claim cannot overwrite the failed authority state.
            return None
        repository_stream_reopen = (
            allow_repository_stream_reopen
            and previous is MilestoneStatus.FAILED
            and status is MilestoneStatus.IN_PROGRESS
        )
        if (
            previous is not None
            and status not in _MILESTONE_TRANSITIONS[previous]
            and not repository_stream_reopen
        ):
            raise ValueError(
                f"illegal Milestone state transition: {previous.value} -> {status.value}"
            )
        cursor = self._next_cursor(conn, run_id)
        state_event_id = stable_id(
            "milstate_",
            {
                "identity": identity_id,
                "status": status.value,
                "source": source_event_id,
                "cursor": cursor,
            },
        )
        conn.execute(
            "INSERT INTO v2_milestone_state_events VALUES(?,?,?,?,?,?,?,?,?)",
            (
                state_event_id,
                run_id,
                identity_id,
                previous.value if previous is not None else None,
                status.value,
                source_event_id,
                revision_id,
                cursor,
                utc_now(),
            ),
        )
        return MilestoneStateEvent(
            state_event_id=state_event_id,
            run_id=run_id,
            identity_id=identity_id,
            previous_status=previous,
            status=status,
            source_event_id=source_event_id,
            revision_id=revision_id,
            cursor=cursor,
        )

    def record_milestone_state(
        self,
        *,
        run_id: str,
        canonical_id: str,
        status: MilestoneStatus | str,
        revision_id: str,
        source_event_id: str,
    ) -> MilestoneStateEvent | None:
        self._require_durable_event(source_event_id)
        checked = self._coerce_milestone_status(status)
        if checked is not MilestoneStatus.IN_PROGRESS:
            raise ValueError(
                "only Milestone activation may be recorded directly; terminal history is "
                "owned by the acceptance/review kernel"
            )
        with self.database.transaction() as conn:
            identity = conn.execute(
                "SELECT identity_id FROM v2_milestone_identities WHERE run_id=? AND canonical_id=?",
                (run_id, canonical_id),
            ).fetchone()
            if identity is None:
                raise KeyError(f"unknown Milestone: {canonical_id}")
            return self._record_milestone_state_conn(
                conn,
                run_id=run_id,
                identity_id=str(identity["identity_id"]),
                status=checked,
                revision_id=revision_id,
                source_event_id=source_event_id,
            )

    def commit_milestone_acceptance(
        self,
        *,
        run_id: str,
        canonical_id: str,
        verdict: MilestoneStatus,
        revision_id: str,
        source_event_id: str,
        satisfied_criterion_ids: Sequence[str],
        failed_criterion_ids: Sequence[str],
        evidence_event_ids: Sequence[str],
        failure_signature: str | None = None,
        unverified_criterion_ids: Sequence[str] = (),
    ) -> str:
        """Commit Criterion verdict, evidence provenance and route state together.

        ``unverified_criterion_ids`` names semantic requirements whose bounded
        review budget was exhausted.  They are reported, never silently marked
        satisfied, and never allowed to hold the route open indefinitely.
        """

        if verdict not in {
            MilestoneStatus.COMPLETED_VERIFIED,
            MilestoneStatus.VERIFICATION_FAILED,
        }:
            raise ValueError("Milestone acceptance supports only terminal verifier verdicts")
        satisfied = tuple(dict.fromkeys(map(str, satisfied_criterion_ids)))
        failed = tuple(dict.fromkeys(map(str, failed_criterion_ids)))
        unverified = tuple(dict.fromkeys(map(str, unverified_criterion_ids)))
        if set(satisfied).intersection(failed):
            raise ValueError("a Criterion cannot be both satisfied and failed")
        if set(unverified).intersection(satisfied) or set(unverified).intersection(failed):
            raise ValueError("an unverified Criterion cannot also be satisfied or failed")
        if verdict is MilestoneStatus.VERIFICATION_FAILED and (
            not failed or not str(failure_signature or "").strip()
        ):
            raise ValueError("failed Milestone acceptance requires Criteria and a signature")
        if verdict is MilestoneStatus.COMPLETED_VERIFIED and (failed or failure_signature):
            raise ValueError("successful Milestone acceptance cannot carry failed Criteria")
        self._require_durable_event(source_event_id)
        with self.database.transaction() as conn:
            identity = conn.execute(
                "SELECT identity_id FROM v2_milestone_identities WHERE run_id=? AND canonical_id=?",
                (run_id, canonical_id),
            ).fetchone()
            if identity is None:
                raise KeyError(canonical_id)
            receipt_id = stable_id(
                "milaccept_",
                {
                    "milestone": identity["identity_id"],
                    "revision": revision_id,
                    "verdict": verdict.value,
                    "source": source_event_id,
                },
            )
            existing = conn.execute(
                "SELECT receipt_id FROM v2_milestone_acceptance_receipts WHERE receipt_id=?",
                (receipt_id,),
            ).fetchone()
            if existing is not None:
                return str(existing["receipt_id"])
            cursor = self._next_cursor(conn, run_id)
            conn.execute(
                "INSERT INTO v2_milestone_acceptance_receipts("
                "receipt_id,run_id,milestone_identity_id,revision_id,verdict,"
                "satisfied_criterion_ids_json,failed_criterion_ids_json,"
                "evidence_event_ids_json,failure_signature,source_event_id,"
                "created_cursor,created_at,unverified_criterion_ids_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    receipt_id,
                    run_id,
                    identity["identity_id"],
                    revision_id,
                    verdict.value,
                    json.dumps(satisfied),
                    json.dumps(failed),
                    json.dumps(tuple(dict.fromkeys(map(str, evidence_event_ids)))),
                    failure_signature,
                    source_event_id,
                    cursor,
                    utc_now(),
                    json.dumps(unverified),
                ),
            )
            self._record_milestone_state_conn(
                conn,
                run_id=run_id,
                identity_id=str(identity["identity_id"]),
                status=verdict,
                revision_id=revision_id,
                source_event_id=source_event_id,
            )
            if verdict is MilestoneStatus.COMPLETED_VERIFIED:
                # Acceptance closes the whole navigation map beneath the
                # Milestone, including Focus nodes opened after the claim.  A
                # Step carries no acceptance meaning, so an open node would
                # only leave a dangling route pointer behind a verified stage.
                open_steps = conn.execute(
                    "SELECT si.step_identity_id,"
                    "COALESCE((SELECT state.status FROM v2_plan_step_state_events state "
                    " WHERE state.step_identity_id=si.step_identity_id "
                    " ORDER BY state.created_cursor DESC LIMIT 1),'PENDING') status "
                    "FROM v2_plan_step_identities si "
                    "WHERE si.run_id=? AND si.milestone_identity_id=? "
                    "AND COALESCE((SELECT state.status FROM v2_plan_step_state_events state "
                    " WHERE state.step_identity_id=si.step_identity_id "
                    " ORDER BY state.created_cursor DESC LIMIT 1),'PENDING') "
                    "IN ('PENDING','IN_PROGRESS','COMPLETED_CLAIMED') "
                    "ORDER BY si.created_cursor",
                    (run_id, str(identity["identity_id"])),
                ).fetchall()
                for row in open_steps:
                    if str(row["status"]) == PlanStepStatus.PENDING.value:
                        self._record_step_state_conn(
                            conn,
                            run_id=run_id,
                            step_identity_id=str(row["step_identity_id"]),
                            status=PlanStepStatus.IN_PROGRESS,
                            revision_id=revision_id,
                            source_event_id=source_event_id,
                        )
                    self._record_step_state_conn(
                        conn,
                        run_id=run_id,
                        step_identity_id=str(row["step_identity_id"]),
                        status=PlanStepStatus.COMPLETED_OBSERVED,
                        revision_id=revision_id,
                        source_event_id=source_event_id,
                    )
                if open_steps:
                    self._project_current_step_conn(
                        conn,
                        run_id=run_id,
                        revision_id=revision_id,
                        source_event_id=source_event_id,
                        cursor=self._next_cursor(conn, run_id),
                    )
            return receipt_id

    def observe_route_frontier(
        self,
        *,
        run_id: str,
        frontier_digest: str,
        source_event_id: str,
        revision_id: str,
    ) -> bool:
        """Record one deterministic execution frontier.

        A frontier is a route observation, not an acceptance receipt.  The
        first observation may continue the same Focus; observing the same
        frontier again is the durable signal that no legal progress was
        produced and the attempt must terminate instead of looping.
        """

        if not frontier_digest.strip():
            raise ValueError("route frontier digest cannot be empty")
        self._require_durable_event(source_event_id)
        with self.database.transaction() as conn:
            current = self.current(run_id, conn=conn)
            existing = conn.execute(
                "SELECT 1 FROM v2_route_frontier_observations "
                "WHERE run_id=? AND frontier_digest=? LIMIT 1",
                (run_id, frontier_digest),
            ).fetchone()
            if existing is not None:
                return False
            cursor = self._next_cursor(conn, run_id)
            conn.execute(
                "INSERT INTO v2_route_frontier_observations VALUES(?,?,?,?,?,?,?,?)",
                (
                    stable_id(
                        "routefrontier_",
                        {"run": run_id, "frontier": frontier_digest},
                    ),
                    run_id,
                    current.identity_id,
                    frontier_digest,
                    source_event_id,
                    revision_id,
                    cursor,
                    utc_now(),
                ),
            )
            return True

    def materialize_acceptance_focus(
        self,
        *,
        run_id: str,
        revision_id: str,
        source_event_id: str,
        criterion_id: str,
        frontier_digest: str,
        reason: str,
        focus_kind: str = "VERIFICATION",
        failure_signature: str | None = None,
    ) -> tuple[str, bool]:
        """Create one bounded Verification/Corrective Focus idempotently.

        Focus is the only dynamic Step creation permitted after planning.  It
        carries no Step acceptance contract; the enclosing Milestone remains
        the sole correctness authority.  The frontier key makes retries after
        a crash converge to the same navigation node.
        """

        kind = str(focus_kind).strip().upper()
        if kind not in {"VERIFICATION", "CORRECTIVE"}:
            raise ValueError("focus_kind must be VERIFICATION or CORRECTIVE")
        if not all(
            value.strip()
            for value in (
                run_id,
                revision_id,
                source_event_id,
                criterion_id,
                frontier_digest,
                reason,
            )
        ):
            raise ValueError("acceptance Focus requires non-empty identity fields")
        self._require_durable_event(source_event_id)
        with self.database.transaction() as conn:
            current = self.current(run_id, conn=conn)
            expected_status = (
                MilestoneStatus.VERIFICATION_FAILED
                if kind == "CORRECTIVE"
                else MilestoneStatus.COMPLETED_CLAIMED
            )
            if MilestoneStatus(current.status) is not expected_status:
                raise ValueError(
                    f"{kind} Focus requires Milestone {expected_status.value}, got {current.status}"
                )
            existing = conn.execute(
                "SELECT si.canonical_step_id FROM v2_milestone_focus_events mf "
                "JOIN v2_plan_step_identities si ON si.step_identity_id=mf.step_identity_id "
                "WHERE mf.run_id=? AND mf.milestone_identity_id=? "
                "AND mf.focus_kind=? AND mf.frontier_digest=?",
                (run_id, current.identity_id, kind, frontier_digest),
            ).fetchone()
            if existing is not None:
                return str(existing["canonical_step_id"]), False
            criterion = conn.execute(
                "SELECT entity_refs_json,observable_outcome FROM v2_completion_criteria "
                "WHERE run_id=? AND milestone_identity_id=? AND local_criterion_id=? "
                "AND required=1 ORDER BY created_cursor DESC LIMIT 1",
                (run_id, current.identity_id, criterion_id),
            ).fetchone()
            if criterion is None:
                raise KeyError(f"unknown required Criterion: {criterion_id}")
            step_id = self._next_step_id_conn(
                conn,
                run_id,
                current.canonical_id,
                corrective=kind == "CORRECTIVE",
            )
            title = (
                f"Correct Milestone acceptance: {criterion_id}"
                if kind == "CORRECTIVE"
                else f"Verify Milestone acceptance: {criterion_id}"
            )
            outcome = str(criterion["observable_outcome"]) or reason
            step = PlanStepSpec(
                step_id=step_id,
                title=title,
                corrective=kind == "CORRECTIVE",
                criterion_ids=(criterion_id,),
                entity_refs=tuple(json.loads(str(criterion["entity_refs_json"]))),
                expected_outcome=outcome,
                minimum_acceptance=(),
                failure_signals=(reason,),
            )
            step_identity_id = self._ensure_step_conn(
                conn,
                run_id=run_id,
                milestone_identity_id=current.identity_id,
                step=step,
                revision_id=revision_id,
                source_event_id=source_event_id,
            )
            cursor = self._next_cursor(conn, run_id)
            focus_id = stable_id(
                "focus_",
                {
                    "run": run_id,
                    "milestone": current.identity_id,
                    "kind": kind,
                    "frontier": frontier_digest,
                },
            )
            conn.execute(
                "INSERT INTO v2_milestone_focus_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    focus_id,
                    run_id,
                    current.identity_id,
                    kind,
                    criterion_id,
                    step_identity_id,
                    frontier_digest,
                    failure_signature,
                    source_event_id,
                    revision_id,
                    cursor,
                    utc_now(),
                ),
            )
            if kind == "CORRECTIVE":
                self._record_milestone_state_conn(
                    conn,
                    run_id=run_id,
                    identity_id=current.identity_id,
                    status=MilestoneStatus.REPAIRING,
                    revision_id=revision_id,
                    source_event_id=source_event_id,
                )
            repository_id, branch_id = self._projection_scope_conn(conn, run_id)
            self._projection().project_plan_step(
                conn,
                repository_id=repository_id,
                run_id=run_id,
                branch_id=branch_id,
                revision_id=revision_id,
                plan_version_id=current.plan_version_id,
                milestone_identity_id=current.identity_id,
                step_identity_id=step_identity_id,
                step=step,
                source_event_id=source_event_id,
                cursor=self._next_cursor(conn, run_id),
            )
            if kind == "CORRECTIVE":
                self._record_runtime_correction_contract_conn(
                    conn,
                    run_id=run_id,
                    current=current,
                    step_identity_id=step_identity_id,
                    criterion_id=criterion_id,
                    reason=reason,
                    revision_id=revision_id,
                    source_event_id=source_event_id,
                    failure_signature=failure_signature,
                    repository_id=repository_id,
                    branch_id=branch_id,
                )
            self._project_current_step_conn(
                conn,
                run_id=run_id,
                revision_id=revision_id,
                source_event_id=source_event_id,
                cursor=self._next_cursor(conn, run_id),
            )
            return step_id, True

    def _record_runtime_correction_contract_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        current: CurrentMilestone,
        step_identity_id: str,
        criterion_id: str,
        reason: str,
        revision_id: str,
        source_event_id: str,
        failure_signature: str | None,
        repository_id: str,
        branch_id: str,
    ) -> None:
        """Persist the causal contract of a runtime-owned corrective Focus.

        The authoritative diagnosis is the Milestone acceptance receipt, not a
        model review.  The corrective Focus inherits the receipt's failed
        Criterion set, Evidence and baseline revision so that the later
        successful post-baseline verification materializes the same
        ``FAILED -> CORRECTED_BY -> SUCCESS`` Page lineage a model-authored
        failure review would have produced.
        """

        receipt = conn.execute(
            "SELECT failed_criterion_ids_json,evidence_event_ids_json,failure_signature "
            "FROM v2_milestone_acceptance_receipts "
            "WHERE run_id=? AND milestone_identity_id=? AND verdict='VERIFICATION_FAILED' "
            "ORDER BY created_cursor DESC LIMIT 1",
            (run_id, current.identity_id),
        ).fetchone()
        failed_criterion_ids: tuple[str, ...] = (criterion_id,)
        evidence_event_ids: tuple[str, ...] = ()
        signature = str(failure_signature or "").strip() or None
        if receipt is not None:
            receipt_failed = tuple(
                dict.fromkeys(
                    value
                    for item in json.loads(str(receipt["failed_criterion_ids_json"]))
                    if (value := str(item).strip())
                )
            )
            if receipt_failed:
                failed_criterion_ids = receipt_failed
            evidence_event_ids = tuple(
                map(str, json.loads(str(receipt["evidence_event_ids_json"])))
            )
            if signature is None and receipt["failure_signature"]:
                signature = str(receipt["failure_signature"])
        criterion_row = conn.execute(
            "SELECT criterion_identity_id FROM v2_completion_criteria "
            "WHERE run_id=? AND milestone_identity_id=? AND local_criterion_id=? "
            "ORDER BY created_cursor DESC LIMIT 1",
            (run_id, current.identity_id, criterion_id),
        ).fetchone()
        baseline_cursor = int(
            conn.execute(
                "SELECT next_cursor FROM v2_registry_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()["next_cursor"]
        )
        correction_cursor = self._next_cursor(conn, run_id)
        conn.execute(
            "INSERT OR IGNORE INTO v2_corrective_steps("
            "corrective_id,run_id,milestone_identity_id,step_identity_id,"
            "criterion_identity_id,reason,evidence_event_ids_json,source_event_id,"
            "revision_id,failure_signature,failure_criterion_ids_json,"
            "baseline_revision_id,baseline_cursor,created_cursor,created_at"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                stable_id("corrective_", {"step": step_identity_id, "source": source_event_id}),
                run_id,
                current.identity_id,
                step_identity_id,
                str(criterion_row["criterion_identity_id"]) if criterion_row else None,
                reason.strip(),
                json.dumps(evidence_event_ids),
                source_event_id,
                revision_id,
                signature,
                json.dumps(failed_criterion_ids),
                revision_id,
                baseline_cursor,
                correction_cursor,
                utc_now(),
            ),
        )
        failed_set = frozenset(failed_criterion_ids)
        causal_rows = conn.execute(
            "SELECT step_identity_id,criterion_ids_json FROM v2_plan_step_identities "
            "WHERE run_id=? AND milestone_identity_id=? AND step_identity_id<>? "
            "ORDER BY created_cursor DESC",
            (run_id, current.identity_id, step_identity_id),
        ).fetchall()
        causal_step_identity_id = next(
            (
                str(row["step_identity_id"])
                for row in causal_rows
                if failed_set.intersection(map(str, json.loads(str(row["criterion_ids_json"]))))
            ),
            None,
        )
        if causal_step_identity_id is None:
            return
        self._projection().project_step_correction(
            conn,
            repository_id=repository_id,
            run_id=run_id,
            branch_id=branch_id,
            revision_id=revision_id,
            plan_version_id=current.plan_version_id,
            failed_step_identity_id=causal_step_identity_id,
            corrective_step_identity_id=step_identity_id,
            source_event_id=source_event_id,
            cursor=correction_cursor,
        )

    def record_route_stall(
        self,
        *,
        run_id: str,
        revision_id: str,
        source_event_id: str,
        frontier_digest: str,
        reason: str,
        terminal: bool = True,
    ) -> str:
        """Persist a route stall and optionally fail the Task atomically.

        Ordinary finite Tasks keep the historical terminal behavior.  A
        repository stream may instead end only the current invocation while
        preserving the same Task, Page Store and TPG for a later official
        release or a fresh physical budget.
        """

        if not all(
            value.strip()
            for value in (run_id, revision_id, source_event_id, frontier_digest, reason)
        ):
            raise ValueError("route stall requires non-empty identity fields")
        self._require_durable_event(source_event_id)
        with self.database.transaction() as conn:
            existing = conn.execute(
                "SELECT stall_id FROM v2_route_stall_events WHERE run_id=? AND frontier_digest=?",
                (run_id, frontier_digest),
            ).fetchone()
            if existing is not None:
                return str(existing["stall_id"])
            current = self.current(run_id, conn=conn)
            cursor = self._next_cursor(conn, run_id)
            stall_id = stable_id("routestall_", {"run": run_id, "frontier": frontier_digest})
            conn.execute(
                "INSERT INTO v2_route_stall_events VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    stall_id,
                    run_id,
                    current.identity_id,
                    frontier_digest,
                    reason,
                    source_event_id,
                    revision_id,
                    cursor,
                    utc_now(),
                ),
            )
            if terminal:
                if MilestoneStatus(current.status) not in {
                    MilestoneStatus.COMPLETED_VERIFIED,
                    MilestoneStatus.FAILED,
                    MilestoneStatus.CANCELLED,
                }:
                    self._record_milestone_state_conn(
                        conn,
                        run_id=run_id,
                        identity_id=current.identity_id,
                        status=MilestoneStatus.FAILED,
                        revision_id=revision_id,
                        source_event_id=source_event_id,
                    )
                task_row = conn.execute(
                    "SELECT status FROM v2_task_state_events WHERE run_id=? "
                    "ORDER BY created_cursor DESC LIMIT 1",
                    (run_id,),
                ).fetchone()
                if task_row is not None and TaskStatus(str(task_row["status"])) not in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                }:
                    self._record_task_state_conn(
                        conn,
                        run_id=run_id,
                        status=TaskStatus.FAILED,
                        revision_id=revision_id,
                        source_event_id=source_event_id,
                    )
            return stall_id

    def freeze_milestone_contract(
        self,
        *,
        run_id: str,
        revision_id: str,
        source_event_id: str,
        canonical_id: str,
        frozen_plan: PlanSpec,
        trigger: str,
        address_resolution: Mapping[str, object],
    ) -> ContractFreezeReceipt:
        """Version one not-yet-started Milestone's executable contract.

        This is the runtime-owned counterpart of the model's future-plan
        review: when a Milestone becomes the route target its semantic
        directions are compiled into an executable acceptance contract with
        resolved addresses.  Started Milestones keep their history; only the
        target Milestone's version changes, inside one PlanVersion transaction.
        """

        if not all(value.strip() for value in (run_id, revision_id, source_event_id, canonical_id)):
            raise ValueError("contract freeze requires non-empty identity fields")
        if not str(trigger).strip():
            raise ValueError("contract freeze requires a trigger")
        self._require_durable_event(source_event_id)
        checked = self._coerce_plan(frozen_plan)
        projection = self._projection()
        with self.database.transaction() as conn:
            current = self.current(run_id, conn=conn)
            identity = conn.execute(
                "SELECT identity_id FROM v2_milestone_identities WHERE run_id=? AND canonical_id=?",
                (run_id, canonical_id),
            ).fetchone()
            if identity is None:
                raise KeyError(canonical_id)
            identity_id = str(identity["identity_id"])
            status = self._latest_milestone_status_conn(conn, identity_id)
            if status not in {MilestoneStatus.PENDING, MilestoneStatus.IN_PROGRESS, None}:
                raise ValueError(
                    f"contract freeze targets a Milestone that already reached {status}"
                )
            current_plan = self._current_plan_conn(conn, run_id)
            if [item.canonical_id for item in checked.milestones] != [
                item.canonical_id for item in current_plan.milestones
            ]:
                raise ValueError("contract freeze cannot add, remove or reorder Milestones")
            for proposed, existing in zip(checked.milestones, current_plan.milestones):
                if proposed.canonical_id == canonical_id:
                    continue
                if digest(self._milestone_structure(proposed)) != digest(
                    self._milestone_structure(existing)
                ):
                    raise ValueError(
                        "contract freeze may rewrite only the activated Milestone contract"
                    )
            previous_row = conn.execute(
                "SELECT milestone_version_id FROM v2_plan_milestones "
                "WHERE plan_version_id=? AND identity_id=?",
                (current.plan_version_id, identity_id),
            ).fetchone()
            if previous_row is None:
                raise RuntimeError("frozen Milestone is missing from the current PlanVersion")
            previous_plan_version_id = current.plan_version_id
            application = self._apply_plan_conn(
                conn,
                projection,
                run_id=run_id,
                revision_id=revision_id,
                plan=checked,
                source_event_id=source_event_id,
                requested_current=current.canonical_id,
            )
            self._link_task_requirements_conn(
                conn,
                run_id=run_id,
                plan_version_id=application.plan_version_id,
                source_event_id=source_event_id,
            )
            resulting_row = conn.execute(
                "SELECT milestone_version_id FROM v2_plan_milestones "
                "WHERE plan_version_id=? AND identity_id=?",
                (application.plan_version_id, identity_id),
            ).fetchone()
            if resulting_row is None:
                raise RuntimeError("frozen Milestone is missing from the resulting PlanVersion")
            freeze_id = stable_id(
                "freeze_",
                {
                    "run": run_id,
                    "milestone": identity_id,
                    "plan": application.plan_version_id,
                },
            )
            existing = conn.execute(
                "SELECT freeze_id FROM v2_milestone_contract_freezes WHERE freeze_id=?",
                (freeze_id,),
            ).fetchone()
            created = existing is None
            if created:
                conn.execute(
                    "INSERT INTO v2_milestone_contract_freezes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        freeze_id,
                        run_id,
                        identity_id,
                        previous_plan_version_id,
                        application.plan_version_id,
                        str(previous_row["milestone_version_id"]),
                        str(resulting_row["milestone_version_id"]),
                        str(trigger),
                        json.dumps(
                            primitive(address_resolution),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        source_event_id,
                        revision_id,
                        self._next_cursor(conn, run_id),
                        utc_now(),
                    ),
                )
            return ContractFreezeReceipt(
                freeze_id=freeze_id,
                milestone_identity_id=identity_id,
                canonical_id=canonical_id,
                previous_plan_version_id=previous_plan_version_id,
                resulting_plan_version_id=application.plan_version_id,
                previous_milestone_version_id=str(previous_row["milestone_version_id"]),
                resulting_milestone_version_id=str(resulting_row["milestone_version_id"]),
                address_resolution=dict(primitive(address_resolution)),
                created=created,
            )

    def contract_freezes(self, run_id: str) -> tuple[dict[str, object], ...]:
        rows = self.database.connection.execute(
            "SELECT f.*,mi.canonical_id FROM v2_milestone_contract_freezes f "
            "JOIN v2_milestone_identities mi ON mi.identity_id=f.milestone_identity_id "
            "WHERE f.run_id=? ORDER BY f.created_cursor",
            (run_id,),
        ).fetchall()
        return tuple(
            {
                "freeze_id": str(row["freeze_id"]),
                "milestone_id": str(row["canonical_id"]),
                "trigger": str(row["trigger"]),
                "previous_plan_version_id": str(row["previous_plan_version_id"]),
                "resulting_plan_version_id": str(row["resulting_plan_version_id"]),
                "address_resolution": json.loads(str(row["address_resolution_json"])),
                "revision_id": str(row["revision_id"]),
            }
            for row in rows
        )

    def observe_acceptance_progress(
        self,
        *,
        run_id: str,
        revision_id: str,
        source_event_id: str,
        boundary_kind: str,
        gap_digest: str,
        bound_evidence_digest: str,
        scoped_revision_digest: str,
        unmet_criterion_ids: Sequence[str],
    ) -> AcceptanceProgress:
        """Classify one acceptance boundary against the previous one.

        Progress is measured on the acceptance gap, the evidence bound to the
        contract and the Milestone-scoped repository state; a mere additional
        Turn, a repeated command or a duplicated observation never counts.
        """

        if not all(
            value.strip()
            for value in (
                run_id,
                revision_id,
                source_event_id,
                boundary_kind,
                gap_digest,
                bound_evidence_digest,
                scoped_revision_digest,
            )
        ):
            raise ValueError("acceptance progress requires non-empty identity fields")
        self._require_durable_event(source_event_id)
        unmet = tuple(dict.fromkeys(map(str, unmet_criterion_ids)))
        with self.database.transaction() as conn:
            current = self.current(run_id, conn=conn)
            existing = conn.execute(
                "SELECT * FROM v2_acceptance_progress_observations "
                "WHERE run_id=? AND milestone_identity_id=? AND source_event_id=? "
                "AND boundary_kind=?",
                (run_id, current.identity_id, source_event_id, boundary_kind),
            ).fetchone()
            if existing is not None:
                count = conn.execute(
                    "SELECT COUNT(*) FROM v2_acceptance_progress_observations "
                    "WHERE run_id=? AND milestone_identity_id=? AND plan_version_id=? "
                    "AND created_cursor<=?",
                    (
                        run_id,
                        current.identity_id,
                        str(existing["plan_version_id"]),
                        int(existing["created_cursor"]),
                    ),
                ).fetchone()[0]
                return AcceptanceProgress(
                    observation_id=str(existing["observation_id"]),
                    milestone_identity_id=current.identity_id,
                    gap_digest=str(existing["gap_digest"]),
                    bound_evidence_digest=str(existing["bound_evidence_digest"]),
                    scoped_revision_digest=str(existing["scoped_revision_digest"]),
                    progress_class=AcceptanceProgressClass(str(existing["progress_class"])),
                    weak_streak=int(existing["weak_streak"]),
                    none_streak=int(existing["none_streak"]),
                    boundary_count=int(count),
                    unmet_criterion_ids=tuple(
                        json.loads(str(existing["unmet_criterion_ids_json"]))
                    ),
                    created=False,
                )
            previous = conn.execute(
                "SELECT * FROM v2_acceptance_progress_observations "
                "WHERE run_id=? AND milestone_identity_id=? AND plan_version_id=? "
                "ORDER BY created_cursor DESC LIMIT 1",
                (run_id, current.identity_id, current.plan_version_id),
            ).fetchone()
            leaves_exploration = (
                previous is not None
                and str(previous["boundary_kind"]) == "EXECUTION"
                and boundary_kind != "EXECUTION"
            )
            if previous is None or leaves_exploration:
                # Exploratory (EXECUTION) boundaries are bounded by the unclaimed
                # budget; the claimed loop that follows (ACCEPTANCE/CORRECTION)
                # opens a fresh streak because the model only now receives the
                # exact acceptance diagnostics and its Focus.  Within the
                # claimed loop the streak is shared across kinds so alternating
                # "no evidence" and "failed evidence" boundaries cannot reset
                # each other forever.
                progress_class = AcceptanceProgressClass.INITIAL
                weak_streak = 0
                none_streak = 0
                boundary_count = (
                    1
                    if previous is None
                    else int(
                        conn.execute(
                            "SELECT COUNT(*) FROM v2_acceptance_progress_observations "
                            "WHERE run_id=? AND milestone_identity_id=? AND plan_version_id=?",
                            (run_id, current.identity_id, current.plan_version_id),
                        ).fetchone()[0]
                    )
                    + 1
                )
            else:
                previous_unmet = set(json.loads(str(previous["unmet_criterion_ids_json"])))
                current_unmet = set(unmet)
                gap_shrank = current_unmet < previous_unmet
                changed = (
                    str(previous["gap_digest"]) != gap_digest
                    or str(previous["bound_evidence_digest"]) != bound_evidence_digest
                    or str(previous["scoped_revision_digest"]) != scoped_revision_digest
                )
                if gap_shrank:
                    progress_class = AcceptanceProgressClass.STRONG
                elif changed:
                    progress_class = AcceptanceProgressClass.WEAK
                else:
                    progress_class = AcceptanceProgressClass.NONE
                previous_weak = int(previous["weak_streak"])
                previous_none = int(previous["none_streak"])
                if progress_class is AcceptanceProgressClass.STRONG:
                    weak_streak = 0
                    none_streak = 0
                elif progress_class is AcceptanceProgressClass.WEAK:
                    weak_streak = previous_weak + 1
                    none_streak = 0
                else:
                    weak_streak = previous_weak
                    none_streak = previous_none + 1
                boundary_count = (
                    int(
                        conn.execute(
                            "SELECT COUNT(*) FROM v2_acceptance_progress_observations "
                            "WHERE run_id=? AND milestone_identity_id=? AND plan_version_id=?",
                            (run_id, current.identity_id, current.plan_version_id),
                        ).fetchone()[0]
                    )
                    + 1
                )
            cursor = self._next_cursor(conn, run_id)
            observation_id = stable_id(
                "acceptprogress_",
                {
                    "run": run_id,
                    "milestone": current.identity_id,
                    "source": source_event_id,
                    "kind": boundary_kind,
                },
            )
            conn.execute(
                "INSERT INTO v2_acceptance_progress_observations "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    observation_id,
                    run_id,
                    current.identity_id,
                    current.plan_version_id,
                    boundary_kind,
                    gap_digest,
                    bound_evidence_digest,
                    scoped_revision_digest,
                    json.dumps(unmet),
                    progress_class.value,
                    weak_streak,
                    none_streak,
                    source_event_id,
                    revision_id,
                    cursor,
                    utc_now(),
                ),
            )
            return AcceptanceProgress(
                observation_id=observation_id,
                milestone_identity_id=current.identity_id,
                gap_digest=gap_digest,
                bound_evidence_digest=bound_evidence_digest,
                scoped_revision_digest=scoped_revision_digest,
                progress_class=progress_class,
                weak_streak=weak_streak,
                none_streak=none_streak,
                boundary_count=boundary_count,
                unmet_criterion_ids=unmet,
                created=True,
            )

    def reset_acceptance_progress(
        self,
        *,
        run_id: str,
        milestone_identity_id: str,
        plan_version_id: str,
        source_event_id: str,
        revision_id: str,
        reason: str,
    ) -> str:
        """Open a fresh acceptance streak for a Milestone the route parked.

        A repository stream leaves a stalled official ID for its released
        siblings and returns later.  The exhausted weak/none streaks belong to
        the abandoned attempt; carried over, they would stall the returning
        route after one boundary (dubbo in an earlier run).  The reset is an
        ``INITIAL`` observation with a unique gap digest, so the next real
        boundary is classified against a clean baseline.
        """

        if not all(
            value.strip()
            for value in (run_id, milestone_identity_id, plan_version_id, source_event_id, revision_id, reason)
        ):
            raise ValueError("acceptance progress reset requires non-empty identity fields")
        self._require_durable_event(source_event_id)
        with self.database.transaction() as conn:
            cursor = self._next_cursor(conn, run_id)
            observation_id = stable_id(
                "acceptreset_",
                {
                    "run": run_id,
                    "milestone": milestone_identity_id,
                    "source": source_event_id,
                    "reason": reason,
                },
            )
            conn.execute(
                "INSERT OR IGNORE INTO v2_acceptance_progress_observations "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    observation_id,
                    run_id,
                    milestone_identity_id,
                    plan_version_id,
                    f"ROUTE_PARKED:{reason}",
                    digest({"reset": observation_id}),
                    digest({"reset_evidence": observation_id}),
                    digest({"reset_revision": observation_id}),
                    json.dumps([]),
                    AcceptanceProgressClass.INITIAL.value,
                    0,
                    0,
                    source_event_id,
                    revision_id,
                    cursor,
                    utc_now(),
                ),
            )
        return observation_id

    def latest_acceptance_progress(
        self,
        run_id: str,
        *,
        milestone_identity_id: str | None = None,
    ) -> AcceptanceProgress | None:
        current = self.current(run_id)
        identity_id = milestone_identity_id or current.identity_id
        row = self.database.connection.execute(
            "SELECT * FROM v2_acceptance_progress_observations "
            "WHERE run_id=? AND milestone_identity_id=? AND plan_version_id=? "
            "ORDER BY created_cursor DESC LIMIT 1",
            (run_id, identity_id, current.plan_version_id),
        ).fetchone()
        if row is None:
            return None
        count = self.database.connection.execute(
            "SELECT COUNT(*) FROM v2_acceptance_progress_observations "
            "WHERE run_id=? AND milestone_identity_id=? AND plan_version_id=?",
            (run_id, identity_id, current.plan_version_id),
        ).fetchone()[0]
        return AcceptanceProgress(
            observation_id=str(row["observation_id"]),
            milestone_identity_id=identity_id,
            gap_digest=str(row["gap_digest"]),
            bound_evidence_digest=str(row["bound_evidence_digest"]),
            scoped_revision_digest=str(row["scoped_revision_digest"]),
            progress_class=AcceptanceProgressClass(str(row["progress_class"])),
            weak_streak=int(row["weak_streak"]),
            none_streak=int(row["none_streak"]),
            boundary_count=int(count),
            unmet_criterion_ids=tuple(json.loads(str(row["unmet_criterion_ids_json"]))),
            created=False,
        )

    def record_semantic_review_round(
        self,
        *,
        run_id: str,
        revision_id: str,
        source_event_id: str,
        gap_digest: str,
        criterion_ids: Sequence[str],
    ) -> int:
        """Count one bounded semantic review request for the current Milestone."""

        if not all(value.strip() for value in (run_id, revision_id, source_event_id, gap_digest)):
            raise ValueError("semantic review round requires non-empty identity fields")
        self._require_durable_event(source_event_id)
        with self.database.transaction() as conn:
            current = self.current(run_id, conn=conn)
            existing = conn.execute(
                "SELECT round_number FROM v2_semantic_review_rounds "
                "WHERE run_id=? AND milestone_identity_id=? AND source_event_id=?",
                (run_id, current.identity_id, source_event_id),
            ).fetchone()
            if existing is not None:
                return int(existing["round_number"])
            round_number = (
                int(
                    conn.execute(
                        "SELECT COUNT(*) FROM v2_semantic_review_rounds "
                        "WHERE run_id=? AND milestone_identity_id=?",
                        (run_id, current.identity_id),
                    ).fetchone()[0]
                )
                + 1
            )
            conn.execute(
                "INSERT INTO v2_semantic_review_rounds VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    stable_id(
                        "semreview_",
                        {
                            "run": run_id,
                            "milestone": current.identity_id,
                            "source": source_event_id,
                        },
                    ),
                    run_id,
                    current.identity_id,
                    revision_id,
                    gap_digest,
                    json.dumps(tuple(dict.fromkeys(map(str, criterion_ids)))),
                    round_number,
                    source_event_id,
                    self._next_cursor(conn, run_id),
                    utc_now(),
                ),
            )
            return round_number

    def semantic_review_rounds(self, run_id: str, canonical_id: str | None = None) -> int:
        current = self.current(run_id)
        identity_id = current.identity_id
        if canonical_id is not None and canonical_id != current.canonical_id:
            row = self.database.connection.execute(
                "SELECT identity_id FROM v2_milestone_identities WHERE run_id=? AND canonical_id=?",
                (run_id, canonical_id),
            ).fetchone()
            if row is None:
                raise KeyError(canonical_id)
            identity_id = str(row["identity_id"])
        return int(
            self.database.connection.execute(
                "SELECT COUNT(*) FROM v2_semantic_review_rounds "
                "WHERE run_id=? AND milestone_identity_id=?",
                (run_id, identity_id),
            ).fetchone()[0]
        )

    def milestone_acceptance_receipts(self, run_id: str) -> tuple[dict[str, object], ...]:
        rows = self.database.connection.execute(
            "SELECT r.*,mi.canonical_id FROM v2_milestone_acceptance_receipts r "
            "JOIN v2_milestone_identities mi ON mi.identity_id=r.milestone_identity_id "
            "WHERE r.run_id=? ORDER BY r.created_cursor",
            (run_id,),
        ).fetchall()
        receipts: list[dict[str, object]] = []
        for row in rows:
            keys = row.keys()
            receipts.append(
                {
                    "receipt_id": str(row["receipt_id"]),
                    "milestone_id": str(row["canonical_id"]),
                    "revision_id": str(row["revision_id"]),
                    "verdict": str(row["verdict"]),
                    "satisfied_criterion_ids": tuple(
                        json.loads(str(row["satisfied_criterion_ids_json"]))
                    ),
                    "failed_criterion_ids": tuple(
                        json.loads(str(row["failed_criterion_ids_json"]))
                    ),
                    "unverified_criterion_ids": tuple(
                        json.loads(str(row["unverified_criterion_ids_json"]))
                        if "unverified_criterion_ids_json" in keys
                        else ()
                    ),
                    "evidence_event_ids": tuple(json.loads(str(row["evidence_event_ids_json"]))),
                    "failure_signature": row["failure_signature"],
                }
            )
        return tuple(receipts)

    def milestone_identity(self, run_id: str, canonical_id: str) -> dict[str, object]:
        """Return the identity row of one Milestone on the current PlanVersion."""

        row = self.database.connection.execute(
            "SELECT mi.identity_id,mi.canonical_id,pm.plan_version_id,pm.milestone_version_id "
            "FROM v2_milestone_identities mi "
            "JOIN v2_plan_milestones pm ON pm.identity_id=mi.identity_id "
            "WHERE mi.run_id=? AND mi.canonical_id=? "
            "AND pm.plan_version_id=(SELECT edge.plan_version_id FROM v2_semantic_edges edge "
            "WHERE edge.run_id=? AND edge.edge_type='CURRENT_MILESTONE' "
            "AND edge.valid_to_cursor IS NULL)",
            (run_id, canonical_id, run_id),
        ).fetchone()
        if row is None:
            row = self.database.connection.execute(
                "SELECT identity_id,canonical_id,NULL plan_version_id,NULL milestone_version_id "
                "FROM v2_milestone_identities WHERE run_id=? AND canonical_id=?",
                (run_id, canonical_id),
            ).fetchone()
        if row is None:
            raise KeyError(canonical_id)
        return {
            "identity_id": str(row["identity_id"]),
            "canonical_id": str(row["canonical_id"]),
            "plan_version_id": (
                str(row["plan_version_id"]) if row["plan_version_id"] is not None else None
            ),
            "milestone_version_id": (
                str(row["milestone_version_id"])
                if row["milestone_version_id"] is not None
                else None
            ),
        }

    def milestone_state_window(self, run_id: str, canonical_id: str) -> tuple[int, int]:
        """Return the WAL cursor window of the Milestone's current activity span.

        The span starts at the first state event after the previous terminal
        acceptance state (so a repair after ``VERIFICATION_FAILED`` opens a new
        window) and ends at the latest state event.  PageSet boundaries are
        taken from these cursors, never from byte sizes.
        """

        rows = self.database.connection.execute(
            "SELECT mse.status,mse.created_cursor FROM v2_milestone_state_events mse "
            "JOIN v2_milestone_identities mi ON mi.identity_id=mse.identity_id "
            "WHERE mi.run_id=? AND mi.canonical_id=? ORDER BY mse.created_cursor",
            (run_id, canonical_id),
        ).fetchall()
        if not rows:
            raise KeyError(canonical_id)
        terminal = {
            MilestoneStatus.COMPLETED_VERIFIED.value,
            MilestoneStatus.VERIFICATION_FAILED.value,
        }
        window_start = int(rows[0]["created_cursor"])
        previous_terminal = False
        for row in rows:
            status = str(row["status"])
            if previous_terminal and status not in terminal:
                window_start = int(row["created_cursor"])
            previous_terminal = status in terminal
        return window_start, int(rows[-1]["created_cursor"])

    def unverified_criterion_ids(self, run_id: str) -> dict[str, tuple[str, ...]]:
        """Semantic requirements accepted as UNVERIFIED, per COMPLETED_VERIFIED Milestone.

        Only the latest acceptance receipt of each Milestone counts, and only
        while that Milestone is still COMPLETED_VERIFIED; a reopened Milestone
        must re-earn its acceptance.
        """

        statuses = self.milestone_statuses(run_id)
        latest: dict[str, tuple[str, ...]] = {}
        for receipt in self.milestone_acceptance_receipts(run_id):
            milestone_id = str(receipt["milestone_id"])
            if str(receipt["verdict"]) != MilestoneStatus.COMPLETED_VERIFIED.value:
                latest.pop(milestone_id, None)
                continue
            latest[milestone_id] = tuple(map(str, receipt["unverified_criterion_ids"]))
        return {
            milestone_id: criterion_ids
            for milestone_id, criterion_ids in latest.items()
            if criterion_ids
            and statuses.get(milestone_id) == MilestoneStatus.COMPLETED_VERIFIED.value
        }

    def route_stalls(self, run_id: str) -> tuple[dict[str, object], ...]:
        rows = self.database.connection.execute(
            "SELECT s.*,mi.canonical_id FROM v2_route_stall_events s "
            "JOIN v2_milestone_identities mi ON mi.identity_id=s.milestone_identity_id "
            "WHERE s.run_id=? ORDER BY s.created_cursor",
            (run_id,),
        ).fetchall()
        return tuple(
            {
                "stall_id": str(row["stall_id"]),
                "milestone_id": str(row["canonical_id"]),
                "reason": str(row["reason"]),
                "frontier_digest": str(row["frontier_digest"]),
                "revision_id": str(row["revision_id"]),
            }
            for row in rows
        )

    def register_alias(
        self,
        *,
        run_id: str,
        canonical_id: str,
        alias: str,
        source_event_id: str,
    ) -> None:
        self._require_durable_event(source_event_id)
        normalized = " ".join(alias.casefold().split())
        if not normalized:
            raise ValueError("Milestone alias cannot be empty")
        with self.database.transaction() as conn:
            identity = conn.execute(
                "SELECT identity_id FROM v2_milestone_identities WHERE run_id=? AND canonical_id=?",
                (run_id, canonical_id),
            ).fetchone()
            if identity is None:
                raise KeyError(f"unknown Milestone: {canonical_id}")
            existing = conn.execute(
                "SELECT identity_id FROM v2_milestone_aliases "
                "WHERE run_id=? AND normalized_alias=?",
                (run_id, normalized),
            ).fetchone()
            if existing is not None and existing["identity_id"] != identity["identity_id"]:
                raise ValueError("Milestone alias is ambiguous")
            conn.execute(
                "INSERT OR IGNORE INTO v2_milestone_aliases VALUES(?,?,?,?,?,?)",
                (run_id, normalized, alias, identity["identity_id"], source_event_id, utc_now()),
            )

    @staticmethod
    def _content_fingerprint(spec: MilestoneSpec) -> str:
        return digest(
            {
                "description": spec.description,
                "completion_criteria": list(spec.completion_criteria),
                "verification": list(spec.verification),
                "target_outcome": spec.target_outcome,
                "downstream_assumptions": list(spec.downstream_assumptions),
                "non_goals": list(spec.non_goals),
            }
        )

    def _current_plan_conn(self, conn: sqlite3.Connection, run_id: str) -> PlanSpec:
        row = conn.execute(
            "SELECT pv.plan_json FROM v2_plan_versions pv "
            "JOIN v2_tasks t ON t.task_id=pv.task_id WHERE t.run_id=? "
            "ORDER BY pv.version_number DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Run has no PlanVersion: {run_id}")
        value = json.loads(str(row["plan_json"]))
        states = {
            str(item["canonical_id"]): str(item["status"])
            for item in conn.execute(
                "SELECT mi.canonical_id, mse.status FROM v2_milestone_identities mi "
                "JOIN v2_milestone_state_events mse ON mse.identity_id=mi.identity_id "
                "WHERE mi.run_id=? AND mse.created_cursor=("
                " SELECT MAX(newer.created_cursor) FROM v2_milestone_state_events newer "
                " WHERE newer.identity_id=mi.identity_id)",
                (run_id,),
            ).fetchall()
        }
        for item in value["milestones"]:
            item["status"] = states.get(str(item["canonical_id"]), MilestoneStatus.PENDING.value)
        return PlanSpec.from_dict(value)

    def active_plan(self, run_id: str) -> PlanSpec:
        """Return the current immutable PlanVersion with live execution states."""

        return self._current_plan_conn(self.database.connection, run_id)

    def append_released_work(self, *, run_id: str, revision_id: str,
                             source_event_id: str, title: str, description: str,
                             navigation_text: str = "") -> PlanApplication | None:
        """Extend the route on a durable external requirements release.

        This is not permission to alter completed work or its acceptance facts.
        Official milestones retain their own identities in the source Page;
        the appended node is merely the next navigable unit of actual work.
        """
        self._require_durable_event(source_event_id)
        if not source_event_id.startswith("stream_"):
            raise ValueError("released work requires repository-stream provenance")
        with self.database.transaction() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS v2_repository_release_routes "
                         "(run_id TEXT, source_event_id TEXT, canonical_id TEXT, "
                         "PRIMARY KEY(run_id,source_event_id))")
            if conn.execute("SELECT 1 FROM v2_repository_release_routes WHERE run_id=? "
                            "AND source_event_id=?", (run_id, source_event_id)).fetchone():
                return None
            plan = self._current_plan_conn(conn, run_id)
            if self.task_status(run_id) in {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.COMPLETED}:
                raise ValueError("cannot reopen a terminal Task as a new requirements release")
            canonical = self._next_canonical_id([m.canonical_id for m in plan.milestones], set())
            criterion = CompletionCriterionSpec(
                criterion_id=f"{canonical}.RELEASE", requirement_id=f"{canonical}.RELEASE",
                requirement_text=description, observable_outcome=description,
                claim_type=ClaimType.BEHAVIORAL, required_evidence_types=(),
                verification_mode=CriterionVerificationMode.SEMANTIC,
                commitment_level=CommitmentLevel.MILESTONE,
            )
            node = MilestoneSpec(canonical_id=canonical, title=title,
                description=description + ("\n" + navigation_text if navigation_text else ""),
                completion_criteria=(), criteria=(criterion,),
                verification=("Bounded semantic review of released requirements; official score remains external",),
                depends_on=(), steps=())
            updated = replace(plan, milestones=(*plan.milestones, node))
            application = self._apply_plan_conn(conn, self._projection(), run_id=run_id,
                revision_id=revision_id, plan=updated, source_event_id=source_event_id,
                requested_current=None)
            self._switch_current_conn(conn, self._projection(), run_id=run_id,
                canonical_id=canonical, revision_id=revision_id, source_event_id=source_event_id,
                plan_version_id=application.plan_version_id, require_future_review=False,
                external_requirement_release=True)
            self._record_milestone_state_conn(conn, run_id=run_id,
                identity_id=self.current(run_id, conn=conn).identity_id,
                status=MilestoneStatus.IN_PROGRESS, revision_id=revision_id,
                source_event_id=source_event_id)
            application = replace(application, current_milestone_id=self.current(run_id, conn=conn).identity_id)
            self._link_task_requirements_conn(conn, run_id=run_id,
                plan_version_id=application.plan_version_id, source_event_id=source_event_id)
            if self.task_status(run_id) is TaskStatus.VERIFYING:
                self._record_task_state_conn(conn, run_id=run_id, status=TaskStatus.EXECUTING,
                    revision_id=revision_id, source_event_id=source_event_id)
            conn.execute("INSERT INTO v2_repository_release_routes VALUES(?,?,?)",
                         (run_id, source_event_id, canonical))
            return application

    def append_released_work_nodes(self, *, run_id: str, revision_id: str,
                                   source_event_id: str,
                                   releases: Sequence[Mapping[str, str]],
                                   navigation_text: str = "",
                                   switch_current: bool = True) -> PlanApplication | None:
        """Append one navigable node per released official milestone.

        ``switch_current=False`` records the nodes but leaves the TPG cursor
        where it is: used while the regression guard holds the route on the
        current official ID, so newly released siblings become navigable
        without pulling the model away from the repair.

        SWE-Milestone can release several sibling official IDs in one queue
        observation.  Bundling them into a single internal node destroys the
        one-official-ID-per-Milestone invariant: the TPG cursor loses its
        official anchor, the submit gate cannot name the required tag, and an
        internal review can drift past untagged official work.  Each release
        therefore owns exactly one internal node; the first untagged release
        becomes current and its siblings stay PENDING until the queue (never
        an internal review) advances the route.
        """
        self._require_durable_event(source_event_id)
        if not source_event_id.startswith("stream_"):
            raise ValueError("released work requires repository-stream provenance")
        cleaned = [
            {"official_id": str(item.get("official_id", "")).strip(),
             "title": str(item.get("title", "")).strip(),
             "description": str(item.get("description", "")).strip()}
            for item in releases
        ]
        cleaned = [item for item in cleaned if item["official_id"] and item["title"]]
        if not cleaned:
            return None
        with self.database.transaction() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS v2_repository_release_nodes "
                         "(run_id TEXT, official_id TEXT, canonical_id TEXT, "
                         "source_event_id TEXT, PRIMARY KEY(run_id,official_id))")
            existing = {
                str(row["official_id"])
                for row in conn.execute(
                    "SELECT official_id FROM v2_repository_release_nodes WHERE run_id=?",
                    (run_id,),
                )
            }
            fresh = [item for item in cleaned if item["official_id"] not in existing]
            if not fresh:
                return None
            plan = self._current_plan_conn(conn, run_id)
            if self.task_status(run_id) in {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.COMPLETED}:
                raise ValueError("cannot reopen a terminal Task as a new requirements release")
            taken = [m.canonical_id for m in plan.milestones]
            nodes: list[MilestoneSpec] = []
            first_canonical: str | None = None
            for index, item in enumerate(fresh):
                canonical = self._next_canonical_id(taken, set())
                taken.append(canonical)
                if first_canonical is None:
                    first_canonical = canonical
                description = item["description"] or item["title"]
                if index == 0 and navigation_text:
                    description += "\n" + navigation_text
                criterion = CompletionCriterionSpec(
                    criterion_id=f"{canonical}.RELEASE", requirement_id=f"{canonical}.RELEASE",
                    requirement_text=item["description"] or item["title"],
                    observable_outcome=item["description"] or item["title"],
                    claim_type=ClaimType.BEHAVIORAL, required_evidence_types=(),
                    verification_mode=CriterionVerificationMode.SEMANTIC,
                    commitment_level=CommitmentLevel.MILESTONE,
                )
                nodes.append(MilestoneSpec(
                    canonical_id=canonical, title=item["title"],
                    description=description,
                    completion_criteria=(), criteria=(criterion,),
                    verification=(
                        "Official evaluator owns acceptance; the runtime submit gate "
                        f"requires git tag agent-impl-{item['official_id']}",
                    ),
                    depends_on=(), steps=()))
            updated = replace(plan, milestones=(*plan.milestones, *nodes))
            application = self._apply_plan_conn(conn, self._projection(), run_id=run_id,
                revision_id=revision_id, plan=updated, source_event_id=source_event_id,
                requested_current=None)
            if switch_current:
                self._switch_current_conn(conn, self._projection(), run_id=run_id,
                    canonical_id=first_canonical, revision_id=revision_id,
                    source_event_id=source_event_id,
                    plan_version_id=application.plan_version_id, require_future_review=False,
                    external_requirement_release=True)
                self._record_milestone_state_conn(conn, run_id=run_id,
                    identity_id=self.current(run_id, conn=conn).identity_id,
                    status=MilestoneStatus.IN_PROGRESS, revision_id=revision_id,
                    source_event_id=source_event_id)
            application = replace(application, current_milestone_id=self.current(run_id, conn=conn).identity_id)
            self._link_task_requirements_conn(conn, run_id=run_id,
                plan_version_id=application.plan_version_id, source_event_id=source_event_id)
            if self.task_status(run_id) is TaskStatus.VERIFYING:
                self._record_task_state_conn(conn, run_id=run_id, status=TaskStatus.EXECUTING,
                    revision_id=revision_id, source_event_id=source_event_id)
            conn.executemany(
                "INSERT INTO v2_repository_release_nodes VALUES(?,?,?,?)",
                [(run_id, item["official_id"], node.canonical_id, source_event_id)
                 for item, node in zip(fresh, nodes)])
            return application

    def task_requirements(self, run_id: str) -> tuple[dict[str, object], ...]:
        """Return the immutable checklist extracted from the original Task."""

        rows = self.database.connection.execute(
            "SELECT r.requirement_id,r.ordinal,r.requirement_text,r.requirement_digest,"
            "r.modality,r.category,r.required,r.source_line,s.raw_request_digest,"
            "s.extractor_version FROM v2_task_requirements r "
            "JOIN v2_task_requirement_sets s "
            "ON s.requirement_set_id=r.requirement_set_id "
            "WHERE r.run_id=? ORDER BY r.ordinal",
            (run_id,),
        ).fetchall()
        return tuple(
            {
                "requirement_id": str(row["requirement_id"]),
                "ordinal": int(row["ordinal"]),
                "requirement_text": str(row["requirement_text"]),
                "requirement_digest": str(row["requirement_digest"]),
                "modality": str(row["modality"]),
                "category": str(row["category"]),
                "required": bool(row["required"]),
                "source_line": int(row["source_line"]),
                "raw_request_digest": str(row["raw_request_digest"]),
                "extractor_version": str(row["extractor_version"]),
            }
            for row in rows
        )

    def requirement_coverage(
        self,
        run_id: str,
        *,
        plan_version_id: str | None = None,
    ) -> tuple[dict[str, object], ...]:
        """Resolve immutable Task clauses onto the current TPG route."""

        connection = self.database.connection
        if plan_version_id is None:
            plan_version_id = self.current(run_id).plan_version_id
        links = connection.execute(
            "SELECT l.requirement_id,l.link_basis,l.link_score,"
            "c.local_criterion_id,c.required_evidence_types_json,"
            "c.observable_outcome,mi.canonical_id,"
            "(SELECT mse.status FROM v2_milestone_state_events mse "
            " WHERE mse.identity_id=mi.identity_id "
            " ORDER BY mse.created_cursor DESC LIMIT 1) milestone_status "
            "FROM v2_requirement_plan_links l "
            "JOIN v2_completion_criteria c "
            "ON c.criterion_identity_id=l.criterion_identity_id "
            "JOIN v2_milestone_versions mv "
            "ON mv.version_id=c.milestone_version_id "
            "JOIN v2_milestone_identities mi ON mi.identity_id=mv.identity_id "
            "WHERE l.run_id=? AND l.plan_version_id=? "
            "ORDER BY l.requirement_id,l.link_score DESC,c.local_criterion_id",
            (run_id, plan_version_id),
        ).fetchall()
        by_requirement: dict[str, list[dict[str, object]]] = {}
        for row in links:
            by_requirement.setdefault(str(row["requirement_id"]), []).append(
                {
                    "criterion_id": str(row["local_criterion_id"]),
                    "milestone_id": str(row["canonical_id"]),
                    "milestone_status": str(row["milestone_status"] or "PENDING"),
                    "observable_outcome": str(row["observable_outcome"]),
                    "required_evidence_types": tuple(
                        map(str, json.loads(str(row["required_evidence_types_json"])))
                    ),
                    "link_basis": str(row["link_basis"]),
                    "link_score": float(row["link_score"]),
                }
            )
        coverage: list[dict[str, object]] = []
        for requirement in self.task_requirements(run_id):
            requirement_id = str(requirement["requirement_id"])
            linked = tuple(by_requirement.get(requirement_id, ()))
            verified = bool(linked) and all(
                item["milestone_status"] == MilestoneStatus.COMPLETED_VERIFIED.value
                for item in linked
            )
            verification_capable = any(
                set(item["required_evidence_types"]).intersection(
                    {
                        "TEST_RESULT",
                        "VERIFIER_RESULT",
                        "REQUIREMENT_REVIEW",
                    }
                )
                for item in linked
            )
            state = (
                "OPTIONAL"
                if not bool(requirement["required"])
                else "UNMAPPED"
                if not linked
                else "ROUTE_VERIFIED"
                if verified
                else "ROUTE_PENDING"
            )
            coverage.append(
                {
                    **requirement,
                    "coverage_state": state,
                    "verification_capable": verification_capable,
                    "links": linked,
                }
            )
        return tuple(coverage)

    @staticmethod
    def _next_canonical_id(existing: Sequence[str], reserved: set[str]) -> str:
        numbers = {
            int(match.group(1))
            for value in (*existing, *reserved)
            if (match := re.fullmatch(r"M(\d{3})", value.upper()))
        }
        candidate = 1
        while candidate in numbers:
            candidate += 1
        return f"M{candidate:03d}"

    def observe_plan(
        self,
        *,
        run_id: str,
        revision_id: str,
        plan: PlanSpec | Mapping[str, object],
        source_event_id: str,
        observation_kind: PlanObservationKind | str,
        coverage: PlanCoverage | str,
        current_milestone_id: str | None = None,
    ) -> PlanObservationResult:
        """Apply a native Plan observation without guessing Milestone identity.

        PATCH and UNKNOWN-coverage observations only merge explicitly resolved
        steps. Only a COMPLETE, unambiguous SNAPSHOT may remove absent steps.
        """

        self._require_durable_event(source_event_id)
        observed = self._coerce_plan(plan)
        kind = (
            observation_kind
            if isinstance(observation_kind, PlanObservationKind)
            else PlanObservationKind(observation_kind)
        )
        checked_coverage = (
            coverage if isinstance(coverage, PlanCoverage) else PlanCoverage(coverage)
        )
        with self.database.transaction() as conn:
            current_plan = self._current_plan_conn(conn, run_id)
            existing_by_id = {item.canonical_id: item for item in current_plan.milestones}
            title_map: dict[str, list[str]] = {}
            fingerprint_map: dict[str, list[str]] = {}
            for item in current_plan.milestones:
                title_map.setdefault(" ".join(item.title.casefold().split()), []).append(
                    item.canonical_id
                )
                fingerprint_map.setdefault(self._content_fingerprint(item), []).append(
                    item.canonical_id
                )
            aliases = {
                str(row["normalized_alias"]): str(row["canonical_id"])
                for row in conn.execute(
                    "SELECT ma.normalized_alias, mi.canonical_id FROM v2_milestone_aliases ma "
                    "JOIN v2_milestone_identities mi ON mi.identity_id=ma.identity_id "
                    "WHERE ma.run_id=?",
                    (run_id,),
                ).fetchall()
            }

            resolved: list[tuple[MilestoneSpec, str, str]] = []
            unresolved_specs: list[tuple[MilestoneSpec, str, tuple[str, ...]]] = []
            reserved: set[str] = set()
            for item in observed.milestones:
                supplied = item.canonical_id.strip()
                title = item.title
                prefix = _MILESTONE_PREFIX.match(title)
                candidate: str | None = None
                reason = ""
                candidates: tuple[str, ...] = ()
                upper_supplied = supplied.upper()
                if re.fullmatch(r"M\d{3}", upper_supplied):
                    candidate, reason = upper_supplied, "EXPLICIT_STABLE_ID"
                elif prefix is not None:
                    candidate, reason = prefix.group(1).upper(), "TITLE_STABLE_ID_PREFIX"
                    title = title[prefix.end() :].strip()
                else:
                    alias_key = " ".join(supplied.casefold().split())
                    if alias_key in aliases:
                        candidate, reason = aliases[alias_key], "REGISTERED_ALIAS"
                    else:
                        exact = title_map.get(" ".join(title.casefold().split()), [])
                        if len(exact) == 1:
                            candidate, reason = exact[0], "UNIQUE_EXACT_TITLE"
                        elif len(exact) > 1:
                            candidates = tuple(sorted(exact))
                            reason = "AMBIGUOUS_EXACT_TITLE"
                        else:
                            fingerprints = fingerprint_map.get(self._content_fingerprint(item), [])
                            if len(fingerprints) == 1:
                                candidate, reason = (
                                    fingerprints[0],
                                    "UNIQUE_CONTENT_FINGERPRINT",
                                )
                            elif len(fingerprints) > 1:
                                candidates = tuple(sorted(fingerprints))
                                reason = "AMBIGUOUS_CONTENT_FINGERPRINT"

                may_add = (
                    kind is PlanObservationKind.SNAPSHOT
                    and checked_coverage is PlanCoverage.COMPLETE
                ) or bool(re.fullmatch(r"M\d{3}", upper_supplied))
                if candidate is None and may_add:
                    candidate = self._next_canonical_id(existing_by_id, reserved)
                    reason = "ALLOCATED_NEW_IDENTITY"
                if candidate is None:
                    unresolved_specs.append((item, reason or "NO_EXACT_IDENTITY_MATCH", candidates))
                    continue
                if candidate in reserved:
                    unresolved_specs.append((item, "DUPLICATE_RESOLVED_IDENTITY", (candidate,)))
                    continue
                reserved.add(candidate)
                resolved.append((item, candidate, title))

            observation_digest = digest(
                {
                    "kind": kind.value,
                    "coverage": checked_coverage.value,
                    "plan": observed,
                }
            )
            observation_cursor = self._next_cursor(conn, run_id)
            unresolved_records: list[UnresolvedMilestoneObservation] = []
            for index, (item, reason, candidates) in enumerate(unresolved_specs):
                unresolved_id = stable_id(
                    "unresolved_",
                    {
                        "run": run_id,
                        "source": source_event_id,
                        "digest": observation_digest,
                        "index": index,
                    },
                )
                conn.execute(
                    "INSERT OR IGNORE INTO v2_unresolved_milestone_observations "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        unresolved_id,
                        run_id,
                        source_event_id,
                        item.title,
                        reason,
                        json.dumps(candidates),
                        json.dumps(self._milestone_structure(item), sort_keys=True),
                        observation_cursor,
                        utc_now(),
                    ),
                )
                unresolved_records.append(
                    UnresolvedMilestoneObservation(
                        observation_id=unresolved_id,
                        run_id=run_id,
                        source_event_id=source_event_id,
                        title=item.title,
                        reason=reason,
                        candidates=candidates,
                        cursor=observation_cursor,
                    )
                )

            observation_id = stable_id(
                "planobs_",
                {"run": run_id, "source": source_event_id, "digest": observation_digest},
            )
            if unresolved_records:
                conn.execute(
                    "INSERT OR IGNORE INTO v2_plan_observations VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        observation_id,
                        run_id,
                        kind.value,
                        checked_coverage.value,
                        source_event_id,
                        observation_digest,
                        None,
                        0,
                        observation_cursor,
                        utc_now(),
                    ),
                )
                return PlanObservationResult(
                    application=None,
                    unresolved=tuple(unresolved_records),
                    observation_kind=kind,
                    coverage=checked_coverage,
                    superseded=False,
                )

            canonical_map = {item.canonical_id: canonical for item, canonical, _ in resolved}
            normalized: list[MilestoneSpec] = []
            for item, canonical, title in resolved:
                dependencies = tuple(canonical_map.get(value, value) for value in item.depends_on)
                normalized.append(
                    MilestoneSpec(
                        canonical_id=canonical,
                        title=title,
                        description=item.description,
                        completion_criteria=item.completion_criteria,
                        verification=item.verification,
                        depends_on=dependencies,
                        status=item.status,
                        entity_refs=item.entity_refs,
                        objective=item.objective,
                        scope=item.scope,
                        target_outcome=item.target_outcome,
                        downstream_assumptions=item.downstream_assumptions,
                        non_goals=item.non_goals,
                        source_plan_item_ids=item.source_plan_item_ids,
                        criteria=item.criteria,
                        steps=item.steps,
                    )
                )
            complete_snapshot = (
                kind is PlanObservationKind.SNAPSHOT and checked_coverage is PlanCoverage.COMPLETE
            )
            if complete_snapshot:
                merged = tuple(normalized)
            else:
                replacements = {item.canonical_id: item for item in normalized}
                merged = tuple(
                    replacements.pop(item.canonical_id, item) for item in current_plan.milestones
                ) + tuple(replacements.values())
            effective = PlanSpec(
                goal=observed.goal if complete_snapshot else current_plan.goal,
                milestones=merged,
                final_verification=(
                    observed.final_verification
                    if complete_snapshot
                    else current_plan.final_verification
                ),
                final_acceptance=(
                    observed.final_acceptance
                    if complete_snapshot
                    else current_plan.final_acceptance
                ),
                native_plan=(
                    observed.native_plan
                    if complete_snapshot and observed.native_plan is not None
                    else current_plan.native_plan
                ),
            )
            before_digest = digest(self._plan_structure(current_plan))
            if digest(self._plan_structure(effective)) != before_digest:
                raise ValueError(
                    "native Plan observations cannot change Milestone structure; "
                    "use the post-verification future-plan review"
                )
            application = self._apply_plan_conn(
                conn,
                self._projection(),
                run_id=run_id,
                revision_id=revision_id,
                plan=effective,
                source_event_id=source_event_id,
                requested_current=current_milestone_id,
            )
            self._link_task_requirements_conn(
                conn,
                run_id=run_id,
                plan_version_id=application.plan_version_id,
                source_event_id=source_event_id,
            )
            superseded = (
                complete_snapshot and digest(self._plan_structure(effective)) != before_digest
            )
            conn.execute(
                "INSERT OR IGNORE INTO v2_plan_observations VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    observation_id,
                    run_id,
                    kind.value,
                    checked_coverage.value,
                    source_event_id,
                    observation_digest,
                    application.plan_version_id,
                    int(superseded),
                    observation_cursor,
                    utc_now(),
                ),
            )
            return PlanObservationResult(
                application=application,
                unresolved=(),
                observation_kind=kind,
                coverage=checked_coverage,
                superseded=superseded,
            )

    def interpret_harness_plan_steps(
        self,
        *,
        run_id: str,
        steps: Sequence[Mapping[str, object]],
    ) -> PlanStepProgressClaim:
        """Interpret one native Plan snapshot without changing route state.

        A native snapshot may report many provider-local items. A completion
        claim may cover only the contiguous prefix beginning at the
        authoritative current Step. This is route projection only; observed
        completion does not assert that the enclosing Milestone is correct.
        """

        current = self.current(run_id)
        owner = self.current_step(run_id)
        route = tuple(self.milestone_steps(run_id, current.canonical_id))
        known_milestone_ids = frozenset(self.milestone_statuses(run_id))
        by_canonical_id = {str(item["step_id"]): item for item in route}
        by_address: dict[PlanStepAddress, list[dict[str, object]]] = {}
        for item in route:
            address = PlanStepAddress.parse(str(item["step_id"]))
            if address is not None:
                by_address.setdefault(address, []).append(item)
        by_title: dict[str, list[dict[str, object]]] = {}
        for item in route:
            normalized = " ".join(str(item["title"]).casefold().split())
            by_title.setdefault(normalized, []).append(item)
        known_rows = self.database.connection.execute(
            "SELECT canonical_step_id,title,milestone_identity_id "
            "FROM v2_plan_step_identities WHERE run_id=?",
            (run_id,),
        ).fetchall()
        known_canonical_ids = frozenset(str(row["canonical_step_id"]) for row in known_rows)
        known_addresses: dict[PlanStepAddress, list[object]] = {}
        for row in known_rows:
            address = PlanStepAddress.parse(str(row["canonical_step_id"]))
            if address is not None:
                known_addresses.setdefault(address, []).append(row)
        known_titles: dict[str, list[object]] = {}
        for row in known_rows:
            normalized = " ".join(str(row["title"]).casefold().split())
            known_titles.setdefault(normalized, []).append(row)

        observed: dict[str, PlanStepStatus] = {}
        unresolved: list[tuple[str, str, tuple[str, ...]]] = []
        for raw in steps:
            title = " ".join(str(raw.get("step", "")).split())
            if not title or _MILESTONE_PREFIX.match(title):
                continue
            canonical_observation = self._persisted_step_id_from_label(
                title,
                known_canonical_ids,
            )
            if canonical_observation is not None:
                item = by_canonical_id.get(canonical_observation)
                # Whole-plan Provider snapshots legitimately include later
                # Milestones.  They are observations of known route nodes but
                # cannot cross the current Milestone's acceptance boundary.
                if item is None:
                    continue
                canonical_step_id = canonical_observation
            elif (address := PlanStepAddress.parse_label(title)) is not None:
                matches = by_address.get(address, ())
                if len(matches) != 1:
                    # A native whole-plan snapshot may contain later
                    # Milestones whose concrete Steps are deliberately not
                    # materialized until activation.  Their explicit labels
                    # are native-plan observations, not unknown current-route
                    # addresses, so they cannot create a reconciliation gate.
                    if (
                        not matches
                        and address.milestone_id != current.canonical_id
                        and address.milestone_id in known_milestone_ids
                    ):
                        continue
                    # Materialized later Milestones stay visible as
                    # observations but can never cross the current Milestone
                    # acceptance gate.
                    if not matches and known_addresses.get(address):
                        continue
                    unresolved.append(
                        (
                            title,
                            (
                                "AMBIGUOUS_EXPLICIT_PLAN_STEP_ID"
                                if matches
                                else "UNKNOWN_EXPLICIT_PLAN_STEP_ID"
                            ),
                            tuple(str(candidate["step_id"]) for candidate in matches),
                        )
                    )
                    continue
                item = matches[0]
                canonical_step_id = str(item["step_id"])
            else:
                normalized_title = " ".join(title.casefold().split())
                matches = by_title.get(normalized_title, ())
                if not matches:
                    # Normal Codex update_plan entries are Provider-local
                    # navigation, not TPG identities.  Preserve the snapshot
                    # as an observation below, but do not turn an ordinary
                    # title such as "Run focused tests" into a route error or
                    # an execution gate.  Only an explicit stable TPG address
                    # (handled above) or an exact current-route title may make
                    # an authoritative progress claim.
                    continue
                if len(matches) != 1:
                    unresolved.append(
                        (
                            title,
                            "AMBIGUOUS_PLAN_STEP_TITLE",
                            tuple(str(item["step_id"]) for item in matches),
                        )
                    )
                    continue
                item = matches[0]
                canonical_step_id = str(item["step_id"])
            status = self._coerce_step_status(str(raw.get("status", "pending")))
            previous = observed.get(canonical_step_id)
            if previous is not None and previous is not status:
                unresolved.append((title, "CONFLICTING_PLAN_STEP_STATUS", (canonical_step_id,)))
                continue
            observed[canonical_step_id] = status

        owner_step_id = str(owner["step_id"]) if owner is not None else None
        route_index = {str(item["step_id"]): index for index, item in enumerate(route)}
        owner_index = route_index.get(owner_step_id) if owner_step_id is not None else None
        completed_statuses = {
            PlanStepStatus.COMPLETED_CLAIMED,
            PlanStepStatus.COMPLETED_VERIFIED,
        }
        completed_span: list[str] = []
        if owner is not None and owner_index is not None:
            for item in route[owner_index:]:
                step_id = str(item["step_id"])
                if observed.get(step_id) not in completed_statuses:
                    break
                if str(item["status"]) in {
                    PlanStepStatus.CANCELLED.value,
                    PlanStepStatus.FAILED.value,
                }:
                    break
                completed_span.append(step_id)

        return PlanStepProgressClaim(
            owner_step_id=owner_step_id,
            completed_span=tuple(completed_span),
            observed_statuses=tuple(
                (str(item["step_id"]), observed[str(item["step_id"])])
                for item in route
                if str(item["step_id"]) in observed
            ),
            unresolved=tuple(unresolved),
        )

    @staticmethod
    def _persisted_step_id_from_label(
        label: str,
        canonical_step_ids: Collection[str],
    ) -> str | None:
        """Resolve an explicit Provider label through the persisted route identity.

        A canonical Step ID is the first-class address regardless of whether
        its Milestone-local component is numeric (``S001``) or semantic
        (``branch-audit``).  Longest-first matching prevents one canonical ID
        from stealing a label that names a longer ID; the following boundary
        must be a normal label separator, so this remains deterministic and
        never becomes fuzzy title matching.
        """

        normalized = label.strip()
        for candidate in sorted(canonical_step_ids, key=len, reverse=True):
            if normalized == candidate:
                return candidate
            if not normalized.startswith(candidate):
                continue
            remainder = normalized[len(candidate) :]
            if remainder.startswith(":") or (remainder and remainder[0].isspace()):
                return candidate
        return None

    def _commit_observed_step_span_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        revision_id: str,
        source_event_id: str,
        accepted_span: Sequence[str],
        evidence_event_ids: Sequence[str],
    ) -> RouteTransitionReceipt:
        """Commit one observed contiguous current-Step span under one WAL fact."""

        accepted = tuple(dict.fromkeys(map(str, accepted_span)))
        if not accepted:
            raise ValueError("observed Step span cannot be empty")
        existing = conn.execute(
            "SELECT milestone_identity_id,owner_step_identity_id,"
            "covered_step_identity_ids_json,evidence_event_ids_json,created_cursor "
            "FROM v2_plan_step_route_commits WHERE run_id=? AND source_event_id=?",
            (run_id, source_event_id),
        ).fetchone()
        if existing is not None:
            committed_identities = tuple(
                map(str, json.loads(existing["covered_step_identity_ids_json"]))
            )
            committed_steps = tuple(
                str(row["canonical_step_id"])
                for identity_id in committed_identities
                if (
                    row := conn.execute(
                        "SELECT canonical_step_id FROM v2_plan_step_identities "
                        "WHERE run_id=? AND step_identity_id=?",
                        (run_id, identity_id),
                    ).fetchone()
                )
                is not None
            )
            if committed_steps != accepted:
                raise RuntimeError("one WAL fact attempted conflicting route commits")
            return self._route_transition_from_commit_conn(
                conn,
                run_id=run_id,
                source_event_id=source_event_id,
                row=existing,
            )

        current = self.current(run_id, conn=conn)
        live = self.current_step(run_id, conn=conn)
        if live is None or str(live["step_id"]) != accepted[0]:
            raise ValueError("route changed before the observed Step span was committed")
        route_rows = conn.execute(
            "SELECT si.canonical_step_id,si.step_identity_id,"
            "(SELECT sse.status FROM v2_plan_step_state_events sse "
            " WHERE sse.step_identity_id=si.step_identity_id "
            " ORDER BY sse.created_cursor DESC LIMIT 1) status "
            "FROM v2_plan_step_identities si WHERE si.run_id=? "
            "AND si.milestone_identity_id=? ORDER BY si.created_cursor",
            (run_id, current.identity_id),
        ).fetchall()
        live_route = tuple(
            str(row["canonical_step_id"])
            for row in route_rows
            if str(row["status"])
            not in {
                PlanStepStatus.COMPLETED_VERIFIED.value,
                PlanStepStatus.CANCELLED.value,
                PlanStepStatus.FAILED.value,
            }
        )
        if accepted != live_route[: len(accepted)]:
            raise ValueError(
                "observed Step span must be the contiguous route prefix beginning at current"
            )
        status_by_step = {
            str(row["canonical_step_id"]): PlanStepStatus(str(row["status"])) for row in route_rows
        }
        identities: list[str] = []
        for step_id in accepted:
            row = conn.execute(
                "SELECT step_identity_id,milestone_identity_id FROM v2_plan_step_identities "
                "WHERE run_id=? AND canonical_step_id=?",
                (run_id, step_id),
            ).fetchone()
            if row is None or str(row["milestone_identity_id"]) != current.identity_id:
                raise ValueError("observed Step span escaped the current Milestone")
            step_identity_id = str(row["step_identity_id"])
            identities.append(step_identity_id)
            if status_by_step[step_id] is PlanStepStatus.PENDING:
                self._record_step_state_conn(
                    conn,
                    run_id=run_id,
                    step_identity_id=step_identity_id,
                    status=PlanStepStatus.IN_PROGRESS,
                    revision_id=revision_id,
                    source_event_id=source_event_id,
                )
            self._record_step_state_conn(
                conn,
                run_id=run_id,
                step_identity_id=step_identity_id,
                status=PlanStepStatus.COMPLETED_OBSERVED,
                revision_id=revision_id,
                source_event_id=source_event_id,
            )
        commit_cursor = self._next_cursor(conn, run_id)
        conn.execute(
            "INSERT INTO v2_plan_step_route_commits VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                stable_id(
                    "step-route-commit_",
                    {"run": run_id, "source": source_event_id, "span": accepted},
                ),
                run_id,
                current.identity_id,
                identities[0],
                json.dumps(identities),
                json.dumps(tuple(dict.fromkeys(map(str, evidence_event_ids)))),
                source_event_id,
                revision_id,
                commit_cursor,
                utc_now(),
            ),
        )
        inserted = conn.execute(
            "SELECT milestone_identity_id,owner_step_identity_id,"
            "covered_step_identity_ids_json,evidence_event_ids_json,created_cursor "
            "FROM v2_plan_step_route_commits WHERE run_id=? AND source_event_id=?",
            (run_id, source_event_id),
        ).fetchone()
        assert inserted is not None
        return self._route_transition_from_commit_conn(
            conn,
            run_id=run_id,
            source_event_id=source_event_id,
            row=inserted,
        )

    @staticmethod
    def _next_step_at_route_commit_conn(
        conn: sqlite3.Connection,
        *,
        run_id: str,
        milestone_identity_id: str,
        commit_cursor: int | None,
        excluded_step_identity_id: str,
    ) -> sqlite3.Row | None:
        """Return the successor that was live at one append-only route commit.

        Looking through the commit cursor makes replay stable even after later
        Steps have completed.  A preview uses the current state and excludes
        the Step whose acceptance is about to be committed.
        """

        cursor_filter = " AND state.created_cursor<=?" if commit_cursor is not None else ""
        parameters: list[object] = [
            run_id,
            milestone_identity_id,
            excluded_step_identity_id,
        ]
        if commit_cursor is not None:
            parameters.extend((commit_cursor, commit_cursor, commit_cursor))
        status_expression = (
            "COALESCE((SELECT state.status FROM v2_plan_step_state_events state "
            "WHERE state.step_identity_id=si.step_identity_id"
            f"{cursor_filter} ORDER BY state.created_cursor DESC LIMIT 1),'PENDING')"
        )
        created_filter = "AND si.created_cursor<=? " if commit_cursor is not None else ""
        query = (
            "SELECT si.step_identity_id,si.canonical_step_id FROM v2_plan_step_identities si "
            "WHERE si.run_id=? AND si.milestone_identity_id=? "
            "AND si.step_identity_id<>? "
            f"{created_filter}"
            f"AND {status_expression} NOT IN ('COMPLETED_VERIFIED','FAILED','CANCELLED') "
            "ORDER BY CASE WHEN si.corrective=1 THEN 0 ELSE 1 END,"
            f"CASE {status_expression} "
            "WHEN 'COMPLETED_CLAIMED' THEN 0 WHEN 'IN_PROGRESS' THEN 1 ELSE 2 END,"
            "si.created_cursor LIMIT 1"
        )
        return conn.execute(query, tuple(parameters)).fetchone()

    def _route_transition_from_commit_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        source_event_id: str,
        row: sqlite3.Row,
    ) -> RouteTransitionReceipt:
        owner = conn.execute(
            "SELECT canonical_step_id FROM v2_plan_step_identities "
            "WHERE run_id=? AND step_identity_id=?",
            (run_id, str(row["owner_step_identity_id"])),
        ).fetchone()
        if owner is None:
            raise RuntimeError("route commit lost its owner Step identity")
        covered_identities = tuple(map(str, json.loads(str(row["covered_step_identity_ids_json"]))))
        covered_by_identity = (
            {
                str(item["step_identity_id"]): str(item["canonical_step_id"])
                for item in conn.execute(
                    "SELECT step_identity_id,canonical_step_id "
                    "FROM v2_plan_step_identities "
                    f"WHERE run_id=? AND step_identity_id IN "
                    f"({','.join('?' for _ in covered_identities)})",
                    (run_id, *covered_identities),
                ).fetchall()
            }
            if covered_identities
            else {}
        )
        covered_step_ids = tuple(
            covered_by_identity[identity]
            for identity in covered_identities
            if identity in covered_by_identity
        )
        successor = self._next_step_at_route_commit_conn(
            conn,
            run_id=run_id,
            milestone_identity_id=str(row["milestone_identity_id"]),
            commit_cursor=int(row["created_cursor"]),
            excluded_step_identity_id=str(row["owner_step_identity_id"]),
        )
        next_step_id = str(successor["canonical_step_id"]) if successor is not None else None
        return RouteTransitionReceipt(
            transition_kind=(
                RouteTransitionKind.STEP_ADVANCED
                if next_step_id is not None
                else RouteTransitionKind.MILESTONE_EXECUTION_BOUNDARY
            ),
            milestone_id=str(row["milestone_identity_id"]),
            completed_step_id=str(owner["canonical_step_id"]),
            next_step_id=next_step_id,
            source_event_id=source_event_id,
            covered_step_ids=covered_step_ids,
        )

    def preview_verified_step_transition(
        self,
        *,
        run_id: str,
        step_id: str,
        source_event_id: str,
    ) -> RouteTransitionReceipt:
        """Describe the transition written into WAL before route mutation."""

        conn = self.database.connection
        current = self.current(run_id, conn=conn)
        live = self.current_step(run_id, conn=conn)
        if live is None or str(live["step_id"]) != step_id:
            raise ValueError("only the authoritative current Step can be previewed")
        successor = self._next_step_at_route_commit_conn(
            conn,
            run_id=run_id,
            milestone_identity_id=current.identity_id,
            commit_cursor=None,
            excluded_step_identity_id=str(live["step_identity_id"]),
        )
        next_step_id = str(successor["canonical_step_id"]) if successor is not None else None
        return RouteTransitionReceipt(
            transition_kind=(
                RouteTransitionKind.STEP_ADVANCED
                if next_step_id is not None
                else RouteTransitionKind.MILESTONE_EXECUTION_BOUNDARY
            ),
            milestone_id=current.identity_id,
            completed_step_id=step_id,
            next_step_id=next_step_id,
            source_event_id=source_event_id,
            covered_step_ids=(step_id,),
        )

    def route_transition_for_source(
        self,
        *,
        run_id: str,
        source_event_id: str,
    ) -> RouteTransitionReceipt | None:
        conn = self.database.connection
        row = conn.execute(
            "SELECT milestone_identity_id,owner_step_identity_id,"
            "covered_step_identity_ids_json,evidence_event_ids_json,created_cursor "
            "FROM v2_plan_step_route_commits WHERE run_id=? AND source_event_id=?",
            (run_id, source_event_id),
        ).fetchone()
        if row is None:
            return None
        return self._route_transition_from_commit_conn(
            conn,
            run_id=run_id,
            source_event_id=source_event_id,
            row=row,
        )

    def commit_verified_step_transition(
        self,
        *,
        run_id: str,
        revision_id: str,
        source_event_id: str,
        accepted_span: Sequence[str],
        evidence_event_ids: Sequence[str],
    ) -> RouteTransitionReceipt:
        """Commit one Step and return its stable typed route transition."""

        self._require_durable_event(source_event_id)
        with self.database.transaction() as conn:
            transition = self._commit_observed_step_span_conn(
                conn,
                run_id=run_id,
                revision_id=revision_id,
                source_event_id=source_event_id,
                accepted_span=accepted_span,
                evidence_event_ids=evidence_event_ids,
            )
            cursor = self._next_cursor(conn, run_id)
            self._project_current_step_conn(
                conn,
                run_id=run_id,
                revision_id=revision_id,
                source_event_id=source_event_id,
                cursor=cursor,
            )
        return transition

    def commit_verified_step_span(
        self,
        *,
        run_id: str,
        revision_id: str,
        source_event_id: str,
        accepted_span: Sequence[str],
        evidence_event_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Compatibility wrapper for pre-refactor callers.

        New execution uses :meth:`commit_observed_step_span`; Step completion
        is navigation state and no verifier receipt is required.
        """

        return self.commit_verified_step_transition(
            run_id=run_id,
            revision_id=revision_id,
            source_event_id=source_event_id,
            accepted_span=accepted_span,
            evidence_event_ids=evidence_event_ids,
        ).committed_step_ids

    def commit_observed_step_span(
        self,
        *,
        run_id: str,
        revision_id: str,
        source_event_id: str,
        observed_span: Sequence[str],
    ) -> tuple[str, ...]:
        """Advance a contiguous native-Plan route observation."""

        return self.commit_verified_step_transition(
            run_id=run_id,
            revision_id=revision_id,
            source_event_id=source_event_id,
            accepted_span=observed_span,
            evidence_event_ids=(),
        ).committed_step_ids

    def observe_focus_progress(
        self,
        *,
        run_id: str,
        revision_id: str,
        source_event_id: str,
        step_id: str,
        basis: str,
        matched_entities: Sequence[str] = (),
    ) -> tuple[str, ...]:
        """Move the lightweight Focus pointer from one durable work fact.

        This is deliberately separate from ``v2_plan_step_route_commits`` and
        ``v2_plan_step_state_events``.  A Focus observation says only that the
        natural execution cursor may move on; it is not a Step receipt, does
        not verify a predicate, and cannot submit Milestone acceptance.
        """

        if not basis.strip():
            raise ValueError("Focus observation requires a deterministic basis")
        self._require_durable_event(source_event_id)
        with self.database.transaction() as conn:
            existing = conn.execute(
                "SELECT step_identity_id FROM v2_focus_observation_events "
                "WHERE run_id=? AND source_event_id=?",
                (run_id, source_event_id),
            ).fetchone()
            if existing is not None:
                row = conn.execute(
                    "SELECT canonical_step_id FROM v2_plan_step_identities "
                    "WHERE step_identity_id=?",
                    (str(existing["step_identity_id"]),),
                ).fetchone()
                if row is None or str(row["canonical_step_id"]) != step_id:
                    raise RuntimeError("one WAL fact attempted conflicting Focus observations")
                return (step_id,)
            current = self.current(run_id, conn=conn)
            live = self.current_step(run_id, conn=conn)
            if live is None or str(live["step_id"]) != step_id:
                raise ValueError("Focus observation must target the current navigation node")
            cursor = self._next_cursor(conn, run_id)
            conn.execute(
                "INSERT INTO v2_focus_observation_events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    stable_id(
                        "focus_",
                        {"run": run_id, "step": step_id, "source": source_event_id},
                    ),
                    run_id,
                    current.plan_version_id,
                    current.identity_id,
                    str(live["step_identity_id"]),
                    basis,
                    json.dumps(tuple(dict.fromkeys(map(str, matched_entities)))),
                    source_event_id,
                    revision_id,
                    cursor,
                    utc_now(),
                ),
            )
            self._project_current_step_conn(
                conn,
                run_id=run_id,
                revision_id=revision_id,
                source_event_id=source_event_id,
                cursor=cursor,
            )
        return (step_id,)

    def observe_harness_plan_steps(
        self,
        *,
        run_id: str,
        revision_id: str,
        steps: Sequence[Mapping[str, object]],
        source_event_id: str,
        observed_span: Sequence[str] | None = None,
        accepted_span: Sequence[str] = (),
        evidence_event_ids: Sequence[str] = (),
        unmet_criteria: Sequence[str] = (),
        missing_evidence_types: Mapping[str, Sequence[str]] | None = None,
        evidence_rejection_reasons: Mapping[str, Sequence[str]] | None = None,
    ) -> tuple[str, ...]:
        """Project native Plan progress into the lightweight route.

        The Provider Plan is an observation surface, not a second route state
        machine.  Pending/in-progress rows cannot activate work and failed or
        cancelled rows cannot decide Milestone correctness. A completed
        contiguous prefix advances navigation directly; Milestone acceptance
        remains the sole correctness boundary.
        """

        self._require_durable_event(source_event_id)
        claim = self.interpret_harness_plan_steps(run_id=run_id, steps=steps)
        legacy_accepted = tuple(map(str, accepted_span))
        observed = tuple(map(str, observed_span)) if observed_span is not None else legacy_accepted
        if legacy_accepted and observed != legacy_accepted:
            raise ValueError("conflicting observed and legacy accepted Step spans")
        if observed and observed != claim.completed_span:
            raise ValueError("observed Step span does not match the native Plan claim")
        if observed and claim.owner_step_id != observed[0]:
            raise ValueError("observed Step span must begin at the current route node")

        touched: list[str] = []
        with self.database.transaction() as conn:
            current = self.current(run_id, conn=conn)
            plan_version_id = current.plan_version_id
            for title, reason, candidates in claim.unresolved:
                self._record_unresolved_conn(
                    conn,
                    run_id=run_id,
                    source_event_id=source_event_id,
                    title=title,
                    reason=reason,
                    candidates=candidates,
                    observation={"plan": list(steps)},
                )
            if observed:
                existing_span = conn.execute(
                    "SELECT covered_step_identity_ids_json "
                    "FROM v2_focus_span_observation_events "
                    "WHERE run_id=? AND source_event_id=?",
                    (run_id, source_event_id),
                ).fetchone()
                identities: list[str] = []
                for step_id in observed:
                    row = conn.execute(
                        "SELECT step_identity_id,milestone_identity_id "
                        "FROM v2_plan_step_identities "
                        "WHERE run_id=? AND canonical_step_id=?",
                        (run_id, step_id),
                    ).fetchone()
                    if row is None or str(row["milestone_identity_id"]) != current.identity_id:
                        raise ValueError("native Plan Focus span escaped the current Milestone")
                    identities.append(str(row["step_identity_id"]))
                if existing_span is not None:
                    if tuple(json.loads(str(existing_span[0]))) != tuple(identities):
                        raise RuntimeError(
                            "one WAL fact attempted conflicting Focus-span observations"
                        )
                else:
                    span_cursor = self._next_cursor(conn, run_id)
                    conn.execute(
                        "INSERT INTO v2_focus_span_observation_events VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (
                            stable_id(
                                "focusspan_",
                                {
                                    "run": run_id,
                                    "source": source_event_id,
                                    "steps": observed,
                                },
                            ),
                            run_id,
                            plan_version_id,
                            current.identity_id,
                            json.dumps(tuple(identities)),
                            "NATIVE_PLAN_CONTIGUOUS_COMPLETION",
                            source_event_id,
                            revision_id,
                            span_cursor,
                            utc_now(),
                        ),
                    )
                touched.extend(observed)
            observation_digest = digest({"kind": "STEP_PROGRESS", "steps": steps})
            cursor = self._next_cursor(conn, run_id)
            conn.execute(
                "INSERT OR IGNORE INTO v2_plan_observations VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    stable_id(
                        "planobs_",
                        {"run": run_id, "source": source_event_id, "digest": observation_digest},
                    ),
                    run_id,
                    PlanObservationKind.PATCH.value,
                    PlanCoverage.UNKNOWN.value,
                    source_event_id,
                    observation_digest,
                    plan_version_id,
                    0,
                    cursor,
                    utc_now(),
                ),
            )
            self._project_current_step_conn(
                conn,
                run_id=run_id,
                revision_id=revision_id,
                source_event_id=source_event_id,
                cursor=cursor,
            )
        return tuple(dict.fromkeys(touched))

    @staticmethod
    def _stale_milestone_progress(
        previous: MilestoneStatus,
        observed: MilestoneStatus,
    ) -> bool:
        rank = {
            MilestoneStatus.PENDING: 0,
            MilestoneStatus.IN_PROGRESS: 1,
            MilestoneStatus.COMPLETED_CLAIMED: 2,
            MilestoneStatus.COMPLETED_VERIFIED: 3,
        }
        return previous in rank and observed in rank and rank[observed] < rank[previous]

    @staticmethod
    def _stale_step_progress(
        previous: PlanStepStatus,
        observed: PlanStepStatus,
    ) -> bool:
        rank = {
            PlanStepStatus.PENDING: 0,
            PlanStepStatus.IN_PROGRESS: 1,
            PlanStepStatus.COMPLETED_CLAIMED: 2,
            PlanStepStatus.COMPLETED_VERIFIED: 3,
        }
        return previous in rank and observed in rank and rank[observed] < rank[previous]

    def _step_effectively_completed_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        step_identity_id: str,
        assumed_verified_step_ids: frozenset[str] = frozenset(),
        visiting: frozenset[str] = frozenset(),
    ) -> bool:
        """Resolve append-only failure history through its validated correction chain."""

        if step_identity_id in visiting:
            raise ValueError("PlanStep correction ancestry contains a cycle")
        row = conn.execute(
            "SELECT si.canonical_step_id,sc.minimum_acceptance_json,"
            "(SELECT sse.status FROM v2_plan_step_state_events sse "
            " WHERE sse.step_identity_id=si.step_identity_id "
            " ORDER BY sse.created_cursor DESC LIMIT 1) status "
            "FROM v2_plan_step_identities si "
            "JOIN v2_effective_plan_step_contracts sc "
            "ON sc.step_identity_id=si.step_identity_id "
            "WHERE si.run_id=? AND si.step_identity_id=?",
            (run_id, step_identity_id),
        ).fetchone()
        if row is None:
            raise KeyError(step_identity_id)
        canonical_step_id = str(row["canonical_step_id"])
        if canonical_step_id in assumed_verified_step_ids:
            return True
        status = PlanStepStatus(str(row["status"]))
        if status is PlanStepStatus.COMPLETED_VERIFIED:
            return True
        if status is PlanStepStatus.COMPLETED_CLAIMED and not bool(
            json.loads(str(row["minimum_acceptance_json"]))
        ):
            return True
        if status is not PlanStepStatus.FAILED:
            return False

        corrections = conn.execute(
            "SELECT correction.corrective_step_identity_id,correction.created_cursor,"
            "(SELECT sse.status FROM v2_plan_step_state_events sse "
            " WHERE sse.step_identity_id=correction.corrective_step_identity_id "
            " ORDER BY sse.created_cursor DESC LIMIT 1) status "
            "FROM v2_plan_step_corrections correction "
            "WHERE correction.run_id=? AND correction.failed_step_identity_id=? "
            "ORDER BY correction.created_cursor",
            (run_id, step_identity_id),
        ).fetchall()
        if not corrections:
            return False
        next_visiting = visiting | {step_identity_id}
        final_correction_id = str(corrections[-1]["corrective_step_identity_id"])
        for correction in corrections:
            corrective_id = str(correction["corrective_step_identity_id"])
            corrective_status = PlanStepStatus(str(correction["status"]))
            if corrective_status is PlanStepStatus.CANCELLED:
                if corrective_id == final_correction_id:
                    return False
                continue
            if not self._step_effectively_completed_conn(
                conn,
                run_id=run_id,
                step_identity_id=corrective_id,
                assumed_verified_step_ids=assumed_verified_step_ids,
                visiting=next_visiting,
            ):
                return False
        return True

    def _milestone_steps_ready_for_claim_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        canonical_id: str,
        assumed_verified_step_ids: frozenset[str] = frozenset(),
    ) -> bool:
        rows = conn.execute(
            "SELECT si.step_identity_id,"
            "(SELECT sse.status FROM v2_plan_step_state_events sse "
            " WHERE sse.step_identity_id=si.step_identity_id "
            " ORDER BY sse.created_cursor DESC LIMIT 1) status "
            "FROM v2_plan_step_identities si "
            "JOIN v2_milestone_identities mi ON mi.identity_id=si.milestone_identity_id "
            "WHERE si.run_id=? AND mi.canonical_id=? ORDER BY si.created_cursor",
            (run_id, canonical_id),
        ).fetchall()
        live = tuple(row for row in rows if str(row["status"]) != PlanStepStatus.CANCELLED.value)
        return bool(live) and all(
            self._step_effectively_completed_conn(
                conn,
                run_id=run_id,
                step_identity_id=str(row["step_identity_id"]),
                assumed_verified_step_ids=assumed_verified_step_ids,
            )
            for row in live
        )

    def milestone_steps_ready_for_claim(
        self,
        run_id: str,
        canonical_id: str,
        *,
        assuming_verified_step_id: str | None = None,
    ) -> bool:
        """Return whether the live route, including correction lineage, is complete."""

        assumed = (
            frozenset({assuming_verified_step_id})
            if assuming_verified_step_id is not None
            else frozenset()
        )
        return self._milestone_steps_ready_for_claim_conn(
            self.database.connection,
            run_id=run_id,
            canonical_id=canonical_id,
            assumed_verified_step_ids=assumed,
        )

    def observe_milestone_completion_claim(
        self,
        *,
        run_id: str,
        canonical_id: str,
        completed_step_ids: Sequence[str],
        criterion_ids: Sequence[str],
        revision_id: str,
        source_event_id: str,
    ) -> tuple[str, ...]:
        """Atomically close navigation and open Milestone acceptance.

        This is the fail-closed fallback for Harnesses/models that finish a
        stage without emitting a native ``turn/plan/updated`` notification.
        It cannot select a different Milestone, omit a live PlanStep, or omit a
        required completion criterion. Steps are marked observed, never
        acceptance-verified; the Milestone verifier remains authoritative.
        """

        self._require_durable_event(source_event_id)
        supplied_steps = tuple(map(str, completed_step_ids))
        supplied_criteria = tuple(map(str, criterion_ids))
        with self.database.transaction() as conn:
            current = self.current(run_id, conn=conn)
            if canonical_id != current.canonical_id:
                raise ValueError(
                    "completion claim does not target the authoritative current Milestone"
                )
            criteria_rows = conn.execute(
                "SELECT c.local_criterion_id FROM v2_completion_criteria c "
                "JOIN v2_plan_milestones pm "
                "ON pm.milestone_version_id=c.milestone_version_id "
                "WHERE c.run_id=? AND c.milestone_identity_id=? "
                "AND pm.plan_version_id=? AND c.required=1 "
                "ORDER BY c.local_criterion_id",
                (run_id, current.identity_id, current.plan_version_id),
            ).fetchall()
            expected_criteria = tuple(str(row["local_criterion_id"]) for row in criteria_rows)
            step_rows = conn.execute(
                "SELECT si.step_identity_id,si.canonical_step_id,"
                "(SELECT sse.status FROM v2_plan_step_state_events sse "
                " WHERE sse.step_identity_id=si.step_identity_id "
                " ORDER BY sse.created_cursor DESC LIMIT 1) status "
                "FROM v2_plan_step_identities si "
                "WHERE si.run_id=? AND si.milestone_identity_id=? "
                "ORDER BY si.canonical_step_id",
                (run_id, current.identity_id),
            ).fetchall()
            live_steps = tuple(
                row for row in step_rows if str(row["status"]) != PlanStepStatus.CANCELLED.value
            )
            expected_steps = tuple(str(row["canonical_step_id"]) for row in live_steps)
            if set(supplied_criteria) != set(expected_criteria) or len(supplied_criteria) != len(
                expected_criteria
            ):
                raise ValueError(
                    "completion claim criterion_ids do not exactly match required criteria: "
                    f"expected {expected_criteria}, got {supplied_criteria}"
                )
            if set(supplied_steps) != set(expected_steps) or len(supplied_steps) != len(
                expected_steps
            ):
                raise ValueError(
                    "completion claim completed_step_ids do not exactly match live PlanSteps: "
                    f"expected {expected_steps}, got {supplied_steps}"
                )
            for row in live_steps:
                step_status = PlanStepStatus(str(row["status"]))
                if step_status is PlanStepStatus.FAILED:
                    raise ValueError(
                        "completion claim cannot bypass an unresolved failed navigation node: "
                        f"{row['canonical_step_id']}"
                    )
                if step_status is PlanStepStatus.COMPLETED_OBSERVED:
                    continue
                if step_status is PlanStepStatus.PENDING:
                    self._record_step_state_conn(
                        conn,
                        run_id=run_id,
                        step_identity_id=str(row["step_identity_id"]),
                        status=PlanStepStatus.IN_PROGRESS,
                        revision_id=revision_id,
                        source_event_id=source_event_id,
                    )
                self._record_step_state_conn(
                    conn,
                    run_id=run_id,
                    step_identity_id=str(row["step_identity_id"]),
                    status=PlanStepStatus.COMPLETED_OBSERVED,
                    revision_id=revision_id,
                    source_event_id=source_event_id,
                )
            if current.status == MilestoneStatus.PENDING.value:
                self._record_milestone_state_conn(
                    conn,
                    run_id=run_id,
                    identity_id=current.identity_id,
                    status=MilestoneStatus.IN_PROGRESS,
                    revision_id=revision_id,
                    source_event_id=source_event_id,
                )
            milestone_state = self._record_milestone_state_conn(
                conn,
                run_id=run_id,
                identity_id=current.identity_id,
                status=MilestoneStatus.COMPLETED_CLAIMED,
                revision_id=revision_id,
                source_event_id=source_event_id,
            )
            self._project_current_step_conn(
                conn,
                run_id=run_id,
                revision_id=revision_id,
                source_event_id=source_event_id,
                cursor=(
                    milestone_state.cursor
                    if milestone_state is not None
                    else self._next_cursor(conn, run_id)
                ),
            )
        return expected_steps

    @staticmethod
    def _coerce_step_status(value: str) -> PlanStepStatus:
        normalized = value.strip().replace("-", "_").casefold()
        mapping = {
            "pending": PlanStepStatus.PENDING,
            "inprogress": PlanStepStatus.IN_PROGRESS,
            "in_progress": PlanStepStatus.IN_PROGRESS,
            "completed": PlanStepStatus.COMPLETED_CLAIMED,
            "completed_claimed": PlanStepStatus.COMPLETED_CLAIMED,
            "completed_verified": PlanStepStatus.COMPLETED_VERIFIED,
            "failed": PlanStepStatus.FAILED,
            "cancelled": PlanStepStatus.CANCELLED,
        }
        try:
            return mapping[normalized]
        except KeyError as exc:
            raise ValueError(f"unsupported PlanStep status: {value}") from exc

    @staticmethod
    def _next_step_id_conn(
        conn: sqlite3.Connection, run_id: str, milestone_canonical_id: str, *, corrective: bool
    ) -> str:
        marker = "R" if corrective else "S"
        prefix = f"{milestone_canonical_id}.{marker}"
        rows = conn.execute(
            "SELECT canonical_step_id FROM v2_plan_step_identities "
            "WHERE run_id=? AND canonical_step_id LIKE ?",
            (run_id, prefix + "%"),
        ).fetchall()
        numbers = [
            int(match.group(1))
            for row in rows
            if (match := re.fullmatch(re.escape(prefix) + r"(\d{3,})", str(row[0])))
        ]
        return f"{prefix}{max(numbers, default=0) + 1:03d}"

    def _record_unresolved_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        source_event_id: str,
        title: str,
        reason: str,
        candidates: tuple[str, ...],
        observation: Mapping[str, object],
    ) -> None:
        cursor = self._next_cursor(conn, run_id)
        observation_id = stable_id(
            "unresolved_",
            {"run": run_id, "source": source_event_id, "title": title, "reason": reason},
        )
        conn.execute(
            "INSERT OR IGNORE INTO v2_unresolved_milestone_observations VALUES(?,?,?,?,?,?,?,?,?)",
            (
                observation_id,
                run_id,
                source_event_id,
                title,
                reason,
                json.dumps(candidates),
                json.dumps(dict(observation), sort_keys=True),
                cursor,
                utc_now(),
            ),
        )

    def current_step(
        self,
        run_id: str,
        canonical_id: str | None = None,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, object] | None:
        """Return the one local working pointer beneath the current Milestone.

        The Registry remains authoritative. The Semantic Graph mirrors this
        pointer for route inspection; it never chooses a Step on its own.
        """

        connection = conn or self.database.connection
        current = self.current(run_id, conn=connection)
        if canonical_id is not None and canonical_id != current.canonical_id:
            raise ValueError("current Step can only be queried for the current Milestone")
        row = connection.execute(
            "SELECT si.step_identity_id,si.canonical_step_id,si.title,si.corrective,"
            "si.created_cursor,"
            "si.criterion_ids_json,si.entity_refs_json,"
            "si.historical_dependency_refs_json,si.source_plan_item_ids_json,"
            "sc.expected_outcome,"
            "sc.minimum_acceptance_json,sc.failure_signals_json,"
            "sc.contract_revision_number,"
            "(SELECT sse.status FROM v2_plan_step_state_events sse "
            " WHERE sse.step_identity_id=si.step_identity_id "
            " ORDER BY sse.created_cursor DESC LIMIT 1) status "
            "FROM v2_plan_step_identities si "
            "JOIN v2_effective_plan_step_contracts sc "
            "ON sc.step_identity_id=si.step_identity_id "
            "WHERE si.run_id=? AND si.milestone_identity_id=? "
            "AND NOT EXISTS (SELECT 1 FROM v2_focus_observation_events focus "
            " WHERE focus.run_id=si.run_id "
            " AND focus.step_identity_id=si.step_identity_id) "
            "AND NOT EXISTS (SELECT 1 "
            " FROM v2_focus_span_observation_events focus_span, "
            " json_each(focus_span.covered_step_identity_ids_json) covered "
            " WHERE focus_span.run_id=si.run_id "
            " AND CAST(covered.value AS TEXT)=si.step_identity_id) "
            "AND COALESCE((SELECT state.status FROM v2_plan_step_state_events state "
            " WHERE state.step_identity_id=si.step_identity_id "
            " ORDER BY state.created_cursor DESC LIMIT 1),'PENDING') "
            "NOT IN ('COMPLETED_VERIFIED','FAILED','CANCELLED') "
            "ORDER BY CASE WHEN si.corrective=1 THEN 0 ELSE 1 END,"
            "CASE COALESCE((SELECT state.status FROM v2_plan_step_state_events state "
            " WHERE state.step_identity_id=si.step_identity_id "
            " ORDER BY state.created_cursor DESC LIMIT 1),'PENDING') "
            "WHEN 'COMPLETED_CLAIMED' THEN 0 WHEN 'IN_PROGRESS' THEN 1 ELSE 2 END,"
            "si.created_cursor LIMIT 1",
            (run_id, current.identity_id),
        ).fetchone()
        if row is None:
            return None
        return self._step_record(row, current)

    @staticmethod
    def _step_record(row: sqlite3.Row, current: CurrentMilestone) -> dict[str, object]:
        return {
            "step_identity_id": str(row["step_identity_id"]),
            "step_id": str(row["canonical_step_id"]),
            "title": str(row["title"]),
            "status": str(row["status"]),
            "corrective": bool(row["corrective"]),
            "created_cursor": int(row["created_cursor"]),
            "criterion_ids": tuple(json.loads(str(row["criterion_ids_json"]))),
            "entity_refs": tuple(json.loads(str(row["entity_refs_json"]))),
            "historical_dependency_refs": tuple(
                json.loads(str(row["historical_dependency_refs_json"]))
            ),
            "source_plan_item_ids": tuple(json.loads(str(row["source_plan_item_ids_json"]))),
            "expected_outcome": str(row["expected_outcome"]),
            "minimum_acceptance": tuple(json.loads(str(row["minimum_acceptance_json"]))),
            "failure_signals": tuple(json.loads(str(row["failure_signals_json"]))),
            "contract_revision_number": int(row["contract_revision_number"]),
            "milestone_identity_id": current.identity_id,
            "milestone_canonical_id": current.canonical_id,
            "plan_version_id": current.plan_version_id,
        }

    def activate_current_work(
        self,
        *,
        run_id: str,
        revision_id: str,
        source_event_id: str,
    ) -> bool:
        """Make the current working pointer match a durable execution fact.

        The model remains authoritative for what work to do. Once a semantic
        execution Event is in the WAL, leaving its current Milestone or
        navigation Step ``PENDING`` would contradict observed execution and
        break provenance attribution. Activation never claims acceptance.
        """

        self._require_durable_event(source_event_id)
        with self.database.transaction() as conn:
            current = self.current(run_id, conn=conn)
            step = self.current_step(run_id, conn=conn)
            cursors: list[int] = []
            if MilestoneStatus(current.status) is MilestoneStatus.PENDING:
                milestone_event = self._record_milestone_state_conn(
                    conn,
                    run_id=run_id,
                    identity_id=current.identity_id,
                    status=MilestoneStatus.IN_PROGRESS,
                    revision_id=revision_id,
                    source_event_id=source_event_id,
                )
                if milestone_event is not None:
                    cursors.append(milestone_event.cursor)
            if step is not None and PlanStepStatus(str(step["status"])) is PlanStepStatus.PENDING:
                cursors.append(
                    self._record_step_state_conn(
                        conn,
                        run_id=run_id,
                        step_identity_id=str(step["step_identity_id"]),
                        status=PlanStepStatus.IN_PROGRESS,
                        revision_id=revision_id,
                        source_event_id=source_event_id,
                    )
                )
            if not cursors:
                return False
            self._project_current_step_conn(
                conn,
                run_id=run_id,
                revision_id=revision_id,
                source_event_id=source_event_id,
                cursor=max(cursors),
            )
            return True

    def step_reviews(self, run_id: str, step_id: str) -> tuple[dict[str, object], ...]:
        rows = self.database.connection.execute(
            "SELECT r.* FROM v2_plan_step_review_events r "
            "JOIN v2_plan_step_identities si ON si.step_identity_id=r.step_identity_id "
            "WHERE r.run_id=? AND si.canonical_step_id=? ORDER BY r.created_cursor",
            (run_id, step_id),
        ).fetchall()
        return tuple(
            {
                "review_id": str(row["review_id"]),
                "decision": str(row["decision"]),
                "summary": str(row["summary"]),
                "evidence_event_ids": tuple(json.loads(str(row["evidence_event_ids_json"]))),
                "created_step_ids": tuple(json.loads(str(row["created_step_ids_json"]))),
                "source_event_id": str(row["source_event_id"]),
                "revision_id": str(row["revision_id"]),
            }
            for row in rows
        )

    def _root_step_contract_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        step_identity_id: str,
    ) -> sqlite3.Row:
        """Resolve a corrective Step to the original failed route Step."""

        root_identity_id = step_identity_id
        visited: set[str] = set()
        while True:
            if root_identity_id in visited:
                raise ValueError("PlanStep correction ancestry contains a cycle")
            visited.add(root_identity_id)
            parent = conn.execute(
                "SELECT failed_step_identity_id FROM v2_plan_step_corrections "
                "WHERE run_id=? AND corrective_step_identity_id=? "
                "ORDER BY created_cursor DESC LIMIT 1",
                (run_id, root_identity_id),
            ).fetchone()
            if parent is None:
                break
            root_identity_id = str(parent["failed_step_identity_id"])
        root = conn.execute(
            "SELECT si.step_identity_id,si.milestone_identity_id,"
            "si.canonical_step_id,si.criterion_ids_json "
            "FROM v2_plan_step_identities si "
            "JOIN v2_effective_plan_step_contracts sc "
            "ON sc.step_identity_id=si.step_identity_id "
            "WHERE si.run_id=? AND si.step_identity_id=?",
            (run_id, root_identity_id),
        ).fetchone()
        if root is None:
            raise KeyError(step_identity_id)
        return root

    def _validate_corrective_chain_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        step_identity_id: str,
        corrective_steps: Sequence[PlanStepSpec],
        revision_id: str | None = None,
    ) -> None:
        """Validate meaningful corrective navigation for the current Milestone.

        ``criterion_ids`` are graph links to the failed Milestone requirements;
        expected outcomes and failure signals keep the route actionable. No
        corrective Step owns a separate acceptance contract.
        """

        if not corrective_steps:
            raise ValueError("CORRECT requires at least one same-Milestone corrective Step")
        root = self._root_step_contract_conn(
            conn,
            run_id=run_id,
            step_identity_id=step_identity_id,
        )
        root_criterion_ids = frozenset(map(str, json.loads(str(root["criterion_ids_json"]))))
        mapped_by_chain = set().union(*(set(step.criterion_ids) for step in corrective_steps))
        known_criteria = {
            str(row["local_criterion_id"])
            for row in conn.execute(
                "SELECT local_criterion_id FROM v2_completion_criteria "
                "WHERE run_id=? AND milestone_identity_id=?",
                (run_id, str(root["milestone_identity_id"])),
            ).fetchall()
        }
        unknown_mappings = mapped_by_chain.difference(known_criteria)
        if unknown_mappings:
            raise ValueError(
                f"corrective Step maps unknown Milestone criteria: {sorted(unknown_mappings)}"
            )
        if root_criterion_ids and not root_criterion_ids.intersection(mapped_by_chain):
            raise ValueError(
                "corrective chain must contribute to at least one failed Milestone criterion: "
                f"{sorted(root_criterion_ids)}"
            )
        if any(
            not step.expected_outcome.strip() or not step.failure_signals
            for step in corrective_steps
        ):
            raise ValueError("every corrective Step requires an expected outcome and bounded risks")
        self._reject_equivalent_corrective_chain_conn(
            conn,
            run_id=run_id,
            milestone_identity_id=str(root["milestone_identity_id"]),
            revision_id=revision_id,
            corrective_steps=corrective_steps,
        )

    @staticmethod
    def _corrective_chain_signature(steps: Sequence[PlanStepSpec]) -> str:
        return digest(
            [
                {
                    "title": " ".join(step.title.split()).casefold(),
                    "criterion_ids": list(step.criterion_ids),
                    "entity_refs": list(step.entity_refs),
                    "expected_outcome": " ".join(step.expected_outcome.split()).casefold(),
                    "minimum_acceptance": primitive(step.minimum_acceptance),
                    "failure_signals": [
                        " ".join(value.split()).casefold() for value in step.failure_signals
                    ],
                }
                for step in steps
            ]
        )

    def _reject_equivalent_corrective_chain_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        milestone_identity_id: str,
        revision_id: str | None,
        corrective_steps: Sequence[PlanStepSpec],
    ) -> None:
        """Reject an exact same-revision correction plan, not a new hypothesis."""

        if revision_id is None:
            return
        proposed_signature = self._corrective_chain_signature(corrective_steps)
        reviews = conn.execute(
            "SELECT created_step_ids_json FROM v2_plan_step_review_events "
            "WHERE run_id=? AND milestone_identity_id=? AND revision_id=? "
            "AND decision='CORRECT' ORDER BY created_cursor",
            (run_id, milestone_identity_id, revision_id),
        ).fetchall()
        for review in reviews:
            step_ids = tuple(map(str, json.loads(str(review["created_step_ids_json"]))))
            if not step_ids:
                continue
            stored: list[PlanStepSpec] = []
            for canonical_step_id in step_ids:
                row = conn.execute(
                    "SELECT si.canonical_step_id,si.title,si.criterion_ids_json,"
                    "si.entity_refs_json,si.historical_dependency_refs_json,"
                    "sc.expected_outcome,sc.minimum_acceptance_json,"
                    "sc.failure_signals_json FROM v2_plan_step_identities si "
                    "JOIN v2_effective_plan_step_contracts sc "
                    "ON sc.step_identity_id=si.step_identity_id "
                    "WHERE si.run_id=? AND si.canonical_step_id=?",
                    (run_id, canonical_step_id),
                ).fetchone()
                if row is None:
                    stored = []
                    break
                stored.append(
                    PlanStepSpec.from_dict(
                        {
                            "step_id": str(row["canonical_step_id"]),
                            "title": str(row["title"]),
                            "criterion_ids": json.loads(str(row["criterion_ids_json"])),
                            "entity_refs": json.loads(str(row["entity_refs_json"])),
                            "historical_dependency_refs": json.loads(
                                str(row["historical_dependency_refs_json"])
                            ),
                            "expected_outcome": str(row["expected_outcome"]),
                            "minimum_acceptance": json.loads(str(row["minimum_acceptance_json"])),
                            "failure_signals": json.loads(str(row["failure_signals_json"])),
                            "corrective": True,
                        },
                        1,
                    )
                )
            if stored and self._corrective_chain_signature(stored) == proposed_signature:
                raise ValueError(
                    "an equivalent corrective chain already exists at the current workspace "
                    "revision; provide a concrete new hypothesis/repair delta, advance the "
                    "revision, or preserve the same Step with CONTINUE"
                )

    @staticmethod
    def _reject_duplicate_failure_contract_conn(
        conn: sqlite3.Connection,
        *,
        run_id: str,
        milestone_identity_id: str,
        revision_id: str,
        failure_signature: str | None,
    ) -> None:
        """Keep one active causal correction per failure at one repository state."""

        if not failure_signature:
            return
        existing = conn.execute(
            "SELECT corrective.step_identity_id,corrective.canonical_step_id,"
            "(SELECT state.status FROM v2_plan_step_state_events state "
            " WHERE state.step_identity_id=corrective.step_identity_id "
            " ORDER BY state.created_cursor DESC LIMIT 1) status "
            "FROM v2_corrective_steps contract "
            "JOIN v2_plan_step_identities corrective "
            "ON corrective.step_identity_id=contract.step_identity_id "
            "WHERE contract.run_id=? AND contract.milestone_identity_id=? "
            "AND contract.baseline_revision_id=? AND contract.failure_signature=? "
            "ORDER BY contract.created_cursor DESC LIMIT 1",
            (run_id, milestone_identity_id, revision_id, failure_signature),
        ).fetchone()
        if existing is None:
            return
        raise ValueError(
            "this Milestone already has a correction for the same failure signature at "
            f"the same workspace revision ({existing['canonical_step_id']}: "
            f"{existing['status']}); continue that causal chain, produce a real new "
            "revision/evidence delta, or report a validated external blocker"
        )

    def validate_corrective_chain(
        self,
        *,
        run_id: str,
        step_id: str,
        corrective_steps: Sequence[PlanStepSpec],
        revision_id: str | None = None,
    ) -> None:
        """Validate a proposed correction without changing durable state."""

        row = self.database.connection.execute(
            "SELECT step_identity_id FROM v2_plan_step_identities "
            "WHERE run_id=? AND canonical_step_id=?",
            (run_id, step_id),
        ).fetchone()
        if row is None:
            raise KeyError(step_id)
        self._validate_corrective_chain_conn(
            self.database.connection,
            run_id=run_id,
            step_identity_id=str(row["step_identity_id"]),
            corrective_steps=corrective_steps,
            revision_id=revision_id,
        )

    def record_step_review(
        self,
        *,
        run_id: str,
        step_id: str,
        decision: PlanStepReviewDecision,
        summary: str,
        revision_id: str,
        source_event_id: str,
        evidence_event_ids: Sequence[str] = (),
        corrective_steps: Sequence[PlanStepSpec] = (),
        failure_signature: str | None = None,
        failure_criterion_ids: Sequence[str] = (),
        failure_baseline_revision_id: str | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Apply a durable local control decision inside the current Milestone."""

        if not summary.strip():
            raise ValueError("PlanStep review requires a non-empty summary")
        self._require_durable_event(source_event_id)
        with self.database.transaction() as conn:
            target = conn.execute(
                "SELECT step_identity_id FROM v2_plan_step_identities "
                "WHERE run_id=? AND canonical_step_id=?",
                (run_id, step_id),
            ).fetchone()
            if target is None:
                raise KeyError(step_id)
            existing = conn.execute(
                "SELECT review_id,created_step_ids_json FROM v2_plan_step_review_events "
                "WHERE step_identity_id=? AND source_event_id=?",
                (target["step_identity_id"], source_event_id),
            ).fetchone()
            if existing is not None:
                return str(existing["review_id"]), tuple(
                    json.loads(str(existing["created_step_ids_json"]))
                )
            current = self.current(run_id, conn=conn)
            live_step = self.current_step(run_id, conn=conn)
            step = live_step
            if step is None or str(step["step_id"]) != step_id:
                raise ValueError("PlanStep review must target the one live current Step")
            if decision is PlanStepReviewDecision.CORRECT and not corrective_steps:
                raise ValueError("CORRECT requires at least one same-Milestone corrective Step")
            if decision is not PlanStepReviewDecision.CORRECT and corrective_steps:
                raise ValueError("only CORRECT may append corrective Steps")
            normalized_evidence_ids = tuple(
                dict.fromkeys(value for item in evidence_event_ids if (value := str(item).strip()))
            )
            required_step_criteria = {
                f"{step_id}:{str(item.get('criterion_id', '')).strip()}"
                for item in step.get("minimum_acceptance", ())
                if isinstance(item, dict)
                and bool(item.get("required", True))
                and str(item.get("criterion_id", "")).strip()
            }
            if (
                decision is PlanStepReviewDecision.SATISFIED
                and required_step_criteria
                and not normalized_evidence_ids
            ):
                raise ValueError(
                    "SATISFIED requires an Evidence-backed local Step acceptance receipt"
                )
            if decision is PlanStepReviewDecision.CORRECT:
                self._validate_corrective_chain_conn(
                    conn,
                    run_id=run_id,
                    step_identity_id=str(step["step_identity_id"]),
                    corrective_steps=corrective_steps,
                    revision_id=revision_id,
                )
                normalized_failure_criteria = tuple(
                    dict.fromkeys(
                        value for item in failure_criterion_ids if (value := str(item).strip())
                    )
                )
                if (
                    not normalized_evidence_ids
                    or not normalized_failure_criteria
                    or not str(failure_signature or "").strip()
                    or not str(failure_baseline_revision_id or "").strip()
                ):
                    raise ValueError(
                        "CORRECT requires a causal failure receipt with Evidence, signature, "
                        "local Criterion addresses, and baseline revision"
                    )
                unknown_failure_criteria = set(normalized_failure_criteria).difference(
                    required_step_criteria
                )
                if not required_step_criteria or unknown_failure_criteria:
                    raise ValueError(
                        "CORRECT failure criteria must address the current Step local contract: "
                        f"{sorted(unknown_failure_criteria or set(normalized_failure_criteria))}"
                    )
                self._reject_duplicate_failure_contract_conn(
                    conn,
                    run_id=run_id,
                    milestone_identity_id=current.identity_id,
                    revision_id=failure_baseline_revision_id or revision_id,
                    failure_signature=str(failure_signature).strip(),
                )
            review_id = stable_id(
                "stepreview_",
                {"step": step["step_identity_id"], "source": source_event_id},
            )
            created_ids: list[str] = []
            created_identity_ids: list[str] = []
            if decision is PlanStepReviewDecision.SATISFIED:
                self._record_step_state_conn(
                    conn,
                    run_id=run_id,
                    step_identity_id=str(step["step_identity_id"]),
                    status=PlanStepStatus.COMPLETED_VERIFIED,
                    revision_id=revision_id,
                    source_event_id=source_event_id,
                )
            elif decision is PlanStepReviewDecision.CONTINUE:
                self._record_step_state_conn(
                    conn,
                    run_id=run_id,
                    step_identity_id=str(step["step_identity_id"]),
                    status=PlanStepStatus.IN_PROGRESS,
                    revision_id=revision_id,
                    source_event_id=source_event_id,
                )
            elif decision is PlanStepReviewDecision.BLOCKED:
                raise ValueError(
                    "a model Step review cannot commit BLOCKED; external feasibility requires "
                    "a separate runtime-owned authority receipt"
                )
            else:
                self._record_step_state_conn(
                    conn,
                    run_id=run_id,
                    step_identity_id=str(step["step_identity_id"]),
                    status=PlanStepStatus.FAILED,
                    revision_id=revision_id,
                    source_event_id=source_event_id,
                )
                for proposal in corrective_steps:
                    generated_id = self._next_step_id_conn(
                        conn,
                        run_id,
                        current.canonical_id,
                        corrective=True,
                    )
                    generated = replace(
                        proposal,
                        step_id=generated_id,
                        status=PlanStepStatus.PENDING,
                        corrective=True,
                    )
                    identity_id = self._ensure_step_conn(
                        conn,
                        run_id=run_id,
                        milestone_identity_id=current.identity_id,
                        step=generated,
                        revision_id=revision_id,
                        source_event_id=source_event_id,
                    )
                    created_ids.append(generated_id)
                    created_identity_ids.append(identity_id)
                    repository_id, branch_id = self._projection_scope_conn(conn, run_id)
                    self._projection().project_plan_step(
                        conn,
                        repository_id=repository_id,
                        run_id=run_id,
                        branch_id=branch_id,
                        revision_id=revision_id,
                        plan_version_id=current.plan_version_id,
                        milestone_identity_id=current.identity_id,
                        step_identity_id=identity_id,
                        step=generated,
                        source_event_id=source_event_id,
                        cursor=self._next_cursor(conn, run_id),
                    )
                if current.status == MilestoneStatus.VERIFICATION_FAILED.value:
                    self._record_milestone_state_conn(
                        conn,
                        run_id=run_id,
                        identity_id=current.identity_id,
                        status=MilestoneStatus.REPAIRING,
                        revision_id=revision_id,
                        source_event_id=source_event_id,
                    )
            cursor = self._next_cursor(conn, run_id)
            conn.execute(
                "INSERT INTO v2_plan_step_review_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    review_id,
                    run_id,
                    current.identity_id,
                    step["step_identity_id"],
                    decision.value,
                    summary.strip(),
                    json.dumps(normalized_evidence_ids),
                    json.dumps(tuple(created_ids)),
                    source_event_id,
                    revision_id,
                    cursor,
                    utc_now(),
                ),
            )
            for identity_id in created_identity_ids:
                correction_cursor = self._next_cursor(conn, run_id)
                criterion_identity_id = None
                if len(tuple(failure_criterion_ids)) == 1:
                    criterion = conn.execute(
                        "SELECT criterion_identity_id FROM v2_completion_criteria "
                        "WHERE run_id=? AND milestone_identity_id=? "
                        "AND local_criterion_id=? ORDER BY created_cursor DESC LIMIT 1",
                        (
                            run_id,
                            current.identity_id,
                            str(tuple(failure_criterion_ids)[0]),
                        ),
                    ).fetchone()
                    if criterion is not None:
                        criterion_identity_id = str(criterion["criterion_identity_id"])
                conn.execute(
                    "INSERT OR IGNORE INTO v2_corrective_steps("
                    "corrective_id,run_id,milestone_identity_id,step_identity_id,"
                    "criterion_identity_id,reason,evidence_event_ids_json,source_event_id,"
                    "revision_id,failure_signature,failure_criterion_ids_json,"
                    "baseline_revision_id,baseline_cursor,created_cursor,created_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        stable_id(
                            "corrective_",
                            {"step": identity_id, "source": source_event_id},
                        ),
                        run_id,
                        current.identity_id,
                        identity_id,
                        criterion_identity_id,
                        summary.strip(),
                        json.dumps(normalized_evidence_ids),
                        source_event_id,
                        revision_id,
                        str(failure_signature).strip() if failure_signature else None,
                        json.dumps(
                            tuple(
                                dict.fromkeys(
                                    value
                                    for item in failure_criterion_ids
                                    if (value := str(item).strip())
                                )
                            )
                        ),
                        failure_baseline_revision_id or revision_id,
                        cursor,
                        correction_cursor,
                        utc_now(),
                    ),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO v2_plan_step_corrections VALUES(?,?,?,?,?,?,?,?)",
                    (
                        stable_id(
                            "stepcorrection_",
                            {
                                "failed": step["step_identity_id"],
                                "corrective": identity_id,
                            },
                        ),
                        run_id,
                        step["step_identity_id"],
                        identity_id,
                        review_id,
                        source_event_id,
                        correction_cursor,
                        utc_now(),
                    ),
                )
                repository_id, branch_id = self._projection_scope_conn(conn, run_id)
                self._projection().project_step_correction(
                    conn,
                    repository_id=repository_id,
                    run_id=run_id,
                    branch_id=branch_id,
                    revision_id=revision_id,
                    plan_version_id=current.plan_version_id,
                    failed_step_identity_id=str(step["step_identity_id"]),
                    corrective_step_identity_id=identity_id,
                    source_event_id=source_event_id,
                    cursor=correction_cursor,
                )
            self._project_current_step_conn(
                conn,
                run_id=run_id,
                revision_id=revision_id,
                source_event_id=source_event_id,
                cursor=cursor,
            )
            return review_id, tuple(created_ids)

    @staticmethod
    def _latest_milestone_status_conn(
        conn: sqlite3.Connection, identity_id: str
    ) -> MilestoneStatus:
        row = conn.execute(
            "SELECT status FROM v2_milestone_state_events WHERE identity_id=? "
            "ORDER BY created_cursor DESC LIMIT 1",
            (identity_id,),
        ).fetchone()
        if row is None:
            return MilestoneStatus.PENDING
        return MilestoneStatus(str(row["status"]))

    def _validated_milestone_review_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        milestone_canonical_id: str,
        decision: MilestoneReviewDecision,
        future_plan: PlanSpec | None,
        require_verified: bool = True,
    ) -> tuple[sqlite3.Row, CurrentMilestone, PlanSpec | None]:
        """Validate one route transition without changing durable state.

        ``require_verified=False`` checks a proposal recorded together with a
        boundary request, before acceptance has run; the structural rules are
        identical, only the ``COMPLETED_VERIFIED`` precondition is deferred to
        the moment the proposal is applied.
        """

        if decision not in {
            MilestoneReviewDecision.CONTINUE,
            MilestoneReviewDecision.REPLAN_FUTURE,
        }:
            raise ValueError("MilestoneReview supports only CONTINUE or REPLAN_FUTURE")
        if decision is MilestoneReviewDecision.REPLAN_FUTURE and future_plan is None:
            raise ValueError("future-plan review decision requires a pending-route delta")
        identity = conn.execute(
            "SELECT identity_id FROM v2_milestone_identities WHERE run_id=? AND canonical_id=?",
            (run_id, milestone_canonical_id),
        ).fetchone()
        if identity is None:
            raise KeyError(milestone_canonical_id)
        if require_verified and (
            self._latest_milestone_status_conn(conn, str(identity["identity_id"]))
            is not MilestoneStatus.COMPLETED_VERIFIED
        ):
            raise ValueError("MilestoneReview requires COMPLETED_VERIFIED")
        current = self.current(run_id, conn=conn)
        if current.identity_id != str(identity["identity_id"]):
            raise ValueError("MilestoneReview must review the current verified Milestone")
        if future_plan is None:
            return identity, current, None

        checked = self._coerce_plan(future_plan)
        milestone_states = {
            str(row["canonical_id"]): MilestoneStatus(str(row["status"]))
            for row in conn.execute(
                "SELECT mi.canonical_id,mse.status FROM v2_milestone_identities mi "
                "JOIN v2_milestone_state_events mse ON mse.identity_id=mi.identity_id "
                "WHERE mi.run_id=? AND mse.created_cursor=(SELECT MAX(x.created_cursor) "
                "FROM v2_milestone_state_events x WHERE x.identity_id=mi.identity_id)",
                (run_id,),
            ).fetchall()
        }
        step_states = {
            str(row["canonical_step_id"]): PlanStepStatus(str(row["status"]))
            for row in conn.execute(
                "SELECT si.canonical_step_id,sse.status FROM v2_plan_step_identities si "
                "JOIN v2_plan_step_state_events sse "
                "ON sse.step_identity_id=si.step_identity_id "
                "WHERE si.run_id=? AND sse.created_cursor=(SELECT MAX(x.created_cursor) "
                "FROM v2_plan_step_state_events x "
                "WHERE x.step_identity_id=si.step_identity_id)",
                (run_id,),
            ).fetchall()
        }
        # A Review versions structure; it is not an execution-state
        # observation. Preserve every existing Milestone/Step state and
        # initialize only genuinely new identities from the proposal.
        checked = PlanSpec(
            goal=checked.goal,
            milestones=tuple(
                replace(
                    item,
                    status=milestone_states.get(
                        item.canonical_id,
                        self._coerce_milestone_status(item.status),
                    ).value,
                    steps=tuple(
                        replace(
                            step,
                            status=step_states.get(step.step_id, step.status),
                        )
                        for step in item.steps
                    ),
                )
                for item in checked.milestones
            ),
            final_verification=checked.final_verification,
            final_acceptance=checked.final_acceptance,
            native_plan=checked.native_plan,
        )
        for item in checked.milestones:
            prior_status = milestone_states.get(item.canonical_id)
            if prior_status is None or prior_status is MilestoneStatus.PENDING:
                if self._coerce_milestone_status(item.status) is not MilestoneStatus.PENDING:
                    raise ValueError(
                        "new and not-yet-started future Milestones must remain PENDING"
                    )
                if any(step.status is not PlanStepStatus.PENDING for step in item.steps):
                    raise ValueError(
                        "future PlanSteps cannot carry execution progress into a review"
                    )
        frozen = {
            str(row["canonical_id"]): str(row["spec_digest"])
            for row in conn.execute(
                "SELECT mi.canonical_id,mv.spec_digest FROM v2_milestone_identities mi "
                "JOIN v2_milestone_state_events mse ON mse.identity_id=mi.identity_id "
                "JOIN v2_milestone_versions mv ON mv.identity_id=mi.identity_id "
                "WHERE mi.run_id=? AND mse.status<>'PENDING' "
                "AND mse.created_cursor=(SELECT MAX(x.created_cursor) "
                "FROM v2_milestone_state_events x WHERE x.identity_id=mi.identity_id) "
                "AND mv.version_number=(SELECT MAX(y.version_number) "
                "FROM v2_milestone_versions y WHERE y.identity_id=mi.identity_id)",
                (run_id,),
            ).fetchall()
        }
        proposed = {
            item.canonical_id: digest(self._milestone_structure(item))
            for item in checked.milestones
        }
        if any(proposed.get(canonical) != spec_digest for canonical, spec_digest in frozen.items()):
            raise ValueError("MilestoneReview cannot rewrite started Milestone history")

        existing_ids = {
            item.canonical_id for item in self._current_plan_conn(conn, run_id).milestones
        }
        existing_numbers = [
            int(match.group(1))
            for canonical in existing_ids
            if (match := re.fullmatch(r"M(\d{3,})", canonical))
        ]
        largest_existing = max(existing_numbers, default=0)
        for item in checked.milestones:
            if item.canonical_id in existing_ids:
                continue
            match = re.fullmatch(r"M(\d{3,})", item.canonical_id)
            if match is None or int(match.group(1)) <= largest_existing:
                raise ValueError("new future Milestones require fresh monotonic IDs")
        return identity, current, checked

    def validate_milestone_review(
        self,
        *,
        run_id: str,
        milestone_canonical_id: str,
        decision: MilestoneReviewDecision,
        reason: str,
        future_plan: PlanSpec | None = None,
        require_verified: bool = True,
    ) -> PlanSpec | None:
        """Validate the model decision before its Provider result enters the WAL."""

        if not reason.strip():
            raise ValueError("MilestoneReview requires a non-empty reason")
        with self.database.transaction() as conn:
            _, _, checked = self._validated_milestone_review_conn(
                conn,
                run_id=run_id,
                milestone_canonical_id=milestone_canonical_id,
                decision=decision,
                future_plan=future_plan,
                require_verified=require_verified,
            )
        return checked

    def record_milestone_review(
        self,
        *,
        run_id: str,
        milestone_canonical_id: str,
        decision: MilestoneReviewDecision,
        reason: str,
        revision_id: str,
        source_event_id: str,
        evidence_event_ids: Sequence[str] = (),
        future_plan: PlanSpec | None = None,
    ) -> str:
        self._require_durable_event(source_event_id)
        if not reason.strip():
            raise ValueError("MilestoneReview requires a non-empty reason")
        projection = self._projection()
        with self.database.transaction() as conn:
            existing = conn.execute(
                "SELECT review_id FROM v2_milestone_review_events "
                "WHERE run_id=? AND source_event_id=? LIMIT 1",
                (run_id, source_event_id),
            ).fetchone()
            if existing is not None:
                return str(existing["review_id"])
            identity, current, checked = self._validated_milestone_review_conn(
                conn,
                run_id=run_id,
                milestone_canonical_id=milestone_canonical_id,
                decision=decision,
                future_plan=future_plan,
            )
            resulting = current.plan_version_id
            if checked is not None:
                application = self._apply_plan_conn(
                    conn,
                    projection,
                    run_id=run_id,
                    revision_id=revision_id,
                    plan=checked,
                    source_event_id=source_event_id,
                    requested_current=milestone_canonical_id,
                )
                resulting = application.plan_version_id
                self._link_task_requirements_conn(
                    conn,
                    run_id=run_id,
                    plan_version_id=resulting,
                    source_event_id=source_event_id,
                )
            cursor = self._next_cursor(conn, run_id)
            review_id = stable_id(
                "review_",
                {
                    "run": run_id,
                    "milestone": identity["identity_id"],
                    "decision": decision.value,
                    "source": source_event_id,
                },
            )
            conn.execute(
                "INSERT OR IGNORE INTO v2_milestone_review_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    review_id,
                    run_id,
                    identity["identity_id"],
                    decision.value,
                    reason,
                    json.dumps(tuple(evidence_event_ids)),
                    current.plan_version_id,
                    resulting,
                    source_event_id,
                    revision_id,
                    cursor,
                    utc_now(),
                ),
            )
            return review_id

    def _validated_milestone_failure_review_conn(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        milestone_canonical_id: str,
        reason: str,
        revision_id: str,
        failure_signature: str,
        failure_criterion_ids: Sequence[str],
        corrective_steps: Sequence[PlanStepSpec],
    ) -> tuple[CurrentMilestone, tuple[str, ...], str | None, tuple[str, ...]]:
        if not reason.strip() or not failure_signature.strip():
            raise ValueError("Milestone failure review requires a causal diagnosis and signature")
        if not corrective_steps:
            raise ValueError("CORRECT_CURRENT requires at least one corrective Step")
        current = self.current(run_id, conn=conn)
        if current.canonical_id != milestone_canonical_id or (
            current.status != MilestoneStatus.VERIFICATION_FAILED.value
        ):
            raise ValueError("failure review must target the current VERIFICATION_FAILED Milestone")
        failed_criterion_order = tuple(dict.fromkeys(map(str, failure_criterion_ids)))
        failed_criteria = frozenset(failed_criterion_order)
        known_criteria = {
            str(row["local_criterion_id"])
            for row in conn.execute(
                "SELECT local_criterion_id FROM v2_completion_criteria "
                "WHERE run_id=? AND milestone_identity_id=?",
                (run_id, current.identity_id),
            ).fetchall()
        }
        if not failed_criteria or not failed_criteria.issubset(known_criteria):
            raise ValueError("failure review contains unknown Milestone criteria")
        receipt = conn.execute(
            "SELECT failed_criterion_ids_json,evidence_event_ids_json,failure_signature "
            "FROM v2_milestone_acceptance_receipts "
            "WHERE run_id=? AND milestone_identity_id=? AND revision_id=? "
            "AND verdict='VERIFICATION_FAILED' ORDER BY created_cursor DESC LIMIT 1",
            (run_id, current.identity_id, revision_id),
        ).fetchone()
        if receipt is None:
            raise ValueError("failure review requires an authoritative acceptance receipt")
        receipt_failed = frozenset(map(str, json.loads(str(receipt["failed_criterion_ids_json"]))))
        if str(receipt["failure_signature"] or "") != failure_signature:
            raise ValueError("failure review signature does not match the acceptance receipt")
        if failed_criteria != receipt_failed:
            raise ValueError("failure review cannot weaken or expand the failed Criterion set")
        receipt_evidence_event_ids = tuple(
            map(str, json.loads(str(receipt["evidence_event_ids_json"])))
        )
        mapped = set().union(*(set(step.criterion_ids) for step in corrective_steps))
        if not failed_criteria.intersection(mapped):
            raise ValueError(
                "corrective Steps must contribute to at least one failed Milestone criterion"
            )
        for step in corrective_steps:
            unknown = set(step.criterion_ids).difference(known_criteria)
            if unknown:
                raise ValueError(
                    f"corrective Step maps unknown Milestone criteria: {sorted(unknown)}"
                )
            if not step.expected_outcome.strip() or not step.failure_signals:
                raise ValueError("corrective Step requires an expected outcome and bounded risks")
        self._reject_duplicate_failure_contract_conn(
            conn,
            run_id=run_id,
            milestone_identity_id=current.identity_id,
            revision_id=revision_id,
            failure_signature=failure_signature,
        )
        self._reject_equivalent_corrective_chain_conn(
            conn,
            run_id=run_id,
            milestone_identity_id=current.identity_id,
            revision_id=revision_id,
            corrective_steps=corrective_steps,
        )
        causal_rows = conn.execute(
            "SELECT step_identity_id,criterion_ids_json FROM v2_plan_step_identities "
            "WHERE run_id=? AND milestone_identity_id=? ORDER BY created_cursor DESC",
            (run_id, current.identity_id),
        ).fetchall()
        causal_step_identity_id = next(
            (
                str(row["step_identity_id"])
                for row in causal_rows
                if failed_criteria.intersection(
                    map(str, json.loads(str(row["criterion_ids_json"])))
                )
            ),
            None,
        )
        return (
            current,
            failed_criterion_order,
            causal_step_identity_id,
            receipt_evidence_event_ids,
        )

    def validate_milestone_failure_review(
        self,
        *,
        run_id: str,
        milestone_canonical_id: str,
        reason: str,
        revision_id: str,
        failure_signature: str,
        failure_criterion_ids: Sequence[str],
        corrective_steps: Sequence[PlanStepSpec],
    ) -> None:
        """Validate the same transition later committed after its WAL event."""

        with self.database.transaction() as conn:
            self._validated_milestone_failure_review_conn(
                conn,
                run_id=run_id,
                milestone_canonical_id=milestone_canonical_id,
                reason=reason,
                revision_id=revision_id,
                failure_signature=failure_signature,
                failure_criterion_ids=failure_criterion_ids,
                corrective_steps=corrective_steps,
            )

    def record_milestone_failure_review(
        self,
        *,
        run_id: str,
        milestone_canonical_id: str,
        reason: str,
        revision_id: str,
        source_event_id: str,
        failure_signature: str,
        failure_criterion_ids: Sequence[str],
        evidence_event_ids: Sequence[str],
        corrective_steps: Sequence[PlanStepSpec],
    ) -> tuple[str, tuple[str, ...]]:
        """Atomically turn one failed Milestone diagnosis into real route Steps."""

        self._require_durable_event(source_event_id)
        projection = self._projection()
        with self.database.transaction() as conn:
            existing = conn.execute(
                "SELECT review_id,created_step_ids_json "
                "FROM v2_milestone_failure_review_events WHERE run_id=? AND source_event_id=?",
                (run_id, source_event_id),
            ).fetchone()
            if existing is not None:
                return str(existing["review_id"]), tuple(
                    json.loads(str(existing["created_step_ids_json"]))
                )
            (
                current,
                failed_criterion_order,
                causal_step_identity_id,
                receipt_evidence_event_ids,
            ) = self._validated_milestone_failure_review_conn(
                conn,
                run_id=run_id,
                milestone_canonical_id=milestone_canonical_id,
                reason=reason,
                revision_id=revision_id,
                failure_signature=failure_signature,
                failure_criterion_ids=failure_criterion_ids,
                corrective_steps=corrective_steps,
            )
            if evidence_event_ids and tuple(dict.fromkeys(map(str, evidence_event_ids))) != (
                receipt_evidence_event_ids
            ):
                raise ValueError("failure review Evidence must match the acceptance receipt")
            baseline_cursor = int(
                conn.execute(
                    "SELECT next_cursor FROM v2_registry_runs WHERE run_id=?",
                    (run_id,),
                ).fetchone()["next_cursor"]
            )
            created_ids: list[str] = []
            created_identity_ids: list[str] = []
            repository_id, branch_id = self._projection_scope_conn(conn, run_id)
            for proposal in corrective_steps:
                generated_id = self._next_step_id_conn(
                    conn,
                    run_id,
                    current.canonical_id,
                    corrective=True,
                )
                generated = replace(
                    proposal,
                    step_id=generated_id,
                    status=PlanStepStatus.PENDING,
                    corrective=True,
                )
                identity_id = self._ensure_step_conn(
                    conn,
                    run_id=run_id,
                    milestone_identity_id=current.identity_id,
                    step=generated,
                    revision_id=revision_id,
                    source_event_id=source_event_id,
                )
                created_ids.append(generated_id)
                created_identity_ids.append(identity_id)
                step_cursor = self._next_cursor(conn, run_id)
                projection.project_plan_step(
                    conn,
                    repository_id=repository_id,
                    run_id=run_id,
                    branch_id=branch_id,
                    revision_id=revision_id,
                    plan_version_id=current.plan_version_id,
                    milestone_identity_id=current.identity_id,
                    step_identity_id=identity_id,
                    step=generated,
                    source_event_id=source_event_id,
                    cursor=step_cursor,
                )
                correction_cursor = self._next_cursor(conn, run_id)
                conn.execute(
                    "INSERT INTO v2_corrective_steps("
                    "corrective_id,run_id,milestone_identity_id,step_identity_id,"
                    "criterion_identity_id,reason,evidence_event_ids_json,source_event_id,"
                    "revision_id,failure_signature,failure_criterion_ids_json,"
                    "baseline_revision_id,baseline_cursor,created_cursor,created_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        stable_id("corrective_", {"step": identity_id, "source": source_event_id}),
                        run_id,
                        current.identity_id,
                        identity_id,
                        None,
                        reason.strip(),
                        json.dumps(receipt_evidence_event_ids),
                        source_event_id,
                        revision_id,
                        failure_signature,
                        json.dumps(failed_criterion_order),
                        revision_id,
                        baseline_cursor,
                        correction_cursor,
                        utc_now(),
                    ),
                )
                if causal_step_identity_id is not None:
                    projection.project_step_correction(
                        conn,
                        repository_id=repository_id,
                        run_id=run_id,
                        branch_id=branch_id,
                        revision_id=revision_id,
                        plan_version_id=current.plan_version_id,
                        failed_step_identity_id=causal_step_identity_id,
                        corrective_step_identity_id=identity_id,
                        source_event_id=source_event_id,
                        cursor=correction_cursor,
                    )
            state_event = self._record_milestone_state_conn(
                conn,
                run_id=run_id,
                identity_id=current.identity_id,
                status=MilestoneStatus.REPAIRING,
                revision_id=revision_id,
                source_event_id=source_event_id,
            )
            review_cursor = self._next_cursor(conn, run_id)
            review_id = stable_id(
                "milfailreview_",
                {
                    "milestone": current.identity_id,
                    "failure": failure_signature,
                    "source": source_event_id,
                },
            )
            conn.execute(
                "INSERT INTO v2_milestone_failure_review_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    review_id,
                    run_id,
                    current.identity_id,
                    MilestoneReviewDecision.CORRECT_CURRENT.value,
                    reason.strip(),
                    failure_signature,
                    json.dumps(failed_criterion_order),
                    json.dumps(receipt_evidence_event_ids),
                    json.dumps(tuple(created_ids)),
                    source_event_id,
                    revision_id,
                    baseline_cursor,
                    review_cursor,
                    utc_now(),
                ),
            )
            self._project_current_step_conn(
                conn,
                run_id=run_id,
                revision_id=revision_id,
                source_event_id=source_event_id,
                cursor=max(
                    review_cursor,
                    state_event.cursor if state_event is not None else review_cursor,
                ),
            )
            return review_id, tuple(created_ids)

    def milestone_statuses(self, run_id: str) -> dict[str, str]:
        rows = self.database.connection.execute(
            "SELECT mi.canonical_id,(SELECT mse.status FROM v2_milestone_state_events mse "
            "WHERE mse.identity_id=mi.identity_id ORDER BY mse.created_cursor DESC LIMIT 1) status "
            "FROM v2_plan_milestones pm "
            "JOIN v2_milestone_identities mi ON mi.identity_id=pm.identity_id "
            "WHERE pm.plan_version_id=(SELECT edge.plan_version_id FROM v2_semantic_edges edge "
            "WHERE edge.run_id=? AND edge.edge_type='CURRENT_MILESTONE' "
            "AND edge.valid_to_cursor IS NULL) ORDER BY pm.ordinal",
            (run_id,),
        ).fetchall()
        return {str(row["canonical_id"]): str(row["status"]) for row in rows}

    def completion_criteria(self, run_id: str, canonical_id: str) -> tuple[dict[str, object], ...]:
        rows = self.database.connection.execute(
            "SELECT c.* FROM v2_completion_criteria c JOIN v2_milestone_identities mi "
            "ON mi.identity_id=c.milestone_identity_id WHERE c.run_id=? AND mi.canonical_id=? "
            "AND c.milestone_version_id=(SELECT pm.milestone_version_id FROM v2_plan_milestones pm "
            "JOIN v2_semantic_edges edge ON edge.plan_version_id=pm.plan_version_id "
            "WHERE edge.run_id=? AND edge.edge_type='CURRENT_MILESTONE' "
            "AND edge.valid_to_cursor IS NULL AND pm.identity_id=mi.identity_id) "
            "ORDER BY c.local_criterion_id",
            (run_id, canonical_id, run_id),
        ).fetchall()
        return tuple(self._criterion_row_to_dict(row) for row in rows)

    @staticmethod
    def _criterion_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
        required_types = tuple(json.loads(str(row["required_evidence_types_json"])))
        requirement_id = str(row["requirement_id"]) or str(row["local_criterion_id"])
        claim_type = str(row["claim_type"])
        commitment_level = str(row["commitment_level"])
        stored_mode = str(row["verification_mode"]) if "verification_mode" in row.keys() else ""
        if stored_mode:
            verification_mode = stored_mode
        else:
            try:
                verification_mode = default_verification_mode(
                    ClaimType(claim_type),
                    requirement_id=requirement_id,
                    required_evidence_types=tuple(FactType(str(item)) for item in required_types),
                    commitment_level=CommitmentLevel(commitment_level),
                ).value
            except ValueError:
                verification_mode = CriterionVerificationMode.EXECUTABLE.value
        return {
            "criterion_id": str(row["local_criterion_id"]),
            "criterion_identity_id": str(row["criterion_identity_id"]),
            "observable_outcome": str(row["observable_outcome"]),
            "required_evidence_types": required_types,
            "entity_refs": tuple(json.loads(str(row["entity_refs_json"]))),
            "test_selectors": tuple(json.loads(str(row["test_selectors_json"]))),
            "required": bool(row["required"]),
            "requirement_id": requirement_id,
            "requirement_text": (str(row["requirement_text"]) or str(row["observable_outcome"])),
            "claim_type": claim_type,
            "commitment_level": commitment_level,
            "verification_mode": verification_mode,
        }

    def native_plan_completed_step_ids(
        self,
        run_id: str,
        canonical_id: str,
    ) -> frozenset[str]:
        """Steps the model's own native Plan has marked completed.

        Only ``turn/plan/updated`` contiguous completions count.  Durable
        actions also advance the lightweight route pointer, but they are the
        runtime's navigation inference, not a statement by the model.
        """

        rows = self.database.connection.execute(
            "SELECT si.canonical_step_id FROM v2_focus_span_observation_events span, "
            "json_each(span.covered_step_identity_ids_json) covered "
            "JOIN v2_plan_step_identities si "
            "ON si.step_identity_id=CAST(covered.value AS TEXT) "
            "JOIN v2_milestone_identities mi ON mi.identity_id=span.milestone_identity_id "
            "WHERE span.run_id=? AND mi.canonical_id=? "
            "AND span.observation_basis='NATIVE_PLAN_CONTIGUOUS_COMPLETION'",
            (run_id, canonical_id),
        ).fetchall()
        return frozenset(str(row["canonical_step_id"]) for row in rows)

    def milestone_steps(self, run_id: str, canonical_id: str) -> tuple[dict[str, object], ...]:
        rows = self.database.connection.execute(
            "SELECT si.step_identity_id,si.milestone_identity_id,si.canonical_step_id,"
            "si.title,si.corrective,si.created_cursor,si.criterion_ids_json,"
            "si.entity_refs_json,si.historical_dependency_refs_json,"
            "si.source_plan_item_ids_json,"
            "sc.expected_outcome,sc.minimum_acceptance_json,"
            "sc.failure_signals_json,sc.contract_revision_number,"
            "(SELECT sse.status FROM v2_plan_step_state_events sse "
            "WHERE sse.step_identity_id=si.step_identity_id "
            "ORDER BY sse.created_cursor DESC LIMIT 1) status "
            "FROM v2_plan_step_identities si "
            "JOIN v2_effective_plan_step_contracts sc "
            "ON sc.step_identity_id=si.step_identity_id "
            "JOIN v2_milestone_identities mi ON mi.identity_id=si.milestone_identity_id "
            "WHERE si.run_id=? AND mi.canonical_id=? ORDER BY si.created_cursor",
            (run_id, canonical_id),
        ).fetchall()
        return tuple(
            {
                "step_identity_id": str(row["step_identity_id"]),
                "milestone_identity_id": str(row["milestone_identity_id"]),
                "step_id": str(row["canonical_step_id"]),
                "title": str(row["title"]),
                "status": str(row["status"]),
                "corrective": bool(row["corrective"]),
                "created_cursor": int(row["created_cursor"]),
                "criterion_ids": tuple(json.loads(str(row["criterion_ids_json"]))),
                "entity_refs": tuple(json.loads(str(row["entity_refs_json"]))),
                "historical_dependency_refs": tuple(
                    json.loads(str(row["historical_dependency_refs_json"]))
                ),
                "source_plan_item_ids": tuple(json.loads(str(row["source_plan_item_ids_json"]))),
                "expected_outcome": str(row["expected_outcome"]),
                "minimum_acceptance": tuple(json.loads(str(row["minimum_acceptance_json"]))),
                "failure_signals": tuple(json.loads(str(row["failure_signals_json"]))),
                "contract_revision_number": int(row["contract_revision_number"]),
            }
            for row in rows
        )

    def switch_current(
        self,
        *,
        run_id: str,
        canonical_id: str,
        revision_id: str,
        source_event_id: str,
        plan_version_id: str | None = None,
    ) -> CurrentMilestone:
        if not all(value.strip() for value in (run_id, canonical_id, revision_id, source_event_id)):
            raise ValueError("Milestone switch requires identity, revision, and source Event")
        self._require_durable_event(source_event_id)
        projection = self._projection()
        with self.database.transaction() as conn:
            self._switch_current_conn(
                conn,
                projection,
                run_id=run_id,
                canonical_id=canonical_id,
                revision_id=revision_id,
                source_event_id=source_event_id,
                plan_version_id=plan_version_id,
            )
            return self.current(run_id, conn=conn)

    def switch_to_released_work(
        self,
        *,
        run_id: str,
        canonical_id: str,
        revision_id: str,
        source_event_id: str,
        plan_version_id: str | None = None,
    ) -> CurrentMilestone:
        '''Switch to an already-planned official release without claiming its predecessor.

        A repository-stream queue is an external release boundary. It may
        expose the next official milestone while an internal review receipt
        for the prior release is still incomplete. The prior milestone stays
        in its recorded state; this method only moves the navigation cursor
        and never marks either milestone verified.
        '''
        if not all(value.strip() for value in (run_id, canonical_id, revision_id, source_event_id)):
            raise ValueError("released-work switch requires non-empty identity fields")
        self._require_durable_event(source_event_id)
        projection = self._projection()
        with self.database.transaction() as conn:
            self._switch_current_conn(
                conn,
                projection,
                run_id=run_id,
                canonical_id=canonical_id,
                revision_id=revision_id,
                source_event_id=source_event_id,
                plan_version_id=plan_version_id,
                require_future_review=False,
                external_requirement_release=True,
            )
            return self.current(run_id, conn=conn)

    def advance_to_next_ready(
        self,
        *,
        run_id: str,
        revision_id: str,
        source_event_id: str,
    ) -> tuple[CurrentMilestone, bool]:
        """Advance the route directly after trusted Milestone acceptance.

        This is a runtime reducer, not a model review.  It is intentionally
        separate from :meth:`switch_current`, whose public compatibility
        contract still protects manual/replan callers with the legacy review
        check.
        """

        self._require_durable_event(source_event_id)
        projection = self._projection()
        with self.database.transaction() as conn:
            current = self.current(run_id, conn=conn)
            if current.status != MilestoneStatus.COMPLETED_VERIFIED.value:
                return current, False
            ordinal_row = conn.execute(
                "SELECT ordinal FROM v2_plan_milestones WHERE plan_version_id=? AND identity_id=?",
                (current.plan_version_id, current.identity_id),
            ).fetchone()
            if ordinal_row is None:
                raise RuntimeError("current Milestone is missing from its PlanVersion")
            rows = conn.execute(
                "SELECT pm.identity_id,mi.canonical_id,pm.ordinal "
                "FROM v2_plan_milestones pm "
                "JOIN v2_milestone_identities mi ON mi.identity_id=pm.identity_id "
                "WHERE pm.plan_version_id=? AND pm.ordinal>? ORDER BY pm.ordinal",
                (current.plan_version_id, int(ordinal_row["ordinal"])),
            ).fetchall()
            target = None
            for row in rows:
                status = self._latest_milestone_status_conn(conn, str(row["identity_id"]))
                if status in {
                    MilestoneStatus.COMPLETED_VERIFIED,
                    MilestoneStatus.CANCELLED,
                }:
                    continue
                dependencies = conn.execute(
                    "SELECT target_identity_id FROM v2_milestone_dependencies "
                    "WHERE plan_version_id=? AND source_identity_id=?",
                    (current.plan_version_id, str(row["identity_id"])),
                ).fetchall()
                if all(
                    self._latest_milestone_status_conn(conn, str(dep["target_identity_id"]))
                    is MilestoneStatus.COMPLETED_VERIFIED
                    for dep in dependencies
                ):
                    target = row
                    break
            if target is None:
                return current, False
            self._switch_current_conn(
                conn,
                projection,
                run_id=run_id,
                canonical_id=str(target["canonical_id"]),
                revision_id=revision_id,
                source_event_id=source_event_id,
                plan_version_id=current.plan_version_id,
                require_future_review=False,
            )
            return self.current(run_id, conn=conn), True

    def _switch_current_conn(
        self,
        conn: sqlite3.Connection,
        projection: PlanProjection,
        *,
        run_id: str,
        canonical_id: str,
        revision_id: str,
        source_event_id: str,
        plan_version_id: str | None,
        cursor: int | None = None,
        require_future_review: bool = True,
        external_requirement_release: bool = False,
    ) -> None:
        if external_requirement_release and not source_event_id.startswith("stream_"):
            raise ValueError("external route switch requires durable stream provenance")
        task = conn.execute("SELECT * FROM v2_tasks WHERE run_id=?", (run_id,)).fetchone()
        if task is None:
            raise KeyError(f"unknown run: {run_id}")
        if plan_version_id is None:
            row = conn.execute(
                "SELECT plan_version_id FROM v2_plan_versions WHERE task_id=? "
                "ORDER BY version_number DESC LIMIT 1",
                (task["task_id"],),
            ).fetchone()
            if row is None:
                raise RuntimeError("Run has no PlanVersion")
            plan_version_id = row["plan_version_id"]
        target = conn.execute(
            "SELECT mi.identity_id FROM v2_plan_milestones pm "
            "JOIN v2_milestone_identities mi ON mi.identity_id=pm.identity_id "
            "WHERE pm.plan_version_id=? AND mi.canonical_id=? AND mi.run_id=?",
            (plan_version_id, canonical_id, run_id),
        ).fetchone()
        if target is None:
            raise ValueError("current Milestone must belong to the selected PlanVersion")

        active = conn.execute(
            "SELECT edge.target_id AS identity_id FROM v2_semantic_edges edge "
            "WHERE edge.run_id=? AND edge.edge_type='CURRENT_MILESTONE' "
            "AND edge.valid_to_cursor IS NULL",
            (run_id,),
        ).fetchone()
        if active is not None and str(active["identity_id"]) != str(target["identity_id"]):
            active_id = str(active["identity_id"])
            active_status = self._latest_milestone_status_conn(conn, active_id)
            if active_status is not MilestoneStatus.COMPLETED_VERIFIED and not external_requirement_release:
                raise ValueError(
                    "cannot switch Milestone before the current one is completion-verified"
                )
            if require_future_review:
                review = conn.execute(
                    "SELECT 1 FROM v2_milestone_review_events "
                    "WHERE run_id=? AND milestone_identity_id=? "
                    "AND resulting_plan_version_id=? "
                    "AND created_cursor>(SELECT MAX(mse.created_cursor) "
                    "FROM v2_milestone_state_events mse "
                    "WHERE mse.identity_id=? AND mse.status='COMPLETED_VERIFIED') "
                    "ORDER BY created_cursor DESC LIMIT 1",
                    (run_id, active_id, plan_version_id, active_id),
                ).fetchone()
                if review is None:
                    raise ValueError(
                        "cannot switch Milestone before post-verification future-plan review"
                    )
            # A repository-stream release is an external, durable route
            # authority. Its queue may release the next official milestone
            # before the previous native-plan node has received an internal
            # semantic review receipt. The provenance check above still
            # protects this escape hatch; applying the native dependency gate
            # here would deadlock the same release on every recovery attempt.
            # Ordinary/manual navigation keeps the dependency gate.
            if not external_requirement_release:
                dependencies = conn.execute(
                    "SELECT target_identity_id FROM v2_milestone_dependencies "
                    "WHERE plan_version_id=? AND source_identity_id=?",
                    (plan_version_id, target["identity_id"]),
                ).fetchall()
                for dependency in dependencies:
                    if (
                        self._latest_milestone_status_conn(conn, str(dependency["target_identity_id"]))
                        is not MilestoneStatus.COMPLETED_VERIFIED
                    ):
                        raise ValueError("cannot switch to a Milestone with unverified dependencies")
        cursor = cursor if cursor is not None else self._next_cursor(conn, run_id)

        conn.execute(
            "UPDATE v2_current_milestones SET valid_to_revision=?, valid_to_cursor=? "
            "WHERE run_id=? AND valid_to_cursor IS NULL",
            (revision_id, cursor, run_id),
        )
        relation_id = stable_id(
            "current_",
            {
                "run": run_id,
                "plan": plan_version_id,
                "milestone": target["identity_id"],
                "cursor": cursor,
            },
        )
        conn.execute(
            "INSERT INTO v2_current_milestones VALUES(?,?,?,?,?,?,?,?,?)",
            (
                relation_id,
                run_id,
                target["identity_id"],
                plan_version_id,
                source_event_id,
                revision_id,
                None,
                cursor,
                None,
            ),
        )

        conn.execute(
            "UPDATE v2_working_set_memberships "
            "SET valid_to_revision=?, valid_to_cursor=? "
            "WHERE run_id=? AND valid_to_cursor IS NULL",
            (revision_id, cursor, run_id),
        )
        rows = conn.execute(
            "SELECT pm.identity_id, pm.ordinal, mi.canonical_id FROM v2_plan_milestones pm "
            "JOIN v2_milestone_identities mi ON mi.identity_id=pm.identity_id "
            "WHERE pm.plan_version_id=? ORDER BY pm.ordinal",
            (plan_version_id,),
        ).fetchall()
        current_ordinal = next(
            int(row["ordinal"]) for row in rows if row["identity_id"] == target["identity_id"]
        )
        dependency_ids = {
            row["target_identity_id"]
            for row in conn.execute(
                "SELECT target_identity_id FROM v2_milestone_dependencies "
                "WHERE plan_version_id=? AND source_identity_id=?",
                (plan_version_id, target["identity_id"]),
            ).fetchall()
        }
        next_ids = {
            row["identity_id"]
            for row in rows
            if current_ordinal < int(row["ordinal"]) <= current_ordinal + self.prefetch_limit
        }
        retained_ids = {
            row["identity_id"]
            for row in rows
            if current_ordinal - self.retain_predecessor_milestones
            <= int(row["ordinal"])
            < current_ordinal
        }
        for row in rows:
            identity_id = row["identity_id"]
            if identity_id == target["identity_id"]:
                state, reason = "HOT", "CURRENT_MILESTONE"
            elif identity_id in dependency_ids:
                state, reason = "HOT", "EXPLICIT_DEPENDENCY"
            elif identity_id in retained_ids:
                state, reason = "HOT", "PREDECESSOR_RETAINED"
            elif identity_id in next_ids:
                state, reason = "PREFETCH", "BOUNDED_NEXT_MILESTONE"
            else:
                state, reason = "COLD", "OUTSIDE_WORKING_ROOTS"
            membership_id = stable_id(
                "ws_",
                {
                    "run": run_id,
                    "plan": plan_version_id,
                    "identity": identity_id,
                    "cursor": cursor,
                    "state": state,
                },
            )
            conn.execute(
                "INSERT INTO v2_working_set_memberships VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    membership_id,
                    run_id,
                    plan_version_id,
                    identity_id,
                    state,
                    reason,
                    source_event_id,
                    revision_id,
                    None,
                    cursor,
                    None,
                ),
            )

        projection.project_current_milestone(
            conn,
            repository_id=task["repository_id"],
            run_id=run_id,
            branch_id=task["branch_id"],
            revision_id=revision_id,
            plan_version_id=plan_version_id,
            milestone_identity_id=target["identity_id"],
            source_event_id=source_event_id,
            cursor=cursor,
        )
        step = self.current_step(run_id, conn=conn)
        projection.project_current_step(
            conn,
            repository_id=task["repository_id"],
            run_id=run_id,
            branch_id=task["branch_id"],
            revision_id=revision_id,
            plan_version_id=plan_version_id,
            milestone_identity_id=target["identity_id"],
            step_identity_id=(str(step["step_identity_id"]) if step is not None else None),
            source_event_id=source_event_id,
            cursor=cursor,
        )

    def current(self, run_id: str, *, conn: sqlite3.Connection | None = None) -> CurrentMilestone:
        sql = (
            "SELECT edge.plan_version_id,edge.target_id AS identity_id,"
            "cm.source_event_id,edge.valid_from_revision,edge.valid_from_cursor,"
            "mi.canonical_id, mv.version_id, mv.title, "
            "COALESCE((SELECT mse.status FROM v2_milestone_state_events mse "
            " WHERE mse.identity_id=edge.target_id ORDER BY mse.created_cursor DESC LIMIT 1), "
            " mv.status) AS execution_status "
            "FROM v2_semantic_edges edge "
            "JOIN v2_current_milestones cm ON cm.run_id=edge.run_id "
            " AND cm.identity_id=edge.target_id AND cm.plan_version_id=edge.plan_version_id "
            " AND cm.valid_to_cursor IS NULL "
            "JOIN v2_milestone_identities mi ON mi.identity_id=edge.target_id "
            "JOIN v2_plan_milestones pm ON pm.plan_version_id=edge.plan_version_id "
            " AND pm.identity_id=edge.target_id "
            "JOIN v2_milestone_versions mv ON mv.version_id=pm.milestone_version_id "
            "WHERE edge.run_id=? AND edge.edge_type='CURRENT_MILESTONE' "
            "AND edge.valid_to_cursor IS NULL"
        )
        row = (conn or self.database.connection).execute(sql, (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Run has no current Milestone: {run_id}")
        return CurrentMilestone(
            run_id=run_id,
            plan_version_id=row["plan_version_id"],
            identity_id=row["identity_id"],
            canonical_id=row["canonical_id"],
            version_id=row["version_id"],
            title=row["title"],
            status=row["execution_status"],
            source_event_id=row["source_event_id"],
            valid_from_revision=row["valid_from_revision"],
            valid_from_cursor=int(row["valid_from_cursor"]),
        )

    def working_set_roots(self, run_id: str) -> tuple[WorkingSetRoot, ...]:
        rows = self.database.connection.execute(
            "SELECT ws.*, mi.canonical_id, pm.ordinal "
            "FROM v2_working_set_memberships ws "
            "JOIN v2_milestone_identities mi ON mi.identity_id=ws.identity_id "
            "JOIN v2_plan_milestones pm ON pm.plan_version_id=ws.plan_version_id "
            " AND pm.identity_id=ws.identity_id "
            "WHERE ws.run_id=? AND ws.valid_to_cursor IS NULL "
            " AND ws.state IN ('HOT', 'PREFETCH') "
            "ORDER BY CASE ws.state WHEN 'HOT' THEN 0 ELSE 1 END, pm.ordinal",
            (run_id,),
        ).fetchall()
        return tuple(
            WorkingSetRoot(
                identity_id=row["identity_id"],
                canonical_id=row["canonical_id"],
                state=row["state"],
                reason=row["reason"],
                ordinal=int(row["ordinal"]),
                source_event_id=row["source_event_id"],
                plan_version_id=row["plan_version_id"],
            )
            for row in rows
        )

    def version_counts(self, run_id: str) -> tuple[int, int]:
        row = self.database.connection.execute(
            "SELECT COUNT(DISTINCT pv.plan_version_id) AS plans, "
            "COUNT(DISTINCT mv.version_id) AS milestones "
            "FROM v2_tasks t "
            "LEFT JOIN v2_plan_versions pv ON pv.task_id=t.task_id "
            "LEFT JOIN v2_milestone_identities mi ON mi.task_id=t.task_id "
            "LEFT JOIN v2_milestone_versions mv ON mv.identity_id=mi.identity_id "
            "WHERE t.run_id=?",
            (run_id,),
        ).fetchone()
        return int(row["plans"]), int(row["milestones"])
