"""SWE-Milestone host-managed regression guard for the five-stage runtime.

The official SWE-Milestone protocol scores a submission only after the agent
tags ``agent-impl-<milestone>``; the agent receives no feedback.  This package
gives the five-stage acceptance kernel the same kind of runtime-owned
``trusted_verifier`` that SWE-EVO already has: it derives the affected scope
from the repository diff since the previous submission, runs the project's own
test runner for that scope offline, calibrates failures against the baseline
revision so pre-existing failures are never counted as regressions, and
refuses submissions whose root manifests or out-of-scope edits could not be
evaluated by the official evaluator.  It never sees hidden benchmark tests and
never edits the repository.
"""

from .adapter import SweMilestoneCodexHarnessAdapter
from .contract import SweMilestoneContract, load_contract
from .maven_reactor import MavenModule, MavenReactor, load_maven_reactor
from .verifier import SweMilestoneVerifier

__all__ = [
    "MavenModule",
    "MavenReactor",
    "SweMilestoneCodexHarnessAdapter",
    "SweMilestoneContract",
    "SweMilestoneVerifier",
    "load_contract",
    "load_maven_reactor",
]
