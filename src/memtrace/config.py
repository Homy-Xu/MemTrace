from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .context_runtime import ContextBudget
from .page_store import PagePolicy


class ConfigurationError(ValueError):
    """Raised before Planning when a V2 runtime combination is not runnable."""


_PROVIDER_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CODEX_SANDBOX_MODES = {"workspace-write", "danger-full-access"}


@dataclass(frozen=True, slots=True)
class CodexProviderConfiguration:
    """Credential-free Codex model Provider configuration.

    ``api_key_env`` is always an environment-variable name.  Literal secrets
    are rejected before the App Server starts and are never accepted as a
    compatibility format.
    """

    id: str = "openai"
    name: str = "OpenAI"
    base_url: str | None = None
    api_key_env: str | None = None
    wire_api: str = "responses"
    model: str | None = None
    model_context_window: int | None = None
    native_compaction_enabled: bool = False
    model_auto_compact_token_limit: int | None = None
    model_auto_compact_token_limit_scope: str = "total"
    compact_prompt: str | None = None

    @property
    def custom(self) -> bool:
        return self.id != "openai"

    def validate(self) -> None:
        if _PROVIDER_ID.fullmatch(self.id) is None:
            raise ConfigurationError("provider.id is not a valid Codex Provider identifier")
        if not self.name.strip():
            raise ConfigurationError("provider.name must be non-empty")
        if self.wire_api != "responses":
            raise ConfigurationError("Codex custom Providers require wire_api=responses")
        if self.model is not None and not self.model.strip():
            raise ConfigurationError("provider.model must be non-empty when provided")
        if self.model_context_window is not None and self.model_context_window < 16_000:
            raise ConfigurationError("provider.model_context_window must be at least 16000")
        if (
            self.model_auto_compact_token_limit is not None
            and self.model_auto_compact_token_limit <= 0
        ):
            raise ConfigurationError(
                "provider.model_auto_compact_token_limit must be positive when provided"
            )
        if self.model_auto_compact_token_limit_scope not in {"total", "body_after_prefix"}:
            raise ConfigurationError(
                "provider.model_auto_compact_token_limit_scope must be total or body_after_prefix"
            )
        if self.compact_prompt is not None and not self.compact_prompt.strip():
            raise ConfigurationError("provider.compact_prompt must be non-empty when provided")
        if not self.native_compaction_enabled and (
            self.model_auto_compact_token_limit is not None or self.compact_prompt is not None
        ):
            raise ConfigurationError(
                "provider native compaction settings require native_compaction_enabled=true"
            )
        if not self.custom:
            if self.base_url is not None or self.api_key_env is not None:
                raise ConfigurationError(
                    "the built-in openai Provider cannot be overridden; use a custom provider.id"
                )
            return
        if self.base_url is None:
            raise ConfigurationError("custom provider.base_url is required")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigurationError("provider.base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ConfigurationError(
                "provider.base_url must not contain credentials, query parameters, or a fragment"
            )
        if self.api_key_env is None or _ENVIRONMENT_NAME.fullmatch(self.api_key_env) is None:
            raise ConfigurationError(
                "custom provider.api_key_env must be an environment-variable name, never a key"
            )


@dataclass(frozen=True, slots=True)
class StageConfiguration:
    planning: bool = True
    page_store: bool = True
    semantic_memory: bool = True
    recall: bool = True
    context_runtime: bool = True
    rich_graph: bool = True

    def validate(self) -> None:
        required = {
            "planning": self.planning,
            "page_store": self.page_store,
            "semantic_memory": self.semantic_memory,
            "recall": self.recall,
            "context_runtime": self.context_runtime,
        }
        disabled = tuple(name for name, enabled in required.items() if not enabled)
        if disabled:
            raise ConfigurationError(
                "V2 formal runner requires the complete memory runtime; disabled: "
                + ", ".join(disabled)
            )


@dataclass(frozen=True, slots=True)
class RunBudgetConfiguration:
    """Per-run hard budget that ends a Task at a durable boundary.

    A benchmark harness usually kills a run that exceeds its slot; the kill
    leaves no ``result.json``, no terminal Memory Episode and no acceptance receipt.
    The runtime instead stops itself: the budget is checked at every Turn end
    (and the wall clock also inside a long Turn) and the current Milestone is
    closed with a ``ROUTE_STALLED`` receipt whose reason names the budget.
    ``None`` leaves a dimension unbounded.
    """

    max_execution_turns: int | None = None
    wall_clock_seconds: float | None = None

    def validate(self) -> None:
        if self.max_execution_turns is not None and self.max_execution_turns < 1:
            raise ConfigurationError("run_budget.max_execution_turns must be >= 1 when provided")
        if self.wall_clock_seconds is not None and not self.wall_clock_seconds > 0.0:
            raise ConfigurationError("run_budget.wall_clock_seconds must be positive when provided")


_ENGAGEMENT_MODES = frozenset({"full", "adaptive", "light", "passthrough"})


@dataclass(frozen=True, slots=True)
class EngagementConfiguration:
    """How deeply the runtime steers the model, decided from task pressure.

    ``full`` is the complete closure-spec behaviour and the default.
    ``adaptive`` starts at PASSTHROUGH or LIGHT for small/medium Plans and
    escalates on real pressure; ``light``/``passthrough`` pin the start level.
    See ``orchestration/engagement.py`` for the level semantics.
    """

    mode: str = "full"
    # PASSTHROUGH when the native Plan, the extracted requirement checklist and
    # the Task text are all at or below these sizes.
    passthrough_max_plan_items: int = 3
    passthrough_max_requirements: int = 4
    passthrough_max_task_chars: int = 1200
    # LIGHT when the native Plan is at or below this many items.
    light_max_plan_items: int = 8
    light_max_milestones: int = 3
    # Runtime escalation triggers (one level per trigger, never downward).
    escalate_on_provider_pressure: bool = True
    escalate_on_epoch: bool = True
    escalate_on_verification_failure: bool = True
    # Predecessor Milestones retained in Working Memory below FULL.
    retain_predecessor_milestones: int = 1

    def validate(self) -> None:
        if self.mode not in _ENGAGEMENT_MODES:
            raise ConfigurationError(
                "engagement.mode must be one of full, adaptive, light, passthrough"
            )
        for name in (
            "passthrough_max_plan_items",
            "passthrough_max_requirements",
            "passthrough_max_task_chars",
            "light_max_plan_items",
            "light_max_milestones",
        ):
            if int(getattr(self, name)) < 1:
                raise ConfigurationError(f"engagement.{name} must be >= 1")
        if self.retain_predecessor_milestones < 0:
            raise ConfigurationError("engagement.retain_predecessor_milestones cannot be negative")
        if self.passthrough_max_plan_items > self.light_max_plan_items:
            raise ConfigurationError(
                "engagement.passthrough_max_plan_items cannot exceed light_max_plan_items"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "EngagementConfiguration":
        if not value:
            return cls()
        base = cls()
        return cls(
            mode=str(value.get("mode", base.mode)),
            passthrough_max_plan_items=int(
                value.get("passthrough_max_plan_items", base.passthrough_max_plan_items)
            ),
            passthrough_max_requirements=int(
                value.get("passthrough_max_requirements", base.passthrough_max_requirements)
            ),
            passthrough_max_task_chars=int(
                value.get("passthrough_max_task_chars", base.passthrough_max_task_chars)
            ),
            light_max_plan_items=int(value.get("light_max_plan_items", base.light_max_plan_items)),
            light_max_milestones=int(value.get("light_max_milestones", base.light_max_milestones)),
            escalate_on_provider_pressure=bool(
                value.get("escalate_on_provider_pressure", base.escalate_on_provider_pressure)
            ),
            escalate_on_epoch=bool(value.get("escalate_on_epoch", base.escalate_on_epoch)),
            escalate_on_verification_failure=bool(
                value.get("escalate_on_verification_failure", base.escalate_on_verification_failure)
            ),
            retain_predecessor_milestones=int(
                value.get("retain_predecessor_milestones", base.retain_predecessor_milestones)
            ),
        )


@dataclass(frozen=True, slots=True)
class V2RuntimeConfig:
    """Validated configuration shared by generic and benchmark entry points."""

    schema_version: int = 2
    provider: CodexProviderConfiguration = field(default_factory=CodexProviderConfiguration)
    stages: StageConfiguration = field(default_factory=StageConfiguration)
    run_budget: RunBudgetConfiguration = field(default_factory=RunBudgetConfiguration)
    acceptance_budgets: Mapping[str, int] = field(default_factory=dict)
    engagement: EngagementConfiguration = field(default_factory=EngagementConfiguration)
    codex_sandbox_mode: str = "workspace-write"
    page_policy: PagePolicy = field(default_factory=PagePolicy)
    context_budget: ContextBudget = field(
        default_factory=lambda: ContextBudget(
            model_limit=32768,
            system_overhead=2048,
            tool_schema_tokens=2048,
            output_reserve=4096,
            safety_margin=2048,
        )
    )
    recall_max_pages: int = 8
    recall_max_tokens: int = 8192
    recall_max_page_bytes_read: int = 8 * 1024 * 1024
    recall_max_blob_bytes_read: int = 1024 * 1024
    recall_max_slice_tokens: int = 4096
    recall_max_recovered_block_tokens: int = 8192
    recall_max_context_admission_tokens: int = 8192
    rich_prefetch_budget: int = 4
    native_compaction_timeout_seconds: float = 180.0
    redaction_env_vars: tuple[str, ...] = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GITHUB_TOKEN",
    )

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ConfigurationError(f"unsupported V2 configuration schema: {self.schema_version}")
        self.provider.validate()
        self.stages.validate()
        self.run_budget.validate()
        self.engagement.validate()
        for name, value in self.acceptance_budgets.items():
            if name not in {
                "weak_progress",
                "no_progress",
                "semantic_review_rounds",
                "unclaimed_boundaries",
                "exploration_turns",
            }:
                raise ConfigurationError(f"unknown acceptance budget: {name}")
            if int(value) < 1:
                raise ConfigurationError(f"acceptance_budgets.{name} must be >= 1")
        if self.codex_sandbox_mode not in _CODEX_SANDBOX_MODES:
            raise ConfigurationError(
                "codex_sandbox_mode must be workspace-write or danger-full-access"
            )
        recall_budgets = (
            self.recall_max_pages,
            self.recall_max_tokens,
            self.recall_max_page_bytes_read,
            self.recall_max_blob_bytes_read,
            self.recall_max_slice_tokens,
            self.recall_max_recovered_block_tokens,
            self.recall_max_context_admission_tokens,
        )
        if any(value <= 0 for value in recall_budgets):
            raise ConfigurationError("Recall budgets must be positive")
        if self.rich_prefetch_budget < 0:
            raise ConfigurationError("Rich prefetch budget cannot be negative")
        if not 30.0 <= self.native_compaction_timeout_seconds <= 600.0:
            raise ConfigurationError("native_compaction_timeout_seconds must be between 30 and 600")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "V2RuntimeConfig":
        try:
            provider_value = dict(value.get("provider", {}))
            stages_value = dict(value.get("stages", {}))
            page_value = dict(value.get("page_policy", {}))
            budget_value = dict(value.get("context_budget", {}))
            run_budget_value = dict(value.get("run_budget", {}))
            acceptance_budgets_value = {
                str(key): int(item)
                for key, item in dict(value.get("acceptance_budgets", {})).items()
            }
            provider = CodexProviderConfiguration(
                id=str(provider_value.get("id", "openai")),
                name=str(provider_value.get("name", "OpenAI")),
                base_url=(
                    None
                    if provider_value.get("base_url") is None
                    else str(provider_value["base_url"]).rstrip("/")
                ),
                api_key_env=(
                    None
                    if provider_value.get("api_key_env") is None
                    else str(provider_value["api_key_env"])
                ),
                wire_api=str(provider_value.get("wire_api", "responses")),
                model=(
                    None if provider_value.get("model") is None else str(provider_value["model"])
                ),
                model_context_window=(
                    None
                    if provider_value.get("model_context_window") is None
                    else int(provider_value["model_context_window"])
                ),
                native_compaction_enabled=bool(
                    provider_value.get("native_compaction_enabled", False)
                ),
                model_auto_compact_token_limit=(
                    None
                    if provider_value.get("model_auto_compact_token_limit") is None
                    else int(provider_value["model_auto_compact_token_limit"])
                ),
                model_auto_compact_token_limit_scope=str(
                    provider_value.get("model_auto_compact_token_limit_scope", "total")
                ),
                compact_prompt=(
                    None
                    if provider_value.get("compact_prompt") is None
                    else str(provider_value["compact_prompt"])
                ),
            )
            redaction_env_vars = list(
                map(
                    str,
                    value.get(
                        "redaction_env_vars",
                        (
                            "OPENAI_API_KEY",
                            "ANTHROPIC_API_KEY",
                            "GITHUB_TOKEN",
                        ),
                    ),
                )
            )
            if provider.api_key_env is not None:
                redaction_env_vars.append(provider.api_key_env)
            config = cls(
                schema_version=int(value.get("schema_version", 2)),
                provider=provider,
                stages=StageConfiguration(
                    planning=bool(stages_value.get("planning", True)),
                    page_store=bool(stages_value.get("page_store", True)),
                    semantic_memory=bool(stages_value.get("semantic_memory", True)),
                    recall=bool(stages_value.get("recall", True)),
                    context_runtime=bool(stages_value.get("context_runtime", True)),
                    rich_graph=bool(stages_value.get("rich_graph", True)),
                ),
                run_budget=RunBudgetConfiguration(
                    max_execution_turns=(
                        None
                        if run_budget_value.get("max_execution_turns") is None
                        else int(run_budget_value["max_execution_turns"])
                    ),
                    wall_clock_seconds=(
                        None
                        if run_budget_value.get("wall_clock_seconds") is None
                        else float(run_budget_value["wall_clock_seconds"])
                    ),
                ),
                acceptance_budgets=acceptance_budgets_value,
                engagement=EngagementConfiguration.from_mapping(
                    dict(value.get("engagement", {}) or {})
                ),
                codex_sandbox_mode=str(value.get("codex_sandbox_mode", "workspace-write")),
                page_policy=PagePolicy(
                    min_tokens=int(page_value.get("min_tokens", 2048)),
                    target_tokens=int(page_value.get("target_tokens", 6144)),
                    nominal_max_tokens=int(page_value.get("nominal_max_tokens", 8192)),
                    absolute_max_tokens=int(page_value.get("absolute_max_tokens", 16384)),
                ),
                context_budget=ContextBudget(
                    model_limit=int(budget_value.get("model_limit", 32768)),
                    system_overhead=int(budget_value.get("system_overhead", 2048)),
                    tool_schema_tokens=int(budget_value.get("tool_schema_tokens", 2048)),
                    output_reserve=int(budget_value.get("output_reserve", 4096)),
                    safety_margin=int(budget_value.get("safety_margin", 2048)),
                    soft_ratio=float(budget_value.get("soft_ratio", 0.60)),
                    urgent_ratio=float(budget_value.get("urgent_ratio", 0.75)),
                    hard_ratio=float(budget_value.get("hard_ratio", 0.85)),
                ),
                recall_max_pages=int(value.get("recall_max_pages", 8)),
                recall_max_tokens=int(value.get("recall_max_tokens", 8192)),
                recall_max_page_bytes_read=int(
                    value.get("recall_max_page_bytes_read", 8 * 1024 * 1024)
                ),
                recall_max_blob_bytes_read=int(
                    value.get("recall_max_blob_bytes_read", 1024 * 1024)
                ),
                recall_max_slice_tokens=int(value.get("recall_max_slice_tokens", 4096)),
                recall_max_recovered_block_tokens=int(
                    value.get("recall_max_recovered_block_tokens", 8192)
                ),
                recall_max_context_admission_tokens=int(
                    value.get("recall_max_context_admission_tokens", 8192)
                ),
                rich_prefetch_budget=int(value.get("rich_prefetch_budget", 4)),
                native_compaction_timeout_seconds=float(
                    value.get("native_compaction_timeout_seconds", 180.0)
                ),
                redaction_env_vars=tuple(dict.fromkeys(redaction_env_vars)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid V2 configuration: {exc}") from exc
        config.validate()
        return config


def load_config(path: Path | None = None) -> V2RuntimeConfig:
    """Load JSON or JSON-compatible YAML without an optional YAML dependency.

    JSON documents are valid YAML 1.2 documents. Release ``.yaml`` files use
    that portable subset so the same strict parser runs on Linux and macOS.
    """

    if path is None:
        config = V2RuntimeConfig()
        config.validate()
        return config
    selected = Path(path).expanduser().resolve()
    try:
        value = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"configuration must be a JSON object (also valid YAML 1.2): {selected}"
        ) from exc
    if not isinstance(value, dict):
        raise ConfigurationError("configuration root must be an object")
    return V2RuntimeConfig.from_mapping(value)
