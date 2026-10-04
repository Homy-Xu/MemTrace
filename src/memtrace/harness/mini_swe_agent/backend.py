"""mini-swe-agent 2.4.6 integration.

The adapter uses mini-swe-agent's public ``get_model``, ``get_environment``
and ``get_agent`` constructors.  An instrumented DefaultAgent records model
and tool boundaries as provider-neutral events while the original trajectory
remains the authoritative usage source.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..base import EventQueueMixin, HarnessCheckpoint, HarnessSession, UsageSnapshot
from ..contracts import HarnessCapabilities, HarnessEvent, HarnessEventType

MINISWE_VERSION = "2.4.6"


def _digest(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()


class MiniSweAgentBackend(EventQueueMixin):
    """Run mini-swe-agent while feeding the same event/receipt contract."""

    def __init__(
        self,
        *,
        repository_path: Path,
        model: str,
        run_root: Path,
        agent_config: Mapping[str, Any] | None = None,
    ) -> None:
        self._init_event_queue()
        self.repository_path = Path(repository_path).resolve()
        self.model = model
        self.run_root = Path(run_root).resolve()
        self.agent_config = dict(agent_config or {})
        self._session: HarnessSession | None = None
        self._started_at = time.monotonic()
        self._sequence = 0
        self._last_usage = UsageSnapshot()
        self._last_trajectory: Path | None = None
        self._closed = False

    def capabilities(self) -> HarnessCapabilities:
        return HarnessCapabilities(
            harness_name="mini-swe-agent",
            model_name=self.model,
            tokenizer_id=None,
            context_limit=None,
            supports_plan_mode=False,
            supports_incremental_plan_updates=False,
            supports_thread_resume=False,
            supports_native_compaction=False,
            supports_token_usage_events=True,
            supports_tool_lifecycle_events=True,
            supports_context_replacement=False,
        )

    def start_session(self, *, run_id: str, branch_id: str, thread_id: str | None = None) -> HarnessSession:
        if thread_id:
            raise ValueError("mini-swe-agent does not support Codex thread IDs")
        actual = f"mini-{uuid.uuid4().hex}"
        self._session = HarnessSession(run_id, branch_id, actual, time.time(), self.capabilities())
        return self._session

    def plan(self, task: str, **_: Any) -> Mapping[str, Any]:
        """Return an explicit capability-aware route without hidden task facts."""
        if self._session is None:
            raise RuntimeError("start_session must be called before plan")
        return {
            "thread_id": self._session.thread_id,
            "native_plan": False,
            "reason": "mini-swe-agent exposes execution turns, not Codex native plan mode",
            "steps": [{"id": "M001", "title": "Execute the supplied task", "status": "pending"}],
            "task_digest": _digest(task),
        }

    def _event(self, event_type: HarnessEventType, payload: Mapping[str, Any]) -> HarnessEvent:
        if self._session is None:
            raise RuntimeError("start_session must be called first")
        self._sequence += 1
        event = HarnessEvent(
            harness_event_id=f"{self._session.run_id}:{self._sequence}:{event_type.value}",
            event_type=event_type,
            thread_id=self._session.thread_id,
            turn_id=f"turn-{self._sequence}",
            sequence=self._sequence,
            provider_time_ms=None,
            run_id=self._session.run_id,
            branch_id=self._session.branch_id,
            revision_id="unknown",
            source_event_id=f"mini:{self._sequence}",
            provider_method="mini-swe-agent",
            payload=dict(payload),
            raw_provider_summary={"harness": "mini-swe-agent", "version": MINISWE_VERSION},
        )
        return self._queue_event(event)

    def execute(self, task: str, **kwargs: Any) -> Iterable[HarnessEvent]:
        if self._session is None:
            raise RuntimeError("start_session must be called before execute")
        try:
            import minisweagent
            from minisweagent.agents import get_agent
            from minisweagent.environments import get_environment
            from minisweagent.models import get_model
        except ImportError as exc:
            raise RuntimeError(
                "mini-swe-agent 2.4.6 is required; install the mini_swe_agent extra"
            ) from exc
        version = str(getattr(minisweagent, "__version__", ""))
        if version and version != MINISWE_VERSION:
            raise RuntimeError(
                f"mini-swe-agent {MINISWE_VERSION} is required, found {version}"
            )

        config = dict(self.agent_config)
        model_config = dict(config.get("model", {}))
        model_config["model_name"] = self.model
        env_config = dict(config.get("environment", {}))
        env_config.setdefault("cwd", str(self.repository_path))
        agent_config = dict(config.get("agent", {}))
        output_path = Path(kwargs.get("trajectory_path", self.run_root / "trajectory.json"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        agent_config["output_path"] = output_path

        model = get_model(config=model_config)
        environment = get_environment(env_config, default_type="local")
        agent = self._build_agent(
            model=model,
            environment=environment,
            agent_config=agent_config,
            get_agent=get_agent,
        )
        self._event(HarnessEventType.THREAD_STARTED, {"trajectory_path": str(output_path)})
        self._event(HarnessEventType.TURN_STARTED, {"task_digest": _digest(task)})
        started = time.monotonic()
        result = agent.run(task)
        self._last_trajectory = output_path
        payload = dict(result or {})
        self._event(HarnessEventType.TURN_COMPLETED, {"exit_status": payload.get("exit_status", "")})
        self._load_usage(output_path, time.monotonic() - started)
        self._event(HarnessEventType.TOKEN_USAGE_UPDATED, self._last_usage.as_dict())
        self._event(HarnessEventType.WORKSPACE_REVISION_ADVANCED, {"patch_digest": self.patch_digest()})
        return tuple(self._event_queue)

    def _build_agent(self, *, model: Any, environment: Any, agent_config: dict[str, Any], get_agent: Any) -> Any:
        """Construct the 2.4.6 default agent with query/action hooks.

        Custom mini-swe-agent agent types continue to use the package factory.
        The default path subclasses the pinned public class so every model call
        and tool batch is represented in the common event stream.
        """
        requested = str(agent_config.get("agent_type", agent_config.get("type", "default")))
        if requested not in ("", "default"):
            return get_agent(model, environment, agent_config, default_type="default")
        try:
            from minisweagent.agents.default import DefaultAgent
        except ImportError:
            return get_agent(model, environment, agent_config, default_type="default")
        backend = self

        class InstrumentedAgent(DefaultAgent):
            def query(self) -> dict[str, Any]:
                message = super().query()
                extra = message.get("extra", {}) if isinstance(message, Mapping) else {}
                if not isinstance(extra, Mapping):
                    extra = {}
                backend._event(
                    HarnessEventType.ITEM_COMPLETED,
                    {
                        "kind": "model_query",
                        "api_call": self.n_calls,
                        "usage": dict(extra.get("usage", {})) if isinstance(extra.get("usage"), Mapping) else {},
                    },
                )
                return message

            def execute_actions(self, message: dict[str, Any]) -> list[dict[str, Any]]:
                actions = message.get("extra", {}).get("actions", [])
                backend._event(
                    HarnessEventType.ITEM_STARTED,
                    {"kind": "tool_batch", "action_count": len(actions) if isinstance(actions, list) else 0},
                )
                outputs = super().execute_actions(message)
                backend._event(
                    HarnessEventType.TOOL_RESULT,
                    {"kind": "tool_batch", "result_count": len(outputs)},
                )
                return outputs

        allowed = {
            "system_template",
            "instance_template",
            "step_limit",
            "cost_limit",
            "wall_time_limit_seconds",
            "max_consecutive_format_errors",
            "output_path",
        }
        agent_config = {key: value for key, value in agent_config.items() if key in allowed}
        agent_config.setdefault(
            "system_template",
            "You are a helpful coding assistant working in the supplied repository.",
        )
        agent_config.setdefault(
            "instance_template",
            "Please solve this issue:\n\n{{task}}",
        )
        return InstrumentedAgent(model, environment, **agent_config)

    def _load_usage(self, trajectory: Path, elapsed: float) -> None:
        data: dict[str, Any] = {}
        if trajectory.is_file():
            try:
                data = json.loads(trajectory.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
        info = data.get("info", {}) if isinstance(data, Mapping) else {}
        stats = info.get("model_stats", {}) if isinstance(info, Mapping) else {}
        if not isinstance(stats, Mapping):
            stats = {}
        input_tokens, output_tokens, total_tokens = self._trajectory_tokens(data)
        self._last_usage = UsageSnapshot(
            api_calls=int(stats.get("api_calls", 0) or 0),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=float(stats["instance_cost"]) if stats.get("instance_cost") is not None else None,
            wall_time_seconds=elapsed,
        )

    @staticmethod
    def _trajectory_tokens(data: Mapping[str, Any]) -> tuple[int | None, int | None, int | None]:
        """Collect token counters from 2.4.6 messages without assuming a provider schema."""
        totals = {"input": 0, "output": 0, "total": 0}
        seen = {"input": False, "output": False, "total": False}

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, target in (
                    ("input_tokens", "input"),
                    ("prompt_tokens", "input"),
                    ("output_tokens", "output"),
                    ("completion_tokens", "output"),
                    ("total_tokens", "total"),
                ):
                    raw = value.get(key)
                    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                        totals[target] += int(raw)
                        seen[target] = True
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(data)
        input_tokens = totals["input"] if seen["input"] else None
        output_tokens = totals["output"] if seen["output"] else None
        total_tokens = totals["total"] if seen["total"] else (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        return input_tokens, output_tokens, total_tokens

    def patch_digest(self) -> str | None:
        patch = self.run_root / "model.patch"
        if not patch.is_file():
            return None
        return hashlib.sha256(patch.read_bytes()).hexdigest()

    def checkpoint(self, revision_id: str, **payload: Any) -> HarnessCheckpoint:
        if self._session is None:
            raise RuntimeError("start_session must be called before checkpoint")
        values = {"thread_id": self._session.thread_id, "branch_id": self._session.branch_id, **payload}
        return HarnessCheckpoint(self._session.run_id, f"{self._session.run_id}:{revision_id}", revision_id, values)

    def resume(self, checkpoint: HarnessCheckpoint) -> HarnessSession:
        return self.start_session(
            run_id=checkpoint.run_id,
            branch_id=str(checkpoint.payload.get("branch_id", "main")),
        )

    def usage(self) -> UsageSnapshot:
        current = self._last_usage
        if current.wall_time_seconds is None:
            return UsageSnapshot(wall_time_seconds=time.monotonic() - self._started_at)
        return current

    def close(self) -> None:
        self._closed = True
