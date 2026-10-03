"""Context and progress guards for the pinned mini-swe-agent backend.

The guards are deliberately provider-neutral.  They operate on the messages
passed to mini-swe-agent and on short workspace/action digests, so the public
runtime does not need to know about a benchmark's private task layout.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Token budget derived from the model context window."""

    model_limit: int = 200_000
    system_overhead: int = 8_000
    tool_schema_tokens: int = 16_000
    output_reserve: int = 16_000
    safety_margin: int = 8_000

    @property
    def input_limit(self) -> int:
        return self.model_limit - (
            self.system_overhead
            + self.tool_schema_tokens
            + self.output_reserve
            + self.safety_margin
        )

    def validate(self) -> None:
        if self.model_limit < 16_000:
            raise ValueError("model_limit must be at least 16000")
        if self.input_limit < 1_024:
            raise ValueError("context reservations leave no usable input budget")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> ContextBudget:
        if not value:
            result = cls()
        else:
            base = cls()
            result = cls(
                model_limit=int(value.get("model_limit", base.model_limit)),
                system_overhead=int(value.get("system_overhead", base.system_overhead)),
                tool_schema_tokens=int(
                    value.get("tool_schema_tokens", base.tool_schema_tokens)
                ),
                output_reserve=int(value.get("output_reserve", base.output_reserve)),
                safety_margin=int(value.get("safety_margin", base.safety_margin)),
            )
        result.validate()
        return result


def _json_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _conservative_tokens(value: object) -> int:
    # Two characters per token is intentionally conservative for source code,
    # JSON tool payloads, and non-ASCII task text.  Provider token counters are
    # used when available and are never allowed to increase the limit silently.
    return max(1, math.ceil(_json_size(value) / 2))


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


class ContextWindowGuard:
    """Wrap a mini model and trim only the oldest complete message groups."""

    def __init__(self, delegate: Any, *, model_name: str, budget: ContextBudget) -> None:
        self.delegate = delegate
        self.model_name = model_name
        self.budget = budget
        self.last_stats: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def _estimate_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
        conservative = _conservative_tokens(list(messages))
        try:
            import litellm

            provider_count = int(
                litellm.token_counter(model=self.model_name, messages=list(messages))
            )
        except Exception:  # noqa: BLE001 - provider tokenizer is optional
            provider_count = 0
        return max(conservative, provider_count)

    @staticmethod
    def _context_synopsis(dropped: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Return a small, provider-neutral summary of discarded history.

        A long coding task can exceed the input budget while the agent is still
        exploring the repository.  Keeping only the newest tool pair is safe
        for the API contract, but it makes the model forget which files it has
        already inspected and restart from ``ls``/``find`` after every trim.
        The synopsis preserves command intent and result status without copying
        source code, prompts, credentials, or full tool output.
        """

        commands: list[str] = []
        results: list[str] = []
        seen_commands: set[str] = set()
        for message in dropped:
            role = message.get("role")
            if role == "assistant":
                for call in message.get("tool_calls", []) or []:
                    function = call.get("function", {}) if isinstance(call, Mapping) else {}
                    arguments = function.get("arguments", "") if isinstance(function, Mapping) else ""
                    command = ""
                    if isinstance(arguments, str):
                        try:
                            parsed = json.loads(arguments)
                        except json.JSONDecodeError:
                            parsed = {}
                        if isinstance(parsed, Mapping):
                            command = str(parsed.get("command", "")).strip()
                    if command:
                        # Keep the shell verb and path context, but cap each
                        # item so a huge generated command cannot consume the
                        # entire synopsis budget.
                        compact = " ".join(command.split())[:240]
                        if compact not in seen_commands:
                            seen_commands.add(compact)
                            commands.append(compact)
            elif role == "tool":
                extra = message.get("extra", {})
                returncode = extra.get("returncode") if isinstance(extra, Mapping) else None
                content = str(message.get("content", ""))
                first_line = " ".join(content.splitlines()[:1]).strip()[:180]
                if returncode is not None or first_line:
                    status = f"rc={returncode}" if returncode is not None else "rc=?"
                    results.append(f"{status} {first_line}".strip())

        # The most recent discarded commands are generally the useful ones;
        # retain a bounded prefix as well so the synopsis remains stable while
        # the history grows.
        commands = commands[-32:]
        results = results[-16:]
        lines = [
            "The context guard compacted earlier agent history.",
            "Continue from this synopsis and do not repeat repository exploration unless it is needed:",
            f"Previously issued bash commands ({len(commands)} retained):",
        ]
        lines.extend(f"- {command}" for command in commands)
        lines.append("Recent tool result summaries:")
        lines.extend(f"- {result}" for result in results)
        return {
            "role": "user",
            "content": "\n".join(lines),
        }

    def _prepare(self, messages: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        original = list(messages)
        total = self._estimate_tokens(original)
        limit = self.budget.input_limit
        if total <= limit:
            self.last_stats = {
                "input_messages": len(original),
                "kept_messages": len(original),
                "estimated_input_tokens": total,
                "input_limit_tokens": limit,
                "trimmed": False,
            }
            return original

        prefix = original[:2]
        prefix_tokens = self._estimate_tokens(prefix)
        remaining = max(1, limit - prefix_tokens)
        tail: list[Mapping[str, Any]] = []
        cursor = len(original) - 1
        while cursor >= 2:
            candidate = original[cursor]
            candidate_tokens = self._estimate_tokens([candidate])
            if tail and sum(self._estimate_tokens([item]) for item in tail) + candidate_tokens > remaining:
                break
            if not tail and candidate_tokens > remaining:
                break
            tail.append(candidate)
            cursor -= 1

        start = cursor + 1
        tail.reverse()
        # A tool result is only valid when its assistant tool-call message is
        # retained as well.  Pull that preceding message into the suffix.
        while start > 2 and tail and original[start].get("role") == "tool":
            start -= 1
            tail.insert(0, original[start])

        # Pulling a paired tool-call message may consume the final few tokens
        # of the budget. Drop only the oldest suffix items until the bound is
        # satisfied; the system/task prefix and newest tool result remain.
        while tail and self._estimate_tokens(prefix + tail) > limit:
            removed = tail.pop(0)
            if (
                removed.get("role") == "assistant"
                and removed.get("tool_calls")
                and tail
                and tail[0].get("role") == "tool"
            ):
                tail.pop(0)

        dropped = original[2:start]
        synopsis = self._context_synopsis(dropped) if dropped else None
        context_prefix = prefix + ([synopsis] if synopsis else [])
        while tail and self._estimate_tokens(context_prefix + tail) > limit:
            removed = tail.pop(0)
            if (
                removed.get("role") == "assistant"
                and removed.get("tool_calls")
                and tail
                and tail[0].get("role") == "tool"
            ):
                tail.pop(0)

        kept = context_prefix + tail
        estimate = self._estimate_tokens(kept)
        self.last_stats = {
            "input_messages": len(original),
            "kept_messages": len(kept),
            "estimated_input_tokens": estimate,
            "input_limit_tokens": limit,
            "trimmed": True,
            "dropped_messages": max(0, len(original) - len(kept)),
            "synopsis_included": synopsis is not None,
            "synopsis_commands": len(
                [line for line in str(synopsis.get("content", "")).splitlines() if line.startswith("- ")]
            )
            if synopsis
            else 0,
        }
        return kept

    def query(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
        prepared = self._prepare(messages)
        result = self.delegate.query(prepared, **kwargs)
        extra = result.setdefault("extra", {})
        extra["context_guard"] = dict(self.last_stats)
        return result


class AgentStalledError(RuntimeError):
    """The agent repeated the same observable action without workspace progress."""

    failure_class = "AGENT_STALLED"


class ProgressGuard:
    """Stop exact repeated action/output cycles while leaving steps unbounded."""

    def __init__(self, *, no_progress_limit: int = 2) -> None:
        if no_progress_limit < 1:
            raise ValueError("no_progress_limit must be positive")
        self.no_progress_limit = no_progress_limit
        self._last_signature: str | None = None
        self._last_workspace_digest: str | None = None
        self._count = 0

    def observe(
        self,
        *,
        actions: Sequence[Mapping[str, Any]],
        outputs: Sequence[Mapping[str, Any]],
        workspace_digest: str,
    ) -> None:
        signature = _digest(
            {
                "actions": list(actions),
                "outputs": list(outputs),
                "workspace": workspace_digest,
            }
        )
        if (
            signature == self._last_signature
            and workspace_digest == self._last_workspace_digest
        ):
            self._count += 1
        else:
            # The first unchanged cycle is already one no-progress observation;
            # the configured limit therefore counts consecutive observations,
            # rather than requiring an extra hidden retry.
            self._count = 1
        self._last_signature = signature
        self._last_workspace_digest = workspace_digest
        if self._count >= self.no_progress_limit:
            raise AgentStalledError(
                f"same action/output/workspace state observed {self._count} times"
            )
