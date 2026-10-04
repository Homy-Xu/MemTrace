from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import CodexProviderConfiguration
from ..contracts import (
    CommitmentLevel,
    MilestoneSpec,
    PlanSpec,
    digest,
    primitive,
    stable_id,
)
from ..orchestration.planning_coordinator import (
    WorkspaceReadOnlyGuard,
    WorkspaceSnapshotReceipt,
)
from .adapter import CodexPlanningResult
from .context_transport import CodexContextTransport
from .contracts import HarnessCapabilities, HarnessEvent, HarnessEventType
from .dynamic_tools import DynamicToolInvocation, DynamicToolResult
from .events import CodexEventMapper
from .memory_tools import (
    code_graph_search_dynamic_tool,
    memory_dynamic_tools,
    milestone_review_dynamic_tool,
)
from .normalizer import CodexPlanNormalizer
from .revision import WorkspaceRevisionTracker

_SECRET_ENV_NAMES = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MEMTENSOR_DOMESTIC_API_KEY",
        "MEMTENSOR_API_KEY",
        "AZURE_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    }
)
_PLAN_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*+])\s+(?P<text>\S.*)$")
_PATH_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.-])(?P<path>"
    r"(?:\.?\.?/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+"
    r"|[A-Za-z0-9_.-]+\.(?:py|pyi|js|jsx|ts|tsx|go|rs|java|groovy|toml|ya?ml|json|md)"
    r")"
)
_UNSAFE_PATH_CHARS = re.compile(r"[\s|&;<>`$\"'\\]")
_MAX_PATH_COMPONENT = 255
MINI_TURN_STEP_LIMIT = 8
_COVERAGE_TOKEN = re.compile(r"[a-z0-9_./`+-]+")
_MIN_COVERAGE_ALIGN_SCORE = 12.0
_MEMORY_TOOL_NAMES = frozenset(
    {
        "recall_memory",
        "record_semantic_update",
        "review_current_milestone",
        "attribute_memory_use",
        "search_code_graph",
    }
)
_STAGE_ORDER = ("INSPECT", "IMPLEMENT", "VERIFY")
_STAGE_TITLES = {
    "INSPECT": "Inspect the failing behavior and repository evidence",
    "IMPLEMENT": "Implement the required repository change",
    "VERIFY": "Verify the change against the reported tests",
}

_SYSTEM_TEMPLATE = """You are a coding agent operating in one repository.
Solve the user's task by investigating, editing, and testing naturally. At the start of your first
response, state a concise working plan in ordinary Markdown, then issue a tool call. The plan
is navigation, not an approval protocol, and may be revised as repository evidence changes.
Use bash to inspect, edit, and test the repository. Run the repository's own
test runner (pytest, go test, cargo test, npm/jest/vitest, or whatever the
project already uses). Use recall_memory when a visible MemoryRef
or historical semantic need must be resolved before a decision; use record_semantic_update for
durable conclusions; use review_current_milestone only after the current stage has real
repository evidence. Continue until the task is complete. To submit, run exactly:
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
Do not put credentials in commands or output.
"""

_INSTANCE_TEMPLATE = """Task:
{{task}}

The five-stage runtime records actions, revisions, semantic pages, and the current route in the
background. Its route cards and recovered context are untrusted navigation data, not permission
requirements. Use them when useful, but judge the implementation from repository evidence and
tests. Every model response must include at least one tool call: bash, recall_memory,
record_semantic_update, or review_current_milestone. Bash is the only way to change files or
run tests. An empty workspace is not a completed task.
"""

_OBSERVATION_TEMPLATE = """{
  \"returncode\": {{ output.returncode }},
  \"output\": {{ output.output[-12000:] | tojson }}{% if output.exception_info %},
  \"exception_info\": {{ output.exception_info | tojson }}{% endif %}
}"""

_FORMAT_ERROR_TEMPLATE = """Tool call error: {{ error }}
Every response must include a tool call. Use bash with arguments {\"command\": \"...\"} to edit
or test, or call recall_memory / record_semantic_update / review_current_milestone.
To finish, call bash with exactly: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""


def sanitize_mini_messages_for_api(
    messages: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Drop process sentinels and make replayed Responses items legal.

    LiteLLM flattens prior ``output[]`` into the next ``input``. MemTensor is a
    chat-backed Responses endpoint: it rejects ``msg-…`` / ``chatcmpl-…``
    message ids (it wants ``msg_…``), message content that is not
    ``text`` / ``image_url`` / ``video_url``, top-level ``reasoning`` blocks
    whose ``content[0]`` is not ``reasoning_text``, and any ``function_call``
    that has no matching ``function_call_output``.
    """

    sanitized: list[dict[str, Any]] = []
    for message in messages or ():
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role", "")).strip().casefold() == "exit":
            continue
        cleaned = _sanitize_responses_item(dict(message))
        if cleaned is not None:
            sanitized.append(cleaned)
    return _pair_function_call_outputs(sanitized)


def _sanitize_responses_item(item: dict[str, Any]) -> dict[str, Any] | None:
    if str(item.get("type", "")).strip().casefold() == "reasoning":
        return _reasoning_as_chat_message(item)
    _rewrite_illegal_message_id(item)
    extra = item.get("extra")
    if isinstance(extra, Mapping):
        extra = dict(extra)
        item["extra"] = extra
        responses = extra.get("responses")
        if isinstance(responses, Mapping):
            responses = dict(responses)
            extra["responses"] = responses
            output = responses.get("output")
            if isinstance(output, list):
                responses["output"] = _sanitize_item_list(output)
        output = extra.get("output")
        if isinstance(output, list):
            extra["output"] = _sanitize_item_list(output)
    content = item.get("content")
    if isinstance(content, list):
        item["content"] = _sanitize_message_content(content)
    output = item.get("output")
    if isinstance(output, list):
        item["output"] = _sanitize_item_list(output)
    _strip_empty_function_call_fields(item)
    return item


def _sanitize_item_list(parts: Sequence[object]) -> list[object]:
    cleaned: list[object] = []
    for part in parts:
        if isinstance(part, Mapping):
            item = _sanitize_responses_item(dict(part))
            if item is not None:
                cleaned.append(item)
        else:
            cleaned.append(part)
    return cleaned


def _sanitize_message_content(parts: Sequence[object]) -> list[object]:
    """MemTensor chat validation only accepts text / image / video parts."""

    cleaned: list[object] = []
    for part in parts:
        if not isinstance(part, Mapping):
            cleaned.append(part)
            continue
        kind = str(part.get("type") or "").strip().casefold()
        if kind == "reasoning":
            text = _extract_item_text(part)
            if text.strip():
                cleaned.append({"type": "text", "text": text})
            continue
        if kind in {"output_text", "input_text", "summary_text", "reasoning_text"}:
            cleaned.append({"type": "text", "text": str(part.get("text") or "")})
            continue
        nested = _sanitize_responses_item(dict(part))
        if nested is not None:
            cleaned.append(nested)
    return cleaned


def _extract_item_text(item: Mapping[str, Any]) -> str:
    texts: list[str] = []
    for key in ("content", "summary"):
        parts = item.get(key)
        if isinstance(parts, str):
            if parts:
                texts.append(parts)
            continue
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, Mapping):
                text = part.get("text")
                if text:
                    texts.append(str(text))
            elif isinstance(part, str) and part:
                texts.append(part)
    return "\n".join(texts)


def _reasoning_as_chat_message(item: Mapping[str, Any]) -> dict[str, Any] | None:
    """Replay hidden thoughts as a chat-safe assistant message.

    MemTensor rejects ``type=reasoning`` with ``summary_text`` content
    (``content[0]`` must be ``reasoning_text``) and also rejects that block
    when the request is validated as a chat message. The five-stage kernel
    does not depend on replayed reasoning items; keeping the text as
    ``role=assistant`` / ``type=text`` preserves the transcript without
    sending an illegal item.
    """

    text = _extract_item_text(item)
    if not text.strip():
        return None
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
    }
    identifier = item.get("id")
    if isinstance(identifier, str) and identifier:
        message["id"] = _legal_message_id(identifier)
    return message


def _legal_message_id(identifier: str) -> str:
    if identifier.startswith("msg_"):
        return identifier
    if identifier.startswith("msg-"):
        return "msg_" + identifier[4:]
    suffix = identifier[9:] if identifier.startswith("chatcmpl-") else identifier
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in suffix)
    return "msg_" + (cleaned or "replayed")


def _rewrite_illegal_message_id(item: dict[str, Any]) -> None:
    identifier = item.get("id")
    if not isinstance(identifier, str) or not identifier:
        return
    kind = str(item.get("type") or "").strip().casefold()
    if kind in {"function_call", "function_call_output"}:
        return
    if item.get("object") == "response":
        return
    if kind == "message" or str(item.get("role") or "") in {"assistant", "user", "system"}:
        item["id"] = _legal_message_id(identifier)


def _strip_empty_function_call_fields(item: dict[str, Any]) -> None:
    kind = str(item.get("type") or "").strip().casefold()
    if kind != "function_call":
        return
    if item.get("content") is None:
        item.pop("content", None)
    for key in ("phase", "quality", "size", "role"):
        if item.get(key) in ("", None):
            item.pop(key, None)


def _pair_function_call_outputs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every replayed function_call must have a function_call_output."""

    seen_outputs: set[str] = set()
    call_ids: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, Mapping):
            kind = str(node.get("type", "")).strip().casefold()
            if kind == "function_call_output":
                call_id = node.get("call_id")
                if call_id:
                    seen_outputs.add(str(call_id))
            elif kind == "function_call":
                call_id = node.get("call_id") or node.get("id")
                if call_id:
                    call_ids.append(str(call_id))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for part in node:
                walk(part)

    walk(items)
    missing = [call_id for call_id in call_ids if call_id not in seen_outputs]
    appended: set[str] = set()
    for call_id in missing:
        if call_id in appended:
            continue
        appended.add(call_id)
        items.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(
                    {
                        "returncode": -1,
                        "output": "tool output was omitted from the replayed transcript",
                    }
                ),
            }
        )
    return items


def drop_exit_messages(messages: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    return [
        dict(message)
        for message in messages or ()
        if isinstance(message, Mapping) and str(message.get("role", "")).casefold() != "exit"
    ]


def workspace_has_progress(
    repository: Path,
    baseline_revision: str | None = None,
    ignore_roots: Sequence[str] = (),
) -> bool:
    """True when tracked task files differ from the planning baseline.

    Harness scratch such as ``.run`` is ignored so a submit cannot claim
    completion from bookkeeping files alone.
    """

    ignored = tuple(root.strip().strip("/") for root in ignore_roots if root.strip())

    def _ignored(path: str) -> bool:
        normalized = path.strip().strip("/")
        return any(
            normalized == root or normalized.startswith(root + "/")
            for root in ignored
        )

    if baseline_revision:
        probe = subprocess.run(
            ("git", "rev-parse", "--verify", baseline_revision),
            cwd=repository,
            check=False,
            capture_output=True,
        )
        if probe.returncode == 0:
            diff = subprocess.run(
                ("git", "diff", "--quiet", baseline_revision),
                cwd=repository,
                check=False,
                capture_output=True,
            )
            if diff.returncode != 0:
                return True
    porcelain = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if porcelain.returncode != 0:
        return False
    for line in porcelain.stdout.splitlines():
        path = line[3:].strip()
        if path.startswith("-> "):
            path = path[3:].strip()
        if path and not _ignored(path):
            return True
    return False


def lite_llm_memory_tools() -> list[dict[str, Any]]:
    """Expose Codex-named memory tools in the LiteLLM Responses tool envelope."""

    converted: list[dict[str, Any]] = []
    tools = list(memory_dynamic_tools())
    if os.environ.get("HOMY_MULTILANG_BASE_COMMIT"):
        tools.append(code_graph_search_dynamic_tool())
    tools.append(milestone_review_dynamic_tool({}))
    for tool in tools:
        schema = tool.get("inputSchema") or tool.get("parameters") or {"type": "object"}
        converted.append(
            {
                "type": "function",
                "name": str(tool["name"]),
                "description": str(tool.get("description", "")),
                "parameters": schema,
            }
        )
    return converted


def _normalized_review_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _criterion_field(criterion: object, name: str) -> str:
    if isinstance(criterion, Mapping):
        return str(criterion.get(name, "") or "")
    return str(getattr(criterion, name, "") or "")


def _is_milestone_review_slot(criterion: object) -> bool:
    if isinstance(criterion, Mapping):
        required = bool(criterion.get("required", True))
        level = str(criterion.get("commitment_level", CommitmentLevel.MILESTONE.value))
    else:
        required = bool(getattr(criterion, "required", True))
        raw_level = getattr(criterion, "commitment_level", CommitmentLevel.MILESTONE)
        level = raw_level.value if hasattr(raw_level, "value") else str(raw_level or "")
    return required and level == CommitmentLevel.MILESTONE.value


def _review_slots(official: Sequence[object]) -> list[object]:
    return [item for item in official if _is_milestone_review_slot(item)]


def _coverage_tokens(value: str) -> set[str]:
    return set(_COVERAGE_TOKEN.findall(value.casefold()))


def _coverage_align_score(slot: object, entry: Mapping[str, Any]) -> float:
    official_req = _normalized_review_text(_criterion_field(slot, "requirement_text"))
    official_out = _normalized_review_text(_criterion_field(slot, "observable_outcome"))
    raw_req = _normalized_review_text(entry.get("requirement_text"))
    raw_out = _normalized_review_text(entry.get("observable_outcome"))
    if official_req == raw_req and official_out == raw_out:
        return 1000.0
    if official_req == raw_req or official_out == raw_out:
        return 800.0
    official = f"{official_req} {official_out}"
    raw = f"{raw_req} {raw_out}"
    official_folded = official.casefold()
    raw_folded = raw.casefold()
    score = 0.0
    if raw_req and raw_req.casefold() in official_folded:
        score += 40.0 + min(len(raw_req), 200) / 10.0
    if official_req and official_req[:160].casefold() in raw_folded:
        score += 30.0
    official_tokens = _coverage_tokens(official)
    raw_tokens = _coverage_tokens(raw)
    if official_tokens and raw_tokens:
        intersection = official_tokens & raw_tokens
        score += 50.0 * len(intersection) / len(official_tokens | raw_tokens)
        score += 20.0 * len(intersection) / len(official_tokens)
    return score


def align_mini_requirement_coverage(
    official: Sequence[object],
    raw_coverage: Sequence[object],
) -> list[dict[str, str]] | None:
    """Rewrite Mini review rows onto official requirement/outcome texts.

    Five-stage still exact-matches ``minimum_acceptance``. Mini often splits the
    one task-anchored Criterion into clause-sized rows and adds Task preamble
    lines such as the branch/commit instruction. Mapping stays in the adapter.
    """

    slots = _review_slots(official)
    entries = [dict(item) for item in raw_coverage if isinstance(item, Mapping)]
    if not slots or not entries:
        return None
    if len(slots) == 1:
        best = max(entries, key=lambda item: _coverage_align_score(slots[0], item))
        if _coverage_align_score(slots[0], best) < _MIN_COVERAGE_ALIGN_SCORE:
            return None
        summaries = [
            _normalized_review_text(item.get("evidence_summary"))
            for item in entries
            if _normalized_review_text(item.get("evidence_summary"))
        ]
        statuses = [
            str(item.get("status", "")).upper().strip()
            for item in entries
        ]
        has_gap = any(status == "NOT_SATISFIED" for status in statuses)
        status = "NOT_SATISFIED" if has_gap else "SATISFIED"
        if status not in {"SATISFIED", "NOT_SATISFIED"}:
            return None
        evidence = " ".join(summaries) or _normalized_review_text(best.get("evidence_summary"))
        if not evidence:
            return None
        return [
            {
                "requirement_text": _normalized_review_text(
                    _criterion_field(slots[0], "requirement_text")
                ),
                "observable_outcome": _normalized_review_text(
                    _criterion_field(slots[0], "observable_outcome")
                ),
                "status": status,
                "evidence_summary": evidence[:4000],
            }
        ]
    pairs: list[tuple[float, int, int]] = []
    for slot_index, slot in enumerate(slots):
        for entry_index, entry in enumerate(entries):
            score = _coverage_align_score(slot, entry)
            if score >= _MIN_COVERAGE_ALIGN_SCORE:
                pairs.append((score, slot_index, entry_index))
    pairs.sort(reverse=True)
    assigned: dict[int, int] = {}
    used_entries: set[int] = set()
    for _score, slot_index, entry_index in pairs:
        if slot_index in assigned or entry_index in used_entries:
            continue
        assigned[slot_index] = entry_index
        used_entries.add(entry_index)
    if len(assigned) != len(slots):
        return None
    aligned: list[dict[str, str]] = []
    for slot_index, slot in enumerate(slots):
        entry = entries[assigned[slot_index]]
        status = str(entry.get("status", "")).upper().strip()
        evidence = _normalized_review_text(entry.get("evidence_summary"))
        if status not in {"SATISFIED", "NOT_SATISFIED"} or not evidence:
            return None
        aligned.append(
            {
                "requirement_text": _normalized_review_text(
                    _criterion_field(slot, "requirement_text")
                ),
                "observable_outcome": _normalized_review_text(
                    _criterion_field(slot, "observable_outcome")
                ),
                "status": status,
                "evidence_summary": evidence,
            }
        )
    return aligned


def _usable_repo_relpath(relative: str) -> str | None:
    """Keep only portable repository-relative paths Mini may emit as file refs.

    The five-stage Rich Graph treats failed-test identities as file signals.
    A Mini bash command (``cd … && go test -run 'A|B|…'`` or a heredoc) is not
    a path; joining it under the repository raises ``ENAMETOOLONG`` in the
    kernel and ``homy-v2`` exits 2 before a five-stage result is written.
    """

    value = str(relative).strip().replace("\\", "/").lstrip("./")
    if not value or value in {".", ".."}:
        return None
    parts = Path(value).parts
    if any(part in {"", ".", ".."} or len(part) > _MAX_PATH_COMPONENT for part in parts):
        return None
    if _UNSAFE_PATH_CHARS.search(value):
        return None
    return value


def mini_test_selector(command: str) -> str:
    """Short test identity so the kernel never stats the raw Mini command."""

    compact = " ".join(str(command).split())
    if not compact:
        return "mini-test:empty"
    portable = _usable_repo_relpath(compact)
    if portable is not None and len(portable) <= 200:
        return portable
    from .command_semantics import command_observation_scope

    scope = command_observation_scope(compact)
    if scope.test_command:
        short_paths = []
        for path in sorted(scope.paths):
            usable = _usable_repo_relpath(path)
            if usable is not None and len(usable) <= 200:
                short_paths.append(usable)
        if len(short_paths) == 1:
            return short_paths[0]
        if short_paths:
            return f"mini-test:{digest({'paths': short_paths})[:24]}"
    return f"mini-test:{digest({'command': compact})[:24]}"


def _task_entity_refs(*texts: str) -> list[str]:
    refs: list[str] = []
    for text in texts:
        for match in _PATH_TOKEN.finditer(text):
            path = _usable_repo_relpath(match.group("path"))
            if path:
                refs.append(f"file:{path}")
    return list(dict.fromkeys(refs))


def _milestone_review_criteria(milestone: MilestoneSpec | None) -> list[dict[str, object]]:
    if milestone is None:
        return []
    slots: list[dict[str, object]] = []
    for item in milestone.minimum_acceptance:
        if not _is_milestone_review_slot(item):
            continue
        slots.append(
            {
                "requirement_text": item.requirement_text,
                "observable_outcome": item.observable_outcome,
                "required": True,
                "commitment_level": CommitmentLevel.MILESTONE.value,
            }
        )
    return slots


def parse_mini_tool_actions(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Accept bash plus the bound five-stage memory tools."""

    extra = message.get("extra")
    if not isinstance(extra, Mapping):
        return []
    actions = extra.get("actions")
    if not isinstance(actions, list):
        return []
    parsed: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            continue
        tool = str(action.get("tool") or action.get("name") or "").strip()
        if tool in _MEMORY_TOOL_NAMES:
            arguments = action.get("arguments", action.get("input", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            parsed.append(
                {
                    "tool": tool,
                    "call_id": str(action.get("call_id") or action.get("id") or f"memory-{index}"),
                    "arguments": dict(arguments) if isinstance(arguments, Mapping) else {},
                }
            )
            continue
        if "command" in action:
            parsed.append({"tool": "bash", "command": str(action["command"])})
    return parsed


def _parse_mini_response_actions(
    output: object,
    *,
    format_error_template: str,
) -> list[dict[str, Any]]:
    """Parse Responses function_call items, including five-stage memory tools."""

    from jinja2 import StrictUndefined, Template
    from minisweagent.exceptions import FormatError
    from minisweagent.models.utils.actions_toolcall_response import _format_error_message

    tool_calls: list[dict[str, Any]] = []
    for item in output if isinstance(output, list) else ():
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type != "function_call":
            continue
        if hasattr(item, "model_dump"):
            tool_calls.append(item.model_dump())
        elif isinstance(item, dict):
            tool_calls.append(item)
        else:
            tool_calls.append(dict(item))
    if not tool_calls:
        error_text = Template(format_error_template, undefined=StrictUndefined).render(
            error="No tool calls found in the response. Every response MUST include at least one tool call.",
            actions=[],
            has_tool_calls=False,
        )
        raise FormatError(_format_error_message(error_text))
    actions: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        name = str(tool_call.get("name") or "")
        call_id = tool_call.get("call_id") or tool_call.get("id")
        error_msg = ""
        try:
            args = json.loads(tool_call.get("arguments", "{}") or "{}")
        except Exception as exc:
            args = {}
            error_msg = f"Error parsing tool call arguments: {exc}."
        if name in _MEMORY_TOOL_NAMES:
            if error_msg:
                error_text = Template(format_error_template, undefined=StrictUndefined).render(
                    error=error_msg.strip(),
                    actions=[],
                    has_tool_calls=True,
                )
                raise FormatError(_format_error_message(error_text))
            actions.append(
                {
                    "tool": name,
                    "call_id": str(call_id or ""),
                    "tool_call_id": call_id,
                    "arguments": args if isinstance(args, dict) else {},
                }
            )
            continue
        if name != "bash":
            error_msg += f"Unknown tool '{name}'."
        if not isinstance(args, dict) or "command" not in args:
            error_msg += "Missing 'command' argument in bash tool call."
        if error_msg:
            error_text = Template(format_error_template, undefined=StrictUndefined).render(
                error=error_msg.strip(),
                actions=[],
                has_tool_calls=True,
            )
            raise FormatError(_format_error_message(error_text))
        actions.append(
            {
                "command": args["command"],
                "tool": "bash",
                "tool_call_id": call_id,
            }
        )
    return actions


@dataclass(slots=True)
class _MiniRuntime:
    agent: Any
    pending_response: Mapping[str, Any] | None = None
    logical_thread_id: str | None = None
    archived_epoch_count: int = 0


class _MiniTransportProxy:
    """Translate context injection into mini's ordinary message history."""

    def __init__(self, adapter: MiniSweAgentHarnessAdapter) -> None:
        self.adapter = adapter

    @staticmethod
    def _texts(value: object) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        output: list[str] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            text = str(item.get("text", "")).strip()
            if text:
                output.append(text)
        return tuple(output)

    def request(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        thread_id = str(params.get("threadId", self.adapter.thread_id or ""))
        if method == "turn/steer":
            self.adapter.queue_context(*self._texts(params.get("input")))
            return {"turnId": str(params.get("expectedTurnId", ""))}
        if method == "thread/inject_items":
            items = params.get("items", ())
            for item in items if isinstance(items, list) else ():
                if isinstance(item, Mapping):
                    self.adapter.queue_context(*self._texts(item.get("content")))
            return {"threadId": thread_id}
        if method == "thread/start":
            replacement = stable_id(
                "mini_thread_",
                {"predecessor": self.adapter.thread_id, "time_ns": time.time_ns()},
            )
            return {"thread": {"id": replacement}}
        if method == "thread/resume":
            raise RuntimeError("mini-swe-agent does not support native Thread resume")
        if method == "thread/compact/start":
            raise RuntimeError("mini-swe-agent does not support native context compaction")
        raise RuntimeError(f"mini-swe-agent does not support transport method {method}")


class MiniSweAgentContextTransport(CodexContextTransport):
    """Page-in transport with adapter-owned logical Epoch replacement.

    mini-swe-agent has no Provider-side Thread API.  It can still honor the
    five-stage physical-context boundary by starting a fresh message history
    and receiving the durable ContinuityCheckpoint through the ordinary
    context queue.  This is deliberately not advertised as native Thread
    resume or native compaction.
    """

    def __init__(self, adapter: MiniSweAgentHarnessAdapter) -> None:
        super().__init__(
            adapter,  # type: ignore[arg-type]
            supports_native_compaction=False,
            native_compaction_policy_enabled=False,
        )

    def start_replacement_thread(self) -> str:
        response = self.adapter.transport.request("thread/start", {})
        thread = response.get("thread")
        if not isinstance(thread, Mapping) or not str(thread.get("id", "")):
            raise RuntimeError("mini logical thread/start returned no Thread ID")
        replacement = str(thread["id"])
        if replacement == self.thread_id:
            raise RuntimeError("mini logical replacement Thread must differ from its predecessor")
        self.adapter.stage_logical_replacement(replacement)
        return replacement

    def resume_replacement_thread(self, thread_id: str) -> None:
        # Recovery is based on the durable five-stage ContextImage, not on an
        # unavailable mini Provider session.  The coordinator will inject that
        # image before the replacement Turn is allowed to sample.
        self.adapter.stage_logical_replacement(thread_id)


class MiniSweAgentHarnessAdapter:
    """Experimental mini-swe-agent 2.4.6 adapter for the five-stage runtime.

    Imports are lazy: installing Homy without the ``mini`` extra leaves the default Codex import
    graph and execution behavior unchanged.
    """

    def __init__(
        self,
        *,
        repository_path: Path,
        model: str,
        run_root: Path,
        provider: CodexProviderConfiguration,
        reasoning_effort: str | None = None,
        command_timeout_seconds: int = 1800,
        runtime_factory: Callable[..., Any] | None = None,
        turn_step_limit: int = MINI_TURN_STEP_LIMIT,
    ) -> None:
        self.repository_path = Path(repository_path).expanduser().resolve()
        self.run_root = Path(run_root).expanduser().resolve()
        self.model = model.strip()
        self.provider = provider
        self.provider.validate()
        self.reasoning_effort = reasoning_effort
        self.command_timeout_seconds = command_timeout_seconds
        self.turn_step_limit = max(1, int(turn_step_limit))
        self.normalizer = CodexPlanNormalizer()
        self.thread_id: str | None = None
        self.protocol_schema = None
        self.transport = _MiniTransportProxy(self)
        self._runtime_factory = runtime_factory or self._default_runtime_factory
        self._runtime: _MiniRuntime | None = None
        self._planning_mapper: CodexEventMapper | None = None
        self._queued_context: list[str] = []
        self._trusted_verification: tuple[str, ...] = ()
        self._staged_replacement_thread_id: str | None = None

    def bind_trusted_verification_contract(self, selectors: Sequence[str]) -> None:
        self._trusted_verification = tuple(map(str, selectors))
        self.normalizer.bind_trusted_verification_contract(self._trusted_verification)

    @staticmethod
    def _safe_command_environment() -> dict[str, str]:
        allowlist = {
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "SHELL",
            "LANG",
            "LC_ALL",
            "TERM",
            "TMPDIR",
            "PYTHONPATH",
        }
        safe = {
            key: value
            for key, value in os.environ.items()
            if key in allowlist and key not in _SECRET_ENV_NAMES
        }
        safe.update(
            {
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            }
        )
        return safe

    def _default_runtime_factory(self, *, task: str, trajectory_path: Path) -> _MiniRuntime:
        try:
            import litellm
            from minisweagent import __version__ as mini_version
            from minisweagent.agents.default import DefaultAgent
            from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
            from minisweagent.models.litellm_response_model import LitellmResponseModel
        except ImportError as error:
            raise RuntimeError(
                "mini-swe-agent runtime is unavailable; install memtrace[mini-swe-agent]"
            ) from error
        if mini_version != "2.4.6":
            raise RuntimeError(f"mini-swe-agent 2.4.6 is required, found {mini_version}")
        if not self.provider.api_key_env:
            raise RuntimeError("mini-swe-agent requires a custom Provider api_key_env")
        api_key = os.environ.get(self.provider.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Provider credential environment {self.provider.api_key_env} is absent"
            )

        class SecretSafeEnvironment(LocalEnvironment):
            def execute(
                runtime_self,
                action: dict,
                cwd: str = "",
                *,
                timeout: int | None = None,
            ) -> dict[str, Any]:
                import subprocess

                from minisweagent.environments.local import _run

                command = str(action.get("command", ""))
                resolved_cwd = cwd or runtime_self.config.cwd
                try:
                    completed = _run(
                        command,
                        resolved_cwd,
                        dict(runtime_self.config.env),
                        timeout or runtime_self.config.timeout,
                    )
                    output = {
                        "output": completed.stdout,
                        "returncode": completed.returncode,
                        "exception_info": "",
                    }
                except Exception as error:
                    raw = getattr(error, "output", "") or ""
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    output = {
                        "output": raw,
                        "returncode": -1,
                        "exception_info": f"{type(error).__name__}: {error}",
                    }
                    if isinstance(error, subprocess.TimeoutExpired):
                        output["extra"] = {"exception_type": "TimeoutExpired"}
                runtime_self._check_finished(output)
                return output

            def get_template_vars(runtime_self, **kwargs: Any) -> dict[str, Any]:
                return {
                    **runtime_self.config.model_dump(),
                    **platform.uname()._asdict(),
                    **kwargs,
                }

        class SecretSafeResponsesModel(LitellmResponseModel):
            abort_exceptions = [
                *LitellmResponseModel.abort_exceptions,
                litellm.exceptions.BadRequestError,
            ]

            def __init__(runtime_self, **kwargs: Any) -> None:
                runtime_self._homy_api_key = api_key
                runtime_self._homy_api_base = self.provider.base_url
                super().__init__(**kwargs)
                runtime_self.extra_tools = lite_llm_memory_tools()

            def _prepare_messages_for_api(runtime_self, messages: list[dict]) -> list[dict]:
                prepared = super()._prepare_messages_for_api(messages)
                return sanitize_mini_messages_for_api(prepared)

            def _parse_actions(runtime_self, response) -> list[dict]:
                return _parse_mini_response_actions(
                    getattr(response, "output", []),
                    format_error_template=runtime_self.config.format_error_template,
                )

            def _query(runtime_self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
                tools = [
                    {
                        "type": "function",
                        "name": "bash",
                        "description": "Execute a bash command",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                    *list(getattr(runtime_self, "extra_tools", None) or ()),
                ]
                extra = dict(runtime_self.config.model_kwargs | kwargs)
                extra.setdefault(
                    "reasoning",
                    {
                        "effort": self.reasoning_effort or "high",
                        "summary": "none",
                    },
                )
                return litellm.responses(
                    model=runtime_self.config.model_name,
                    input=sanitize_mini_messages_for_api(messages),
                    tools=tools,
                    api_key=runtime_self._homy_api_key,
                    api_base=runtime_self._homy_api_base,
                    reasoning_effort=self.reasoning_effort,
                    **extra,
                )

        environment = SecretSafeEnvironment(
            config_class=LocalEnvironmentConfig,
            cwd=str(self.repository_path),
            env=self._safe_command_environment(),
            timeout=self.command_timeout_seconds,
        )
        model_name = self.model if "/" in self.model else f"openai/{self.model}"
        model = SecretSafeResponsesModel(
            model_name=model_name,
            model_kwargs={"drop_params": True},
            cost_tracking="ignore_errors",
            observation_template=_OBSERVATION_TEMPLATE,
            format_error_template=_FORMAT_ERROR_TEMPLATE,
        )
        agent = DefaultAgent(
            model,
            environment,
            system_template=_SYSTEM_TEMPLATE,
            instance_template=_INSTANCE_TEMPLATE,
            step_limit=0,
            cost_limit=0,
            wall_time_limit_seconds=0,
            max_consecutive_format_errors=3,
            output_path=trajectory_path,
        )
        agent.extra_template_vars = {"task": task}
        agent.messages = []
        agent.add_messages(
            model.format_message(role="system", content=agent._render_template(_SYSTEM_TEMPLATE)),
            model.format_message(role="user", content=agent._render_template(_INSTANCE_TEMPLATE)),
        )
        return _MiniRuntime(agent=agent)

    def stage_logical_replacement(self, thread_id: str) -> None:
        """Prepare one fresh mini message history for a durable five-stage Epoch."""

        normalized = thread_id.strip()
        if not normalized:
            raise ValueError("mini logical replacement Thread ID must be non-empty")
        existing = self._staged_replacement_thread_id
        if existing is not None and existing != normalized:
            raise RuntimeError("multiple mini logical replacement Threads were staged")
        self._staged_replacement_thread_id = normalized

    def _archive_current_epoch(self, runtime: _MiniRuntime) -> None:
        """Persist the old physical history before releasing it from memory."""

        output = getattr(runtime.agent.config, "output_path", None)
        if output:
            epoch_dir = self.run_root / "mini-swe-agent" / "epochs"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            active_thread = runtime.logical_thread_id or self.thread_id or "unknown"
            safe_thread = re.sub(r"[^A-Za-z0-9_.-]+", "_", active_thread)[:96]
            snapshot = epoch_dir / (
                f"epoch-{runtime.archived_epoch_count:04d}-{safe_thread}.json"
            )
            runtime.agent.save(snapshot)
        runtime.archived_epoch_count += 1

    @staticmethod
    def _render_system_message(agent: Any) -> Mapping[str, Any]:
        renderer = getattr(agent, "_render_template", None)
        content = renderer(_SYSTEM_TEMPLATE) if callable(renderer) else _SYSTEM_TEMPLATE
        return agent.model.format_message(role="system", content=content)

    def begin_logical_thread(self, thread_id: str) -> None:
        """Activate a staged mini Epoch without claiming native Thread support.

        The coordinator still owns activation: ``adapter.thread_id`` changes
        only after a real model/tool event proves that the injected checkpoint
        was observed.  Here we replace only mini's physical message history.
        """

        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("mini runtime is unavailable for Context replacement")
        normalized = thread_id.strip()
        if runtime.logical_thread_id == normalized:
            return
        if self._staged_replacement_thread_id != normalized:
            raise RuntimeError("mini logical replacement was not staged")
        self._archive_current_epoch(runtime)
        runtime.agent.messages = []
        runtime.agent.add_messages(self._render_system_message(runtime.agent))
        runtime.pending_response = None
        runtime.logical_thread_id = normalized
        self._staged_replacement_thread_id = None

    @staticmethod
    def _message_text(message: Mapping[str, Any]) -> str:
        direct = message.get("content")
        if isinstance(direct, str):
            return direct.strip()
        texts: list[str] = []
        output = message.get("output")
        for item in output if isinstance(output, list) else ():
            if not isinstance(item, Mapping) or item.get("type") != "message":
                continue
            content = item.get("content")
            for block in content if isinstance(content, list) else ():
                if isinstance(block, Mapping) and block.get("type") in {"output_text", "text"}:
                    text = str(block.get("text", "")).strip()
                    if text:
                        texts.append(text)
        return "\n".join(texts).strip()

    @classmethod
    def _plan_steps(cls, text: str) -> tuple[Mapping[str, str], ...]:
        # Same section rules as Codex --multilang-plan, but Mini never speaks
        # App Server Plan items. Only activate when the Pier gate is on.
        if os.environ.get("HOMY_MULTILANG_BASE_COMMIT"):
            from .multilang_normalizer import MultilangPlanNormalizer

            sectioned = MultilangPlanNormalizer.native_plan_artifact_steps(text)
            if sectioned:
                return sectioned
        items: list[str] = []
        for line in text.splitlines():
            match = _PLAN_ITEM.match(line)
            if match is None:
                continue
            item = " ".join(match.group("text").split())
            if item and item not in items:
                items.append(item)
        return tuple(
            {"source_step_id": f"N{index:03d}", "step": item, "status": "pending"}
            for index, item in enumerate(items[:12], start=1)
        )

    @staticmethod
    def _work_kind(title: str) -> str:
        lowered = title.casefold()
        if any(word in lowered for word in ("test", "verify", "validate", "check")):
            return "VERIFY"
        if any(word in lowered for word in ("inspect", "investigate", "analy", "read", "find")):
            return "INSPECT"
        if any(word in lowered for word in ("implement", "fix", "edit", "change", "add")):
            return "IMPLEMENT"
        return "PROCESS"

    def _group_plan_stages(
        self,
        steps: tuple[Mapping[str, str], ...],
    ) -> tuple[tuple[Mapping[str, str], ...], list[tuple[str, list[Mapping[str, str]]]], str | None]:
        """Keep parsed plan items; synthesize missing inspect/implement/verify stages."""

        if not steps:
            synthesized = tuple(
                {
                    "source_step_id": f"N{index:03d}",
                    "step": _STAGE_TITLES[kind],
                    "status": "pending",
                }
                for index, kind in enumerate(_STAGE_ORDER, start=1)
            )
            groups = [
                (kind, [item])
                for kind, item in zip(_STAGE_ORDER, synthesized, strict=True)
            ]
            return synthesized, groups, "PLANNING_UNAVAILABLE"

        grouped: list[tuple[str, list[Mapping[str, str]]]] = []
        for item in steps:
            kind = self._work_kind(str(item["step"]))
            if kind == "PROCESS":
                kind = "IMPLEMENT"
            if grouped and grouped[-1][0] == kind:
                grouped[-1][1].append(item)
            else:
                grouped.append((kind, [item]))
        return steps, grouped, None

    def _project_plan(
        self,
        *,
        user_task: str,
        plan_text: str,
    ) -> tuple[PlanSpec, bool, tuple[Mapping[str, str], ...]]:
        parsed = self._plan_steps(plan_text)
        steps, groups, reason = self._group_plan_stages(parsed)
        available = reason != "PLANNING_UNAVAILABLE"
        task_anchor = user_task.strip()
        task_refs = _task_entity_refs(task_anchor, plan_text)
        milestones: list[dict[str, Any]] = []
        for ordinal, (kind, items) in enumerate(groups, start=1):
            del kind
            source_ids = [str(item["source_step_id"]) for item in items]
            milestone: dict[str, Any] = {
                "source_plan_item_ids": source_ids,
                "task_requirement": task_anchor,
                "target_outcome": task_anchor,
                # C001 is the whole Task. Stage kind only groups navigation
                # Steps; leaking INSPECT as STRUCTURAL made acceptance demand
                # unbound CODE_CHANGE and then stall the review.
                "claim_type": "BEHAVIORAL",
                "entity_refs": list(task_refs) if ordinal == 1 else [],
                "non_goals": [],
            }
            if ordinal == 1:
                milestone["steps"] = [
                    {
                        "source_plan_item_ids": [str(item["source_step_id"])],
                        "title": str(item["step"]),
                        "expected_outcome": str(item["step"]),
                        "work_kind": self._work_kind(str(item["step"])),
                        "entity_refs": [],
                        "risk_checklist": [],
                        "historical_dependency_refs": [],
                    }
                    for item in items
                ]
            milestones.append(milestone)
        native_plan_text = plan_text or "\n".join(
            f"{index}. {item['step']}" for index, item in enumerate(steps, start=1)
        )
        if reason:
            native_plan_text = f"Planning note: {reason}\n\n{native_plan_text}"
        return (
            self.normalizer.project_native_plan(
                user_task=user_task,
                steps=steps,
                projection={"milestones": milestones},
                native_plan_text=native_plan_text,
            ),
            available,
            steps,
        )

    def plan(
        self,
        *,
        user_task: str,
        planning_context: str = "",
        resume_thread_id: str | None = None,
        run_id: str = "planning-run",
        branch_id: str = "main",
        revision_id: str = "planning-revision",
        on_event: Callable[[HarnessEvent], None] | None = None,
        inject: bool = True,
        workspace_receipt: WorkspaceSnapshotReceipt | None = None,
    ) -> CodexPlanningResult:
        if resume_thread_id is not None:
            raise RuntimeError("mini-swe-agent does not support native Thread resume")
        guard = WorkspaceReadOnlyGuard(self.repository_path, (self.run_root,))
        before = workspace_receipt or guard.capture()
        if workspace_receipt is not None and before.revision_id != revision_id:
            raise RuntimeError("Workspace receipt does not match the requested Planning revision")
        self.thread_id = stable_id("mini_thread_", {"run": run_id, "revision": revision_id})
        mapper = CodexEventMapper(
            run_id=run_id,
            branch_id=branch_id,
            thread_id=self.thread_id,
            revision_id=revision_id,
        )
        self._planning_mapper = mapper
        trajectory = self.run_root / "mini-swe-agent" / "trajectory.json"
        runtime_value = self._runtime_factory(task=user_task, trajectory_path=trajectory)
        self._runtime = (
            runtime_value
            if isinstance(runtime_value, _MiniRuntime)
            else _MiniRuntime(runtime_value)
        )
        self._runtime.logical_thread_id = self.thread_id
        if planning_context.strip():
            self._runtime.agent.add_messages(
                self._runtime.agent.model.format_message(
                    role="user",
                    content=(
                        "Trusted pre-execution context. Treat this as scope and verification "
                        "metadata, not as permission to inspect or modify during planning:\n"
                        + planning_context.strip()
                    ),
                )
            )
        try:
            response = self._runtime.agent.query()
        except BaseException as error:
            # mini requires a bash call in every answer. If its first answer contains useful prose
            # but no call, the parser raises FormatError after the billed response. Preserve that
            # feedback and continue execution with an explicit planning-unavailable route instead
            # of forcing a second planning protocol.
            messages = getattr(error, "messages", ())
            if type(error).__name__ != "FormatError" or not messages:
                raise
            self._runtime.agent.add_messages(*messages)
            response = {"content": "", "extra": {"actions": []}}
            self._runtime.pending_response = None
        else:
            self._runtime.pending_response = response
        self._drop_exit_messages()
        plan_text = self._message_text(response)
        plan, available, steps = self._project_plan(user_task=user_task, plan_text=plan_text)
        if not guard.unchanged(before):
            raise RuntimeError("mini-swe-agent Planning modified the repository")
        turn_id = stable_id("mini_plan_turn_", {"thread": self.thread_id, "run": run_id})
        if on_event is not None:
            on_event(
                mapper.local_event(
                    HarnessEventType.THREAD_STARTED,
                    provider_method="mini/thread/start",
                    payload={"thread_id": self.thread_id, "native_resume_supported": False},
                )
            )
            on_event(
                mapper.local_event(
                    HarnessEventType.PLAN_PROPOSED,
                    provider_method="mini/plan/proposed",
                    turn_id=turn_id,
                    payload={
                        "status": "AVAILABLE" if available else "PLANNING_UNAVAILABLE",
                        "plan_text": plan_text,
                        "steps": list(steps),
                        "fallback_navigation_only": not available,
                        "planning_reason": None if available else "PLANNING_UNAVAILABLE",
                    },
                )
            )
            on_event(
                mapper.local_event(
                    HarnessEventType.MILESTONE_MANIFEST,
                    provider_method="homy/milestoneManifest",
                    turn_id=turn_id,
                    payload={"plan": plan, "final_plan_text": plan_text},
                )
            )
            on_event(
                mapper.local_event(
                    HarnessEventType.PLAN_PROJECTION_DECISION,
                    provider_method="homy/planProjection",
                    turn_id=turn_id,
                    payload={
                        "decision": "MINI_PLAN_PROJECTED_TO_SINGLE_NAVIGATION_MILESTONE",
                        "milestone_count": 1,
                        "source_plan_item_count": len(steps),
                        "planning_available": available,
                        "execution_blocking": False,
                    },
                )
            )
            on_event(
                mapper.local_event(
                    HarnessEventType.PLANNING_VALIDATED,
                    provider_method="homy/planningReadOnlyValidated",
                    turn_id=turn_id,
                    payload={"read_only_verified": True, "planning_available": available},
                )
            )
        if inject:
            self.inject_normalized_plan(plan, on_event=on_event)
        return CodexPlanningResult(
            thread_id=self.thread_id,
            turn_id=turn_id,
            model=self.model,
            plan=plan,
            final_plan_text=plan_text or None,
            read_only_verified=True,
        )

    def inject_normalized_plan(
        self,
        plan: PlanSpec,
        *,
        on_event: Callable[[HarnessEvent], None] | None = None,
    ) -> None:
        if self.thread_id is None or self._runtime is None:
            raise RuntimeError("cannot inject a Plan before mini Planning")
        self.queue_context(
            "Canonical five-stage route projection. This is navigation, not an approval gate.\n"
            + self.normalizer.render_for_thread(plan)
        )
        if on_event is not None and self._planning_mapper is not None:
            on_event(
                self._planning_mapper.local_event(
                    HarnessEventType.CANONICAL_PLAN_INJECTION,
                    provider_method="mini/context/queued",
                    payload={
                        "state": "TRANSPORT_ACCEPTED",
                        "plan_digest": digest(plan),
                        "native_thread_injection": False,
                    },
                )
            )

    def resume_planned_thread(self, **_: Any) -> None:
        raise RuntimeError("mini-swe-agent does not support native Thread resume")

    def _progress_ignore_roots(self) -> tuple[str, ...]:
        ignored = [".run", "mini-swe-agent"]
        try:
            relative = self.run_root.resolve().relative_to(self.repository_path)
            ignored.append(relative.as_posix())
        except ValueError:
            pass
        return tuple(dict.fromkeys(ignored))

    def _drop_exit_messages(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        messages = getattr(runtime.agent, "messages", None)
        if isinstance(messages, list):
            runtime.agent.messages = drop_exit_messages(messages)

    def _enable_execution_tools(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        model = getattr(runtime.agent, "model", None)
        if model is None:
            return
        tools = lite_llm_memory_tools()
        existing = list(getattr(model, "extra_tools", None) or [])
        names = {
            str(item.get("name"))
            for item in existing
            if isinstance(item, Mapping)
        }
        for tool in tools:
            if tool["name"] not in names:
                existing.append(tool)
        try:
            model.extra_tools = existing
        except Exception:
            return

    def queue_context(self, *texts: str) -> None:
        for text in texts:
            normalized = text.strip()
            if normalized:
                self._queued_context.append(normalized)

    def take_context(self) -> tuple[str, ...]:
        values = tuple(self._queued_context)
        self._queued_context.clear()
        return values

    def build_driver(
        self,
        *,
        native_compaction_timeout_seconds: float,
        native_compaction_enabled: bool,
    ) -> MiniSweAgentHarnessDriver:
        del native_compaction_timeout_seconds, native_compaction_enabled
        return MiniSweAgentHarnessDriver(self)

    def close(self) -> None:
        if self._runtime is None:
            return
        output = self._runtime.agent.config.output_path
        if output:
            self._runtime.agent.save(output)


class MiniSweAgentHarnessDriver:
    """Map one mini trajectory to the provider-neutral five-stage event contract."""

    def __init__(self, adapter: MiniSweAgentHarnessAdapter) -> None:
        if adapter.thread_id is None or adapter._runtime is None:
            raise ValueError("MiniSweAgentHarnessDriver requires completed Planning")
        self.adapter = adapter
        self._context_transport = MiniSweAgentContextTransport(adapter)
        self._memory_tool_handler: Callable[[DynamicToolInvocation], DynamicToolResult] | None = None
        self._memory_response_observer: (
            Callable[[DynamicToolInvocation, DynamicToolResult], None] | None
        ) = None
        self._review_criteria_provider: Callable[[], Sequence[object]] | None = None
        self._fallback_review_criteria: list[dict[str, object]] = []

    def bind_memory_tool_handler(
        self,
        handler: Callable[[DynamicToolInvocation], DynamicToolResult],
        response_observer: Callable[[DynamicToolInvocation, DynamicToolResult], None] | None = None,
    ) -> None:
        if self._memory_tool_handler is not None:
            raise RuntimeError("a dynamic memory-tool handler is already bound")
        self._memory_tool_handler = handler
        self._memory_response_observer = response_observer

    def bind_current_review_criteria(
        self,
        provider: Callable[[], Sequence[object]],
    ) -> None:
        if self._review_criteria_provider is not None:
            raise RuntimeError("current review criteria are already bound")
        self._review_criteria_provider = provider

    def _official_review_criteria(self) -> list[object]:
        if self._review_criteria_provider is not None:
            return list(self._review_criteria_provider() or ())
        return list(self._fallback_review_criteria)

    def _align_review_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        official = self._official_review_criteria()
        raw_coverage = arguments.get("requirement_coverage")
        if not official or not isinstance(raw_coverage, list):
            return arguments
        aligned = align_mini_requirement_coverage(official, raw_coverage)
        if aligned is None:
            return arguments
        rewritten = dict(arguments)
        rewritten["requirement_coverage"] = aligned
        return rewritten

    def _queue_official_review_texts(self) -> None:
        official = self._official_review_criteria()
        slots = _review_slots(official)
        if not slots:
            return
        lines = []
        for index, slot in enumerate(slots, start=1):
            requirement = _normalized_review_text(_criterion_field(slot, "requirement_text"))
            outcome = _normalized_review_text(_criterion_field(slot, "observable_outcome"))
            lines.append(
                f"{index}. requirement_text: {requirement}\n"
                f"   observable_outcome: {outcome}"
            )
        self.adapter.queue_context(
            "Official current-milestone acceptance texts for review_current_milestone. "
            "Copy requirement_text and observable_outcome exactly, one row per item. "
            "Do not add Task preamble or branch/commit instructions as extra coverage.\n"
            + "\n".join(lines)
        )

    def capabilities(self) -> HarnessCapabilities:
        return HarnessCapabilities(
            harness_name="mini-swe-agent",
            model_name=self.adapter.model,
            tokenizer_id=None,
            context_limit=self.adapter.provider.model_context_window,
            supports_plan_mode=True,
            supports_incremental_plan_updates=False,
            supports_thread_resume=False,
            supports_native_compaction=False,
            supports_token_usage_events=True,
            supports_tool_lifecycle_events=True,
            supports_context_replacement=True,
        )

    def context_transport(self) -> MiniSweAgentContextTransport:
        return self._context_transport

    @staticmethod
    def _usage(message: Mapping[str, Any]) -> Mapping[str, Any]:
        usage = message.get("usage")
        if not isinstance(usage, Mapping):
            extra = message.get("extra")
            response = extra.get("response") if isinstance(extra, Mapping) else None
            usage = response.get("usage") if isinstance(response, Mapping) else None
        return usage if isinstance(usage, Mapping) else {}

    @classmethod
    def _usage_payload(
        cls,
        message: Mapping[str, Any],
        *,
        cumulative_input: int,
        cumulative_output: int,
        context_limit: int | None,
    ) -> tuple[Mapping[str, Any], int, int]:
        usage = cls._usage(message)
        input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
        cumulative_input += max(0, input_tokens)
        cumulative_output += max(0, output_tokens)
        last_total = max(0, input_tokens) + max(0, output_tokens)
        return (
            {
                "token_usage": {
                    "last": {
                        "inputTokens": max(0, input_tokens),
                        "outputTokens": max(0, output_tokens),
                        "totalTokens": last_total,
                    },
                    "cumulative": {
                        "inputTokens": cumulative_input,
                        "outputTokens": cumulative_output,
                        "totalTokens": cumulative_input + cumulative_output,
                    },
                    "modelContextWindow": context_limit,
                    "nativeCompaction": "UNSUPPORTED",
                }
            },
            cumulative_input,
            cumulative_output,
        )

    def _append_context(self) -> None:
        runtime = self.adapter._runtime
        assert runtime is not None
        for text in self.adapter.take_context():
            runtime.agent.add_messages(runtime.agent.model.format_message(role="user", content=text))

    def _pending_fence_reason(self, turn_id: str) -> str | None:
        """Take the one semantic fence for this Mini Turn, if the coordinator raised it.

        Codex honors the fence at the next item boundary and stamps a single
        ``TURN_COMPLETED``. Mini used to skip that take after memory tools, then
        emit ``TURN_QUIESCED`` plus an unstamped ``TURN_COMPLETED`` after bash —
        the first crashes on a second ``review_current_milestone``, the second
        double-submits acceptance. The five-stage kernel (one fence per Turn,
        acceptance only at a stamped semantic boundary) stays unchanged.
        """

        fence = self._context_transport.take_turn_fence(turn_id)
        return None if fence is None else fence.reason

    @staticmethod
    def _accessed_paths(command: str, repository: Path) -> tuple[str, ...]:
        paths: list[str] = []
        tokens = [match.group("path") for match in _PATH_TOKEN.finditer(command)]
        from .command_semantics import command_observation_scope

        tokens.extend(sorted(command_observation_scope(command).paths))
        for raw in tokens:
            token = _usable_repo_relpath(raw)
            if token is None:
                continue
            candidate = Path(token)
            if candidate.is_absolute():
                try:
                    relative = candidate.resolve().relative_to(repository)
                except (OSError, ValueError):
                    continue
            else:
                try:
                    relative = (repository / candidate).resolve().relative_to(repository)
                except (OSError, ValueError):
                    continue
            value = _usable_repo_relpath(relative.as_posix())
            if value is None or value in paths:
                continue
            paths.append(value)
        return tuple(paths[:32])

    @staticmethod
    def _revision_event(
        *,
        mapper: CodexEventMapper,
        tracker: WorkspaceRevisionTracker,
        source: HarnessEvent,
    ) -> HarnessEvent | None:
        try:
            receipt = tracker.capture(source_event_id=source.source_event_id)
        except OSError:
            return None
        if not receipt.changed:
            return None
        mapper.update_revision(receipt.revision_id)
        changed_symbols = tuple(
            item
            for item in receipt.symbol_bindings
            if item.reference_kind == "SymbolReference" and item.change_scope == "CHANGED_SYMBOL"
        )
        return mapper.local_event(
            HarnessEventType.WORKSPACE_REVISION_ADVANCED,
            provider_method="workspace/revision/advanced",
            turn_id=source.turn_id,
            discriminator=source.harness_event_id,
            payload={
                "revision_id": receipt.revision_id,
                "previous_revision_id": receipt.previous_revision_id,
                "paths": list(receipt.changed_paths),
                "source_harness_event_id": source.harness_event_id,
                "source_provider_event_id": source.source_event_id,
                "detection": "MINI_POST_ACTION_RECONCILIATION",
                "semantic_progress": receipt.semantic_progress,
                "recent_symbols": [item.canonical_entity_id for item in changed_symbols],
                "symbol_bindings": [primitive(item) for item in changed_symbols],
            },
        )

    def _run_memory_tool(
        self,
        *,
        mapper: CodexEventMapper,
        turn_id: str,
        action: Mapping[str, Any],
        action_index: int,
    ) -> tuple[HarnessEvent, dict[str, Any]]:
        tool = str(action.get("tool") or "")
        call_id = str(action.get("call_id") or action.get("tool_call_id") or f"memory-{action_index}")
        raw_arguments = action.get("arguments", {})
        arguments = dict(raw_arguments) if isinstance(raw_arguments, Mapping) else {}
        if tool == "review_current_milestone":
            arguments = self._align_review_arguments(arguments)
        if self._memory_tool_handler is None:
            result = DynamicToolResult(
                success=False,
                text=f"memory tool {tool} was called before runtime binding",
            )
        else:
            invocation = DynamicToolInvocation(
                request_id=f"mini-{call_id}",
                call_id=call_id,
                tool=tool,
                arguments=arguments,
                thread_id=mapper.thread_id,
                turn_id=turn_id,
            )
            result = self._memory_tool_handler(invocation)
            if self._memory_response_observer is not None:
                self._memory_response_observer(invocation, result)
        event = mapper.local_event(
            HarnessEventType.MEMORY_TOOL_RESULT,
            provider_method="item/tool/call",
            turn_id=turn_id,
            discriminator=call_id,
            payload={
                "call_id": call_id,
                "tool": tool,
                "arguments": arguments,
                "success": result.success,
                "result_digest": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
                "delivery_id": result.delivery_id,
                "entity_refs": list(result.entity_refs),
                "evidence_handles": list(result.evidence_handles),
                "runtime_metadata": dict(result.runtime_metadata),
            },
        )
        return event, {
            "output": result.text,
            "returncode": 0 if result.success else 1,
            "exception_info": "" if result.success else tool,
        }

    def _terminal_event(
        self,
        mapper: CodexEventMapper,
        turn_id: str,
        *,
        status: str,
        submission: str = "",
        error: BaseException | None = None,
        quiescence_reason: str | None = None,
    ) -> HarnessEvent:
        error_payload = (
            {"type": type(error).__name__, "message": str(error)[:1000]}
            if error is not None
            else None
        )
        payload: dict[str, Any] = {
            "turn": {"id": turn_id, "status": status},
            "submission": submission,
            "unsupported_capabilities": [
                "native_thread_resume",
                "native_context_compaction",
                "codex_app_server_events",
            ],
        }
        if error_payload is not None:
            payload["error"] = error_payload
            payload["provider_error"] = str(error)[:1000]
        if quiescence_reason:
            payload["quiescence_reason"] = quiescence_reason
        return mapper.local_event(
            HarnessEventType.TURN_COMPLETED,
            provider_method="mini/turn/completed",
            turn_id=turn_id,
            discriminator=f"{status}:{time.time_ns()}",
            payload=payload,
        )

    def events(
        self,
        *,
        user_task: str,
        run_id: str,
        branch_id: str,
        revision_tracker: WorkspaceRevisionTracker,
        initial_milestone: MilestoneSpec | None = None,
        initial_semantic_route: Mapping[str, object] | None = None,
    ) -> Iterator[HarnessEvent]:
        del user_task
        self._fallback_review_criteria = _milestone_review_criteria(initial_milestone)
        runtime = self.adapter._runtime
        assert runtime is not None
        self.adapter._enable_execution_tools()
        self.adapter._drop_exit_messages()
        thread_id = self.adapter.thread_id
        assert thread_id is not None
        revision_id = revision_tracker.current() or revision_tracker.capture(
            source_event_id="mini-execution-start"
        ).revision_id
        git_baseline = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=self.adapter.repository_path,
            check=False,
            capture_output=True,
            text=True,
        )
        progress_baseline = git_baseline.stdout.strip() if git_baseline.returncode == 0 else None
        mapper = CodexEventMapper(
            run_id=run_id,
            branch_id=branch_id,
            thread_id=thread_id,
            revision_id=revision_id,
        )
        cumulative_input = 0
        cumulative_output = 0
        turn_number = 0
        pending = runtime.pending_response
        runtime.pending_response = None
        while True:
            pending_thread_id = self._context_transport.pending_thread_id
            if pending_thread_id is not None and pending_thread_id != thread_id:
                self.adapter.begin_logical_thread(pending_thread_id)
                thread_id = pending_thread_id
                mapper = CodexEventMapper(
                    run_id=run_id,
                    branch_id=branch_id,
                    thread_id=thread_id,
                    revision_id=mapper.revision_id,
                )
                pending = None
            turn_number += 1
            turn_id = stable_id(
                "mini_turn_",
                {"run": run_id, "thread": thread_id, "ordinal": turn_number},
            )
            yield mapper.turn_started(turn_id)
            route = json.dumps(
                initial_semantic_route or {},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            self.adapter.queue_context(
                "Current five-stage route card (navigation only; continue natural coding):\n" + route
            )
            self._queue_official_review_texts()
            self._append_context()
            submitted = False
            terminal_status = "completed"
            submission = ""
            terminal_error: BaseException | None = None
            terminal_reason: str | None = None
            consecutive_format_errors = 0
            steps_in_turn = 0
            while not submitted:
                try:
                    response = runtime.agent.query() if pending is None else pending
                    pending = None
                    self.adapter._drop_exit_messages()
                    consecutive_format_errors = 0
                    steps_in_turn += 1
                    text = self.adapter._message_text(response)
                    if text:
                        yield mapper.local_event(
                            HarnessEventType.ITEM_COMPLETED,
                            provider_method="mini/model/response",
                            turn_id=turn_id,
                            discriminator=f"model:{runtime.agent.n_calls}:{digest(text)}",
                            payload={
                                "item": {
                                    "id": stable_id(
                                        "mini_message_", {"turn": turn_id, "call": runtime.agent.n_calls}
                                    ),
                                    "type": "agentMessage",
                                    "status": "completed",
                                    "text": text,
                                }
                            },
                        )
                    usage, cumulative_input, cumulative_output = self._usage_payload(
                        response,
                        cumulative_input=cumulative_input,
                        cumulative_output=cumulative_output,
                        context_limit=self.adapter.provider.model_context_window,
                    )
                    yield mapper.local_event(
                        HarnessEventType.TOKEN_USAGE_UPDATED,
                        provider_method="mini/tokenUsage/updated",
                        turn_id=turn_id,
                        discriminator=f"usage:{runtime.agent.n_calls}",
                        payload=usage,
                    )
                    # TOKEN_USAGE can raise PROVIDER_CONTEXT_LIMIT. Take that
                    # fence here so a later review/bash cannot request a second
                    # semantic fence on the same Mini Turn.
                    fence_reason = self._pending_fence_reason(turn_id)
                    if fence_reason is not None:
                        submitted = True
                        terminal_status = "completed"
                        terminal_reason = fence_reason
                        break
                    actions = parse_mini_tool_actions(response)
                    observations: list[dict[str, Any]] = []
                    for action_index, action in enumerate(actions, start=1):
                        if str(action.get("tool") or "") in _MEMORY_TOOL_NAMES:
                            memory_event, memory_output = self._run_memory_tool(
                                mapper=mapper,
                                turn_id=turn_id,
                                action=action,
                                action_index=action_index,
                            )
                            yield memory_event
                            observations.append(memory_output)
                            fence_reason = self._pending_fence_reason(turn_id)
                            if fence_reason is not None:
                                submitted = True
                                terminal_status = "completed"
                                terminal_reason = fence_reason
                                break
                            self._append_context()
                            continue
                        command = str(action.get("command", ""))
                        item_id = stable_id(
                            "mini_action_",
                            {"turn": turn_id, "call": runtime.agent.n_calls, "index": action_index},
                        )
                        accessed = self._accessed_paths(command, self.adapter.repository_path)
                        started_item = {
                            "id": item_id,
                            "type": "commandExecution",
                            "status": "inProgress",
                            "command": command,
                            "testSelector": mini_test_selector(command),
                            "cwd": str(self.adapter.repository_path),
                            "commandActions": [{"type": "read", "path": path} for path in accessed],
                        }
                        yield mapper.local_event(
                            HarnessEventType.ITEM_STARTED,
                            provider_method="mini/action/started",
                            turn_id=turn_id,
                            discriminator=item_id,
                            payload={"item": started_item},
                        )
                        yield mapper.local_event(
                            HarnessEventType.TOOL_INTENT,
                            provider_method="mini/action/intent",
                            turn_id=turn_id,
                            discriminator=item_id,
                            payload={"item": started_item, "accessed_paths": list(accessed)},
                        )
                        try:
                            output = runtime.agent.env.execute(action)
                        except BaseException as error:
                            interrupt_name = type(error).__name__
                            if interrupt_name not in {
                                "Submitted",
                                "InterruptAgentFlow",
                                "LimitsExceeded",
                                "TimeExceeded",
                            }:
                                raise
                            runtime.agent.add_messages(*getattr(error, "messages", ()))
                            last = (getattr(error, "messages", None) or [None])[-1] or {}
                            extra = last.get("extra", {}) if isinstance(last, Mapping) else {}
                            is_submitted = interrupt_name == "Submitted"
                            if is_submitted and not workspace_has_progress(
                                self.adapter.repository_path,
                                progress_baseline,
                                ignore_roots=self.adapter._progress_ignore_roots(),
                            ):
                                self.adapter._drop_exit_messages()
                                rejected = (
                                    "Submitted was rejected: the workspace has no change "
                                    "relative to the planning baseline. Continue editing and testing."
                                )
                                runtime.agent.add_messages(
                                    runtime.agent.model.format_message(role="user", content=rejected)
                                )
                                output = {
                                    "output": rejected,
                                    "returncode": 1,
                                    "exception_info": "EmptySubmissionRejected",
                                }
                            else:
                                submission = str(extra.get("submission", ""))
                                terminal_status = (
                                    "completed"
                                    if is_submitted
                                    else str(extra.get("exit_status", "failed"))
                                )
                                output = {
                                    "output": submission or str(last.get("content", "")),
                                    "returncode": 0 if is_submitted else -1,
                                    "exception_info": "" if is_submitted else type(error).__name__,
                                }
                                submitted = True
                                if is_submitted:
                                    self.adapter._drop_exit_messages()
                        observations.append(output)
                        completed_item = {
                            **started_item,
                            "status": "completed",
                            "success": int(output.get("returncode", -1)) == 0,
                            "exitCode": int(output.get("returncode", -1)),
                            "aggregatedOutput": str(output.get("output", "")),
                        }
                        result_event = mapper.local_event(
                            HarnessEventType.TOOL_RESULT,
                            provider_method="mini/action/completed",
                            turn_id=turn_id,
                            discriminator=item_id,
                            payload={
                                "item": completed_item,
                                "partial": False,
                                "accessed_paths": list(accessed),
                            },
                        )
                        yield result_event
                        fence_reason = self._pending_fence_reason(turn_id)
                        if fence_reason is not None:
                            submitted = True
                            terminal_status = "completed"
                            terminal_reason = fence_reason
                            break
                        revision_event = self._revision_event(
                            mapper=mapper,
                            tracker=revision_tracker,
                            source=result_event,
                        )
                        if revision_event is not None:
                            yield mapper.local_event(
                                HarnessEventType.FILE_CHANGED,
                                provider_method="mini/workspace/changed",
                                turn_id=turn_id,
                                discriminator=item_id,
                                payload={
                                    "item": {
                                        "id": stable_id("mini_file_change_", item_id),
                                        "type": "fileChange",
                                        "status": "completed",
                                        "changes": [],
                                    },
                                    "paths": list(revision_event.payload.get("paths", ())),
                                    "provisional": True,
                                },
                            )
                            fence_reason = self._pending_fence_reason(turn_id)
                            if fence_reason is not None:
                                submitted = True
                                terminal_status = "completed"
                                terminal_reason = fence_reason
                                break
                            yield revision_event
                        fence_reason = self._pending_fence_reason(turn_id)
                        if fence_reason is not None:
                            submitted = True
                            terminal_status = "completed"
                            terminal_reason = fence_reason
                            break
                        self._append_context()
                    if observations:
                        # Always pair function_call with function_call_output,
                        # including when the turn fences after the bash step.
                        runtime.agent.add_messages(
                            *runtime.agent.model.format_observation_messages(
                                response,
                                observations,
                                runtime.agent.get_template_vars(),
                            )
                        )
                    if not submitted and steps_in_turn >= self.adapter.turn_step_limit:
                        self._context_transport.request_task_continuation(
                            "Mini step boundary. Continue the current milestone from the "
                            "durable route and repository evidence. Do not treat this fence "
                            "as task completion."
                        )
                        terminal_status = "interrupted"
                        terminal_reason = "MINI_STEP_BOUNDARY"
                        submitted = True
                except BaseException as error:
                    messages = getattr(error, "messages", ())
                    if type(error).__name__ == "FormatError" and messages:
                        consecutive_format_errors += 1
                        runtime.agent.add_messages(*messages)
                        if consecutive_format_errors < 3:
                            continue
                    terminal_error = error
                    terminal_status = "failed"
                    try:
                        runtime.agent.handle_uncaught_exception(error)
                    except Exception:
                        pass
                    self.adapter._drop_exit_messages()
                    submitted = True
            runtime.agent.save(runtime.agent.config.output_path)
            self.adapter._drop_exit_messages()
            yield self._terminal_event(
                mapper,
                turn_id,
                status=terminal_status,
                submission=submission,
                error=terminal_error,
                quiescence_reason=terminal_reason,
            )
            if not self._context_transport.needs_followup_turn:
                return
            continuation = self._context_transport.take_continuation_request()
            if continuation is not None:
                self.adapter.queue_context(continuation.prompt)
            self._append_context()
