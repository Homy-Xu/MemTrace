from __future__ import annotations

import re
from dataclasses import dataclass

from ..contracts import digest, stable_id


@dataclass(frozen=True, slots=True)
class TaskRequirement:
    requirement_id: str
    ordinal: int
    text: str
    text_digest: str
    modality: str
    category: str
    required: bool
    source_line: int


_LIST_ITEM = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|[a-zA-Z][.)]\s+)(.+?)\s*$")
_NORMATIVE = re.compile(
    r"\b(?:must|shall|required|should|need(?:s)?\s+to|ensure|support|implement|"
    r"preserve|return|reject|accept|do\s+not|never|without)\b|"
    r"(?:必须|不得|不能|需要|应当|应该|确保|支持|实现|保持|返回|拒绝|禁止|不要|"
    r"不(?:修改|读取|创建|允许|应|可|使用|删除|跳过)|请勿|避免)",
    re.IGNORECASE,
)
# Imperative or subject-led requirement lines often omit an English modal.
# Treat those lines as checklist clauses while preserving their full text.
_REQUIREMENT_LEAD = re.compile(
    r"^(?:add|define|use|generate|implement|verify|ensure|preserve|return|"
    r"direct|explicit|implicit|public|before|after|important|create|"
    r"增加|添加|定义|使用|生成|实现|验证|确保|保持|返回|显式|隐式|"
    r"公共|在|然后|请|创建|检查)\b",
    re.IGNORECASE,
)
_OPTIONAL = re.compile(r"\b(?:may|optional|if\s+possible)\b|(?:可以|可选|尽量)", re.IGNORECASE)
_SHOULD = re.compile(r"\bshould\b|(?:应当|应该|建议)", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(?:must\s+not|shall\s+not|do\s+not|never|forbid(?:den)?)\b|"
    r"(?:不得|不能|禁止|不要)",
    re.IGNORECASE,
)

# Issue-tracker noise.  Bug-report templates, quoted logs, tracebacks and
# bare test/file identifiers are context for the model, never checklist
# clauses: a Requirement must be something the final workspace can satisfy.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG_LINE = re.compile(r"^\s*</?[a-zA-Z][^>]*>\s*$")
_TEMPLATE_HEADINGS = frozenset(
    {
        "describe the bug",
        "description",
        "expected behavior",
        "expected behaviour",
        "expected result",
        "expected results",
        "actual behavior",
        "actual behaviour",
        "actual result",
        "actual results",
        "steps to reproduce",
        "to reproduce",
        "how to reproduce",
        "reproduction",
        "reproducer",
        "minimal example",
        "minimal reproducible example",
        "code sample",
        "code example",
        "system details",
        "system information",
        "system info",
        "environment",
        "versions",
        "version",
        "additional context",
        "additional information",
        "screenshots",
        "problem description",
        "proposed solution",
        "possible solution",
        "possible fix",
        "alternatives",
        "checklist",
        "context",
        "summary",
        "motivation",
        "traceback",
        "error message",
        "output",
        "logs",
    }
)
_TRACEBACK_LINE = re.compile(
    r"^\s*(?:Traceback \(most recent call last\)|File \"|\s+File \"|"
    r"[A-Za-z_][\w.]*(?:Error|Exception|Warning)\b\s*[:(]|>>>\s|\.\.\.\s|\$\s|In \[\d+\]:|"
    r"Out\[\d+\]:|at 0x[0-9a-fA-F]+)"
)
_BARE_IDENTIFIER_LINE = re.compile(
    r"^\s*(?:https?://\S+|[\w./\\-]+\.(?:py|rs|ts|js|go|java|rb|cpp|c|h|md|txt|yml|yaml|toml|cfg|ini|json)"
    r"(?:::[\w\[\]\-.:]+)*|[\w./\\-]+::[\w\[\]\-.:]+)\s*$"
)
_TEMPLATE_CHECKBOX = re.compile(r"^\s*\[[ xX]\]\s*(?:I have|I've|I am|I'm|I checked|I searched)", re.IGNORECASE)


def _heading_text(stripped: str) -> str:
    text = stripped.strip("#*_ \t:")
    return " ".join(text.split()).casefold()


def _is_noise_line(stripped: str) -> bool:
    if not stripped:
        return True
    if stripped.startswith(("#", ">", "|", "…", "...")):
        return True
    if _HTML_TAG_LINE.match(stripped):
        return True
    if _TRACEBACK_LINE.match(stripped):
        return True
    if _BARE_IDENTIFIER_LINE.match(stripped):
        return True
    heading = _heading_text(stripped)
    if heading in _TEMPLATE_HEADINGS:
        return True
    # Emphasis-only short lines ("**Expected behavior**") are section titles.
    if stripped.startswith(("**", "__")) and stripped.endswith(("**", "__")) and len(heading.split()) <= 5:
        return True
    letters = sum(1 for char in stripped if char.isalpha() or "\u4e00" <= char <= "\u9fff")
    if letters < max(4, len(stripped) // 4):
        return True
    return False


def _modality(text: str, *, listed: bool) -> tuple[str, bool]:
    if _FORBIDDEN.search(text):
        return "FORBIDDEN", True
    if _OPTIONAL.search(text):
        return "MAY", False
    if _SHOULD.search(text):
        return "SHOULD", True
    return ("MUST", True) if listed or _NORMATIVE.search(text) else ("CONTEXT", False)


def _category(text: str) -> str:
    lowered = text.casefold()
    if any(word in lowered for word in ("non-goal", "out of scope", "不得", "不能", "不要")):
        return "CONSTRAINT"
    if any(word in lowered for word in ("test", "verify", "regression", "测试", "验证", "回归")):
        return "VERIFICATION"
    if any(word in lowered for word in ("performance", "latency", "memory", "性能", "延迟", "内存")):
        return "PERFORMANCE"
    if any(word in lowered for word in ("doc", "readme", "documentation", "文档", "说明")):
        return "DOCUMENTATION"
    if any(word in lowered for word in ("api", "behavior", "return", "request", "response", "行为", "返回")):
        return "BEHAVIORAL"
    if any(word in lowered for word in ("file", "class", "function", "module", "schema", "文件", "函数", "模块")):
        return "STRUCTURAL"
    return "FUNCTIONAL"


def extract_task_requirements(task_id: str, user_request: str) -> tuple[TaskRequirement, ...]:
    """Build an immutable, Plan-independent checklist from the original Task.

    The complete request remains separately digest-bound by the Registry.  This
    extractor adds addressable clauses for route coverage and final review; it
    never rewrites or summarizes their text.
    """

    candidates: list[tuple[int, str, bool]] = []
    in_fence = False
    # HTML comments are removed as blocks first: bug-report templates hide
    # multi-line instructions ("<!-- ... so you do not need to remove them! -->")
    # that would otherwise surface as FORBIDDEN clauses.  Line numbers of the
    # surviving text are preserved by blanking the comment span line by line.
    cleaned = _HTML_COMMENT.sub(lambda match: "\n" * match.group(0).count("\n"), user_request)
    for line_number, raw_line in enumerate(cleaned.splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence or _is_noise_line(stripped):
            continue
        match = _LIST_ITEM.match(raw_line)
        if match is not None:
            text = match.group(1).strip()
            if _TEMPLATE_CHECKBOX.match(text) or _is_noise_line(text):
                continue
            text = re.sub(r"^\[[ xX]\]\s*", "", text)
            if text:
                candidates.append((line_number, text, True))
            continue
        # Preserve list items as one user-authored clause. For prose, split
        # only at sentence boundaries and keep normative or imperative clauses.
        segments = tuple(
            segment.strip()
            for segment in re.split(r"(?<=[.!?。！？；;])\s+", stripped)
            if segment.strip()
        )
        for segment in segments:
            # A short lead-in that ends with a colon introduces the list that
            # follows ("Implement the feature:"); the clauses are the items.
            if segment.endswith((":", "：")) and len(segment.split()) <= 8:
                continue
            if _NORMATIVE.search(segment) or _REQUIREMENT_LEAD.search(segment):
                candidates.append((line_number, segment, False))

    if not candidates:
        compact = " ".join(user_request.split())
        if compact:
            candidates.append((1, compact, True))

    requirements: list[TaskRequirement] = []
    seen: set[str] = set()
    for source_line, text, listed in candidates:
        normalized = " ".join(text.split()).casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        modality, required = _modality(text, listed=listed)
        ordinal = len(requirements) + 1
        requirements.append(
            TaskRequirement(
                requirement_id=stable_id(
                    "requirement_",
                    {"task": task_id, "ordinal": ordinal, "text": normalized},
                ),
                ordinal=ordinal,
                text=text,
                text_digest=digest({"text": text}),
                modality=modality,
                category=_category(text),
                required=required,
                source_line=source_line,
            )
        )
    return tuple(requirements)


def semantic_tokens(text: str) -> frozenset[str]:
    lowered = " ".join(text.casefold().split())
    latin = re.findall(r"[a-z0-9_]+", lowered)
    chinese = "".join(char for char in lowered if "\u4e00" <= char <= "\u9fff")
    bigrams = [chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))]
    return frozenset((*latin, *bigrams))


def requirement_link_score(requirement: str, criterion_text: str) -> tuple[float, str]:
    left = " ".join(requirement.casefold().split())
    right = " ".join(criterion_text.casefold().split())
    if not left or not right:
        return 0.0, "EMPTY"
    if left == right:
        return 1.0, "EXACT_TEXT"
    if left in right:
        ratio = len(left) / max(len(right), 1)
        return 0.55 + 0.4 * ratio, "REQUIREMENT_CONTAINED"
    if right in left:
        ratio = len(right) / max(len(left), 1)
        return 0.5 + 0.35 * ratio, "CRITERION_CONTAINED"
    left_tokens = semantic_tokens(left)
    right_tokens = semantic_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0, "NO_TOKENS"
    overlap = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    return (overlap / union if union else 0.0), "TOKEN_OVERLAP"
