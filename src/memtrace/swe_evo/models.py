from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return (value,) if value else ()
        value = decoded
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    return ()


@dataclass(frozen=True, slots=True)
class SweEvoInstance:
    repo: str
    instance_id: str
    base_commit: str
    test_patch: str
    problem_statement: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    start_version: str
    end_version: str
    image: str
    test_cmds: str
    official_row: Mapping[str, object]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SweEvoInstance":
        instance_id = str(value.get("instance_id", ""))
        repo = str(value.get("repo", ""))
        base_commit = str(value.get("base_commit", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", instance_id):
            raise ValueError("SWE-EVO instance_id contains unsafe path characters")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("SWE-EVO repository identifier is invalid")
        if not re.fullmatch(r"[0-9a-f]{40}", base_commit):
            raise ValueError("SWE-EVO base_commit must be a full SHA-1")
        problem = str(value.get("problem_statement", "")).strip()
        if not problem:
            raise ValueError("SWE-EVO problem_statement is empty")
        sanitized = dict(value)
        sanitized["patch"] = ""
        return cls(
            repo=repo,
            instance_id=instance_id,
            base_commit=base_commit,
            test_patch=str(value.get("test_patch", "")),
            problem_statement=problem,
            fail_to_pass=_strings(value.get("FAIL_TO_PASS", ())),
            pass_to_pass=_strings(value.get("PASS_TO_PASS", ())),
            start_version=str(value.get("start_version", "")),
            end_version=str(value.get("end_version", value.get("version", ""))),
            image=str(value.get("image", "")),
            test_cmds=str(value.get("test_cmds", "")),
            official_row=sanitized,
        )

    def public_metadata(self) -> dict[str, object]:
        fail_to_pass_payload = json.dumps(
            self.fail_to_pass,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "start_version": self.start_version,
            "end_version": self.end_version,
            "image": self.image,
            "fail_to_pass_count": len(self.fail_to_pass),
            "fail_to_pass_sha256": hashlib.sha256(fail_to_pass_payload).hexdigest(),
            "pass_to_pass_count": len(self.pass_to_pass),
            "problem_chars": len(self.problem_statement),
            "test_patch_lines": len(self.test_patch.splitlines()),
            "gold_patch_exposed": False,
        }


def _load_json_dataset(path: Path, instance_id: str) -> Mapping[str, object] | None:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    elif path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else [value]
    else:
        return None
    matches = [row for row in rows if str(row.get("instance_id")) == instance_id]
    if len(matches) != 1:
        raise ValueError(f"expected one SWE-EVO instance {instance_id!r}, found {len(matches)}")
    result = dict(matches[0])
    result["patch"] = ""
    return result


def load_instance(
    instance_id: str,
    dataset_path: Path,
    *,
    dataset_python: Path | None = None,
) -> SweEvoInstance:
    path = dataset_path.expanduser().resolve()
    local = _load_json_dataset(path, instance_id) if path.is_file() else None
    if local is not None:
        return SweEvoInstance.from_mapping(local)
    # Preserve a venv interpreter path. Resolving its symlink to the base
    # interpreter would silently lose the venv's datasets installation.
    python = (dataset_python or Path(sys.executable)).expanduser().absolute()
    bridge = Path(__file__).resolve().with_name("dataset_bridge.py")
    completed = subprocess.run(
        [str(python), str(bridge), "--dataset", str(path), "--instance-id", instance_id],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SWE-EVO dataset bridge failed: "
            + (completed.stderr.strip() or completed.stdout.strip())[:2000]
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, Mapping):
        raise ValueError("SWE-EVO dataset bridge returned a non-object")
    return SweEvoInstance.from_mapping(value)


def list_instance_ids(
    dataset_path: Path,
    *,
    dataset_python: Path | None = None,
) -> tuple[str, ...]:
    path = dataset_path.expanduser().resolve()
    if path.is_file() and path.suffix in {".json", ".jsonl"}:
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            rows = value if isinstance(value, list) else [value]
        identifiers = tuple(str(row.get("instance_id", "")) for row in rows)
    else:
        python = (dataset_python or Path(sys.executable)).expanduser().absolute()
        bridge = Path(__file__).resolve().with_name("dataset_bridge.py")
        completed = subprocess.run(
            [str(python), str(bridge), "--dataset", str(path), "--list-instance-ids"],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "SWE-EVO dataset bridge failed: "
                + (completed.stderr.strip() or completed.stdout.strip())[:2000]
            )
        value = json.loads(completed.stdout)
        if not isinstance(value, list):
            raise ValueError("SWE-EVO dataset bridge returned a non-list")
        identifiers = tuple(str(item) for item in value)
    if not identifiers or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", item) for item in identifiers):
        raise ValueError("SWE-EVO dataset contains an invalid or empty instance_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("SWE-EVO dataset contains duplicate instance IDs")
    return identifiers


def _test_preview(instance: SweEvoInstance, *, limit: int = 5) -> str:
    visible = instance.fail_to_pass[:limit]
    lines = [f"- {name}" for name in visible]
    remaining = len(instance.fail_to_pass) - len(visible)
    if remaining:
        lines.append(f"- ……其余 {remaining} 项由验证入口读取，不在任务文本中重复展开")
    return "\n".join(lines) or "- 数据集指定测试"


def build_task_text(instance: SweEvoInstance, verification_command: str) -> str:
    tests = _test_preview(instance)
    return (
        f"SWE-EVO 实例：{instance.instance_id}\n"
        f"仓库：{instance.repo}\n"
        f"基线版本：{instance.start_version}\n"
        f"目标版本：{instance.end_version}\n\n"
        f"任务说明：\n{instance.problem_statement}\n\n"
        f"最低验收测试数量：{len(instance.fail_to_pass)}\n"
        f"测试标识预览：\n{tests}\n"
        "完整修复测试与私有回归测试清单由 benchmark runner 保存在只读验证资产中；"
        "不要复制、修改或展开它。\n\n"
        "统一的隔离验证入口由 benchmark runner 在 Codex 沙箱外托管（HOST_MANAGED）：\n"
        f"{verification_command}\n"
        "不要在 shell 中执行该入口，也不要尝试访问 Docker socket。完成预期代码修改后，"
        "调用 verify_current_milestone；运行时会同时执行目标修复测试和 PASS_TO_PASS 回归"
        "矩阵，并把不可由模型改写的真实结果作为当前 revision 的验收证据返回。\n\n"
        "约束：\n"
        "- 先调查当前实现，再完成最小且完整的代码修复。\n"
        "- 不修改、删除或跳过测试，不读取任何 gold patch。\n"
        "- 不创建 Git commit；最终说明修改内容和验证证据。\n"
    )
