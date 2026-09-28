from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..config import CodexProviderConfiguration
from .contracts import COMPACTION_TOOL_ABI_INVARIANT
from .resource_envelope import resource_envelope_environment
from .transport import AppServerProtocolError

_DISABLED_NATIVE_COMPACTION_TOKEN_LIMIT = 2_147_483_647


@dataclass(frozen=True, slots=True)
class CodexRuntimeLaunch:
    executable: str
    provider_id: str
    app_server_args: tuple[str, ...]
    environment: Mapping[str, str]
    sqlite_home: str | None


def resolve_codex_executable(explicit: str | None = None) -> str:
    """Resolve one reproducible Codex Runtime without silently changing versions."""

    configured = explicit or os.environ.get("HOMY_CODEX_BIN")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_absolute() or candidate.parent != Path("."):
            resolved = candidate.resolve()
            if not resolved.is_file() or not os.access(resolved, os.X_OK):
                raise AppServerProtocolError(
                    f"configured Codex Runtime is not executable: {resolved}"
                )
            return str(resolved)
        on_path = shutil.which(configured)
        if on_path is None:
            raise AppServerProtocolError(f"configured Codex Runtime is not on PATH: {configured}")
        return on_path

    try:
        from codex_cli_bin import bundled_codex_path

        bundled = bundled_codex_path().resolve()
        if bundled.is_file() and os.access(bundled, os.X_OK):
            return str(bundled)
    except (ImportError, OSError):
        pass

    on_path = shutil.which("codex")
    if on_path is not None:
        return on_path
    raise AppServerProtocolError(
        "no Codex Runtime found; install the pinned project dependency, set HOMY_CODEX_BIN, "
        "or pass --codex-bin"
    )


def _override(key: str, value: object) -> tuple[str, str]:
    return "--config", f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"


def build_runtime_launch(
    provider: CodexProviderConfiguration,
    *,
    executable: str | None = None,
    environment: Mapping[str, str] | None = None,
    sqlite_home: Path | None = None,
) -> CodexRuntimeLaunch:
    """Build credential-safe App Server arguments for one configured Provider."""

    provider.validate()
    runtime = resolve_codex_executable(executable)
    inherited = dict(os.environ if environment is None else environment)
    # The model's shell commands inherit this environment.  Pin the
    # "auto"-sizing knobs of common parallel tools to the sandbox's real CPU
    # budget so `pytest -n auto` cannot spawn a host-sized worker swarm inside
    # a cgroup-limited container (2.2.93 pressure run r4: 124 workers, 8 GiB
    # cgroup, OOM killer took the App Server with them).
    inherited.update(resource_envelope_environment(inherited))
    arguments: list[str] = []
    resolved_sqlite_home: str | None = None
    if sqlite_home is not None:
        resolved = sqlite_home.expanduser().resolve()
        if not resolved.is_dir():
            raise AppServerProtocolError(
                f"configured Codex SQLite state directory does not exist: {resolved}"
            )
        resolved_sqlite_home = str(resolved)
        # CODEX_SQLITE_HOME is the public shell-scoped location override. The
        # explicit config value is also supplied because user config.toml has
        # higher precedence than the environment variable. Both therefore name
        # the same per-Run durable directory.
        inherited["CODEX_SQLITE_HOME"] = resolved_sqlite_home
        arguments.extend(_override("sqlite_home", resolved_sqlite_home))
    if provider.custom:
        assert provider.base_url is not None
        assert provider.api_key_env is not None
        if not inherited.get(provider.api_key_env, "").strip():
            raise AppServerProtocolError(
                f"custom Provider credential environment variable is not set: "
                f"{provider.api_key_env}"
            )
        prefix = f"model_providers.{provider.id}"
        excluded = sorted(
            {
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                provider.api_key_env,
            }
        )
        overrides = (
            _override("model_provider", provider.id),
            _override(f"{prefix}.name", provider.name),
            _override(f"{prefix}.base_url", provider.base_url),
            _override(f"{prefix}.env_key", provider.api_key_env),
            _override(f"{prefix}.wire_api", provider.wire_api),
            _override(f"{prefix}.requires_openai_auth", False),
            _override("features.shell_snapshot", False),
            _override("shell_environment_policy.exclude", excluded),
            _override("model_reasoning_summary", "none"),
            _override("model_supports_reasoning_summaries", False),
        )
        for pair in overrides:
            arguments.extend(pair)
    if provider.model_context_window is not None:
        arguments.extend(_override("model_context_window", provider.model_context_window))
    if not provider.native_compaction_enabled:
        # Current Codex builds do not expose an auto-compaction opt-out. Move
        # the threshold beyond every supported model window; any unsolicited
        # Provider compaction event is still observation-only in the runtime.
        arguments.extend(
            _override(
                "model_auto_compact_token_limit",
                _DISABLED_NATIVE_COMPACTION_TOKEN_LIMIT,
            )
        )
        arguments.extend(_override("model_auto_compact_token_limit_scope", "total"))
    elif provider.model_auto_compact_token_limit is not None:
        arguments.extend(
            _override(
                "model_auto_compact_token_limit",
                provider.model_auto_compact_token_limit,
            )
        )
        arguments.extend(
            _override(
                "model_auto_compact_token_limit_scope",
                provider.model_auto_compact_token_limit_scope,
            )
        )
    # The compactor is allowed to summarize task history, but not to rewrite
    # the pinned Codex tool ABI.  Append this invariant even when a deployment
    # supplies its own quality-oriented compaction prompt.
    if provider.native_compaction_enabled:
        effective_compact_prompt = "\n\n".join(
            part
            for part in (provider.compact_prompt, COMPACTION_TOOL_ABI_INVARIANT)
            if part is not None and part.strip()
        )
        arguments.extend(_override("compact_prompt", effective_compact_prompt))
    return CodexRuntimeLaunch(
        executable=runtime,
        provider_id=provider.id,
        app_server_args=tuple(arguments),
        environment=inherited,
        sqlite_home=resolved_sqlite_home,
    )
