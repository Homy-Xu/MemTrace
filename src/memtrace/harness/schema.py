from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..contracts import digest
from .transport import AppServerProtocolError


@dataclass(frozen=True, slots=True)
class CodexProtocolSchemaReceipt:
    schema_digest: str
    turn_start_schema: str
    collaboration_mode_field: str
    sandbox_policy_field: str
    plan_mode_value: str
    supports_context_injection: bool = False
    supports_native_compaction: bool = False
    supports_context_compaction_events: bool = False
    supports_turn_effort: bool = False
    supports_dynamic_tools: bool = False
    supports_turn_interrupt: bool = False
    supports_output_schema: bool = False
    supports_thread_fork: bool = False


class CodexProtocolSchemaProbe:
    """Generate and inspect the installed Codex App Server schema at runtime."""

    def __init__(self, executable: str = "codex", *, timeout_seconds: float = 30.0) -> None:
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def probe(self) -> CodexProtocolSchemaReceipt:
        with tempfile.TemporaryDirectory(prefix="homy-codex-schema-") as directory:
            completed = subprocess.run(
                [
                    self.executable,
                    "app-server",
                    "generate-json-schema",
                    "--experimental",
                    "--out",
                    directory,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if completed.returncode != 0:
                raise AppServerProtocolError(
                    "unable to generate installed Codex App Server schema: "
                    + completed.stderr.strip()
                )
            schema_path = Path(directory) / "v2" / "TurnStartParams.json"
            try:
                document = json.loads(schema_path.read_text(encoding="utf-8"))
                client_requests = json.loads(
                    (Path(directory) / "ClientRequest.json").read_text(encoding="utf-8")
                )
                server_notifications = json.loads(
                    (Path(directory) / "ServerNotification.json").read_text(encoding="utf-8")
                )
                server_requests = json.loads(
                    (Path(directory) / "ServerRequest.json").read_text(encoding="utf-8")
                )
                thread_start = json.loads(
                    (Path(directory) / "v2" / "ThreadStartParams.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise AppServerProtocolError(
                    "installed Codex App Server schema is invalid"
                ) from exc
            return self.inspect(
                document,
                schema_path.name,
                client_requests=client_requests,
                server_notifications=server_notifications,
                server_requests=server_requests,
                thread_start=thread_start,
            )

    @staticmethod
    def inspect(
        document: object,
        source_name: str = "TurnStartParams.json",
        *,
        client_requests: object | None = None,
        server_notifications: object | None = None,
        server_requests: object | None = None,
        thread_start: object | None = None,
    ) -> CodexProtocolSchemaReceipt:
        if not isinstance(document, dict):
            raise AppServerProtocolError("Codex TurnStartParams schema is not an object")
        encoded = json.dumps(document, ensure_ascii=False, sort_keys=True)
        for field in ("collaborationMode", "sandboxPolicy"):
            if f'"{field}"' not in encoded:
                raise AppServerProtocolError(
                    f"installed Codex schema does not expose required {field}"
                )
        if '"plan"' not in encoded or '"default"' not in encoded:
            raise AppServerProtocolError(
                "installed Codex schema does not expose plan/default collaboration modes"
            )
        client_encoded = json.dumps(client_requests, ensure_ascii=False, sort_keys=True)
        notification_encoded = json.dumps(server_notifications, ensure_ascii=False, sort_keys=True)
        server_request_encoded = json.dumps(server_requests, ensure_ascii=False, sort_keys=True)
        thread_start_encoded = json.dumps(thread_start, ensure_ascii=False, sort_keys=True)
        properties = document.get("properties")
        supports_turn_effort = isinstance(properties, dict) and "effort" in properties
        supports_output_schema = isinstance(properties, dict) and "outputSchema" in properties
        supports_native_request = '"thread/compact/start"' in client_encoded
        supports_context_events = '"contextCompaction"' in notification_encoded
        return CodexProtocolSchemaReceipt(
            schema_digest=digest(
                {
                    "turn_start": document,
                    "client_requests": client_requests,
                    "server_notifications": server_notifications,
                    "server_requests": server_requests,
                    "thread_start": thread_start,
                }
            ),
            turn_start_schema=source_name,
            collaboration_mode_field="collaborationMode",
            sandbox_policy_field="sandboxPolicy",
            plan_mode_value="plan",
            supports_context_injection='"thread/inject_items"' in client_encoded,
            supports_native_compaction=supports_native_request and supports_context_events,
            supports_context_compaction_events=supports_context_events,
            supports_turn_effort=supports_turn_effort,
            supports_dynamic_tools=(
                '"dynamicTools"' in thread_start_encoded
                and '"item/tool/call"' in server_request_encoded
            ),
            supports_turn_interrupt='"turn/interrupt"' in client_encoded,
            supports_output_schema=supports_output_schema,
            supports_thread_fork='"thread/fork"' in client_encoded,
        )
