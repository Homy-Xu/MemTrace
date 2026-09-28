from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


class AppServerProtocolError(RuntimeError):
    """JSON-RPC failure with optional structured Provider error details."""

    def __init__(
        self,
        message: str,
        *,
        method: str | None = None,
        code: int | None = None,
        server_message: str | None = None,
        stream_closed: bool = False,
    ) -> None:
        super().__init__(message)
        self.method = method
        self.code = code
        self.server_message = server_message
        # The App Server process ended (crash, OOM kill, sandbox failure); the
        # Provider session is gone but the runtime's durable state is intact.
        self.stream_closed = stream_closed

    def is_closed_turn_steer(self) -> bool:
        return (
            self.method == "turn/steer"
            and self.code == -32600
            and self.server_message is not None
            and "no active turn to steer" in self.server_message.casefold()
        )

    def is_closed_turn_interrupt(self) -> bool:
        """Whether an interrupt raced with an already terminal Provider Turn."""

        return (
            self.method == "turn/interrupt"
            and self.code == -32600
            and self.server_message is not None
            and "no active turn to interrupt" in self.server_message.casefold()
        )


class AppServerTransport(Protocol):
    def request(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None: ...

    def respond(self, request_id: str | int, result: Mapping[str, Any]) -> None: ...

    def next_message(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class SubprocessJsonlTransport:
    """Codex App Server stdio transport using one JSON object per line."""

    def __init__(
        self,
        *,
        executable: str = "codex",
        cwd: Path | None = None,
        timeout_seconds: float = 600.0,
        extra_args: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._executable = executable
        self._cwd = cwd
        self._extra_args = tuple(extra_args)
        self._environment = None if environment is None else dict(environment)
        self._next_request_id = 1
        self._pending: deque[Mapping[str, Any]] = deque()
        self._write_lock = threading.Lock()
        self.restart_count = 0
        self.last_exit_status: int | None = None
        self.last_failure_stderr: tuple[str, ...] = ()
        self._spawn()

    def _spawn(self) -> None:
        self._incoming: queue.Queue[Mapping[str, Any] | BaseException | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=100)
        self._closed = False
        self._process = subprocess.Popen(
            [self._executable, "app-server", "--stdio", *self._extra_args],
            cwd=str(self._cwd) if self._cwd is not None else None,
            env=dict(os.environ if self._environment is None else self._environment),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        process = self._process
        incoming = self._incoming
        stderr = self._stderr
        threading.Thread(
            target=self._read_stdout, args=(process, incoming), daemon=True
        ).start()
        threading.Thread(target=self._read_stderr, args=(process, stderr), daemon=True).start()

    def restart(self) -> None:
        """Replace a dead App Server process with a fresh one.

        The Provider session (threads, turns, in-flight items) is lost with the
        old process; the caller re-runs the ``initialize`` handshake and opens a
        replacement Thread from the runtime's durable ContextImage.  Messages
        from the dead process are discarded: they belong to a session that no
        longer exists and were already applied if they were ever received.
        """

        self.last_exit_status = self._process.poll()
        self.last_failure_stderr = tuple(self._stderr)
        self.close()
        self._pending.clear()
        self.restart_count += 1
        self._spawn()

    @property
    def process_alive(self) -> bool:
        return not self._closed and self._process.poll() is None

    @staticmethod
    def _read_stdout(
        process: subprocess.Popen[str],
        incoming: queue.Queue[Mapping[str, Any] | BaseException | None],
    ) -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AppServerProtocolError("App Server emitted a non-object JSON message")
                incoming.put(value)
        except BaseException as exc:
            incoming.put(exc)
        finally:
            incoming.put(None)

    @staticmethod
    def _read_stderr(process: subprocess.Popen[str], stderr: deque[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr.append(line.rstrip())

    def _send(self, value: Mapping[str, Any]) -> None:
        if self._closed or self._process.poll() is not None:
            raise AppServerProtocolError(
                "Codex App Server is not running",
                stream_closed=not self._closed,
            )
        assert self._process.stdin is not None
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._write_lock:
            self._process.stdin.write(encoded)
            self._process.stdin.flush()

    def _receive(self, timeout_seconds: float | None = None) -> Mapping[str, Any]:
        try:
            item = self._incoming.get(
                timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds
            )
        except queue.Empty as exc:
            raise TimeoutError("timed out waiting for Codex App Server") from exc
        if isinstance(item, BaseException):
            raise AppServerProtocolError("invalid App Server output") from item
        if item is None:
            stderr = "\n".join(self._stderr)
            raise AppServerProtocolError(
                f"Codex App Server closed the stream unexpectedly{': ' + stderr if stderr else ''}",
                stream_closed=True,
            )
        return item

    def request(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        request_id = self._next_request_id
        self._next_request_id += 1
        self._send({"id": request_id, "method": method, "params": dict(params)})
        while True:
            message = self._receive()
            if message.get("id") != request_id or "method" in message:
                self._pending.append(message)
                continue
            if "error" in message:
                raw_error = message["error"]
                error = raw_error if isinstance(raw_error, Mapping) else {}
                raw_code = error.get("code")
                code = (
                    raw_code
                    if isinstance(raw_code, int) and not isinstance(raw_code, bool)
                    else None
                )
                server_message = str(error.get("message", "")) or None
                raise AppServerProtocolError(
                    f"App Server {method} failed: {json.dumps(raw_error, ensure_ascii=False)}",
                    method=method,
                    code=code,
                    server_message=server_message,
                )
            result = message.get("result", {})
            if not isinstance(result, Mapping):
                raise AppServerProtocolError(f"App Server {method} returned a non-object result")
            return result

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"method": method}
        if params is not None:
            message["params"] = dict(params)
        self._send(message)

    def respond(self, request_id: str | int, result: Mapping[str, Any]) -> None:
        """Reply to a JSON-RPC request initiated by App Server."""

        if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
            raise AppServerProtocolError("App Server request has an invalid request ID")
        self._send({"id": request_id, "result": dict(result)})

    def next_message(self) -> Mapping[str, Any]:
        return self._pending.popleft() if self._pending else self._receive()

    def next_message_with_timeout(self, timeout_seconds: float) -> Mapping[str, Any]:
        if timeout_seconds <= 0:
            raise TimeoutError("timed out waiting for Codex App Server")
        return (
            self._pending.popleft()
            if self._pending
            else self._receive(timeout_seconds=timeout_seconds)
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        if self._process.poll() is not None:
            return
        try:
            self._process.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            try:
                self._process.terminate()
            except OSError:
                pass
        try:
            self._process.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            try:
                self._process.kill()
            except OSError:
                return
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # close() is best-effort cleanup.  A stuck child must not mask
                # the authoritative runtime exception already being unwound.
                return


class FakeAppServerTransport:
    """Protocol-level transport; it never starts Codex or calls a Provider."""

    def __init__(
        self,
        *,
        responses: Mapping[str, Sequence[Mapping[str, Any]]],
        messages: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self._responses = defaultdict(deque)
        self._supported_methods = frozenset(responses)
        for method, values in responses.items():
            self._responses[method].extend(dict(value) for value in values)
        self._messages = deque(dict(item) for item in messages)
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    def supports_method(self, method: str) -> bool:
        """Declare only protocol methods explicitly provided by this Fake."""

        return method in self._supported_methods

    def request(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        self.sent.append({"kind": "request", "method": method, "params": dict(params)})
        if not self._responses[method]:
            raise AppServerProtocolError(f"unexpected Fake App Server request: {method}")
        return self._responses[method].popleft()

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        self.sent.append({"kind": "notification", "method": method, "params": dict(params or {})})

    def respond(self, request_id: str | int, result: Mapping[str, Any]) -> None:
        self.sent.append({"kind": "response", "id": request_id, "result": dict(result)})

    def next_message(self) -> Mapping[str, Any]:
        if not self._messages:
            raise AppServerProtocolError("Fake App Server message stream is exhausted")
        return self._messages.popleft()

    def next_message_with_timeout(self, timeout_seconds: float) -> Mapping[str, Any]:
        del timeout_seconds
        return self.next_message()

    def close(self) -> None:
        self.closed = True
