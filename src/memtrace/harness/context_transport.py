from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from ..contracts import DeliveryState
from .contracts import HarnessEvent, HarnessEventType
from .transport import AppServerProtocolError

if TYPE_CHECKING:
    from .adapter import CodexHarnessAdapter


class NativeCompactionRequestState(StrEnum):
    UNSUPPORTED = "UNSUPPORTED"
    SCHEDULED = "SCHEDULED"
    REQUEST_ACCEPTED = "REQUEST_ACCEPTED"
    FENCING_TIMEOUT = "FENCING_TIMEOUT"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ContextTransportReceipt:
    delivery_id: str
    thread_id: str
    context_digest: str
    request_accepted: bool
    provider_payload_tokens: int = 0
    delivery_state: DeliveryState = DeliveryState.TRANSPORT_ACCEPTED
    transport_method: str = "thread/inject_items"
    active_turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class ContextDeliverySignal:
    delivery_id: str
    context_digest: str
    state: DeliveryState
    turn_id: str | None
    source_event_id: str
    delivery_kind: str = "RECOVERED_CONTEXT"


@dataclass(frozen=True, slots=True)
class TurnContinuation:
    prompt: str
    collaboration_mode: str = "default"
    read_only: bool = False
    source_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class TurnFenceRequest:
    """One operational fence for a Provider Turn that lost route ownership.

    Durable route state remains in the WAL/registry.  This request only tells
    the stream driver to quiesce the physical Turn before the Provider can act
    beyond the authoritative current Step.
    """

    turn_id: str
    reason: str
    source_event_id: str


@dataclass(frozen=True, slots=True)
class ContextCompactionSignal:
    source_event_id: str
    origin: str


@dataclass(frozen=True, slots=True)
class NativeCompactionCompletion:
    """One completed Provider compaction episode, retained for the stream driver."""

    verified: bool | None
    timed_out: bool
    origin: str
    provider_turn_id: str | None
    completion_event_observed: bool = False


@dataclass(slots=True)
class _PendingTransportDelivery:
    delivery_id: str
    thread_id: str
    context_digest: str
    committed_turn_id: str | None = None
    replacement_epoch: bool = False
    recovered_state: DeliveryState = DeliveryState.TRANSPORT_ACCEPTED
    provider_payload_tokens: int = 0
    delivery_kind: str = "RECOVERED_CONTEXT"


class CodexContextTransport:
    """Deliver immutable context to Codex and derive states from protocol facts.

    A successful ``thread/inject_items`` response proves only transport
    acceptance. A later real Turn and its first model/tool event provide the
    two stronger delivery states.
    """

    _MODEL_OBSERVATION_EVENTS = frozenset(
        {
            HarnessEventType.PLAN_PROPOSED,
            HarnessEventType.PLAN_UPDATED,
            HarnessEventType.ITEM_STARTED,
            HarnessEventType.ITEM_COMPLETED,
            HarnessEventType.TOOL_INTENT,
            HarnessEventType.TOOL_RESULT,
            HarnessEventType.FILE_CHANGED,
        }
    )

    def __init__(
        self,
        adapter: CodexHarnessAdapter,
        *,
        supports_native_compaction: bool = True,
        native_compaction_policy_enabled: bool | None = None,
        native_compaction_timeout_seconds: float = 60.0,
    ) -> None:
        if adapter.thread_id is None:
            raise ValueError("Codex Context Transport requires an established Thread")
        self.adapter = adapter
        self.supports_native_compaction = supports_native_compaction
        # Policy and request capability are deliberately separate.  A Provider
        # may emit an automatic context-compaction item even when its protocol
        # schema exposes no method for requesting one.  Production can disable
        # adoption of either form without pretending that the method exists.
        self.native_compaction_policy_enabled = (
            supports_native_compaction
            if native_compaction_policy_enabled is None
            else native_compaction_policy_enabled
        )
        if native_compaction_timeout_seconds <= 0:
            raise ValueError("native compaction timeout must be positive")
        self.native_compaction_timeout_seconds = native_compaction_timeout_seconds
        self._pending: dict[str, _PendingTransportDelivery] = {}
        self._continuations: list[TurnContinuation] = []
        self._turn_fences: dict[str, TurnFenceRequest] = {}
        # Payloads accepted by Codex but not yet reflected in a later physical
        # token observation are admission debt.  Without this ledger, several
        # individually bounded Memory Loading operations can collectively refill a freshly
        # compacted Provider context before the next usage event arrives.
        self._unaccounted_provider_tokens = 0
        self._awaiting_usage_reconciliation_tokens = 0
        self._inline_provider_payloads: dict[str, int] = {}
        self._accounted_inline_payloads: set[str] = set()
        self._native_state = NativeCompactionRequestState.UNSUPPORTED
        self._native_event_observed = False
        self._native_episode_complete = False
        # Latest Provider usage outside an active episode.  The request-time
        # baseline below is immutable for the episode; keeping the two values
        # separate makes verification tolerant of Provider event reordering.
        self._native_baseline_tokens: int | None = None
        self._native_request_baseline_tokens: int | None = None
        self._native_post_request_tokens: int | None = None
        self._native_usage_after_event_observed = False
        self._native_verified: bool | None = None
        self._native_timed_out = False
        self._native_turn_id: str | None = None
        self._native_scheduled_from_turn_id: str | None = None
        self._native_origin: str | None = None
        self._native_requested_at: float | None = None
        self._compaction_signals: list[ContextCompactionSignal] = []
        self._native_completions: list[NativeCompactionCompletion] = []
        # Run budget: a monotonic deadline the driver polls inside a Turn, and
        # the closed flag the coordinator raises once the budget is spent so no
        # follow-up Turn is started on this Thread.
        self.run_deadline_monotonic: float | None = None
        self._run_budget_closed_reason: str | None = None
        self._invocation_closed_reason: str | None = None

    @property
    def run_budget_closed(self) -> bool:
        return self._run_budget_closed_reason is not None

    @property
    def run_budget_closed_reason(self) -> str | None:
        return self._run_budget_closed_reason

    def run_deadline_reached(self) -> bool:
        return (
            self.run_deadline_monotonic is not None
            and time.monotonic() >= self.run_deadline_monotonic
        )

    def close_for_run_budget(self, reason: str) -> int:
        """Stop scheduling Turns: the run budget is spent.

        Queued continuations are discarded and reported; deliveries that were
        pending for a next Turn stay in their durable state so the run report
        can count them as unobserved.  Idempotent.
        """

        normalized = " ".join(reason.split())
        if not normalized:
            raise ValueError("run budget closure requires a reason")
        if self._run_budget_closed_reason is None:
            self._run_budget_closed_reason = normalized
        return self.close_invocation(normalized)

    def close_invocation(self, reason: str) -> int:
        """Quiesce this physical invocation without terminating its Task.

        A repository-stream Task can span several bounded process invocations.
        Closing one invocation discards its queued physical continuations,
        while the trace ledger, MTG, Trace Store and Codex Thread remain resumable by the
        next invocation. This is deliberately distinct from a Task verdict.
        """

        normalized = " ".join(reason.split())
        if not normalized:
            raise ValueError("invocation closure requires a reason")
        if self._invocation_closed_reason is None:
            self._invocation_closed_reason = normalized
        discarded = len(self._continuations)
        self._continuations = []
        return discarded

    @property
    def thread_id(self) -> str:
        thread_id = self.adapter.thread_id
        if thread_id is None:
            raise RuntimeError("Codex Thread is no longer available")
        return thread_id

    def submit_recovered(
        self,
        *,
        delivery_id: str,
        context_digest: str,
        rendered_content: str,
        max_provider_tokens: int | None = None,
        active_turn_id: str | None = None,
    ) -> ContextTransportReceipt:
        if not delivery_id or not context_digest or not rendered_content:
            raise ValueError("recovered context delivery fields must be non-empty")
        provider_text = self.render_recovered_payload(
            delivery_id=delivery_id,
            context_digest=context_digest,
            rendered_content=rendered_content,
        )
        return self._submit_context_payload(
            delivery_id=delivery_id,
            context_digest=context_digest,
            provider_text=provider_text,
            max_provider_tokens=max_provider_tokens,
            active_turn_id=active_turn_id,
            delivery_kind="RECOVERED_CONTEXT",
        )

    def submit_provider_compaction_refresh(
        self,
        *,
        delivery_id: str,
        context_digest: str,
        rendered_content: str,
        active_turn_id: str | None,
    ) -> ContextTransportReceipt:
        if not delivery_id or not context_digest or not rendered_content:
            raise ValueError("Provider compaction refresh fields must be non-empty")
        provider_text = self.render_provider_compaction_refresh_payload(
            delivery_id=delivery_id,
            context_digest=context_digest,
            rendered_content=rendered_content,
        )
        return self._submit_context_payload(
            delivery_id=delivery_id,
            context_digest=context_digest,
            provider_text=provider_text,
            max_provider_tokens=None,
            active_turn_id=active_turn_id,
            delivery_kind="PROVIDER_COMPACTION_REFRESH",
        )

    def _submit_context_payload(
        self,
        *,
        delivery_id: str,
        context_digest: str,
        provider_text: str,
        max_provider_tokens: int | None,
        active_turn_id: str | None,
        delivery_kind: str,
    ) -> ContextTransportReceipt:
        provider_tokens = self.provider_token_estimate(provider_text)
        if max_provider_tokens is not None and provider_tokens > max_provider_tokens:
            raise ValueError("Provider Context payload exceeds admission budget")
        steer_turn_id = active_turn_id
        if steer_turn_id is not None:
            try:
                response = self.adapter.transport.request(
                    "turn/steer",
                    {
                        "threadId": self.thread_id,
                        "input": [{"type": "text", "text": provider_text}],
                        "expectedTurnId": steer_turn_id,
                    },
                )
            except AppServerProtocolError as error:
                if not error.is_closed_turn_steer():
                    raise
            # The context-refresh request was already durable before transport.  A
                # Provider may close the Turn in the narrow interval between
                # its last item notification and this request. Preserve the
                # same Delivery for the next Turn instead of losing the task
                # or pretending that the closed Turn observed it.
                steer_turn_id = None
            else:
                if str(response.get("turnId", "")) != steer_turn_id:
                    raise RuntimeError("turn/steer did not accept the active Context delivery")
                delivery_state = DeliveryState.CONTEXT_COMMITTED
                transport_method = "turn/steer"
        if steer_turn_id is None:
            self.adapter.transport.request(
                "thread/inject_items",
                {
                    "threadId": self.thread_id,
                    "items": [
                        {
                            "type": "message",
                            "role": "developer",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": provider_text,
                                }
                            ],
                        }
                    ],
                },
            )
            delivery_state = DeliveryState.TRANSPORT_ACCEPTED
            transport_method = (
                "thread/inject_items_after_closed_turn"
                if active_turn_id is not None
                else "thread/inject_items"
            )
        self._pending[delivery_id] = _PendingTransportDelivery(
            delivery_id=delivery_id,
            thread_id=self.thread_id,
            context_digest=context_digest,
            committed_turn_id=steer_turn_id,
            recovered_state=delivery_state,
            provider_payload_tokens=provider_tokens,
            delivery_kind=delivery_kind,
        )
        self._unaccounted_provider_tokens += provider_tokens
        return ContextTransportReceipt(
            delivery_id=delivery_id,
            thread_id=self.thread_id,
            context_digest=context_digest,
            request_accepted=True,
            provider_payload_tokens=provider_tokens,
            delivery_state=delivery_state,
            transport_method=transport_method,
            active_turn_id=steer_turn_id,
        )

    @staticmethod
    def render_provider_compaction_refresh_payload(
        *, delivery_id: str, context_digest: str, rendered_content: str
    ) -> str:
        return (
            "ProviderCompactionRecovery (durable Working Memory refresh)\n"
            f"delivery_id={delivery_id}\n"
            f"context_digest={context_digest}\n\n"
            f"{rendered_content}"
        )

    def steer_active_turn(self, turn_id: str, text: str) -> bool:
        """Append a control fact while the Provider's Turn lease remains valid.

        ``turn_id`` is evidence that the Turn was active when its latest event
        was observed, not a lock on Provider state.  A normal completion can
        therefore race this request.  Report that closed-lease outcome to the
        caller so the same durable fact can move to a continuation Turn; all
        other protocol failures remain fatal.
        """

        normalized = " ".join(text.split())
        if not turn_id or not normalized:
            raise ValueError("active Turn steering fields must be non-empty")
        try:
            response = self.adapter.transport.request(
                "turn/steer",
                {
                    "threadId": self.thread_id,
                    "input": [{"type": "text", "text": normalized}],
                    "expectedTurnId": turn_id,
                },
            )
        except AppServerProtocolError as error:
            if error.is_closed_turn_steer():
                return False
            raise
        if str(response.get("turnId", "")) != turn_id:
            raise RuntimeError("turn/steer did not accept the runtime control fact")
        return True

    def recover_delivery(
        self,
        *,
        delivery_id: str,
        thread_id: str,
        context_digest: str,
        state: DeliveryState,
        replacement_epoch: bool = False,
        delivery_kind: str = "RECOVERED_CONTEXT",
    ) -> None:
        if state not in {DeliveryState.TRANSPORT_ACCEPTED, DeliveryState.CONTEXT_COMMITTED}:
            raise ValueError("only accepted/committed deliveries can rejoin transport")
        self._pending[delivery_id] = _PendingTransportDelivery(
            delivery_id=delivery_id,
            thread_id=thread_id,
            context_digest=context_digest,
            replacement_epoch=replacement_epoch,
            recovered_state=state,
            delivery_kind=delivery_kind,
        )

    def has_pending_delivery(self, delivery_id: str) -> bool:
        return delivery_id in self._pending

    @staticmethod
    def render_recovered_payload(
        *, delivery_id: str, context_digest: str, rendered_content: str
    ) -> str:
        return (
            "RecoveredContextBlock (immutable, evidence-backed)\n"
            f"delivery_id={delivery_id}\n"
            f"content_digest={context_digest}\n\n"
            f"{rendered_content}"
        )

    @staticmethod
    def provider_token_estimate(text: str) -> int:
        return max(1, (len(text.encode("utf-8")) + 2) // 3)

    @classmethod
    def recovered_content_budget(cls, provider_payload_budget: int) -> int:
        """Translate a wire-level budget into a safe recovered-body budget."""

        if provider_payload_budget <= 0:
            return 0
        # Stable IDs and SHA-256 digests have fixed widths.  Reserving the
        # exact empty envelope before Page slicing guarantees that the final
        # Provider payload—not merely its evidence body—fits admission.
        envelope = cls.render_recovered_payload(
            delivery_id="delivery_" + ("0" * 32),
            context_digest="sha256:" + ("0" * 64),
            rendered_content="",
        )
        envelope_tokens = cls.provider_token_estimate(envelope)
        return max(0, provider_payload_budget - envelope_tokens)

    def account_inline_provider_payload(self, payload_id: str, text: str) -> int:
        """Reserve physical headroom for a synchronous dynamic-tool result."""

        if not payload_id or not text:
            raise ValueError("inline Provider payload fields must be non-empty")
        if payload_id in self._accounted_inline_payloads:
            return self._inline_provider_payloads.get(payload_id, 0)
        tokens = self.provider_token_estimate(text)
        self._accounted_inline_payloads.add(payload_id)
        self._inline_provider_payloads[payload_id] = tokens
        self._unaccounted_provider_tokens += tokens
        return tokens

    def inline_provider_payload_observed(self, payload_id: str) -> None:
        """Move a dynamic result's debt to the next usage reconciliation."""

        tokens = self._inline_provider_payloads.pop(payload_id, 0)
        if tokens:
            self._awaiting_usage_reconciliation_tokens += tokens

    def start_replacement_thread(self) -> str:
        response = self.adapter.transport.request(
            "thread/start",
            self.adapter.thread_parameters(
                developer_instructions=(
                    "Replacement Epoch. Continue only from the injected ContinuityCheckpoint "
                    "and ContextImage. Confirm that the injected workspace revision still matches "
                    "the live workspace receipt before side effects, then reuse the durable "
                    "execution handoff instead of repeating unchanged investigation. Re-read "
                    "exact code only when the relevant Page detail is absent or truncated, or "
                    "when the workspace revision changed. Call recall_memory before relying on "
                    "a compressed Memory Anchor."
                )
            ),
        )
        thread = response.get("thread")
        if not isinstance(thread, dict) or not str(thread.get("id", "")):
            raise RuntimeError("replacement thread/start returned no real Thread ID")
        replacement = str(thread["id"])
        if replacement == self.thread_id:
            raise RuntimeError("replacement Thread must differ from its predecessor")
        return replacement

    def resume_replacement_thread(self, thread_id: str) -> None:
        """Reattach a persisted candidate Thread without activating its Epoch."""

        response = self.adapter.transport.request("thread/resume", {"threadId": thread_id})
        thread = response.get("thread")
        if not isinstance(thread, dict) or str(thread.get("id", "")) != thread_id:
            raise RuntimeError("could not resume persisted replacement Thread")

    def submit_replacement(
        self,
        *,
        delivery_id: str,
        thread_id: str,
        context_digest: str,
        rendered_content: str,
        max_provider_tokens: int | None = None,
    ) -> ContextTransportReceipt:
        provider_text = self.render_replacement_payload(
            delivery_id=delivery_id,
            context_digest=context_digest,
            rendered_content=rendered_content,
        )
        provider_tokens = self.provider_token_estimate(provider_text)
        if max_provider_tokens is not None and provider_tokens > max_provider_tokens:
            raise ValueError("replacement Context payload exceeds safe Provider budget")
        self.adapter.transport.request(
            "thread/inject_items",
            {
                "threadId": thread_id,
                "items": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [
                            {
                                "type": "input_text",
                                "text": provider_text,
                            }
                        ],
                    }
                ],
            },
        )
        self._pending[delivery_id] = _PendingTransportDelivery(
            delivery_id=delivery_id,
            thread_id=thread_id,
            context_digest=context_digest,
            replacement_epoch=True,
            provider_payload_tokens=provider_tokens,
        )
        self._unaccounted_provider_tokens += provider_tokens
        return ContextTransportReceipt(
            delivery_id=delivery_id,
            thread_id=thread_id,
            context_digest=context_digest,
            request_accepted=True,
            provider_payload_tokens=provider_tokens,
        )

    @staticmethod
    def render_replacement_payload(
        *, delivery_id: str, context_digest: str, rendered_content: str
    ) -> str:
        return (
            "ContinuityCheckpoint and canonical ContextImage\n"
            f"delivery_id={delivery_id}\n"
            f"context_digest={context_digest}\n\n"
            f"{rendered_content}"
        )

    @classmethod
    def replacement_provider_token_estimate(cls, rendered_content: str) -> int:
        """Measure the exact wire shape before creating a pending Epoch."""

        payload = cls.render_replacement_payload(
            delivery_id="delivery_" + ("0" * 32),
            context_digest="sha256:" + ("0" * 64),
            rendered_content=rendered_content,
        )
        return cls.provider_token_estimate(payload)

    def observe(self, event: HarnessEvent) -> tuple[ContextDeliverySignal, ...]:
        if (
            event.event_type is HarnessEventType.TOKEN_USAGE_UPDATED
            and self._awaiting_usage_reconciliation_tokens
        ):
            self._unaccounted_provider_tokens = max(
                0,
                self._unaccounted_provider_tokens - self._awaiting_usage_reconciliation_tokens,
            )
            self._awaiting_usage_reconciliation_tokens = 0
        if event.event_type is HarnessEventType.CONTEXT_COMPACTED:
            origin = self._native_origin or "AUTOMATIC_OR_PROVIDER"
            managed_episode = not (
                origin == "AUTOMATIC_OR_PROVIDER"
                and not self.native_compaction_policy_enabled
            )
            if managed_episode and not any(
                item.source_event_id == event.source_event_id for item in self._compaction_signals
            ):
                self._compaction_signals.append(
                    ContextCompactionSignal(event.source_event_id, origin)
                )
            if self._native_state is NativeCompactionRequestState.REQUEST_ACCEPTED:
                self._native_event_observed = True
                if self._native_origin == "AUTOMATIC_OR_PROVIDER":
                    # This runtime did not request the compaction. Its explicit
                    # completion event is sufficient to restore the logical
                    # Working Memory and continue; token reduction remains
                    # observability and never gates recovery.
                    self._native_episode_complete = True
                else:
                    self._verify_native_compaction_if_ready()
        elif event.event_type is HarnessEventType.TOKEN_USAGE_UPDATED:
            usage = event.payload.get("token_usage", {})
            last = usage.get("last", {}) if isinstance(usage, dict) else {}
            tokens = (
                last.get("totalTokens", last.get("total_tokens"))
                if isinstance(last, dict)
                else None
            )
            if isinstance(tokens, int) and not isinstance(tokens, bool):
                if (
                    self._native_state is NativeCompactionRequestState.REQUEST_ACCEPTED
                    and self._native_origin == "MANUAL"
                ):
                    # Codex may emit the reduced usage observation either
                    # before or after CONTEXT_COMPACTED.  Capture every usage
                    # produced after request acceptance and verify once both
                    # independent facts are present.
                    self._native_post_request_tokens = (
                        tokens
                        if self._native_post_request_tokens is None
                        else min(self._native_post_request_tokens, tokens)
                    )
                    if self._native_event_observed:
                        self._native_usage_after_event_observed = True
                    self._verify_native_compaction_if_ready()
                # Retain the latest physical observation for the *next*
                # episode without mutating the current request baseline.
                self._native_baseline_tokens = tokens
        signals: list[ContextDeliverySignal] = []
        for delivery_id, pending in tuple(self._pending.items()):
            if event.thread_id != pending.thread_id:
                continue
            if pending.committed_turn_id is None:
                if event.event_type is not HarnessEventType.TURN_STARTED:
                    continue
                # ``thread/compact/start`` returns before the compact Turn is
                # observable.  Memory Loading accepted during that interval belongs
                # to the next task continuation, not to the compaction request
                # itself.  Binding it to the compact Turn would falsely claim
                # that the coding model observed evidence which was only added
                # to subsequent model-visible history.
                if self.native_compaction_waiting:
                    continue
                pending.committed_turn_id = event.turn_id
                if pending.recovered_state is DeliveryState.TRANSPORT_ACCEPTED:
                    pending.recovered_state = DeliveryState.CONTEXT_COMMITTED
                    signals.append(
                        ContextDeliverySignal(
                            delivery_id=delivery_id,
                            context_digest=pending.context_digest,
                            state=DeliveryState.CONTEXT_COMMITTED,
                            turn_id=event.turn_id,
                            source_event_id=event.source_event_id,
                            delivery_kind=pending.delivery_kind,
                        )
                    )
                continue
            if event.turn_id == pending.committed_turn_id and (
                event.event_type in self._MODEL_OBSERVATION_EVENTS
                or (
                    # A runtime tool call is a model action too: the model
                    # sampled after the committed payload became visible.
                    # The call that produced this very delivery is excluded;
                    # its response is what the model has yet to observe.
                    event.event_type is HarnessEventType.MEMORY_TOOL_RESULT
                    and str(event.payload.get("delivery_id") or "") != delivery_id
                )
            ):
                signals.append(
                    ContextDeliverySignal(
                        delivery_id=delivery_id,
                        context_digest=pending.context_digest,
                        state=DeliveryState.MODEL_OBSERVED,
                        turn_id=event.turn_id,
                        source_event_id=event.source_event_id,
                        delivery_kind=pending.delivery_kind,
                    )
                )
                self._awaiting_usage_reconciliation_tokens += pending.provider_payload_tokens
                self._pending.pop(delivery_id, None)
                continue
            if event.turn_id == pending.committed_turn_id and event.event_type in {
                HarnessEventType.TURN_COMPLETED,
                HarnessEventType.TURN_QUIESCED,
            }:
                # The payload was committed, but there was no later model/tool
                # action proving observation.  The developer message stays in
                # the Thread history, so preserve this delivery for a successor
                # Turn on the same Thread instead of leaving it committed to a
                # closed Turn forever.
                pending.committed_turn_id = None
                if pending.delivery_kind == "PROVIDER_COMPACTION_REFRESH":
                    self.request_task_continuation(
                        "Continue the same task from the durable post-compaction "
                        "Working Memory refresh."
                    )
        return tuple(signals)

    def take_compaction_signals(self) -> tuple[ContextCompactionSignal, ...]:
        signals = tuple(self._compaction_signals)
        self._compaction_signals.clear()
        return signals

    def _verify_native_compaction_if_ready(self) -> None:
        if self._native_state is not NativeCompactionRequestState.REQUEST_ACCEPTED:
            return
        if not self._native_event_observed or self._native_post_request_tokens is None:
            return
        baseline = self._native_request_baseline_tokens
        if baseline is not None and self._native_post_request_tokens < baseline:
            self._native_verified = True
        elif self._native_usage_after_event_observed:
            # A post-completion usage fact that does not fall below the immutable
            # baseline completes the episode as a verified negative result.
            self._native_verified = False

    @property
    def needs_followup_turn(self) -> bool:
        if self._invocation_closed_reason is not None:
            return False
        return (
            self.native_compaction_waiting
            or bool(self._continuations)
            or any(item.committed_turn_id is None for item in self._pending.values())
        )

    @property
    def pending_thread_id(self) -> str | None:
        threads = {
            item.thread_id for item in self._pending.values() if item.committed_turn_id is None
        }
        if len(threads) > 1:
            raise RuntimeError("multiple Context delivery Threads are pending")
        return next(iter(threads), self.thread_id if self._continuations else None)

    @property
    def unaccounted_provider_tokens(self) -> int:
        """Accepted Memory Loading payload not represented by a later usage fact yet."""

        return self._unaccounted_provider_tokens

    def request_task_continuation(
        self,
        prompt: str,
        *,
        collaboration_mode: str = "default",
        read_only: bool = False,
        source_event_id: str | None = None,
    ) -> None:
        normalized = " ".join(prompt.split())
        if collaboration_mode not in {"default", "plan"}:
            raise ValueError("unsupported continuation collaboration mode")
        request = TurnContinuation(
            normalized,
            collaboration_mode,
            read_only,
            source_event_id,
        )
        if not normalized:
            return
        if source_event_id is not None:
            self._continuations = [
                existing
                for existing in self._continuations
                if existing.source_event_id != source_event_id
            ]
            self._continuations.append(request)
        elif request not in self._continuations:
            self._continuations.append(request)

    def cancel_task_continuations(self, *, source_event_id: str) -> int:
        """Remove only continuations superseded by a later route commit."""

        retained = [
            request
            for request in self._continuations
            if request.source_event_id != source_event_id
        ]
        removed = len(self._continuations) - len(retained)
        self._continuations = retained
        return removed

    def take_continuation_request(self) -> TurnContinuation | None:
        return self._continuations.pop(0) if self._continuations else None

    def take_continuation_prompt(self) -> str | None:
        request = self.take_continuation_request()
        return request.prompt if request is not None else None

    def request_turn_fence(
        self,
        *,
        turn_id: str,
        reason: str,
        source_event_id: str,
    ) -> None:
        """Fence an active Turn after the rejecting route fact is durable.

        A Turn has exactly one semantic fence.  Replaying the same durable
        request is idempotent; conflicting requests indicate two control
        authorities and therefore fail closed.
        """

        request = TurnFenceRequest(
            turn_id=turn_id.strip(),
            reason=reason.strip(),
            source_event_id=source_event_id.strip(),
        )
        if not request.turn_id or not request.reason or not request.source_event_id:
            raise ValueError("Turn fence requires turn_id, reason, and source_event_id")
        existing = self._turn_fences.get(request.turn_id)
        if existing is not None and existing != request:
            raise RuntimeError("one Provider Turn received conflicting semantic fences")
        self._turn_fences[request.turn_id] = request

    def take_turn_fence(self, turn_id: str) -> TurnFenceRequest | None:
        return self._turn_fences.pop(turn_id, None)

    def turn_fence_pending(self, turn_id: str) -> bool:
        """Return whether another durable boundary already fences this Turn."""

        return turn_id in self._turn_fences

    def take_pending_turn_fence(self) -> TurnFenceRequest | None:
        """Take one recovered fence before any successor Turn is started."""

        if not self._turn_fences:
            return None
        turn_id = next(iter(self._turn_fences))
        return self._turn_fences.pop(turn_id)

    def knows_thread(self, thread_id: str) -> bool:
        return thread_id == self.thread_id or any(
            item.thread_id == thread_id for item in self._pending.values()
        )

    def activate_replacement(self, thread_id: str) -> None:
        if thread_id == self.thread_id:
            return
        self.adapter.thread_id = thread_id

    def request_native_compaction(self) -> NativeCompactionRequestState:
        if not self.supports_native_compaction:
            self._native_state = NativeCompactionRequestState.UNSUPPORTED
            return self._native_state
        if self.native_compaction_waiting:
            return self._native_state
        # Capture the latest observed physical size immediately before this
        # pressure episode.  It is updated on every ordinary usage event in
        # ``observe`` and is deliberately not the first-ever usage value.
        baseline = self._native_baseline_tokens
        try:
            response = self.adapter.transport.request(
                "thread/compact/start",
                {"threadId": self.thread_id},
            )
        except Exception:
            self._native_state = NativeCompactionRequestState.FAILED
            return self._native_state
        # Request acceptance is deliberately not reported as compaction
        # success. Success requires a later CONTEXT_COMPACTED event.
        self._native_state = NativeCompactionRequestState.REQUEST_ACCEPTED
        self._native_event_observed = False
        self._native_episode_complete = False
        self._native_verified = None
        self._native_timed_out = False
        self._native_request_baseline_tokens = baseline
        self._native_post_request_tokens = None
        self._native_usage_after_event_observed = False
        self._native_origin = "MANUAL"
        self._native_scheduled_from_turn_id = None
        turn = response.get("turn")
        self._native_turn_id = (
            str(turn.get("id")) if isinstance(turn, dict) and str(turn.get("id", "")) else None
        )
        self._native_requested_at = time.monotonic()
        return self._native_state

    def schedule_native_compaction(
        self, *, active_turn_id: str | None
    ) -> NativeCompactionRequestState:
        """Defer manual compaction until the current semantic unit is durable.

        The Provider emits a ``contextCompaction`` item inside the active task
        Turn. Starting it while a command is in flight can split a side effect
        from its result, so the driver waits for a completed tool/file/message
        boundary that has already been committed to the raw WAL.
        """

        if not self.supports_native_compaction:
            self._native_state = NativeCompactionRequestState.UNSUPPORTED
            return self._native_state
        if self.native_compaction_waiting:
            return self._native_state
        if self.native_compaction_scheduled:
            return self._native_state
        self._native_state = NativeCompactionRequestState.SCHEDULED
        self._native_scheduled_from_turn_id = active_turn_id
        return self._native_state

    def observe_compaction_started(self, turn_id: str | None) -> bool:
        """Observe one Provider compaction and return whether it is adopted.

        When native compaction policy is disabled, Provider-owned compression
        is not a correctness mechanism.  Retain its identity for diagnostics,
        but never enter the waiting state: the orchestration layer will fence
        the affected Thread and recover through the durable Epoch path.
        """

        if self.native_compaction_waiting:
            return True
        if not self.native_compaction_policy_enabled:
            self._native_state = NativeCompactionRequestState.UNSUPPORTED
            self._native_event_observed = False
            self._native_episode_complete = False
            self._native_verified = None
            self._native_timed_out = False
            self._native_request_baseline_tokens = None
            self._native_post_request_tokens = None
            self._native_usage_after_event_observed = False
            self._native_turn_id = turn_id
            self._native_scheduled_from_turn_id = None
            self._native_origin = "AUTOMATIC_OR_PROVIDER"
            self._native_requested_at = None
            return False
        self._native_state = NativeCompactionRequestState.REQUEST_ACCEPTED
        self._native_event_observed = False
        self._native_episode_complete = False
        self._native_verified = None
        self._native_timed_out = False
        self._native_request_baseline_tokens = self._native_baseline_tokens
        self._native_post_request_tokens = None
        self._native_usage_after_event_observed = False
        self._native_turn_id = turn_id
        self._native_scheduled_from_turn_id = None
        self._native_origin = "AUTOMATIC_OR_PROVIDER"
        self._native_requested_at = time.monotonic()
        return True

    def native_compaction_remaining_seconds(self) -> float:
        if not self.native_compaction_waiting or self._native_requested_at is None:
            return 0.0
        return max(
            0.0,
            self.native_compaction_timeout_seconds - (time.monotonic() - self._native_requested_at),
        )

    def begin_native_compaction_timeout_fence(self) -> None:
        """Freeze a timed-out episode while its predecessor Turn is quiesced.

        A late Provider compaction notification must not reverse a timeout after
        the runtime has begun fencing the active coding Turn.  Keeping the
        episode pending until that Turn is terminal also prevents the
        coordinator from creating a replacement Epoch while predecessor side
        effects are still arriving.
        """

        if self._native_state is NativeCompactionRequestState.REQUEST_ACCEPTED:
            self._native_state = NativeCompactionRequestState.FENCING_TIMEOUT
            self._native_timed_out = True

    def fail_native_compaction_timeout(self) -> None:
        if self.native_compaction_waiting:
            self._native_timed_out = True
            self._native_verified = False
            self._native_episode_complete = True

    def reset_native_compaction_episode(self) -> None:
        """Allow a later independent pressure episode after verification."""

        if self.native_compaction_waiting:
            raise RuntimeError("cannot reset a native compaction still awaiting evidence")
        if self._native_episode_complete or self._native_verified is not None:
            self._native_completions.append(
                NativeCompactionCompletion(
                    verified=self._native_verified,
                    timed_out=self._native_timed_out,
                    origin=self._native_origin or "UNKNOWN",
                    provider_turn_id=self._native_turn_id,
                    completion_event_observed=self._native_event_observed,
                )
            )
        self._native_state = NativeCompactionRequestState.UNSUPPORTED
        self._native_event_observed = False
        self._native_episode_complete = False
        self._native_verified = None
        self._native_timed_out = False
        self._native_request_baseline_tokens = None
        self._native_post_request_tokens = None
        self._native_usage_after_event_observed = False
        self._native_turn_id = None
        self._native_scheduled_from_turn_id = None
        self._native_origin = None
        self._native_requested_at = None

    def take_native_compaction_completion(self) -> NativeCompactionCompletion | None:
        return self._native_completions.pop(0) if self._native_completions else None

    @property
    def native_compaction_waiting(self) -> bool:
        if self._native_state not in {
            NativeCompactionRequestState.REQUEST_ACCEPTED,
            NativeCompactionRequestState.FENCING_TIMEOUT,
        }:
            return False
        if self._native_origin == "AUTOMATIC_OR_PROVIDER":
            return not self._native_episode_complete
        return self._native_verified is None

    @property
    def native_compaction_complete(self) -> bool:
        return self._native_episode_complete or self._native_verified is not None

    @property
    def native_compaction_scheduled(self) -> bool:
        return self._native_state is NativeCompactionRequestState.SCHEDULED

    @property
    def native_compaction_pending(self) -> bool:
        return self.native_compaction_scheduled or self.native_compaction_waiting

    def can_steer_turn(self, turn_id: str | None) -> bool:
        """Whether a task Turn is still the Provider's steerable active Turn."""

        if not turn_id:
            return False
        # Manual compaction is asynchronous and its request may not return a
        # Turn ID.  Once scheduled or accepted, the predecessor task Turn has
        # no safe steering lease: App Server may already have made the compact
        # Turn active before its first notification reaches this client.
        if self.native_compaction_scheduled or self.native_compaction_waiting:
            return False
        return True

    @property
    def native_compaction_verified(self) -> bool | None:
        return self._native_verified

    @property
    def native_compaction_turn_id(self) -> str | None:
        return self._native_turn_id

    @property
    def native_compaction_origin(self) -> str | None:
        return self._native_origin
