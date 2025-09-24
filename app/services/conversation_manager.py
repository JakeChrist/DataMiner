"""Stateful helpers for coordinating LMStudio conversations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from .lmstudio_client import (
    AnswerLength,
    ChatMessage,
    LMStudioClient,
    LMStudioConnectionError,
    LMStudioError,
)


class ResponseMode(Enum):
    """Requested shape of the assistant response."""

    GENERATIVE = "generative"
    SOURCES_ONLY = "sources_only"


class ReasoningVerbosity(Enum):
    """How much structured reasoning the assistant should emit."""

    MINIMAL = "minimal"
    BRIEF = "brief"
    EXTENDED = "extended"

    def to_request_options(self) -> dict[str, Any]:
        """Translate the verbosity preset into LMStudio request options."""

        payload: dict[str, Any] = {
            "reasoning": {
                "verbosity": self.value,
                "include_summary": True,
                "include_assumptions": True,
                "include_self_check": True,
            }
        }
        if self is ReasoningVerbosity.MINIMAL:
            payload["reasoning"].update({
                "include_plan": False,
                "max_bullets": 2,
            })
        elif self is ReasoningVerbosity.BRIEF:
            payload["reasoning"].update({
                "include_plan": True,
                "max_bullets": 4,
                "max_plan_items": 3,
            })
        else:  # EXTENDED
            payload["reasoning"].update({
                "include_plan": True,
                "max_bullets": 8,
                "max_plan_items": 6,
            })
        return payload


@dataclass
class PlanItem:
    """A single plan entry returned by the model."""

    description: str
    status: str = "pending"

    @property
    def is_complete(self) -> bool:
        return self.status.lower() in {"complete", "completed", "done"}


@dataclass
class SelfCheckResult:
    """Outcome of the model's self-check routine."""

    passed: bool
    flags: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class AssumptionDecision:
    """Record how ambiguity was handled for a turn."""

    mode: Literal["clarify", "assume", "unspecified"]
    rationale: str | None = None
    clarifying_question: str | None = None


@dataclass
class ReasoningArtifacts:
    """Structured reasoning data parsed from LMStudio metadata."""

    summary_bullets: list[str] = field(default_factory=list)
    plan_items: list[PlanItem] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    assumption_decision: AssumptionDecision | None = None
    self_check: SelfCheckResult | None = None


@dataclass
class ConversationTurn:
    """Record of a single question/answer exchange."""

    question: str
    answer: str
    citations: list[Any] = field(default_factory=list)
    reasoning: dict[str, Any] | None = None
    reasoning_artifacts: ReasoningArtifacts | None = None
    response_mode: ResponseMode = ResponseMode.GENERATIVE

    @property
    def reasoning_bullets(self) -> list[str]:
        if self.reasoning_artifacts is None:
            return []
        return list(self.reasoning_artifacts.summary_bullets)

    @property
    def plan(self) -> list[PlanItem]:
        if self.reasoning_artifacts is None:
            return []
        return list(self.reasoning_artifacts.plan_items)

    @property
    def assumptions(self) -> list[str]:
        if self.reasoning_artifacts is None:
            return []
        return list(self.reasoning_artifacts.assumptions)

    @property
    def assumption_decision(self) -> AssumptionDecision | None:
        if self.reasoning_artifacts is None:
            return None
        return self.reasoning_artifacts.assumption_decision

    @property
    def self_check(self) -> SelfCheckResult | None:
        if self.reasoning_artifacts is None:
            return None
        return self.reasoning_artifacts.self_check


@dataclass(frozen=True)
class ConnectionState:
    """Connection status event payload."""

    connected: bool
    message: str | None = None


class ConversationManager:
    """Track conversation history and orchestrate LMStudio requests."""

    def __init__(
        self,
        client: LMStudioClient,
        *,
        system_prompt: str | None = None,
        context_window: int = 4,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.context_window = max(0, context_window)
        self.turns: list[ConversationTurn] = []
        self._connected = True
        self._connection_error: str | None = None
        self._listeners: list[Callable[[ConnectionState], None]] = []

    def add_connection_listener(
        self, listener: Callable[[ConnectionState], None]
    ) -> Callable[[], None]:
        """Subscribe to connection changes and return an unsubscribe callable."""

        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    @property
    def connection_state(self) -> ConnectionState:
        return ConnectionState(self._connected, self._connection_error)

    def check_connection(self) -> ConnectionState:
        """Probe LMStudio and update connection state."""

        healthy = self.client.health_check()
        if healthy:
            self._update_connection(True, None)
        else:
            self._update_connection(False, "LMStudio is unreachable. Check the base URL.")
        return self.connection_state

    def can_ask(self) -> bool:
        """Return ``True`` when it is safe to issue a new query."""

        return self._connected

    def ask(
        self,
        question: str,
        *,
        context_snippets: Sequence[str] | None = None,
        preset: AnswerLength = AnswerLength.NORMAL,
        reasoning_verbosity: ReasoningVerbosity | None = ReasoningVerbosity.BRIEF,
        response_mode: ResponseMode = ResponseMode.GENERATIVE,
        extra_options: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        """Send ``question`` to LMStudio and append the resulting turn."""

        if not self._connected:
            message = self._connection_error or "LMStudio is disconnected."
            raise LMStudioConnectionError(message)

        messages = self._build_messages(question, context_snippets)
        request_options = self._build_request_options(
            reasoning_verbosity, response_mode, extra_options
        )
        try:
            response = self.client.chat(
                messages,
                preset=preset,
                extra_options=request_options or None,
            )
        except LMStudioError as exc:
            self._update_connection(False, str(exc) or "Unable to reach LMStudio.")
            raise

        self._update_connection(True, None)
        turn = self._register_turn(question, response, response_mode)
        return turn

    def _register_turn(
        self, question: str, response: ChatMessage, response_mode: ResponseMode
    ) -> ConversationTurn:
        artifacts = self._parse_reasoning_artifacts(response.reasoning)
        turn = ConversationTurn(
            question=question,
            answer=response.content,
            citations=response.citations,
            reasoning=response.reasoning,
            reasoning_artifacts=artifacts,
            response_mode=response_mode,
        )
        self.turns.append(turn)
        return turn

    @staticmethod
    def _build_request_options(
        reasoning_verbosity: ReasoningVerbosity | None,
        response_mode: ResponseMode,
        extra_options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if reasoning_verbosity is not None:
            options.update(reasoning_verbosity.to_request_options())
        if response_mode is ResponseMode.SOURCES_ONLY:
            options["response_mode"] = response_mode.value
        if extra_options:
            for key, value in extra_options.items():
                if (
                    key == "reasoning"
                    and isinstance(value, dict)
                    and isinstance(options.get("reasoning"), dict)
                ):
                    options["reasoning"].update(value)
                else:
                    options[key] = value
        return options

    @staticmethod
    def _parse_reasoning_artifacts(
        reasoning: dict[str, Any] | None,
    ) -> ReasoningArtifacts | None:
        if not reasoning:
            return None

        summary_candidates: list[str] = []
        for key in ("summary_bullets", "bullets", "summary", "points"):
            summary_candidates.extend(
                ConversationManager._coerce_str_list(reasoning.get(key))
            )
        summary_bullets = ConversationManager._deduplicate_preserve_order(summary_candidates)

        plan_items: list[PlanItem] = []
        plan_raw = reasoning.get("plan") or reasoning.get("plan_items")
        if isinstance(plan_raw, Sequence) and not isinstance(plan_raw, (str, bytes, dict)):
            for entry in plan_raw:
                if isinstance(entry, dict):
                    description = str(
                        entry.get("description")
                        or entry.get("step")
                        or entry.get("text")
                        or ""
                    ).strip()
                    status_raw = entry.get("status") or entry.get("state")
                    status = (
                        str(status_raw).strip().lower() if status_raw is not None else ""
                    )
                    if not status:
                        status = "pending"
                elif isinstance(entry, str):
                    description = entry.strip()
                    status = "pending"
                else:
                    continue
                if description:
                    plan_items.append(PlanItem(description=description, status=status))

        assumptions_list: list[str] = []
        decision: AssumptionDecision | None = None
        assumptions_raw = reasoning.get("assumptions")
        if isinstance(assumptions_raw, dict):
            assumptions_list = ConversationManager._coerce_str_list(
                assumptions_raw.get("used")
                or assumptions_raw.get("list")
                or assumptions_raw.get("items")
                or assumptions_raw.get("assumptions")
            )
            question_raw = (
                assumptions_raw.get("clarifying_question")
                or assumptions_raw.get("question")
            )
            clarifying_question = (
                str(question_raw).strip() if isinstance(question_raw, str) else None
            )
            rationale_raw = assumptions_raw.get("rationale") or assumptions_raw.get("reason")
            rationale = (
                str(rationale_raw).strip() if isinstance(rationale_raw, str) else None
            )
            decision_raw = assumptions_raw.get("decision")
            if isinstance(decision_raw, str):
                mode = decision_raw.strip().lower() or "unspecified"
            else:
                mode = "unspecified"
            should_ask = assumptions_raw.get("should_ask")
            if isinstance(should_ask, bool) and should_ask:
                mode = "clarify"
            if mode not in {"clarify", "assume"}:
                if assumptions_list:
                    mode = "assume"
                elif clarifying_question:
                    mode = "clarify"
                else:
                    mode = "unspecified"
            decision = AssumptionDecision(
                mode=mode if mode in {"clarify", "assume", "unspecified"} else "unspecified",
                rationale=rationale,
                clarifying_question=clarifying_question,
            )
        elif isinstance(assumptions_raw, Sequence) and not isinstance(
            assumptions_raw, (str, bytes)
        ):
            assumptions_list = ConversationManager._coerce_str_list(assumptions_raw)
            mode = "assume" if assumptions_list else "unspecified"
            decision = AssumptionDecision(mode=mode, rationale=None, clarifying_question=None)

        self_check_data = reasoning.get("self_check") or reasoning.get("selfCheck")
        self_check: SelfCheckResult | None = None
        if isinstance(self_check_data, dict):
            passed = bool(self_check_data.get("passed"))
            flags = ConversationManager._coerce_str_list(
                self_check_data.get("flags")
                or self_check_data.get("issues")
                or self_check_data.get("notes")
            )
            notes_raw = self_check_data.get("notes") or self_check_data.get("explanation")
            notes = str(notes_raw).strip() if isinstance(notes_raw, str) else None
            self_check = SelfCheckResult(passed=passed, flags=flags, notes=notes)

        if not any([summary_bullets, plan_items, assumptions_list, self_check, decision]):
            return None

        return ReasoningArtifacts(
            summary_bullets=summary_bullets,
            plan_items=plan_items,
            assumptions=assumptions_list,
            assumption_decision=decision,
            self_check=self_check,
        )

    @staticmethod
    def _coerce_str_list(value: Any) -> list[str]:
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, dict)):
            results: list[str] = []
            for item in value:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        results.append(text)
            return results
        return []

    @staticmethod
    def _deduplicate_preserve_order(values: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def _build_messages(
        self, question: str, context_snippets: Sequence[str] | None
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        history = self.turns[-self.context_window :] if self.context_window else []
        for turn in history:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})

        user_content = question
        if context_snippets:
            formatted_context = "\n\n".join(context_snippets)
            user_content = f"{question}\n\nContext:\n{formatted_context}"
        messages.append({"role": "user", "content": user_content})
        return messages

    def _update_connection(self, connected: bool, message: str | None) -> None:
        state_changed = connected != self._connected or message != self._connection_error
        self._connected = connected
        self._connection_error = message
        if state_changed:
            self._emit_connection_state()

    def _emit_connection_state(self) -> None:
        state = self.connection_state
        for listener in list(self._listeners):
            try:
                listener(state)
            except Exception:  # pragma: no cover - defensive guard
                continue


__all__ = [
    "AnswerLength",
    "AssumptionDecision",
    "ConnectionState",
    "ConversationManager",
    "ConversationTurn",
    "PlanItem",
    "ReasoningArtifacts",
    "ReasoningVerbosity",
    "ResponseMode",
    "SelfCheckResult",
]

