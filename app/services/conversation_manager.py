"""Stateful helpers for coordinating LMStudio conversations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .lmstudio_client import (
    AnswerLength,
    ChatMessage,
    LMStudioClient,
    LMStudioConnectionError,
    LMStudioError,
)


@dataclass
class ConversationTurn:
    """Record of a single question/answer exchange."""

    question: str
    answer: str
    citations: list[Any] = field(default_factory=list)
    reasoning: dict[str, Any] | None = None


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
        extra_options: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        """Send ``question`` to LMStudio and append the resulting turn."""

        if not self._connected:
            message = self._connection_error or "LMStudio is disconnected."
            raise LMStudioConnectionError(message)

        messages = self._build_messages(question, context_snippets)
        try:
            response = self.client.chat(messages, preset=preset, extra_options=extra_options)
        except LMStudioError as exc:
            self._update_connection(False, str(exc) or "Unable to reach LMStudio.")
            raise

        self._update_connection(True, None)
        turn = self._register_turn(question, response)
        return turn

    def _register_turn(self, question: str, response: ChatMessage) -> ConversationTurn:
        turn = ConversationTurn(
            question=question,
            answer=response.content,
            citations=response.citations,
            reasoning=response.reasoning,
        )
        self.turns.append(turn)
        return turn

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
    "ConnectionState",
    "ConversationManager",
    "ConversationTurn",
]

