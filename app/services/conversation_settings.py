"""Reactive conversation settings shared between UI components."""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from .conversation_manager import ReasoningVerbosity, ResponseMode


class ConversationSettings(QObject):
    """Expose conversation level toggles to coordinate UI and requests."""

    reasoning_verbosity_changed = pyqtSignal(object)
    show_plan_changed = pyqtSignal(bool)
    show_assumptions_changed = pyqtSignal(bool)
    sources_only_mode_changed = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._reasoning_verbosity = ReasoningVerbosity.BRIEF
        self._show_plan = True
        self._show_assumptions = True
        self._sources_only_mode = False

    # ------------------------------------------------------------------
    @property
    def reasoning_verbosity(self) -> ReasoningVerbosity:
        return self._reasoning_verbosity

    def set_reasoning_verbosity(self, verbosity: ReasoningVerbosity) -> None:
        if not isinstance(verbosity, ReasoningVerbosity):
            raise TypeError("verbosity must be a ReasoningVerbosity value")
        if verbosity is self._reasoning_verbosity:
            return
        self._reasoning_verbosity = verbosity
        self.reasoning_verbosity_changed.emit(verbosity)

    # ------------------------------------------------------------------
    @property
    def show_plan(self) -> bool:
        return self._show_plan

    def set_show_plan(self, enabled: bool) -> None:
        value = bool(enabled)
        if value == self._show_plan:
            return
        self._show_plan = value
        self.show_plan_changed.emit(value)

    # ------------------------------------------------------------------
    @property
    def show_assumptions(self) -> bool:
        return self._show_assumptions

    def set_show_assumptions(self, enabled: bool) -> None:
        value = bool(enabled)
        if value == self._show_assumptions:
            return
        self._show_assumptions = value
        self.show_assumptions_changed.emit(value)

    # ------------------------------------------------------------------
    @property
    def sources_only_mode(self) -> bool:
        return self._sources_only_mode

    def set_sources_only_mode(self, enabled: bool) -> None:
        value = bool(enabled)
        if value == self._sources_only_mode:
            return
        self._sources_only_mode = value
        self.sources_only_mode_changed.emit(value)

    # ------------------------------------------------------------------
    @property
    def response_mode(self) -> ResponseMode:
        return (
            ResponseMode.SOURCES_ONLY
            if self._sources_only_mode
            else ResponseMode.GENERATIVE
        )


__all__ = ["ConversationSettings"]

