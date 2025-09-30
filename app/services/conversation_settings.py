"""Reactive conversation settings shared between UI components."""

from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal

from .conversation_manager import ReasoningVerbosity, ResponseMode
from .lmstudio_client import AnswerLength


logger = logging.getLogger(__name__)


class ConversationSettings(QObject):
    """Expose conversation level toggles to coordinate UI and requests."""

    reasoning_verbosity_changed = pyqtSignal(object)
    show_plan_changed = pyqtSignal(bool)
    show_assumptions_changed = pyqtSignal(bool)
    sources_only_mode_changed = pyqtSignal(bool)
    answer_length_changed = pyqtSignal(object)
    model_changed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._reasoning_verbosity = ReasoningVerbosity.BRIEF
        self._show_plan = True
        self._show_assumptions = True
        self._sources_only_mode = False
        self._answer_length = AnswerLength.NORMAL
        self._model_name = "lmstudio"

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
        logger.info(
            "Reasoning verbosity changed",
            extra={"verbosity": verbosity.name},
        )
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
        logger.info("Show plan toggled", extra={"enabled": value})
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
        logger.info("Show assumptions toggled", extra={"enabled": value})
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
        logger.info("Sources only mode toggled", extra={"enabled": value})
        self.sources_only_mode_changed.emit(value)

    # ------------------------------------------------------------------
    @property
    def response_mode(self) -> ResponseMode:
        return (
            ResponseMode.SOURCES_ONLY
            if self._sources_only_mode
            else ResponseMode.GENERATIVE
        )

    # ------------------------------------------------------------------
    @property
    def answer_length(self) -> AnswerLength:
        return self._answer_length

    def set_answer_length(self, preset: AnswerLength) -> None:
        if not isinstance(preset, AnswerLength):
            raise TypeError("preset must be an AnswerLength value")
        if preset is self._answer_length:
            return
        self._answer_length = preset
        logger.info(
            "Answer length changed",
            extra={"answer_length": preset.name},
        )
        self.answer_length_changed.emit(preset)

    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._model_name

    def set_model_name(self, name: str) -> None:
        cleaned = str(name).strip()
        if not cleaned:
            return
        if cleaned == self._model_name:
            return
        self._model_name = cleaned
        logger.info("Model changed", extra={"model": cleaned})
        self.model_changed.emit(cleaned)


__all__ = ["ConversationSettings"]

