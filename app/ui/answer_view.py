"""Widgets for rendering conversation turns with metadata and actions."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..services.conversation_manager import ConversationTurn
from ..services.conversation_settings import ConversationSettings
from ..services.progress_service import ProgressService


def _format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "—"
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _format_token_usage(token_usage: dict[str, int] | None) -> str:
    if not token_usage:
        return "Tokens: —"
    ordered_keys = ["prompt_tokens", "completion_tokens", "total_tokens"]
    parts: list[str] = []
    for key in ordered_keys:
        if key in token_usage:
            parts.append(f"{key.replace('_', ' ').title()}: {token_usage[key]}")
    for key, value in token_usage.items():
        if key not in ordered_keys:
            parts.append(f"{key.replace('_', ' ').title()}: {value}")
    return ", ".join(parts) if parts else "Tokens: —"


class TurnCardWidget(QFrame):
    """Card summarizing a single conversation turn."""

    def __init__(
        self,
        turn: ConversationTurn,
        settings: ConversationSettings,
        progress_service: ProgressService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.turn = turn
        self._settings = settings
        self._progress = progress_service
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("turnCard")

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self.question_label = QLabel(f"Q: {turn.question}", self)
        self.question_label.setWordWrap(True)
        self.question_label.setObjectName("turnQuestion")
        layout.addWidget(self.question_label)

        self.answer_browser = QTextBrowser(self)
        self.answer_browser.setReadOnly(True)
        self.answer_browser.setOpenExternalLinks(True)
        self.answer_browser.setObjectName("turnAnswer")
        self.answer_browser.setPlainText(turn.answer)
        self.answer_browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        layout.addWidget(self.answer_browser)

        self.metadata_label = QLabel(self)
        self.metadata_label.setObjectName("turnMetadata")
        layout.addWidget(self.metadata_label)

        self.citations_label = QLabel(self)
        self.citations_label.setObjectName("turnCitations")
        self.citations_label.setWordWrap(True)
        layout.addWidget(self.citations_label)

        self.reasoning_section = self._create_section("Reasoning summary", [])
        layout.addWidget(self.reasoning_section)

        self.plan_section = self._create_section("Plan", [])
        layout.addWidget(self.plan_section)

        self.assumptions_section = self._create_section("Assumptions", [])
        layout.addWidget(self.assumptions_section)

        self.self_check_section = self._create_section("Self-check", [])
        layout.addWidget(self.self_check_section)

        actions_row = QHBoxLayout()
        actions_row.addStretch(1)
        self.copy_answer_button = QPushButton("Copy answer", self)
        self.copy_answer_button.clicked.connect(self._copy_answer)
        actions_row.addWidget(self.copy_answer_button)
        self.copy_citations_button = QPushButton("Copy citations", self)
        self.copy_citations_button.clicked.connect(self._copy_citations)
        actions_row.addWidget(self.copy_citations_button)
        layout.addLayout(actions_row)

        self._apply_turn_data()
        self.apply_settings(settings)

    # ------------------------------------------------------------------
    def _apply_turn_data(self) -> None:
        turn = self.turn
        asked = _format_timestamp(turn.asked_at)
        answered = _format_timestamp(turn.answered_at)
        latency = f"Latency: {turn.latency_ms} ms" if turn.latency_ms is not None else "Latency: —"
        tokens = _format_token_usage(turn.token_usage)
        metadata_text = f"Asked: {asked} | Answered: {answered} | {latency} | {tokens}"
        self.metadata_label.setText(metadata_text)

        if turn.citations:
            if all(isinstance(cite, str) for cite in turn.citations):
                formatted = "\n".join(f"• {cite}" for cite in turn.citations)
            else:
                formatted = "\n".join(f"• {str(cite)}" for cite in turn.citations)
            self.citations_label.setText(f"Citations:\n{formatted}")
        else:
            self.citations_label.setText("Citations: —")

        reasoning_bullets = list(turn.reasoning_bullets)
        self._set_section_content(self.reasoning_section, reasoning_bullets)

        plan_rows: list[str] = []
        for index, item in enumerate(turn.plan, start=1):
            status = item.status.replace("_", " ") if item.status else "pending"
            plan_rows.append(f"{index}. {item.description} [{status}]")
        self._set_section_content(self.plan_section, plan_rows)

        self._set_section_content(self.assumptions_section, list(turn.assumptions))

        decision = turn.assumption_decision
        assumptions_extra: list[str] = []
        if decision:
            mode = decision.mode.title()
            rationale = decision.rationale or ""
            question = decision.clarifying_question or ""
            parts = [f"Decision: {mode}"]
            if rationale:
                parts.append(f"Rationale: {rationale}")
            if question:
                parts.append(f"Follow-up: {question}")
            assumptions_extra.append(" | ".join(parts))
        if assumptions_extra:
            current = self.assumptions_section.findChild(QLabel, "contentLabel")
            if current and current.text():
                current.setText(current.text() + "\n" + "\n".join(assumptions_extra))

        self_check = turn.self_check
        self_check_lines: list[str] = []
        if self_check is not None:
            status = "Passed" if self_check.passed else "Flagged"
            self_check_lines.append(f"Status: {status}")
            if self_check.flags:
                self_check_lines.extend(f"• {flag}" for flag in self_check.flags)
            if self_check.notes:
                self_check_lines.append(self_check.notes)
        self._set_section_content(self.self_check_section, self_check_lines)

    def apply_settings(self, settings: ConversationSettings) -> None:
        self._settings = settings
        self.plan_section.setVisible(settings.show_plan and bool(self.turn.plan))
        assumptions_visible = settings.show_assumptions and (
            bool(self.turn.assumptions) or self.turn.assumption_decision is not None
        )
        self.assumptions_section.setVisible(assumptions_visible)

    def to_plain_text(self) -> str:
        parts = [
            self.question_label.text(),
            self.answer_browser.toPlainText(),
            self.metadata_label.text(),
            self.citations_label.text(),
        ]
        for section in (
            self.reasoning_section,
            self.plan_section,
            self.assumptions_section,
            self.self_check_section,
        ):
            label = section.findChild(QLabel, "titleLabel")
            content = section.findChild(QLabel, "contentLabel")
            if label and content and section.isVisible() and content.text():
                parts.append(f"{label.text()}:\n{content.text()}")
        return "\n".join(filter(None, parts))

    # ------------------------------------------------------------------
    def _copy_answer(self) -> None:
        QApplication.clipboard().setText(self.answer_browser.toPlainText())
        self._progress.notify("Answer copied to clipboard", level="info", duration_ms=1500)

    def _copy_citations(self) -> None:
        QApplication.clipboard().setText(self.citations_label.text())
        self._progress.notify("Citations copied", level="info", duration_ms=1500)

    # ------------------------------------------------------------------
    def _create_section(self, title: str, lines: Iterable[str]) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName(f"section_{title.lower().replace(' ', '_')}")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        title_label = QLabel(title, frame)
        title_label.setObjectName("titleLabel")
        title_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(title_label)
        content_label = QLabel("", frame)
        content_label.setObjectName("contentLabel")
        content_label.setWordWrap(True)
        layout.addWidget(content_label)
        self._set_section_content(frame, list(lines))
        return frame

    def _set_section_content(self, section: QFrame, lines: Iterable[str]) -> None:
        label = section.findChild(QLabel, "contentLabel")
        if label is None:
            return
        filtered = [line for line in lines if line]
        if filtered:
            label.setText("\n".join(filtered))
            section.setVisible(True)
        else:
            label.clear()
            section.setVisible(False)


class AnswerView(QScrollArea):
    """Scrollable list of :class:`ConversationTurn` cards."""

    def __init__(
        self,
        settings: ConversationSettings,
        progress_service: ProgressService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._progress = progress_service
        self._cards: list[TurnCardWidget] = []
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget(self)
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._layout.addStretch(1)
        self.setWidget(container)

        settings.show_plan_changed.connect(self._apply_settings_to_cards)
        settings.show_assumptions_changed.connect(self._apply_settings_to_cards)

    # ------------------------------------------------------------------
    @property
    def cards(self) -> list[TurnCardWidget]:
        return list(self._cards)

    def clear(self) -> None:
        for card in self._cards:
            card.setParent(None)
        self._cards.clear()

    def render_turns(self, turns: Iterable[ConversationTurn]) -> None:
        self.clear()
        for turn in turns:
            self.add_turn(turn)

    def add_turn(self, turn: ConversationTurn) -> TurnCardWidget:
        card = TurnCardWidget(turn, self._settings, self._progress, parent=self.widget())
        self._cards.append(card)
        self._layout.insertWidget(self._layout.count() - 1, card)
        self._scroll_to_bottom()
        return card

    def to_plain_text(self) -> str:
        return "\n\n".join(card.to_plain_text() for card in self._cards)

    # ------------------------------------------------------------------
    def _apply_settings_to_cards(self) -> None:
        for card in self._cards:
            card.apply_settings(self._settings)

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())


__all__ = ["AnswerView", "TurnCardWidget"]

