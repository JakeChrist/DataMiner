"""Widgets for rendering conversation turns with metadata and actions."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Iterable

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
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

    citation_activated = pyqtSignal(int)

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
        self.answer_browser.setOpenExternalLinks(False)
        self.answer_browser.setOpenLinks(False)
        self.answer_browser.setObjectName("turnAnswer")
        self.answer_browser.anchorClicked.connect(self._on_anchor_clicked)
        self.answer_browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        layout.addWidget(self.answer_browser)

        self.metadata_label = QLabel(self)
        self.metadata_label.setObjectName("turnMetadata")
        layout.addWidget(self.metadata_label)

        self.citations_browser = QTextBrowser(self)
        self.citations_browser.setObjectName("turnCitations")
        self.citations_browser.setReadOnly(True)
        self.citations_browser.setOpenExternalLinks(False)
        self.citations_browser.setOpenLinks(False)
        self.citations_browser.anchorClicked.connect(self._on_anchor_clicked)
        layout.addWidget(self.citations_browser)

        self.reasoning_section = self._create_section("Reasoning summary", [])
        layout.addWidget(self.reasoning_section)

        self.plan_section = self._create_section("Plan", [])
        layout.addWidget(self.plan_section)

        self.step_results_section = self._create_section("Step results", [])
        layout.addWidget(self.step_results_section)

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

        self._selected_citation: int | None = None
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

        self._render_answer()
        self._render_citations()

        reasoning_bullets = list(turn.reasoning_bullets)
        self._set_section_content(self.reasoning_section, reasoning_bullets)

        plan_rows: list[str] = []
        for index, item in enumerate(turn.plan, start=1):
            status = item.status.replace("_", " ") if item.status else "pending"
            plan_rows.append(f"{index}. {item.description} [{status}]")
        self._set_section_content(self.plan_section, plan_rows)

        step_lines: list[str] = []
        for result in turn.step_results:
            description = result.description.strip() or f"Step {result.index}"
            step_lines.append(f"{result.index}. {description}")
            answer_text = result.answer.strip()
            markers = "".join(f"[{idx}]" for idx in result.citation_indexes)
            if answer_text or markers:
                summary = answer_text or "No evidence recorded"
                if markers:
                    summary = f"{summary} {markers}".strip()
                step_lines.append(f"    {summary}")
        self._set_section_content(self.step_results_section, step_lines)

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
        self.step_results_section.setVisible(bool(self.turn.step_results))
        assumptions_visible = settings.show_assumptions and (
            bool(self.turn.assumptions) or self.turn.assumption_decision is not None
        )
        self.assumptions_section.setVisible(assumptions_visible)

    def to_plain_text(self) -> str:
        parts = [
            self.question_label.text(),
            self.answer_browser.toPlainText(),
            self.metadata_label.text(),
            self.citations_browser.toPlainText(),
        ]
        for section in (
            self.reasoning_section,
            self.plan_section,
            self.step_results_section,
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
        QApplication.clipboard().setText(self.citations_browser.toPlainText())
        self._progress.notify("Citations copied", level="info", duration_ms=1500)

    def set_selected_citation(self, index: int | None) -> None:
        if index == self._selected_citation:
            return
        self._selected_citation = index
        self._render_answer()
        self._render_citations()

    @property
    def selected_citation(self) -> int | None:
        return self._selected_citation

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

    def _render_answer(self) -> None:
        answer = html.escape(self.turn.answer or "")
        answer = answer.replace("\n", "<br/>")
        total_citations = len(self.turn.citations)
        for index in range(1, total_citations + 1):
            placeholder = f"[{index}]"
            highlighted = placeholder
            classes = []
            if self._selected_citation == index:
                classes.append("selected-citation")
                highlighted = f"<span class='selected-citation'>{placeholder}</span>"
            anchor = f"<a href='cite-{index}'>{highlighted}</a>"
            answer = answer.replace(placeholder, anchor)
        if self._selected_citation is not None:
            style = "<style>.selected-citation{background-color:rgba(255,230,128,0.6);}</style>"
        else:
            style = ""
        self.answer_browser.setHtml(style + answer)

    def _render_citations(self) -> None:
        if not self.turn.citations:
            self.citations_browser.setHtml("<p>Citations: —</p>")
            return
        rows: list[str] = ["<p>Citations:</p><ul>"]
        for idx, citation in enumerate(self.turn.citations, start=1):
            if isinstance(citation, str):
                label = citation
            elif isinstance(citation, dict):
                base_label = str(
                    citation.get("source")
                    or citation.get("title")
                    or citation.get("document")
                    or citation.get("path")
                    or citation.get("snippet")
                    or citation
                )
                steps = citation.get("steps")
                if isinstance(steps, Iterable) and not isinstance(steps, (str, bytes, dict)):
                    step_ids = sorted({str(step) for step in steps if str(step).strip()})
                    if step_ids:
                        label = f"Steps {', '.join(step_ids)} – {base_label}" if base_label else f"Steps {', '.join(step_ids)}"
                    else:
                        label = base_label
                else:
                    label = base_label
            else:
                label = str(citation)
            safe_label = html.escape(label)
            if self._selected_citation == idx:
                safe_label = (
                    "<span class='selected-citation'>"
                    + safe_label
                    + "</span>"
                )
            rows.append(f"<li><a href='cite-{idx}'>[{idx}]</a> {safe_label}</li>")
        rows.append("</ul>")
        if self._selected_citation is not None:
            style = "<style>.selected-citation{background-color:rgba(255,230,128,0.6);}</style>"
        else:
            style = ""
        self.citations_browser.setHtml(style + "".join(rows))

    def _on_anchor_clicked(self, url: QUrl) -> None:
        target = url.toString()
        if target.startswith("cite-"):
            try:
                index = int(target.split("-", 1)[1])
            except (ValueError, IndexError):
                return
            self.set_selected_citation(index)
            self.citation_activated.emit(index)


class AnswerView(QScrollArea):
    """Scrollable list of :class:`ConversationTurn` cards."""

    citation_activated = pyqtSignal(object, int)

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
        card.citation_activated.connect(lambda index, card=card: self._emit_citation(card, index))
        card.set_selected_citation(None)
        self._scroll_to_bottom()
        return card

    def to_plain_text(self) -> str:
        return "\n\n".join(card.to_plain_text() for card in self._cards)

    # ------------------------------------------------------------------
    def highlight_citation(self, card: TurnCardWidget | None, index: int | None) -> None:
        for current in self._cards:
            if current is card:
                current.set_selected_citation(index)
            else:
                current.set_selected_citation(None)

    def _apply_settings_to_cards(self) -> None:
        for card in self._cards:
            card.apply_settings(self._settings)

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())

    def _emit_citation(self, card: TurnCardWidget, index: int) -> None:
        self.citation_activated.emit(card, index)


__all__ = ["AnswerView", "TurnCardWidget"]

