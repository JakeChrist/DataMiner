"""Chat-style conversation widgets for displaying LLM responses."""

from __future__ import annotations

import html
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from PyQt6.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QTextOption
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..services.conversation_manager import ConversationTurn
from ..services.conversation_settings import ConversationSettings
from ..services.progress_service import ProgressService
from ..services.settings_service import ChatStyleSettings


logger = logging.getLogger(__name__)

_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_CITATION_RE = re.compile(r'\[(\d+)\](?!\()')
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")
_CODE_FENCE_RE = re.compile(r"^```(.*)$")


@dataclass(slots=True)
class MessageBlock:
    """Representation of a block-level segment of a chat message."""

    type: str
    text: str = ""
    items: list[str] = field(default_factory=list)
    language: str | None = None


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


def _parse_blocks(text: str) -> list[MessageBlock]:
    """Parse raw assistant text into renderable blocks."""

    blocks: list[MessageBlock] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        fence_match = _CODE_FENCE_RE.match(line)
        if fence_match:
            language = fence_match.group(1).strip() or None
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not _CODE_FENCE_RE.match(lines[index]):
                code_lines.append(lines[index])
                index += 1
            if index < len(lines) and _CODE_FENCE_RE.match(lines[index]):
                index += 1
            blocks.append(MessageBlock(type="code", text="\n".join(code_lines), language=language))
            continue

        if not line.strip():
            index += 1
            continue

        if _LIST_ITEM_RE.match(line):
            items: list[str] = []
            while index < len(lines) and _LIST_ITEM_RE.match(lines[index]):
                stripped = _LIST_ITEM_RE.sub("", lines[index], count=1)
                items.append(stripped.rstrip())
                index += 1
            blocks.append(MessageBlock(type="list", items=items))
            continue

        paragraph_lines = [line.rstrip()]
        index += 1
        while index < len(lines):
            if not lines[index].strip():
                break
            if _LIST_ITEM_RE.match(lines[index]) or _CODE_FENCE_RE.match(lines[index]):
                break
            paragraph_lines.append(lines[index].rstrip())
            index += 1
        blocks.append(MessageBlock(type="paragraph", text="\n".join(paragraph_lines)))
    return blocks


def _render_inline_html(text: str, *, highlight: int | None, accent: str) -> tuple[str, bool]:
    """Return HTML for a paragraph/list item with inline styling."""

    fragments: list[str] = []
    last_index = 0
    has_citation = False
    for match in _INLINE_CODE_RE.finditer(text):
        fragments.append(html.escape(text[last_index:match.start()]))
        fragments.append(
            f"<code class='inline'>{html.escape(match.group(1))}</code>"
        )
        last_index = match.end()
    fragments.append(html.escape(text[last_index:]))
    escaped = "".join(fragments)

    def _replace(match: re.Match[str]) -> str:
        nonlocal has_citation
        index = int(match.group(1))
        has_citation = True
        classes = ["citation"]
        if highlight == index:
            classes.append("selected")
        joined = " ".join(classes)
        return (
            f"<a href='cite-{index}' class='{joined}' data-citation='{index}' "
            f"style='color:{accent};text-decoration:none;'>[{index}]</a>"
        )

    html_text = _CITATION_RE.sub(_replace, escaped)
    html_text = html_text.replace("\n", "<br/>")
    return html_text, has_citation


class CitationPopover(QFrame):
    """Floating preview for citation snippets."""

    title_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.ToolTip)
        self.setObjectName("citationPopover")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        self._title_button = QPushButton("", self)
        self._title_button.setFlat(True)
        self._title_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._title_button.clicked.connect(self.title_clicked)
        self._title_button.setStyleSheet("text-align: left;")
        self._snippet_label = QLabel("", self)
        self._snippet_label.setWordWrap(True)
        layout.addWidget(self._title_button)
        layout.addWidget(self._snippet_label)
        self._timer = QTimer(self)
        self._timer.setInterval(4500)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_payload(self, title: str, snippet: str, *, position: QPoint) -> None:
        self._title_button.setText(title or "Open source")
        self._snippet_label.setText(snippet or "No preview available.")
        self.adjustSize()
        self.move(position)
        self.show()
        self.raise_()
        self._timer.start()


class TextBlockWidget(QTextBrowser):
    """Display a paragraph or bullet list item with inline citations."""

    citation_activated = pyqtSignal(int)

    def __init__(
        self,
        text: str,
        *,
        accent: str,
        highlight: int | None = None,
    ) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.document().setDocumentMargin(0)
        self.document().setDefaultTextOption(QTextOption(Qt.AlignmentFlag.AlignLeft))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._raw_text = text
        self._accent = accent
        self._highlight = highlight
        self._has_citation = False
        self._sources_only = False
        self.anchorClicked.connect(self._on_anchor)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self._render()

    @property
    def has_citation(self) -> bool:
        return self._has_citation

    def set_highlight(self, citation: int | None) -> None:
        if citation == self._highlight:
            return
        self._highlight = citation
        self._render()

    def set_sources_only(self, enabled: bool) -> None:
        if enabled == self._sources_only:
            return
        self._sources_only = enabled
        self._update_visibility()

    def set_accent(self, color: str) -> None:
        if color == self._accent:
            return
        self._accent = color
        self._render()

    def _render(self) -> None:
        html_text, has_citation = _render_inline_html(
            self._raw_text, highlight=self._highlight, accent=self._accent
        )
        self._has_citation = has_citation
        self.setHtml(f"<div class='chat-paragraph'>{html_text}</div>")
        self._update_visibility()
        self.document().adjustSize()
        self.setMinimumHeight(math.ceil(self.document().size().height()))

    def _update_visibility(self) -> None:
        self.setVisible(not (self._sources_only and not self._has_citation))

    def _on_anchor(self, url):  # pragma: no cover - signal routing
        target = url.toString()
        if target.startswith("cite-"):
            try:
                index = int(target.split("-", 1)[1])
            except (ValueError, IndexError):
                return
            self.citation_activated.emit(index)


class CodeBlockWidget(QFrame):
    """Dedicated widget for fenced code blocks with copy/collapse controls."""

    MAX_VISIBLE_LINES = 16

    def __init__(self, code: str, language: str | None, *, background: str) -> None:
        super().__init__()
        self.setObjectName("codeBlock")
        self._code = code.rstrip("\n")
        self._language = (language or "").strip()
        self._collapsed = False
        self._background = background

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        language_label = QLabel(self._language or "Code", self)
        language_label.setObjectName("codeLanguageLabel")
        header.addWidget(language_label)
        header.addStretch(1)

        self._copy_button = QToolButton(self)
        self._copy_button.setText("Copy")
        self._copy_button.clicked.connect(self._copy_code)
        header.addWidget(self._copy_button)

        self._toggle_button = QToolButton(self)
        self._toggle_button.setText("Collapse")
        self._toggle_button.clicked.connect(self._toggle_collapsed)
        header.addWidget(self._toggle_button)

        layout.addLayout(header)

        self._editor = QPlainTextEdit(self)
        self._editor.setObjectName("codeEditor")
        self._editor.setReadOnly(True)
        self._editor.setFrameShape(QFrame.Shape.NoFrame)
        self._editor.setPlainText(self._code)
        self._editor.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        self._editor.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        layout.addWidget(self._editor)

        self._apply_background()
        self._update_toggle_state(initial=True)

    def _apply_background(self) -> None:
        self.setStyleSheet(
            f"QFrame#codeBlock {{border-radius: 8px; background-color: {self._background};}}"
        )
        self._editor.setStyleSheet(
            f"QPlainTextEdit#codeEditor {{background-color: {self._background}; border: none;}}"
        )

    def set_background(self, color: str) -> None:
        if color == self._background:
            return
        self._background = color
        self._apply_background()

    def expand(self) -> None:
        if not self._collapsed:
            return
        self._collapsed = False
        self._update_toggle_state()

    def collapse(self) -> None:
        if self._collapsed:
            return
        self._collapsed = True
        self._update_toggle_state()

    def _toggle_collapsed(self) -> None:
        self._collapsed = not self._collapsed
        self._update_toggle_state()

    def _update_toggle_state(self, *, initial: bool = False) -> None:
        total_lines = self._code.count("\n") + 1
        if total_lines <= self.MAX_VISIBLE_LINES:
            self._toggle_button.setVisible(False)
            full_height = math.ceil(self._editor.document().size().height()) + 12
            self._editor.setMaximumHeight(full_height)
            return
        if initial and total_lines > self.MAX_VISIBLE_LINES:
            self._collapsed = True
        self._toggle_button.setVisible(True)
        if self._collapsed:
            metrics = self._editor.fontMetrics()
            height = metrics.height() * self.MAX_VISIBLE_LINES + 12
            self._editor.setMaximumHeight(height)
            self._toggle_button.setText("Expand code")
        else:
            full_height = math.ceil(self._editor.document().size().height()) + 12
            self._editor.setMaximumHeight(full_height)
            self._toggle_button.setText("Collapse")

    def _copy_code(self) -> None:  # pragma: no cover - clipboard access
        QApplication.clipboard().setText(self._code)


class PlanSection(QFrame):
    """Accordion-style section controlled by a chip button."""

    def __init__(self, title: str, rows: list[str]) -> None:
        super().__init__()
        self.setObjectName("planSection")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for row in rows:
            label = QLabel(row, self)
            label.setWordWrap(True)
            layout.addWidget(label)
        self.setVisible(False)


class ChatBubbleWidget(QFrame):
    """Base widget representing a speaker bubble in the chat transcript."""

    def __init__(
        self,
        *,
        speaker: str,
        background: str,
        accent: str,
        progress: ProgressService,
    ) -> None:
        super().__init__()
        self._speaker = speaker
        self._progress = progress
        self._accent = accent
        self._background = background
        self.setObjectName(f"chatBubble_{speaker}")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._hover_actions: list[QPushButton] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(8)

        self._content_layout = QVBoxLayout()
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(10)
        outer.addLayout(self._content_layout)

        self._meta_label = QLabel("", self)
        self._meta_label.setObjectName("bubbleMeta")
        self._meta_label.setWordWrap(True)
        self._meta_label.hide()
        outer.addWidget(self._meta_label)

        self._actions_bar = QHBoxLayout()
        self._actions_bar.setContentsMargins(0, 0, 0, 0)
        self._actions_bar.setSpacing(6)
        self._actions_widget = QFrame(self)
        self._actions_widget.setLayout(self._actions_bar)
        self._actions_widget.hide()
        outer.addWidget(self._actions_widget)

        self._typing_label = QLabel("", self)
        self._typing_label.setObjectName("typingIndicator")
        self._typing_label.hide()
        outer.addWidget(self._typing_label)

        radius = 18
        self._apply_background()

    def add_widget(self, widget: QWidget) -> None:
        self._content_layout.addWidget(widget)

    def add_layout(self, layout: QVBoxLayout) -> None:
        self._content_layout.addLayout(layout)

    def set_metadata(self, text: str) -> None:
        self._meta_label.setText(text)

    def set_metadata_visible(self, visible: bool) -> None:
        self._meta_label.setVisible(visible)

    def add_action(self, label: str, callback) -> None:
        button = QPushButton(label, self)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(callback)
        self._actions_bar.addWidget(button)
        self._hover_actions.append(button)

    def set_typing(self, active: bool) -> None:
        self._typing_label.setVisible(active)
        if active:
            self._typing_label.setText("Assistant is typing…")

    def enterEvent(self, event):  # pragma: no cover - UI behaviour
        super().enterEvent(event)
        self._actions_widget.setVisible(True)

    def leaveEvent(self, event):  # pragma: no cover
        super().leaveEvent(event)
        self._actions_widget.setVisible(False)

    def notify_copy(self, message: str) -> None:
        self._progress.notify(message, level="info", duration_ms=1500)

    def set_background(self, color: str) -> None:
        if color == self._background:
            return
        self._background = color
        self._apply_background()

    @property
    def accent(self) -> str:
        return self._accent

    def set_accent(self, color: str) -> None:
        self._accent = color

    def _apply_background(self) -> None:
        radius = 18
        self.setStyleSheet(
            f"QFrame#chatBubble_{self._speaker} {{"
            f"background-color: {self._background};"
            "border: 0;"
            f"border-radius: {radius}px;"
            "color: palette(text);"
            "}}"
        )


class UserBubbleWidget(ChatBubbleWidget):
    """Bubble rendering the user prompt."""

    def __init__(
        self,
        *,
        text: str,
        background: str,
        accent: str,
        progress: ProgressService,
    ) -> None:
        super().__init__(speaker="user", background=background, accent=accent, progress=progress)
        self._raw_text = text
        self._text_widget = TextBlockWidget(text, accent=accent)
        self.add_widget(self._text_widget)
        self.add_action("Copy", self._copy_text)

    def _copy_text(self) -> None:  # pragma: no cover - clipboard
        QApplication.clipboard().setText(self._raw_text)
        self.notify_copy("Prompt copied")

    def apply_colors(self, *, background: str, accent: str) -> None:
        self.set_background(background)
        self.set_accent(accent)
        self._text_widget.set_accent(accent)


class AssistantBubbleWidget(ChatBubbleWidget):
    """Bubble responsible for rendering the assistant response."""

    citation_activated = pyqtSignal(int)

    def __init__(
        self,
        turn: ConversationTurn,
        *,
        settings: ConversationSettings,
        background: str,
        accent: str,
        code_background: str,
        progress: ProgressService,
    ) -> None:
        super().__init__(speaker="assistant", background=background, accent=accent, progress=progress)
        self.turn = turn
        self._settings = settings
        self._code_background = code_background
        self._selected_citation: int | None = None
        self._sources_only = settings.sources_only_mode
        self._code_blocks: list[CodeBlockWidget] = []
        self._text_blocks: list[TextBlockWidget] = []
        self._plan_sections: dict[str, tuple[QToolButton, PlanSection]] = {}
        self._popover = CitationPopover(self)
        self._popover.title_clicked.connect(self._emit_current_citation)

        blocks = _parse_blocks(turn.answer or "")
        for block in blocks:
            if block.type == "code":
                code_widget = CodeBlockWidget(block.text, block.language, background=code_background)
                self._code_blocks.append(code_widget)
                self.add_widget(code_widget)
            elif block.type == "list":
                container = QVBoxLayout()
                container.setContentsMargins(0, 0, 0, 0)
                container.setSpacing(4)
                for item in block.items:
                    text_widget = TextBlockWidget(item, accent=accent)
                    text_widget.citation_activated.connect(self._on_citation_activated)
                    self._text_blocks.append(text_widget)
                    container.addWidget(text_widget)
                wrapper = QFrame(self)
                wrapper.setLayout(container)
                wrapper.setObjectName("listWrapper")
                self.add_widget(wrapper)
            else:
                text_widget = TextBlockWidget(block.text, accent=accent)
                text_widget.citation_activated.connect(self._on_citation_activated)
                self._text_blocks.append(text_widget)
                self.add_widget(text_widget)

        for block in self._text_blocks:
            block.set_sources_only(self._sources_only)

        self._create_plan_sections()

        metadata = [
            f"Model: {turn.model_name or '—'}",
            f"Asked: {_format_timestamp(turn.asked_at)}",
            f"Answered: {_format_timestamp(turn.answered_at)}",
            f"Latency: {turn.latency_ms or '—'} ms",
            _format_token_usage(turn.token_usage),
        ]
        self.set_metadata(" | ".join(metadata))

        self.add_action("Copy", self._copy_answer)
        self.add_action("Expand code", self._expand_all_code)
        self.add_action("Toggle plan", self._toggle_plan_sections)
        self.add_action("Info", self._toggle_metadata)

        self._settings.sources_only_mode_changed.connect(self._on_sources_only_changed)
        self._settings.show_assumptions_changed.connect(self._on_assumptions_toggle)

    def _create_plan_sections(self) -> None:
        chips = QHBoxLayout()
        chips.setContentsMargins(0, 0, 0, 0)
        chips.setSpacing(6)
        chip_wrapper = QFrame(self)
        chip_wrapper.setLayout(chips)
        has_chip = False

        def _add_chip(label: str, section: PlanSection) -> None:
            nonlocal has_chip
            button = QToolButton(self)
            button.setText(label)
            button.setCheckable(True)
            button.setChecked(False)
            button.toggled.connect(section.setVisible)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setObjectName("planChip")
            chips.addWidget(button)
            has_chip = True
            self._plan_sections[label] = (button, section)

        plan_rows: list[str] = []
        for index, item in enumerate(self.turn.plan, start=1):
            status = item.status.replace("_", " ") if item.status else "pending"
            plan_rows.append(f"{index}. {item.description} [{status}]")
        if plan_rows:
            section = PlanSection("Plan", plan_rows)
            _add_chip("Plan", section)
            self.add_widget(section)

        assumptions: list[str] = list(self.turn.assumptions)
        decision = self.turn.assumption_decision
        if decision:
            decision_parts = [f"Decision: {decision.mode.title()}"]
            if decision.rationale:
                decision_parts.append(f"Rationale: {decision.rationale}")
            if decision.clarifying_question:
                decision_parts.append(f"Follow-up: {decision.clarifying_question}")
            assumptions.append(" | ".join(decision_parts))
        if assumptions and self._settings.show_assumptions:
            section = PlanSection("Assumptions", assumptions)
            _add_chip("Assumptions", section)
            self.add_widget(section)

        evidence_rows: list[str] = []
        for result in self.turn.step_results:
            prefix = f"{result.index}. {result.description.strip() or 'Step'}"
            summary = result.answer.strip() or "No summary recorded"
            markers = " ".join(f"[{idx}]" for idx in result.citation_indexes)
            evidence_rows.append(f"{prefix}\n    {summary} {markers}".strip())
        if evidence_rows:
            section = PlanSection("Evidence Map", evidence_rows)
            _add_chip("Evidence Map", section)
            self.add_widget(section)

        critiques: list[str] = []
        if self.turn.self_check:
            status = "Passed" if self.turn.self_check.passed else "Flagged"
            critiques.append(f"Self-check: {status}")
            for flag in self.turn.self_check.flags:
                critiques.append(f"• {flag}")
            if self.turn.self_check.notes:
                critiques.append(self.turn.self_check.notes)
        if self.turn.adversarial_review:
            critiques.append(f"Judge: {self.turn.adversarial_review.decision}")
            for reason in self.turn.adversarial_review.reasons:
                critiques.append(f"• {reason}")
        if critiques:
            section = PlanSection("Critiques", critiques)
            _add_chip("Critiques", section)
            self.add_widget(section)

        if has_chip:
            self.layout().insertWidget(0, chip_wrapper)
        else:
            chip_wrapper.deleteLater()

    def _on_sources_only_changed(self, enabled: bool) -> None:
        self._sources_only = enabled
        for block in self._text_blocks:
            block.set_sources_only(enabled)

    def _toggle_plan_sections(self) -> None:
        if not self._plan_sections:
            return
        expanded = any(section.isVisible() for _button, section in self._plan_sections.values())
        target = not expanded
        for button, section in self._plan_sections.values():
            button.blockSignals(True)
            button.setChecked(target)
            section.setVisible(target)
            button.blockSignals(False)

    def _toggle_metadata(self) -> None:
        self.set_metadata_visible(not self._meta_label.isVisible())

    def _copy_answer(self) -> None:  # pragma: no cover - clipboard
        QApplication.clipboard().setText(self.turn.answer or "")
        self.notify_copy("Answer copied")

    def _expand_all_code(self) -> None:
        for block in self._code_blocks:
            block.expand()

    def _on_citation_activated(self, index: int) -> None:
        self._selected_citation = index
        for block in self._text_blocks:
            block.set_highlight(index)
        self._show_citation_preview(index)
        self.citation_activated.emit(index)

    def _emit_current_citation(self) -> None:
        if self._selected_citation is None:
            return
        self.citation_activated.emit(self._selected_citation)

    def _show_citation_preview(self, index: int) -> None:
        if index < 1 or index > len(self.turn.citations):
            self._popover.hide()
            return
        citation = self.turn.citations[index - 1]
        title, snippet = _describe_citation(citation)
        pos = QCursor.pos()
        self._popover.show_payload(title, snippet, position=pos + QPoint(12, 12))

    def set_selected_citation(self, index: int | None) -> None:
        self._selected_citation = index
        for block in self._text_blocks:
            block.set_highlight(index)
        if index is None:
            self._popover.hide()

    def set_plan_expanded(self, expanded: bool) -> None:
        for button, section in self._plan_sections.values():
            button.blockSignals(True)
            button.setChecked(expanded)
            section.setVisible(expanded)
            button.blockSignals(False)

    def set_background_colors(
        self, *, background: str, code_background: str, accent: str
    ) -> None:
        self.set_background(background)
        self.set_accent(accent)
        for block in self._text_blocks:
            block.set_accent(accent)
        for code_block in self._code_blocks:
            code_block.set_background(code_background)

    def _on_assumptions_toggle(self, enabled: bool) -> None:
        entry = self._plan_sections.get("Assumptions")
        if not entry:
            return
        button, section = entry
        button.setVisible(enabled)
        if not enabled:
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
            section.setVisible(False)
        else:
            section.setVisible(button.isChecked())


def _describe_citation(citation: Any) -> tuple[str, str]:
    if isinstance(citation, dict):
        title = str(
            citation.get("title")
            or citation.get("source")
            or citation.get("document")
            or citation.get("path")
            or "Source"
        )
        snippet = str(citation.get("snippet") or citation.get("preview") or "")
        return title, snippet
    return str(citation), ""


class ChatTurnWidget(QFrame):
    """Container widget grouping the user and assistant bubbles."""

    citation_activated = pyqtSignal(int)

    def __init__(
        self,
        turn: ConversationTurn,
        *,
        settings: ConversationSettings,
        colors: ChatStyleSettings,
        progress: ProgressService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.turn = turn
        self.setObjectName("chatTurn")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        user_row = QHBoxLayout()
        user_row.setContentsMargins(0, 0, 0, 0)
        user_row.setSpacing(4)
        self.user_bubble = UserBubbleWidget(
            text=turn.question,
            background=colors.user_bubble_color,
            accent=colors.citation_accent,
            progress=progress,
        )
        user_row.addWidget(self.user_bubble, alignment=Qt.AlignmentFlag.AlignLeft)
        user_row.addStretch(1)
        layout.addLayout(user_row)

        assistant_row = QHBoxLayout()
        assistant_row.setContentsMargins(0, 0, 0, 0)
        assistant_row.setSpacing(4)
        self.assistant_bubble = AssistantBubbleWidget(
            turn,
            settings=settings,
            background=colors.ai_bubble_color,
            accent=colors.citation_accent,
            code_background=colors.code_block_background,
            progress=progress,
        )
        self.assistant_bubble.citation_activated.connect(self.citation_activated.emit)
        assistant_row.addStretch(1)
        assistant_row.addWidget(self.assistant_bubble, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(assistant_row)

    def to_plain_text(self) -> str:
        parts = [
            f"You: {self.turn.question}",
            f"Assistant: {self.turn.answer}",
        ]
        return "\n".join(parts)

    def set_plan_expanded(self, expanded: bool) -> None:
        self.assistant_bubble.set_plan_expanded(expanded)

    def set_selected_citation(self, index: int | None) -> None:
        self.assistant_bubble.set_selected_citation(index)

    def apply_colors(self, colors: ChatStyleSettings) -> None:
        self.user_bubble.apply_colors(
            background=colors.user_bubble_color,
            accent=colors.citation_accent,
        )
        self.assistant_bubble.set_background_colors(
            background=colors.ai_bubble_color,
            code_background=colors.code_block_background,
            accent=colors.citation_accent,
        )


class AnswerView(QScrollArea):
    """Scrollable container of chat turns rendered in chronological order."""

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
        self._colors = ChatStyleSettings()
        self._turns: list[ChatTurnWidget] = []
        self._density = "comfortable"

        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget(self)
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(12)
        self._layout.addStretch(1)
        self.setWidget(container)

        settings.show_plan_changed.connect(self._toggle_plan_sections)
        settings.sources_only_mode_changed.connect(self._update_sources_only_state)

    # ------------------------------------------------------------------
    @property
    def turns(self) -> list[ChatTurnWidget]:
        return list(self._turns)

    def clear(self) -> None:
        for turn in self._turns:
            turn.setParent(None)
        self._turns.clear()

    def render_turns(self, turns: Iterable[ConversationTurn]) -> None:
        self.clear()
        for turn in turns:
            self.add_turn(turn)

    def add_turn(self, turn: ConversationTurn) -> ChatTurnWidget:
        widget = ChatTurnWidget(
            turn,
            settings=self._settings,
            colors=self._colors,
            progress=self._progress,
            parent=self.widget(),
        )
        widget.citation_activated.connect(lambda index, item=widget: self._emit_citation(item, index))
        self._turns.append(widget)
        self._layout.insertWidget(self._layout.count() - 1, widget)
        widget.assistant_bubble.set_plan_expanded(self._settings.show_plan)
        self._scroll_to_bottom()
        return widget

    def to_plain_text(self) -> str:
        return "\n\n".join(turn.to_plain_text() for turn in self._turns)

    def highlight_citation(self, item: ChatTurnWidget | None, index: int | None) -> None:
        for turn in self._turns:
            if turn is item:
                turn.set_selected_citation(index)
            else:
                turn.set_selected_citation(None)

    def apply_colors(self, colors: ChatStyleSettings) -> None:
        self._colors = colors
        for turn in self._turns:
            turn.apply_colors(colors)

    def _toggle_plan_sections(self, enabled: bool) -> None:
        for turn in self._turns[-1:]:
            turn.set_plan_expanded(enabled)

    def _update_sources_only_state(self, enabled: bool) -> None:
        self.highlight_citation(None, None)

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())

    def _emit_citation(self, turn: ChatTurnWidget, index: int) -> None:
        logger.info(
            "Citation activated in chat view",
            extra={"question": turn.turn.question, "citation_index": index},
        )
        self.citation_activated.emit(turn, index)

    def set_density(self, density: str) -> None:
        normalized = "compact" if density == "compact" else "comfortable"
        if normalized == self._density:
            return
        self._density = normalized
        spacing = 8 if normalized == "compact" else 12
        if isinstance(self._layout, QVBoxLayout):
            self._layout.setSpacing(spacing)


__all__ = ["AnswerView", "ChatTurnWidget"]
