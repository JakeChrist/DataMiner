"""Chat-style conversation widgets for displaying LLM responses."""

from __future__ import annotations

import html
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from PyQt6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
    QTimer,
    pyqtSignal,
)
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
        self._title_label = QLabel("", self)
        self._title_label.setWordWrap(True)
        self._title_label.setObjectName("citationTitle")
        self._snippet_label = QLabel("", self)
        self._snippet_label.setWordWrap(True)
        self._snippet_label.setObjectName("citationSnippet")
        self._action_button = QPushButton("View in Evidence", self)
        self._action_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._action_button.clicked.connect(self.title_clicked)
        self._action_button.setDefault(False)
        self._action_button.setAutoDefault(False)
        layout.addWidget(self._title_label)
        layout.addWidget(self._snippet_label)
        layout.addWidget(self._action_button)
        self._timer = QTimer(self)
        self._timer.setInterval(4500)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_payload(self, title: str, snippet: str, *, position: QPoint) -> None:
        header = title or "Source"
        self._title_label.setText(header)
        body = snippet or "No preview available."
        self._snippet_label.setText(body)
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


class CollapsibleSection(QFrame):
    """Disclosure component with an animated body and header chevron."""

    toggled = pyqtSignal(bool)
    user_toggled = pyqtSignal(bool, bool)

    def __init__(self, title: str, *, accent: str, expanded: bool = False) -> None:
        super().__init__()
        self.setObjectName("collapsibleSection")
        self._title = title
        self._accent = accent
        self._expanded = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle_button = QToolButton(self)
        self._toggle_button.setObjectName("sectionHeaderButton")
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(False)
        self._toggle_button.setText(title)
        self._toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._toggle_button.clicked.connect(self._handle_clicked)
        outer.addWidget(self._toggle_button)

        self._content = QFrame(self)
        self._content.setObjectName("sectionBody")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 6, 12, 10)
        self._content_layout.setSpacing(6)
        outer.addWidget(self._content)

        self._animation = QPropertyAnimation(self._content, b"maximumHeight", self)
        self._animation.setDuration(180)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self._content.setMaximumHeight(0)
        self._content.setVisible(False)

        self._apply_accent()
        self.set_expanded(expanded, animate=False)

    # -----------------------------------------------------------------
    def _apply_accent(self) -> None:
        self._toggle_button.setStyleSheet(
            "QToolButton#sectionHeaderButton {"
            "  border: none;"
            "  text-align: left;"
            "  padding: 6px 4px;"
            f"  color: {self._accent};"
            "  font-weight: 600;"
            "}"
        )
        self._content.setStyleSheet(
            "QFrame#sectionBody { border: none; }"
        )

    def _handle_clicked(self, checked: bool) -> None:  # pragma: no cover - UI
        modifiers = QApplication.keyboardModifiers()
        shift_pressed = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        self._apply_expanded_state(checked, animate=True)
        self.user_toggled.emit(checked, shift_pressed)

    def _apply_expanded_state(self, expanded: bool, *, animate: bool) -> None:
        if expanded == self._expanded and self._animation.state() == QAbstractAnimation.State.Stopped:
            return
        self._expanded = expanded
        self._toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._animation.stop()
        try:
            self._animation.finished.disconnect(self._on_animation_finished)
        except TypeError:
            pass
        if animate:
            start = self._content.maximumHeight()
            if expanded:
                self._content.setVisible(True)
                target = self._content.sizeHint().height()
                self._animation.setStartValue(start)
                self._animation.setEndValue(target)
            else:
                target = 0
                if start == 0:
                    start = self._content.sizeHint().height()
                self._animation.setStartValue(start)
                self._animation.setEndValue(target)
                self._animation.finished.connect(self._on_animation_finished)
            self._animation.start()
        else:
            if expanded:
                self._content.setVisible(True)
                self._content.setMaximumHeight(self._content.sizeHint().height())
            else:
                self._content.setMaximumHeight(0)
                self._content.setVisible(False)
        self.toggled.emit(self._expanded)

    def _on_animation_finished(self) -> None:  # pragma: no cover - animation
        if not self._expanded:
            self._content.setVisible(False)

    # -----------------------------------------------------------------
    def add_text(self, text: str) -> QLabel:
        label = QLabel(text, self._content)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._content_layout.addWidget(label)
        self._refresh_height()
        return label

    def add_widget(self, widget: QWidget) -> None:
        self._content_layout.addWidget(widget)
        self._refresh_height()

    def add_spacing(self, pixels: int) -> None:
        spacer = QFrame(self._content)
        spacer.setFixedHeight(max(0, pixels))
        self._content_layout.addWidget(spacer)
        self._refresh_height()

    def _refresh_height(self) -> None:
        if self._expanded:
            self._content.setMaximumHeight(self._content.sizeHint().height())

    def set_accent(self, color: str) -> None:
        if color == self._accent:
            return
        self._accent = color
        self._apply_accent()

    def set_expanded(self, expanded: bool, *, animate: bool = False) -> None:
        if expanded == self._expanded and self._animation.state() == QAbstractAnimation.State.Stopped:
            return
        self._toggle_button.blockSignals(True)
        self._toggle_button.setChecked(expanded)
        self._toggle_button.blockSignals(False)
        self._apply_expanded_state(expanded, animate=animate)

    @property
    def expanded(self) -> bool:
        return self._expanded


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

    def add_action(self, label: str, callback) -> QPushButton:
        button = QPushButton(label, self)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(callback)
        self._actions_bar.addWidget(button)
        self._hover_actions.append(button)
        return button

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
        self._sections: dict[str, CollapsibleSection] = {}
        self._section_order: list[CollapsibleSection] = []
        self._assumption_widgets: list[QWidget] = []
        self._citation_entries: list[QToolButton] = []
        self._expand_controls: QFrame | None = None
        self._expand_all_button: QToolButton | None = None
        self._collapse_all_button: QToolButton | None = None
        self._hover_expand_button: QPushButton | None = None
        self._hover_jump_button: QPushButton | None = None
        self._plan_section: CollapsibleSection | None = None
        self._plan_has_rows = False
        self._plan_has_assumptions = False
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

        self._build_sections()

        copy_button = self.add_action("Copy message", self._copy_answer)
        copy_button.setObjectName("bubbleActionCopy")
        self._hover_expand_button = self.add_action("Expand all", self._expand_all_sections)
        self._hover_expand_button.setObjectName("bubbleActionExpand")
        self._hover_jump_button = self.add_action("Jump to Evidence", self._jump_to_evidence_panel)
        self._hover_jump_button.setObjectName("bubbleActionEvidence")
        if not self.turn.citations:
            self._hover_jump_button.setEnabled(False)

        self._settings.sources_only_mode_changed.connect(self._on_sources_only_changed)
        self._settings.show_assumptions_changed.connect(self._on_assumptions_toggle)

        self._update_expand_control_state()
        self._update_citation_entry_highlight(None)

    def _build_sections(self) -> None:
        accent = self.accent
        metadata_lines = [
            f"Model: {self.turn.model_name or '—'}",
            f"Asked: {_format_timestamp(self.turn.asked_at)}",
            f"Answered: {_format_timestamp(self.turn.answered_at)}",
            f"Latency: {self.turn.latency_ms or '—'} ms",
            _format_token_usage(self.turn.token_usage),
        ]

        plan_rows: list[str] = []
        for index, item in enumerate(self.turn.plan, start=1):
            status = (item.status or "pending").replace("_", " ")
            plan_rows.append(f"{index}. {item.description} [{status}]")
        self._plan_has_rows = bool(plan_rows)

        assumptions: list[str] = list(self.turn.assumptions)
        decision = self.turn.assumption_decision
        if decision:
            decision_parts = [f"Decision: {decision.mode.title()}"]
            if decision.rationale:
                decision_parts.append(f"Rationale: {decision.rationale}")
            if decision.clarifying_question:
                decision_parts.append(f"Follow-up: {decision.clarifying_question}")
            assumptions.append(" | ".join(decision_parts))
        self._plan_has_assumptions = bool(assumptions)

        if plan_rows or assumptions:
            section = CollapsibleSection("Plan", accent=accent, expanded=self._settings.show_plan)
            for row in plan_rows:
                section.add_text(row)
            if assumptions:
                if plan_rows:
                    section.add_spacing(4)
                header = QLabel("Assumptions", section)
                header.setObjectName("sectionSubheading")
                header.setWordWrap(True)
                section.add_widget(header)
                self._assumption_widgets.append(header)
                for assumption in assumptions:
                    widget = section.add_text(f"• {assumption}")
                    self._assumption_widgets.append(widget)
            self._register_section("Plan", section)
            self._plan_section = section
            self._update_assumptions_visibility(self._settings.show_assumptions)
            self._refresh_plan_section_visibility()

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
            section = CollapsibleSection("Critiques", accent=accent, expanded=False)
            for row in critiques:
                section.add_text(row)
            self._register_section("Critiques", section)

        logs_section = CollapsibleSection("Logs", accent=accent, expanded=False)
        for line in metadata_lines:
            logs_section.add_text(line)
        if self.turn.step_results:
            logs_section.add_spacing(4)
        for result in self.turn.step_results:
            prefix = f"{result.index}. {result.description.strip() or 'Step'}"
            summary = result.answer.strip() or "No summary recorded"
            markers = " ".join(f"[{idx}]" for idx in result.citation_indexes)
            text = f"{prefix}\n{summary} {markers}".strip()
            logs_section.add_text(text)
        self._register_section("Logs", logs_section)

        if self.turn.citations:
            section = CollapsibleSection(
                f"Citations ({len(self.turn.citations)})",
                accent=accent,
                expanded=False,
            )
            self._citation_entries.clear()
            for display_index, citation in enumerate(self.turn.citations, start=1):
                title, snippet = _describe_citation(citation)
                entry = QFrame(section)
                entry_layout = QVBoxLayout(entry)
                entry_layout.setContentsMargins(0, 0, 0, 6)
                entry_layout.setSpacing(2)
                button = QToolButton(entry)
                button.setObjectName("citationEntryButton")
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                button.setCheckable(True)
                button.setText(f"[{display_index}] {title}")
                button.clicked.connect(lambda _checked=False, idx=display_index: self._on_citation_list_clicked(idx))
                entry_layout.addWidget(button)
                if snippet:
                    snippet_label = QLabel(snippet, entry)
                    snippet_label.setObjectName("citationEntrySnippet")
                    snippet_label.setWordWrap(True)
                    snippet_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                    entry_layout.addWidget(snippet_label)
                section.add_widget(entry)
                self._citation_entries.append(button)
            self._register_section("Citations", section)

    def _ensure_expand_controls(self) -> None:
        if self._expand_controls is not None:
            return
        controls = QFrame(self)
        controls.setObjectName("sectionControls")
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(12, 0, 12, 4)
        layout.setSpacing(4)
        layout.addStretch(1)
        self._expand_all_button = QToolButton(controls)
        self._expand_all_button.setText("Expand all")
        self._expand_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._expand_all_button.clicked.connect(self._expand_all_sections)
        layout.addWidget(self._expand_all_button)
        self._collapse_all_button = QToolButton(controls)
        self._collapse_all_button.setText("Collapse all")
        self._collapse_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._collapse_all_button.clicked.connect(self._collapse_all_sections)
        layout.addWidget(self._collapse_all_button)
        self._expand_controls = controls
        self.add_widget(controls)

    def _register_section(self, key: str, section: CollapsibleSection) -> None:
        if key not in self._sections:
            self._ensure_expand_controls()
            self._sections[key] = section
            self._section_order.append(section)
            section.user_toggled.connect(
                lambda expanded, shift, sec=section: self._on_section_user_toggled(sec, expanded, shift)
            )
            section.toggled.connect(lambda _expanded, sec=section: self._update_expand_control_buttons())
            self.add_widget(section)

    def _on_section_user_toggled(
        self, section: CollapsibleSection, expanded: bool, shift: bool
    ) -> None:
        if shift and len(self._section_order) > 1:
            QTimer.singleShot(0, lambda: self._set_all_sections(expanded))
        else:
            self._update_expand_control_buttons()

    def _set_all_sections(self, expanded: bool) -> None:
        for section in self._section_order:
            section.set_expanded(expanded, animate=True)
        self._update_expand_control_buttons()

    def _expand_all_sections(self) -> None:
        self._set_all_sections(True)
        for block in self._code_blocks:
            block.expand()

    def _collapse_all_sections(self) -> None:
        self._set_all_sections(False)

    def _update_expand_control_buttons(self) -> None:
        total = len(self._section_order)
        expanded = sum(1 for section in self._section_order if section.expanded)
        if self._expand_all_button:
            self._expand_all_button.setEnabled(total > 0 and expanded < total)
        if self._collapse_all_button:
            self._collapse_all_button.setEnabled(expanded > 0)
        if self._hover_expand_button:
            self._hover_expand_button.setEnabled(total > 0 and expanded < total)

    def _update_expand_control_state(self) -> None:
        has_sections = bool(self._section_order)
        if self._expand_controls:
            self._expand_controls.setVisible(has_sections)
        if self._hover_expand_button:
            self._hover_expand_button.setEnabled(has_sections)
        self._update_expand_control_buttons()

    def _on_sources_only_changed(self, enabled: bool) -> None:
        self._sources_only = enabled
        for block in self._text_blocks:
            block.set_sources_only(enabled)

    def _copy_answer(self) -> None:  # pragma: no cover - clipboard
        QApplication.clipboard().setText(self.turn.answer or "")
        self.notify_copy("Answer copied")

    def _on_citation_activated(self, index: int) -> None:
        self._selected_citation = index
        for block in self._text_blocks:
            block.set_highlight(index)
        self._update_citation_entry_highlight(index)
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

    def _on_citation_list_clicked(self, index: int) -> None:
        self._selected_citation = index
        for block in self._text_blocks:
            block.set_highlight(index)
        self._update_citation_entry_highlight(index)
        self.citation_activated.emit(index)
        self._show_citation_preview(index)

    def _jump_to_evidence_panel(self) -> None:
        if not self.turn.citations:
            return
        target = self._selected_citation or 1
        self._selected_citation = target
        self.citation_activated.emit(target)

    def _update_citation_entry_highlight(self, index: int | None) -> None:
        for position, button in enumerate(self._citation_entries, start=1):
            button.blockSignals(True)
            button.setChecked(position == index)
            button.blockSignals(False)

    def set_selected_citation(self, index: int | None) -> None:
        self._selected_citation = index
        for block in self._text_blocks:
            block.set_highlight(index)
        self._update_citation_entry_highlight(index)
        if index is None:
            self._popover.hide()

    def set_plan_expanded(self, expanded: bool) -> None:
        section = self._sections.get("Plan")
        if section:
            section.set_expanded(expanded, animate=False)
        self._update_expand_control_buttons()

    def set_background_colors(
        self, *, background: str, code_background: str, accent: str
    ) -> None:
        self.set_background(background)
        self.set_accent(accent)
        for block in self._text_blocks:
            block.set_accent(accent)
        for code_block in self._code_blocks:
            code_block.set_background(code_background)
        for section in self._section_order:
            section.set_accent(accent)

    def _update_assumptions_visibility(self, enabled: bool) -> None:
        for widget in self._assumption_widgets:
            widget.setVisible(enabled)
        self._refresh_plan_section_visibility()

    def _on_assumptions_toggle(self, enabled: bool) -> None:
        self._update_assumptions_visibility(enabled)

    def _refresh_plan_section_visibility(self) -> None:
        if not self._plan_section:
            return
        visible_assumptions = self._plan_has_assumptions and self._settings.show_assumptions
        should_show = self._plan_has_rows or visible_assumptions
        self._plan_section.setVisible(should_show)


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
