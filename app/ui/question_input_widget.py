"""Reusable widget housing the chat question input controls."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
)

from ..services.conversation_manager import AnswerLength, ConnectionState


class _HistoryTextEdit(QTextEdit):
    """Text edit that surfaces history navigation shortcuts."""

    submit_requested = pyqtSignal()
    history_previous_requested = pyqtSignal()
    history_next_requested = pyqtSignal()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        modifiers = event.modifiers()
        if key in {Qt.Key.Key_Return, Qt.Key.Key_Enter} and (
            modifiers & Qt.KeyboardModifier.ControlModifier
        ):
            event.accept()
            self.submit_requested.emit()
            return
        if key == Qt.Key.Key_Up and modifiers == Qt.KeyboardModifier.NoModifier:
            if self.textCursor().atStart():
                event.accept()
                self.history_previous_requested.emit()
                return
        if key == Qt.Key.Key_Down and modifiers == Qt.KeyboardModifier.NoModifier:
            if self.textCursor().atEnd():
                event.accept()
                self.history_next_requested.emit()
                return
        super().keyPressEvent(event)


class QuestionInputWidget(QFrame):
    """Composite widget handling question entry, history, and controls."""

    ask_requested = pyqtSignal(str)
    cleared = pyqtSignal()
    scope_cleared = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("questionInput")
        self._history: list[str] = []
        self._history_index = 0
        self._prerequisites_met = True
        self._status_message: str | None = None
        self._busy = False
        self._settings_menu: QMenu | None = None
        self._model_name = "lmstudio"
        self._answer_length = AnswerLength.NORMAL

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.editor = _HistoryTextEdit(self)
        self.editor.setAcceptRichText(False)
        self.editor.setPlaceholderText("Ask a question about your data (Ctrl+Enter to send)")
        self.editor.textChanged.connect(self._update_button_state)
        self.editor.submit_requested.connect(self._trigger_ask)
        self.editor.history_previous_requested.connect(self._recall_previous)
        self.editor.history_next_requested.connect(self._recall_next)
        layout.addWidget(self.editor, 1)

        self._top_row = QHBoxLayout()
        self._top_row.setSpacing(6)
        layout.addLayout(self._top_row)

        self.scope_button = QToolButton(self)
        self.scope_button.setObjectName("scopeChip")
        self.scope_button.setVisible(False)
        self.scope_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.scope_button.setToolTip("Scope: entire corpus")
        self.scope_button.clicked.connect(lambda: self.scope_cleared.emit())
        self._top_row.addWidget(self.scope_button)

        self.status_label = QLabel("", self)
        font = self.status_label.font()
        font.setPointSizeF(font.pointSizeF() - 1)
        self.status_label.setFont(font)
        self.status_label.setVisible(False)
        self._top_row.addWidget(self.status_label, 1)

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(8)
        layout.addLayout(buttons_row)
        self._buttons_row = buttons_row

        self.status_pill = QLabel("", self)
        self.status_pill.setObjectName("statusPill")
        self.status_pill.setVisible(False)
        buttons_row.addWidget(self.status_pill)
        buttons_row.addStretch(1)

        self.settings_button = QToolButton(self)
        self.settings_button.setObjectName("settingsChip")
        self.settings_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.settings_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.settings_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_button.setEnabled(False)
        buttons_row.addWidget(self.settings_button)

        self.ask_button = QPushButton("Ask", self)
        self.ask_button.setDefault(True)
        self.ask_button.clicked.connect(self._trigger_ask)
        buttons_row.addWidget(self.ask_button)

        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear)
        buttons_row.addWidget(self.clear_button)

        self._update_button_state()
        self._update_settings_summary()

    # ------------------------------------------------------------------
    def text(self) -> str:
        return self.editor.toPlainText().strip()

    def set_text(self, text: str) -> None:
        self.editor.setPlainText(text)
        cursor = self.editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.editor.setTextCursor(cursor)
        self._update_button_state()

    def clear(self) -> None:  # type: ignore[override]
        self.editor.clear()
        self._history_index = len(self._history)
        self._update_button_state()
        self.cleared.emit()

    # ------------------------------------------------------------------
    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.editor.setReadOnly(self._busy)
        self.editor.setEnabled(not self._busy)
        self.clear_button.setEnabled(not self._busy)
        self._update_button_state()

    def set_prerequisites_met(self, ok: bool, message: str | None = None) -> None:
        self._prerequisites_met = bool(ok)
        self._status_message = message
        if self._prerequisites_met or not message:
            self.status_label.setVisible(False)
        else:
            display = message.strip()
            if display:
                metrics = QFontMetrics(self.status_label.font())
                elided = metrics.elidedText(display, Qt.TextElideMode.ElideMiddle, 320)
                self.status_label.setText(elided)
                self.status_label.setVisible(True)
        self._update_button_state()

    def set_settings_menu(self, menu: QMenu | None) -> None:
        self._settings_menu = menu
        self.settings_button.setMenu(menu)
        self.settings_button.setEnabled(menu is not None)

    def set_model_name(self, name: str) -> None:
        cleaned = str(name).strip()
        if not cleaned:
            return
        if cleaned == self._model_name:
            return
        self._model_name = cleaned
        self._update_settings_summary()

    def set_answer_length(self, preset: AnswerLength) -> None:
        if not isinstance(preset, AnswerLength):
            return
        if preset is self._answer_length:
            return
        self._answer_length = preset
        self._update_settings_summary()

    def set_density(self, density: str) -> None:
        mode = str(density).lower()
        spacing = 4 if mode == "compact" else 8
        margins = 0 if mode == "compact" else 4
        layout = self.layout()
        if isinstance(layout, QVBoxLayout):
            layout.setSpacing(spacing)
            layout.setContentsMargins(margins, margins, margins, margins)
        self._top_row.setSpacing(spacing)
        self._buttons_row.setSpacing(spacing + 2)

    def update_scope_chip(self, include_count: int, exclude_count: int) -> None:
        if include_count <= 0 and exclude_count <= 0:
            self.scope_button.setVisible(False)
            self.scope_button.setToolTip("Scope: entire corpus")
            return
        parts: list[str] = []
        if include_count > 0:
            parts.append(f"+{include_count}")
        if exclude_count > 0:
            parts.append(f"-{exclude_count}")
        label = "Scope " + " · ".join(parts)
        self.scope_button.setText(label)
        self.scope_button.setToolTip("Clear scope filters")
        self.scope_button.setVisible(True)

    def set_connection_state(self, state: ConnectionState) -> None:
        if state.connected:
            tooltip = state.message or "Connected to LMStudio"
            self._set_status_pill("LMStudio Connected", "connected", tooltip)
        else:
            severity = "error" if state.message else "warning"
            label = state.message or "LMStudio Offline"
            tooltip = state.message or "LMStudio is not responding."
            self._set_status_pill(label, severity, tooltip)

    def set_status_message(self, text: str, level: str = "info") -> None:
        normalized = str(level).lower()
        state = "info"
        if normalized in {"warning", "warn"}:
            state = "warning"
        elif normalized in {"error", "danger"}:
            state = "error"
        elif normalized in {"success", "ok"}:
            state = "connected"
        self._set_status_pill(text, state)

    def _set_status_pill(self, text: str, state: str, tooltip: str | None = None) -> None:
        display = text.strip() if isinstance(text, str) else ""
        self.status_pill.setText(display)
        self.status_pill.setVisible(bool(display))
        self.status_pill.setToolTip(tooltip or "")
        self.status_pill.setProperty("state", state or "info")
        self.status_pill.style().unpolish(self.status_pill)
        self.status_pill.style().polish(self.status_pill)

    def _update_settings_summary(self) -> None:
        length_label = self._answer_length.value.title()
        summary = f"{self._model_name} · {length_label}"
        self.settings_button.setText(summary)

    # ------------------------------------------------------------------
    def _trigger_ask(self) -> None:
        if self._busy:
            return
        text = self.text()
        if not text:
            return
        if not self._prerequisites_met:
            return
        self._remember_entry(text)
        self.editor.clear()
        self.ask_requested.emit(text)

    def _remember_entry(self, text: str) -> None:
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._history_index = len(self._history)

    def _recall_previous(self) -> None:
        if not self._history:
            return
        self._history_index = max(0, self._history_index - 1)
        self._apply_history()

    def _recall_next(self) -> None:
        if not self._history:
            return
        self._history_index = min(len(self._history), self._history_index + 1)
        self._apply_history()

    def _apply_history(self) -> None:
        if 0 <= self._history_index < len(self._history):
            self.set_text(self._history[self._history_index])
        else:
            self.editor.clear()

    def _update_button_state(self) -> None:
        has_text = bool(self.text())
        enabled = has_text and self._prerequisites_met and not self._busy
        self.ask_button.setEnabled(enabled)
        if not self._prerequisites_met and self._status_message:
            self.ask_button.setToolTip(self._status_message)
        else:
            self.ask_button.setToolTip("")


__all__ = ["QuestionInputWidget"]

