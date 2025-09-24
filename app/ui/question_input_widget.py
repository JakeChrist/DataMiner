"""Reusable widget housing the chat question input controls."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout


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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("questionInput")
        self._history: list[str] = []
        self._history_index = 0
        self._prerequisites_met = True
        self._status_message: str | None = None
        self._busy = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.editor = _HistoryTextEdit(self)
        self.editor.setAcceptRichText(False)
        self.editor.setPlaceholderText("Ask a question about your data (Ctrl+Enter to send)")
        self.editor.textChanged.connect(self._update_button_state)
        self.editor.submit_requested.connect(self._trigger_ask)
        self.editor.history_previous_requested.connect(self._recall_previous)
        self.editor.history_next_requested.connect(self._recall_next)
        layout.addWidget(self.editor, 1)

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(6)
        layout.addLayout(buttons_row)

        self.status_label = QLabel("", self)
        font = self.status_label.font()
        font.setPointSizeF(font.pointSizeF() - 1)
        self.status_label.setFont(font)
        self.status_label.setStyleSheet("color: palette(dark)")
        self.status_label.setVisible(False)
        buttons_row.addWidget(self.status_label, 1)

        self.ask_button = QPushButton("Ask", self)
        self.ask_button.setDefault(True)
        self.ask_button.clicked.connect(self._trigger_ask)
        buttons_row.addWidget(self.ask_button)

        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear)
        buttons_row.addWidget(self.clear_button)

        self._update_button_state()

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

