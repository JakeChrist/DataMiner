"""Dialog for presenting crash tracebacks and recent log output."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QFont, QGuiApplication, QTextOption
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QWidget,
)


LOG_PREVIEW_LIMIT = 20_000


class LogViewerDialog(QDialog):
    """Display a formatted traceback alongside recent log file output."""

    def __init__(
        self,
        *,
        log_path: Optional[Path],
        traceback_text: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._log_path = log_path
        self._traceback_text = traceback_text.strip()
        self.setWindowTitle("DataMiner Crash Report")
        self.resize(960, 640)

        layout = QGridLayout(self)
        layout.setColumnStretch(0, 1)

        header = QLabel(
            "An unexpected error occurred. The traceback and recent log output "
            "are shown below so that the issue can be diagnosed quickly."
        )
        header.setWordWrap(True)
        header.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout.addWidget(header, 0, 0)

        traceback_label = QLabel("Traceback")
        traceback_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        traceback_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(traceback_label, 1, 0)

        traceback_view = QTextEdit(self)
        traceback_view.setReadOnly(True)
        traceback_view.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        monospaced = _monospace_font()
        traceback_view.setFont(monospaced)
        traceback_view.setPlainText(self._traceback_text or "Traceback unavailable.")
        layout.addWidget(traceback_view, 2, 0)

        log_label = QLabel("Recent log output" + (f" ({log_path})" if log_path else ""))
        log_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        log_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(log_label, 3, 0)

        log_view = QTextEdit(self)
        log_view.setReadOnly(True)
        log_view.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        log_view.setFont(monospaced)
        log_view.setPlainText(self._load_log_preview())
        layout.addWidget(log_view, 4, 0)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        button_box.rejected.connect(self.reject)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box, 5, 0, alignment=Qt.AlignmentFlag.AlignRight)

        if log_path:
            open_button = QPushButton("Open Log File", self)
            open_button.clicked.connect(self._open_log_location)
            button_box.addButton(open_button, QDialogButtonBox.ButtonRole.ActionRole)

        copy_button = QPushButton("Copy Traceback", self)
        copy_button.clicked.connect(self._copy_traceback)
        button_box.addButton(copy_button, QDialogButtonBox.ButtonRole.ActionRole)

    def _load_log_preview(self) -> str:
        if not self._log_path:
            return "Log file location unknown."

        try:
            data = self._log_path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"Unable to read log file: {exc}"

        if not data.strip():
            return "Log file is currently empty."

        if len(data) > LOG_PREVIEW_LIMIT:
            data = data[-LOG_PREVIEW_LIMIT:]
            # Ensure we start from a newline boundary when truncating.
            newline = data.find("\n")
            if newline > 0:
                data = data[newline + 1 :]
            data = "… (log truncated)\n" + data
        return data

    def _open_log_location(self) -> None:
        if not self._log_path:
            return
        if QDesktopServices is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._log_path)))

    def _copy_traceback(self) -> None:
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self._traceback_text)

try:  # pragma: no cover - optional import depending on Qt bindings
    from PyQt6.QtGui import QDesktopServices, QFontDatabase
except ImportError:  # pragma: no cover
    QDesktopServices = None
    QFontDatabase = None


def _monospace_font() -> QFont:
    if QFontDatabase is None:
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        return font
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)

