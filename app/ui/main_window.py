"""Main window for the DataMiner desktop application."""

from __future__ import annotations

import threading
from functools import partial
from importlib import metadata
from typing import Callable

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PyQt6.QtGui import QAction, QCloseEvent, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QShortcut,
    QSplitter,
    QStatusBar,
    QTextBrowser,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..services.lmstudio_client import LMStudioClient
from ..services.progress_service import ProgressService, ProgressUpdate
from ..services.settings_service import SettingsService


class ChatInput(QTextEdit):
    """Text edit that emits a signal when the user submits a message."""

    def __init__(self, on_send: Callable[[str], None], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._on_send = on_send
        self.setPlaceholderText("Type your question...")
        self.setAcceptRichText(False)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
            text = self.toPlainText().strip()
            if text:
                self._on_send(text)
                self.clear()
            event.accept()
            return
        super().keyPressEvent(event)


class ToastWidget(QFrame):
    """Transient toast notification overlay."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.ToolTip)
        self.setObjectName("toast")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._label = QLabel(self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.addWidget(self._label)
        self._animation = QPropertyAnimation(self, b"windowOpacity", self)
        self._animation.setDuration(250)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

    def show_message(self, message: str, level: str = "info", duration_ms: int = 4000) -> None:
        palette = self.palette()
        if level == "error":
            palette.setColor(palette.ColorRole.Window, Qt.GlobalColor.darkRed)
            palette.setColor(palette.ColorRole.WindowText, Qt.GlobalColor.white)
        elif level == "warning":
            palette.setColor(palette.ColorRole.Window, Qt.GlobalColor.darkYellow)
            palette.setColor(palette.ColorRole.WindowText, Qt.GlobalColor.black)
        else:
            palette.setColor(palette.ColorRole.Window, Qt.GlobalColor.black)
            palette.setColor(palette.ColorRole.WindowText, Qt.GlobalColor.white)
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self._label.setText(message)
        self.adjustSize()
        parent = self.parentWidget()
        if parent:
            margin = 24
            geo = parent.geometry()
            self.move(geo.right() - self.width() - margin, geo.top() + margin)
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self._animation.stop()
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.start()
        QTimer.singleShot(duration_ms, self._fade_out)

    def _fade_out(self) -> None:
        self._animation.stop()
        self._animation.setStartValue(1.0)
        self._animation.setEndValue(0.0)
        try:
            self._animation.finished.disconnect(self.hide)
        except TypeError:
            pass
        self._animation.finished.connect(self.hide)
        self._animation.start()


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(
        self,
        *,
        settings_service: SettingsService,
        progress_service: ProgressService,
        lmstudio_client: LMStudioClient,
        enable_health_monitor: bool = True,
    ) -> None:
        super().__init__()
        self.settings_service = settings_service
        self.progress_service = progress_service
        self.lmstudio_client = lmstudio_client
        self.enable_health_monitor = enable_health_monitor
        self._setup_window()
        self._create_actions()
        self._create_menus_and_toolbar()
        self._create_status_bar()
        self._create_layout()
        self._connect_services()
        self._toast = ToastWidget(self)
        self.settings_service.apply_theme()
        self.settings_service.apply_font_scale()
        self._update_lmstudio_status()
        if self.enable_health_monitor:
            self._health_timer = QTimer(self)
            self._health_timer.timeout.connect(self._update_lmstudio_status)
            self._health_timer.start(15000)

    # ------------------------------------------------------------------
    def _setup_window(self) -> None:
        self.setWindowTitle("DataMiner")
        self.resize(1280, 800)

    def _create_actions(self) -> None:
        self.settings_action = QAction("Settings", self)
        self.settings_action.triggered.connect(self._open_settings)

        self.help_action = QAction("Help", self)
        self.help_action.triggered.connect(self._open_help)

        self.toggle_theme_action = QAction("Toggle Theme", self)
        self.toggle_theme_action.triggered.connect(self.settings_service.toggle_theme)

        self.send_button = QPushButton("Send", self)
        self.send_button.clicked.connect(self._handle_send)

    def _create_menus_and_toolbar(self) -> None:
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        file_menu = QMenu("File", self)
        file_menu.addAction(self.settings_action)
        menubar.addMenu(file_menu)

        help_menu = QMenu("Help", self)
        help_menu.addAction(self.help_action)
        menubar.addMenu(help_menu)

        view_menu = QMenu("View", self)
        view_menu.addAction(self.toggle_theme_action)
        menubar.addMenu(view_menu)

        toolbar = QToolBar("Main Toolbar", self)
        toolbar.addAction(self.settings_action)
        toolbar.addAction(self.help_action)
        toolbar.addAction(self.toggle_theme_action)
        self.addToolBar(toolbar)

    def _create_status_bar(self) -> None:
        status = QStatusBar(self)
        self.setStatusBar(status)

        project_label = QLabel(self._project_label_text())
        status.addWidget(project_label)

        self._lmstudio_indicator = QLabel("LMStudio: Checking...")
        status.addPermanentWidget(self._lmstudio_indicator)

        self._progress_bar = QProgressBar(self)
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximumWidth(200)
        status.addPermanentWidget(self._progress_bar)

    def _create_layout(self) -> None:
        central = QWidget(self)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter)

        # Left panel - corpus selector
        self._corpus_list = QListWidget(splitter)
        self._corpus_list.setObjectName("corpusSelector")
        self._corpus_list.addItems(["All Documents", "Recent", "Favorites"])
        splitter.addWidget(self._corpus_list)

        # Center panel - chat conversation
        chat_panel = QWidget(splitter)
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(8, 8, 8, 8)
        chat_layout.setSpacing(8)
        self._chat_log = QTextBrowser(chat_panel)
        self._chat_log.setObjectName("chatArea")
        self._chat_log.setOpenLinks(False)
        self._chat_log.setPlaceholderText("Conversation will appear here.")
        chat_layout.addWidget(self._chat_log, 1)

        input_container = QFrame(chat_panel)
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(6)
        self._chat_input = ChatInput(self._handle_send, input_container)
        input_layout.addWidget(self._chat_input, 1)
        input_layout.addWidget(self.send_button)
        chat_layout.addWidget(input_container)
        splitter.addWidget(chat_panel)

        # Right panel - evidence viewer
        self._evidence_panel = QTextBrowser(splitter)
        self._evidence_panel.setObjectName("evidencePanel")
        self._evidence_panel.setPlaceholderText("Evidence and citations will appear here.")
        splitter.addWidget(self._evidence_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([240, 640, 320])

        self.setCentralWidget(central)

        # Keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self._focus_corpus)
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self._copy_chat_text)

    # ------------------------------------------------------------------
    def _connect_services(self) -> None:
        self.settings_service.theme_changed.connect(self._apply_theme)
        self.settings_service.font_scale_changed.connect(self._apply_font_scale)
        self.progress_service.progress_started.connect(self._on_progress_started)
        self.progress_service.progress_updated.connect(self._on_progress_updated)
        self.progress_service.progress_finished.connect(self._on_progress_finished)
        self.progress_service.toast_requested.connect(self._show_toast)

    # ------------------------------------------------------------------
    def _handle_send(self, text: str | None = None) -> None:
        text = text or self._chat_input.toPlainText().strip()
        if not text:
            return
        self._chat_log.append(f"<b>You:</b> {text}")
        self._chat_input.clear()
        self.progress_service.start("chat-send", "Sending message...")
        self.progress_service.notify("Message queued", level="info")
        QTimer.singleShot(400, lambda: self.progress_service.finish("chat-send", "Message sent"))

    def _focus_corpus(self) -> None:
        self._corpus_list.setFocus()

    def _copy_chat_text(self) -> None:
        selected = self._chat_log.textCursor().selectedText()
        if not selected:
            selected = self._chat_log.toPlainText()
        QApplication.clipboard().setText(selected)
        self.progress_service.notify("Chat copied to clipboard", level="info", duration_ms=2000)

    # ------------------------------------------------------------------
    def _open_settings(self) -> None:
        QMessageBox.information(self, "Settings", "Settings dialog coming soon.")

    def _open_help(self) -> None:
        QMessageBox.information(self, "Help", "Visit the documentation for assistance.")

    # ------------------------------------------------------------------
    def _apply_theme(self, theme: str) -> None:
        self.settings_service.apply_theme()
        self._show_toast(f"Theme set to {theme.title()}.", level="info", duration_ms=1500)

    def _apply_font_scale(self, _scale: float) -> None:
        self.settings_service.apply_font_scale()

    # ------------------------------------------------------------------
    def _on_progress_started(self, update: ProgressUpdate) -> None:
        self._progress_bar.setFormat(update.message or "Working...")
        self._progress_bar.setRange(0, 0 if update.indeterminate else 100)
        self._progress_bar.setValue(0 if update.percent is None else int(update.percent))
        self._progress_bar.setVisible(True)

    def _on_progress_updated(self, update: ProgressUpdate) -> None:
        if update.message:
            self._progress_bar.setFormat(update.message)
        if update.indeterminate:
            self._progress_bar.setRange(0, 0)
        elif update.percent is not None:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(int(max(0, min(100, update.percent))))

    def _on_progress_finished(self, update: ProgressUpdate) -> None:
        if update.message:
            self.statusBar().showMessage(update.message, 3000)
        self._progress_bar.setVisible(False)

    def _show_toast(self, message: str, level: str, duration_ms: int) -> None:
        self._toast.show_message(message, level=level, duration_ms=duration_ms)

    # ------------------------------------------------------------------
    def _project_label_text(self) -> str:
        try:
            version = metadata.version("DataMiner")
        except metadata.PackageNotFoundError:
            version = "0.0.0"
        return f"DataMiner v{version}"

    def _update_lmstudio_status(self) -> None:
        if not self.enable_health_monitor:
            self._lmstudio_indicator.setText("LMStudio: Disabled")
            return

        def worker() -> None:
            healthy = self.lmstudio_client.health_check()
            QTimer.singleShot(0, partial(self._set_lmstudio_status, healthy))

        threading.Thread(target=worker, daemon=True).start()

    def _set_lmstudio_status(self, healthy: bool) -> None:
        text = "LMStudio: Connected" if healthy else "LMStudio: Offline"
        self._lmstudio_indicator.setText(text)

    # ------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        if hasattr(self, "_health_timer"):
            self._health_timer.stop()
        return super().closeEvent(event)


__all__ = ["MainWindow"]

