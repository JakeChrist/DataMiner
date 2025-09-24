"""Main window for the DataMiner desktop application."""

from __future__ import annotations

from datetime import datetime
import threading
from functools import partial
from importlib import metadata

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PyQt6.QtGui import QAction, QCloseEvent, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QShortcut,
    QSplitter,
    QStatusBar,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..services.conversation_manager import (
    ConnectionState,
    ConversationManager,
    ConversationTurn,
    LMStudioError,
    ReasoningVerbosity,
    ResponseMode,
)
from ..services.conversation_settings import ConversationSettings
from ..services.lmstudio_client import LMStudioClient
from ..services.progress_service import ProgressService, ProgressUpdate
from ..services.settings_service import SettingsService
from .answer_view import AnswerView
from .question_input_widget import QuestionInputWidget


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
        self.conversation_settings = ConversationSettings()
        self._conversation_manager = ConversationManager(self.lmstudio_client)
        self._turns: list[ConversationTurn] = []
        self._connection_unsubscribe = self._conversation_manager.add_connection_listener(
            self._on_connection_state_changed
        )
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

        # Center panel - conversation and controls
        chat_panel = QWidget(splitter)
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(8, 8, 8, 8)
        chat_layout.setSpacing(8)

        self.answer_view = AnswerView(
            settings=self.conversation_settings,
            progress_service=self.progress_service,
            parent=chat_panel,
        )
        self.answer_view.setObjectName("answerView")
        chat_layout.addWidget(self.answer_view, 1)

        self._controls_frame = self._create_conversation_controls(chat_panel)
        chat_layout.addWidget(self._controls_frame)

        self.question_input = QuestionInputWidget(chat_panel)
        self.question_input.ask_requested.connect(self._handle_ask)
        chat_layout.addWidget(self.question_input)
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
        self._update_question_prerequisites(self._conversation_manager.connection_state)

    def _create_conversation_controls(self, parent: QWidget) -> QFrame:
        frame = QFrame(parent)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        reasoning_label = QLabel("Reasoning:", frame)
        layout.addWidget(reasoning_label)

        self._verbosity_combo = QComboBox(frame)
        for option in ReasoningVerbosity:
            self._verbosity_combo.addItem(option.name.title(), userData=option)
        index = self._verbosity_combo.findData(self.conversation_settings.reasoning_verbosity)
        if index >= 0:
            self._verbosity_combo.setCurrentIndex(index)
        self._verbosity_combo.currentIndexChanged.connect(self._on_reasoning_verbosity_changed)
        layout.addWidget(self._verbosity_combo)

        self._plan_checkbox = QCheckBox("Show plan", frame)
        self._plan_checkbox.setObjectName("togglePlan")
        self._plan_checkbox.setChecked(self.conversation_settings.show_plan)
        self._plan_checkbox.toggled.connect(self.conversation_settings.set_show_plan)
        layout.addWidget(self._plan_checkbox)

        self._assumptions_checkbox = QCheckBox("Show assumptions", frame)
        self._assumptions_checkbox.setObjectName("toggleAssumptions")
        self._assumptions_checkbox.setChecked(self.conversation_settings.show_assumptions)
        self._assumptions_checkbox.toggled.connect(self.conversation_settings.set_show_assumptions)
        layout.addWidget(self._assumptions_checkbox)

        layout.addStretch(1)

        self._sources_only_checkbox = QCheckBox("Sources only", frame)
        self._sources_only_checkbox.setObjectName("toggleSourcesOnly")
        self._sources_only_checkbox.setChecked(self.conversation_settings.sources_only_mode)
        self._sources_only_checkbox.toggled.connect(self._on_sources_only_toggled)
        layout.addWidget(self._sources_only_checkbox)

        self.conversation_settings.reasoning_verbosity_changed.connect(self._sync_reasoning_combo)
        self.conversation_settings.show_plan_changed.connect(self._sync_plan_checkbox)
        self.conversation_settings.show_assumptions_changed.connect(
            self._sync_assumptions_checkbox
        )
        self.conversation_settings.sources_only_mode_changed.connect(
            self._sync_sources_checkbox
        )

        return frame

    # ------------------------------------------------------------------
    def _connect_services(self) -> None:
        self.settings_service.theme_changed.connect(self._apply_theme)
        self.settings_service.font_scale_changed.connect(self._apply_font_scale)
        self.progress_service.progress_started.connect(self._on_progress_started)
        self.progress_service.progress_updated.connect(self._on_progress_updated)
        self.progress_service.progress_finished.connect(self._on_progress_finished)
        self.progress_service.toast_requested.connect(self._show_toast)

    def _on_reasoning_verbosity_changed(self, index: int) -> None:
        data = self._verbosity_combo.itemData(index)
        if isinstance(data, ReasoningVerbosity):
            self.conversation_settings.set_reasoning_verbosity(data)

    def _sync_reasoning_combo(self, verbosity: ReasoningVerbosity) -> None:
        index = self._verbosity_combo.findData(verbosity)
        if index >= 0 and index != self._verbosity_combo.currentIndex():
            self._verbosity_combo.blockSignals(True)
            self._verbosity_combo.setCurrentIndex(index)
            self._verbosity_combo.blockSignals(False)

    def _sync_plan_checkbox(self, enabled: bool) -> None:
        if self._plan_checkbox.isChecked() != enabled:
            self._plan_checkbox.blockSignals(True)
            self._plan_checkbox.setChecked(enabled)
            self._plan_checkbox.blockSignals(False)

    def _sync_assumptions_checkbox(self, enabled: bool) -> None:
        if self._assumptions_checkbox.isChecked() != enabled:
            self._assumptions_checkbox.blockSignals(True)
            self._assumptions_checkbox.setChecked(enabled)
            self._assumptions_checkbox.blockSignals(False)

    def _on_sources_only_toggled(self, enabled: bool) -> None:
        self.conversation_settings.set_sources_only_mode(enabled)

    def _sync_sources_checkbox(self, enabled: bool) -> None:
        if self._sources_only_checkbox.isChecked() != enabled:
            self._sources_only_checkbox.blockSignals(True)
            self._sources_only_checkbox.setChecked(enabled)
            self._sources_only_checkbox.blockSignals(False)

    # ------------------------------------------------------------------
    def _handle_ask(self, text: str) -> None:
        if not text.strip():
            return

        state = self._conversation_manager.connection_state
        if not state.connected:
            self._update_question_prerequisites(state)
            self.progress_service.notify(
                state.message or "LMStudio is unavailable.", level="error", duration_ms=4000
            )
            return

        self.question_input.set_busy(True)
        self.progress_service.start("chat-send", "Submitting question...")
        asked_at = datetime.now()
        try:
            turn = self._conversation_manager.ask(
                text,
                reasoning_verbosity=self.conversation_settings.reasoning_verbosity,
                response_mode=self.conversation_settings.response_mode,
            )
        except LMStudioError as exc:
            self.progress_service.finish("chat-send", "Send failed")
            self.progress_service.notify(str(exc) or "Failed to contact LMStudio", level="error")
            self.question_input.set_busy(False)
            self._update_question_prerequisites(self._conversation_manager.connection_state)
            return

        answered_at = datetime.now()
        turn.asked_at = asked_at
        turn.answered_at = answered_at
        turn.latency_ms = int((answered_at - asked_at).total_seconds() * 1000)
        turn.token_usage = self._extract_token_usage(turn)
        self._turns.append(turn)
        self.answer_view.add_turn(turn)
        self._update_evidence_panel(turn)
        self.progress_service.finish("chat-send", "Answer received")
        self.question_input.set_busy(False)
        self._update_question_prerequisites(self._conversation_manager.connection_state)

    def _focus_corpus(self) -> None:
        self._corpus_list.setFocus()

    def _copy_chat_text(self) -> None:
        text = self.answer_view.to_plain_text()
        QApplication.clipboard().setText(text)
        self.progress_service.notify("Conversation copied", level="info", duration_ms=2000)

    def _extract_token_usage(self, turn: ConversationTurn) -> dict[str, int] | None:
        raw = turn.raw_response if isinstance(turn.raw_response, dict) else None
        usage = raw.get("usage") if raw else None
        if not isinstance(usage, dict):
            return None
        parsed: dict[str, int] = {}
        for key, value in usage.items():
            try:
                parsed[key] = int(value)
            except (TypeError, ValueError):
                continue
        return parsed or None

    def _update_evidence_panel(self, turn: ConversationTurn) -> None:
        if not turn.citations:
            self._evidence_panel.setPlainText("No citations provided.")
            return
        lines = []
        for index, citation in enumerate(turn.citations, start=1):
            lines.append(f"{index}. {citation}")
        self._evidence_panel.setPlainText("\n".join(lines))

    def _update_question_prerequisites(self, state: ConnectionState) -> None:
        message = state.message if not state.connected else None
        self.question_input.set_prerequisites_met(state.connected, message)

    def _on_connection_state_changed(self, state: ConnectionState) -> None:
        text = "LMStudio: Connected" if state.connected else "LMStudio: Offline"
        self._lmstudio_indicator.setText(text)
        if state.message:
            self.statusBar().showMessage(state.message, 4000)
        self._update_question_prerequisites(state)

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
        unsubscribe = getattr(self, "_connection_unsubscribe", None)
        if callable(unsubscribe):
            try:
                unsubscribe()
            except Exception:
                pass
        return super().closeEvent(event)


__all__ = ["MainWindow"]

