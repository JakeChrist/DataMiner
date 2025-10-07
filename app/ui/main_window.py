"""Main window for the DataMiner desktop application."""

from __future__ import annotations

import logging
import queue
from datetime import datetime
from functools import partial
from importlib import metadata
from pathlib import Path
import threading
from typing import Any, Callable, Iterable

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer, QUrl
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QDesktopServices,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..ingest.parsers import SUPPORTED_PATTERNS
from ..ingest.service import IngestService, TaskStatus
from ..retrieval import SearchService
from ..services.conversation_manager import (
    ConnectionState,
    ConversationManager,
    ConversationTurn,
    LMStudioError,
    PlanItem,
    ReasoningVerbosity,
    ResponseMode,
    StepContextBatch,
)
from ..services.conversation_settings import ConversationSettings
from ..services.document_hierarchy import DocumentHierarchyService
from ..services.lmstudio_client import AnswerLength, LMStudioClient
from ..services.progress_service import ProgressService, ProgressUpdate
from ..services.project_service import ProjectRecord, ProjectService
from ..services.backup_service import BackupService
from ..services.export_service import ExportService
from ..services.settings_service import SettingsService, ChatStyleSettings
from ..storage import DatabaseError
from ..logging import get_log_file_path
from .answer_view import AnswerView, ChatTurnWidget
from .evidence_panel import EvidencePanel
from .question_input_widget import QuestionInputWidget
from .log_viewer_dialog import LogViewerDialog
from .settings_dialog import SettingsDialog


LOGGER = logging.getLogger(__name__)


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
        project_service: ProjectService,
        ingest_service: IngestService,
        document_hierarchy: DocumentHierarchyService,
        export_service: ExportService,
        backup_service: BackupService,
        enable_health_monitor: bool = True,
    ) -> None:
        super().__init__()
        self.settings_service = settings_service
        self.progress_service = progress_service
        self.lmstudio_client = lmstudio_client
        self.project_service = project_service
        self.ingest_service = ingest_service
        self.document_hierarchy = document_hierarchy
        self.export_service = export_service
        self.backup_service = backup_service
        self.enable_health_monitor = enable_health_monitor
        self._last_splitter_sizes = list(self.settings_service.splitter_sizes)
        self._setup_window()
        self._create_actions()
        self._create_menus_and_toolbar()
        self._create_status_bar()
        self.conversation_settings = ConversationSettings()
        self._conversation_manager = ConversationManager(self.lmstudio_client)
        self.search_service = SearchService(
            project_service.ingest,
            project_service.documents,
            project_service.chats,
        )
        self._turns: list[ConversationTurn] = []
        self._project_sessions: dict[int, dict[str, Any]] = {}
        self._ingest_jobs: dict[int, dict[str, Any]] = {}
        self._ingest_updates: "queue.Queue[tuple[int, dict[str, Any]]]" = queue.Queue()
        self._connection_unsubscribe = self._conversation_manager.add_connection_listener(
            self._on_connection_state_changed
        )
        self._current_retrieval_scope: dict[str, list[str]] = {"include": [], "exclude": []}
        self._last_question: str | None = None
        self._active_card: ChatTurnWidget | None = None
        self._has_documents: bool = False
        self._toast = ToastWidget(self)
        self._rescan_button: QPushButton | None = None
        self._create_layout()
        self._apply_density(self.settings_service.density)
        self.question_input.set_model_name(self.conversation_settings.model_name)
        self.question_input.set_answer_length(self.conversation_settings.answer_length)
        self._configure_question_settings_menu()
        self._connect_services()
        self.settings_service.apply_theme()
        self.settings_service.apply_font_scale()
        self._update_lmstudio_status()
        self.project_service.projects_changed.connect(self._on_projects_changed)
        self.project_service.active_project_changed.connect(self._on_active_project_changed)
        self._initialise_project_state()
        self._ingest_unsubscribe = self.ingest_service.subscribe(self._enqueue_ingest_update)
        self._ingest_timer = QTimer(self)
        self._ingest_timer.setInterval(150)
        self._ingest_timer.timeout.connect(self._drain_ingest_updates)
        self._ingest_timer.start()
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
        self.toggle_theme_action.triggered.connect(self._on_toggle_theme_triggered)

        self.view_logs_action = QAction("View Logs…", self)
        self.view_logs_action.triggered.connect(self._open_log_viewer)

        self.new_project_action = QAction("New Project…", self)
        self.new_project_action.triggered.connect(self._prompt_new_project)

        self.rename_project_action = QAction("Rename Project…", self)
        self.rename_project_action.triggered.connect(self._prompt_rename_project)

        self.delete_project_action = QAction("Delete Project", self)
        self.delete_project_action.triggered.connect(self._delete_current_project)

        self.reveal_storage_action = QAction("Reveal Project Storage", self)
        self.reveal_storage_action.triggered.connect(self._reveal_project_storage)

        self.remove_project_data_action = QAction("Remove Project Data", self)
        self.remove_project_data_action.triggered.connect(self._purge_project_data)

        self.backup_action = QAction("Create Backup…", self)
        self.backup_action.triggered.connect(self._create_backup)

        self.restore_action = QAction("Restore from Backup…", self)
        self.restore_action.triggered.connect(self._restore_backup)

        self.export_markdown_action = QAction("Export Conversation to Markdown…", self)
        self.export_markdown_action.triggered.connect(self._export_conversation_markdown)

        self.export_html_action = QAction("Export Conversation to HTML…", self)
        self.export_html_action.triggered.connect(self._export_conversation_html)

        self.export_snippet_action = QAction("Export Selected Snippet…", self)
        self.export_snippet_action.triggered.connect(self._export_selected_snippet)

        self.export_markdown_action.setEnabled(False)
        self.export_html_action.setEnabled(False)
        self.export_snippet_action.setEnabled(False)

        self.add_folder_action = QAction("Add Folder to Corpus…", self)
        self.add_folder_action.triggered.connect(self._add_folder_to_corpus)

        self.add_files_action = QAction("Add Files to Corpus…", self)
        self.add_files_action.triggered.connect(self._add_files_to_corpus)

        self.rescan_corpus_action = QAction("Rescan Indexed Folders", self)
        self.rescan_corpus_action.triggered.connect(self._rescan_corpus)
        self.rescan_corpus_action.setEnabled(False)

    def _create_menus_and_toolbar(self) -> None:
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        file_menu = QMenu("File", self)
        file_menu.addAction(self.settings_action)
        file_menu.addSeparator()
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.rename_project_action)
        file_menu.addAction(self.delete_project_action)
        file_menu.addAction(self.remove_project_data_action)
        file_menu.addSeparator()
        corpus_menu = file_menu.addMenu("Corpus")
        corpus_menu.addAction(self.add_folder_action)
        corpus_menu.addAction(self.add_files_action)
        corpus_menu.addSeparator()
        corpus_menu.addAction(self.rescan_corpus_action)
        file_menu.addSeparator()
        file_menu.addAction(self.reveal_storage_action)
        file_menu.addSeparator()
        export_menu = file_menu.addMenu("Export")
        export_menu.addAction(self.export_markdown_action)
        export_menu.addAction(self.export_html_action)
        export_menu.addAction(self.export_snippet_action)
        file_menu.addSeparator()
        file_menu.addAction(self.backup_action)
        file_menu.addAction(self.restore_action)
        menubar.addMenu(file_menu)

        help_menu = QMenu("Help", self)
        help_menu.addAction(self.help_action)
        menubar.addMenu(help_menu)

        view_menu = QMenu("View", self)
        view_menu.addAction(self.view_logs_action)
        view_menu.addSeparator()
        view_menu.addAction(self.toggle_theme_action)
        view_menu.addSeparator()

        self.toggle_corpus_panel_action = QAction("Show corpus pane", self, checkable=True)
        self.toggle_corpus_panel_action.setChecked(self.settings_service.show_corpus_panel)
        self.toggle_corpus_panel_action.toggled.connect(self._toggle_corpus_panel)
        view_menu.addAction(self.toggle_corpus_panel_action)

        self.toggle_evidence_panel_action = QAction(
            "Show evidence pane", self, checkable=True
        )
        self.toggle_evidence_panel_action.setChecked(
            self.settings_service.show_evidence_panel
        )
        self.toggle_evidence_panel_action.toggled.connect(self._toggle_evidence_panel)
        view_menu.addAction(self.toggle_evidence_panel_action)

        view_menu.addSeparator()
        density_menu = view_menu.addMenu("Density")
        self._density_group = QActionGroup(self)
        for label, value in [("Comfortable", "comfortable"), ("Compact", "compact")]:
            action = QAction(label, self, checkable=True)
            action.setData(value)
            if self.settings_service.density == value:
                action.setChecked(True)
            self._density_group.addAction(action)
            density_menu.addAction(action)
        self._density_group.triggered.connect(self._on_density_action_triggered)
        menubar.addMenu(view_menu)

        toolbar = QToolBar("Main Toolbar", self)
        toolbar.addAction(self.settings_action)
        toolbar.addAction(self.help_action)
        toolbar.addAction(self.toggle_theme_action)
        toolbar.addSeparator()
        self._project_combo = QComboBox(toolbar)
        self._project_combo.setToolTip("Switch active project")
        self._project_combo.currentIndexChanged.connect(self._on_project_combo_changed)
        toolbar.addWidget(self._project_combo)
        toolbar.addSeparator()
        toolbar.addAction(self.new_project_action)
        toolbar.addAction(self.add_folder_action)
        toolbar.addAction(self.backup_action)
        self.addToolBar(toolbar)

    def _create_status_bar(self) -> None:
        status = QStatusBar(self)
        self.setStatusBar(status)

        project_label = QLabel(self._project_label_text())
        status.addWidget(project_label)

        self._progress_bar = QProgressBar(self)
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximumWidth(200)
        status.addPermanentWidget(self._progress_bar)

    def _create_layout(self) -> None:
        central = QWidget(self)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self._splitter = QSplitter(Qt.Orientation.Horizontal, central)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.splitterMoved.connect(lambda *_args: self._store_splitter_sizes())
        root_layout.addWidget(self._splitter)

        # Left panel - corpus selector and ingest controls
        self._corpus_panel = QWidget(self._splitter)
        self._corpus_panel.setObjectName("corpusPane")
        corpus_layout = QVBoxLayout(self._corpus_panel)
        corpus_layout.setContentsMargins(12, 12, 12, 12)
        corpus_layout.setSpacing(12)

        controls_frame = QFrame(self._corpus_panel)
        controls_frame.setObjectName("corpusControls")
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(8)

        controls_header = QHBoxLayout()
        controls_header.setContentsMargins(0, 0, 0, 0)
        controls_header.setSpacing(6)
        controls_label = QLabel("Corpus", controls_frame)
        controls_label.setObjectName("corpusLabel")
        controls_header.addWidget(controls_label)
        controls_header.addStretch(1)
        collapse_left = QPushButton("Collapse", controls_frame)
        collapse_left.setObjectName("collapseCorpus")
        collapse_left.clicked.connect(
            lambda: self.toggle_corpus_panel_action.setChecked(False)
        )
        controls_header.addWidget(collapse_left)
        controls_layout.addLayout(controls_header)

        add_folder_button = QPushButton("Index Folder…", controls_frame)
        add_folder_button.setObjectName("indexFolderButton")
        add_folder_button.clicked.connect(
            lambda _checked=False: self.add_folder_action.trigger()
        )
        controls_layout.addWidget(add_folder_button)

        add_files_button = QPushButton("Index Files…", controls_frame)
        add_files_button.setObjectName("indexFilesButton")
        add_files_button.clicked.connect(
            lambda _checked=False: self.add_files_action.trigger()
        )
        controls_layout.addWidget(add_files_button)

        self._rescan_button = QPushButton("Rescan Indexed Folders", controls_frame)
        self._rescan_button.setObjectName("rescanCorpusButton")
        self._rescan_button.setEnabled(False)
        self._rescan_button.clicked.connect(
            lambda _checked=False: self.rescan_corpus_action.trigger()
        )
        controls_layout.addWidget(self._rescan_button)

        controls_layout.addStretch(1)
        corpus_layout.addWidget(controls_frame)

        self._corpus_tree = QTreeWidget(self._corpus_panel)
        self._corpus_tree.setObjectName("corpusSelector")
        self._corpus_tree.setHeaderHidden(True)
        self._corpus_tree.setIndentation(16)
        corpus_layout.addWidget(self._corpus_tree, 1)
        self._splitter.addWidget(self._corpus_panel)

        # Center panel - conversation and controls
        chat_panel = QWidget(self._splitter)
        chat_panel.setObjectName("chatPane")
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(12, 12, 12, 12)
        chat_layout.setSpacing(12)

        self.answer_view = AnswerView(
            settings=self.conversation_settings,
            progress_service=self.progress_service,
            parent=chat_panel,
        )
        self.answer_view.setObjectName("answerView")
        self.answer_view.apply_colors(self.settings_service.chat_style)
        self.answer_view.citation_activated.connect(self._on_card_citation)
        chat_layout.addWidget(self.answer_view, 1)

        self._controls_frame = self._create_conversation_controls(chat_panel)
        chat_layout.addWidget(self._controls_frame)

        self.question_input = QuestionInputWidget(chat_panel)
        self.question_input.ask_requested.connect(self._handle_ask)
        self.question_input.scope_cleared.connect(self._on_scope_chip_cleared)
        chat_layout.addWidget(self.question_input)
        self._splitter.addWidget(chat_panel)

        # Right panel - evidence viewer
        self._evidence_panel = EvidencePanel(self._splitter)
        self._evidence_panel.setObjectName("evidencePanel")
        self._evidence_panel.scope_changed.connect(self._on_evidence_scope_changed)
        self._evidence_panel.evidence_selected.connect(self._on_evidence_selected)
        self._evidence_panel.locate_requested.connect(self._on_evidence_locate_requested)
        self._evidence_panel.copy_requested.connect(self._on_copy_evidence_snippet)
        self._evidence_panel.reask_requested.connect(self._on_reask_requested)
        self._splitter.addWidget(self._evidence_panel)

        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 3)
        self._splitter.setStretchFactor(2, 2)

        self.setCentralWidget(central)

        self._apply_splitter_preferences()

        # Keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self._focus_corpus)
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self._copy_chat_text)
        self._update_question_prerequisites(self._conversation_manager.connection_state)

    # ------------------------------------------------------------------
    def _apply_splitter_preferences(self) -> None:
        if not hasattr(self, "_splitter"):
            return
        sizes = list(self.settings_service.splitter_sizes)
        if len(sizes) == 3 and any(size > 0 for size in sizes):
            self._splitter.setSizes([max(80, int(size)) for size in sizes])
        self._set_panel_visibility(0, self.settings_service.show_corpus_panel)
        self._set_panel_visibility(2, self.settings_service.show_evidence_panel)
        if hasattr(self, "toggle_corpus_panel_action"):
            self.toggle_corpus_panel_action.blockSignals(True)
            self.toggle_corpus_panel_action.setChecked(self.settings_service.show_corpus_panel)
            self.toggle_corpus_panel_action.blockSignals(False)
        if hasattr(self, "toggle_evidence_panel_action"):
            self.toggle_evidence_panel_action.blockSignals(True)
            self.toggle_evidence_panel_action.setChecked(
                self.settings_service.show_evidence_panel
            )
            self.toggle_evidence_panel_action.blockSignals(False)

    def _configure_question_settings_menu(self) -> None:
        menu = QMenu("Response settings", self.question_input)
        length_menu = menu.addMenu("Answer length")
        self._length_action_group = QActionGroup(menu)
        self._length_action_group.setExclusive(True)
        for preset in AnswerLength:
            action = QAction(preset.name.title(), menu)
            action.setCheckable(True)
            action.setData(preset)
            if preset is self.conversation_settings.answer_length:
                action.setChecked(True)
            self._length_action_group.addAction(action)
            length_menu.addAction(action)
        self._length_action_group.triggered.connect(self._on_answer_length_action_triggered)
        menu.addSeparator()
        model_action = QAction("Set model…", menu)
        model_action.triggered.connect(self._prompt_model_name)
        menu.addAction(model_action)
        self.question_input.set_settings_menu(menu)
        self._sync_answer_length_actions(self.conversation_settings.answer_length)

    def _set_panel_visibility(self, index: int, visible: bool) -> None:
        widget = self._splitter.widget(index) if hasattr(self, "_splitter") else None
        if widget is None:
            return
        widget.setVisible(bool(visible))
        if visible:
            widget.show()
        else:
            widget.hide()

    def _store_splitter_sizes(self) -> None:
        if not hasattr(self, "_splitter"):
            return
        sizes = self._splitter.sizes()
        if len(sizes) != 3:
            return
        if any(size > 0 for size in sizes):
            self._last_splitter_sizes = list(sizes)
            self.settings_service.set_splitter_sizes(sizes)

    def _restore_splitter_sizes(self) -> None:
        sizes = getattr(self, "_last_splitter_sizes", None)
        if not sizes:
            sizes = list(self.settings_service.splitter_sizes)
        if hasattr(self, "_splitter") and sizes and len(sizes) == 3:
            self._splitter.setSizes([max(80, int(size)) for size in sizes])

    def _on_toggle_theme_triggered(self, _: bool) -> None:
        self.settings_service.toggle_theme()

    def _toggle_corpus_panel(self, visible: bool) -> None:
        if visible:
            self.settings_service.set_show_corpus_panel(True)
            self._set_panel_visibility(0, True)
            self._restore_splitter_sizes()
        else:
            self._store_splitter_sizes()
            self._set_panel_visibility(0, False)
            self.settings_service.set_show_corpus_panel(False)

    def _toggle_evidence_panel(self, visible: bool) -> None:
        if visible:
            self.settings_service.set_show_evidence_panel(True)
            self._set_panel_visibility(2, True)
            self._restore_splitter_sizes()
        else:
            self._store_splitter_sizes()
            self._set_panel_visibility(2, False)
            self.settings_service.set_show_evidence_panel(False)

    def _on_density_action_triggered(self, action: QAction) -> None:
        value = action.data()
        if isinstance(value, str):
            self.settings_service.set_density(value)
            self._apply_density(value)

    def _apply_density(self, density: str | None = None) -> None:
        mode = (density or self.settings_service.density or "comfortable").lower()
        spacing = 8 if mode == "compact" else 12
        if hasattr(self, "_splitter"):
            for index in range(self._splitter.count()):
                widget = self._splitter.widget(index)
                if widget:
                    layout = widget.layout()
                    if isinstance(layout, (QVBoxLayout, QHBoxLayout)):
                        layout.setSpacing(spacing)
        self.answer_view.set_density(mode)
        self._evidence_panel.set_density(mode)
        self.question_input.set_density(mode)

    def _on_scope_chip_cleared(self) -> None:
        if not any(self._current_retrieval_scope.values()):
            return
        self._current_retrieval_scope = {"include": [], "exclude": []}
        self._update_session(scope=self._current_retrieval_scope)
        self._evidence_panel.reset_scope()
        self._update_scope_chip()
        if self._last_question:
            self._ask_question(self._last_question, triggered_by_scope=True)

    def _update_scope_chip(self) -> None:
        include = len(self._current_retrieval_scope.get("include", []))
        exclude = len(self._current_retrieval_scope.get("exclude", []))
        self.question_input.update_scope_chip(include, exclude)

    def _on_evidence_locate_requested(self, payload: dict[str, Any]) -> None:
        document_id = payload.get("document_id") if isinstance(payload, dict) else None
        if document_id is not None and self._select_document_in_tree(int(document_id)):
            return
        path = payload.get("path") if isinstance(payload, dict) else None
        if path:
            self._focus_tree_by_path(str(path))

    def _on_copy_evidence_snippet(self, snippet: str) -> None:
        if not snippet:
            return
        QApplication.clipboard().setText(snippet)
        self.progress_service.notify("Evidence snippet copied", level="info", duration_ms=1800)

    def _select_document_in_tree(self, document_id: int) -> bool:
        root = self._corpus_tree.invisibleRootItem()

        def visit(item: QTreeWidgetItem) -> bool:
            if item.data(0, Qt.ItemDataRole.UserRole) == document_id:
                current = item
                while current:
                    current.setExpanded(True)
                    current = current.parent()
                self._corpus_tree.setCurrentItem(item)
                self._corpus_tree.scrollToItem(item)
                return True
            for idx in range(item.childCount()):
                if visit(item.child(idx)):
                    return True
            return False

        for index in range(root.childCount()):
            if visit(root.child(index)):
                return True
        return False

    def _focus_tree_by_path(self, path: str) -> None:
        target = Path(path)
        root = self._corpus_tree.invisibleRootItem()

        def visit(item: QTreeWidgetItem) -> bool:
            tooltip = item.toolTip(0)
            if tooltip:
                try:
                    current_path = Path(tooltip)
                except Exception:
                    current_path = None
                if current_path and current_path == target:
                    current = item
                    while current:
                        current.setExpanded(True)
                        current = current.parent()
                    self._corpus_tree.setCurrentItem(item)
                    self._corpus_tree.scrollToItem(item)
                    return True
            for idx in range(item.childCount()):
                if visit(item.child(idx)):
                    return True
            return False

        for index in range(root.childCount()):
            if visit(root.child(index)):
                return
    # Project coordination
    def _initialise_project_state(self) -> None:
        self._refresh_project_selector()
        active = self.project_service.active_project()
        self._load_project_session(active.id)

    def _refresh_project_selector(self) -> None:
        projects = self.project_service.list_projects()
        current_id = self.project_service.active_project_id
        self._project_combo.blockSignals(True)
        self._project_combo.clear()
        current_index = -1
        for record in projects:
            self._project_combo.addItem(record.name, record.id)
            if record.id == current_id:
                current_index = self._project_combo.count() - 1
        if current_index >= 0:
            self._project_combo.setCurrentIndex(current_index)
        self._project_combo.blockSignals(False)
        self.delete_project_action.setEnabled(len(projects) > 1)

    def _on_projects_changed(self, _projects: list[ProjectRecord]) -> None:
        self._refresh_project_selector()

    def _on_active_project_changed(self, project: ProjectRecord) -> None:
        self._refresh_project_selector()
        self.search_service = SearchService(
            self.project_service.ingest,
            self.project_service.documents,
            self.project_service.chats,
        )
        self._load_project_session(project.id)

    def _on_project_combo_changed(self, index: int) -> None:
        project_id = self._project_combo.itemData(index)
        if not isinstance(project_id, int):
            return
        if project_id == self.project_service.active_project_id:
            return
        self._store_active_project_session()
        self.project_service.set_active_project(project_id)

    def _load_project_session(self, project_id: int) -> None:
        session = self._project_sessions.get(project_id)
        if session is None:
            snapshot = self.project_service.load_conversation_settings(project_id)
            session = {"settings": snapshot}
            self._project_sessions[project_id] = session
        self._turns = list(session.get("turns", []))
        self._conversation_manager.turns = list(self._turns)
        self.answer_view.render_turns(self._turns)
        turns = self.answer_view.turns
        self._active_card = turns[-1] if turns else None
        scope = session.get("scope")
        if isinstance(scope, dict):
            include = list(scope.get("include", []))
            exclude = list(scope.get("exclude", []))
            self._current_retrieval_scope = {"include": include, "exclude": exclude}
        else:
            self._current_retrieval_scope = {"include": [], "exclude": []}
        self._last_question = session.get("last_question")
        self._apply_conversation_settings_snapshot(session.get("settings"))
        if self._turns and self._turns[-1].citations:
            self._evidence_panel.set_evidence(self._turns[-1].citations)
        else:
            self._evidence_panel.clear()
        self._evidence_panel.set_reask_enabled(bool(self._last_question))
        self._update_export_actions()
        self.export_snippet_action.setEnabled(self._evidence_panel.selected_index is not None)
        self._refresh_corpus_view()
        self._update_corpus_actions()

    def _snapshot_conversation_settings(self) -> dict[str, Any]:
        return {
            "reasoning_verbosity": self.conversation_settings.reasoning_verbosity.value,
            "show_plan": self.conversation_settings.show_plan,
            "show_assumptions": self.conversation_settings.show_assumptions,
            "sources_only": self.conversation_settings.sources_only_mode,
            "answer_length": self.conversation_settings.answer_length.value,
            "model": self.conversation_settings.model_name,
        }

    def _apply_conversation_settings_snapshot(self, snapshot: Any) -> None:
        data = snapshot if isinstance(snapshot, dict) else {}
        verbosity = data.get("reasoning_verbosity")
        resolved: ReasoningVerbosity | None = None
        if isinstance(verbosity, str):
            try:
                resolved = ReasoningVerbosity[verbosity.upper()]
            except KeyError:
                try:
                    resolved = ReasoningVerbosity(verbosity)
                except ValueError:
                    resolved = None
        if isinstance(verbosity, ReasoningVerbosity):
            resolved = verbosity
        if resolved and resolved is not self.conversation_settings.reasoning_verbosity:
            self.conversation_settings.set_reasoning_verbosity(resolved)
        plan = data.get("show_plan")
        if isinstance(plan, bool):
            self.conversation_settings.set_show_plan(plan)
        assumptions = data.get("show_assumptions")
        if isinstance(assumptions, bool):
            self.conversation_settings.set_show_assumptions(assumptions)
        sources_only = data.get("sources_only")
        if isinstance(sources_only, bool):
            self.conversation_settings.set_sources_only_mode(sources_only)
        answer_length = data.get("answer_length")
        resolved_length: AnswerLength | None = None
        if isinstance(answer_length, str):
            try:
                resolved_length = AnswerLength[answer_length.upper()]
            except KeyError:
                try:
                    resolved_length = AnswerLength(answer_length)
                except ValueError:
                    resolved_length = None
        if isinstance(answer_length, AnswerLength):
            resolved_length = answer_length
        if resolved_length and resolved_length is not self.conversation_settings.answer_length:
            self.conversation_settings.set_answer_length(resolved_length)
        model = data.get("model")
        if isinstance(model, str) and model.strip():
            self.conversation_settings.set_model_name(model)
            self.lmstudio_client.configure(model=model.strip())

    def _update_session(self, project_id: int | None = None, **fields: Any) -> None:
        if project_id is None:
            project_id = self.project_service.active_project_id
        session = self._project_sessions.setdefault(project_id, {})
        session.update(fields)

    def _store_active_project_session(self) -> None:
        project_id = self.project_service.active_project_id
        snapshot = self._snapshot_conversation_settings()
        self._update_session(
            project_id,
            turns=list(self._turns),
            scope=dict(self._current_retrieval_scope),
            last_question=self._last_question,
            settings=snapshot,
        )
        self.project_service.save_conversation_settings(project_id, snapshot)

    def _build_default_export_path(self, suffix: str) -> str:
        project = self.project_service.active_project()
        safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in project.name)
        slug = "-".join(filter(None, safe.split("-"))) or "project"
        location = self.project_service.get_project_storage_location(project.id)
        base = Path(location) if location is not None else Path(self.project_service.storage_root)
        return str(base / f"{slug}{suffix}")

    # ------------------------------------------------------------------
    # Corpus management
    def _ingest_include_patterns(self) -> list[str]:
        return list(SUPPORTED_PATTERNS)

    def _ingest_file_filter_spec(self) -> str:
        patterns = " ".join(self._ingest_include_patterns())
        return f"Documents ({patterns});;All Files (*)"

    def _add_folder_to_corpus(self, _checked: bool = False) -> None:
        project = self.project_service.active_project()
        roots = self.project_service.list_corpus_roots(project.id)
        initial = roots[-1] if roots else ""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Index",
            initial,
        )
        if not folder:
            return
        include = self._ingest_include_patterns()
        try:
            job_id = self.ingest_service.queue_folder_crawl(
                project.id,
                folder,
                include=include,
            )
        except Exception as exc:  # pragma: no cover - user feedback path
            QMessageBox.critical(
                self,
                "Add Folder",
                str(exc) or "Unable to start indexing for the selected folder.",
            )
            return
        folder_name = Path(folder).name or Path(folder).resolve().name or folder
        description = f"Indexing {folder_name}"
        self._register_ingest_job(
            job_id,
            project_id=project.id,
            description=description,
            root=folder,
        )
        self.project_service.add_corpus_root(project.id, folder)
        self._update_corpus_actions()
        self._show_toast("Folder queued for indexing.", level="info", duration_ms=2500)

    def _add_files_to_corpus(self, _checked: bool = False) -> None:
        project = self.project_service.active_project()
        filter_spec = self._ingest_file_filter_spec()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Files to Index",
            "",
            filter_spec,
        )
        if not files:
            return
        include = self._ingest_include_patterns()
        try:
            job_id = self.ingest_service.queue_file_add(
                project.id,
                files,
                include=include,
            )
        except Exception as exc:  # pragma: no cover - user feedback path
            QMessageBox.critical(
                self,
                "Add Files",
                str(exc) or "Unable to start indexing for the selected files.",
            )
            return
        description = f"Indexing {len(files)} file(s)"
        root = str(Path(files[0]).resolve().parent)
        self._register_ingest_job(
            job_id,
            project_id=project.id,
            description=description,
            root=root,
        )
        self._show_toast("Files queued for indexing.", level="info", duration_ms=2500)

    def _rescan_corpus(self, _checked: bool = False) -> None:
        project = self.project_service.active_project()
        roots = self.project_service.list_corpus_roots(project.id)
        if not roots:
            QMessageBox.information(
                self,
                "Rescan Corpus",
                "No indexed folders are available to rescan.",
            )
            return
        include = self._ingest_include_patterns()
        queued = 0
        for root in roots:
            try:
                job_id = self.ingest_service.queue_rescan(
                    project.id,
                    root,
                    include=include,
                )
            except Exception as exc:  # pragma: no cover - user feedback path
                QMessageBox.warning(
                    self,
                    "Rescan Corpus",
                    f"Failed to queue rescan for {root}: {exc}",
                )
                continue
            root_name = Path(root).name or Path(root).resolve().name or root
            description = f"Rescanning {root_name}"
            self._register_ingest_job(
                job_id,
                project_id=project.id,
                description=description,
                root=root,
            )
            queued += 1
        if queued:
            self._show_toast(
                f"Queued rescan for {queued} folder(s).",
                level="info",
                duration_ms=2500,
            )

    def _register_ingest_job(
        self,
        job_id: int,
        *,
        project_id: int,
        description: str,
        root: str | None = None,
    ) -> None:
        task_id = f"ingest-{job_id}"
        self._ingest_jobs[job_id] = {
            "task_id": task_id,
            "project_id": project_id,
            "description": description,
            "root": root,
        }
        self.progress_service.start(task_id, description)

    def _enqueue_ingest_update(self, job_id: int, payload: dict[str, Any]) -> None:
        self._ingest_updates.put((job_id, payload))

    def _drain_ingest_updates(self) -> None:
        while True:
            try:
                job_id, payload = self._ingest_updates.get_nowait()
            except queue.Empty:
                break
            self._handle_ingest_update(job_id, payload)

    def _handle_ingest_update(self, job_id: int, payload: dict[str, Any]) -> None:
        job_info = self._ingest_jobs.get(job_id)
        if job_info is None:
            return
        task_id = job_info["task_id"]
        description = job_info.get("description", "Indexing corpus")
        status = payload.get("status")
        progress = payload.get("progress", {}) if isinstance(payload, dict) else {}
        total = int(progress.get("total", 0) or 0)
        processed = int(progress.get("processed", 0) or 0)
        if status == TaskStatus.RUNNING:
            message = (
                f"{description} ({processed}/{total})" if total else description
            )
            percent = (processed / total * 100.0) if total else None
            self.progress_service.update(task_id, message=message, percent=percent)
            return
        if status == TaskStatus.PAUSED:
            message = f"{description} (paused)"
            self.progress_service.update(task_id, message=message, indeterminate=True)
            return
        if status not in TaskStatus.FINAL:
            return

        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        errors = payload.get("errors") or summary.get("errors")

        if status == TaskStatus.COMPLETED:
            self.progress_service.finish(task_id, f"{description} complete")
            self._apply_ingest_results(job_info, payload)
            success = int(summary.get("success_count", 0) or 0)
            removed = summary.get("removed") or []
            details = f"Indexed {success} document(s)"
            if removed:
                details += f", removed {len(removed)}"
            self._show_toast(details + ".", level="info", duration_ms=3500)
        elif status == TaskStatus.CANCELLED:
            self.progress_service.finish(task_id, f"{description} cancelled")
            self._show_toast("Ingest cancelled.", level="warning", duration_ms=3000)
        elif status == TaskStatus.FAILED:
            self.progress_service.finish(task_id, f"{description} failed")
            message = "; ".join(str(err) for err in errors) if errors else "Ingest failed"
            self._show_toast(message, level="error", duration_ms=5000)
        self._ingest_jobs.pop(job_id, None)

    def _apply_ingest_results(self, job_info: dict[str, Any], payload: dict[str, Any]) -> None:
        project_id = job_info.get("project_id")
        if not isinstance(project_id, int):
            return
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        known_files = summary.get("known_files", {}) if isinstance(summary, dict) else {}
        removed = summary.get("removed", []) if isinstance(summary, dict) else []
        self._sync_documents_with_known_files(project_id, known_files, removed)
        self._refresh_corpus_view()
        self._update_corpus_actions()
        try:
            snapshot = self.project_service.export_project_database_snapshot(project_id)
        except DatabaseError:
            LOGGER.exception(
                "Failed to write database snapshot for project %s", project_id
            )
        else:
            if snapshot is not None:
                session = self._project_sessions.setdefault(project_id, {})
                session["database_snapshot"] = str(snapshot)

    def _sync_documents_with_known_files(
        self,
        project_id: int,
        known_files: Any,
        removed: Any,
    ) -> None:
        repo = self.project_service.documents
        existing_docs = {
            str(Path(doc["source_path"]).resolve()): doc
            for doc in repo.list_for_project(project_id)
            if doc.get("source_path")
        }
        normalized_known: dict[str, dict[str, Any]] = {}
        if isinstance(known_files, dict):
            for path, metadata in known_files.items():
                if not isinstance(path, str):
                    continue
                normalized_path = str(Path(path).resolve())
                normalized_known[normalized_path] = (
                    dict(metadata) if isinstance(metadata, dict) else {}
                )
                document = existing_docs.get(normalized_path)
                payload = {"file": normalized_known[normalized_path]}
                if document is None:
                    title = Path(normalized_path).stem or Path(normalized_path).name
                    repo.create(
                        project_id,
                        title,
                        source_type="file",
                        source_path=normalized_path,
                        metadata=payload,
                    )
                else:
                    updates: dict[str, Any] = {}
                    if document.get("source_path") != normalized_path:
                        updates["source_path"] = normalized_path
                    current_meta = document.get("metadata") or {}
                    if current_meta.get("file") != normalized_known[normalized_path]:
                        updates["metadata"] = payload
                    if updates:
                        repo.update(document["id"], **updates)
        removed_paths = []
        if isinstance(removed, (list, tuple, set)):
            removed_paths = [str(Path(path).resolve()) for path in removed if isinstance(path, str)]
        if removed_paths:
            for document in repo.list_for_project(project_id):
                source_path = document.get("source_path")
                if not source_path:
                    continue
                normalized = str(Path(source_path).resolve())
                if normalized in removed_paths:
                    repo.delete(document["id"])

    def _refresh_corpus_view(self) -> None:
        self._corpus_tree.clear()
        try:
            project_id = self.project_service.active_project_id
        except RuntimeError:
            self._has_documents = False
            self._corpus_tree.setEnabled(False)
            return
        documents = self.project_service.documents.list_for_project(project_id)
        self._has_documents = bool(documents)
        if not documents:
            placeholder = QTreeWidgetItem(["No documents indexed"])
            placeholder.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._corpus_tree.addTopLevelItem(placeholder)
            self._corpus_tree.setEnabled(False)
            return
        self._corpus_tree.setEnabled(True)
        tree = self.document_hierarchy.build_folder_tree(project_id)
        label_path = tree.get("path")
        if label_path:
            try:
                root_label = Path(label_path).name or str(Path(label_path))
            except Exception:
                root_label = str(label_path)
        else:
            root_label = self.project_service.active_project().name or "Corpus"
        if not root_label:
            root_label = "Corpus"
        root_item = QTreeWidgetItem([root_label])
        if label_path:
            root_item.setToolTip(0, str(label_path))
        self._corpus_tree.addTopLevelItem(root_item)
        self._populate_corpus_tree(root_item, tree)
        root_item.setExpanded(True)
        self._corpus_tree.resizeColumnToContents(0)
        self._update_question_prerequisites(self._conversation_manager.connection_state)

    def _populate_corpus_tree(self, parent: QTreeWidgetItem, node: dict[str, Any]) -> None:
        for child in node.get("children", []):
            name = child.get("name") or "(root)"
            item = QTreeWidgetItem([name])
            path = child.get("path")
            if path:
                item.setToolTip(0, str(path))
            parent.addChild(item)
            self._populate_corpus_tree(item, child)
        for document in node.get("documents", []):
            source_path = document.get("source_path")
            title = document.get("title")
            if not title and source_path:
                title = Path(source_path).name
            title = title or "Untitled"
            item = QTreeWidgetItem([title])
            if source_path:
                item.setToolTip(0, str(source_path))
            doc_id = document.get("id")
            if doc_id is not None:
                item.setData(0, Qt.ItemDataRole.UserRole, doc_id)
            parent.addChild(item)

    def _update_corpus_actions(self) -> None:
        try:
            project_id = self.project_service.active_project_id
        except RuntimeError:
            self.rescan_corpus_action.setEnabled(False)
            if self._rescan_button is not None:
                self._rescan_button.setEnabled(False)
            return
        roots = self.project_service.list_corpus_roots(project_id)
        self.rescan_corpus_action.setEnabled(bool(roots))
        if self._rescan_button is not None:
            self._rescan_button.setEnabled(bool(roots))

    def _prompt_new_project(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "New Project",
            "Project name:",
            QLineEdit.EchoMode.Normal,
        )
        if not ok:
            return
        cleaned = name.strip()
        if not cleaned:
            return
        project = self.project_service.create_project(cleaned)
        self._show_toast(f"Project '{project.name}' created.", level="info", duration_ms=2500)

    def _prompt_rename_project(self) -> None:
        project = self.project_service.active_project()
        name, ok = QInputDialog.getText(
            self,
            "Rename Project",
            "Project name:",
            QLineEdit.EchoMode.Normal,
            project.name,
        )
        if not ok:
            return
        cleaned = name.strip()
        if not cleaned or cleaned == project.name:
            return
        updated = self.project_service.rename_project(project.id, name=cleaned)
        self._show_toast(f"Project renamed to {updated.name}.", level="info", duration_ms=2000)

    def _delete_current_project(self) -> None:
        project = self.project_service.active_project()
        alternatives = [p for p in self.project_service.list_projects() if p.id != project.id]
        if not alternatives:
            QMessageBox.information(
                self,
                "Delete Project",
                "Create another project before deleting this one.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete Project",
            f"Delete project '{project.name}'? This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._store_active_project_session()
        replacement = alternatives[0]
        self.project_service.set_active_project(replacement.id)
        self.project_service.delete_project(project.id)
        self._project_sessions.pop(project.id, None)
        self._show_toast("Project deleted.", level="warning", duration_ms=2500)

    def _reveal_project_storage(self) -> None:
        project = self.project_service.active_project()
        path = self.project_service.get_project_storage(project.id)
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _purge_project_data(self) -> None:
        project = self.project_service.active_project()
        reply = QMessageBox.question(
            self,
            "Remove Project Data",
            "Remove indexed data, chats, and cached assets for this project?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.project_service.purge_project_data(project.id)
        self._turns.clear()
        self._conversation_manager.turns.clear()
        self.answer_view.clear()
        self._current_retrieval_scope = {"include": [], "exclude": []}
        self._evidence_panel.clear()
        self._last_question = None
        self._evidence_panel.set_reask_enabled(False)
        self._active_card = None
        self.export_snippet_action.setEnabled(False)
        self._update_export_actions()
        self._update_session(turns=[], scope=self._current_retrieval_scope, last_question=None)
        self._show_toast("Project data removed.", level="info", duration_ms=3000)
        self._refresh_corpus_view()
        self._update_corpus_actions()

    def _create_backup(self) -> None:
        default_path = self._build_default_export_path("-backup.zip")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Create Backup",
            default_path,
            "Zip Archives (*.zip)",
        )
        if not path:
            return
        try:
            saved = self.backup_service.create_backup(path)
        except Exception as exc:  # pragma: no cover - user feedback path
            QMessageBox.critical(self, "Backup Failed", str(exc) or "Unable to create backup.")
            return
        self._show_toast(f"Backup saved to {saved}", level="info", duration_ms=3000)

    def _restore_backup(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Restore Backup",
            "",
            "Zip Archives (*.zip)",
        )
        if not path:
            return
        try:
            self._project_sessions.clear()
            self.backup_service.restore_backup(path)
        except Exception as exc:  # pragma: no cover - user feedback path
            QMessageBox.critical(self, "Restore Failed", str(exc) or "Unable to restore backup.")
            return
        self._show_toast("Backup restored.", level="info", duration_ms=3000)

    def _export_conversation_markdown(self) -> None:
        if not self._turns:
            QMessageBox.information(
                self, "Export Conversation", "No conversation available to export."
            )
            return
        project = self.project_service.active_project()
        default_path = self._build_default_export_path("-conversation.md")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Conversation to Markdown",
            default_path,
            "Markdown Files (*.md)",
        )
        if not path:
            return
        try:
            self.export_service.export_conversation_markdown(
                path,
                self._turns,
                title=f"{project.name} Conversation",
                metadata={"Project": project.name},
            )
        except Exception as exc:  # pragma: no cover - user feedback path
            QMessageBox.critical(self, "Export Failed", str(exc) or "Unable to export conversation.")
            return
        self._show_toast("Conversation exported.", level="info", duration_ms=2500)

    def _export_conversation_html(self) -> None:
        if not self._turns:
            QMessageBox.information(
                self, "Export Conversation", "No conversation available to export."
            )
            return
        project = self.project_service.active_project()
        default_path = self._build_default_export_path("-conversation.html")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Conversation to HTML",
            default_path,
            "HTML Files (*.html)",
        )
        if not path:
            return
        try:
            self.export_service.export_conversation_html(
                path,
                self._turns,
                title=f"{project.name} Conversation",
                metadata={"Project": project.name},
            )
        except Exception as exc:  # pragma: no cover - user feedback path
            QMessageBox.critical(self, "Export Failed", str(exc) or "Unable to export conversation.")
            return
        self._show_toast("Conversation exported.", level="info", duration_ms=2500)

    def _export_selected_snippet(self) -> None:
        record = self._evidence_panel.selected_record()
        if record is None:
            QMessageBox.information(
                self, "Export Snippet", "Select an evidence snippet to export first."
            )
            return
        default_path = self._build_default_export_path("-snippet.txt")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Selected Snippet",
            default_path,
            "Text Files (*.txt)",
        )
        if not path:
            return
        payload = {
            "label": record.label,
            "snippet_html": record.snippet_html,
            "metadata_text": record.metadata_text,
        }
        try:
            self.export_service.export_snippets_text(path, [payload])
        except Exception as exc:  # pragma: no cover - user feedback path
            QMessageBox.critical(self, "Export Failed", str(exc) or "Unable to export snippet.")
            return
        self._show_toast("Snippet exported.", level="info", duration_ms=2000)

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
        self.settings_service.font_family_changed.connect(self._apply_font_scale)
        self.settings_service.font_point_size_changed.connect(self._apply_font_scale)
        self.settings_service.density_changed.connect(self._apply_density)
        self.settings_service.chat_style_changed.connect(self._on_chat_style_changed)
        self.progress_service.progress_started.connect(self._on_progress_started)
        self.progress_service.progress_updated.connect(self._on_progress_updated)
        self.progress_service.progress_finished.connect(self._on_progress_finished)
        self.progress_service.toast_requested.connect(self._show_toast)
        self.conversation_settings.reasoning_verbosity_changed.connect(
            self._persist_conversation_settings
        )
        self.conversation_settings.show_plan_changed.connect(self._persist_conversation_settings)
        self.conversation_settings.show_assumptions_changed.connect(
            self._persist_conversation_settings
        )
        self.conversation_settings.sources_only_mode_changed.connect(
            self._persist_conversation_settings
        )
        self.conversation_settings.answer_length_changed.connect(
            self._on_answer_length_changed
        )
        self.conversation_settings.model_changed.connect(self._on_model_changed)

    def _on_chat_style_changed(self, style: ChatStyleSettings) -> None:
        self.answer_view.apply_colors(style)

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

    def _persist_conversation_settings(self, *_args: object) -> None:
        snapshot = self._snapshot_conversation_settings()
        project_id = self.project_service.active_project_id
        self._update_session(project_id, settings=snapshot)
        self.project_service.save_conversation_settings(project_id, snapshot)

    def _update_export_actions(self) -> None:
        has_turns = bool(self._turns)
        self.export_markdown_action.setEnabled(has_turns)
        self.export_html_action.setEnabled(has_turns)

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
        self._ask_question(text, triggered_by_scope=False)

    def _focus_corpus(self) -> None:
        self._corpus_tree.setFocus()

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

    def _ask_question(self, question: str, *, triggered_by_scope: bool) -> None:
        if not question.strip():
            return

        state = self._conversation_manager.connection_state
        if not state.connected:
            self._update_question_prerequisites(state)
            self.progress_service.notify(
                state.message or "LMStudio is unavailable.", level="error", duration_ms=4000
            )
            return

        self._last_question = question
        self._update_session(last_question=question)
        self.question_input.set_busy(True)
        progress_message = "Refreshing evidence..." if triggered_by_scope else "Submitting question..."
        self.progress_service.start("chat-send", progress_message)
        asked_at = datetime.now()
        step_context_provider = self._build_step_context_provider(question)
        context_snippets, retrieval_documents = self._prepare_retrieval_context(question)
        extra_options = self._build_extra_request_options(
            question, retrieval_documents=retrieval_documents or None
        )
        try:
            turn = self._conversation_manager.ask(
                question,
                context_snippets=context_snippets or None,
                reasoning_verbosity=self.conversation_settings.reasoning_verbosity,
                response_mode=self.conversation_settings.response_mode,
                preset=self.conversation_settings.answer_length,
                extra_options=extra_options or None,
                context_provider=step_context_provider,
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
        review = getattr(turn, "adversarial_review", None)
        if review and getattr(review, "revised", False):
            self._show_toast(
                "Answer revised for accuracy/clarity.",
                level="info",
                duration_ms=3500,
            )
        self._turns.append(turn)
        self._update_session(turns=list(self._turns))
        card = self.answer_view.add_turn(turn)
        self._active_card = card
        self._update_evidence_panel(turn)
        self._update_export_actions()
        finish_message = "Evidence refreshed" if triggered_by_scope else "Answer received"
        self.progress_service.finish("chat-send", finish_message)
        self.question_input.set_busy(False)
        self._update_question_prerequisites(self._conversation_manager.connection_state)

    def _build_extra_request_options(
        self,
        question: str | None = None,
        *,
        retrieval_documents: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        scope = self._current_retrieval_scope
        include = list(scope.get("include", [])) if scope else []
        exclude = list(scope.get("exclude", [])) if scope else []
        retrieval: dict[str, Any] = {}
        if question and question.strip():
            retrieval["query"] = question.strip()
        if include:
            retrieval["include"] = include
        if exclude:
            retrieval["exclude"] = exclude
        if retrieval_documents:
            retrieval["documents"] = retrieval_documents
        if retrieval:
            options["retrieval"] = retrieval
        return options

    def _build_step_context_provider(
        self, question: str
    ) -> Callable[[PlanItem, int, int], Iterable[StepContextBatch]]:
        normalized = question.strip()
        try:
            project_id = self.project_service.active_project_id
        except RuntimeError:
            return lambda *_args, **_kwargs: []

        scope = self._current_retrieval_scope or {"include": [], "exclude": []}
        include = list(scope.get("include") or [])
        exclude = list(scope.get("exclude") or [])
        chunk_size = 3
        limit = 9

        def provider(plan_item: PlanItem, step_index: int, total_steps: int) -> list[StepContextBatch]:
            step_prompt = plan_item.description.strip()
            combined_query = "\n\n".join(
                part for part in [normalized, step_prompt] if part
            )
            records = self.search_service.collect_context_records(
                combined_query,
                project_id=project_id,
                include_identifiers=include,
                exclude_identifiers=exclude,
                limit=limit,
            )
            batches: list[StepContextBatch] = []
            for chunk in self._chunk_records(records, chunk_size):
                snippets, documents = self._context_payload_from_records(
                    chunk,
                    project_id,
                    step_index=step_index,
                )
                if snippets or documents:
                    batches.append(StepContextBatch(snippets=snippets, documents=documents))
            return batches

        return provider

    def _prepare_retrieval_context(
        self, question: str
    ) -> tuple[list[str], list[dict[str, Any]]]:
        if not question.strip():
            return [], []
        try:
            project_id = self.project_service.active_project_id
        except RuntimeError:
            return [], []
        scope = self._current_retrieval_scope or {"include": [], "exclude": []}
        include = scope.get("include") or []
        exclude = scope.get("exclude") or []
        records = self.search_service.collect_context_records(
            question,
            project_id=project_id,
            include_identifiers=include,
            exclude_identifiers=exclude,
        )
        return self._context_payload_from_records(records, project_id)

    def _context_payload_from_records(
        self,
        records: Iterable[dict[str, Any]],
        project_id: int,
        *,
        step_index: int | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        snippets: list[str] = []
        retrieval_documents: list[dict[str, Any]] = []
        prefix = f"Step {step_index}: " if step_index is not None else ""
        for record in records:
            document = record.get("document") or {}
            chunk = record.get("chunk") or {}
            context_text = str(record.get("context") or chunk.get("text") or "").strip()
            if not context_text:
                continue
            source_path = document.get("source_path") or record.get("path")
            title = document.get("title")
            if not title and source_path:
                try:
                    path_obj = Path(source_path)
                    title = path_obj.name or path_obj.stem
                except Exception:
                    title = str(source_path)
            if not title:
                title = "Document"
            identifiers = sorted(SearchService._document_identifiers(document))
            primary_identifier = identifiers[0] if identifiers else str(document.get("id") or title)
            header_parts = []
            if prefix:
                header_parts.append(prefix.strip())
            if primary_identifier:
                header_parts.append(f"[{primary_identifier}] {title}")
            else:
                header_parts.append(title)
            snippet_header = " ".join(header_parts)
            snippets.append(f"{snippet_header}\n{context_text}")

            payload: dict[str, Any] = {
                "id": primary_identifier,
                "source": title,
                "identifiers": identifiers,
                "text": context_text,
                "project_id": project_id,
            }
            score = record.get("score")
            if score is not None:
                try:
                    payload["score"] = float(score)
                except (TypeError, ValueError):
                    pass
            if source_path:
                payload["path"] = str(source_path)
            document_id = document.get("id")
            if document_id is not None:
                payload["document_id"] = int(document_id)
            chunk_id = chunk.get("id") if isinstance(chunk, dict) else None
            if chunk_id is not None:
                payload["chunk_id"] = int(chunk_id)
            chunk_index = chunk.get("index") if isinstance(chunk, dict) else None
            if chunk_index is not None:
                payload["chunk_index"] = int(chunk_index)
            start_offset = chunk.get("start_offset") if isinstance(chunk, dict) else None
            if start_offset is not None:
                payload["start_offset"] = int(start_offset)
            end_offset = chunk.get("end_offset") if isinstance(chunk, dict) else None
            if end_offset is not None:
                payload["end_offset"] = int(end_offset)
            highlight = record.get("highlight")
            if highlight:
                payload["snippet"] = str(highlight)
            ingest_doc = record.get("ingest_document") or {}
            ingest_id = ingest_doc.get("id")
            if ingest_id is not None:
                payload["ingest_document_id"] = int(ingest_id)
            ingest_version = ingest_doc.get("version")
            if ingest_version is not None:
                payload["ingest_version"] = int(ingest_version)
            retrieval_documents.append(payload)

        return snippets, retrieval_documents

    @staticmethod
    def _chunk_records(
        records: Iterable[dict[str, Any]], chunk_size: int
    ) -> Iterable[list[dict[str, Any]]]:
        chunk: list[dict[str, Any]] = []
        for record in records:
            chunk.append(record)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def _update_evidence_panel(self, turn: ConversationTurn) -> None:
        if not turn.citations:
            self._evidence_panel.clear()
            self._evidence_panel.set_reask_enabled(bool(self._last_question))
            self._current_retrieval_scope = {"include": [], "exclude": []}
            self._update_session(scope=self._current_retrieval_scope)
            self.export_snippet_action.setEnabled(False)
            self._update_scope_chip()
            return
        self._evidence_panel.set_evidence(turn.citations)
        self._evidence_panel.set_reask_enabled(bool(self._last_question))
        self._current_retrieval_scope = self._evidence_panel.current_scope
        self._update_session(scope=self._current_retrieval_scope)
        self.export_snippet_action.setEnabled(self._evidence_panel.evidence_count > 0)
        self._update_scope_chip()
        self.answer_view.highlight_citation(self._active_card, None)

    def _on_card_citation(self, card: ChatTurnWidget, index: int) -> None:
        if card is not self._active_card:
            self._active_card = card
            self._evidence_panel.set_evidence(card.turn.citations)
            self._current_retrieval_scope = self._evidence_panel.current_scope
            self._evidence_panel.set_reask_enabled(bool(self._last_question))
        self.answer_view.highlight_citation(card, index)
        self._evidence_panel.select_index(index - 1)

    def _on_evidence_selected(self, index: int, _identifier: str) -> None:
        if self._active_card is None:
            return
        self.answer_view.highlight_citation(self._active_card, index + 1)
        self.export_snippet_action.setEnabled(index >= 0)

    def _on_evidence_scope_changed(self, include: list[str], exclude: list[str]) -> None:
        self._current_retrieval_scope = {"include": list(include), "exclude": list(exclude)}
        self._update_session(scope=self._current_retrieval_scope)
        self._update_scope_chip()
        if self._last_question:
            self._ask_question(self._last_question, triggered_by_scope=True)

    def _on_reask_requested(self) -> None:
        if not self._last_question:
            self.progress_service.notify(
                "Ask a question first to enable re-asking.",
                level="warning",
                duration_ms=2200,
            )
            return
        self._ask_question(self._last_question, triggered_by_scope=True)

    def _update_question_prerequisites(self, state: ConnectionState) -> None:
        ok = state.connected
        message: str | None = None
        if not state.connected:
            message = state.message or "LMStudio connection unavailable."
        if ok and not self._has_documents:
            ok = False
            message = "Index at least one document to enable questions."
        self.question_input.set_prerequisites_met(ok, message)

    def _on_connection_state_changed(self, state: ConnectionState) -> None:
        self.question_input.set_connection_state(state)
        if state.message:
            self.statusBar().showMessage(state.message, 4000)
        elif not state.connected:
            self.statusBar().showMessage("LMStudio connection unavailable.", 4000)
        self._update_question_prerequisites(state)

    # ------------------------------------------------------------------
    def _open_settings(self) -> None:
        dialog = SettingsDialog(settings_service=self.settings_service, parent=self)
        dialog.exec()

    def _open_help(self) -> None:
        QMessageBox.information(self, "Help", "Visit the documentation for assistance.")

    def _open_log_viewer(self) -> None:
        root_logger = logging.getLogger("DataMiner")
        log_path = get_log_file_path(root_logger)
        if log_path is None:
            QMessageBox.warning(
                self,
                "Logs Unavailable",
                "Unable to determine the log file location.",
            )
            return

        dialog = LogViewerDialog(
            log_path=log_path,
            traceback_text="",
            message="Recent log output is shown below.",
            window_title="DataMiner Logs",
            parent=self,
        )
        dialog.exec()

    # ------------------------------------------------------------------
    def _apply_theme(self, theme: str) -> None:
        self.settings_service.apply_theme()
        self._show_toast(f"Theme set to {theme.title()}.", level="info", duration_ms=1500)

    def _apply_font_scale(self, *_args: object) -> None:
        self.settings_service.apply_font_scale()

    def _on_model_changed(self, name: str) -> None:
        cleaned = str(name).strip()
        if not cleaned:
            return
        self.lmstudio_client.configure(model=cleaned)
        self.question_input.set_model_name(cleaned)
        self._persist_conversation_settings()

    def _on_answer_length_changed(self, preset: AnswerLength) -> None:
        if isinstance(preset, AnswerLength):
            self.question_input.set_answer_length(preset)
            self._sync_answer_length_actions(preset)
        self._persist_conversation_settings()

    def _on_answer_length_action_triggered(self, action: QAction) -> None:
        preset = action.data()
        if isinstance(preset, AnswerLength):
            self.conversation_settings.set_answer_length(preset)

    def _sync_answer_length_actions(self, preset: AnswerLength) -> None:
        group = getattr(self, "_length_action_group", None)
        if group is None:
            return
        for action in group.actions():
            data = action.data()
            if isinstance(data, AnswerLength):
                action.setChecked(data is preset)

    def _prompt_model_name(self) -> None:
        current = self.conversation_settings.model_name
        name, ok = QInputDialog.getText(
            self,
            "Set LMStudio Model",
            "Model identifier:",
            QLineEdit.EchoMode.Normal,
            current,
        )
        if not ok:
            return
        cleaned = name.strip()
        if cleaned:
            self.conversation_settings.set_model_name(cleaned)

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
            self.question_input.set_status_message("LMStudio monitor off", "warning")
            return

        def worker() -> None:
            healthy = self.lmstudio_client.health_check()
            QTimer.singleShot(0, partial(self._set_lmstudio_status, healthy))

        threading.Thread(target=worker, daemon=True).start()

    def _set_lmstudio_status(self, healthy: bool) -> None:
        message = None if healthy else "LMStudio health probe failed"
        self.question_input.set_connection_state(ConnectionState(healthy, message))

    # ------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        if hasattr(self, "_health_timer"):
            self._health_timer.stop()
        if hasattr(self, "_ingest_timer"):
            self._ingest_timer.stop()
        unsubscribe = getattr(self, "_connection_unsubscribe", None)
        if callable(unsubscribe):
            try:
                unsubscribe()
            except Exception:
                pass
        ingest_unsubscribe = getattr(self, "_ingest_unsubscribe", None)
        if callable(ingest_unsubscribe):
            try:
                ingest_unsubscribe()
            except Exception:
                pass
        try:
            self._store_splitter_sizes()
        except Exception:
            pass
        try:
            self._store_active_project_session()
        except Exception:
            pass
        return super().closeEvent(event)


__all__ = ["MainWindow"]

