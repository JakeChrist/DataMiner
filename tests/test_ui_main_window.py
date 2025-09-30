"""UI tests for the DataMiner main window."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for UI tests", exc_type=ImportError)
pytest.importorskip(
    "PyQt6.QtWidgets",
    reason="PyQt6 widgets require a Qt runtime",
    exc_type=ImportError,
)

from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QApplication, QSplitter, QPushButton

from app.config import ConfigManager
from app.ingest.parsers import SUPPORTED_PATTERNS
from app.services.conversation_manager import (
    ConnectionState,
    ConversationTurn,
    ReasoningVerbosity,
)
from app.services.progress_service import ProgressService
from app.services.settings_service import SettingsService
from app.services.project_service import ProjectService
from app.services.document_hierarchy import DocumentHierarchyService
from app.services.export_service import ExportService
from app.services.backup_service import BackupService
from app.ui.main_window import MainWindow
from app.services.lmstudio_client import ChatMessage, LMStudioClient


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def build_settings_service(tmp_path, monkeypatch) -> SettingsService:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    manager = ConfigManager(app_name="DataMinerTest", filename="ui.json")
    return SettingsService(config_manager=manager)


@pytest.fixture()
def project_service(tmp_path, monkeypatch) -> ProjectService:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = ConfigManager(app_name="DataMinerTest", filename="projects.json")
    service = ProjectService(storage_root=tmp_path / "storage", config_manager=config)
    yield service
    service.shutdown()


class DummyLMStudioClient:
    """Deterministic LMStudio stub for UI interaction tests."""

    def __init__(self) -> None:
        self.last_messages: list[dict] = []
        self.last_options: dict | None = None
        self.calls = 0
        self.last_question: str | None = None

    def health_check(self) -> bool:
        return True

    def chat(self, messages, *, preset, extra_options=None) -> ChatMessage:  # type: ignore[override]
        self.last_messages = list(messages)
        self.last_options = extra_options or {}
        self.calls += 1
        if messages:
            last = messages[-1]
            if isinstance(last, dict):
                self.last_question = str(last.get("content", ""))
        reasoning = {
            "summary_bullets": ["Reviewed document A for relevant data."],
            "plan": [
                {"description": "Search repository", "status": "complete"},
                {"description": "Summarize findings", "status": "pending"},
            ],
            "assumptions": {
                "used": ["Assumed latest dataset applies"],
                "decision": "assume",
                "rationale": "No timeframe specified",
            },
            "self_check": {
                "passed": True,
                "flags": ["Validated citations"],
                "notes": "No issues detected",
            },
        }
        citations = [
            {
                "id": "doc-a",
                "source": "Doc A",
                "snippet": "<mark>Metric</mark> value is 42.",
                "page": 5,
                "section": "Summary",
                "path": "/tmp/doc_a.txt",
            },
            {
                "id": "doc-b",
                "source": "Doc B",
                "snippet": "Supporting details from Doc B.",
                "page": 2,
                "section": "Context",
                "path": "/tmp/doc_b.txt",
            },
        ]
        metadata = {"citations": citations, "reasoning": reasoning}
        raw_response = {
            "choices": [
                {
                    "message": {
                        "content": "Deterministic answer [1] references evidence [2] as well.",
                        "metadata": metadata,
                    }
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 24, "total_tokens": 36},
        }
        return ChatMessage(
            content="Deterministic answer [1] references evidence [2] as well.",
            citations=citations,
            reasoning=reasoning,
            raw_response=raw_response,
        )


def test_main_window_splitter_layout(qt_app, tmp_path, monkeypatch, project_service):
    settings_service = build_settings_service(tmp_path, monkeypatch)
    progress_service = ProgressService()
    lmstudio_client = LMStudioClient()
    export_service = ExportService()
    backup_service = BackupService(project_service)
    window = MainWindow(
        settings_service=settings_service,
        progress_service=progress_service,
        lmstudio_client=lmstudio_client,
        project_service=project_service,
        export_service=export_service,
        backup_service=backup_service,
        enable_health_monitor=False,
    )
    window.resize(1400, 900)
    qt_app.processEvents()
    splitter = window.findChild(QSplitter)
    assert splitter is not None
    assert splitter.count() == 3
    sizes = splitter.sizes()
    assert sum(sizes) > 0
    assert sizes[1] >= sizes[0]
    assert sizes[1] >= sizes[2]
    index_button = window.findChild(QPushButton, "indexFolderButton")
    assert index_button is not None
    rescan_button = window.findChild(QPushButton, "rescanCorpusButton")
    assert rescan_button is not None
    assert not rescan_button.isEnabled()
    window.close()


def test_main_window_ingest_patterns_match_parsers(
    qt_app, tmp_path, monkeypatch, project_service
) -> None:
    settings_service = build_settings_service(tmp_path, monkeypatch)
    progress_service = ProgressService()
    lmstudio_client = LMStudioClient()
    export_service = ExportService()
    backup_service = BackupService(project_service)
    window = MainWindow(
        settings_service=settings_service,
        progress_service=progress_service,
        lmstudio_client=lmstudio_client,
        project_service=project_service,
        export_service=export_service,
        backup_service=backup_service,
        enable_health_monitor=False,
    )
    try:
        patterns = window._ingest_include_patterns()
        assert patterns == list(SUPPORTED_PATTERNS)

        filter_spec = window._ingest_file_filter_spec()
        for pattern in SUPPORTED_PATTERNS:
            assert pattern in filter_spec
    finally:
        window.close()


def test_settings_persist_theme_and_font(tmp_path, monkeypatch):
    settings_service = build_settings_service(tmp_path, monkeypatch)
    settings_service.set_theme("dark")
    settings_service.set_font_scale(1.5)

    reloaded = build_settings_service(tmp_path, monkeypatch)
    assert reloaded.theme == "dark"
    assert reloaded.font_scale == pytest.approx(1.5)


def test_submission_flow_and_controls(qt_app, tmp_path, monkeypatch, project_service):
    settings_service = build_settings_service(tmp_path, monkeypatch)
    progress_service = ProgressService()
    client = DummyLMStudioClient()
    export_service = ExportService()
    backup_service = BackupService(project_service)
    window = MainWindow(
        settings_service=settings_service,
        progress_service=progress_service,
        lmstudio_client=client,
        project_service=project_service,
        export_service=export_service,
        backup_service=backup_service,
        enable_health_monitor=False,
    )

    window.question_input.set_text("What is the latest metric?")
    window.question_input.ask_button.click()
    qt_app.processEvents()

    assert len(window.answer_view.cards) == 1
    first_card = window.answer_view.cards[0]
    assert first_card.question_label.text().endswith("What is the latest metric?")
    assert "Deterministic answer" in first_card.answer_browser.toPlainText()
    assert "Doc A" in first_card.citations_browser.toPlainText()
    assert first_card.reasoning_section.isVisible()
    assert first_card.plan_section.isVisible()
    assert first_card.assumptions_section.isVisible()
    assert first_card.self_check_section.isVisible()
    assert "Latency" in first_card.metadata_label.text()
    assert "Prompt Tokens" in first_card.metadata_label.text()
    assert client.last_options is not None
    reasoning_options = client.last_options.get("reasoning", {})
    assert reasoning_options.get("verbosity") == ReasoningVerbosity.BRIEF.value
    assert reasoning_options.get("include_plan") is True
    assert window._evidence_panel.evidence_count == 2
    assert window._evidence_panel.evidence_items[0].label.startswith("[1] Doc A")

    window._plan_checkbox.setChecked(False)
    window._assumptions_checkbox.setChecked(False)
    qt_app.processEvents()
    assert not first_card.plan_section.isVisible()
    assert not first_card.assumptions_section.isVisible()
    window._plan_checkbox.setChecked(True)
    window._assumptions_checkbox.setChecked(True)
    qt_app.processEvents()

    minimal_index = window._verbosity_combo.findData(ReasoningVerbosity.MINIMAL)
    window._verbosity_combo.setCurrentIndex(minimal_index)
    window.question_input.set_text("Summarize context")
    window.question_input.ask_button.click()
    qt_app.processEvents()
    assert len(window.answer_view.cards) == 2
    assert client.last_options is not None
    reasoning_options = client.last_options.get("reasoning", {})
    assert reasoning_options.get("verbosity") == ReasoningVerbosity.MINIMAL.value
    assert reasoning_options.get("include_plan") is False

    window._sources_only_checkbox.setChecked(True)
    window.question_input.set_text("Only sources please")
    window.question_input.ask_button.click()
    qt_app.processEvents()
    assert len(window.answer_view.cards) == 3
    assert client.last_options is not None
    assert client.last_options.get("response_mode") == "sources_only"

    latest_card = window.answer_view.cards[-1]
    latest_card.copy_answer_button.click()
    qt_app.processEvents()
    assert QApplication.clipboard().text() == latest_card.answer_browser.toPlainText()

    latest_card.copy_citations_button.click()
    qt_app.processEvents()
    assert "Doc A" in QApplication.clipboard().text()

    window._copy_chat_text()
    qt_app.processEvents()
    copied_conversation = QApplication.clipboard().text()
    assert "What is the latest metric?" in copied_conversation
    assert "Only sources please" in copied_conversation

    window.close()


def test_evidence_scope_requery_and_preview(qt_app, tmp_path, monkeypatch, project_service):
    settings_service = build_settings_service(tmp_path, monkeypatch)
    progress_service = ProgressService()
    client = DummyLMStudioClient()
    export_service = ExportService()
    backup_service = BackupService(project_service)
    window = MainWindow(
        settings_service=settings_service,
        progress_service=progress_service,
        lmstudio_client=client,
        project_service=project_service,
        export_service=export_service,
        backup_service=backup_service,
        enable_health_monitor=False,
    )

    window.question_input.set_text("Explain the metric")
    window.question_input.ask_button.click()
    qt_app.processEvents()

    assert client.calls == 1
    assert window._evidence_panel.evidence_count == 2

    window._evidence_panel.select_index(0)
    qt_app.processEvents()
    assert "Metric" in window._evidence_panel.evidence_items[0].snippet_html
    assert "Page 5" in window._evidence_panel._metadata_label.text()

    first_card = window.answer_view.cards[0]
    first_card.citations_browser.anchorClicked.emit(QUrl("cite-2"))
    qt_app.processEvents()
    assert window._evidence_panel.selected_index == 1
    assert first_card.selected_citation == 2

    row_widget = window._evidence_panel._list.itemWidget(window._evidence_panel._list.item(0))
    row_widget.exclude_button.setChecked(True)
    qt_app.processEvents()

    assert client.calls >= 2
    assert client.last_options is not None
    retrieval = client.last_options.get("retrieval", {})
    assert retrieval.get("exclude") == ["doc-a"]
    assert "doc-b" in retrieval.get("include", [])
    documents = retrieval.get("documents")
    assert isinstance(documents, list) and documents
    assert documents[0]["text"]
    assert documents[0]["id"]

    reask_button = window._evidence_panel._reask_button
    assert reask_button.isEnabled()
    previous_calls = client.calls
    reask_button.click()
    qt_app.processEvents()
    assert client.calls >= previous_calls + 1

    window.close()


def test_question_includes_retrieval_context(
    qt_app, tmp_path, monkeypatch, project_service
) -> None:
    context_path = tmp_path / "docs" / "context.txt"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_text = "Solar adoption is accelerating across regions."
    context_path.write_text(context_text, encoding="utf-8")

    class RecordingConversationManager:
        def __init__(self, _client) -> None:
            self.turns: list[ConversationTurn] = []
            self.last_call: dict[str, object] | None = None
            self._state = ConnectionState(True, None)

        def add_connection_listener(self, _listener):
            return lambda: None

        @property
        def connection_state(self) -> ConnectionState:
            return self._state

        def ask(self, question: str, **kwargs) -> ConversationTurn:  # type: ignore[override]
            self.last_call = {"question": question, **kwargs}
            turn = ConversationTurn(question=question, answer="Stub answer")
            self.turns.append(turn)
            return turn

    class StubSearchService:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def collect_context_records(self, *_args, **_kwargs) -> list[dict[str, object]]:
            return [
                {
                    "document": {
                        "id": 1,
                        "title": "Context Doc",
                        "source_path": str(context_path),
                    },
                    "chunk": {"id": 11, "index": 0, "text": context_text},
                    "context": context_text,
                    "score": 0.12,
                    "ingest_document": {"id": 5, "version": 1},
                }
            ]

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr("app.ui.main_window.ConversationManager", RecordingConversationManager)
    monkeypatch.setattr("app.ui.main_window.SearchService", StubSearchService)

    settings_service = build_settings_service(tmp_path, monkeypatch)
    progress_service = ProgressService()
    client = DummyLMStudioClient()
    export_service = ExportService()
    backup_service = BackupService(project_service)

    class StubIngestService:
        def subscribe(self, _callback):
            return lambda: None

        def shutdown(self, *, wait: bool = True) -> None:  # pragma: no cover - compatibility
            pass

    ingest_service = StubIngestService()
    document_hierarchy = DocumentHierarchyService(project_service.documents)

    window = MainWindow(
        settings_service=settings_service,
        progress_service=progress_service,
        lmstudio_client=client,
        project_service=project_service,
        ingest_service=ingest_service,
        document_hierarchy=document_hierarchy,
        export_service=export_service,
        backup_service=backup_service,
        enable_health_monitor=False,
    )

    try:
        window.question_input.set_text("Summarize solar adoption trends")
        window.question_input.ask_button.click()
        qt_app.processEvents()

        manager = window._conversation_manager  # type: ignore[attr-defined]
        assert isinstance(manager, RecordingConversationManager)
        assert manager.last_call is not None
        context_snippets = manager.last_call.get("context_snippets")
        assert isinstance(context_snippets, list) and context_snippets
        assert any("Solar adoption" in snippet for snippet in context_snippets)
        extra_options = manager.last_call.get("extra_options")
        assert isinstance(extra_options, dict)
        documents = extra_options.get("retrieval", {}).get("documents", [])  # type: ignore[assignment]
        assert documents and documents[0]["text"] == context_text
    finally:
        window.close()


def test_evidence_open_in_system_app(qt_app, tmp_path, monkeypatch, project_service):
    settings_service = build_settings_service(tmp_path, monkeypatch)
    progress_service = ProgressService()
    client = DummyLMStudioClient()
    export_service = ExportService()
    backup_service = BackupService(project_service)
    window = MainWindow(
        settings_service=settings_service,
        progress_service=progress_service,
        lmstudio_client=client,
        project_service=project_service,
        export_service=export_service,
        backup_service=backup_service,
        enable_health_monitor=False,
    )

    window.question_input.set_text("Launch source")
    window.question_input.ask_button.click()
    qt_app.processEvents()

    called: dict[str, QUrl] = {}

    def fake_open(url: QUrl) -> bool:  # pragma: no cover - executed in test
        called["url"] = url
        return True

    monkeypatch.setattr("PyQt6.QtGui.QDesktopServices.openUrl", staticmethod(fake_open))

    window._evidence_panel.select_index(0)
    qt_app.processEvents()
    row = window._evidence_panel._list.itemWidget(window._evidence_panel._list.item(0))
    assert row is not None
    row.open_button.click()
    qt_app.processEvents()

    assert "url" in called
    assert called["url"].toLocalFile() == "/tmp/doc_a.txt"

    window.close()
