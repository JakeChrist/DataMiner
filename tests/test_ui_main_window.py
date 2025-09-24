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

from PyQt6.QtWidgets import QApplication, QSplitter

from app.config import ConfigManager
from app.services.conversation_manager import ReasoningVerbosity
from app.services.progress_service import ProgressService
from app.services.settings_service import SettingsService
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


class DummyLMStudioClient:
    """Deterministic LMStudio stub for UI interaction tests."""

    def __init__(self) -> None:
        self.last_messages: list[dict] = []
        self.last_options: dict | None = None

    def health_check(self) -> bool:
        return True

    def chat(self, messages, *, preset, extra_options=None) -> ChatMessage:  # type: ignore[override]
        self.last_messages = list(messages)
        self.last_options = extra_options or {}
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
        metadata = {"citations": ["Doc A"], "reasoning": reasoning}
        raw_response = {
            "choices": [
                {
                    "message": {
                        "content": "Deterministic answer",
                        "metadata": metadata,
                    }
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 24, "total_tokens": 36},
        }
        return ChatMessage(
            content="Deterministic answer",
            citations=["Doc A"],
            reasoning=reasoning,
            raw_response=raw_response,
        )


def test_main_window_splitter_layout(qt_app, tmp_path, monkeypatch):
    settings_service = build_settings_service(tmp_path, monkeypatch)
    progress_service = ProgressService()
    lmstudio_client = LMStudioClient()
    window = MainWindow(
        settings_service=settings_service,
        progress_service=progress_service,
        lmstudio_client=lmstudio_client,
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
    window.close()


def test_settings_persist_theme_and_font(tmp_path, monkeypatch):
    settings_service = build_settings_service(tmp_path, monkeypatch)
    settings_service.set_theme("dark")
    settings_service.set_font_scale(1.5)

    reloaded = build_settings_service(tmp_path, monkeypatch)
    assert reloaded.theme == "dark"
    assert reloaded.font_scale == pytest.approx(1.5)


def test_submission_flow_and_controls(qt_app, tmp_path, monkeypatch):
    settings_service = build_settings_service(tmp_path, monkeypatch)
    progress_service = ProgressService()
    client = DummyLMStudioClient()
    window = MainWindow(
        settings_service=settings_service,
        progress_service=progress_service,
        lmstudio_client=client,
        enable_health_monitor=False,
    )

    window.question_input.set_text("What is the latest metric?")
    window.question_input.ask_button.click()
    qt_app.processEvents()

    assert len(window.answer_view.cards) == 1
    first_card = window.answer_view.cards[0]
    assert first_card.question_label.text().endswith("What is the latest metric?")
    assert "Deterministic answer" in first_card.answer_browser.toPlainText()
    assert "Doc A" in first_card.citations_label.text()
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
    assert "Doc A" in window._evidence_panel.toPlainText()

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
