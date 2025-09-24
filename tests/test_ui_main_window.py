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
from app.services.progress_service import ProgressService
from app.services.settings_service import SettingsService
from app.ui.main_window import MainWindow
from app.services.lmstudio_client import LMStudioClient


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
