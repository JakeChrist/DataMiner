"""Application settings management for UI preferences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWidgets import QApplication

from ..config import ConfigManager


DEFAULT_THEME = "light"
MIN_FONT_SCALE = 0.5
MAX_FONT_SCALE = 2.5


@dataclass(slots=True)
class UISettings:
    """Container for persisted UI settings."""

    theme: str = DEFAULT_THEME
    font_scale: float = 1.0


class SettingsService(QObject):
    """Persist and broadcast UI level settings such as theme and fonts."""

    theme_changed = pyqtSignal(str)
    font_scale_changed = pyqtSignal(float)

    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        super().__init__()
        self._config = config_manager or ConfigManager()
        self._settings = UISettings()
        self._base_font_point_size: float | None = None
        self.reload()

    # ------------------------------------------------------------------
    # Persistence helpers
    def reload(self) -> None:
        """Reload settings from disk."""

        data = self._config.load()
        ui_settings = data.get("ui") if isinstance(data, dict) else {}
        theme = str(ui_settings.get("theme", DEFAULT_THEME)).lower()
        if theme not in {"light", "dark"}:
            theme = DEFAULT_THEME
        font_scale = ui_settings.get("font_scale", 1.0)
        try:
            font_scale = float(font_scale)
        except (TypeError, ValueError):
            font_scale = 1.0
        font_scale = float(min(MAX_FONT_SCALE, max(MIN_FONT_SCALE, font_scale)))
        self._settings = UISettings(theme=theme, font_scale=font_scale)
        self._base_font_point_size = None

    def save(self) -> None:
        """Persist the current settings to disk."""

        data = self._config.load()
        if not isinstance(data, dict):
            data = {}
        ui_data = data.get("ui") if isinstance(data.get("ui"), dict) else {}
        ui_data.update(
            {
                "theme": self._settings.theme,
                "font_scale": self._settings.font_scale,
            }
        )
        data["ui"] = ui_data
        self._config.save(data)

    # ------------------------------------------------------------------
    # Accessors
    @property
    def theme(self) -> str:
        return self._settings.theme

    @property
    def font_scale(self) -> float:
        return self._settings.font_scale

    # ------------------------------------------------------------------
    # Mutators
    def set_theme(self, theme: str) -> None:
        normalized = "dark" if str(theme).lower() == "dark" else "light"
        if normalized == self._settings.theme:
            return
        self._settings.theme = normalized
        self.save()
        self.theme_changed.emit(normalized)

    def toggle_theme(self) -> None:
        self.set_theme("dark" if self._settings.theme == "light" else "light")

    def set_font_scale(self, scale: float) -> None:
        try:
            value = float(scale)
        except (TypeError, ValueError):
            return
        value = float(min(MAX_FONT_SCALE, max(MIN_FONT_SCALE, value)))
        if abs(value - self._settings.font_scale) < 1e-6:
            return
        self._settings.font_scale = value
        self.save()
        self.font_scale_changed.emit(value)

    # ------------------------------------------------------------------
    # Application helpers
    def apply_theme(self, app: QApplication | None = None) -> None:
        """Apply the current theme palette to the ``QApplication``."""

        app = app or QApplication.instance()
        if app is None:
            return
        if self._settings.theme == "dark":
            palette = QPalette()
            palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
            palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
            palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(220, 220, 220))
            palette.setColor(QPalette.ColorRole.ToolTipText, QColor(25, 25, 25))
            palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
            palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
            palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        else:
            palette = app.style().standardPalette()
        app.setPalette(palette)

    def apply_font_scale(self, app: QApplication | None = None) -> None:
        """Scale the application's default font according to settings."""

        app = app or QApplication.instance()
        if app is None:
            return
        default_font = QFont(app.font())
        if self._base_font_point_size is None:
            point_size = default_font.pointSizeF()
            if point_size <= 0:
                point_size = float(default_font.pointSize())
            self._base_font_point_size = point_size or 10.0
        default_font.setPointSizeF(self._base_font_point_size * self._settings.font_scale)
        app.setFont(default_font)


__all__ = ["SettingsService", "UISettings", "DEFAULT_THEME"]

