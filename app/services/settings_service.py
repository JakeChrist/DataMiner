"""Application settings management for UI preferences."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
import os
from typing import Any, Iterable

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWidgets import QApplication

from ..config import ConfigManager
from ..logging import log_call


logger = logging.getLogger(__name__)


DEFAULT_THEME = "light"
STANDARD_THEMES = {"light", "dark"}
ALL_THEMES = {"light", "dark", "futuristic"}
DEFAULT_DENSITY = "comfortable"
DEFAULT_SPLITTER_SIZES = (280, 720, 360)
MIN_FONT_SCALE = 0.5
MAX_FONT_SCALE = 2.5


_UNSET = object()


@log_call(logger=logger, include_result=True)
def _clamp(value: float, *, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def _normalize_color(value: Any, *, fallback: str) -> str:
    """Return a normalized hex color string or the fallback if invalid."""

    if isinstance(value, str):
        candidate = QColor(value)
    elif isinstance(value, (tuple, list)) and len(value) >= 3:
        candidate = QColor(*value)
    else:
        candidate = QColor()
    if not candidate.isValid():
        candidate = QColor(fallback)
    if not candidate.isValid():
        candidate = QColor("#000000")
    return candidate.name()


@dataclass(slots=True)
class ChatStyleSettings:
    """Themeable colors used by the chat style answer view."""

    ai_bubble_color: str = "#315389"
    ai_text_color: str = "#f2f5f9"
    user_bubble_color: str = "#ffffff"
    user_text_color: str = "#1e2430"
    code_block_background: str = "#1f2530"
    citation_accent: str = "#d8893a"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)  # type: ignore[return-value]


@dataclass(slots=True)
class UISettings:
    """Container for persisted UI settings."""

    theme: str = DEFAULT_THEME
    standard_theme: str = DEFAULT_THEME
    font_scale: float = 1.0
    font_family: str | None = None
    font_point_size: float | None = None
    density: str = DEFAULT_DENSITY
    splitter_sizes: tuple[int, int, int] = DEFAULT_SPLITTER_SIZES
    show_corpus_panel: bool = True
    show_evidence_panel: bool = True
    chat_style: ChatStyleSettings = field(default_factory=ChatStyleSettings)


class SettingsService(QObject):
    """Persist and broadcast UI level settings such as theme and fonts."""

    theme_changed = pyqtSignal(str)
    font_scale_changed = pyqtSignal(float)
    font_family_changed = pyqtSignal(object)
    font_point_size_changed = pyqtSignal(object)
    density_changed = pyqtSignal(str)
    chat_style_changed = pyqtSignal(object)

    @log_call(logger=logger)
    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        super().__init__()
        self._config = config_manager or ConfigManager()
        self._settings = UISettings()
        self._base_font_point_size: float | None = None
        self._glass_translucent: bool = False
        self.reload()
        self._previous_standard_theme = (
            self._settings.standard_theme
            if self._settings.standard_theme in STANDARD_THEMES
            else DEFAULT_THEME
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    @log_call(logger=logger)
    def reload(self) -> None:
        """Reload settings from disk."""

        data = self._config.load()
        if not isinstance(data, dict):
            data = {}
        ui_settings = data.get("ui", {})
        if not isinstance(ui_settings, dict):
            ui_settings = {}
        theme = str(ui_settings.get("theme", DEFAULT_THEME)).lower()
        if theme not in ALL_THEMES:
            theme = DEFAULT_THEME
        standard_theme_value = str(ui_settings.get("standard_theme", theme)).lower()
        if standard_theme_value not in STANDARD_THEMES:
            standard_theme_value = DEFAULT_THEME
        if theme not in ALL_THEMES:
            theme = DEFAULT_THEME
        if theme not in STANDARD_THEMES and standard_theme_value not in STANDARD_THEMES:
            standard_theme_value = DEFAULT_THEME
        font_scale = ui_settings.get("font_scale", 1.0)
        try:
            font_scale = float(font_scale)
        except (TypeError, ValueError):
            font_scale = 1.0
        font_scale = _clamp(font_scale, low=MIN_FONT_SCALE, high=MAX_FONT_SCALE)
        density = str(ui_settings.get("density", DEFAULT_DENSITY)).lower()
        if density not in {"compact", "comfortable"}:
            density = DEFAULT_DENSITY
        splitter = ui_settings.get("splitter_sizes")
        if (
            isinstance(splitter, (list, tuple))
            and len(splitter) == 3
            and all(isinstance(value, (int, float)) for value in splitter)
        ):
            splitter_sizes = tuple(int(max(80, value)) for value in splitter)  # type: ignore[assignment]
        else:
            splitter_sizes = DEFAULT_SPLITTER_SIZES
        show_corpus = bool(ui_settings.get("show_corpus_panel", True))
        show_evidence = bool(ui_settings.get("show_evidence_panel", True))
        family_value = ui_settings.get("font_family")
        if isinstance(family_value, str):
            normalized_family = family_value.strip()
            font_family = normalized_family or None
        else:
            font_family = None
        size_value = ui_settings.get("font_point_size")
        font_point_size: float | None
        try:
            font_point_size = float(size_value)
        except (TypeError, ValueError):
            font_point_size = None
        if font_point_size is not None and font_point_size <= 0:
            font_point_size = None
        chat_data = ui_settings.get("chat_style", {})
        if not isinstance(chat_data, dict):
            chat_data = {}

        def _load_chat_color(*keys: str, fallback: str) -> str:
            for key in keys:
                if key in chat_data:
                    return _normalize_color(chat_data.get(key), fallback=fallback)
            return fallback

        defaults = ChatStyleSettings()
        chat_style = ChatStyleSettings(
            ai_bubble_color=_load_chat_color(
                "ai_bubble_color", "ai_bubble", fallback=defaults.ai_bubble_color
            ),
            ai_text_color=_load_chat_color(
                "ai_text_color",
                "ai_text",
                "answer_text_color",
                fallback=defaults.ai_text_color,
            ),
            user_bubble_color=_load_chat_color(
                "user_bubble_color", "user_bubble", fallback=defaults.user_bubble_color
            ),
            user_text_color=_load_chat_color(
                "user_text_color",
                "user_text",
                "question_text_color",
                fallback=defaults.user_text_color,
            ),
            code_block_background=_load_chat_color(
                "code_block_background", "code_background", fallback=defaults.code_block_background
            ),
            citation_accent=_load_chat_color(
                "citation_accent", fallback=defaults.citation_accent
            ),
        )
        self._settings = UISettings(
            theme=theme,
            standard_theme=standard_theme_value,
            font_scale=font_scale,
            font_family=font_family,
            font_point_size=font_point_size,
            density=density,
            splitter_sizes=splitter_sizes,
            show_corpus_panel=show_corpus,
            show_evidence_panel=show_evidence,
            chat_style=chat_style,
        )
        self._base_font_point_size = None
        if hasattr(self, "_previous_standard_theme"):
            self._previous_standard_theme = (
                self._settings.standard_theme
                if self._settings.standard_theme in STANDARD_THEMES
                else DEFAULT_THEME
            )

    @log_call(logger=logger)
    def save(self) -> None:
        """Persist the current settings to disk."""

        data = self._config.load()
        if not isinstance(data, dict):
            data = {}
        ui_data = data.get("ui")
        if not isinstance(ui_data, dict):
            ui_data = {}
        ui_data.update(
            {
                "theme": self._settings.theme,
                "font_scale": self._settings.font_scale,
                "font_family": self._settings.font_family,
                "font_point_size": self._settings.font_point_size,
                "density": self._settings.density,
                "splitter_sizes": list(self._settings.splitter_sizes),
                "show_corpus_panel": self._settings.show_corpus_panel,
                "show_evidence_panel": self._settings.show_evidence_panel,
                "chat_style": self._settings.chat_style.as_dict(),
                "standard_theme": self._settings.standard_theme,
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
    def glass_translucency_enabled(self) -> bool:
        """Return ``True`` when the glass theme can render translucency."""

        return self._glass_translucent

    @property
    def standard_theme(self) -> str:
        return self._settings.standard_theme

    @property
    def last_standard_theme(self) -> str:
        """Return the cached non-futuristic theme used for fallbacks."""

        if self._previous_standard_theme not in STANDARD_THEMES:
            self._previous_standard_theme = DEFAULT_THEME
        return self._previous_standard_theme

    @property
    def font_scale(self) -> float:
        return self._settings.font_scale

    @property
    def font_family(self) -> str | None:
        return self._settings.font_family

    @property
    def font_point_size(self) -> float | None:
        return self._settings.font_point_size

    @property
    def density(self) -> str:
        return self._settings.density

    @property
    def splitter_sizes(self) -> tuple[int, int, int]:
        return self._settings.splitter_sizes

    @property
    def show_corpus_panel(self) -> bool:
        return self._settings.show_corpus_panel

    @property
    def show_evidence_panel(self) -> bool:
        return self._settings.show_evidence_panel

    @property
    def chat_style(self) -> ChatStyleSettings:
        style = self._settings.chat_style
        return ChatStyleSettings(
            ai_bubble_color=style.ai_bubble_color,
            ai_text_color=style.ai_text_color,
            user_bubble_color=style.user_bubble_color,
            user_text_color=style.user_text_color,
            code_block_background=style.code_block_background,
            citation_accent=style.citation_accent,
        )

    # ------------------------------------------------------------------
    # Mutators
    @log_call(logger=logger)
    def set_theme(self, theme: str) -> None:
        normalized = str(theme).lower()
        if normalized not in ALL_THEMES:
            normalized = DEFAULT_THEME
        if normalized in STANDARD_THEMES:
            self._settings.standard_theme = normalized
            self._previous_standard_theme = normalized
        if normalized == self._settings.theme:
            return
        self._settings.theme = normalized
        self.save()
        logger.info("Theme changed", extra={"theme": normalized})
        self.theme_changed.emit(normalized)

    @log_call(logger=logger)
    def toggle_theme(self) -> None:
        current = self._settings.theme
        if current == "futuristic":
            target = "dark" if self.last_standard_theme == "light" else "light"
        else:
            target = "dark" if current == "light" else "light"
        self.set_theme(target)

    @log_call(logger=logger)
    def use_futuristic_theme(self, enabled: bool) -> None:
        """Enable or disable the layered glass theme."""

        if enabled:
            self.set_theme("futuristic")
        else:
            self.set_theme(self.last_standard_theme)

    @log_call(logger=logger)
    def set_font_scale(self, scale: float) -> None:
        try:
            value = float(scale)
        except (TypeError, ValueError):
            return
        value = _clamp(value, low=MIN_FONT_SCALE, high=MAX_FONT_SCALE)
        if abs(value - self._settings.font_scale) < 1e-6:
            return
        self._settings.font_scale = value
        self.save()
        logger.info("Font scale changed", extra={"font_scale": value})
        self.font_scale_changed.emit(value)
        self.apply_font_preferences()

    @log_call(logger=logger)
    def set_font_preferences(
        self,
        *,
        family: str | None | object = _UNSET,
        point_size: float | None | object = _UNSET,
    ) -> None:
        self._ensure_base_font_point_size()
        current_family = self._settings.font_family
        current_size = self._settings.font_point_size
        new_family = current_family
        new_size = current_size

        if family is not _UNSET:
            if isinstance(family, QFont):
                normalized_family = family.family().strip() or None
            elif isinstance(family, str):
                normalized_family = family.strip() or None
            elif family is None:
                normalized_family = None
            else:
                normalized_family = None
            new_family = normalized_family

        if point_size is not _UNSET:
            if point_size is None:
                normalized_size = None
            else:
                try:
                    numeric = float(point_size)
                except (TypeError, ValueError):
                    return
                if numeric <= 0:
                    normalized_size = None
                else:
                    normalized_size = numeric
            new_size = normalized_size

        if new_family == current_family and new_size == current_size:
            return

        self._settings.font_family = new_family
        self._settings.font_point_size = new_size
        self.save()
        if new_family != current_family:
            self.font_family_changed.emit(new_family)
        if new_size != current_size:
            self.font_point_size_changed.emit(new_size)
        self.apply_font_preferences()

    @log_call(logger=logger)
    def set_density(self, density: str) -> None:
        normalized = str(density).lower()
        if normalized not in {"compact", "comfortable"}:
            return
        if normalized == self._settings.density:
            return
        self._settings.density = normalized
        self.save()
        logger.info("Density changed", extra={"density": normalized})
        self.density_changed.emit(normalized)

    @log_call(logger=logger)
    def set_splitter_sizes(self, sizes: Iterable[int]) -> None:
        values = [int(max(80, value)) for value in sizes]
        if len(values) != 3:
            return
        if tuple(values) == self._settings.splitter_sizes:
            return
        self._settings.splitter_sizes = tuple(values)
        self.save()

    @log_call(logger=logger)
    def set_show_corpus_panel(self, visible: bool) -> None:
        value = bool(visible)
        if value == self._settings.show_corpus_panel:
            return
        self._settings.show_corpus_panel = value
        self.save()

    @log_call(logger=logger)
    def set_show_evidence_panel(self, visible: bool) -> None:
        value = bool(visible)
        if value == self._settings.show_evidence_panel:
            return
        self._settings.show_evidence_panel = value
        self.save()

    @log_call(logger=logger)
    def set_chat_style(
        self,
        *,
        ai_bubble_color: str | None = None,
        ai_text_color: str | None = None,
        user_bubble_color: str | None = None,
        user_text_color: str | None = None,
        code_block_background: str | None = None,
        citation_accent: str | None = None,
    ) -> None:
        current = self._settings.chat_style
        updated = ChatStyleSettings(
            ai_bubble_color=_normalize_color(
                ai_bubble_color, fallback=current.ai_bubble_color
            ),
            ai_text_color=_normalize_color(
                ai_text_color, fallback=current.ai_text_color
            ),
            user_bubble_color=_normalize_color(
                user_bubble_color, fallback=current.user_bubble_color
            ),
            user_text_color=_normalize_color(
                user_text_color, fallback=current.user_text_color
            ),
            code_block_background=_normalize_color(
                code_block_background, fallback=current.code_block_background
            ),
            citation_accent=_normalize_color(
                citation_accent, fallback=current.citation_accent
            ),
        )
        if updated == current:
            return
        self._settings.chat_style = updated
        self.save()
        self.chat_style_changed.emit(self.chat_style)

    # ------------------------------------------------------------------
    # Application helpers
    def apply_theme(self, app: QApplication | None = None) -> None:
        """Apply the current theme palette to the ``QApplication``."""

        app = app or QApplication.instance()
        if app is None:
            return
        if self._settings.theme == "futuristic":
            palette, stylesheet = self._build_futuristic_theme(app)
        else:
            palette, stylesheet = self._build_standard_theme(app, dark=self._settings.theme == "dark")
        app.setPalette(palette)
        app.setStyleSheet(stylesheet)

    # ------------------------------------------------------------------
    def _build_standard_theme(
        self, app: QApplication, *, dark: bool
    ) -> tuple[QPalette, str]:
        self._glass_translucent = False
        palette = QPalette()
        if dark:
            window = QColor("#10131a")
            surface = QColor("#1a1f29")
            border = QColor("#2b3240")
            text = QColor("#f2f5f9")
            muted = QColor("#bac4d4")
            accent = QColor("#d8893a")
            success = QColor("#56c28c")
            warning = QColor("#f1c04e")
            error = QColor("#f06c65")
        else:
            window = QColor("#f4f6f9")
            surface = QColor("#ffffff")
            border = QColor("#d9dee7")
            text = QColor("#1e2430")
            muted = QColor("#566072")
            accent = QColor("#b96a1d")
            success = QColor("#2f8f58")
            warning = QColor("#d5881a")
            error = QColor("#c94b45")

        palette.setColor(QPalette.ColorRole.Window, window)
        palette.setColor(QPalette.ColorRole.WindowText, text)
        palette.setColor(QPalette.ColorRole.Base, surface)
        palette.setColor(QPalette.ColorRole.AlternateBase, surface)
        palette.setColor(QPalette.ColorRole.ToolTipBase, surface)
        palette.setColor(QPalette.ColorRole.ToolTipText, text)
        palette.setColor(QPalette.ColorRole.Text, text)
        palette.setColor(QPalette.ColorRole.PlaceholderText, muted)
        palette.setColor(QPalette.ColorRole.Button, surface)
        palette.setColor(QPalette.ColorRole.ButtonText, text)
        palette.setColor(QPalette.ColorRole.Highlight, accent)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#0b0d11"))
        palette.setColor(QPalette.ColorRole.BrightText, error)
        palette.setColor(QPalette.ColorRole.Link, accent)
        palette.setColor(QPalette.ColorRole.LinkVisited, accent.darker(120))
        palette.setColor(QPalette.ColorRole.Light, surface.lighter(105))
        palette.setColor(QPalette.ColorRole.Midlight, border)
        palette.setColor(QPalette.ColorRole.Mid, border)
        palette.setColor(QPalette.ColorRole.Dark, border)
        palette.setColor(QPalette.ColorRole.Shadow, border.darker(140))

        accent_hex = accent.name()
        accent_hover_hex = accent.darker(110).name()
        text_hex = text.name()
        muted_hex = muted.name()
        border_hex = border.name()
        surface_hex = surface.name()
        window_hex = window.name()
        success_hex = success.name()
        warning_hex = warning.name()
        error_hex = error.name()
        tooltip_background_hex = (
            surface.lighter(115).name() if dark else surface.darker(105).name()
        )
        tooltip_font = QFont(app.font())
        tooltip_font_size = tooltip_font.pointSizeF()
        if tooltip_font_size <= 0:
            tooltip_font_size = float(tooltip_font.pointSize())
        if tooltip_font_size <= 0:
            tooltip_font_size = 10.0
        tooltip_font_size_css = f"{tooltip_font_size:.1f}pt"

        stylesheet = f"""
            QWidget {{
                background-color: {window_hex};
                color: {text_hex};
            }}
            QToolTip {{
                background-color: {tooltip_background_hex};
                color: {text_hex};
                border: 1px solid {border_hex};
                border-radius: 6px;
                padding: 6px 8px;
                font-size: {tooltip_font_size_css};
            }}
            QFrame#chatBubble_user, QFrame#chatBubble_assistant {{
                border: 1px solid {border_hex};
            }}
            QLabel#bubbleMeta, QLabel#typingIndicator {{
                color: {muted_hex};
            }}
            QFrame#planSection {{
                background-color: {surface_hex};
                border-radius: 10px;
                border: 1px solid {border_hex};
                padding: 8px 12px;
            }}
            QToolButton#planChip {{
                background-color: {surface_hex};
                border-radius: 14px;
                border: 1px solid {border_hex};
                padding: 2px 10px;
            }}
            QToolButton#planChip:checked {{
                background-color: {accent_hex};
                color: #0b0d11;
                border-color: {accent_hex};
            }}
            QFrame#codeBlock {{
                border: 1px solid {border_hex};
            }}
            QPlainTextEdit#codeEditor {{
                color: {text_hex};
            }}
            QTextBrowser {{
                background-color: transparent;
                border: none;
            }}
            QPushButton, QToolButton {{
                background-color: {surface_hex};
                border: 1px solid {border_hex};
                border-radius: 16px;
                padding: 4px 12px;
            }}
            QPushButton:disabled, QToolButton:disabled {{
                color: {muted_hex};
                border-color: {border_hex};
                background-color: {surface_hex};
            }}
            QPushButton:hover, QToolButton:hover {{
                border-color: {accent_hex};
            }}
            QPushButton:pressed, QToolButton:pressed {{
                background-color: {accent_hex};
                color: #0b0d11;
            }}
            QPushButton#accent, QToolButton#accent {{
                background-color: {accent_hex};
                color: #0b0d11;
                border: none;
            }}
            QPushButton#accent:hover, QToolButton#accent:hover {{
                background-color: {accent_hover_hex};
            }}
            QLabel#statusPill {{
                border-radius: 14px;
                padding: 2px 10px;
                border: 1px solid {border_hex};
                background-color: {surface_hex};
                color: {muted_hex};
            }}
            QLabel#statusPill[state="info"] {{
                background-color: {accent_hex};
                color: #0b0d11;
                border-color: {accent_hex};
            }}
            QLabel#statusPill[state="connected"] {{
                background-color: {success_hex};
                color: #0b0d11;
                border-color: {success_hex};
            }}
            QLabel#statusPill[state="warning"] {{
                background-color: {warning_hex};
                color: #0b0d11;
                border-color: {warning_hex};
            }}
            QLabel#statusPill[state="error"] {{
                background-color: {error_hex};
                color: #0b0d11;
                border-color: {error_hex};
            }}
        """
        return palette, stylesheet

    def _build_futuristic_theme(self, app: QApplication) -> tuple[QPalette, str]:
        palette = QPalette()
        window = QColor("#05070d")
        surface = QColor("#0f1a2a")
        chrome = QColor("#18263b")
        border = QColor("#233145")
        text = QColor("#edf6ff")
        muted = QColor("#8fa4c9")
        accent_cyan = QColor("#52e0ff")
        accent_blue = QColor("#4f75ff")
        accent_violet = QColor("#b17dff")
        warm_warning = QColor("#ff8c5c")
        success = QColor("#63ffc8")
        warning = QColor("#ffc35b")
        error = QColor("#ff6b9d")

        palette.setColor(QPalette.ColorRole.Window, window)
        palette.setColor(QPalette.ColorRole.WindowText, text)
        palette.setColor(QPalette.ColorRole.Base, surface)
        palette.setColor(QPalette.ColorRole.AlternateBase, surface)
        palette.setColor(QPalette.ColorRole.ToolTipBase, chrome)
        palette.setColor(QPalette.ColorRole.ToolTipText, text)
        palette.setColor(QPalette.ColorRole.Text, text)
        palette.setColor(QPalette.ColorRole.PlaceholderText, muted)
        palette.setColor(QPalette.ColorRole.Button, chrome)
        palette.setColor(QPalette.ColorRole.ButtonText, text)
        palette.setColor(QPalette.ColorRole.Highlight, accent_cyan)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#051119"))
        palette.setColor(QPalette.ColorRole.BrightText, error)
        palette.setColor(QPalette.ColorRole.Link, accent_blue)
        palette.setColor(QPalette.ColorRole.LinkVisited, accent_violet)
        palette.setColor(QPalette.ColorRole.Light, chrome.lighter(110))
        palette.setColor(QPalette.ColorRole.Midlight, border)
        palette.setColor(QPalette.ColorRole.Mid, border)
        palette.setColor(QPalette.ColorRole.Dark, border.darker(140))
        palette.setColor(QPalette.ColorRole.Shadow, QColor(0, 0, 0, 180))

        translucent = _supports_translucency(app)
        self._glass_translucent = translucent
        glass_rgba = "rgba(18, 30, 49, 0.72)" if translucent else "#121e31"
        film_rgba = "rgba(18, 30, 49, 0.55)" if translucent else "#101a2b"
        chrome_rgba = "rgba(24, 38, 59, 0.92)" if translucent else chrome.name()
        toolbar_rgba = "rgba(8, 13, 22, 0.85)" if translucent else "#080d16"
        window_rgba = "rgba(5, 7, 13, 0.55)" if translucent else window.name()
        selection_rgba = "rgba(82, 224, 255, 0.25)"
        chrome_hover_rgba = "rgba(28, 44, 66, 0.95)" if translucent else chrome.lighter(110).name()

        text_hex = text.name()
        muted_hex = muted.name()
        border_hex = border.name()
        success_hex = success.name()
        warning_hex = warning.name()
        error_hex = error.name()
        accent_cyan_hex = accent_cyan.name()
        accent_blue_hex = accent_blue.name()
        accent_violet_hex = accent_violet.name()
        warm_warning_hex = warm_warning.name()

        tooltip_font = QFont(app.font())
        tooltip_font_size = tooltip_font.pointSizeF()
        if tooltip_font_size <= 0:
            tooltip_font_size = float(tooltip_font.pointSize())
        if tooltip_font_size <= 0:
            tooltip_font_size = 10.0
        tooltip_font_size_css = f"{tooltip_font_size:.1f}pt"

        stylesheet = f"""
            QWidget {{
                background-color: {window_rgba};
                color: {text_hex};
            }}
            QMainWindow, QDialog {{
                background-color: {window_rgba};
                color: {text_hex};
            }}
            QToolTip {{
                background-color: {chrome_rgba};
                color: {text_hex};
                border: 1px solid {border_hex};
                border-radius: 10px;
                padding: 8px 10px;
                font-size: {tooltip_font_size_css};
            }}
            QMenu, QToolBar {{
                background-color: {film_rgba};
                border: 1px solid {border_hex};
                border-radius: 14px;
            }}
            QToolBar {{
                background-color: {toolbar_rgba};
                spacing: 6px;
            }}
            QStatusBar {{
                background-color: {toolbar_rgba};
                border-top: 1px solid {border_hex};
            }}
            QGroupBox {{
                background-color: {glass_rgba};
                border: 1px solid {border_hex};
                border-radius: 18px;
                margin-top: 18px;
                padding: 12px 14px 14px 14px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 6px;
                background-color: transparent;
                color: {muted_hex};
            }}
            QFrame#planSection, QFrame#chatBubble_user, QFrame#chatBubble_assistant,
            QFrame#codeBlock {{
                background-color: {glass_rgba};
                border-radius: 18px;
                border: 1px solid {border_hex};
            }}
            QWidget#corpusPane, QWidget#chatPane, QWidget#evidencePanel {{
                background-color: {glass_rgba};
                border: 1px solid {border_hex};
                border-radius: 22px;
            }}
            QLabel#bubbleMeta, QLabel#typingIndicator {{
                color: {muted_hex};
            }}
            QTextBrowser, QPlainTextEdit, QTreeWidget, QTableView, QListView, QListWidget {{
                background-color: {glass_rgba};
                border: 1px solid {border_hex};
                border-radius: 14px;
                selection-background-color: {selection_rgba};
                selection-color: {text_hex};
            }}
            QHeaderView::section {{
                background-color: {chrome_rgba};
                border: 0px;
                border-bottom: 1px solid {border_hex};
                color: {muted_hex};
                padding: 6px 8px;
            }}
            QPushButton, QToolButton {{
                background-color: {chrome_rgba};
                border: 1px solid {border_hex};
                border-radius: 18px;
                padding: 6px 14px;
                color: {text_hex};
            }}
            QPushButton:hover, QToolButton:hover {{
                border-color: {accent_cyan_hex};
                background-color: {chrome_hover_rgba};
            }}
            QPushButton:pressed, QToolButton:pressed {{
                background-color: {accent_cyan_hex};
                color: #041018;
                border-color: {accent_cyan_hex};
            }}
            QPushButton:disabled, QToolButton:disabled {{
                color: {muted_hex};
                border-color: {border_hex};
            }}
            QPushButton:focus, QToolButton:focus {{
                border: 2px solid {accent_cyan_hex};
                padding: 5px 13px;
                background-color: {chrome_hover_rgba};
            }}
            QPushButton#accent, QToolButton#accent {{
                background-color: {accent_blue_hex};
                border: none;
                color: #05070d;
            }}
            QPushButton#accent:hover, QToolButton#accent:hover {{
                background-color: {accent_violet_hex};
                color: {text_hex};
            }}
            QPushButton#warn, QToolButton#warn {{
                background-color: {warm_warning_hex};
                color: #05070d;
            }}
            QPushButton#warn:hover, QToolButton#warn:hover {{
                border: 2px solid {warm_warning_hex};
                padding: 5px 13px;
            }}
            QScrollBar::handle {{
                background: {accent_blue_hex};
                border-radius: 8px;
                min-height: 24px;
            }}
            QScrollBar::handle:hover {{
                background: {accent_cyan_hex};
            }}
            QScrollBar::add-page, QScrollBar::sub-page {{
                background: transparent;
            }}
            QTreeView::item:selected, QListView::item:selected, QTableView::item:selected {{
                background-color: {selection_rgba};
                color: {text_hex};
            }}
            QLabel#statusPill {{
                border-radius: 16px;
                padding: 4px 12px;
                border: 1px solid {border_hex};
                background-color: {glass_rgba};
                color: {muted_hex};
            }}
            QLabel#statusPill[state="info"] {{
                background-color: {accent_cyan_hex};
                color: #041018;
                border-color: {accent_cyan_hex};
            }}
            QLabel#statusPill[state="connected"] {{
                background-color: {success_hex};
                color: #041018;
                border-color: {success_hex};
            }}
            QLabel#statusPill[state="warning"] {{
                background-color: {warning_hex};
                color: #041018;
                border-color: {warning_hex};
            }}
            QLabel#statusPill[state="error"] {{
                background-color: {error_hex};
                color: #041018;
                border-color: {error_hex};
            }}
            QWidget#toast {{
                background-color: {chrome_rgba};
                border: 1px solid {border_hex};
                border-radius: 16px;
                color: {text_hex};
            }}
            QProgressBar {{
                background-color: {film_rgba};
                border: 1px solid {border_hex};
                border-radius: 10px;
                color: {text_hex};
            }}
            QProgressBar::chunk {{
                background: QLinearGradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {accent_cyan_hex}, stop:0.5 {accent_blue_hex}, stop:1 {accent_violet_hex});
                border-radius: 8px;
            }}
        """
        return palette, stylesheet

    def apply_font_preferences(self, app: QApplication | None = None) -> None:
        """Apply the configured font family and size to the application."""

        app = app or QApplication.instance()
        if app is None:
            return
        default_font = QFont(app.font())
        if self._settings.font_family:
            default_font.setFamily(self._settings.font_family)

        base_point_size = self._settings.font_point_size
        if base_point_size is None:
            self._ensure_base_font_point_size(default_font)
            point_size = default_font.pointSizeF()
            if point_size <= 0:
                point_size = float(default_font.pointSize())
            if point_size <= 0:
                point_size = 10.0
            self._base_font_point_size = self._base_font_point_size or point_size
            base_point_size = self._base_font_point_size
        else:
            base_point_size = float(base_point_size)

        default_font.setPointSizeF(base_point_size * self._settings.font_scale)
        app.setFont(default_font)

    def apply_font_scale(self, app: QApplication | None = None) -> None:
        """Backward compatible wrapper to apply font preferences."""

        self.apply_font_preferences(app)

    def _ensure_base_font_point_size(self, font: QFont | None = None) -> None:
        if self._base_font_point_size is not None:
            return
        if font is None:
            app = QApplication.instance()
            if app is not None:
                font = QFont(app.font())
            else:
                font = QFont()
        point_size = font.pointSizeF()
        if point_size <= 0:
            point_size = float(font.pointSize())
        if point_size <= 0:
            point_size = 10.0
        self._base_font_point_size = point_size


__all__ = ["SettingsService", "UISettings", "DEFAULT_THEME"]

def _supports_translucency(app: QApplication) -> bool:
    """Best effort check for platform translucency support."""

    override = os.environ.get("DATAMINER_DISABLE_GLASS", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return False
    platform = (app.platformName() or "").lower()
    if platform in {"offscreen", "minimal", "minimalistic"}:
        return False
    screen = app.primaryScreen()
    if screen is None:
        return False
    try:
        return screen.depth() >= 24
    except Exception:  # pragma: no cover - defensive fallback
        return True

