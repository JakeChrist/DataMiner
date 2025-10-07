"""Application settings management for UI preferences."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWidgets import QApplication

from ..config import ConfigManager
from ..logging import log_call


logger = logging.getLogger(__name__)


DEFAULT_THEME = "light"
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
    user_bubble_color: str = "#ffffff"
    code_block_background: str = "#1f2530"
    citation_accent: str = "#d8893a"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)  # type: ignore[return-value]


@dataclass(slots=True)
class UISettings:
    """Container for persisted UI settings."""

    theme: str = DEFAULT_THEME
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
        self.reload()

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
        if theme not in {"light", "dark"}:
            theme = DEFAULT_THEME
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
        defaults = ChatStyleSettings()
        chat_style = ChatStyleSettings(
            ai_bubble_color=_normalize_color(
                chat_data.get("ai_bubble"), fallback=defaults.ai_bubble_color
            ),
            user_bubble_color=_normalize_color(
                chat_data.get("user_bubble"), fallback=defaults.user_bubble_color
            ),
            code_block_background=_normalize_color(
                chat_data.get("code_background"), fallback=defaults.code_block_background
            ),
            citation_accent=_normalize_color(
                chat_data.get("citation_accent"), fallback=defaults.citation_accent
            ),
        )
        self._settings = UISettings(
            theme=theme,
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
            user_bubble_color=style.user_bubble_color,
            code_block_background=style.code_block_background,
            citation_accent=style.citation_accent,
        )

    # ------------------------------------------------------------------
    # Mutators
    @log_call(logger=logger)
    def set_theme(self, theme: str) -> None:
        normalized = "dark" if str(theme).lower() == "dark" else "light"
        if normalized == self._settings.theme:
            return
        self._settings.theme = normalized
        self.save()
        logger.info("Theme changed", extra={"theme": normalized})
        self.theme_changed.emit(normalized)

    @log_call(logger=logger)
    def toggle_theme(self) -> None:
        self.set_theme("dark" if self._settings.theme == "light" else "light")

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
        user_bubble_color: str | None = None,
        code_block_background: str | None = None,
        citation_accent: str | None = None,
    ) -> None:
        current = self._settings.chat_style
        updated = ChatStyleSettings(
            ai_bubble_color=_normalize_color(
                ai_bubble_color, fallback=current.ai_bubble_color
            ),
            user_bubble_color=_normalize_color(
                user_bubble_color, fallback=current.user_bubble_color
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
        palette = QPalette()
        if self._settings.theme == "dark":
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
        app.setPalette(palette)

        accent_hex = accent.name()
        text_hex = text.name()
        muted_hex = muted.name()
        border_hex = border.name()
        surface_hex = surface.name()
        window_hex = window.name()
        success_hex = success.name()
        warning_hex = warning.name()
        error_hex = error.name()

        stylesheet = f"""
            QWidget {{
                background-color: {window_hex};
                color: {text_hex};
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
                background-color: {accent.darker(110).name()};
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
        app.setStyleSheet(stylesheet)

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

