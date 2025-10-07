"""Settings dialog housing chat style configuration."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QColorDialog,
)

from ..services.settings_service import ChatStyleSettings, SettingsService


@dataclass(slots=True)
class _ColorControl:
    label: str
    attribute: str


class ColorSwatchButton(QPushButton):
    """Button showing the currently selected color."""

    def __init__(self, color: str, parent: QDialog | None = None) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(64, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh()

    @property
    def color(self) -> QColor:
        return QColor(self._color)

    def set_color(self, color: str) -> None:
        candidate = QColor(color)
        if candidate.isValid():
            self._color = candidate
            self._refresh()

    def _refresh(self) -> None:
        radius = 8
        self.setStyleSheet(
            f"border-radius: {radius}px; border: 1px solid rgba(0,0,0,0.25);"
            f"background-color: {self._color.name()};"
        )


class SettingsDialog(QDialog):
    """Modal dialog exposing configurable UI settings."""

    _CHAT_CONTROLS = (
        _ColorControl("AI bubble", "ai_bubble_color"),
        _ColorControl("User bubble", "user_bubble_color"),
        _ColorControl("Code background", "code_block_background"),
        _ColorControl("Citation accent", "citation_accent"),
    )

    def __init__(
        self,
        *,
        settings_service: SettingsService,
        parent: QDialog | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self._settings = settings_service
        self._swatches: dict[str, ColorSwatchButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        chat_group = QGroupBox("Chat style", self)
        chat_layout = QFormLayout(chat_group)
        chat_layout.setSpacing(12)
        chat_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        chat_style = self._settings.chat_style
        for control in self._CHAT_CONTROLS:
            swatch = ColorSwatchButton(getattr(chat_style, control.attribute), chat_group)
            swatch.clicked.connect(lambda _=False, attr=control.attribute: self._choose_color(attr))
            self._swatches[control.attribute] = swatch
            wrapper = QHBoxLayout()
            wrapper.setContentsMargins(0, 0, 0, 0)
            wrapper.addWidget(swatch)
            wrapper.addStretch(1)
            chat_layout.addRow(QLabel(control.label, chat_group), wrapper)

        layout.addWidget(chat_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close,
            Qt.Orientation.Horizontal,
            self,
        )
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("Close")
        layout.addWidget(buttons)

        self._settings.chat_style_changed.connect(self._on_chat_style_changed)

    # ------------------------------------------------------------------
    def _choose_color(self, attribute: str) -> None:
        current = self._settings.chat_style
        base = getattr(current, attribute)
        color = QColorDialog.getColor(QColor(base), self, "Select color")
        if not color.isValid():
            return
        kwargs = {attribute: color.name()}
        self._settings.set_chat_style(**kwargs)

    def _on_chat_style_changed(self, style: ChatStyleSettings) -> None:  # pragma: no cover - UI update
        for attribute, swatch in self._swatches.items():
            swatch.set_color(getattr(style, attribute))


__all__ = ["SettingsDialog"]
