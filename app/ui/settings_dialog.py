"""Settings dialog housing chat style configuration."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFontComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
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
        self._font_combo: QFontComboBox | None = None
        self._font_size_spin: QDoubleSpinBox | None = None
        self._preview_label: QLabel | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        layout.addWidget(self._build_typography_group())

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
        self._settings.font_family_changed.connect(self._sync_font_family)
        self._settings.font_point_size_changed.connect(self._sync_font_point_size)
        self._settings.font_scale_changed.connect(self._update_font_preview)

        self._sync_font_family(self._settings.font_family)
        self._sync_font_point_size(self._settings.font_point_size)

    # ------------------------------------------------------------------
    def _build_typography_group(self) -> QGroupBox:
        group = QGroupBox("Typography", self)
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(12)

        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._font_combo = QFontComboBox(group)
        self._font_combo.setEditable(False)
        self._font_combo.setToolTip("Choose the font family used throughout the interface.")
        self._font_combo.currentFontChanged.connect(self._on_font_family_selected)
        form_layout.addRow(QLabel("Font family", group), self._font_combo)

        self._font_size_spin = QDoubleSpinBox(group)
        self._font_size_spin.setRange(0.0, 48.0)
        self._font_size_spin.setDecimals(1)
        self._font_size_spin.setSingleStep(0.5)
        self._font_size_spin.setSuffix(" pt")
        self._font_size_spin.setSpecialValueText("System default")
        self._font_size_spin.setToolTip(
            "Specify an absolute base font size, or choose the system default by selecting 0."
        )
        self._font_size_spin.valueChanged.connect(self._on_font_point_size_changed)
        form_layout.addRow(QLabel("Font size", group), self._font_size_spin)

        group_layout.addLayout(form_layout)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        reset_button = QPushButton("Reset to defaults", group)
        reset_button.setAutoDefault(False)
        reset_button.setToolTip("Restore the application font family and size.")
        reset_button.clicked.connect(self._reset_font_preferences)
        button_row.addWidget(reset_button)
        button_row.addStretch(1)
        group_layout.addLayout(button_row)

        preview_caption = QLabel("Preview", group)
        preview_caption.setObjectName("fontPreviewCaption")
        group_layout.addWidget(preview_caption)

        preview_frame = QFrame(group)
        preview_frame.setFrameShape(QFrame.Shape.StyledPanel)
        preview_frame.setFrameShadow(QFrame.Shadow.Plain)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(12, 8, 12, 8)
        self._preview_label = QLabel(
            "The quick brown fox jumps over the lazy dog. 1234567890", preview_frame
        )
        self._preview_label.setWordWrap(True)
        preview_layout.addWidget(self._preview_label)
        group_layout.addWidget(preview_frame)

        return group

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

    def _on_font_family_selected(self, font: QFont) -> None:
        self._settings.set_font_preferences(family=font.family())

    def _on_font_point_size_changed(self, value: float) -> None:
        if value <= 0:
            self._settings.set_font_preferences(point_size=None)
        else:
            self._settings.set_font_preferences(point_size=value)

    def _reset_font_preferences(self) -> None:
        self._settings.set_font_preferences(family=None, point_size=None)

    def _sync_font_family(self, family: str | None) -> None:
        if not self._font_combo:
            return
        target_family = family or QFont().defaultFamily()
        current_family = self._font_combo.currentFont().family()
        if current_family != target_family:
            index = self._font_combo.findText(target_family, Qt.MatchFlag.MatchExactly)
            self._font_combo.blockSignals(True)
            if index >= 0:
                self._font_combo.setCurrentIndex(index)
            else:
                self._font_combo.setCurrentFont(QFont(target_family))
            self._font_combo.blockSignals(False)
        self._update_font_preview()

    def _sync_font_point_size(self, size: float | None) -> None:
        if not self._font_size_spin:
            return
        target = float(size) if size and size > 0 else 0.0
        if abs(self._font_size_spin.value() - target) > 1e-3:
            self._font_size_spin.blockSignals(True)
            self._font_size_spin.setValue(target)
            self._font_size_spin.blockSignals(False)
        self._update_font_preview()

    def _update_font_preview(self, *_args: object) -> None:
        if not (self._preview_label and self._font_combo and self._font_size_spin):
            return
        font = QFont(self._font_combo.currentFont())
        point_size = self._font_size_spin.value()
        if point_size <= 0:
            base_size = font.pointSizeF()
            if base_size <= 0:
                base_size = float(font.pointSize())
            if base_size <= 0:
                base_size = 10.0
        else:
            base_size = point_size
        scaled_size = base_size * self._settings.font_scale
        font.setPointSizeF(scaled_size)
        self._preview_label.setFont(font)


__all__ = ["SettingsDialog"]
