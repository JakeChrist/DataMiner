"""Evidence panel widget with scope controls and source preview."""

from __future__ import annotations

from dataclasses import dataclass
import html
from typing import Any, Iterable

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


@dataclass
class EvidenceRecord:
    """Normalized representation of a citation entry."""

    identifier: str
    label: str
    snippet_html: str
    metadata_text: str
    path: str | None
    raw: Any
    state: str = "include"

    def copy_with_state(self, state: str) -> "EvidenceRecord":
        updated = EvidenceRecord(
            identifier=self.identifier,
            label=self.label,
            snippet_html=self.snippet_html,
            metadata_text=self.metadata_text,
            path=self.path,
            raw=self.raw,
            state=state,
        )
        return updated


class _EvidenceRow(QFrame):
    """Row widget housing include/exclude toggles for an evidence item."""

    state_changed = pyqtSignal(str)

    def __init__(self, record: EvidenceRecord, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.record = record
        self.setObjectName("evidenceRow")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self.label = QLabel(record.label, self)
        self.label.setWordWrap(True)
        layout.addWidget(self.label, 1)

        self.include_button = QToolButton(self)
        self.include_button.setObjectName("includeButton")
        self.include_button.setText("Include")
        self.include_button.setCheckable(True)
        self.include_button.setChecked(record.state == "include")
        self.include_button.toggled.connect(self._on_include_toggled)
        layout.addWidget(self.include_button)

        self.exclude_button = QToolButton(self)
        self.exclude_button.setObjectName("excludeButton")
        self.exclude_button.setText("Exclude")
        self.exclude_button.setCheckable(True)
        self.exclude_button.setChecked(record.state == "exclude")
        self.exclude_button.toggled.connect(self._on_exclude_toggled)
        layout.addWidget(self.exclude_button)

    # ------------------------------------------------------------------
    def set_state(self, state: str) -> None:
        with _SignalBlocker(self.include_button), _SignalBlocker(self.exclude_button):
            self.include_button.setChecked(state == "include")
            self.exclude_button.setChecked(state == "exclude")
        self.record = self.record.copy_with_state(state)

    def _on_include_toggled(self, checked: bool) -> None:
        if checked:
            if self.exclude_button.isChecked():
                self.exclude_button.setChecked(False)
            state = "include"
        else:
            state = "neutral" if not self.exclude_button.isChecked() else "exclude"
        self.record = self.record.copy_with_state(state)
        self.state_changed.emit(state)

    def _on_exclude_toggled(self, checked: bool) -> None:
        if checked:
            if self.include_button.isChecked():
                self.include_button.setChecked(False)
            state = "exclude"
        else:
            state = "include" if self.include_button.isChecked() else "neutral"
        self.record = self.record.copy_with_state(state)
        self.state_changed.emit(state)


class _SignalBlocker:
    """Context manager temporarily blocking signals on a QObject."""

    def __init__(self, widget: QWidget) -> None:
        self.widget = widget
        self._previous = False

    def __enter__(self) -> None:
        self._previous = self.widget.blockSignals(True)

    def __exit__(self, *_exc: object) -> None:
        self.widget.blockSignals(self._previous)


class EvidencePanel(QWidget):
    """Composite widget displaying citations with scope controls."""

    scope_changed = pyqtSignal(list, list)
    evidence_selected = pyqtSignal(int, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records: list[EvidenceRecord] = []
        self._suppress_scope = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self._list = QListWidget(self)
        self._list.setObjectName("evidenceList")
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.currentRowChanged.connect(self._on_row_selected)
        layout.addWidget(self._list, 1)

        self._preview = QTextBrowser(self)
        self._preview.setObjectName("evidencePreview")
        self._preview.setOpenExternalLinks(True)
        self._preview.setPlaceholderText("Select evidence to see highlighted passages.")
        layout.addWidget(self._preview, 2)

        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.setSpacing(6)

        self._metadata_label = QLabel("No evidence selected.", self)
        self._metadata_label.setObjectName("evidenceMetadata")
        self._metadata_label.setWordWrap(True)
        meta_row.addWidget(self._metadata_label, 1)

        self._open_button = QPushButton("Open in default app", self)
        self._open_button.setObjectName("openInAppButton")
        self._open_button.clicked.connect(self._open_in_app)
        self._open_button.setEnabled(False)
        meta_row.addWidget(self._open_button)

        layout.addLayout(meta_row)

        self._empty_state()

    # ------------------------------------------------------------------
    @property
    def evidence_items(self) -> list[EvidenceRecord]:
        return list(self._records)

    @property
    def evidence_count(self) -> int:
        return len(self._records)

    @property
    def selected_index(self) -> int | None:
        row = self._list.currentRow()
        return row if row >= 0 else None

    def selected_record(self) -> EvidenceRecord | None:
        index = self.selected_index
        if index is None:
            return None
        try:
            return self._records[index]
        except IndexError:  # pragma: no cover - defensive guard
            return None

    @property
    def current_scope(self) -> dict[str, list[str]]:
        include = [record.identifier for record in self._records if record.state == "include"]
        exclude = [record.identifier for record in self._records if record.state == "exclude"]
        return {"include": include, "exclude": exclude}

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._records.clear()
        self._list.clear()
        self._empty_state()

    def set_evidence(self, citations: Iterable[Any], *, emit_scope: bool = False) -> None:
        self._suppress_scope = True
        try:
            self._records = [self._normalize_citation(index, raw) for index, raw in enumerate(citations, start=1)]
        finally:
            self._suppress_scope = False
        self._populate_list()
        if self._records:
            self._list.setCurrentRow(0)
            self._update_preview(0)
        else:
            self._empty_state()
        if emit_scope:
            scope = self.current_scope
            self.scope_changed.emit(scope["include"], scope["exclude"])

    def select_index(self, index: int) -> None:
        if 0 <= index < self._list.count():
            self._list.setCurrentRow(index)

    # ------------------------------------------------------------------
    def _empty_state(self) -> None:
        self._preview.setHtml("<p>No evidence available.</p>")
        self._metadata_label.setText("No evidence selected.")
        self._open_button.setEnabled(False)

    def _populate_list(self) -> None:
        self._list.clear()
        for record in self._records:
            item = QListWidgetItem(self._list)
            widget = _EvidenceRow(record, self._list)
            widget.state_changed.connect(lambda _state, rec=record: self._on_state_changed(rec, _state))
            item.setSizeHint(widget.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, widget)

    def _normalize_citation(self, index: int, citation: Any) -> EvidenceRecord:
        if isinstance(citation, str):
            label = f"[{index}] {citation}"
            snippet = html.escape(citation)
            metadata = ""
            path = None
        elif isinstance(citation, dict):
            source = str(citation.get("source") or citation.get("title") or citation.get("document") or f"Source {index}").strip()
            label = f"[{index}] {source}" if source else f"[{index}] Source"
            snippet_raw = citation.get("snippet") or citation.get("highlight") or citation.get("text") or ""
            snippet = snippet_raw if isinstance(snippet_raw, str) else html.escape(str(snippet_raw))
            if "<" not in snippet:
                snippet = html.escape(snippet)
            location_parts: list[str] = []
            page = citation.get("page") or citation.get("page_number")
            if page:
                location_parts.append(f"Page {page}")
            section = citation.get("section") or citation.get("heading")
            if section:
                location_parts.append(str(section))
            metadata = " | ".join(location_parts)
            path = citation.get("path") or citation.get("file_path")
            if isinstance(path, str) and not path.strip():
                path = None
        else:
            label = f"[{index}] {str(citation)}"
            snippet = html.escape(str(citation))
            metadata = ""
            path = None
        identifier = self._build_identifier(citation, fallback=f"citation-{index}")
        metadata_text = metadata or "Location unknown"
        return EvidenceRecord(
            identifier=identifier,
            label=label,
            snippet_html=snippet,
            metadata_text=metadata_text,
            path=path,
            raw=citation,
            state="include",
        )

    def _build_identifier(self, citation: Any, *, fallback: str) -> str:
        if isinstance(citation, dict):
            for key in ("id", "document_id", "source", "path", "file_path"):
                value = citation.get(key)
                if value:
                    return str(value)
        if isinstance(citation, str) and citation.strip():
            return citation.strip()
        return fallback

    def _on_row_selected(self, row: int) -> None:
        if not (0 <= row < len(self._records)):
            self._empty_state()
            return
        self._update_preview(row)
        record = self._records[row]
        self.evidence_selected.emit(row, record.identifier)

    def _update_preview(self, index: int) -> None:
        if not (0 <= index < len(self._records)):
            self._empty_state()
            return
        record = self._records[index]
        html_content = record.snippet_html
        if "<" not in html_content:
            html_content = html.escape(html_content)
        self._preview.setHtml(f"<p>{html_content}</p>")
        self._metadata_label.setText(record.metadata_text)
        self._open_button.setEnabled(bool(record.path))

    def _on_state_changed(self, record: EvidenceRecord, state: str) -> None:
        if self._suppress_scope:
            return
        for idx, existing in enumerate(self._records):
            if existing.identifier == record.identifier:
                self._records[idx] = existing.copy_with_state(state)
                break
        if not self._suppress_scope:
            scope = self.current_scope
            self.scope_changed.emit(scope["include"], scope["exclude"])

    def _open_in_app(self) -> None:
        index = self._list.currentRow()
        if not (0 <= index < len(self._records)):
            return
        record = self._records[index]
        if not record.path:
            return
        url = QUrl.fromLocalFile(str(record.path))
        QDesktopServices.openUrl(url)


__all__ = ["EvidencePanel", "EvidenceRecord"]
