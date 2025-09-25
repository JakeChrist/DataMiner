"""Evidence panel widget with scope controls and source preview."""

from __future__ import annotations

from dataclasses import dataclass, field
import html
from typing import Any, Iterable

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QTextDocument
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
    score: float | None = None
    step_badges: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    conflict_summary: str | None = None
    conflict_sources: list[dict[str, Any]] = field(default_factory=list)
    document_id: int | None = None
    passage_id: str | None = None

    def copy_with_state(self, state: str) -> "EvidenceRecord":
        updated = EvidenceRecord(
            identifier=self.identifier,
            label=self.label,
            snippet_html=self.snippet_html,
            metadata_text=self.metadata_text,
            path=self.path,
            raw=self.raw,
            state=state,
            score=self.score,
            step_badges=list(self.step_badges),
            tags=list(self.tags),
            conflict_summary=self.conflict_summary,
            conflict_sources=list(self.conflict_sources),
            document_id=self.document_id,
            passage_id=self.passage_id,
        )
        return updated


class _EvidenceRow(QFrame):
    """Presentation of a single evidence entry with quick actions."""

    state_changed = pyqtSignal(str)
    copy_requested = pyqtSignal()
    locate_requested = pyqtSignal()
    open_requested = pyqtSignal()

    def __init__(self, record: EvidenceRecord, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.record = record
        self.setObjectName("evidenceRow")
        self._density = "comfortable"

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        self._layout.addLayout(header)

        self.title_label = QLabel(record.label, self)
        self.title_label.setObjectName("evidenceTitle")
        self.title_label.setWordWrap(True)
        header.addWidget(self.title_label, 1)

        toggles = QHBoxLayout()
        toggles.setSpacing(4)
        header.addLayout(toggles)

        self.include_button = QToolButton(self)
        self.include_button.setObjectName("includeButton")
        self.include_button.setText("Include")
        self.include_button.setCheckable(True)
        self.include_button.setChecked(record.state == "include")
        self.include_button.toggled.connect(self._on_include_toggled)
        toggles.addWidget(self.include_button)

        self.exclude_button = QToolButton(self)
        self.exclude_button.setObjectName("excludeButton")
        self.exclude_button.setText("Exclude")
        self.exclude_button.setCheckable(True)
        self.exclude_button.setChecked(record.state == "exclude")
        self.exclude_button.toggled.connect(self._on_exclude_toggled)
        toggles.addWidget(self.exclude_button)

        self._badge_row = QHBoxLayout()
        self._badge_row.setSpacing(6)
        self._layout.addLayout(self._badge_row)
        self._populate_badges()

        self.snippet_label = QLabel(self)
        self.snippet_label.setObjectName("evidenceSnippet")
        self.snippet_label.setTextFormat(Qt.TextFormat.RichText)
        self.snippet_label.setWordWrap(True)
        self.snippet_label.setText(record.snippet_html or "<i>No snippet provided.</i>")
        self.snippet_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._layout.addWidget(self.snippet_label)

        self.metadata_label = QLabel(record.metadata_text, self)
        self.metadata_label.setObjectName("evidenceMeta")
        self.metadata_label.setWordWrap(True)
        self._layout.addWidget(self.metadata_label)

        self._actions_row = QHBoxLayout()
        self._actions_row.setSpacing(6)
        self._layout.addLayout(self._actions_row)

        self.copy_button = QToolButton(self)
        self.copy_button.setObjectName("copyEvidence")
        self.copy_button.setText("Copy")
        self.copy_button.clicked.connect(self.copy_requested.emit)
        self._actions_row.addWidget(self.copy_button)

        self.open_button = QToolButton(self)
        self.open_button.setObjectName("openEvidence")
        self.open_button.setText("Open")
        self.open_button.setEnabled(bool(record.path))
        self.open_button.clicked.connect(self.open_requested.emit)
        self._actions_row.addWidget(self.open_button)

        self.locate_button = QToolButton(self)
        self.locate_button.setObjectName("locateEvidence")
        self.locate_button.setText("Locate")
        self.locate_button.clicked.connect(self.locate_requested.emit)
        self._actions_row.addWidget(self.locate_button)

        self._actions_row.addStretch(1)

    # ------------------------------------------------------------------
    def set_state(self, state: str) -> None:
        with _SignalBlocker(self.include_button), _SignalBlocker(self.exclude_button):
            self.include_button.setChecked(state == "include")
            self.exclude_button.setChecked(state == "exclude")
        self.record = self.record.copy_with_state(state)

    def set_density(self, density: str) -> None:
        self._density = density
        compact = density == "compact"
        margin = 8 if compact else 12
        spacing = 4 if compact else 8
        self._layout.setContentsMargins(margin, margin, margin, margin)
        self._layout.setSpacing(spacing)
        self._badge_row.setSpacing(4 if compact else 6)
        self._actions_row.setSpacing(spacing)

    def refresh(self) -> None:
        self.title_label.setText(self.record.label)
        self.snippet_label.setText(self.record.snippet_html or "<i>No snippet provided.</i>")
        self.metadata_label.setText(self.record.metadata_text)
        self.open_button.setEnabled(bool(self.record.path))
        self._populate_badges()

    def _populate_badges(self) -> None:
        while self._badge_row.count():
            item = self._badge_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for step in self.record.step_badges:
            self._badge_row.addWidget(self._make_badge(step, "step"))
        for tag in self.record.tags:
            self._badge_row.addWidget(self._make_badge(tag, "tag"))
        if self.record.score is not None:
            self._badge_row.addWidget(
                self._make_badge(f"Score {self.record.score:.2f}", "score")
            )
        if self.record.conflict_summary:
            badge = self._make_badge("Conflict", "conflict")
            badge.setToolTip(self.record.conflict_summary)
            self._badge_row.addWidget(badge)
        self._badge_row.addStretch(1)

    def _make_badge(self, text: str, kind: str) -> QLabel:
        badge = QLabel(text, self)
        badge.setObjectName("evidenceBadge")
        badge.setProperty("kind", kind)
        return badge

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
    locate_requested = pyqtSignal(dict)
    copy_requested = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records: list[EvidenceRecord] = []
        self._suppress_scope = False
        self._density = "comfortable"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._conflict_banner = QLabel("", self)
        self._conflict_banner.setObjectName("conflictBanner")
        self._conflict_banner.setWordWrap(True)
        self._conflict_banner.setVisible(False)
        layout.addWidget(self._conflict_banner)

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

        self._reset_scope_button = QPushButton("Reset scope", self)
        self._reset_scope_button.setObjectName("resetScopeButton")
        self._reset_scope_button.clicked.connect(self._reset_scope_filters)
        self._reset_scope_button.setEnabled(False)
        meta_row.addWidget(self._reset_scope_button)

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
        self._update_reset_button_state()

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
        self._update_reset_button_state()
        if emit_scope:
            scope = self.current_scope
            self.scope_changed.emit(scope["include"], scope["exclude"])

    def select_index(self, index: int) -> None:
        if 0 <= index < self._list.count():
            self._list.setCurrentRow(index)

    def set_density(self, density: str) -> None:
        self._density = density
        spacing = 6 if density == "compact" else 8
        layout = self.layout()
        if isinstance(layout, QVBoxLayout):
            layout.setSpacing(spacing)
        for row in range(self._list.count()):
            widget = self._list.itemWidget(self._list.item(row))
            if isinstance(widget, _EvidenceRow):
                widget.set_density(density)

    def reset_scope(self) -> None:
        self._suppress_scope = True
        try:
            for index, record in enumerate(self._records):
                self._records[index] = record.copy_with_state("include")
                item = self._list.item(index)
                widget = self._list.itemWidget(item) if item is not None else None
                if isinstance(widget, _EvidenceRow):
                    widget.set_state("include")
        finally:
            self._suppress_scope = False
        scope = self.current_scope
        self.scope_changed.emit(scope["include"], scope["exclude"])
        self._update_reset_button_state()

    # ------------------------------------------------------------------
    def _empty_state(self) -> None:
        self._preview.setHtml("<p>No evidence available.</p>")
        self._metadata_label.setText("No evidence selected.")
        self._conflict_banner.setVisible(False)
        self._update_reset_button_state()

    def _populate_list(self) -> None:
        self._list.clear()
        conflict_messages: list[str] = []
        for index, record in enumerate(self._records):
            item = QListWidgetItem(self._list)
            widget = _EvidenceRow(record, self._list)
            widget.state_changed.connect(
                lambda state, rec=record: self._on_state_changed(rec, state)
            )
            widget.copy_requested.connect(
                lambda _checked=False, rec=record, item=item: self._handle_copy(rec, item)
            )
            widget.locate_requested.connect(
                lambda _checked=False, rec=record, item=item: self._handle_locate(rec, item)
            )
            widget.open_requested.connect(
                lambda _checked=False, rec=record, item=item: self._handle_open(rec, item)
            )
            widget.set_density(self._density)
            item.setSizeHint(widget.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, widget)
            if record.conflict_summary:
                conflict_messages.append(record.conflict_summary)
        self._set_conflict_banner(conflict_messages)
        self._update_reset_button_state()

    def _normalize_citation(self, index: int, citation: Any) -> EvidenceRecord:
        score: float | None = None
        steps: list[str] = []
        tags: list[str] = []
        conflict_summary: str | None = None
        conflict_sources: list[dict[str, Any]] = []
        document_id: int | None = None
        passage_id: str | None = None

        if isinstance(citation, str):
            label = f"[{index}] {citation}"
            snippet = html.escape(citation)
            metadata = ""
            path = None
        elif isinstance(citation, dict):
            data = dict(citation)
            nested = data.get("citation")
            if isinstance(nested, dict):
                merged = dict(nested)
                for key, value in data.items():
                    if key not in merged:
                        merged[key] = value
                data = merged
            source = str(
                data.get("source")
                or data.get("title")
                or data.get("document")
                or data.get("path")
                or f"Source {index}"
            ).strip()
            label = f"[{index}] {source}" if source else f"[{index}] Source"
            snippet_raw = (
                data.get("snippet")
                or data.get("highlight")
                or data.get("preview")
                or data.get("text")
                or ""
            )
            snippet = snippet_raw if isinstance(snippet_raw, str) else html.escape(str(snippet_raw))
            if "<" not in snippet:
                snippet = html.escape(snippet)
            location_parts: list[str] = []
            page = data.get("page") or data.get("page_number")
            if page:
                location_parts.append(f"Page {page}")
            section = data.get("section") or data.get("heading")
            if section:
                location_parts.append(str(section))
            raw_steps = data.get("steps")
            if isinstance(raw_steps, Iterable) and not isinstance(raw_steps, (str, bytes, dict)):
                step_ids = [str(step).strip() for step in raw_steps if str(step).strip()]
                if step_ids:
                    steps = [f"Step {value}" for value in step_ids]
                    location_parts.insert(0, f"Steps {', '.join(step_ids)}")
            else:
                single_step = data.get("step") or data.get("step_index")
                if isinstance(single_step, (int, str)):
                    step_value = str(single_step).strip()
                    if step_value:
                        steps = [f"Step {step_value}"]
                        location_parts.insert(0, f"Step {step_value}")
            metadata = " | ".join(location_parts)
            path = data.get("path") or data.get("file_path") or data.get("source_path")
            if isinstance(path, str) and not path.strip():
                path = None
            score_value = data.get("score") or data.get("similarity")
            try:
                score = float(score_value) if score_value is not None else None
            except (TypeError, ValueError):
                score = None
            tags_raw = data.get("tag_names") or data.get("tags") or []
            if isinstance(tags_raw, Iterable) and not isinstance(tags_raw, (str, bytes, dict)):
                tags = [str(tag).strip() for tag in tags_raw if str(tag).strip()]
            elif isinstance(tags_raw, str):
                tag_value = tags_raw.strip()
                if tag_value:
                    tags = [tag_value]
            if not tags:
                single_tag = data.get("tag") or data.get("tag_name")
                if isinstance(single_tag, str) and single_tag.strip():
                    tags = [single_tag.strip()]
            summary_raw = data.get("conflict_summary") or data.get("conflictSummary")
            if isinstance(summary_raw, str) and summary_raw.strip():
                conflict_summary = summary_raw.strip()
            conflicts_raw = data.get("conflicts")
            if isinstance(conflicts_raw, Iterable) and not isinstance(conflicts_raw, (str, bytes)):
                conflict_sources = [c for c in conflicts_raw if isinstance(c, dict)]
                if conflict_sources:
                    ids = [
                        str(conflict.get("passage_id") or conflict.get("document_id") or "")
                        for conflict in conflict_sources
                    ]
                    names = [value for value in ids if value]
                    label_text = ", ".join(names[:3])
                    remaining = max(0, len(names) - 3)
                    if remaining:
                        label_text = f"{label_text} +{remaining}"
                    if label_text:
                        conflict_summary = (
                            f"Conflicts with {len(conflict_sources)} passages ({label_text})"
                        )
                    else:
                        conflict_summary = f"Conflicts with {len(conflict_sources)} passages"
            doc_id_raw = data.get("document_id")
            try:
                document_id = int(doc_id_raw) if doc_id_raw is not None else None
            except (TypeError, ValueError):
                document_id = None
            passage = data.get("passage_id") or data.get("id")
            passage_id = str(passage).strip() if passage else None
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
            score=score,
            step_badges=steps,
            tags=tags,
            conflict_summary=conflict_summary,
            conflict_sources=conflict_sources,
            document_id=document_id,
            passage_id=passage_id,
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

    def _handle_copy(self, record: EvidenceRecord, item: QListWidgetItem) -> None:
        row = self._list.row(item)
        if row >= 0:
            self._list.setCurrentRow(row)
        snippet = self._plain_text(record.snippet_html)
        payload = "\n".join(
            part
            for part in [record.label, snippet, record.metadata_text]
            if part and part.strip()
        )
        self.copy_requested.emit(payload.strip())

    def _handle_locate(self, record: EvidenceRecord, item: QListWidgetItem) -> None:
        row = self._list.row(item)
        if row >= 0:
            self._list.setCurrentRow(row)
        payload = {
            "document_id": record.document_id,
            "path": record.path,
            "passage_id": record.passage_id,
        }
        self.locate_requested.emit(payload)

    def _handle_open(self, record: EvidenceRecord, item: QListWidgetItem) -> None:
        row = self._list.row(item)
        if row >= 0:
            self._list.setCurrentRow(row)
        self._open_record(record)

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
        snippet_html = record.snippet_html or "<i>No snippet provided.</i>"
        style = (
            "<style>"
            ".snippet{font-size:13px;line-height:1.5;}"
            ".meta{margin-top:8px;color:rgba(140,150,165,0.9);}"
            ".conflict{margin-top:6px;color:#d8893a;font-weight:600;}"
            ".tags{margin-top:4px;color:rgba(140,150,165,0.9);}"
            "</style>"
        )
        fragments = [f"<div class='snippet'>{snippet_html}</div>"]
        if record.metadata_text:
            fragments.append(f"<p class='meta'>{html.escape(record.metadata_text)}</p>")
        if record.conflict_summary:
            fragments.append(f"<p class='conflict'>{html.escape(record.conflict_summary)}</p>")
        if record.tags:
            tags = ", ".join(html.escape(tag) for tag in record.tags)
            fragments.append(f"<p class='tags'>Tags: {tags}</p>")
        self._preview.setHtml(style + "".join(fragments))
        header = record.label
        if record.metadata_text:
            header = f"{record.label} — {record.metadata_text}"
        self._metadata_label.setText(header)
        self._update_reset_button_state()

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
        self._update_reset_button_state()

    def _reset_scope_filters(self) -> None:
        if not any(record.state != "include" for record in self._records):
            return
        self.reset_scope()

    def _set_conflict_banner(self, messages: list[str]) -> None:
        if messages:
            unique = list(dict.fromkeys(messages))
            joined = " • ".join(html.escape(msg) for msg in unique)
            self._conflict_banner.setText(f"⚠️ Conflict detected: {joined}")
            self._conflict_banner.setVisible(True)
        else:
            self._conflict_banner.setVisible(False)

    def _update_reset_button_state(self) -> None:
        if not hasattr(self, "_reset_scope_button"):
            return
        dirty = any(record.state != "include" for record in self._records)
        enabled = dirty and bool(self._records)
        self._reset_scope_button.setEnabled(enabled)

    def _plain_text(self, snippet: str) -> str:
        document = QTextDocument()
        document.setHtml(snippet or "")
        return document.toPlainText().strip()

    def _open_record(self, record: EvidenceRecord | None) -> None:
        if record is None or not record.path:
            return
        url = QUrl.fromLocalFile(str(record.path))
        QDesktopServices.openUrl(url)


__all__ = ["EvidencePanel", "EvidenceRecord"]
