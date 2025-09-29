"""Utilities for extracting text and metadata from PDF files."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from statistics import median
from typing import Any

from . import DocumentSection, PageContent, ParsedDocument, ParserError


def _to_datetime(value: Any) -> str | None:
    """Normalize a PyMuPDF metadata value into ISO format when possible."""

    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_dt.timezone.utc)
        return value.astimezone(_dt.timezone.utc).isoformat()
    if isinstance(value, str):
        try:
            # PyMuPDF encodes dates as D:YYYYMMDDHHmmSS
            cleaned = value.strip().lstrip("D:")
            if cleaned:
                parsed = _dt.datetime.strptime(cleaned[:14], "%Y%m%d%H%M%S")
                return parsed.replace(tzinfo=_dt.timezone.utc).isoformat()
        except ValueError:
            return value
        return None
    return None


def parse_pdf(path: Path) -> ParsedDocument:
    """Parse ``path`` and extract textual content using PyMuPDF."""

    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - defensive
        raise ParserError("PyMuPDF is required to parse PDF documents") from exc

    try:
        document = fitz.open(path)  # type: ignore[arg-type]
    except fitz.FileDataError as exc:  # pragma: no cover - defensive
        raise ParserError(str(exc)) from exc

    try:
        document_lines: list[str] = []
        sections: list[DocumentSection] = []
        pages: list[PageContent] = []
        needs_ocr = True
        heading_count = 0
        current_section: DocumentSection | None = None
        char_offset = 0
        line_number = 0

        for index, page in enumerate(document, start=1):
            page_dict = page.get_text("dict")
            line_candidates: list[dict[str, Any]] = []
            span_sizes: list[float] = []

            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans") or []
                    if not spans:
                        continue
                    raw_text = "".join(span.get("text", "") for span in spans)
                    text_line = raw_text.rstrip()
                    is_blank = not text_line.strip()
                    size = 0.0
                    bold = False
                    if not is_blank:
                        numeric_sizes = [
                            float(span.get("size", 0.0))
                            for span in spans
                            if str(span.get("text", "")).strip()
                        ]
                        if numeric_sizes:
                            size = max(numeric_sizes)
                            span_sizes.extend(numeric_sizes)
                        bold = any(int(span.get("flags", 0)) & 2 for span in spans)
                    line_candidates.append(
                        {
                            "text": text_line,
                            "blank": is_blank,
                            "size": size,
                            "bold": bold,
                            "page": index,
                        }
                    )

            body_size = median(span_sizes) if span_sizes else 0.0
            page_lines: list[str] = []

            for entry in line_candidates:
                text_line = entry["text"]
                page_lines.append(text_line)
                document_lines.append(text_line)
                if entry["blank"]:
                    char_offset += len(text_line) + 1 if text_line else 1
                    line_number += 1
                    continue

                stripped = text_line.strip()
                if stripped:
                    needs_ocr = False

                level: int | None = None
                if body_size and stripped:
                    relative = entry["size"] / body_size if body_size else 1.0
                    word_count = len(stripped.split())
                    if relative >= 1.8 and word_count <= 20:
                        level = 1
                    elif relative >= 1.45 and word_count <= 24:
                        level = 2
                    elif relative >= 1.25 and word_count <= 28:
                        level = 3
                    elif entry["bold"] and relative >= 1.1 and word_count <= 20:
                        level = 3

                if level is not None:
                    heading_count += 1
                    current_section = DocumentSection(
                        title=stripped,
                        content="",
                        level=level,
                        page_number=index,
                        start_offset=char_offset,
                        end_offset=char_offset + len(text_line),
                        line_start=line_number + 1,
                        line_end=line_number + 1,
                    )
                    sections.append(current_section)
                    char_offset += len(text_line) + 1
                    line_number += 1
                    continue

                if current_section is None:
                    current_section = DocumentSection(
                        title=None,
                        content="",
                        level=None,
                        page_number=index,
                        start_offset=char_offset,
                        line_start=line_number + 1,
                    )
                    sections.append(current_section)

                if current_section.content:
                    current_section.content += "\n"
                current_section.content += text_line
                current_section.end_offset = char_offset + len(text_line)
                current_section.line_end = line_number + 1
                if current_section.line_start is None:
                    current_section.line_start = line_number + 1

                char_offset += len(text_line) + 1
                line_number += 1

            pages.append(PageContent(number=index, text="\n".join(page_lines)))

        metadata = {key: value for key, value in (document.metadata or {}).items() if value}
        if document.page_count is not None:
            metadata["page_count"] = int(document.page_count)

        # Normalize datetime fields for consistency.
        for key in ("creationDate", "modDate", "CreationDate", "ModDate"):
            if key in metadata:
                normalized = _to_datetime(metadata[key])
                if normalized:
                    metadata[key] = normalized

        ocr_hint = (
            "No extractable text detected. Consider running OCR before re-ingesting this PDF."
            if needs_ocr
            else None
        )

        sections = [section for section in sections if section.content or section.title]
        combined_text = "\n".join(document_lines).strip()
        return ParsedDocument(
            text=combined_text,
            metadata={**metadata, "heading_count": heading_count},
            sections=sections,
            pages=pages,
            needs_ocr=needs_ocr,
            ocr_hint=ocr_hint,
        )
    finally:
        document.close()
