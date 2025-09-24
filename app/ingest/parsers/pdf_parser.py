"""Utilities for extracting text and metadata from PDF files."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
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
        texts: list[str] = []
        sections: list[DocumentSection] = []
        pages: list[PageContent] = []
        needs_ocr = True

        for index, page in enumerate(document, start=1):
            text = page.get_text("text")
            if text.strip():
                needs_ocr = False
            texts.append(text.strip())
            sections.append(
                DocumentSection(
                    title=f"Page {index}",
                    content=text.strip(),
                    level=1,
                    page_number=index,
                )
            )
            pages.append(PageContent(number=index, text=text))

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

        combined_text = "\n\n".join(filter(None, texts))
        return ParsedDocument(
            text=combined_text,
            metadata=metadata,
            sections=sections,
            pages=pages,
            needs_ocr=needs_ocr,
            ocr_hint=ocr_hint,
        )
    finally:
        document.close()
