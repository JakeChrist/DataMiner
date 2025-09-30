"""Plain text parser with encoding detection."""

from __future__ import annotations

from pathlib import Path

from . import DocumentSection, PageContent, ParsedDocument


def parse_text(path: Path) -> ParsedDocument:
    """Parse a plain-text file while attempting to detect encoding."""

    raw = path.read_bytes()
    encoding = "utf-8"
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        try:
            import chardet  # type: ignore[import-untyped]
        except ImportError:  # pragma: no cover - defensive
            text = raw.decode(encoding, errors="replace")
        else:
            detected = chardet.detect(raw)
            encoding = detected.get("encoding") or "utf-8"
            text = raw.decode(encoding, errors="replace")

    metadata = {"encoding": encoding}
    sections = [DocumentSection(title=None, content=text)]
    pages = [PageContent(number=1, text=text)]
    return ParsedDocument(text=text, metadata=metadata, sections=sections, pages=pages)
