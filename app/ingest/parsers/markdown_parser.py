"""Markdown parser that preserves heading hierarchy."""

from __future__ import annotations

import re
from pathlib import Path

from . import DocumentSection, PageContent, ParsedDocument

_HEADING_RE = re.compile(r"^(?P<level>#{1,6})\s*(?P<title>.+)$")


def parse_markdown(path: Path) -> ParsedDocument:
    """Parse a Markdown file into sections based on heading levels."""

    text = path.read_text(encoding="utf-8")
    sections: list[DocumentSection] = []
    current_section: DocumentSection | None = None
    body_lines: list[str] = []

    for line in text.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match:
            level = len(match.group("level"))
            title = match.group("title").strip()
            current_section = DocumentSection(title=title, content="", level=level)
            sections.append(current_section)
            continue
        if current_section is None:
            current_section = DocumentSection(title=None, content="")
            sections.append(current_section)
        current_section.content += ("\n" if current_section.content else "") + line.rstrip()
        body_lines.append(line.rstrip())

    sections = [section for section in sections if section.content or section.title]
    combined = "\n".join(body_lines)
    pages = [PageContent(number=1, text=combined)]
    metadata = {"format": "markdown", "heading_count": len([s for s in sections if s.title])}
    return ParsedDocument(text=combined, metadata=metadata, sections=sections, pages=pages)
