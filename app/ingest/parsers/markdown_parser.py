"""Markdown parser that preserves heading hierarchy."""

from __future__ import annotations

import re
from pathlib import Path

from . import DocumentSection, PageContent, ParsedDocument

_HEADING_RE = re.compile(r"^(?P<level>#{1,6})\s*(?P<title>.+?)\s*$")


def parse_markdown(path: Path) -> ParsedDocument:
    """Parse a Markdown file into sections based on heading levels."""

    text = path.read_text(encoding="utf-8")
    sections: list[DocumentSection] = []
    current_section: DocumentSection | None = None
    body_lines: list[str] = []
    heading_count = 0
    char_offset = 0
    line_number = 0

    for raw_line in text.splitlines():
        line_number += 1
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        match = _HEADING_RE.match(stripped)
        if match:
            level = len(match.group("level"))
            title = match.group("title").strip()
            heading_count += 1
            current_section = DocumentSection(
                title=title,
                content="",
                level=level,
                page_number=1,
                start_offset=char_offset,
                end_offset=char_offset + len(line),
                line_start=line_number,
                line_end=line_number,
            )
            sections.append(current_section)
            body_lines.append(line)
            char_offset += len(line) + 1
            continue

        body_lines.append(line)
        if current_section is None:
            current_section = DocumentSection(
                title=None,
                content="",
                level=None,
                page_number=1,
            )
            sections.append(current_section)

        if stripped:
            if current_section.content:
                current_section.content += "\n"
            current_section.content += line
            if current_section.start_offset is None:
                current_section.start_offset = char_offset
            current_section.end_offset = char_offset + len(line)
            if current_section.line_start is None:
                current_section.line_start = line_number
            current_section.line_end = line_number
        char_offset += len(line) + 1

    sections = [section for section in sections if section.content or section.title]
    combined = "\n".join(body_lines)
    pages = [PageContent(number=1, text=combined)]
    metadata = {"format": "markdown", "heading_count": heading_count}
    return ParsedDocument(text=combined, metadata=metadata, sections=sections, pages=pages)
