"""Document parsers that extract text and structural metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(slots=True)
class PageContent:
    """Representation of a single logical page."""

    number: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentSection:
    """Hierarchical section extracted from a document."""

    title: str | None
    content: str
    level: int | None = None
    page_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ParsedDocument:
    """Normalized representation returned by document parsers."""

    text: str
    metadata: dict[str, Any]
    sections: list[DocumentSection] = field(default_factory=list)
    pages: list[PageContent] = field(default_factory=list)
    needs_ocr: bool = False
    ocr_hint: str | None = None

    def to_json(self) -> str:
        """Return a JSON representation of structured data for storage."""

        payload = {
            "metadata": self.metadata,
            "sections": [section.to_dict() for section in self.sections],
            "pages": [page.to_dict() for page in self.pages],
            "needs_ocr": self.needs_ocr,
            "ocr_hint": self.ocr_hint,
        }
        return json.dumps(payload)


class ParserError(RuntimeError):
    """Raised when a parser cannot handle the supplied document."""
from .docx_parser import parse_docx
from .markdown_parser import parse_markdown
from .pdf_parser import parse_pdf
from .text_parser import parse_text


Parser = Callable[[Path], ParsedDocument]


_PARSERS: dict[str, Parser] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_text,
    ".text": parse_text,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".mkd": parse_markdown,
    ".html": parse_text,
    ".htm": parse_text,
    ".py": parse_text,
    ".pyw": parse_text,
    ".m": parse_text,
    ".cpp": parse_text,
}


def register_parser(suffixes: Iterable[str], parser: Parser) -> None:
    """Register ``parser`` for the provided ``suffixes``."""

    for suffix in suffixes:
        _PARSERS[suffix.lower()] = parser


def parse_file(path: str | Path) -> ParsedDocument:
    """Parse ``path`` and return a :class:`ParsedDocument` instance."""

    resolved = Path(path).resolve()
    if not resolved.exists():
        raise ParserError(f"Missing document: {resolved}")
    suffix = resolved.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise ParserError(f"Unsupported file type: {resolved.suffix}")
    return parser(resolved)


__all__ = [
    "DocumentSection",
    "PageContent",
    "ParsedDocument",
    "ParserError",
    "parse_file",
    "register_parser",
]
