"""Helpers for generating highlighted previews of ingested documents."""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from app.storage import DatabaseManager, IngestDocumentRepository


class PreviewService:
    """Provide highlighted passages and page retrieval for stored documents."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.documents = IngestDocumentRepository(db)

    def get_highlighted_passages(
        self,
        document_id: int,
        terms: Sequence[str] | None = None,
        *,
        context_chars: int = 240,
    ) -> dict[str, Any]:
        """Return a snippet with highlights for ``document_id``.

        ``terms`` is treated as a case-insensitive set of tokens to emphasise in the
        returned snippet. If no terms are supplied, the stored preview is returned.
        """

        record = self.documents.get(document_id)
        if record is None:
            raise LookupError(f"Unknown document id: {document_id}")

        normalized = record.get("normalized_text") or record.get("text") or ""
        snippet_source = normalized
        if not snippet_source:
            snippet_source = record.get("preview") or ""

        if not terms:
            snippet = record.get("preview") or snippet_source[:context_chars]
            return self._assemble_preview(record, snippet, page=None, offset=0)

        snippet, offset = self._build_highlight(snippet_source, terms, context_chars)
        page_number = self._find_page(record.get("pages") or [], terms)
        return self._assemble_preview(record, snippet, page=page_number, offset=offset)

    def get_page(self, document_id: int, page_number: int) -> dict[str, Any]:
        """Return the full text of ``page_number`` for ``document_id``."""

        record = self.documents.get(document_id)
        if record is None:
            raise LookupError(f"Unknown document id: {document_id}")
        for page in record.get("pages", []):
            if int(page.get("number", -1)) == int(page_number):
                return {
                    "document_id": record["id"],
                    "page_number": page_number,
                    "text": page.get("text", ""),
                }
        raise LookupError(f"Page {page_number} not found for document {document_id}")

    @staticmethod
    def _apply_highlight(text: str, terms: Iterable[str]) -> str:
        highlighted = text
        for term in sorted({term for term in terms if term}, key=len, reverse=True):
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(lambda match: f"<mark>{match.group(0)}</mark>", highlighted)
        return highlighted

    def _build_highlight(
        self, text: str, terms: Sequence[str], context_chars: int
    ) -> tuple[str, int]:
        if not text:
            return "", 0
        lowered = text.lower()
        best_index = None
        for term in terms:
            if not term:
                continue
            index = lowered.find(term.lower())
            if index != -1 and (best_index is None or index < best_index):
                best_index = index
        if best_index is None:
            snippet = text[:context_chars]
            return self._apply_highlight(snippet, terms), 0
        start = max(best_index - context_chars // 2, 0)
        end = min(start + context_chars, len(text))
        snippet = text[start:end]
        return self._apply_highlight(snippet, terms), start

    @staticmethod
    def _find_page(pages: list[dict[str, Any]], terms: Sequence[str]) -> int | None:
        lowered_terms = [term.lower() for term in terms if term]
        for page in pages:
            content = (page.get("text") or "").lower()
            if any(term in content for term in lowered_terms):
                try:
                    return int(page.get("number"))
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _assemble_preview(
        record: dict[str, Any],
        snippet: str,
        *,
        page: int | None,
        offset: int,
    ) -> dict[str, Any]:
        return {
            "document_id": record.get("id"),
            "path": record.get("path"),
            "version": record.get("version"),
            "snippet": snippet,
            "offset": offset,
            "preview": record.get("preview"),
            "needs_ocr": record.get("needs_ocr", False),
            "ocr_message": record.get("ocr_message"),
            "page": page,
            "pages": record.get("pages"),
            "sections": record.get("sections"),
            "metadata": record.get("metadata"),
        }


__all__ = ["PreviewService"]
