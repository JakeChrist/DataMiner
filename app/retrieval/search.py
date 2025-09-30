"""Keyword search helpers that bridge the ingest index and project metadata."""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

from app.storage import ChatRepository, DocumentRepository, IngestDocumentRepository


class SearchService:
    """Expose keyword search results with metadata and reusable scopes."""

    def __init__(
        self,
        ingest_repository: IngestDocumentRepository,
        document_repository: DocumentRepository,
        chat_repository: ChatRepository,
    ) -> None:
        self.ingest = ingest_repository
        self.documents = document_repository
        self.chats = chat_repository

    def search_documents(
        self,
        query: str,
        *,
        project_id: int,
        limit: int = 5,
        chat_id: int | None = None,
        tags: Iterable[int] | None = None,
        folder: str | Path | None = None,
        recursive: bool = True,
        save_scope: bool = False,
    ) -> list[dict[str, Any]]:
        """Search the ingest index and join results with project documents."""

        scope_tags, scope_folder = self._resolve_scope(
            chat_id,
            tags,
            folder,
            save_scope=save_scope,
        )
        candidate_documents = self.documents.list_for_scope(
            project_id,
            tags=scope_tags,
            folder=scope_folder,
            recursive=recursive,
        )
        documents_by_path = self._build_path_index(candidate_documents)
        seen_documents: set[int] = set()
        results: list[dict[str, Any]] = []
        for record in self._search_with_fallback(query, limit=limit * 6):
            doc_payload = record.get("document") or {}
            path = record.get("path") or doc_payload.get("path")
            if not path:
                continue
            document = documents_by_path.get(self._normalize_path(path))
            if document is None:
                continue
            doc_id = int(document.get("id"))
            if doc_id in seen_documents:
                continue
            seen_documents.add(doc_id)
            chunk: dict[str, Any] = record.get("chunk") or {}
            chunk_text = chunk.get("text") if isinstance(chunk, dict) else None
            highlight = record.get("highlight") or chunk_text or doc_payload.get("preview")
            results.append(
                {
                    "document": document,
                    "highlight": highlight,
                    "context": chunk_text or "",
                    "chunk": chunk,
                    "ingest_document": doc_payload,
                    "score": record.get("score"),
                }
            )
            if len(results) >= limit:
                break
        return results

    def collect_context_records(
        self,
        query: str,
        *,
        project_id: int,
        limit: int = 5,
        tags: Iterable[int] | None = None,
        folder: str | Path | None = None,
        recursive: bool = True,
        include_identifiers: Iterable[str] | None = None,
        exclude_identifiers: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return structured retrieval records for ``query`` within scope."""

        scope_tags = list(tags) if tags is not None else None
        scope_folder = self._normalize_folder(folder) if folder is not None else None
        candidate_documents = self.documents.list_for_scope(
            project_id,
            tags=scope_tags,
            folder=scope_folder,
            recursive=recursive,
        )
        documents_by_path = self._build_path_index(candidate_documents)
        include_set = {str(item) for item in (include_identifiers or []) if str(item)}
        exclude_set = {str(item) for item in (exclude_identifiers or []) if str(item)}
        records: list[dict[str, Any]] = []
        seen_chunks: set[int] = set()
        for record in self._search_with_fallback(query, limit=limit * 6):
            doc_payload = record.get("document") or {}
            path = record.get("path") or doc_payload.get("path")
            if not path:
                continue
            document = documents_by_path.get(self._normalize_path(path))
            if document is None:
                continue
            identifiers = self._document_identifiers(document)
            if include_set and include_set.isdisjoint(identifiers):
                continue
            if exclude_set and not exclude_set.isdisjoint(identifiers):
                continue
            chunk: dict[str, Any] = record.get("chunk") or {}
            chunk_id = int(chunk.get("id", -1)) if isinstance(chunk, dict) else -1
            if chunk_id in seen_chunks:
                continue
            text = chunk.get("text") if isinstance(chunk, dict) else None
            if not text:
                continue
            highlight = (
                record.get("highlight")
                or chunk.get("highlight")
                or doc_payload.get("preview")
                or text
            )
            context_record = {
                "document": document,
                "chunk": chunk,
                "context": text,
                "highlight": highlight,
                "ingest_document": doc_payload,
                "score": record.get("score"),
            }
            records.append(context_record)
            seen_chunks.add(chunk_id)
            if len(records) >= limit:
                break
        return records

    def retrieve_context_snippets(
        self,
        query: str,
        *,
        project_id: int,
        limit: int = 5,
        tags: Iterable[int] | None = None,
        folder: str | Path | None = None,
        recursive: bool = True,
        include_identifiers: Iterable[str] | None = None,
        exclude_identifiers: Iterable[str] | None = None,
    ) -> list[str]:
        records = self.collect_context_records(
            query,
            project_id=project_id,
            limit=limit,
            tags=tags,
            folder=folder,
            recursive=recursive,
            include_identifiers=include_identifiers,
            exclude_identifiers=exclude_identifiers,
        )
        snippets: list[str] = []
        for entry in records:
            document = entry.get("document") or {}
            chunk = entry.get("chunk") or {}
            context = entry.get("context") or chunk.get("text") or ""
            path = document.get("source_path") or entry.get("path")
            title = document.get("title") or (Path(path).stem if path else None)
            if not title and path:
                title = Path(path).name
            label = title or "Document"
            snippets.append(f"{label}: {context.strip()}")
        return snippets

    def _resolve_scope(
        self,
        chat_id: int | None,
        tags: Iterable[int] | None,
        folder: str | Path | None,
        *,
        save_scope: bool,
    ) -> tuple[list[int] | None, str | None]:
        explicit_tags = list(tags) if tags is not None else None
        explicit_folder = self._normalize_folder(folder) if folder is not None else None

        stored_scope: dict[str, Any] = {}
        if chat_id is not None:
            stored_scope = self.chats.get_query_scope(chat_id) or {}

        scope_tags = explicit_tags if explicit_tags is not None else stored_scope.get("tags")
        scope_folder = explicit_folder if folder is not None else stored_scope.get("folder")

        if chat_id is not None and save_scope:
            payload = {
                "tags": explicit_tags or [],
                "folder": explicit_folder,
            }
            self.chats.set_query_scope(chat_id, payload)

        return scope_tags, scope_folder

    def _build_path_index(self, documents: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for document in documents:
            source_path = document.get("source_path")
            if not source_path:
                continue
            lookup_key = self._normalize_path(source_path)
            if not lookup_key:
                continue
            index[lookup_key] = document
            raw_key = str(source_path)
            if raw_key not in index:
                index[raw_key] = document
        return index

    @staticmethod
    def _document_identifiers(document: dict[str, Any]) -> set[str]:
        identifiers: set[str] = set()
        doc_id = document.get("id")
        if doc_id is not None:
            identifiers.add(str(doc_id))
        source_path = document.get("source_path")
        if source_path:
            identifiers.add(str(source_path))
            identifiers.add(SearchService._normalize_path(source_path))
        title = document.get("title")
        if isinstance(title, str) and title:
            identifiers.add(title)
        return identifiers

    @staticmethod
    def _normalize_path(path: str | Path) -> str:
        """Return a canonical lookup key for ``path``."""

        return SearchService._canonical_path_key(path)

    @staticmethod
    def _canonical_path_key(path: str | Path) -> str:
        """Normalise paths for reliable cross-platform comparisons."""

        text = str(path).strip()
        if not text:
            return ""

        collapsed = text.replace("\\", "/")
        drive_match = re.search(r"(?i)([a-z]):/", collapsed)
        if drive_match:
            collapsed = collapsed[drive_match.start(1) :]
        else:
            unc_index = collapsed.find("//")
            if unc_index > 0:
                collapsed = collapsed[unc_index:]

        if collapsed.startswith("//"):
            prefix = "//"
            remainder = collapsed[2:]
        elif collapsed.startswith("/"):
            prefix = "/"
            remainder = collapsed[1:]
        else:
            prefix = ""
            remainder = collapsed

        remainder = re.sub(r"/+", "/", remainder)
        normalized = prefix + remainder
        return normalized.lower()

    @staticmethod
    def _normalize_folder(folder: str | Path | None) -> str | None:
        if folder in (None, ""):
            return None
        return str(Path(folder).expanduser().resolve())

    # ------------------------------------------------------------------
    _TOKEN_PATTERN = re.compile(r"[\w-]+", re.UNICODE)
    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
    }

    def _search_with_fallback(self, query: str, *, limit: int) -> Iterable[dict[str, Any]]:
        """Yield ingest search hits while progressively relaxing the query."""

        normalized = (query or "").strip()
        if not normalized:
            return

        attempts: list[str] = [normalized]
        tokens = self._tokenize_query(normalized)
        if tokens:
            wildcard_tokens = [f"{token}*" for token in tokens]
            attempts.append(" OR ".join(wildcard_tokens))
            if len(tokens) > 1:
                attempts.append(" ".join(wildcard_tokens))

        seen_queries: set[str] = set()
        seen_results: set[tuple[Any, ...]] = set()
        yielded = 0

        def _result_identity(record: dict[str, Any]) -> tuple[Any, ...]:
            chunk = record.get("chunk") if isinstance(record, dict) else None
            chunk_id: Any = None
            chunk_index: Any = None
            if isinstance(chunk, dict):
                chunk_id = chunk.get("id")
                chunk_index = chunk.get("index")
            document = record.get("document") if isinstance(record, dict) else None
            document_id = document.get("id") if isinstance(document, dict) else None
            path = record.get("path") if isinstance(record, dict) else None
            return (chunk_id, document_id, chunk_index, path)

        for candidate in attempts:
            candidate = candidate.strip()
            if not candidate or candidate in seen_queries:
                continue
            seen_queries.add(candidate)
            try:
                results = self.ingest.search(candidate, limit=limit)
            except sqlite3.OperationalError:
                continue
            for record in results:
                identity = _result_identity(record)
                if identity in seen_results:
                    continue
                seen_results.add(identity)
                yield record
                yielded += 1
                if yielded >= limit:
                    return

    def _tokenize_query(self, query: str) -> list[str]:
        """Extract significant terms for relaxed fallback queries."""

        tokens = [match.group(0).lower() for match in self._TOKEN_PATTERN.finditer(query)]
        filtered = [
            token
            for token in tokens
            if len(token) >= 3 and token not in self._STOPWORDS
        ]
        seen: set[str] = set()
        unique: list[str] = []
        for token in filtered:
            if token not in seen:
                seen.add(token)
                unique.append(token)
        return unique
