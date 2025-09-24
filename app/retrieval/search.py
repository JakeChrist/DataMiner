"""Keyword search helpers that bridge the ingest index and project metadata."""

from __future__ import annotations

from pathlib import Path
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
        results: list[dict[str, Any]] = []
        for record in self.ingest.search(query, limit=limit * 3):
            path = record.get("path")
            if not path:
                continue
            document = documents_by_path.get(path)
            if document is None:
                continue
            results.append(
                {
                    "document": document,
                    "highlight": record.get("highlight") or record.get("preview"),
                    "ingest_document": record,
                }
            )
            if len(results) >= limit:
                break
        return results

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
            normalized = self._normalize_path(source_path)
            index[normalized] = document
        return index

    @staticmethod
    def _normalize_path(path: str | Path) -> str:
        return str(Path(path))

    @staticmethod
    def _normalize_folder(folder: str | Path | None) -> str | None:
        if folder in (None, ""):
            return None
        return str(Path(folder))
