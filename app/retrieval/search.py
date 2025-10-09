"""Keyword search helpers that bridge the ingest index and project metadata."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

from app.storage import ChatRepository, DocumentRepository, IngestDocumentRepository


logger = logging.getLogger(__name__)


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

        tags_list = list(tags) if tags is not None else None
        logger.info(
            "Executing RAG document search",
            extra={
                "query_preview": query.strip()[:120],
                "project_id": project_id,
                "limit": limit,
                "tags": tags_list,
                "folder": str(folder) if folder is not None else None,
                "recursive": recursive,
                "save_scope": save_scope,
            },
        )

        scope_tags, scope_folder = self._resolve_scope(
            chat_id,
            tags_list,
            folder,
            save_scope=save_scope,
        )
        candidate_documents = self.documents.list_for_scope(
            project_id,
            tags=scope_tags,
            folder=scope_folder,
            recursive=recursive,
        )
        logger.debug(
            "Loaded candidate documents for RAG search",
            extra={
                "candidate_count": len(candidate_documents),
                "scope_tags": scope_tags,
                "scope_folder": scope_folder,
            },
        )
        documents_by_path = self._build_path_index(candidate_documents)
        seen_documents: set[int] = set()
        results: list[dict[str, Any]] = []
        for record in self._search_with_fallback(query, limit=limit * 6):
            doc_payload = record.get("document") or {}
            path = record.get("path") or doc_payload.get("path")
            if not path:
                logger.debug(
                    "Skipping retrieval record without path",
                    extra={"record_keys": sorted(record.keys())},
                )
                continue
            document = documents_by_path.get(self._normalize_path(path))
            if document is None:
                logger.debug(
                    "No project document matched retrieval record",
                    extra={
                        "path": path,
                        "normalized_path": self._normalize_path(path),
                    },
                )
                continue
            doc_id = int(document.get("id"))
            if doc_id in seen_documents:
                logger.debug(
                    "Skipping duplicate document in RAG search",
                    extra={
                        "document_id": doc_id,
                        "path": document.get("source_path"),
                    },
                )
                continue
            seen_documents.add(doc_id)
            chunk: dict[str, Any] = record.get("chunk") or {}
            chunk_text = chunk.get("text") if isinstance(chunk, dict) else None
            highlight = self._build_evidence_snippet(
                query,
                chunk_text=chunk_text,
                highlight_html=record.get("highlight"),
                preview=doc_payload.get("preview"),
            )
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
            logger.debug(
                "Appended RAG search result",
                extra={
                    "document_id": doc_id,
                    "context_length": len(chunk_text or ""),
                    "highlight_length": len(highlight or ""),
                    "score": record.get("score"),
                },
            )
            if len(results) >= limit:
                break
        logger.info(
            "RAG document search completed",
            extra={
                "query_preview": query.strip()[:120],
                "project_id": project_id,
                "result_count": len(results),
                "candidate_count": len(candidate_documents),
            },
        )
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

        tags_list = list(tags) if tags is not None else None
        include_list = list(include_identifiers or [])
        exclude_list = list(exclude_identifiers or [])
        logger.info(
            "Collecting RAG context records",
            extra={
                "query_preview": query.strip()[:120],
                "project_id": project_id,
                "limit": limit,
                "tags": tags_list,
                "folder": str(folder) if folder is not None else None,
                "recursive": recursive,
                "include_identifiers": include_list,
                "exclude_identifiers": exclude_list,
            },
        )

        scope_tags = list(tags_list) if tags_list is not None else None
        scope_folder = self._normalize_folder(folder) if folder is not None else None
        candidate_documents = self.documents.list_for_scope(
            project_id,
            tags=scope_tags,
            folder=scope_folder,
            recursive=recursive,
        )
        logger.debug(
            "Loaded candidate documents for context collection",
            extra={
                "candidate_count": len(candidate_documents),
                "scope_tags": scope_tags,
                "scope_folder": scope_folder,
            },
        )
        documents_by_path = self._build_path_index(candidate_documents)
        include_set = {str(item) for item in include_list if str(item)}
        exclude_set = {str(item) for item in exclude_list if str(item)}
        records: list[dict[str, Any]] = []
        seen_chunks: set[int] = set()
        for record in self._search_with_fallback(query, limit=limit * 6):
            doc_payload = record.get("document") or {}
            path = record.get("path") or doc_payload.get("path")
            if not path:
                logger.debug(
                    "Skipping retrieval record without path during context collection",
                    extra={"record_keys": sorted(record.keys())},
                )
                continue
            document = documents_by_path.get(self._normalize_path(path))
            if document is None:
                logger.debug(
                    "No document match for context record",
                    extra={
                        "path": path,
                        "normalized_path": self._normalize_path(path),
                    },
                )
                continue
            identifiers = self._document_identifiers(document)
            if include_set and include_set.isdisjoint(identifiers):
                logger.debug(
                    "Excluding document due to include filter",
                    extra={
                        "document_id": document.get("id"),
                        "identifiers": sorted(identifiers),
                        "include_filter": sorted(include_set),
                    },
                )
                continue
            if exclude_set and not exclude_set.isdisjoint(identifiers):
                logger.debug(
                    "Excluding document due to exclude filter",
                    extra={
                        "document_id": document.get("id"),
                        "identifiers": sorted(identifiers),
                        "exclude_filter": sorted(exclude_set),
                    },
                )
                continue
            chunk: dict[str, Any] = record.get("chunk") or {}
            chunk_id = int(chunk.get("id", -1)) if isinstance(chunk, dict) else -1
            if chunk_id in seen_chunks:
                logger.debug(
                    "Skipping duplicate chunk in context records",
                    extra={
                        "chunk_id": chunk_id,
                        "document_id": document.get("id"),
                    },
                )
                continue
            text = chunk.get("text") if isinstance(chunk, dict) else None
            if not text:
                logger.debug(
                    "Skipping chunk without text",
                    extra={
                        "chunk_id": chunk_id,
                        "document_id": document.get("id"),
                    },
                )
                continue
            highlight = self._build_evidence_snippet(
                query,
                chunk_text=text,
                highlight_html=(
                    record.get("highlight")
                    or chunk.get("highlight")
                    or doc_payload.get("preview")
                ),
                preview=doc_payload.get("preview"),
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
            logger.debug(
                "Collected context record",
                extra={
                    "chunk_id": chunk_id,
                    "document_id": document.get("id"),
                    "context_length": len(text or ""),
                    "highlight_length": len(highlight or ""),
                },
            )
            if len(records) >= limit:
                break
        logger.info(
            "Collected RAG context records",
            extra={
                "query_preview": query.strip()[:120],
                "project_id": project_id,
                "record_count": len(records),
                "candidate_count": len(candidate_documents),
            },
        )
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
        logger.info(
            "Retrieved RAG context snippets",
            extra={
                "query_preview": query.strip()[:120],
                "project_id": project_id,
                "snippet_count": len(snippets),
            },
        )
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
            logger.debug(
                "Loaded stored retrieval scope",
                extra={
                    "chat_id": chat_id,
                    "stored_scope_keys": sorted(stored_scope.keys()),
                },
            )

        scope_tags = explicit_tags if explicit_tags is not None else stored_scope.get("tags")
        scope_folder = explicit_folder if folder is not None else stored_scope.get("folder")

        if chat_id is not None and save_scope:
            payload = {
                "tags": explicit_tags or [],
                "folder": explicit_folder,
            }
            self.chats.set_query_scope(chat_id, payload)

        logger.info(
            "Resolved retrieval scope",
            extra={
                "chat_id": chat_id,
                "explicit_tags": explicit_tags,
                "explicit_folder": explicit_folder,
                "resolved_tags": scope_tags,
                "resolved_folder": scope_folder,
                "saved": bool(chat_id is not None and save_scope),
            },
        )

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
        if os.name == "nt":
            return normalized.lower()
        return normalized

    @staticmethod
    def _normalize_folder(folder: str | Path | None) -> str | None:
        if folder in (None, ""):
            return None
        return str(Path(folder).expanduser().resolve())

    def _build_evidence_snippet(
        self,
        query: str,
        *,
        chunk_text: str | None,
        highlight_html: str | None,
        preview: str | None,
    ) -> str:
        """Return an evidence snippet that contains meaningful support."""

        chunk_text = (chunk_text or "").strip()
        highlight_html = (highlight_html or "").strip()
        preview = (preview or "").strip()

        base_text = chunk_text or self._strip_tags(highlight_html) or preview
        if not base_text:
            return highlight_html or ""

        keywords = self._extract_marked_terms(highlight_html)
        if not keywords:
            keywords = self._tokenize_query(query)
        if not keywords:
            keywords = self._extract_keywords(base_text)

        snippet = self._select_relevant_text(base_text, keywords)
        if not snippet:
            snippet = base_text

        snippet = self._normalize_whitespace(snippet)
        snippet = self._limit_length(snippet)
        highlighted = self._apply_highlights(snippet, keywords)
        logger.debug(
            "Constructed evidence snippet",
            extra={
                "base_length": len(base_text),
                "keyword_count": len(keywords),
                "snippet_length": len(highlighted or ""),
            },
        )
        return highlighted

    # ------------------------------------------------------------------
    _TOKEN_PATTERN = re.compile(r"[\w-]+", re.UNICODE)
    _SAFE_MATCH_TOKEN = re.compile(r"^[0-9A-Za-z_]+$")
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
            logger.info("Skipping empty retrieval query")
            return

        attempts: list[str] = []
        base_terms = [match.group(0) for match in self._TOKEN_PATTERN.finditer(normalized)]
        sanitized_terms = [self._escape_match_token(term) for term in base_terms if term]
        logger.debug(
            "Tokenised retrieval query",
            extra={
                "normalized_query": normalized,
                "base_terms": base_terms,
                "sanitized_terms": sanitized_terms,
            },
        )
        if sanitized_terms:
            attempts.append(" ".join(sanitized_terms))
        else:
            attempts.append(normalized)

        tokens = self._tokenize_query(normalized)
        if tokens:
            escaped_tokens = [self._escape_match_token(token) for token in tokens]
            wildcard_tokens = [f"{token}*" for token in escaped_tokens]
            attempts.append(" OR ".join(wildcard_tokens))
            if len(wildcard_tokens) > 1:
                attempts.append(" ".join(wildcard_tokens))

        logger.debug(
            "Constructed retrieval attempts",
            extra={
                "attempts": attempts,
                "token_count": len(tokens),
            },
        )
        logger.info(
            "Preparing retrieval query attempts",
            extra={
                "original_query": normalized,
                "token_count": len(tokens),
                "attempt_count": len(attempts),
            },
        )

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
            logger.info(
                "Executing retrieval attempt",
                extra={
                    "attempt_query": candidate,
                    "attempt_index": len(seen_queries),
                    "limit": limit,
                },
            )
            try:
                results = self.ingest.search(candidate, limit=limit)
            except sqlite3.OperationalError as exc:
                logger.warning(
                    "Ingest search failed for attempt",
                    extra={"attempt_query": candidate, "error": str(exc)},
                )
                continue
            logger.info(
                "Retrieved ingest results",
                extra={
                    "attempt_query": candidate,
                    "result_count": len(results),
                },
            )
            logger.debug(
                "Processing ingest results",
                extra={
                    "attempt_query": candidate,
                    "result_sample": results[:3],
                },
            )
            for record in results:
                identity = _result_identity(record)
                if identity in seen_results:
                    logger.debug(
                        "Skipping duplicate ingest result",
                        extra={
                            "attempt_query": candidate,
                            "identity": identity,
                        },
                    )
                    continue
                seen_results.add(identity)
                yield record
                yielded += 1
                if yielded >= limit:
                    logger.info(
                        "Reached retrieval limit",
                        extra={
                            "attempts_executed": len(seen_queries),
                            "yielded": yielded,
                        },
                    )
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

    @classmethod
    def _escape_match_token(cls, token: str) -> str:
        """Escape tokens so they are safe for use in SQLite FTS MATCH queries."""

        if not token:
            return token
        if cls._SAFE_MATCH_TOKEN.fullmatch(token):
            return token
        escaped = token.replace("\"", "\"\"")
        return f'"{escaped}"'

    @staticmethod
    def _strip_tags(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"<[^>]+>", "", text)

    @staticmethod
    def _extract_marked_terms(highlight_html: str) -> list[str]:
        if not highlight_html:
            return []
        matches = re.findall(r"<mark>(.*?)</mark>", highlight_html, flags=re.IGNORECASE)
        terms: list[str] = []
        seen: set[str] = set()
        for match in matches:
            token = SearchService._normalize_whitespace(match)
            if not token:
                continue
            lowered = token.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            terms.append(token)
        return terms

    def _extract_keywords(self, text: str) -> list[str]:
        tokens = [match.group(0) for match in self._TOKEN_PATTERN.finditer(text)]
        keywords: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            lowered = token.lower()
            if lowered in self._STOPWORDS or len(lowered) < 3:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            keywords.append(token)
        return keywords

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _select_relevant_text(self, text: str, keywords: Iterable[str]) -> str:
        normalized = self._normalize_whitespace(text)
        if not normalized:
            return ""
        sentences = self._split_sentences(normalized)
        if not sentences:
            return normalized
        keyword_list = [term.lower() for term in keywords if term]
        keyword_list = [term for i, term in enumerate(keyword_list) if term not in keyword_list[:i]]
        if not keyword_list:
            snippet = sentences[0]
            if len(snippet) < 80 and len(sentences) > 1:
                snippet = " ".join(sentences[:2])
            return snippet

        matched_indices: list[int] = []
        for idx, sentence in enumerate(sentences):
            lower_sentence = sentence.lower()
            if any(keyword in lower_sentence for keyword in keyword_list):
                matched_indices.append(idx)

        if not matched_indices:
            snippet = sentences[0]
            if len(snippet) < 80 and len(sentences) > 1:
                snippet = " ".join(sentences[:2])
            return snippet

        start = matched_indices[0]
        end = matched_indices[-1] + 1
        snippet = " ".join(sentences[start:end])
        while len(snippet) < 120 and start > 0:
            start -= 1
            snippet = " ".join(sentences[start:end])
        while len(snippet) < 120 and end < len(sentences):
            end += 1
            snippet = " ".join(sentences[start:end])
        return snippet

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text)
        sentences = [part.strip() for part in parts if part.strip()]
        return sentences if sentences else [text]

    @staticmethod
    def _limit_length(text: str, *, max_length: int = 500) -> str:
        if len(text) <= max_length:
            return text
        truncated = text[:max_length].rstrip()
        if not truncated.endswith("…"):
            truncated = truncated.rstrip(".,;: ") + "…"
        return truncated

    @staticmethod
    def _apply_highlights(text: str, keywords: Iterable[str]) -> str:
        terms = [term for term in keywords if term]
        if not terms:
            return text
        unique_terms: list[str] = []
        seen: set[str] = set()
        for term in terms:
            lowered = term.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique_terms.append(term)
        lower_text = text.lower()
        spans: list[tuple[int, int]] = []

        def _overlaps(start: int, end: int) -> bool:
            for existing_start, existing_end in spans:
                if start < existing_end and end > existing_start:
                    return True
            return False

        for term in sorted(unique_terms, key=lambda value: len(value), reverse=True):
            search = term.lower()
            start_index = 0
            while start_index < len(lower_text):
                match_index = lower_text.find(search, start_index)
                if match_index == -1:
                    break
                end_index = match_index + len(search)
                if not _overlaps(match_index, end_index):
                    spans.append((match_index, end_index))
                start_index = end_index

        if not spans:
            return text

        spans.sort()
        parts: list[str] = []
        cursor = 0
        for start, end in spans:
            parts.append(text[cursor:start])
            parts.append(f"<mark>{text[start:end]}</mark>")
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts)
