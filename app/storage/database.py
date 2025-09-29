"""Database utilities and migration helpers for the storage layer."""

from __future__ import annotations

import contextlib
import datetime as _dt
import difflib
import json
import re
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

if TYPE_CHECKING:
    from app.ingest.parsers import ParsedDocument

from app.ingest.parsers import DocumentSection

SCHEMA_VERSION = 5
SCHEMA_FILENAME = "schema.sql"


class DatabaseError(RuntimeError):
    """Raised when database bootstrap or migrations fail."""


class DatabaseManager:
    """Manage the SQLite database connection and schema migrations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection_lock = threading.RLock()
        self._connections: dict[int, sqlite3.Connection] = {}

    def connect(self) -> sqlite3.Connection:
        """Return a singleton SQLite connection with sensible defaults."""
        thread_id = threading.get_ident()
        with self._connection_lock:
            connection = self._connections.get(thread_id)
            if connection is None:
                connection = sqlite3.connect(self.path, check_same_thread=False)
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                self._connections[thread_id] = connection
            return connection

    def close(self) -> None:
        """Close the underlying database connection if it exists."""
        with self._connection_lock:
            connections = list(self._connections.values())
            self._connections.clear()
        for connection in connections:
            connection.close()

    def initialize(self) -> None:
        """Bootstrap the database schema, applying migrations if required."""
        connection = self.connect()
        try:
            with connection:  # Start a transaction so initialization is atomic.
                version = self._get_user_version(connection)
                if version > SCHEMA_VERSION:
                    raise DatabaseError(
                        "Database schema version is newer than this application supports"
                    )
                if version == 0:
                    self._install_base_schema(connection)
                elif version < SCHEMA_VERSION:
                    self._apply_migrations(connection, version)
        except sqlite3.DatabaseError as exc:  # pragma: no cover - defensive
            raise DatabaseError(str(exc)) from exc

    def _install_base_schema(self, connection: sqlite3.Connection) -> None:
        schema_path = Path(__file__).with_name(SCHEMA_FILENAME)
        schema_sql = schema_path.read_text(encoding="utf-8")
        connection.executescript(schema_sql)
        self._set_user_version(connection, SCHEMA_VERSION)

    def _apply_migrations(self, connection: sqlite3.Connection, current: int) -> None:
        """Placeholder for future migrations from ``current`` to ``SCHEMA_VERSION``."""
        if current >= SCHEMA_VERSION:
            return
        # No incremental migrations yet; reapply schema to fill gaps.
        self._install_base_schema(connection)

    @staticmethod
    def _get_user_version(connection: sqlite3.Connection) -> int:
        cursor = connection.execute("PRAGMA user_version")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _set_user_version(connection: sqlite3.Connection, version: int) -> None:
        connection.execute(f"PRAGMA user_version = {version}")

    @contextlib.contextmanager
    def transaction(self) -> Iterable[sqlite3.Connection]:
        """Context manager that wraps operations in a transaction."""
        connection = self.connect()
        try:
            with connection:
                yield connection
        except sqlite3.DatabaseError as exc:  # pragma: no cover - defensive
            raise DatabaseError(str(exc)) from exc

    def export_database(self, destination: str | Path) -> Path:
        """Create a consistent snapshot of the database at ``destination``."""
        destination_path = Path(destination)
        if destination_path.is_dir():
            timestamp = _dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            destination_path = destination_path / f"{self.path.stem}-backup-{timestamp}.db"
        if not destination_path.parent.exists():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
        connection = self.connect()
        with sqlite3.connect(destination_path) as backup_conn:
            connection.backup(backup_conn)
        return destination_path

    def import_database(self, source: str | Path) -> Path:
        """Replace the current database with ``source`` ensuring schema compatibility."""
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        self.close()
        staging_path = self.path.with_suffix(".staging")
        shutil.copy2(source_path, staging_path)
        with sqlite3.connect(staging_path) as staging_conn:
            version = self._get_user_version(staging_conn)
            if version > SCHEMA_VERSION:
                raise DatabaseError(
                    "Imported database has a newer schema version than supported"
                )
        staging_path.replace(self.path)
        self.initialize()
        return self.path


class BaseRepository:
    """Common utilities shared by repository implementations."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    @contextlib.contextmanager
    def transaction(self) -> Iterable[sqlite3.Connection]:
        with self.db.transaction() as connection:
            yield connection

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {key: row[key] for key in row.keys()}


class ProjectRepository(BaseRepository):
    """CRUD helpers for project rows."""

    def create(self, name: str, description: str | None = None) -> dict[str, Any]:
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO projects (name, description) VALUES (?, ?)",
                (name, description),
            )
            project_id = cursor.lastrowid
            return self.get(project_id)  # type: ignore[return-value]

    def get(self, project_id: int) -> dict[str, Any] | None:
        connection = self.db.connect()
        row = connection.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        return self._row_to_dict(row)

    def list(self) -> list[dict[str, Any]]:
        connection = self.db.connect()
        rows = connection.execute(
            "SELECT * FROM projects ORDER BY created_at ASC"
        ).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]  # type: ignore[list-item]

    def update(self, project_id: int, **fields: Any) -> dict[str, Any] | None:
        if not fields:
            return self.get(project_id)
        columns = ", ".join(f"{key} = ?" for key in fields.keys())
        values = list(fields.values()) + [project_id]
        with self.transaction() as connection:
            connection.execute(f"UPDATE projects SET {columns} WHERE id = ?", values)
        return self.get(project_id)

    def delete(self, project_id: int) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))


class DocumentRepository(BaseRepository):
    """Manage documents, file versions, tags, and embeddings."""

    def create(
        self,
        project_id: int,
        title: str,
        *,
        source_type: str | None = None,
        source_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        folder_path: str | Path | None = None,
    ) -> dict[str, Any]:
        metadata_json = json.dumps(metadata) if metadata is not None else None
        stored_path = self._normalize_path(source_path)
        folder = self._normalize_folder(folder_path)
        if folder is None and stored_path is not None:
            folder = self._normalize_folder(Path(stored_path).parent)
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO documents (
                    project_id, title, source_type, source_path, folder_path, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (project_id, title, source_type, stored_path, folder, metadata_json),
            )
            document_id = cursor.lastrowid
        return self.get(document_id)  # type: ignore[return-value]

    def get(self, document_id: int) -> dict[str, Any] | None:
        row = self.db.connect().execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        return self._decode_document_row(row)

    def list_for_project(self, project_id: int) -> list[dict[str, Any]]:
        rows = self.db.connect().execute(
            "SELECT * FROM documents WHERE project_id = ? ORDER BY created_at ASC",
            (project_id,),
        ).fetchall()
        return [record for record in (self._decode_document_row(row) for row in rows) if record]

    def list_for_folder(
        self,
        project_id: int,
        folder_path: str | Path,
        *,
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        return self.list_for_scope(project_id, folder=folder_path, recursive=recursive)

    def list_for_tag(self, project_id: int, tag_id: int) -> list[dict[str, Any]]:
        return self.list_for_scope(project_id, tags=[tag_id])

    def list_for_scope(
        self,
        project_id: int,
        *,
        tags: Iterable[int] | None = None,
        folder: str | Path | None = None,
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        tag_ids = list(dict.fromkeys(tags or []))
        connection = self.db.connect()
        params: list[Any] = [project_id]
        where_clauses = ["documents.project_id = ?"]
        joins: list[str] = []
        group_by = ""
        having = ""

        if tag_ids:
            joins.append("INNER JOIN tag_links ON tag_links.document_id = documents.id")
            placeholders = ",".join("?" for _ in tag_ids)
            where_clauses.append(f"tag_links.tag_id IN ({placeholders})")
            params.extend(tag_ids)
            group_by = " GROUP BY documents.id"
            having = " HAVING COUNT(DISTINCT tag_links.tag_id) = ?"

        normalized_folder = self._normalize_folder(folder) if folder is not None else None
        if normalized_folder is not None:
            if recursive:
                where_clauses.append(
                    "(documents.folder_path = ? OR documents.folder_path LIKE ? ESCAPE '\\')"
                )
                params.extend(
                    [
                        normalized_folder,
                        self._build_folder_like_pattern(normalized_folder),
                    ]
                )
            else:
                where_clauses.append("documents.folder_path = ?")
                params.append(normalized_folder)

        if tag_ids:
            params.append(len(tag_ids))

        query = "SELECT documents.* FROM documents"
        if joins:
            query = " ".join([query] + joins)
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += group_by + having + " ORDER BY documents.created_at ASC"

        rows = connection.execute(query, params).fetchall()
        return [record for record in (self._decode_document_row(row) for row in rows) if record]

    def update(self, document_id: int, **fields: Any) -> dict[str, Any] | None:
        if not fields:
            return self.get(document_id)

        updates: list[str] = []
        values: list[Any] = []

        if "metadata" in fields:
            metadata_value = fields["metadata"]
            fields["metadata"] = json.dumps(metadata_value) if metadata_value is not None else None

        if "source_path" in fields:
            normalized = self._normalize_path(fields["source_path"])
            fields["source_path"] = normalized
            if "folder_path" not in fields:
                folder_value: str | Path | None
                if normalized is not None:
                    folder_value = Path(normalized).parent
                else:
                    folder_value = None
                fields["folder_path"] = self._normalize_folder(folder_value)

        if "folder_path" in fields:
            fields["folder_path"] = self._normalize_folder(fields["folder_path"])

        for key, value in fields.items():
            updates.append(f"{key} = ?")
            values.append(value)
        values.append(document_id)

        with self.transaction() as connection:
            connection.execute(
                f"UPDATE documents SET {', '.join(updates)} WHERE id = ?",
                values,
            )
        return self.get(document_id)

    def delete(self, document_id: int) -> None:
        tag_ids = [tag["id"] for tag in self.list_tags_for_document(document_id)]
        with self.transaction() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            if tag_ids:
                connection.executemany(
                    """
                    UPDATE tags
                    SET document_count = CASE
                        WHEN document_count > 0 THEN document_count - 1
                        ELSE 0
                    END
                    WHERE id = ?
                    """,
                    [(tag_id,) for tag_id in set(tag_ids)],
                )

    def add_file_version(
        self,
        document_id: int,
        *,
        file_path: str | Path,
        checksum: str | None = None,
        file_size: int | None = None,
    ) -> dict[str, Any]:
        connection = self.db.connect()
        row = connection.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM file_versions WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        version = int(row[0]) if row else 1
        with self.transaction() as tx:
            cursor = tx.execute(
                """
                INSERT INTO file_versions (document_id, version, file_path, checksum, file_size)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    version,
                    str(Path(file_path).resolve()),
                    checksum,
                    file_size,
                ),
            )
            version_id = cursor.lastrowid
        return self.get_file_version(version_id)  # type: ignore[return-value]

    def get_file_version(self, version_id: int) -> dict[str, Any] | None:
        row = self.db.connect().execute(
            "SELECT * FROM file_versions WHERE id = ?", (version_id,)
        ).fetchone()
        return self._row_to_dict(row)

    def list_file_versions(self, document_id: int) -> list[dict[str, Any]]:
        rows = self.db.connect().execute(
            "SELECT * FROM file_versions WHERE document_id = ? ORDER BY version ASC",
            (document_id,),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]  # type: ignore[list-item]

    def create_tag(
        self, project_id: int, name: str, description: str | None = None, color: str | None = None
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO tags (project_id, name, description, color) VALUES (?, ?, ?, ?)",
                (project_id, name, description, color),
            )
            tag_id = cursor.lastrowid
        return self.get_tag(tag_id)  # type: ignore[return-value]

    def get_tag(self, tag_id: int) -> dict[str, Any] | None:
        row = self.db.connect().execute(
            "SELECT * FROM tags WHERE id = ?", (tag_id,)
        ).fetchone()
        return self._row_to_dict(row)

    def tag_document(self, document_id: int, tag_id: int) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO tag_links (tag_id, document_id) VALUES (?, ?)",
                (tag_id, document_id),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE tags SET document_count = document_count + 1 WHERE id = ?",
                    (tag_id,),
                )

    def untag_document(self, document_id: int, tag_id: int) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM tag_links WHERE tag_id = ? AND document_id = ?",
                (tag_id, document_id),
            )
            if cursor.rowcount:
                connection.execute(
                    """
                    UPDATE tags
                    SET document_count = CASE
                        WHEN document_count > 0 THEN document_count - 1
                        ELSE 0
                    END
                    WHERE id = ?
                    """,
                    (tag_id,),
                )

    def list_tags_for_document(self, document_id: int) -> list[dict[str, Any]]:
        rows = self.db.connect().execute(
            """
            SELECT tags.*
            FROM tags
            INNER JOIN tag_links ON tag_links.tag_id = tags.id
            WHERE tag_links.document_id = ?
            ORDER BY tags.name ASC
            """,
            (document_id,),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]  # type: ignore[list-item]

    def list_tags_for_project(self, project_id: int) -> list[dict[str, Any]]:
        rows = self.db.connect().execute(
            "SELECT * FROM tags WHERE project_id = ? ORDER BY name ASC",
            (project_id,),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]  # type: ignore[list-item]

    def delete_tag(self, tag_id: int) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM tags WHERE id = ?", (tag_id,))

    def refresh_tag_counts(self, project_id: int | None = None) -> None:
        connection = self.db.connect()
        if project_id is None:
            connection.execute(
                """
                UPDATE tags
                SET document_count = (
                    SELECT COUNT(*) FROM tag_links WHERE tag_links.tag_id = tags.id
                )
                """
            )
            return
        connection.execute(
            """
            UPDATE tags
            SET document_count = (
                SELECT COUNT(*)
                FROM tag_links
                INNER JOIN documents ON documents.id = tag_links.document_id
                WHERE tag_links.tag_id = tags.id AND documents.project_id = ?
            )
            WHERE project_id = ?
            """,
            (project_id, project_id),
        )

    @staticmethod
    def _normalize_path(path: str | Path | None) -> str | None:
        if path is None:
            return None
        return str(Path(path).resolve())

    @staticmethod
    def _normalize_folder(path: str | Path | None) -> str | None:
        if path in (None, ""):
            return None
        return str(Path(path).resolve())

    @staticmethod
    def _build_folder_like_pattern(folder: str) -> str:
        sanitized = folder.replace("%", "\\%").replace("_", "\\_")
        if sanitized.endswith(("/", "\\")):
            return f"{sanitized}%"
        separator = "\\" if "\\" in folder and "/" not in folder else "/"
        return f"{sanitized}{separator}%"

    @staticmethod
    def _decode_document_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        record = {key: row[key] for key in row.keys()}
        if record.get("metadata"):
            record["metadata"] = json.loads(record["metadata"])
        return record


class IngestDocumentRepository(BaseRepository):
    """Manage parsed document text, previews, and search indexes."""

    def store_version(
        self,
        *,
        path: str,
        checksum: str | None,
        size: int | None,
        mtime: float | None,
        ctime: float | None,
        parsed: "ParsedDocument",
        base_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_path = str(Path(path).resolve())
        metadata = dict(base_metadata or {})
        metadata.update(parsed.metadata)
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        sections_json = json.dumps(
            [section.to_dict() for section in parsed.sections], ensure_ascii=False
        )
        pages_json = json.dumps([page.to_dict() for page in parsed.pages], ensure_ascii=False)
        normalized_text = self._normalize_text(parsed.text)
        preview = self._build_preview(normalized_text)
        ocr_message = parsed.ocr_hint if parsed.needs_ocr else None

        with self.transaction() as connection:
            existing_rows = connection.execute(
                "SELECT id FROM ingest_documents WHERE path = ?",
                (normalized_path,),
            ).fetchall()
            previous_ids = [int(row["id"]) for row in existing_rows]
            row = connection.execute(
                """
                SELECT version
                FROM ingest_documents
                WHERE path = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (normalized_path,),
            ).fetchone()
            version = int(row["version"]) + 1 if row else 1
            cursor = connection.execute(
                """
                INSERT INTO ingest_documents (
                    path, version, checksum, size, mtime, ctime, metadata, text,
                    normalized_text, preview, sections, pages, needs_ocr, ocr_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_path,
                    version,
                    checksum,
                    size,
                    mtime,
                    ctime,
                    metadata_json,
                    parsed.text,
                    normalized_text,
                    preview,
                    sections_json,
                    pages_json,
                    int(parsed.needs_ocr),
                    ocr_message,
                ),
            )
            document_id = cursor.lastrowid
            if previous_ids:
                self._delete_chunks_for_documents(connection, previous_ids)
            self._replace_chunks(
                connection,
                document_id,
                normalized_path,
                parsed,
                normalized_text=normalized_text,
            )
        return self.get(document_id)  # type: ignore[return-value]

    def get(self, document_id: int) -> dict[str, Any] | None:
        row = self.db.connect().execute(
            "SELECT * FROM ingest_documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        return self._decode_document_row(row)

    def get_latest_by_path(self, path: str | Path) -> dict[str, Any] | None:
        normalized_path = str(Path(path).resolve())
        row = self.db.connect().execute(
            """
            SELECT * FROM ingest_documents
            WHERE path = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (normalized_path,),
        ).fetchone()
        return self._decode_document_row(row)

    def list_versions(self, path: str | Path) -> list[dict[str, Any]]:
        normalized_path = str(Path(path).resolve())
        rows = self.db.connect().execute(
            """
            SELECT * FROM ingest_documents
            WHERE path = ?
            ORDER BY version DESC
            """,
            (normalized_path,),
        ).fetchall()
        return [record for record in (self._decode_document_row(row) for row in rows) if record]

    def list_all(self) -> list[dict[str, Any]]:
        rows = self.db.connect().execute(
            "SELECT * FROM ingest_documents ORDER BY created_at ASC",
        ).fetchall()
        return [record for record in (self._decode_document_row(row) for row in rows) if record]

    def delete_by_paths(self, paths: Iterable[str | Path]) -> int:
        """Remove all ingest records associated with ``paths``.

        Returns the number of rows deleted across both the ingest table and the
        associated FTS index. Paths are normalised to absolute form before
        matching so callers can provide either relative or absolute values.
        """

        normalized = [str(Path(path).resolve()) for path in paths]
        unique_paths = list(dict.fromkeys(normalized))
        if not unique_paths:
            return 0

        removed = 0
        with self.transaction() as connection:
            for path in unique_paths:
                rows = connection.execute(
                    "SELECT id FROM ingest_documents WHERE path = ?",
                    (path,),
                ).fetchall()
                document_ids = [int(row["id"]) for row in rows]
                if not document_ids:
                    continue
                self._delete_chunks_for_documents(connection, document_ids)
                placeholders = ",".join("?" for _ in document_ids)
                connection.execute(
                    f"DELETE FROM ingest_documents WHERE id IN ({placeholders})",
                    document_ids,
                )
                removed += len(document_ids)
        return removed

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        return self.search_chunks(query, limit=limit)

    def search_chunks(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.db.connect().execute(
            """
            SELECT
                ingest_document_index.rowid AS chunk_id,
                ingest_document_index.document_id AS document_id,
                ingest_document_index.chunk_index AS chunk_index,
                ingest_document_index.path AS path,
                highlight(ingest_document_index, 0, '<mark>', '</mark>') AS snippet,
                bm25(ingest_document_index) AS score,
                chunks.text AS chunk_text,
                chunks.token_count AS token_count,
                chunks.start_offset AS start_offset,
                chunks.end_offset AS end_offset,
                chunks.metadata AS metadata
            FROM ingest_document_index
            INNER JOIN ingest_document_chunks AS chunks ON chunks.id = ingest_document_index.rowid
            WHERE ingest_document_index MATCH ?
            ORDER BY score ASC, chunk_index ASC
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()

        if not rows:
            return []

        document_ids = {int(row["document_id"]) for row in rows}
        documents: dict[int, dict[str, Any]] = {}
        for doc_id in document_ids:
            document = self.get(doc_id)
            if document is not None:
                documents[doc_id] = document

        results: list[dict[str, Any]] = []
        normalized_query = (query or "").lower()
        results: list[dict[str, Any]] = []
        for row in rows:
            doc_id = int(row["document_id"])
            document = documents.get(doc_id)
            if document is None:
                continue
            metadata_raw = row["metadata"]
            chunk_metadata = json.loads(metadata_raw) if metadata_raw else {}
            chunk = {
                "id": int(row["chunk_id"]),
                "document_id": doc_id,
                "index": int(row["chunk_index"]),
                "text": row["chunk_text"],
                "token_count": int(row["token_count"]),
                "start_offset": int(row["start_offset"]),
                "end_offset": int(row["end_offset"]),
                "metadata": chunk_metadata,
            }
            chunk_metadata.setdefault("bm25", float(row["score"]))
            hierarchy_weight = float(chunk_metadata.get("hierarchy_weight", 1.0) or 1.0)
            raw_score = float(row["score"])
            keyword_score = 1.0 / (1.0 + max(raw_score, 0.0))
            semantic_score = 0.0
            chunk_text_lower = (row["chunk_text"] or "").lower()
            if normalized_query and chunk_text_lower:
                matcher = difflib.SequenceMatcher(None, normalized_query, chunk_text_lower)
                semantic_score = matcher.ratio()
            combined_score = (0.5 * keyword_score + 0.5 * semantic_score) * hierarchy_weight
            score_breakdown = {
                "keyword": keyword_score,
                "semantic": semantic_score,
                "hierarchy_weight": hierarchy_weight,
            }
            results.append(
                {
                    "chunk": chunk,
                    "document": document,
                    "highlight": row["snippet"],
                    "score": combined_score,
                    "score_breakdown": score_breakdown,
                    "path": document.get("path"),
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    @staticmethod
    def _normalize_text(text: str) -> str:
        collapsed = re.sub(r"\s+", " ", text.strip())
        return collapsed

    def _delete_chunks_for_documents(
        self, connection: sqlite3.Connection, document_ids: Iterable[int]
    ) -> None:
        doc_ids = [int(doc_id) for doc_id in document_ids]
        if not doc_ids:
            return
        placeholders = ",".join("?" for _ in doc_ids)
        rows = connection.execute(
            f"SELECT id FROM ingest_document_chunks WHERE document_id IN ({placeholders})",
            doc_ids,
        ).fetchall()
        chunk_ids = [int(row["id"]) for row in rows]
        if not chunk_ids:
            return
        chunk_placeholders = ",".join("?" for _ in chunk_ids)
        connection.execute(
            f"DELETE FROM ingest_document_index WHERE rowid IN ({chunk_placeholders})",
            chunk_ids,
        )
        connection.execute(
            f"DELETE FROM ingest_document_chunks WHERE id IN ({chunk_placeholders})",
            chunk_ids,
        )

    def _replace_chunks(
        self,
        connection: sqlite3.Connection,
        document_id: int,
        path: str,
        parsed: "ParsedDocument",
        *,
        normalized_text: str,
    ) -> None:
        chunks = self._semantic_chunk_document(parsed, normalized_text=normalized_text, path=path)
        connection.execute(
            "DELETE FROM ingest_document_chunks WHERE document_id = ?",
            (document_id,),
        )
        for chunk in chunks:
            cursor = connection.execute(
                """
                INSERT INTO ingest_document_chunks (
                    document_id, chunk_index, text, token_count, start_offset, end_offset, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    chunk["index"],
                    chunk["text"],
                    chunk["token_count"],
                    chunk["start_offset"],
                    chunk["end_offset"],
                    json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                ),
            )
            chunk_id = cursor.lastrowid
            connection.execute(
                """
                INSERT INTO ingest_document_index (
                    rowid, content, path, document_id, chunk_id, chunk_index
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    chunk["search_text"],
                    path,
                    document_id,
                    chunk_id,
                    chunk["index"],
                ),
            )

    def _semantic_chunk_document(
        self,
        parsed: "ParsedDocument",
        *,
        normalized_text: str,
        path: str,
        min_chars: int = 400,
    ) -> list[dict[str, Any]]:
        sections: list[DocumentSection] = list(parsed.sections or [])
        if not sections:
            sections = [
                DocumentSection(
                    title=None,
                    content=parsed.text,
                    level=None,
                    page_number=1,
                    start_offset=0,
                    end_offset=len(parsed.text),
                    line_start=1 if parsed.text else None,
                    line_end=parsed.text.count("\n") + 1 if parsed.text else None,
                )
            ]

        section_paths = self._build_section_paths(sections)
        windows: list[list[int]] = []
        index = 0
        total = len(sections)
        while index < total:
            section = sections[index]
            level = section.level if section.level is not None else 6
            if section.title and level <= 3:
                next_index = index + 1
                while next_index < total:
                    candidate = sections[next_index]
                    candidate_level = candidate.level if candidate.level is not None else 6
                    if candidate.title and candidate_level <= level:
                        break
                    next_index += 1
                windows.append(list(range(index, next_index)))
                index = next_index
            else:
                windows.append([index])
                index += 1

        def window_length(indices: Sequence[int]) -> int:
            return len(self._render_sections(sections, indices))

        final_windows: list[list[int]] = []
        buffer: list[int] | None = None
        for window in windows:
            if buffer is None:
                buffer = list(window)
                continue
            if window_length(buffer) < min_chars:
                buffer.extend(window)
                continue
            final_windows.append(buffer)
            buffer = list(window)

        if buffer is not None:
            if final_windows and window_length(buffer) < min_chars:
                final_windows[-1].extend(buffer)
            else:
                final_windows.append(buffer)

        if not final_windows:
            final_windows = [[0]]

        chunks: list[dict[str, Any]] = []
        chunk_index = 0
        for indices in final_windows:
            chunk_text = self._render_sections(sections, indices)
            if not chunk_text.strip():
                continue
            start_offset = min(
                (sections[i].start_offset for i in indices if sections[i].start_offset is not None),
                default=0,
            )
            end_offset = max(
                (sections[i].end_offset for i in indices if sections[i].end_offset is not None),
                default=start_offset + len(chunk_text),
            )
            token_count = len(chunk_text.split())
            metadata = self._build_chunk_metadata(sections, section_paths, indices, path)
            metadata.setdefault("token_count", token_count)
            search_terms = " ".join(metadata.get("section_path", []))
            search_basis = f"{search_terms} {chunk_text}" if search_terms else chunk_text
            chunks.append(
                {
                    "index": chunk_index,
                    "text": chunk_text,
                    "token_count": token_count,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "search_text": self._normalize_text(search_basis or normalized_text),
                    "metadata": metadata,
                }
            )
            chunk_index += 1

        if not chunks:
            chunks.append(
                {
                    "index": 0,
                    "text": parsed.text or normalized_text,
                    "token_count": len((parsed.text or "").split()),
                    "start_offset": 0,
                    "end_offset": len(parsed.text or normalized_text),
                    "search_text": self._normalize_text(parsed.text or normalized_text),
                    "metadata": {
                        "section_path": [],
                        "sections": [],
                        "hierarchy_weight": 1.0,
                        "source_path": path,
                    },
                }
            )
        return chunks

    @staticmethod
    def _build_section_paths(sections: Sequence[DocumentSection]) -> list[list[str]]:
        paths: list[list[str]] = []
        stack: list[tuple[int, str]] = []
        for section in sections:
            level = section.level if section.level is not None else 6
            if section.title:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, section.title))
            paths.append([title for _, title in stack])
        return paths

    @staticmethod
    def _render_sections(
        sections: Sequence[DocumentSection], indices: Sequence[int]
    ) -> str:
        parts: list[str] = []
        for index in indices:
            section = sections[index]
            local_parts: list[str] = []
            if section.title:
                level = section.level if section.level and section.level > 0 else 1
                prefix = "#" * min(level, 6)
                header = f"{prefix} {section.title}" if prefix else section.title
                local_parts.append(header.strip())
            if section.content:
                local_parts.append(section.content.strip("\n"))
            if local_parts:
                parts.append("\n".join(local_parts))
        return "\n\n".join(parts).strip()

    def _build_chunk_metadata(
        self,
        sections: Sequence[DocumentSection],
        section_paths: Sequence[Sequence[str]],
        indices: Sequence[int],
        source_path: str,
    ) -> dict[str, Any]:
        first_idx = indices[0]
        path = list(section_paths[first_idx])
        hierarchy_weight = self._hierarchy_weight(path)
        page_numbers = [
            sections[i].page_number for i in indices if sections[i].page_number is not None
        ]
        line_start = min(
            (sections[i].line_start for i in indices if sections[i].line_start is not None),
            default=None,
        )
        line_end = max(
            (sections[i].line_end for i in indices if sections[i].line_end is not None),
            default=None,
        )
        start_offset = min(
            (sections[i].start_offset for i in indices if sections[i].start_offset is not None),
            default=None,
        )
        end_offset = max(
            (sections[i].end_offset for i in indices if sections[i].end_offset is not None),
            default=None,
        )
        section_summaries = [
            {
                "index": i,
                "title": sections[i].title,
                "level": sections[i].level,
                "path": list(section_paths[i]),
                "page": sections[i].page_number,
                "line_start": sections[i].line_start,
                "line_end": sections[i].line_end,
            }
            for i in indices
        ]
        return {
            "section_path": path,
            "sections": section_summaries,
            "hierarchy_weight": hierarchy_weight,
            "page_range": {
                "start": min(page_numbers) if page_numbers else None,
                "end": max(page_numbers) if page_numbers else None,
            },
            "position": {
                "start_offset": start_offset,
                "end_offset": end_offset,
                "line_start": line_start,
                "line_end": line_end,
            },
            "source_path": source_path,
        }

    @staticmethod
    def _hierarchy_weight(path: Sequence[str]) -> float:
        if not path:
            return 1.0
        title = path[-1].lower()
        if any(keyword in title for keyword in ("appendix", "annex", "supplement")):
            base = 0.6
        elif any(keyword in title for keyword in ("reference", "bibliography")):
            base = 0.7
        elif any(keyword in title for keyword in ("abstract", "summary")):
            base = 0.85
        else:
            base = 1.0
        depth_penalty = min(0.05 * max(len(path) - 1, 0), 0.25)
        return max(0.4, base - depth_penalty)

    @staticmethod
    def _build_preview(text: str, *, limit: int = 320) -> str:
        if len(text) <= limit:
            return text
        cutoff = text.rfind(" ", 0, limit)
        if cutoff == -1:
            cutoff = limit
        return text[:cutoff].rstrip() + "…"

    @staticmethod
    def _decode_document_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        record = {key: row[key] for key in row.keys()}
        for key in ("metadata", "sections", "pages"):
            if record.get(key):
                record[key] = json.loads(record[key])
        record["needs_ocr"] = bool(record.get("needs_ocr"))
        return record

class ChatRepository(BaseRepository):
    """Manage chat sessions, citations, and reasoning summaries."""

    def create(
        self,
        project_id: int,
        title: str | None = None,
        *,
        query_scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = json.dumps(query_scope) if query_scope is not None else None
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO chats (project_id, title, query_scope) VALUES (?, ?, ?)",
                (project_id, title, payload),
            )
            chat_id = cursor.lastrowid
        return self.get(chat_id)  # type: ignore[return-value]

    def get(self, chat_id: int) -> dict[str, Any] | None:
        row = self.db.connect().execute(
            "SELECT * FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        return self._decode_chat_row(row)

    def list_for_project(self, project_id: int) -> list[dict[str, Any]]:
        rows = self.db.connect().execute(
            "SELECT * FROM chats WHERE project_id = ? ORDER BY created_at ASC",
            (project_id,),
        ).fetchall()
        return [record for record in (self._decode_chat_row(row) for row in rows) if record]

    def get_query_scope(self, chat_id: int) -> dict[str, Any] | None:
        chat = self.get(chat_id)
        if not chat:
            return None
        scope = chat.get("query_scope")
        return scope if isinstance(scope, dict) else None

    def set_query_scope(
        self, chat_id: int, scope: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        payload = json.dumps(scope) if scope is not None else None
        with self.transaction() as connection:
            connection.execute(
                "UPDATE chats SET query_scope = ? WHERE id = ?",
                (payload, chat_id),
            )
        return self.get(chat_id)

    def delete(self, chat_id: int) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def add_citation(
        self,
        chat_id: int,
        *,
        document_id: int | None = None,
        file_version_id: int | None = None,
        snippet: str | None = None,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO citations (chat_id, document_id, file_version_id, snippet)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, document_id, file_version_id, snippet),
            )
            citation_id = cursor.lastrowid
        return self.get_citation(citation_id)  # type: ignore[return-value]

    def get_citation(self, citation_id: int) -> dict[str, Any] | None:
        row = self.db.connect().execute(
            "SELECT * FROM citations WHERE id = ?",
            (citation_id,),
        ).fetchone()
        return self._row_to_dict(row)

    def list_citations(self, chat_id: int) -> list[dict[str, Any]]:
        rows = self.db.connect().execute(
            "SELECT * FROM citations WHERE chat_id = ? ORDER BY created_at ASC",
            (chat_id,),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]  # type: ignore[list-item]

    def add_reasoning_summary(self, chat_id: int, content: str) -> dict[str, Any]:
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO reasoning_summaries (chat_id, content) VALUES (?, ?)",
                (chat_id, content),
            )
            summary_id = cursor.lastrowid
        return self.get_reasoning_summary(summary_id)  # type: ignore[return-value]

    def get_reasoning_summary(self, summary_id: int) -> dict[str, Any] | None:
        row = self.db.connect().execute(
            "SELECT * FROM reasoning_summaries WHERE id = ?",
            (summary_id,),
        ).fetchone()
        return self._row_to_dict(row)

    def list_reasoning_summaries(self, chat_id: int) -> list[dict[str, Any]]:
        rows = self.db.connect().execute(
            "SELECT * FROM reasoning_summaries WHERE chat_id = ? ORDER BY created_at ASC",
            (chat_id,),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]  # type: ignore[list-item]

    @staticmethod
    def _decode_chat_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        record = {key: row[key] for key in row.keys()}
        if record.get("query_scope"):
            record["query_scope"] = json.loads(record["query_scope"])
        return record


class BackgroundTaskLogRepository(BaseRepository):
    """Persist and query background task execution metadata."""

    TABLE = "background_task_logs"

    def create(
        self,
        task_name: str,
        *,
        status: str,
        message: str | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = json.dumps(extra_data) if extra_data is not None else None
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO background_task_logs (task_name, status, message, extra_data)
                VALUES (?, ?, ?, ?)
                """,
                (task_name, status, message, payload),
            )
            task_id = cursor.lastrowid
        return self.get(task_id)  # type: ignore[return-value]

    def get(self, task_id: int) -> dict[str, Any] | None:
        row = self.db.connect().execute(
            f"SELECT * FROM {self.TABLE} WHERE id = ?",
            (task_id,),
        ).fetchone()
        record = self._row_to_dict(row)
        if record and record.get("extra_data"):
            record["extra_data"] = json.loads(record["extra_data"])
        return record

    def update(
        self,
        task_id: int,
        *,
        status: str | None = None,
        message: str | None = None,
        extra_data: dict[str, Any] | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any] | None:
        updates: list[str] = []
        values: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            values.append(status)
        if message is not None:
            updates.append("message = ?")
            values.append(message)
        if extra_data is not None:
            updates.append("extra_data = ?")
            values.append(json.dumps(extra_data))
        if completed_at is not None:
            updates.append("completed_at = ?")
            values.append(completed_at)
        if not updates:
            return self.get(task_id)
        values.append(task_id)
        with self.transaction() as connection:
            connection.execute(
                f"UPDATE {self.TABLE} SET {', '.join(updates)} WHERE id = ?",
                values,
            )
        return self.get(task_id)

    def list_incomplete(self) -> list[dict[str, Any]]:
        rows = self.db.connect().execute(
            f"""
            SELECT * FROM {self.TABLE}
            WHERE status IN ('queued', 'running', 'paused')
            ORDER BY created_at ASC
            """
        ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = self._row_to_dict(row)
            if record is None:
                continue
            if record.get("extra_data"):
                record["extra_data"] = json.loads(record["extra_data"])
            records.append(record)
        return records

    def find_latest_completed(self, task_name: str) -> dict[str, Any] | None:
        row = self.db.connect().execute(
            f"""
            SELECT * FROM {self.TABLE}
            WHERE task_name = ? AND status = 'completed'
            ORDER BY completed_at DESC NULLS LAST, created_at DESC
            LIMIT 1
            """,
            (task_name,),
        ).fetchone()
        record = self._row_to_dict(row)
        if record and record.get("extra_data"):
            record["extra_data"] = json.loads(record["extra_data"])
        return record

    def list_completed(self, task_name: str) -> list[dict[str, Any]]:
        rows = self.db.connect().execute(
            f"""
            SELECT * FROM {self.TABLE}
            WHERE task_name = ? AND status = 'completed'
            ORDER BY completed_at DESC NULLS LAST, created_at DESC
            """,
            (task_name,),
        ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = self._row_to_dict(row)
            if record and record.get("extra_data"):
                record["extra_data"] = json.loads(record["extra_data"])
            if record:
                records.append(record)
        return records


__all__ = [
    "DatabaseError",
    "DatabaseManager",
    "BaseRepository",
    "ProjectRepository",
    "DocumentRepository",
    "ChatRepository",
    "BackgroundTaskLogRepository",
]
