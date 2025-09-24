"""Database utilities and migration helpers for the storage layer."""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import re
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from app.ingest.parsers import ParsedDocument

SCHEMA_VERSION = 3
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
            connection.execute(
                """
                INSERT INTO ingest_document_index (rowid, content, document_id)
                VALUES (?, ?, ?)
                """,
                (document_id, normalized_text, document_id),
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
                placeholders = ",".join("?" for _ in document_ids)
                connection.execute(
                    f"DELETE FROM ingest_document_index WHERE rowid IN ({placeholders})",
                    document_ids,
                )
                connection.execute(
                    f"DELETE FROM ingest_documents WHERE id IN ({placeholders})",
                    document_ids,
                )
                removed += len(document_ids)
        return removed

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.db.connect().execute(
            """
            SELECT ingest_documents.*, highlight(ingest_document_index, 0, '<mark>', '</mark>') AS snippet
            FROM ingest_document_index
            INNER JOIN ingest_documents ON ingest_documents.id = ingest_document_index.rowid
            WHERE ingest_document_index MATCH ?
            ORDER BY bm25(ingest_document_index)
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            record = self._decode_document_row(row)
            if record is not None:
                record["highlight"] = row["snippet"]
                results.append(record)
        return results

    @staticmethod
    def _normalize_text(text: str) -> str:
        collapsed = re.sub(r"\s+", " ", text.strip())
        return collapsed

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
