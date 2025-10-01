"""Database utilities and migration helpers for the storage layer."""

from __future__ import annotations

import ast
import contextlib
import datetime as _dt
import json
import logging
import re
import shutil
import sqlite3
import threading
from bisect import bisect_left
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from app.ingest.parsers import DocumentSection, ParsedDocument

SCHEMA_VERSION = 4
SCHEMA_FILENAME = "schema.sql"


def _parse_schema_objects() -> dict[str, set[str]]:
    """Extract schema object names from ``schema.sql`` for compatibility checks."""

    schema_path = Path(__file__).with_name(SCHEMA_FILENAME)
    schema_sql = schema_path.read_text(encoding="utf-8")
    patterns = {
        "table": re.compile(
            r"CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>[A-Za-z_][\w]*)",
            re.IGNORECASE,
        ),
        "index": re.compile(
            r"CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>[A-Za-z_][\w]*)",
            re.IGNORECASE,
        ),
        "trigger": re.compile(
            r"CREATE\s+TRIGGER\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>[A-Za-z_][\w]*)",
            re.IGNORECASE,
        ),
    }
    objects: dict[str, set[str]] = {"table": set(), "index": set(), "trigger": set()}
    for kind, pattern in patterns.items():
        for match in pattern.finditer(schema_sql):
            objects[kind].add(match.group("name"))
    return objects


EXPECTED_SCHEMA_OBJECTS = _parse_schema_objects()
logger = logging.getLogger(__name__)


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
                    version = self._handle_newer_schema(connection, version)
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

    def _handle_newer_schema(self, connection: sqlite3.Connection, version: int) -> int:
        """Downgrade ``user_version`` if the schema is still compatible."""

        if not self._is_schema_compatible(connection):
            raise DatabaseError(
                "Database schema version is newer than this application supports"
            )
        logger.warning(
            "Detected newer database schema version %s; resetting to supported version %s",
            version,
            SCHEMA_VERSION,
        )
        self._install_base_schema(connection)
        return SCHEMA_VERSION

    def _is_schema_compatible(self, connection: sqlite3.Connection) -> bool:
        """Check that required schema objects exist in the connected database."""

        existing: dict[str, set[str]] = {"table": set(), "index": set(), "trigger": set()}
        cursor = connection.execute(
            "SELECT type, name FROM sqlite_master WHERE type IN ('table', 'index', 'trigger')"
        )
        for row in cursor.fetchall():
            kind = row["type"]
            if kind in existing:
                existing[kind].add(row["name"])
        for kind, expected in EXPECTED_SCHEMA_OBJECTS.items():
            missing = expected - existing.get(kind, set())
            if missing:
                return False
        return True

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

        logger.info(
            "Storing ingest document version",
            extra={
                "path": normalized_path,
                "checksum": checksum,
                "size": size,
                "metadata_keys": sorted(metadata.keys()),
                "needs_ocr": bool(parsed.needs_ocr),
            },
        )

        chunk_count = 0
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
                logger.info(
                    "Superseding previous ingest versions",
                    extra={
                        "path": normalized_path,
                        "previous_version_count": len(previous_ids),
                        "previous_document_ids": previous_ids[:10],
                    },
                )
                self._delete_chunks_for_documents(connection, previous_ids)
            chunk_count = self._replace_chunks(
                connection,
                document_id,
                normalized_path,
                parsed.text,
                normalized_text,
                metadata,
                parsed.sections,
            )
        logger.info(
            "Stored ingest document version",
            extra={
                "document_id": document_id,
                "path": normalized_path,
                "version": version,
                "chunk_count": chunk_count,
            },
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

        logger.info(
            "Deleting ingest documents by path",
            extra={
                "path_count": len(unique_paths),
                "sample_paths": unique_paths[:10],
            },
        )

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
                logger.info(
                    "Deleted ingest documents",
                    extra={
                        "path": path,
                        "document_count": len(document_ids),
                    },
                )
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
                chunks.end_offset AS end_offset
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
        for row in rows:
            doc_id = int(row["document_id"])
            document = documents.get(doc_id)
            if document is None:
                continue
            chunk = {
                "id": int(row["chunk_id"]),
                "document_id": doc_id,
                "index": int(row["chunk_index"]),
                "text": row["chunk_text"],
                "token_count": int(row["token_count"]),
                "start_offset": int(row["start_offset"]),
                "end_offset": int(row["end_offset"]),
            }
            results.append(
                {
                    "chunk": chunk,
                    "document": document,
                    "highlight": row["snippet"],
                    "score": float(row["score"]),
                    "path": document.get("path"),
                }
            )
        return results

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
        logger.info(
            "Removing existing document chunks",
            extra={
                "document_ids": doc_ids[:10],
                "document_count": len(doc_ids),
                "chunk_count": len(chunk_ids),
            },
        )
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
        text: str,
        normalized_text: str,
        metadata: dict[str, Any] | None,
        sections: Sequence["DocumentSection"] | Sequence[dict[str, Any]] | None = None,
    ) -> int:
        chunks = self._chunk_document(
            text,
            normalized_text=normalized_text,
            path=path,
            metadata=metadata,
            sections=sections,
        )
        logger.info(
            "Persisting document chunks",
            extra={
                "document_id": document_id,
                "path": path,
                "chunk_count": len(chunks),
            },
        )
        connection.execute(
            "DELETE FROM ingest_document_chunks WHERE document_id = ?",
            (document_id,),
        )
        for chunk in chunks:
            cursor = connection.execute(
                """
                INSERT INTO ingest_document_chunks (
                    document_id, chunk_index, text, token_count, start_offset, end_offset
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    chunk["index"],
                    chunk["text"],
                    chunk["token_count"],
                    chunk["start_offset"],
                    chunk["end_offset"],
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
        return len(chunks)

    _CODE_SUFFIXES = {".py", ".pyw", ".m", ".cpp"}
    _HTML_SUFFIXES = {".html", ".htm"}
    _SENTENCE_BOUNDARY_CHARS = ".!?:;"
    _CLOSING_PUNCTUATION = "\"')]}»”"
    _BOUNDARY_LOOKBACK = 1200
    _BOUNDARY_LOOKAHEAD = 1200
    _TEXT_TARGET_TOKENS = 450
    _TEXT_MIN_TOKENS = 180
    _TEXT_MAX_TOKENS = 620

    def _chunk_document(
        self,
        text: str,
        *,
        normalized_text: str,
        path: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_tokens: int = 450,
        overlap: int = 60,
        sections: Sequence["DocumentSection"] | Sequence[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        (
            strategy,
            resolved_max_tokens,
            resolved_overlap,
        ) = self._resolve_chunking_settings(
            path=path,
            metadata=metadata,
            default_max_tokens=max_tokens,
            default_overlap=overlap,
        )

        if strategy == "code":
            chunks = self._chunk_code_document(
                text,
                normalized_text=normalized_text,
                max_tokens=resolved_max_tokens,
                path=path,
            )
        elif strategy == "html":
            chunks = self._chunk_markup_document(
                text,
                normalized_text=normalized_text,
                max_tokens=resolved_max_tokens,
                overlap=resolved_overlap,
                strategy=strategy,
            )
        else:
            resolved_sections = self._resolve_section_spans(text, sections)
            chunks = self._chunk_text_document(
                text,
                normalized_text=normalized_text,
                target_tokens=resolved_max_tokens,
                sections=resolved_sections,
            )

        if not chunks:
            fallback_text = normalized_text or text.strip()
            if not fallback_text:
                fallback_text = text
            chunks = [
                {
                    "index": 0,
                    "text": fallback_text,
                    "token_count": max(1, self._count_tokens(fallback_text)),
                    "start_offset": 0,
                    "end_offset": len(fallback_text),
                    "search_text": self._normalize_text(fallback_text),
                }
            ]

        logger.info(
            "Chunked document",
            extra={
                "path": path,
                "chunk_count": len(chunks),
                "strategy": strategy,
                "max_tokens": resolved_max_tokens,
                "overlap": resolved_overlap,
            },
        )
        return chunks

    def _chunk_text_document(
        self,
        text: str,
        *,
        normalized_text: str,
        target_tokens: int,
        sections: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        target, min_tokens, max_tokens = self._resolve_text_chunk_bounds(target_tokens)
        chunks: list[dict[str, Any]] = []
        chunk_index = 0
        for section_start, section_end in sections:
            if section_start >= section_end:
                continue
            for start_offset, end_offset in self._chunk_text_section(
                text,
                section_start,
                section_end,
                target,
                min_tokens,
                max_tokens,
            ):
                snippet = text[start_offset:end_offset]
                if not snippet.strip():
                    continue
                chunks.append(
                    {
                        "index": chunk_index,
                        "text": snippet,
                        "token_count": max(1, self._count_tokens(snippet)),
                        "start_offset": start_offset,
                        "end_offset": end_offset,
                        "search_text": self._normalize_text(snippet),
                    }
                )
                chunk_index += 1
        if not chunks and text.strip():
            snippet = normalized_text or text.strip()
            chunks.append(
                {
                    "index": 0,
                    "text": snippet,
                    "token_count": max(1, self._count_tokens(snippet)),
                    "start_offset": 0,
                    "end_offset": len(snippet),
                    "search_text": self._normalize_text(snippet),
                }
            )
        return chunks

    def _chunk_text_section(
        self,
        text: str,
        section_start: int,
        section_end: int,
        target_tokens: int,
        min_tokens: int,
        max_tokens: int,
    ) -> list[tuple[int, int]]:
        sentences = self._collect_sentence_spans(text, section_start, section_end)
        if not sentences:
            if section_end > section_start:
                return [(section_start, section_end)]
            return []
        chunks: list[tuple[int, int]] = []
        index = 0
        total = len(sentences)
        while index < total:
            chunk_start_idx = index
            cursor = index
            accumulated = 0
            while cursor < total:
                start, end = sentences[cursor]
                tokens = max(1, self._count_tokens(text[start:end]))
                if accumulated and accumulated + tokens > max_tokens:
                    break
                accumulated += tokens
                cursor += 1
                if accumulated >= target_tokens:
                    if cursor >= total:
                        break
                    next_start = sentences[cursor][0]
                    last_end = sentences[cursor - 1][1]
                    if self._has_paragraph_break(text, last_end, next_start):
                        break
                    if accumulated >= min_tokens:
                        break
            if cursor == chunk_start_idx:
                start, end = sentences[chunk_start_idx]
                accumulated = max(1, self._count_tokens(text[start:end]))
                cursor += 1
            if accumulated < min_tokens and cursor < total:
                start, end = sentences[cursor]
                accumulated += max(1, self._count_tokens(text[start:end]))
                cursor += 1
            chunk_start_offset = sentences[chunk_start_idx][0]
            chunk_end_offset = sentences[cursor - 1][1]
            if chunk_start_idx == 0:
                chunk_start_offset = min(chunk_start_offset, section_start)
            if cursor >= total:
                chunk_end_offset = section_end
            chunks.append((chunk_start_offset, max(chunk_start_offset + 1, chunk_end_offset)))
            if cursor >= total:
                break
            next_start = sentences[cursor][0]
            last_end = sentences[cursor - 1][1]
            if not self._has_paragraph_break(text, last_end, next_start):
                overlap_index = cursor - 1
                index = overlap_index if overlap_index > chunk_start_idx else cursor
            else:
                index = cursor
        return chunks

    def _collect_sentence_spans(
        self, text: str, start: int, end: int
    ) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        offset = max(0, start)
        limit = min(len(text), end)
        while offset < limit:
            while offset < limit and text[offset].isspace():
                offset += 1
            if offset >= limit:
                break
            sentence_end = self._advance_to_sentence_end(text, offset)
            if sentence_end <= offset:
                sentence_end = min(limit, offset + self._BOUNDARY_LOOKAHEAD)
            sentence_end = min(sentence_end, limit)
            spans.append((offset, sentence_end))
            offset = sentence_end
        return spans

    def _has_paragraph_break(self, text: str, start: int, end: int) -> bool:
        if start >= end:
            return False
        return bool(re.search(r"\n\s*\n", text[start:end]))

    def _resolve_section_spans(
        self,
        text: str,
        sections: Sequence["DocumentSection"] | Sequence[dict[str, Any]] | None,
    ) -> list[tuple[int, int]]:
        if not sections:
            return [(0, len(text))]
        spans: list[tuple[int, int]] = []
        cursor = 0
        length = len(text)
        for section in sections:
            content = None
            if hasattr(section, "content"):
                content = getattr(section, "content")
            elif isinstance(section, dict):
                content = section.get("content")
            if not content:
                continue
            search_value = content
            start_offset = text.find(search_value, cursor)
            if start_offset == -1:
                trimmed = search_value.strip()
                if trimmed:
                    start_offset = text.find(trimmed, cursor)
                    if start_offset != -1:
                        search_value = trimmed
            if start_offset == -1:
                continue
            if start_offset < cursor:
                start_offset = text.find(search_value, cursor)
                if start_offset == -1:
                    continue
            end_offset = start_offset + len(search_value)
            if cursor < start_offset:
                spans.append((cursor, start_offset))
            spans.append((start_offset, end_offset))
            cursor = max(cursor, end_offset)
        if cursor < length:
            spans.append((cursor, length))
        merged: list[tuple[int, int]] = []
        for start_offset, end_offset in spans:
            if start_offset >= end_offset:
                continue
            if merged and start_offset < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end_offset))
            else:
                merged.append((start_offset, end_offset))
        return merged or [(0, len(text))]

    def _resolve_text_chunk_bounds(self, target_tokens: int) -> tuple[int, int, int]:
        base_target = max(1, target_tokens)
        if base_target < self._TEXT_MIN_TOKENS:
            target = max(220, base_target)
            min_tokens = max(90, int(target * 0.6))
        else:
            target = min(self._TEXT_MAX_TOKENS, base_target)
            min_tokens = max(self._TEXT_MIN_TOKENS, int(target * 0.5))
        max_tokens = min(self._TEXT_MAX_TOKENS, max(target + 120, int(target * 1.25)))
        if min_tokens >= target:
            min_tokens = max(1, int(target * 0.75))
        if min_tokens >= max_tokens:
            min_tokens = max(1, max_tokens - 1)
        return target, min_tokens, max_tokens

    def _chunk_code_document(
        self,
        text: str,
        *,
        normalized_text: str,
        max_tokens: int,
        path: str | None,
    ) -> list[dict[str, Any]]:
        suffix = Path(path).suffix.lower() if path else ""
        spans: list[tuple[int, int]]
        if suffix in {".py", ".pyw"}:
            spans = self._chunk_python_code(text)
            if not spans:
                spans = self._chunk_markup_document(
                    text,
                    normalized_text=normalized_text,
                    max_tokens=max_tokens,
                    overlap=min(max(0, max_tokens // 4), max_tokens - 1),
                    strategy="code",
                    return_spans=True,
                )
        else:
            spans = self._chunk_markup_document(
                text,
                normalized_text=normalized_text,
                max_tokens=max_tokens,
                overlap=min(max(0, max_tokens // 5), max_tokens - 1),
                strategy="code",
                return_spans=True,
            )
        if not spans:
            snippet = normalized_text or text.strip()
            if not snippet:
                return []
            return [
                {
                    "index": 0,
                    "text": snippet,
                    "token_count": max(1, self._count_tokens(snippet)),
                    "start_offset": 0,
                    "end_offset": len(snippet),
                    "search_text": self._normalize_text(snippet),
                }
            ]
        chunks: list[dict[str, Any]] = []
        for index, (start_offset, end_offset) in enumerate(spans):
            if suffix not in {".py", ".pyw"}:
                adjusted_start = self._rewind_to_line_start(text, start_offset)
                if adjusted_start <= start_offset:
                    start_offset = max(0, adjusted_start)
                end_offset = min(len(text), max(end_offset, start_offset + 1))
                if end_offset < len(text) and text[end_offset - 1] != "\n":
                    end_offset = self._advance_to_code_boundary(text, end_offset)
            snippet = text[start_offset:end_offset]
            if not snippet.strip():
                continue
            chunks.append(
                {
                    "index": index,
                    "text": snippet,
                    "token_count": max(1, self._count_tokens(snippet)),
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "search_text": self._normalize_text(snippet),
                }
            )
        return chunks

    def _chunk_python_code(self, text: str) -> list[tuple[int, int]]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        line_offsets = self._compute_line_offsets(text)
        return self._chunk_python_body(tree.body, 0, len(text), line_offsets, text)

    def _chunk_python_body(
        self,
        nodes: list[ast.stmt],
        start_offset: int,
        end_offset: int,
        line_offsets: list[int],
        text: str,
    ) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        block_start = start_offset
        for node in nodes:
            child_start = self._python_symbol_start(node, text, line_offsets)
            child_end = self._python_node_end(node, text, line_offsets)
            if isinstance(node, ast.ClassDef):
                if block_start < child_start:
                    snippet = text[block_start:child_start]
                    if snippet.strip():
                        spans.append((block_start, child_start))
                spans.extend(self._chunk_python_class(node, line_offsets, text))
                block_start = max(block_start, child_end)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if block_start < child_start:
                    snippet = text[block_start:child_start]
                    if snippet.strip():
                        spans.append((block_start, child_start))
                spans.append((child_start, child_end))
                block_start = max(block_start, child_end)
            else:
                continue
        if block_start < end_offset:
            snippet = text[block_start:end_offset]
            if snippet.strip():
                spans.append((block_start, end_offset))
        return spans

    def _chunk_python_class(
        self, node: ast.ClassDef, line_offsets: list[int], text: str
    ) -> list[tuple[int, int]]:
        class_start = self._python_symbol_start(node, text, line_offsets)
        class_end = self._python_node_end(node, text, line_offsets)
        spans: list[tuple[int, int]] = []
        block_start = class_start
        for child in node.body:
            child_start = self._python_symbol_start(child, text, line_offsets)
            child_end = self._python_node_end(child, text, line_offsets)
            if isinstance(child, ast.ClassDef):
                if block_start < child_start:
                    snippet = text[block_start:child_start]
                    if snippet.strip():
                        spans.append((block_start, child_start))
                spans.extend(self._chunk_python_class(child, line_offsets, text))
                block_start = max(block_start, child_end)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if block_start < child_start:
                    snippet = text[block_start:child_start]
                    if snippet.strip():
                        spans.append((block_start, child_start))
                spans.append((child_start, child_end))
                block_start = max(block_start, child_end)
            else:
                continue
        if block_start < class_end:
            snippet = text[block_start:class_end]
            if snippet.strip():
                spans.append((block_start, class_end))
        return spans

    def _compute_line_offsets(self, text: str) -> list[int]:
        offsets = [0]
        for match in re.finditer(r"\n", text):
            offsets.append(match.end())
        if offsets[-1] != len(text):
            offsets.append(len(text))
        return offsets

    def _offset_from_line_col(
        self, lineno: int | None, col_offset: int | None, line_offsets: list[int], length: int
    ) -> int:
        if lineno is None or lineno < 1:
            return 0
        index = min(len(line_offsets) - 1, lineno - 1)
        base = line_offsets[index]
        column = col_offset or 0
        return max(0, min(length, base + column))

    def _python_symbol_start(
        self, node: ast.AST, text: str, line_offsets: list[int]
    ) -> int:
        lineno = getattr(node, "lineno", None)
        col_offset = getattr(node, "col_offset", 0)
        start = self._offset_from_line_col(lineno, col_offset, line_offsets, len(text))
        if lineno is None:
            return start
        line_index = lineno - 2
        best_start = start
        while line_index >= 0:
            line_start = line_offsets[line_index]
            line_end = line_offsets[line_index + 1] if line_index + 1 < len(line_offsets) else len(text)
            line_text = text[line_start:line_end]
            stripped = line_text.strip()
            if not stripped:
                break
            if stripped.startswith("#"):
                best_start = line_start
                line_index -= 1
                continue
            break
        return best_start

    def _python_node_end(
        self, node: ast.AST, text: str, line_offsets: list[int]
    ) -> int:
        end_lineno = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)
        if end_lineno is None or end_col is None:
            segment = ast.get_source_segment(text, node)
            if segment is None:
                return len(text)
            start = self._python_symbol_start(node, text, line_offsets)
            return min(len(text), start + len(segment))
        return self._offset_from_line_col(end_lineno, end_col, line_offsets, len(text))

    def _count_tokens(self, text: str) -> int:
        return len(re.findall(r"\S+", text))

    def _chunk_markup_document(
        self,
        text: str,
        *,
        normalized_text: str,
        max_tokens: int,
        overlap: int,
        strategy: str,
        return_spans: bool = False,
    ) -> list[dict[str, Any]] | list[tuple[int, int]]:
        matches = list(re.finditer(r"\S+", text))
        if not matches:
            trimmed = text.strip()
            chunk_text = trimmed if trimmed else normalized_text
            if return_spans:
                return [(0, len(chunk_text))] if chunk_text else []
            return [
                {
                    "index": 0,
                    "text": chunk_text,
                    "token_count": max(1, self._count_tokens(chunk_text)),
                    "start_offset": 0,
                    "end_offset": len(chunk_text),
                    "search_text": self._normalize_text(chunk_text),
                }
            ]
        token_positions = [match.start() for match in matches]
        spans: list[tuple[int, int]] = []
        start_index = 0
        text_length = len(text)
        step = max(1, max_tokens - overlap)
        while start_index < len(matches):
            end_index = min(start_index + max_tokens, len(matches))
            start_offset = matches[start_index].start()
            end_offset = (
                matches[end_index - 1].end() if end_index > start_index else start_offset
            )
            if end_index >= len(matches):
                end_offset = text_length
            else:
                adjusted_start = self._adjust_chunk_start(text, start_offset, strategy)
                adjusted_end = self._adjust_chunk_end(text, end_offset, strategy)
                if adjusted_start <= start_offset:
                    start_offset = max(0, adjusted_start)
                if adjusted_end >= end_offset:
                    end_offset = min(text_length, adjusted_end)
            end_offset = min(text_length, max(end_offset, start_offset + 1))
            start_offset = max(0, min(start_offset, end_offset - 1))
            snippet = text[start_offset:end_offset]
            if not snippet.strip():
                start_index = end_index if end_index > start_index else start_index + 1
                continue
            actual_start = bisect_left(token_positions, start_offset)
            actual_end = bisect_left(token_positions, end_offset)
            if (
                strategy == "code"
                and end_offset < text_length
                and (end_offset <= 0 or text[end_offset - 1] != "\n")
            ):
                line_break = text.rfind("\n", start_offset, end_offset)
                if line_break != -1 and line_break >= start_offset:
                    end_offset = line_break + 1
                    actual_end = bisect_left(token_positions, end_offset)
                    snippet = text[start_offset:end_offset]
            if actual_end <= actual_start:
                actual_end = min(len(matches), actual_start + 1)
                end_offset = matches[actual_end - 1].end()
                snippet = text[start_offset:end_offset]
                if not snippet.strip():
                    start_index = end_index if end_index > start_index else start_index + 1
                    continue
            spans.append((start_offset, end_offset))
            if end_index >= len(matches):
                break
            next_index = max(actual_start, actual_end - overlap)
            if next_index <= start_index:
                next_index = start_index + step
            start_index = min(next_index, len(matches))
        if return_spans:
            return spans
        chunks: list[dict[str, Any]] = []
        for index, (start_offset, end_offset) in enumerate(spans):
            snippet = text[start_offset:end_offset]
            if not snippet.strip():
                continue
            chunks.append(
                {
                    "index": index,
                    "text": snippet,
                    "token_count": max(1, self._count_tokens(snippet)),
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "search_text": self._normalize_text(snippet),
                }
            )
        return chunks

    def _resolve_chunking_settings(
        self,
        *,
        path: str | None,
        metadata: dict[str, Any] | None,
        default_max_tokens: int,
        default_overlap: int,
    ) -> tuple[str, int, int]:
        strategy = "text"
        max_tokens = max(1, int(default_max_tokens))
        overlap = max(0, min(int(default_overlap), max_tokens - 1))
        suffix = Path(path).suffix.lower() if path else ""
        if suffix in self._CODE_SUFFIXES:
            strategy = "code"
            max_tokens = 220
            overlap = min(32, max_tokens - 1)
        elif suffix in self._HTML_SUFFIXES:
            strategy = "html"
            max_tokens = 360
            overlap = min(72, max_tokens - 1)

        chunking_meta = None
        if isinstance(metadata, dict):
            chunking_meta = metadata.get("chunking")
        if isinstance(chunking_meta, dict):
            max_tokens = max(1, int(chunking_meta.get("max_tokens", max_tokens)))
            overlap = max(
                0,
                min(int(chunking_meta.get("overlap", overlap)), max_tokens - 1),
            )
            override_strategy = chunking_meta.get("strategy")
            if (
                isinstance(override_strategy, str)
                and override_strategy.lower() in {"text", "code", "html"}
            ):
                strategy = override_strategy.lower()

        return strategy, max_tokens, overlap

    def _adjust_chunk_start(self, text: str, offset: int, strategy: str) -> int:
        offset = max(0, min(offset, len(text)))
        if strategy == "code":
            candidate = self._rewind_to_line_start(text, offset)
            return min(offset, candidate)
        if strategy == "html":
            tag_start = self._rewind_to_tag_start(text, offset)
            if tag_start is not None:
                return min(offset, tag_start)
            candidate = self._rewind_to_line_start(text, offset)
            return min(offset, candidate)
        candidate = self._rewind_to_sentence_start(text, offset)
        return min(offset, candidate)

    def _adjust_chunk_end(self, text: str, offset: int, strategy: str) -> int:
        offset = max(0, min(offset, len(text)))
        if strategy == "code":
            candidate = self._advance_to_code_boundary(text, offset)
            return max(offset, candidate)
        if strategy == "html":
            tag_end = self._advance_to_tag_end(text, offset)
            if tag_end is not None:
                return max(offset, tag_end)
            candidate = self._advance_to_sentence_end(text, offset)
            return max(offset, candidate)
        candidate = self._advance_to_sentence_end(text, offset)
        return max(offset, candidate)

    @staticmethod
    def _rewind_to_line_start(text: str, offset: int) -> int:
        newline = text.rfind("\n", 0, offset)
        return newline + 1 if newline != -1 else 0

    def _rewind_to_tag_start(self, text: str, offset: int) -> int | None:
        lt = text.rfind("<", 0, offset)
        gt = text.rfind(">", 0, offset)
        if lt != -1 and (gt < lt):
            return lt
        return None

    def _rewind_to_sentence_start(self, text: str, offset: int) -> int:
        if offset <= 0:
            return 0
        pos = offset - 1
        limit = max(0, offset - self._BOUNDARY_LOOKBACK)
        while pos > limit and text[pos] in " \t":
            pos -= 1
        while (
            pos > limit
            and text[pos] not in self._SENTENCE_BOUNDARY_CHARS
            and text[pos] != "\n"
        ):
            pos -= 1
        if pos <= limit:
            newline = text.rfind("\n", limit, offset)
            if newline != -1:
                return newline + 1
            return limit
        if text[pos] == "\n":
            newline = text.rfind("\n", 0, pos)
            return newline + 1 if newline != -1 else 0
        pos += 1
        while pos < len(text) and text[pos] in self._CLOSING_PUNCTUATION:
            pos += 1
        while pos < len(text) and text[pos] in " \t":
            pos += 1
        if pos < len(text) and text[pos] in "\r\n":
            while pos < len(text) and text[pos] in "\r\n":
                pos += 1
        return pos

    def _advance_to_sentence_end(self, text: str, offset: int) -> int:
        length = len(text)
        if offset >= length:
            return length
        pos = offset
        limit = min(length, offset + self._BOUNDARY_LOOKAHEAD)
        while pos < limit and text[pos] in " \t":
            pos += 1
        while (
            pos < limit
            and text[pos] not in self._SENTENCE_BOUNDARY_CHARS
            and text[pos] != "\n"
        ):
            pos += 1
        if pos >= limit or pos >= length:
            return min(length, limit)
        if text[pos] == "\n":
            return pos
        pos += 1
        while pos < length and text[pos] in self._CLOSING_PUNCTUATION:
            pos += 1
        while pos < length and text[pos] == " ":
            pos += 1
        if pos < length and text[pos] == "\n":
            while pos < length and text[pos] in "\r\n":
                pos += 1
        return pos

    def _advance_to_code_boundary(self, text: str, offset: int) -> int:
        length = len(text)
        if offset >= length:
            return length
        limit = min(length, offset + self._BOUNDARY_LOOKAHEAD)
        newline_newline = text.find("\n\n", offset, limit)
        if newline_newline != -1:
            return newline_newline
        newline = text.find("\n", offset, limit)
        if newline != -1:
            return newline
        return limit

    def _advance_to_tag_end(self, text: str, offset: int) -> int | None:
        length = len(text)
        if offset >= length:
            return length
        gt = text.find(">", offset)
        if gt != -1:
            return gt + 1
        return None

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
