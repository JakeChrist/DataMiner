"""Database utilities and migration helpers for the storage layer."""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA_VERSION = 1
SCHEMA_FILENAME = "schema.sql"


class DatabaseError(RuntimeError):
    """Raised when database bootstrap or migrations fail."""


class DatabaseManager:
    """Manage the SQLite database connection and schema migrations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """Return a singleton SQLite connection with sensible defaults."""
        if self._connection is None:
            connection = sqlite3.connect(self.path, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            self._connection = connection
        return self._connection

    def close(self) -> None:
        """Close the underlying database connection if it exists."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

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
    ) -> dict[str, Any]:
        metadata_json = None
        if metadata is not None:
            import json

            metadata_json = json.dumps(metadata)
        stored_path = str(source_path) if source_path is not None else None
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO documents (project_id, title, source_type, source_path, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (project_id, title, source_type, stored_path, metadata_json),
            )
            document_id = cursor.lastrowid
        return self.get(document_id)  # type: ignore[return-value]

    def get(self, document_id: int) -> dict[str, Any] | None:
        row = self.db.connect().execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        data = self._row_to_dict(row)
        if data and data.get("metadata"):
            import json

            data["metadata"] = json.loads(data["metadata"])
        return data

    def list_for_project(self, project_id: int) -> list[dict[str, Any]]:
        rows = self.db.connect().execute(
            "SELECT * FROM documents WHERE project_id = ? ORDER BY created_at ASC",
            (project_id,),
        ).fetchall()
        documents: list[dict[str, Any]] = []
        for row in rows:
            document = self._row_to_dict(row)
            if document and document.get("metadata"):
                import json

                document["metadata"] = json.loads(document["metadata"])
            if document:
                documents.append(document)
        return documents

    def delete(self, document_id: int) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

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
                (document_id, version, str(file_path), checksum, file_size),
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
            connection.execute(
                "INSERT OR IGNORE INTO tag_links (tag_id, document_id) VALUES (?, ?)",
                (tag_id, document_id),
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


class ChatRepository(BaseRepository):
    """Manage chat sessions, citations, and reasoning summaries."""

    def create(self, project_id: int, title: str | None = None) -> dict[str, Any]:
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO chats (project_id, title) VALUES (?, ?)",
                (project_id, title),
            )
            chat_id = cursor.lastrowid
        return self.get(chat_id)  # type: ignore[return-value]

    def get(self, chat_id: int) -> dict[str, Any] | None:
        row = self.db.connect().execute(
            "SELECT * FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        return self._row_to_dict(row)

    def list_for_project(self, project_id: int) -> list[dict[str, Any]]:
        rows = self.db.connect().execute(
            "SELECT * FROM chats WHERE project_id = ? ORDER BY created_at ASC",
            (project_id,),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]  # type: ignore[list-item]

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
