"""Project management utilities for coordinating application state."""

from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from ..config import ConfigManager, get_user_config_dir
from ..storage import (
    BackgroundTaskLogRepository,
    ChatRepository,
    DatabaseManager,
    DocumentRepository,
    IngestDocumentRepository,
    ProjectRepository,
)


DEFAULT_PROJECT_NAME = "Default Project"


@dataclass(slots=True)
class ProjectRecord:
    """Lightweight representation of a stored project."""

    id: int
    name: str
    description: str | None = None


class ProjectService(QObject):
    """Coordinate project state, storage paths, and per-project settings."""

    projects_changed = pyqtSignal(object)
    active_project_changed = pyqtSignal(object)

    def __init__(
        self,
        *,
        storage_root: str | Path | None = None,
        config_manager: ConfigManager | None = None,
    ) -> None:
        super().__init__()
        self._config = config_manager or ConfigManager()
        if storage_root is None:
            storage_root = get_user_config_dir() / "storage"
        self._storage_root = Path(storage_root)
        self._storage_root.mkdir(parents=True, exist_ok=True)
        self._db_path = self._storage_root / "dataminer.db"
        self._db = DatabaseManager(self._db_path)
        self._db.initialize()
        self.projects = ProjectRepository(self._db)
        self.documents = DocumentRepository(self._db)
        self.chats = ChatRepository(self._db)
        self.ingest = IngestDocumentRepository(self._db)
        self.background_tasks = BackgroundTaskLogRepository(self._db)
        self._lock = threading.RLock()
        self._active_project_id: int | None = None
        self._ensure_default_project()
        self._load_active_project()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    def shutdown(self) -> None:
        """Close resources held by the service."""

        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------
    @property
    def storage_root(self) -> Path:
        return self._storage_root

    @property
    def database_path(self) -> Path:
        return self._db_path

    @property
    def database_manager(self) -> DatabaseManager:
        return self._db

    @property
    def active_project_id(self) -> int:
        if self._active_project_id is None:
            raise RuntimeError("Active project not initialised")
        return self._active_project_id

    def active_project(self) -> ProjectRecord:
        record = self.get_project(self.active_project_id)
        if record is None:
            raise RuntimeError("Active project record missing")
        return record

    # ------------------------------------------------------------------
    def list_projects(self) -> list[ProjectRecord]:
        records: list[ProjectRecord] = []
        for entry in self.projects.list():
            if not entry:
                continue
            records.append(
                ProjectRecord(
                    id=int(entry["id"]),
                    name=str(entry.get("name", "")),
                    description=entry.get("description"),
                )
            )
        return records

    def get_project(self, project_id: int) -> ProjectRecord | None:
        entry = self.projects.get(project_id)
        if not entry:
            return None
        return ProjectRecord(
            id=int(entry["id"]),
            name=str(entry.get("name", "")),
            description=entry.get("description"),
        )

    def create_project(
        self, name: str, *, description: str | None = None, activate: bool = True
    ) -> ProjectRecord:
        with self._lock:
            created = self.projects.create(name, description=description)
            project = ProjectRecord(
                id=int(created["id"]),
                name=str(created.get("name", name)),
                description=created.get("description"),
            )
            self._ensure_project_storage(project.id)
            self._emit_projects_changed()
            if activate:
                self.set_active_project(project.id)
            return project

    def rename_project(
        self, project_id: int, *, name: str | None = None, description: str | None = None
    ) -> ProjectRecord:
        updates: dict[str, Any] = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if not updates:
            record = self.get_project(project_id)
            if record is None:
                raise LookupError(f"Unknown project id {project_id}")
            return record
        with self._lock:
            updated = self.projects.update(project_id, **updates)
            if not updated:
                raise LookupError(f"Unknown project id {project_id}")
            record = ProjectRecord(
                id=int(updated["id"]),
                name=str(updated.get("name", "")),
                description=updated.get("description"),
            )
            self._emit_projects_changed()
            if project_id == self._active_project_id:
                self._emit_active_project_changed(record)
            return record

    def delete_project(self, project_id: int) -> None:
        with self._lock:
            if project_id == self._active_project_id:
                raise RuntimeError("Cannot delete the active project")
            self.projects.delete(project_id)
            storage = self._project_storage_dir(project_id)
            if storage.exists():
                shutil.rmtree(storage, ignore_errors=True)
            self._remove_project_settings(project_id)
            self._emit_projects_changed()

    def set_active_project(self, project_id: int) -> None:
        project = self.get_project(project_id)
        if project is None:
            raise LookupError(f"Unknown project id {project_id}")
        with self._lock:
            self._active_project_id = project_id
            self._ensure_project_storage(project_id)
            self._store_active_project(project_id)
            self._emit_active_project_changed(project)

    def reload(self) -> None:
        """Reload persisted state after an external change such as restore."""

        with self._lock:
            self._db.close()
            self._db = DatabaseManager(self._db_path)
            self._db.initialize()
            self.projects = ProjectRepository(self._db)
            self.documents = DocumentRepository(self._db)
            self.chats = ChatRepository(self._db)
            self.ingest = IngestDocumentRepository(self._db)
            self.background_tasks = BackgroundTaskLogRepository(self._db)
            self._ensure_default_project()
            self._load_active_project()
            self._emit_projects_changed()
            self._emit_active_project_changed(self.active_project())

    # ------------------------------------------------------------------
    # Storage helpers
    def get_project_storage(self, project_id: int) -> Path:
        return self._project_storage_dir(project_id)

    def purge_project_data(self, project_id: int) -> None:
        """Remove derived data for ``project_id`` without deleting source files."""

        with self._lock:
            with self._db.transaction() as connection:
                connection.execute("DELETE FROM documents WHERE project_id = ?", (project_id,))
                connection.execute("DELETE FROM chats WHERE project_id = ?", (project_id,))
                connection.execute("DELETE FROM tags WHERE project_id = ?", (project_id,))
            storage = self._project_storage_dir(project_id)
            if storage.exists():
                shutil.rmtree(storage, ignore_errors=True)
            self._ensure_project_storage(project_id)

    # ------------------------------------------------------------------
    # Conversation setting helpers
    def load_conversation_settings(self, project_id: int) -> dict[str, Any]:
        data = self._config.load()
        projects = data.get("projects") if isinstance(data, dict) else {}
        if not isinstance(projects, dict):
            return {}
        settings = projects.get("conversation")
        if not isinstance(settings, dict):
            return {}
        record = settings.get(str(project_id), {})
        return record if isinstance(record, dict) else {}

    def save_conversation_settings(self, project_id: int, settings: dict[str, Any]) -> None:
        data = self._config.load()
        if not isinstance(data, dict):
            data = {}
        projects = data.setdefault("projects", {})
        if not isinstance(projects, dict):
            projects = {}
            data["projects"] = projects
        conversation = projects.setdefault("conversation", {})
        if not isinstance(conversation, dict):
            conversation = {}
            projects["conversation"] = conversation
        conversation[str(project_id)] = settings
        self._config.save(data)

    # ------------------------------------------------------------------
    # Corpus root helpers
    def list_corpus_roots(self, project_id: int) -> list[str]:
        """Return a list of indexed corpus root folders for ``project_id``."""

        data = self._config.load()
        if not isinstance(data, dict):
            return []
        projects = data.get("projects")
        if not isinstance(projects, dict):
            return []
        corpus = projects.get("corpus_roots")
        if not isinstance(corpus, dict):
            return []
        roots = corpus.get(str(project_id))
        if not isinstance(roots, list):
            return []
        normalized: list[str] = []
        for entry in roots:
            if not isinstance(entry, str):
                continue
            normalized.append(str(Path(entry).resolve()))
        return normalized

    def add_corpus_root(self, project_id: int, path: str | Path) -> None:
        """Persist ``path`` as an indexed corpus root for ``project_id``."""

        normalized = str(Path(path).resolve())
        with self._lock:
            data = self._config.load()
            if not isinstance(data, dict):
                data = {}
            projects = data.setdefault("projects", {})
            if not isinstance(projects, dict):
                projects = {}
                data["projects"] = projects
            corpus = projects.setdefault("corpus_roots", {})
            if not isinstance(corpus, dict):
                corpus = {}
                projects["corpus_roots"] = corpus
            roots = corpus.setdefault(str(project_id), [])
            if not isinstance(roots, list):
                roots = []
                corpus[str(project_id)] = roots
            if normalized not in roots:
                roots.append(normalized)
            self._config.save(data)

    def remove_corpus_root(self, project_id: int, path: str | Path) -> None:
        """Remove ``path`` from the stored corpus roots for ``project_id``."""

        normalized = str(Path(path).resolve())
        with self._lock:
            data = self._config.load()
            if not isinstance(data, dict):
                return
            projects = data.get("projects")
            if not isinstance(projects, dict):
                return
            corpus = projects.get("corpus_roots")
            if not isinstance(corpus, dict):
                return
            roots = corpus.get(str(project_id))
            if not isinstance(roots, list):
                return
            filtered = [root for root in roots if isinstance(root, str) and str(Path(root).resolve()) != normalized]
            if filtered:
                corpus[str(project_id)] = filtered
            else:
                corpus.pop(str(project_id), None)
            self._config.save(data)

    def clear_corpus_roots(self, project_id: int) -> None:
        """Remove all stored corpus roots for ``project_id``."""

        with self._lock:
            data = self._config.load()
            if not isinstance(data, dict):
                return
            projects = data.get("projects")
            if not isinstance(projects, dict):
                return
            corpus = projects.get("corpus_roots")
            if not isinstance(corpus, dict):
                return
            if str(project_id) in corpus:
                corpus.pop(str(project_id), None)
                self._config.save(data)

    # ------------------------------------------------------------------
    # Internal helpers
    def _ensure_default_project(self) -> None:
        if self.projects.list():
            return
        created = self.projects.create(DEFAULT_PROJECT_NAME)
        self._ensure_project_storage(int(created["id"]))
        self._store_active_project(int(created["id"]))

    def _load_active_project(self) -> None:
        stored = self._load_active_project_id()
        available = {record.id for record in self.list_projects()}
        if stored in available:
            self._active_project_id = stored
        elif available:
            self._active_project_id = sorted(available)[0]
        else:
            raise RuntimeError("No projects available")
        self._ensure_project_storage(self._active_project_id)

    def _load_active_project_id(self) -> int | None:
        data = self._config.load()
        if not isinstance(data, dict):
            return None
        projects = data.get("projects")
        if not isinstance(projects, dict):
            return None
        value = projects.get("active_id")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _store_active_project(self, project_id: int) -> None:
        data = self._config.load()
        if not isinstance(data, dict):
            data = {}
        projects = data.setdefault("projects", {})
        if not isinstance(projects, dict):
            projects = {}
            data["projects"] = projects
        projects["active_id"] = project_id
        self._config.save(data)

    def _remove_project_settings(self, project_id: int) -> None:
        data = self._config.load()
        if not isinstance(data, dict):
            return
        projects = data.get("projects")
        if not isinstance(projects, dict):
            return
        modified = False
        conversation = projects.get("conversation")
        if isinstance(conversation, dict) and str(project_id) in conversation:
            conversation.pop(str(project_id), None)
            modified = True
        corpus = projects.get("corpus_roots")
        if isinstance(corpus, dict) and str(project_id) in corpus:
            corpus.pop(str(project_id), None)
            modified = True
        if modified:
            self._config.save(data)

    def _project_storage_dir(self, project_id: int) -> Path:
        return self._storage_root / "projects" / str(project_id)

    def _ensure_project_storage(self, project_id: int) -> None:
        storage = self._project_storage_dir(project_id)
        storage.mkdir(parents=True, exist_ok=True)

    def _emit_projects_changed(self) -> None:
        self.projects_changed.emit(self.list_projects())

    def _emit_active_project_changed(self, project: ProjectRecord) -> None:
        self.active_project_changed.emit(project)


__all__ = ["ProjectService", "ProjectRecord", "DEFAULT_PROJECT_NAME"]

