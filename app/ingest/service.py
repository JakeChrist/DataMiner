"""Background ingestion service with resumable job processing."""

from __future__ import annotations

import hashlib
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from app.ingest.parsers import ParserError, parse_file
from app.storage import (
    BackgroundTaskLogRepository,
    DatabaseManager,
    DocumentRepository,
    IngestDocumentRepository,
)


TaskCallback = Callable[[int, dict[str, Any]], None]


class TaskStatus:
    """Canonical status values persisted in the task log."""

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"

    FINAL = {CANCELLED, FAILED, COMPLETED}


class _JobPaused(RuntimeError):
    """Raised internally to unwind the worker loop when a pause is requested."""


class _JobCancelled(RuntimeError):
    """Raised internally when a cancellation request is observed."""


@dataclass
class IngestJob:
    """Representation of a single ingest job managed by the worker."""

    job_id: int
    job_type: str
    params: dict[str, Any]
    status: str
    progress: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    pause_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def ensure_defaults(self) -> None:
        """Populate default bookkeeping keys for newly restored jobs."""

        self.progress.setdefault("total", 0)
        self.progress.setdefault("processed", 0)
        self.progress.setdefault("succeeded", 0)
        self.progress.setdefault("failed", 0)
        self.progress.setdefault("skipped", 0)
        self.state.setdefault("pending_files", None)
        self.state.setdefault("position", 0)
        self.state.setdefault("processed_files", [])
        known_files = self.state.get("known_files")
        if known_files is None:
            known_files = self.params.get("known_files", {})
            # Ensure we store a copy so mutation does not leak back to params.
            self.state["known_files"] = dict(known_files)
        else:
            # Work with a copy to keep the serialized representation stable.
            self.state["known_files"] = dict(known_files)
        if "known_files_snapshot" not in self.state:
            self.state["known_files_snapshot"] = dict(self.state["known_files"])


class IngestService:
    """Coordinate ingest jobs using a background worker thread."""

    def __init__(self, db: DatabaseManager, *, worker_idle_sleep: float = 0.1) -> None:
        self.db = db
        self.repo = BackgroundTaskLogRepository(db)
        self.documents = IngestDocumentRepository(db)
        self.project_documents = DocumentRepository(db)
        self._queue: "queue.Queue[Optional[int]]" = queue.Queue()
        self._jobs: dict[int, IngestJob] = {}
        self._jobs_lock = threading.RLock()
        self._subscribers: list[TaskCallback] = []
        self._stop_event = threading.Event()
        self._worker_idle_sleep = worker_idle_sleep
        self._worker = threading.Thread(target=self._worker_loop, name="IngestWorker", daemon=True)
        self._restore_incomplete_jobs()
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def subscribe(self, callback: TaskCallback) -> Callable[[], None]:
        """Register ``callback`` for task updates and return an unsubscribe handle."""

        with self._jobs_lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._jobs_lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def queue_folder_crawl(
        self,
        project_id: int | None,
        root: str | Path,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
    ) -> int:
        """Discover and ingest all files underneath ``root``."""

        root_path = Path(root).resolve()
        params = {
            "project_id": project_id,
            "root": str(root_path),
            "include": list(include or []),
            "exclude": list(exclude or []),
        }
        return self._create_job("folder_crawl", params, f"Folder crawl for {root_path}")

    def queue_file_add(
        self,
        project_id: int | None,
        files: str | Path | Iterable[str | Path],
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
    ) -> int:
        """Queue a job that ingests one or multiple explicit files."""

        if isinstance(files, (str, Path)):
            file_list = [files]
        else:
            file_list = list(files)
        normalized = [str(Path(path).resolve()) for path in file_list]
        base_dir = str(Path(normalized[0]).resolve().parent) if normalized else None
        params = {
            "project_id": project_id,
            "files": normalized,
            "include": list(include or []),
            "exclude": list(exclude or []),
            "root": base_dir,
        }
        return self._create_job("single_file", params, "Single file ingest")

    def queue_rescan(
        self,
        project_id: int | None,
        root: str | Path,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        known_files: dict[str, Any] | None = None,
    ) -> int:
        """Re-ingest files that have changed since the previous crawl."""

        root_path = Path(root).resolve()
        if known_files is None:
            known_files = self._load_known_files(str(root_path))
        params = {
            "project_id": project_id,
            "root": str(root_path),
            "include": list(include or []),
            "exclude": list(exclude or []),
            "known_files": known_files or {},
        }
        return self._create_job("rescan", params, f"Rescan for {root_path}")

    def queue_remove(
        self,
        project_id: int | None,
        root: str | Path,
        files: Iterable[str | Path],
    ) -> int:
        """Remove tracked files from the ingest index."""

        root_path = Path(root).resolve()
        normalized = [str(Path(path).resolve()) for path in files]
        known_files = self._load_known_files(str(root_path))
        params = {
            "project_id": project_id,
            "root": str(root_path),
            "files": normalized,
            "known_files": known_files or {},
        }
        return self._create_job("remove", params, "Remove files from ingest index")

    def pause_job(self, job_id: int) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.pause_event.set()

    def resume_job(self, job_id: int) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.pause_event.clear()
        with self._jobs_lock:
            if job.status == TaskStatus.PAUSED:
                job.status = TaskStatus.QUEUED
                self._persist(job, status=TaskStatus.QUEUED)
                self._queue.put(job.job_id)

    def cancel_job(self, job_id: int) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.cancel_event.set()

    def wait_for_completion(self, job_id: int, timeout: float | None = None) -> bool:
        """Block until ``job_id`` reaches a terminal state or ``timeout`` expires."""

        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            job = self._jobs.get(job_id)
            if job and job.status in TaskStatus.FINAL:
                return True
            record = self.repo.get(job_id)
            if record is None:
                return False
            if record["status"] in TaskStatus.FINAL:
                if job is not None:
                    job.status = record["status"]
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def shutdown(self, *, wait: bool = True) -> None:
        """Request the worker to stop and optionally wait for completion."""

        self._stop_event.set()
        self._queue.put(None)
        if wait:
            self._worker.join()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_job(self, job_type: str, params: dict[str, Any], message: str) -> int:
        payload = {
            "job": {"type": job_type, "params": params},
            "progress": {
                "total": 0,
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "skipped": 0,
            },
            "errors": [],
            "summary": {},
            "state": {
                "pending_files": None,
                "position": 0,
                "processed_files": [],
                "known_files": dict(params.get("known_files", {})),
                "known_files_snapshot": dict(params.get("known_files", {})),
            },
        }
        record = self.repo.create(
            f"ingest.{job_type}",
            status=TaskStatus.QUEUED,
            message=message,
            extra_data=payload,
        )
        job = self._record_to_job(record)
        with self._jobs_lock:
            self._jobs[job.job_id] = job
        self._queue.put(job.job_id)
        self._emit(job.job_id)
        return job.job_id

    def _restore_incomplete_jobs(self) -> None:
        for record in self.repo.list_incomplete():
            job = self._record_to_job(record)
            with self._jobs_lock:
                self._jobs[job.job_id] = job
            if job.status == TaskStatus.PAUSED:
                job.pause_event.set()
                self._persist(job, status=TaskStatus.PAUSED)
            else:
                if job.status != TaskStatus.QUEUED:
                    job.status = TaskStatus.QUEUED
                    self._persist(job, status=TaskStatus.QUEUED)
                self._queue.put(job.job_id)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=self._worker_idle_sleep)
            except queue.Empty:
                continue

            if job_id is None:
                self._queue.task_done()
                continue

            job = self._jobs.get(job_id)
            if job is None:
                self._queue.task_done()
                continue

            if job.pause_event.is_set():
                # Remain paused until resume clears the flag.
                self._queue.task_done()
                continue

            try:
                self._start_job(job)
                self._process_job(job)
                if job.cancel_event.is_set():
                    raise _JobCancelled
                if job.pause_event.is_set():
                    raise _JobPaused
                self._complete_job(job)
            except _JobPaused:
                # Already persisted by _check_pause_cancel.
                pass
            except _JobCancelled:
                self._handle_cancel(job)
            except Exception as exc:  # pragma: no cover - defensive
                self._fail_job(job, exc)
            finally:
                self._queue.task_done()

        # Drain remaining queue items so join() on Queue won't block when shutting down.
        while True:
            try:
                job_id = self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()

    def _start_job(self, job: IngestJob) -> None:
        job.ensure_defaults()
        job.status = TaskStatus.RUNNING
        if not job.state.get("known_files_snapshot"):
            job.state["known_files_snapshot"] = dict(job.state.get("known_files", {}))
        self._persist(job, status=TaskStatus.RUNNING)

    def _process_job(self, job: IngestJob) -> None:
        files = job.state.get("pending_files")
        if files is None:
            files = self._discover_files(job)
            job.state["pending_files"] = files
            job.state["position"] = 0
            job.progress["total"] = len(files)
            self._persist(job)

        position = int(job.state.get("position", 0))
        known_files: dict[str, Any] = job.state.get("known_files", {})
        processed_files: list[dict[str, Any]] = job.state.get("processed_files", [])

        while position < len(files):
            self._check_pause_cancel(job)
            file_path = Path(files[position])
            result = self._handle_path(job, file_path)
            position += 1
            job.state["position"] = position
            job.progress["processed"] += 1

            if result["status"] == "success":
                job.progress["succeeded"] += 1
                metadata = result["metadata"]
                processed_files.append(metadata)
                known_files[metadata["path"]] = {
                    "checksum": metadata["checksum"],
                    "mtime": metadata["mtime"],
                    "size": metadata["size"],
                }
                if metadata.get("needs_ocr"):
                    job.summary.setdefault("needs_ocr", []).append(
                        {
                            "path": metadata["path"],
                            "message": metadata.get("ocr_message"),
                        }
                    )
            elif result["status"] == "skipped":
                job.progress["skipped"] += 1
            elif result["status"] == "removed":
                job.progress["succeeded"] += 1
                removed_path = result["metadata"]["path"]
                if removed_path in known_files:
                    known_files.pop(removed_path, None)
                job.summary.setdefault("removed", []).append(removed_path)
            if result.get("error"):
                job.errors.append(result["error"])
                job.progress["failed"] += 1
            self._persist(job)

        if job.job_type == "rescan":
            snapshot = job.state.get("known_files_snapshot", {})
            current_paths = {self._normalize_path(Path(path)) for path in files}
            removed = [path for path in snapshot.keys() if path not in current_paths]
            if removed:
                job.summary.setdefault("removed", []).extend(removed)
                for path in removed:
                    known_files.pop(path, None)
                self.documents.delete_by_paths(removed)
        job.state["processed_files"] = processed_files
        job.state["pending_files"] = files

    def _handle_path(self, job: IngestJob, path: Path) -> dict[str, Any]:
        if job.job_type == "remove":
            normalized = self._normalize_path(path)
            removed = self.documents.delete_by_paths([normalized])
            metadata = {"path": normalized, "removed_versions": removed}
            return {"status": "removed", "metadata": metadata}

        if not path.exists():
            return {
                "status": "skipped",
                "metadata": {"path": self._normalize_path(path)},
                "error": f"Missing file: {path}",
            }

        stat = path.stat()
        metadata = {
            "path": self._normalize_path(path),
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "checksum": self._hash_file(path),
        }

        if job.job_type == "rescan":
            previous = job.state.get("known_files_snapshot", {}).get(metadata["path"])
            if previous and previous.get("mtime") == metadata["mtime"] and previous.get("checksum") == metadata["checksum"]:
                return {"status": "skipped", "metadata": metadata}

        try:
            parsed = parse_file(path)
        except ParserError as exc:
            metadata["parser_error"] = str(exc)
            return {
                "status": "failed",
                "metadata": metadata,
                "error": str(exc),
            }
        except Exception as exc:  # pragma: no cover - defensive
            metadata["parser_error"] = str(exc)
            return {
                "status": "failed",
                "metadata": metadata,
                "error": str(exc),
            }

        base_metadata = {
            "file": {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "ctime": stat.st_ctime,
                "checksum": metadata["checksum"],
            }
        }
        record = self.documents.store_version(
            path=metadata["path"],
            checksum=metadata["checksum"],
            size=stat.st_size,
            mtime=stat.st_mtime,
            ctime=stat.st_ctime,
            parsed=parsed,
            base_metadata=base_metadata,
        )
        metadata["document_id"] = record.get("id")
        metadata["version"] = record.get("version")
        metadata["preview"] = record.get("preview")
        metadata["needs_ocr"] = record.get("needs_ocr", False)
        if record.get("ocr_message"):
            metadata["ocr_message"] = record["ocr_message"]
        metadata["parser_metadata"] = parsed.metadata

        return {"status": "success", "metadata": metadata}

    def _discover_files(self, job: IngestJob) -> list[str]:
        if job.job_type == "single_file":
            files = [Path(path) for path in job.params.get("files", [])]
            include = job.params.get("include") or []
            exclude = job.params.get("exclude") or []
            base = Path(job.params.get("root") or Path(files[0]).parent) if files else None
            result: list[str] = []
            for file_path in files:
                if not file_path.exists():
                    continue
                if self._matches_filters(file_path, include, exclude, base):
                    result.append(self._normalize_path(file_path))
            return sorted(dict.fromkeys(result))

        if job.job_type == "remove":
            files = [self._normalize_path(Path(path)) for path in job.params.get("files", [])]
            return sorted(dict.fromkeys(files))

        root = Path(job.params.get("root", ".")).resolve()
        include = job.params.get("include") or []
        exclude = job.params.get("exclude") or []
        if not root.exists():
            return []
        result: list[str] = []
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if self._matches_filters(file_path, include, exclude, root):
                result.append(self._normalize_path(file_path))
        return sorted(dict.fromkeys(result))

    def _matches_filters(
        self,
        path: Path,
        include: Iterable[str],
        exclude: Iterable[str],
        base: Path | None,
    ) -> bool:
        from fnmatch import fnmatch

        relative = path.name
        if base is not None:
            try:
                relative = str(path.resolve().relative_to(base.resolve()))
            except ValueError:
                relative = path.name

        include_patterns = list(include)
        exclude_patterns = list(exclude)

        if include_patterns and not any(fnmatch(relative, pattern) for pattern in include_patterns):
            return False
        if exclude_patterns and any(fnmatch(relative, pattern) for pattern in exclude_patterns):
            return False
        return True

    def _handle_cancel(self, job: IngestJob) -> None:
        self._rollback_job(job)
        job.status = TaskStatus.CANCELLED
        self._persist(job, status=TaskStatus.CANCELLED, completed=True)

    def _rollback_job(self, job: IngestJob) -> None:
        snapshot = job.state.get("known_files_snapshot", {})
        job.state["known_files"] = dict(snapshot)
        job.summary.setdefault("rolled_back", 0)
        job.summary["rolled_back"] += job.progress.get("succeeded", 0)
        # Reset counters that represent committed work so the UI reflects rollback.
        job.progress["succeeded"] = 0
        job.progress["processed"] = job.progress.get("skipped", 0) + job.progress.get("failed", 0)
        job.summary["known_files"] = dict(snapshot)
        job.state["processed_files"] = []

    def _fail_job(self, job: IngestJob, exc: Exception) -> None:
        job.status = TaskStatus.FAILED
        job.errors.append(str(exc))
        self._persist(job, status=TaskStatus.FAILED, message=str(exc), completed=True)

    def _complete_job(self, job: IngestJob) -> None:
        job.status = TaskStatus.COMPLETED
        job.summary.setdefault("success_count", job.progress.get("succeeded", 0))
        job.summary.setdefault("failure_count", job.progress.get("failed", 0))
        job.summary.setdefault("skipped_count", job.progress.get("skipped", 0))
        if job.summary.get("removed"):
            job.summary.setdefault("removed_count", len(job.summary.get("removed", [])))
        job.summary.setdefault("rolled_back", 0)
        job.summary["known_files"] = job.state.get("known_files", {})
        job.summary["total_files"] = job.progress.get("total", 0)
        job.summary["errors"] = job.errors
        job.state.pop("known_files_snapshot", None)
        job.state["processed_files"] = []
        self._sync_project_documents(job)
        self._persist(job, status=TaskStatus.COMPLETED, completed=True)

    def _check_pause_cancel(self, job: IngestJob) -> None:
        if job.cancel_event.is_set():
            self._persist(job)
            raise _JobCancelled
        if job.pause_event.is_set():
            job.status = TaskStatus.PAUSED
            self._persist(job, status=TaskStatus.PAUSED)
            raise _JobPaused

    def _persist(
        self,
        job: IngestJob,
        *,
        status: str | None = None,
        message: str | None = None,
        completed: bool = False,
    ) -> None:
        payload = self._serialize(job)
        completed_at = self._utcnow() if completed else None
        self.repo.update(
            job.job_id,
            status=status,
            message=message,
            extra_data=payload,
            completed_at=completed_at,
        )
        self._emit(job.job_id)

    def _serialize(self, job: IngestJob) -> dict[str, Any]:
        return {
            "job": {"type": job.job_type, "params": job.params},
            "progress": dict(job.progress),
            "errors": list(job.errors),
            "summary": dict(job.summary),
            "state": self._serialize_state(job.state),
        }

    def _sync_project_documents(self, job: IngestJob) -> None:
        project_id = job.params.get("project_id")
        if not isinstance(project_id, int):
            return

        repo = self.project_documents
        try:
            connection = repo.db.connect()
            project_exists = connection.execute(
                "SELECT 1 FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if project_exists is None:
                return
            existing_docs = {
                str(Path(doc["source_path"]).resolve()): doc
                for doc in repo.list_for_project(project_id)
                if doc.get("source_path")
            }
        except Exception:  # pragma: no cover - defensive
            return

        known_files_param = job.summary.get("known_files", {})
        if not isinstance(known_files_param, dict):
            known_files_param = {}

        normalized_known: dict[str, dict[str, Any]] = {}
        for path, metadata in known_files_param.items():
            if not isinstance(path, str):
                continue
            normalized_path = str(Path(path).resolve())
            data = dict(metadata) if isinstance(metadata, dict) else {}
            normalized_known[normalized_path] = data
            document = existing_docs.get(normalized_path)
            payload = {"file": data}
            if document is None:
                title = Path(normalized_path).stem or Path(normalized_path).name
                try:
                    repo.create(
                        project_id,
                        title,
                        source_type="file",
                        source_path=normalized_path,
                        metadata=payload,
                    )
                except Exception:  # pragma: no cover - defensive
                    continue
            else:
                updates: dict[str, Any] = {}
                if document.get("source_path") != normalized_path:
                    updates["source_path"] = normalized_path
                current_meta = document.get("metadata") or {}
                if current_meta.get("file") != data:
                    updates["metadata"] = payload
                if updates:
                    try:
                        repo.update(document["id"], **updates)
                    except Exception:  # pragma: no cover - defensive
                        continue

        removed_paths: set[str] = set()
        removed = job.summary.get("removed")
        if isinstance(removed, (list, tuple, set)):
            for entry in removed:
                if not isinstance(entry, str):
                    continue
                removed_paths.add(str(Path(entry).resolve()))

        if removed_paths:
            for document in repo.list_for_project(project_id):
                source_path = document.get("source_path")
                if not source_path:
                    continue
                normalized = str(Path(source_path).resolve())
                if normalized in removed_paths:
                    try:
                        repo.delete(document["id"])
                    except Exception:  # pragma: no cover - defensive
                        continue

    @staticmethod
    def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for key, value in state.items():
            if isinstance(value, Path):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized

    def _record_to_job(self, record: dict[str, Any]) -> IngestJob:
        extra = record.get("extra_data") or {}
        job_data = extra.get("job", {})
        progress = extra.get("progress", {})
        errors = extra.get("errors", [])
        summary = extra.get("summary", {})
        state = extra.get("state", {})
        job = IngestJob(
            job_id=record["id"],
            job_type=job_data.get("type") or record.get("task_name", "ingest.unknown").split(".")[-1],
            params=job_data.get("params", {}),
            status=record.get("status", TaskStatus.QUEUED),
            progress=dict(progress),
            errors=list(errors),
            summary=dict(summary),
            state=dict(state),
        )
        job.ensure_defaults()
        if job.status == TaskStatus.PAUSED:
            job.pause_event.set()
        return job

    def _emit(self, job_id: int) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        payload = self._serialize(job)
        payload["status"] = job.status
        for callback in list(self._subscribers):
            try:
                callback(job_id, payload)
            except Exception:  # pragma: no cover - defensive
                continue

    def _load_known_files(self, root: str) -> dict[str, Any]:
        root_path = str(Path(root).resolve())
        known_files: dict[str, Any] = {}
        for task_name in ("ingest.rescan", "ingest.folder_crawl"):
            for record in self.repo.list_completed(task_name):
                extra = record.get("extra_data") or {}
                job_data = extra.get("job", {})
                params = job_data.get("params", {})
                if str(params.get("root")) != root_path:
                    continue
                summary = extra.get("summary", {})
                files = summary.get("known_files")
                if files:
                    known_files = dict(files)
                    return known_files
        return known_files

    @staticmethod
    def _normalize_path(path: Path) -> str:
        return str(path.resolve())

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _utcnow() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

