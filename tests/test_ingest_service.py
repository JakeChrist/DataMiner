from __future__ import annotations

import os
import threading
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.ingest.service import IngestService, TaskStatus
from app.storage import (
    DatabaseManager,
    DocumentRepository,
    IngestDocumentRepository,
    ProjectRepository,
)


@pytest.fixture()
def db_manager(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "ingest.db"
    manager = DatabaseManager(db_path)
    manager.initialize()
    yield manager
    manager.close()


def _wait_for_status(service: IngestService, job_id: int, status: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = service.repo.get(job_id)
        if record and record["status"] == status:
            return
        time.sleep(0.05)
    pytest.fail(f"Job {job_id} did not reach status {status}")


def _collect_known_files(record: dict[str, object]) -> dict[str, object]:
    extra = record.get("extra_data") or {}
    return extra.get("summary", {}).get("known_files", {})  # type: ignore[return-value]


def test_folder_crawl_tracks_progress_and_summary(tmp_path: Path, db_manager: DatabaseManager) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "keep.txt").write_text("alpha", encoding="utf-8")
    (docs / "skip.log").write_text("should be ignored", encoding="utf-8")
    (docs / "other.txt").write_text("beta", encoding="utf-8")

    service = IngestService(db_manager, worker_idle_sleep=0.01)
    updates: list[tuple[int, str]] = []

    def on_update(job_id: int, payload: dict[str, object]) -> None:
        updates.append((job_id, payload.get("status", "") if isinstance(payload, dict) else ""))

    service.subscribe(on_update)

    job_id = service.queue_folder_crawl(
        project_id=1,
        root=docs,
        include=["*.txt"],
        exclude=["other.*"],
    )

    assert service.wait_for_completion(job_id, timeout=5.0)
    record = service.repo.get(job_id)
    assert record is not None
    assert record["status"] == TaskStatus.COMPLETED
    extra = record["extra_data"]
    assert extra["progress"]["total"] == 1
    assert extra["summary"]["success_count"] == 1
    known_files = extra["summary"]["known_files"]
    assert list(known_files.keys()) == [str((docs / "keep.txt").resolve())]
    doc_repo = IngestDocumentRepository(db_manager)
    stored = doc_repo.get_latest_by_path(docs / "keep.txt")
    assert stored is not None
    assert stored["preview"].startswith("alpha")
    assert stored["needs_ocr"] is False
    assert any(status == TaskStatus.COMPLETED for _, status in updates)

    service.shutdown()


def test_folder_crawl_recurses_into_subdirectories(
    tmp_path: Path, db_manager: DatabaseManager
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    nested = root / "nested"
    nested.mkdir()
    nested_file = nested / "note.txt"
    nested_file.write_text("payload", encoding="utf-8")

    service = IngestService(db_manager, worker_idle_sleep=0.01)
    job_id = service.queue_folder_crawl(None, root, include=["*.txt"])
    assert service.wait_for_completion(job_id, timeout=5.0)

    record = service.repo.get(job_id)
    assert record is not None
    known_files = record["extra_data"]["summary"]["known_files"]
    assert str(nested_file.resolve()) in known_files

    service.shutdown()


def test_completed_job_populates_document_repository(
    tmp_path: Path, db_manager: DatabaseManager
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tracked = root / "tracked.txt"
    payload = "indexed payload"
    tracked.write_text(payload, encoding="utf-8")

    service = IngestService(db_manager, worker_idle_sleep=0.01)
    projects = ProjectRepository(db_manager)
    project = projects.create("Test Project")
    project_id = int(project["id"])
    job_id = service.queue_folder_crawl(project_id, root, include=["*.txt"])
    assert service.wait_for_completion(job_id, timeout=5.0)

    repo = DocumentRepository(db_manager)
    ingest_repo = IngestDocumentRepository(db_manager)
    documents = repo.list_for_project(project_id)
    assert len(documents) == 1
    document = documents[0]
    assert document["source_path"] == str(tracked.resolve())
    assert document["metadata"]["file"]["size"] == tracked.stat().st_size
    assert ingest_repo.get_latest_by_path(tracked) is not None

    remove_job = service.queue_remove(project_id, root, [tracked])
    assert service.wait_for_completion(remove_job, timeout=5.0)
    assert repo.list_for_project(project_id) == []
    assert ingest_repo.get_latest_by_path(tracked) is None

    service.shutdown()


def test_rescan_detects_changes_and_removals(tmp_path: Path, db_manager: DatabaseManager) -> None:
    root = tmp_path / "library"
    root.mkdir()
    file_a = root / "a.txt"
    file_b = root / "b.txt"
    file_a.write_text("one", encoding="utf-8")
    file_b.write_text("two", encoding="utf-8")

    service = IngestService(db_manager, worker_idle_sleep=0.01)
    first_job = service.queue_folder_crawl(None, root, include=["*.txt"])
    assert service.wait_for_completion(first_job, timeout=5.0)
    record = service.repo.get(first_job)
    assert record is not None
    initial_known = _collect_known_files(record)
    assert len(initial_known) == 2

    # Modify a file, remove another, and add a new one before rescanning.
    file_a.write_text("one-modified", encoding="utf-8")
    os.remove(file_b)
    file_c = root / "c.txt"
    file_c.write_text("three", encoding="utf-8")

    second_job = service.queue_rescan(None, root, include=["*.txt"])
    assert service.wait_for_completion(second_job, timeout=5.0)
    rescan_record = service.repo.get(second_job)
    assert rescan_record is not None
    summary = rescan_record["extra_data"]["summary"]
    assert summary["success_count"] >= 2  # modified + new
    assert set(summary.get("removed", [])) == {str(file_b.resolve())}
    known_files = summary["known_files"]
    assert str(file_a.resolve()) in known_files
    assert str(file_c.resolve()) in known_files
    assert str(file_b.resolve()) not in known_files
    ingest_repo = IngestDocumentRepository(db_manager)
    assert ingest_repo.get_latest_by_path(file_b) is None
    assert ingest_repo.get_latest_by_path(file_a) is not None
    assert ingest_repo.get_latest_by_path(file_c) is not None

    service.shutdown()


def test_folder_crawl_handles_unreadable_files(
    tmp_path: Path, db_manager: DatabaseManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "restricted"
    root.mkdir()
    unreadable = root / "secret.txt"
    unreadable.write_text("top secret", encoding="utf-8")

    def fake_hash(_path: Path) -> str:
        raise OSError("denied")

    monkeypatch.setattr(
        IngestService,
        "_hash_file",
        staticmethod(fake_hash),
    )

    service = IngestService(db_manager, worker_idle_sleep=0.01)
    job_id = service.queue_folder_crawl(None, root, include=["*.txt"])
    assert service.wait_for_completion(job_id, timeout=5.0)

    record = service.repo.get(job_id)
    assert record is not None
    assert record["status"] == TaskStatus.COMPLETED
    summary = record["extra_data"]["summary"]
    assert summary["success_count"] == 0
    assert summary["failure_count"] == 1
    assert summary["known_files"] == {}
    errors = record["extra_data"].get("errors", [])
    assert any("Unable to read" in str(error) for error in errors)

    service.shutdown()


def test_pause_and_resume_persists_progress(tmp_path: Path, db_manager: DatabaseManager) -> None:
    root = tmp_path / "pause"
    root.mkdir()
    for idx in range(5):
        (root / f"file-{idx}.txt").write_text(f"payload-{idx}", encoding="utf-8")

    service = IngestService(db_manager, worker_idle_sleep=0.01)
    pause_triggered = threading.Event()

    def on_update(job_id: int, payload: dict[str, object]) -> None:
        if payload.get("status") == TaskStatus.RUNNING:
            progress = payload.get("progress", {})
            if isinstance(progress, dict) and progress.get("processed", 0) >= 1 and not pause_triggered.is_set():
                pause_triggered.set()
                service.pause_job(job_id)

    service.subscribe(on_update)
    job_id = service.queue_folder_crawl(None, root, include=["*.txt"])

    assert pause_triggered.wait(timeout=5.0)
    _wait_for_status(service, job_id, TaskStatus.PAUSED)
    record = service.repo.get(job_id)
    assert record is not None
    state = record["extra_data"]["state"]
    assert 0 < state["position"] < len(state["pending_files"])

    service.resume_job(job_id)
    assert service.wait_for_completion(job_id, timeout=5.0)
    final_record = service.repo.get(job_id)
    assert final_record is not None
    assert final_record["status"] == TaskStatus.COMPLETED
    assert final_record["extra_data"]["summary"]["success_count"] == 5
    service.shutdown()


def test_cancel_rolls_back_partial_progress(tmp_path: Path, db_manager: DatabaseManager) -> None:
    root = tmp_path / "cancel"
    root.mkdir()
    for idx in range(3):
        (root / f"doc-{idx}.txt").write_text("data", encoding="utf-8")

    service = IngestService(db_manager, worker_idle_sleep=0.01)
    cancel_triggered = threading.Event()

    def on_update(job_id: int, payload: dict[str, object]) -> None:
        if payload.get("status") == TaskStatus.RUNNING:
            progress = payload.get("progress", {})
            if isinstance(progress, dict) and progress.get("processed", 0) >= 1 and not cancel_triggered.is_set():
                cancel_triggered.set()
                service.cancel_job(job_id)

    service.subscribe(on_update)
    job_id = service.queue_folder_crawl(None, root, include=["*.txt"])
    assert cancel_triggered.wait(timeout=5.0)
    _wait_for_status(service, job_id, TaskStatus.CANCELLED)
    record = service.repo.get(job_id)
    assert record is not None
    summary = record["extra_data"]["summary"]
    assert summary["rolled_back"] >= 1
    assert summary["known_files"] == {}

    service.shutdown()


def test_load_known_files_reflects_removals(tmp_path: Path, db_manager: DatabaseManager) -> None:
    root = tmp_path / "removal"
    root.mkdir()
    tracked = root / "tracked.txt"
    tracked.write_text("payload", encoding="utf-8")

    service = IngestService(db_manager, worker_idle_sleep=0.01)
    first_job = service.queue_folder_crawl(None, root, include=["*.txt"])
    assert service.wait_for_completion(first_job, timeout=5.0)

    remove_job = service.queue_remove(None, root, [tracked])
    assert service.wait_for_completion(remove_job, timeout=5.0)

    known_files = service._load_known_files(str(root))
    assert known_files == {}

    service.shutdown()


def test_pdf_without_text_flags_ocr(tmp_path: Path, db_manager: DatabaseManager) -> None:
    pdf_path = tmp_path / "blank.pdf"
    fitz = pytest.importorskip("fitz")

    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    service = IngestService(db_manager, worker_idle_sleep=0.01)
    job_id = service.queue_file_add(None, [pdf_path], include=["*.pdf"])
    assert service.wait_for_completion(job_id, timeout=5.0)
    record = service.repo.get(job_id)
    assert record is not None
    assert record["status"] == TaskStatus.COMPLETED
    summary = record["extra_data"]["summary"]
    assert summary.get("needs_ocr")
    needs_ocr_entry = summary["needs_ocr"][0]
    assert "OCR" in needs_ocr_entry["message"]
    repo = IngestDocumentRepository(db_manager)
    stored = repo.get_latest_by_path(pdf_path)
    assert stored is not None
    assert stored["needs_ocr"] is True
    assert stored["ocr_message"]
    service.shutdown()


def test_resume_after_restart(tmp_path: Path, db_manager: DatabaseManager) -> None:
    root = tmp_path / "restart"
    root.mkdir()
    for idx in range(4):
        (root / f"item-{idx}.txt").write_text("restart", encoding="utf-8")

    service_one = IngestService(db_manager, worker_idle_sleep=0.01)
    paused = threading.Event()

    def on_update(job_id: int, payload: dict[str, object]) -> None:
        if payload.get("status") == TaskStatus.RUNNING:
            progress = payload.get("progress", {})
            if isinstance(progress, dict) and progress.get("processed", 0) >= 1 and not paused.is_set():
                paused.set()
                service_one.pause_job(job_id)

    service_one.subscribe(on_update)
    job_id = service_one.queue_folder_crawl(None, root, include=["*.txt"])
    assert paused.wait(timeout=5.0)
    _wait_for_status(service_one, job_id, TaskStatus.PAUSED)
    service_one.shutdown()

    service_two = IngestService(db_manager, worker_idle_sleep=0.01)
    # On restore the job remains paused until explicitly resumed.
    record = service_two.repo.get(job_id)
    assert record is not None
    assert record["status"] == TaskStatus.PAUSED

    service_two.resume_job(job_id)
    assert service_two.wait_for_completion(job_id, timeout=5.0)
    final_record = service_two.repo.get(job_id)
    assert final_record is not None
    assert final_record["status"] == TaskStatus.COMPLETED
    assert final_record["extra_data"]["summary"]["success_count"] == 4

    service_two.shutdown()

