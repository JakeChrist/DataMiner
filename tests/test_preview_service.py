from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.ingest.parsers import parse_file
from app.ingest.preview import PreviewService
from app.storage import DatabaseManager, IngestDocumentRepository


@pytest.fixture()
def db_manager(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "preview.db"
    manager = DatabaseManager(db_path)
    manager.initialize()
    yield manager
    manager.close()


def test_preview_highlights_terms(tmp_path: Path, db_manager: DatabaseManager) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Alpha beta gamma delta epsilon", encoding="utf-8")
    parsed = parse_file(file_path)
    checksum = hashlib.sha256(file_path.read_bytes()).hexdigest()
    repo = IngestDocumentRepository(db_manager)
    stored = repo.store_version(
        path=str(file_path),
        checksum=checksum,
        size=file_path.stat().st_size,
        mtime=file_path.stat().st_mtime,
        ctime=file_path.stat().st_ctime,
        parsed=parsed,
        base_metadata={"file": {"size": file_path.stat().st_size}},
    )

    preview = PreviewService(db_manager)
    result = preview.get_highlighted_passages(stored["id"], ["gamma"])
    assert "<mark>gamma</mark>" in result["snippet"]
    assert result["page"] == 1 or result["page"] is None
    page = preview.get_page(stored["id"], 1)
    assert "Alpha" in page["text"]

    results = repo.search("gamma")
    assert results
    assert "<mark>gamma</mark>" in results[0]["highlight"]
