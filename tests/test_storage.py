from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.parsers import ParsedDocument
from app.storage import (
    ChatRepository,
    DatabaseManager,
    DocumentRepository,
    IngestDocumentRepository,
    ProjectRepository,
)


@pytest.fixture()
def database(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "dataminer.db"
    manager = DatabaseManager(db_path)
    manager.initialize()
    yield manager
    manager.close()


@pytest.fixture()
def project_repo(database: DatabaseManager) -> ProjectRepository:
    return ProjectRepository(database)


@pytest.fixture()
def document_repo(database: DatabaseManager) -> DocumentRepository:
    return DocumentRepository(database)


@pytest.fixture()
def chat_repo(database: DatabaseManager) -> ChatRepository:
    return ChatRepository(database)


@pytest.fixture()
def ingest_repo(database: DatabaseManager) -> IngestDocumentRepository:
    return IngestDocumentRepository(database)


def test_project_crud(project_repo: ProjectRepository) -> None:
    project = project_repo.create("Project Alpha", description="Primary")
    assert project["name"] == "Project Alpha"

    fetched = project_repo.get(project["id"])
    assert fetched == project

    updated = project_repo.update(project["id"], name="Renamed")
    assert updated["name"] == "Renamed"

    all_projects = project_repo.list()
    assert len(all_projects) == 1

    project_repo.delete(project["id"])
    assert project_repo.get(project["id"]) is None


def test_document_crud_and_cascade(
    tmp_path: Path,
    project_repo: ProjectRepository,
    document_repo: DocumentRepository,
    chat_repo: ChatRepository,
) -> None:
    project = project_repo.create("Project Beta")
    document = document_repo.create(
        project["id"],
        "Spec Sheet",
        source_type="file",
        source_path=tmp_path / "spec.pdf",
        metadata={"pages": 12},
    )
    assert document["metadata"] == {"pages": 12}

    file_path = tmp_path / "spec.pdf"
    file_path.write_text("dummy data")
    version = document_repo.add_file_version(
        document["id"],
        file_path=file_path,
        checksum="abc123",
        file_size=file_path.stat().st_size,
    )
    assert version["version"] == 1

    tag = document_repo.create_tag(project["id"], "urgent")
    document_repo.tag_document(document["id"], tag["id"])
    tags = document_repo.list_tags_for_document(document["id"])
    assert tags[0]["name"] == "urgent"

    chat = chat_repo.create(project["id"], "Review")
    citation = chat_repo.add_citation(chat["id"], document_id=document["id"], snippet="See section 2")
    assert citation["document_id"] == document["id"]
    summary = chat_repo.add_reasoning_summary(chat["id"], "Relevant to requirements")
    assert "Relevant" in summary["content"]

    # deleting the project should cascade to documents, versions, chats, etc.
    project_repo.delete(project["id"])
    assert document_repo.get(document["id"]) is None
    assert document_repo.get_file_version(version["id"]) is None
    assert chat_repo.get(chat["id"]) is None
    assert chat_repo.get_citation(citation["id"]) is None
    assert chat_repo.get_reasoning_summary(summary["id"]) is None

    # ensure the backing file remains untouched
    assert file_path.exists()


def test_project_isolation(
    project_repo: ProjectRepository, document_repo: DocumentRepository
) -> None:
    project_a = project_repo.create("Project A")
    project_b = project_repo.create("Project B")

    doc_a = document_repo.create(project_a["id"], "Doc A")
    doc_b = document_repo.create(project_b["id"], "Doc B")

    docs_for_a = document_repo.list_for_project(project_a["id"])
    docs_for_b = document_repo.list_for_project(project_b["id"])

    assert {doc["id"] for doc in docs_for_a} == {doc_a["id"]}
    assert {doc["id"] for doc in docs_for_b} == {doc_b["id"]}

    # deleting one project should not remove the other project's data
    project_repo.delete(project_a["id"])
    assert document_repo.get(doc_b["id"]) is not None


def test_tag_counts_and_scope_queries(
    tmp_path: Path, project_repo: ProjectRepository, document_repo: DocumentRepository
) -> None:
    project = project_repo.create("Project Tags")
    doc_path = tmp_path / "corpus" / "alpha" / "plan.txt"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text("content")

    document = document_repo.create(
        project["id"],
        "Launch Plan",
        source_path=doc_path,
        metadata={"pages": 3},
    )
    tag_focus = document_repo.create_tag(project["id"], "focus")
    tag_aux = document_repo.create_tag(project["id"], "aux")

    assert tag_focus["document_count"] == 0

    document_repo.tag_document(document["id"], tag_focus["id"])
    document_repo.tag_document(document["id"], tag_aux["id"])

    refreshed = document_repo.get_tag(tag_focus["id"])
    assert refreshed["document_count"] == 1

    tags_for_project = document_repo.list_tags_for_project(project["id"])
    assert {tag["name"] for tag in tags_for_project} == {"focus", "aux"}

    document_repo.untag_document(document["id"], tag_focus["id"])
    assert document_repo.get_tag(tag_focus["id"])["document_count"] == 0

    folder_docs = document_repo.list_for_folder(project["id"], document["folder_path"])
    assert [doc["id"] for doc in folder_docs] == [document["id"]]

    tag_docs = document_repo.list_for_tag(project["id"], tag_aux["id"])
    assert [doc["id"] for doc in tag_docs] == [document["id"]]

    scoped_docs = document_repo.list_for_scope(
        project["id"], tags=[tag_aux["id"]], folder=document["folder_path"]
    )
    assert [doc["id"] for doc in scoped_docs] == [document["id"]]

    document_repo.delete(document["id"])
    assert document_repo.get_tag(tag_aux["id"])["document_count"] == 0


def test_ingest_chunk_storage(
    tmp_path: Path, ingest_repo: IngestDocumentRepository
) -> None:
    path = tmp_path / "evidence.txt"
    path.write_text(
        "Alpha beta gamma delta epsilon zeta eta theta iota.",
        encoding="utf-8",
    )
    parsed = ParsedDocument(text=path.read_text(encoding="utf-8"), metadata={})
    record = ingest_repo.store_version(
        path=str(path),
        checksum=None,
        size=None,
        mtime=None,
        ctime=None,
        parsed=parsed,
    )
    assert record is not None
    chunk_results = ingest_repo.search_chunks("gamma", limit=5)
    assert chunk_results
    first_chunk = chunk_results[0]["chunk"]
    assert "metadata" in first_chunk
    assert "hierarchy_weight" in first_chunk["metadata"]
    assert "score_breakdown" in chunk_results[0]
    texts = [entry["chunk"]["text"] for entry in chunk_results if entry.get("chunk")]
    assert any("gamma" in text for text in texts)
    count = ingest_repo.db.connect().execute(
        "SELECT COUNT(*) FROM ingest_document_chunks"
    ).fetchone()[0]
    assert count >= 1
