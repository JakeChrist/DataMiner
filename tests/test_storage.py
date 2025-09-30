from __future__ import annotations

from pathlib import Path
import json
import sqlite3
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


def test_initialize_migrates_context_chunking_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE ingest_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                version INTEGER NOT NULL,
                checksum TEXT,
                size INTEGER,
                mtime REAL,
                ctime REAL,
                metadata TEXT,
                text TEXT,
                normalized_text TEXT,
                preview TEXT,
                sections TEXT,
                pages TEXT,
                needs_ocr INTEGER NOT NULL DEFAULT 0,
                ocr_message TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE ingest_document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE VIRTUAL TABLE ingest_document_index
            USING fts5(
                content,
                path UNINDEXED,
                document_id UNINDEXED,
                tokenize='porter'
            );
            INSERT INTO ingest_documents (path, version) VALUES ('/tmp/doc.txt', 1);
            INSERT INTO ingest_document_chunks (
                document_id, chunk_index, text, token_count, start_offset, end_offset
            )
            VALUES (1, 0, 'Legacy content for migration', 4, 0, 24);
            INSERT INTO ingest_document_index (rowid, content, path, document_id)
            VALUES (1, 'Legacy content for migration', '/tmp/doc.txt', 1);
            PRAGMA user_version = 4;
            """
        )

    manager = DatabaseManager(db_path)
    manager.initialize()

    connection = manager.connect()
    columns = connection.execute("PRAGMA table_info(ingest_document_chunks)").fetchall()
    assert any(row["name"] == "metadata" for row in columns)

    schema_sql_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'ingest_document_index'"
    ).fetchone()
    assert schema_sql_row is not None
    assert "chunk_id" in schema_sql_row["sql"]
    assert "chunk_index" in schema_sql_row["sql"]

    migrated_row = connection.execute(
        "SELECT chunk_id, chunk_index, path FROM ingest_document_index"
    ).fetchone()
    assert migrated_row is not None
    assert migrated_row["chunk_id"] == 1
    assert migrated_row["chunk_index"] == 0
    assert migrated_row["path"] == "/tmp/doc.txt"

    metadata_value = connection.execute(
        "SELECT metadata FROM ingest_document_chunks WHERE id = 1"
    ).fetchone()["metadata"]
    assert metadata_value == "{}"


def test_initialize_creates_chunk_table_from_version_three(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy_v3.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE ingest_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                version INTEGER NOT NULL,
                checksum TEXT,
                size INTEGER,
                mtime REAL,
                ctime REAL,
                metadata TEXT,
                text TEXT,
                normalized_text TEXT,
                preview TEXT,
                sections TEXT,
                pages TEXT,
                needs_ocr INTEGER NOT NULL DEFAULT 0,
                ocr_message TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE VIRTUAL TABLE ingest_document_index
            USING fts5(
                content,
                document_id UNINDEXED,
                tokenize='porter'
            );
            INSERT INTO ingest_documents (
                path, version, text, normalized_text
            )
            VALUES ('/tmp/legacy.txt', 1, 'Legacy payload', 'Legacy payload');
            INSERT INTO ingest_document_index (rowid, content, document_id)
            VALUES (1, 'Legacy payload', 1);
            PRAGMA user_version = 3;
            """
        )

    manager = DatabaseManager(db_path)
    manager.initialize()

    connection = manager.connect()
    columns = connection.execute("PRAGMA table_info(ingest_document_chunks)").fetchall()
    names = {row["name"] for row in columns}
    assert {"id", "document_id", "metadata"}.issubset(names)

    chunk_row = connection.execute(
        "SELECT id, document_id, chunk_index, text, metadata FROM ingest_document_chunks"
    ).fetchone()
    assert chunk_row is not None
    assert chunk_row["document_id"] == 1
    assert chunk_row["chunk_index"] == 0
    assert chunk_row["text"] == "Legacy payload"
    metadata = json.loads(chunk_row["metadata"])
    assert metadata["source_path"] == "/tmp/legacy.txt"

    index_row = connection.execute(
        "SELECT chunk_id, chunk_index, path FROM ingest_document_index"
    ).fetchone()
    assert index_row is not None
    assert index_row["chunk_id"] == chunk_row["id"]
    assert index_row["chunk_index"] == 0
    assert index_row["path"] == "/tmp/legacy.txt"

    assert connection.execute("PRAGMA user_version").fetchone()[0] == 5

