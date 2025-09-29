from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.parsers import ParsedDocument
from app.retrieval import SearchService
from app.services import DocumentHierarchyService
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
def ingest_repo(database: DatabaseManager) -> IngestDocumentRepository:
    return IngestDocumentRepository(database)


@pytest.fixture()
def chat_repo(database: DatabaseManager) -> ChatRepository:
    return ChatRepository(database)


def _store_ingest_document(
    repo: IngestDocumentRepository,
    path: Path,
    text: str,
) -> None:
    parsed = ParsedDocument(text=text, metadata={})
    repo.store_version(
        path=str(path),
        checksum=None,
        size=None,
        mtime=None,
        ctime=None,
        parsed=parsed,
    )


def test_search_service_scope_and_highlight(
    tmp_path: Path,
    project_repo: ProjectRepository,
    document_repo: DocumentRepository,
    ingest_repo: IngestDocumentRepository,
    chat_repo: ChatRepository,
) -> None:
    base = tmp_path / "corpus"
    alpha_path = base / "alpha" / "spec.txt"
    beta_path = base / "beta" / "notes.txt"
    alpha_path.parent.mkdir(parents=True)
    beta_path.parent.mkdir(parents=True)
    alpha_path.write_text("Alpha specification includes the starlight keyword.")
    beta_path.write_text("Beta notes mention starlight from another source.")

    project = project_repo.create("Search Project")
    doc_alpha = document_repo.create(project["id"], "Alpha", source_path=alpha_path)
    doc_beta = document_repo.create(project["id"], "Beta", source_path=beta_path)

    tag_focus = document_repo.create_tag(project["id"], "focus")
    tag_other = document_repo.create_tag(project["id"], "other")
    document_repo.tag_document(doc_alpha["id"], tag_focus["id"])
    document_repo.tag_document(doc_beta["id"], tag_other["id"])

    _store_ingest_document(ingest_repo, alpha_path, alpha_path.read_text())
    _store_ingest_document(ingest_repo, beta_path, beta_path.read_text())

    chat = chat_repo.create(project["id"], "Analysis")
    service = SearchService(ingest_repo, document_repo, chat_repo)

    first_results = service.search_documents(
        "starlight",
        project_id=project["id"],
        chat_id=chat["id"],
        tags=[tag_focus["id"]],
        save_scope=True,
    )
    assert [item["document"]["id"] for item in first_results] == [doc_alpha["id"]]
    assert "<mark>starlight</mark>" in first_results[0]["highlight"]
    assert "starlight" in first_results[0]["context"]

    follow_up = service.search_documents(
        "starlight",
        project_id=project["id"],
        chat_id=chat["id"],
    )
    assert [item["document"]["id"] for item in follow_up] == [doc_alpha["id"]]

    beta_folder = document_repo.get(doc_beta["id"])["folder_path"]
    updated_results = service.search_documents(
        "starlight",
        project_id=project["id"],
        chat_id=chat["id"],
        tags=[tag_other["id"]],
        folder=beta_folder,
        save_scope=True,
    )
    assert [item["document"]["id"] for item in updated_results] == [doc_beta["id"]]

    contexts = service.retrieve_context_snippets(
        "starlight",
        project_id=project["id"],
        include_identifiers=[str(doc_alpha["id"])],
    )
    assert any("starlight" in snippet for snippet in contexts)


def test_context_snippets_relax_query_terms(
    tmp_path: Path,
    project_repo: ProjectRepository,
    document_repo: DocumentRepository,
    ingest_repo: IngestDocumentRepository,
    chat_repo: ChatRepository,
) -> None:
    corpus_dir = tmp_path / "docs"
    doc_path = corpus_dir / "design.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    content = """
    # DataMiner Architecture

    The DataMiner system includes a retrieval pipeline with ranking heuristics.
    """
    doc_path.write_text(content)

    project = project_repo.create("Relaxed Query")
    document_repo.create(project["id"], "Design", source_path=doc_path)
    _store_ingest_document(ingest_repo, doc_path, content)

    service = SearchService(ingest_repo, document_repo, chat_repo)

    question = "What does the DataMiner system include for its architecture?"
    snippets = service.retrieve_context_snippets(question, project_id=project["id"])

    assert any("DataMiner" in snippet for snippet in snippets)


def test_collect_context_records_includes_metadata(
    tmp_path: Path,
    project_repo: ProjectRepository,
    document_repo: DocumentRepository,
    ingest_repo: IngestDocumentRepository,
    chat_repo: ChatRepository,
) -> None:
    base = tmp_path / "corpus"
    doc_path = base / "report.txt"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("The quarterly report covers revenue and expenses in detail.")

    project = project_repo.create("Metadata Project")
    document = document_repo.create(project["id"], "Report", source_path=doc_path)
    _store_ingest_document(ingest_repo, doc_path, doc_path.read_text())

    service = SearchService(ingest_repo, document_repo, chat_repo)

    records = service.collect_context_records("revenue", project_id=project["id"])

    assert records
    first = records[0]
    assert first["document"]["id"] == document["id"]
    assert "context" in first and "revenue" in first["context"].lower()
    assert first["chunk"]["id"]


def test_collect_context_records_handles_path_casing(
    tmp_path: Path,
    project_repo: ProjectRepository,
    document_repo: DocumentRepository,
    ingest_repo: IngestDocumentRepository,
    chat_repo: ChatRepository,
) -> None:
    base = tmp_path / "casefold"
    base.mkdir(parents=True, exist_ok=True)
    doc_path = base / "Report.TXT"
    doc_path.write_text("Casefolding matters for retrieval.")

    project = project_repo.create("Casefold Project")
    document_repo.create(project["id"], "Report", source_path=doc_path)

    ingest_repo.store_version(
        path=str(doc_path).lower(),
        checksum=None,
        size=None,
        mtime=None,
        ctime=None,
        parsed=ParsedDocument(text=doc_path.read_text(), metadata={}),
    )

    service = SearchService(ingest_repo, document_repo, chat_repo)

    records = service.collect_context_records("casefolding", project_id=project["id"])

    assert records, "Expected retrieval records despite path casing differences"
    assert any(
        record.get("context", "").lower().startswith("casefolding") for record in records
    )


def test_search_service_fallback_skips_unscoped_results(
    tmp_path: Path,
    project_repo: ProjectRepository,
    document_repo: DocumentRepository,
    ingest_repo: IngestDocumentRepository,
    chat_repo: ChatRepository,
) -> None:
    base = tmp_path / "fallback"
    tracked_path = base / "tracked.txt"
    tracked_path.parent.mkdir(parents=True, exist_ok=True)
    tracked_path.write_text("Delta-only insight for the project.", encoding="utf-8")

    external_path = base / "external.txt"
    external_path.write_text("Alpha delta combined in external corpus.", encoding="utf-8")

    project = project_repo.create("Fallback Project")
    tracked_doc = document_repo.create(project["id"], "Tracked", source_path=tracked_path)

    _store_ingest_document(ingest_repo, tracked_path, tracked_path.read_text(encoding="utf-8"))
    _store_ingest_document(ingest_repo, external_path, external_path.read_text(encoding="utf-8"))

    service = SearchService(ingest_repo, document_repo, chat_repo)

    results = service.search_documents("alpha delta", project_id=project["id"], limit=3)

    assert results
    assert [item["document"]["id"] for item in results] == [tracked_doc["id"]]


def test_document_hierarchy_service(
    tmp_path: Path,
    project_repo: ProjectRepository,
    document_repo: DocumentRepository,
) -> None:
    base = tmp_path / "hier"
    first = base / "alpha" / "doc1.txt"
    second = base / "alpha" / "deep" / "doc2.txt"
    third = base / "beta" / "doc3.txt"
    third.parent.mkdir(parents=True, exist_ok=True)
    second.parent.mkdir(parents=True, exist_ok=True)
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("one")
    second.write_text("two")
    third.write_text("three")

    project = project_repo.create("Hierarchy")
    doc_one = document_repo.create(project["id"], "One", source_path=first)
    doc_two = document_repo.create(project["id"], "Two", source_path=second)
    doc_three = document_repo.create(project["id"], "Three", source_path=third)

    tag_deep = document_repo.create_tag(project["id"], "deep")
    document_repo.tag_document(doc_two["id"], tag_deep["id"])

    hierarchy = DocumentHierarchyService(document_repo)
    tree = hierarchy.build_folder_tree(project["id"])
    top_level = [child["name"] for child in tree["children"]]
    assert top_level == sorted(top_level)
    assert {child["name"] for child in tree["children"]} == {"alpha", "beta"}

    alpha_node = next(child for child in tree["children"] if child["name"] == "alpha")
    assert {doc["title"] for doc in alpha_node["documents"]} == {"One"}

    deep_node = next(child for child in alpha_node["children"] if child["name"] == "deep")
    assert {doc["title"] for doc in deep_node["documents"]} == {"Two"}

    beta_node = next(child for child in tree["children"] if child["name"] == "beta")
    assert {doc["title"] for doc in beta_node["documents"]} == {"Three"}

    view = hierarchy.get_document_view(doc_two["id"])
    assert view is not None
    assert {tag["name"] for tag in view["tags"]} == {"deep"}

    scoped = hierarchy.list_documents_for_scope(project["id"], tags=[tag_deep["id"]])
    assert [doc["id"] for doc in scoped] == [doc_two["id"]]
    assert scoped[0]["tags"][0]["name"] == "deep"
