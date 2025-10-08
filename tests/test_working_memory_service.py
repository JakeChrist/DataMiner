from __future__ import annotations

from pathlib import Path

from app.services.conversation_manager import (
    EvidenceRecord,
    PlanItem,
    StateDigestEntry,
    StepContextBatch,
    StepResult,
)
from app.services.working_memory import WorkingMemoryService
from app.storage.database import (
    DatabaseManager,
    ProjectRepository,
    WorkingMemoryRepository,
)


def test_working_memory_service_persists_and_searches(tmp_path) -> None:
    db_path = Path(tmp_path) / "memory.db"
    manager = DatabaseManager(db_path)
    manager.initialize()
    projects = ProjectRepository(manager)
    project = projects.create("Test Project")
    project_id = int(project["id"])
    repository = WorkingMemoryRepository(manager)
    service = WorkingMemoryService(repository)

    plan = [PlanItem(description="Collect metrics", status="done")]
    step_result = StepResult(
        index=1,
        description="Collect metrics",
        answer="Metrics improved over the quarter.",
        citation_indexes=[1],
        contexts=[StepContextBatch(snippets=["Metrics snippet"], documents=[])],
    )
    digest = [
        StateDigestEntry(
            step_index=1,
            summary="Metrics highlight sustained improvement.",
            citation_indexes=[1],
        )
    ]
    evidence = [
        EvidenceRecord(
            step_index=1,
            intent="Review quarterly metrics",
            documents=[{"id": "doc-1", "title": "Metrics Report"}],
            snippets=["Quarterly metrics show growth"],
        )
    ]

    service.store_turn_memory(
        project_id=project_id,
        turn_index=1,
        question="Summarize quarterly metrics performance.",
        answer="Metrics improved over the quarter.",
        plan=plan,
        step_results=[step_result],
        state_digest=digest,
        evidence_log=evidence,
    )

    entries = repository.list_for_project(project_id)
    kinds = {entry["kind"] for entry in entries}
    assert {"question", "final_answer", "plan", "step_result", "state_digest", "evidence"}.issubset(kinds)

    results = service.collect_context_records("metrics", project_id=project_id, limit=10)
    assert results
    assert any(record["document"].get("working_memory") for record in results)

    manager.close()
