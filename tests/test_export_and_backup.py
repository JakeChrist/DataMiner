from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.config import ConfigManager
from app.services.backup_service import BackupService
from app.services.conversation_manager import (
    AssumptionDecision,
    ConversationTurn,
    PlanItem,
    ReasoningArtifacts,
    SelfCheckResult,
)
from app.services.export_service import ExportService
from app.services.project_service import ProjectService


def build_project_service(tmp_path: Path) -> ProjectService:
    config = ConfigManager(app_name="DataMinerTest", filename="projects.json")
    service = ProjectService(storage_root=tmp_path / "storage", config_manager=config)
    return service


def test_export_service_includes_reasoning_and_citations() -> None:
    export = ExportService()
    artifacts = ReasoningArtifacts(
        summary_bullets=["Reviewed primary source"],
        plan_items=[PlanItem(description="Check tables", status="complete")],
        assumptions=["Assumed latest revision"],
        assumption_decision=AssumptionDecision(
            mode="assume", rationale="No conflicting versions"
        ),
        self_check=SelfCheckResult(passed=True, flags=["Verified citations"], notes="All good"),
    )
    turn = ConversationTurn(
        question="What is the outcome?",
        answer="Outcome is 42 [1].",
        citations=[{"source": "Doc A", "snippet": "<mark>42</mark> is the result", "page": 3}],
        reasoning_artifacts=artifacts,
        asked_at=datetime(2024, 1, 1, 12, 0, 0),
        answered_at=datetime(2024, 1, 1, 12, 0, 5),
        latency_ms=5000,
        token_usage={"prompt_tokens": 10, "completion_tokens": 15},
    )

    markdown = export.conversation_to_markdown([turn], metadata={"Project": "Demo"})
    assert "Reasoning" in markdown
    assert "Doc A" in markdown
    assert "Plan" in markdown
    assert "Token Usage" in markdown

    html = export.conversation_to_html([turn], metadata={"Project": "Demo"})
    assert "<h2>Turn 1</h2>" in html
    assert "Doc A" in html
    assert "Reasoning" in html
    assert "Self-check" in html


def test_project_service_lifecycle_and_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    service = build_project_service(tmp_path)
    try:
        default_project = service.active_project()
        service.save_conversation_settings(default_project.id, {"show_plan": False})

        created = service.create_project("Analysis", activate=True)
        service.save_conversation_settings(created.id, {"show_plan": True, "sources_only": True})

        service.set_active_project(default_project.id)
        original_settings = service.load_conversation_settings(default_project.id)
        assert original_settings.get("show_plan") is False

        service.set_active_project(created.id)
        created_settings = service.load_conversation_settings(created.id)
        assert created_settings.get("show_plan") is True
        assert created_settings.get("sources_only") is True

        storage_path = service.get_project_storage(created.id)
        assert storage_path.exists()

        service.set_active_project(default_project.id)
        service.delete_project(created.id)
        remaining_ids = {record.id for record in service.list_projects()}
        assert created.id not in remaining_ids
    finally:
        service.shutdown()


def test_backup_restore_persists_audit_trail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    service = build_project_service(tmp_path)
    backup = BackupService(service)
    try:
        project = service.active_project()
        log = service.background_tasks.create(
            "ingest", status="running", message="Started"
        )
        storage_dir = service.get_project_storage(project.id)
        cache_file = storage_dir / "cache" / "data.txt"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("cached", encoding="utf-8")

        archive = backup.create_backup(tmp_path)

        with service.database_manager.transaction() as connection:
            connection.execute("DELETE FROM background_task_logs")
        shutil.rmtree(storage_dir)

        backup.restore_backup(archive)

        restored = service.background_tasks.get(log["id"])
        assert restored is not None
        assert restored["status"] == "running"
        assert (service.get_project_storage(project.id) / "cache" / "data.txt").exists()
    finally:
        service.shutdown()
