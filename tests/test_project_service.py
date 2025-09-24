from __future__ import annotations

import json
from pathlib import Path

from app.services.project_service import ProjectService


class InMemoryConfig:
    """Minimal config manager stub for testing."""

    def __init__(self) -> None:
        self._data: dict[str, object] = {}

    def load(self) -> dict[str, object]:
        return json.loads(json.dumps(self._data))

    def save(self, data: dict[str, object]) -> None:
        self._data = json.loads(json.dumps(data))


def test_corpus_root_management(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    config = InMemoryConfig()
    service = ProjectService(storage_root=storage_root, config_manager=config)
    try:
        project = service.active_project()
        assert service.list_corpus_roots(project.id) == []

        first = tmp_path / "docs"
        first.mkdir()
        service.add_corpus_root(project.id, first)
        assert service.list_corpus_roots(project.id) == [str(first.resolve())]

        # Adding the same folder again should not create duplicates.
        service.add_corpus_root(project.id, first)
        assert service.list_corpus_roots(project.id) == [str(first.resolve())]

        second = tmp_path / "more"
        second.mkdir()
        service.add_corpus_root(project.id, second)
        roots = service.list_corpus_roots(project.id)
        assert set(roots) == {str(first.resolve()), str(second.resolve())}

        service.remove_corpus_root(project.id, first)
        assert service.list_corpus_roots(project.id) == [str(second.resolve())]

        service.clear_corpus_roots(project.id)
        assert service.list_corpus_roots(project.id) == []
    finally:
        service.shutdown()
