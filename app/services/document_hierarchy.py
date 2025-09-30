"""Services for building document hierarchies and metadata views."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from app.storage import DocumentRepository

from ..logging import log_call


logger = logging.getLogger(__name__)


class DocumentHierarchyService:
    """Provide hierarchical views of documents for the UI layer."""

    @log_call(logger=logger)
    def __init__(self, documents: DocumentRepository) -> None:
        self.documents = documents

    @log_call(logger=logger, include_result=True)
    def build_folder_tree(self, project_id: int) -> dict[str, Any]:
        """Return a nested folder tree with documents attached at each node."""

        documents = self.documents.list_for_project(project_id)
        base_path = self._determine_base_path(documents)
        root = self._create_node(name="", path=str(base_path) if base_path is not None else None)
        index: dict[str | None, dict[str, Any]] = {None: root}
        for document in documents:
            folder = self._normalize_folder(document.get("folder_path"))
            relative = self._relative_folder(folder, base_path)
            for ancestor in self._iter_folder_paths(relative):
                if ancestor not in index:
                    parent = self._parent_folder_key(ancestor)
                    parent_node = index[parent]
                    absolute_path = self._absolute_path(base_path, ancestor)
                    node = self._create_node(name=self._node_name(ancestor), path=absolute_path)
                    parent_node["children"].append(node)
                    index[ancestor] = node
            index[relative]["documents"].append(document)
        self._sort_tree(root)
        logger.info(
            "Built folder tree",
            extra={
                "project_id": project_id,
                "document_count": len(documents),
                "node_count": len(index),
            },
        )
        return root

    @log_call(logger=logger, include_result=True)
    def list_documents_for_scope(
        self,
        project_id: int,
        *,
        tags: Iterable[int] | None = None,
        folder: str | Path | None = None,
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        """Return documents matching ``tags``/``folder`` enriched with tag metadata."""

        documents = self.documents.list_for_scope(
            project_id,
            tags=tags,
            folder=folder,
            recursive=recursive,
        )
        return [self._with_tags(document) for document in documents]

    @log_call(logger=logger, include_result=True)
    def get_document_view(self, document_id: int) -> dict[str, Any] | None:
        """Return a metadata view for ``document_id`` including its tags."""

        document = self.documents.get(document_id)
        if document is None:
            return None
        return self._with_tags(document)

    @log_call(logger=logger)
    def refresh_tag_counts(self, project_id: int | None = None) -> None:
        """Recalculate tag counts via the repository helper."""

        self.documents.refresh_tag_counts(project_id)

    @staticmethod
    @log_call(logger=logger, include_result=True)
    def _normalize_folder(folder: Any) -> str | None:
        if folder in (None, ""):
            return None
        return str(Path(folder))

    @staticmethod
    @log_call(logger=logger, include_result=True)
    def _relative_folder(folder: str | None, base_path: Path | None) -> str | None:
        if folder is None:
            return None
        if base_path is None:
            return str(Path(folder))
        folder_path = Path(folder)
        try:
            relative = folder_path.relative_to(base_path)
        except ValueError:
            return str(folder_path)
        relative_str = str(relative)
        return None if relative_str in ("", ".") else relative_str

    @staticmethod
    @log_call(logger=logger, include_result=True)
    def _absolute_path(base_path: Path | None, relative: str | None) -> str | None:
        if relative in (None, ""):
            return str(base_path) if base_path is not None else None
        if base_path is None:
            return str(Path(relative))
        return str((base_path / relative))

    @staticmethod
    @log_call(logger=logger, include_result=True)
    def _node_name(path: str | None) -> str:
        if path is None:
            return ""
        name = Path(path).name
        return name or str(Path(path))

    @staticmethod
    @log_call(logger=logger, include_result=True)
    def _create_node(*, name: str, path: str | None) -> dict[str, Any]:
        return {"name": name, "path": path, "documents": [], "children": []}

    @staticmethod
    @log_call(logger=logger, include_result=True)
    def _parent_folder_key(path: str | None) -> str | None:
        if path in (None, "", "."):
            return None
        current = Path(path)
        parent = current.parent
        parent_str = str(parent)
        if parent == current or parent_str in ("", "."):
            return None
        return parent_str

    @staticmethod
    @log_call(logger=logger, include_result=True)
    def _iter_folder_paths(folder: str | None) -> Iterable[str | None]:
        yield None
        if folder in (None, "", "."):
            return
        path = Path(folder)
        ancestors: list[Path] = []
        current = path
        while True:
            ancestors.append(current)
            parent = current.parent
            parent_str = str(parent)
            if parent == current or parent_str in ("", "."):
                break
            current = parent
        for ancestor in reversed(ancestors):
            yield str(ancestor)

    @staticmethod
    @log_call(logger=logger, include_result=True)
    def _determine_base_path(documents: Iterable[dict[str, Any]]) -> Path | None:
        folders = [Path(doc["folder_path"]) for doc in documents if doc.get("folder_path")]
        if not folders:
            return None
        base = folders[0]
        for folder in folders[1:]:
            while not folder.is_relative_to(base):
                base = base.parent
                if base.parent == base:
                    break
        return base

    @log_call(logger=logger, include_result=True)
    def _with_tags(self, document: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(document)
        enriched["tags"] = self.documents.list_tags_for_document(document["id"])
        return enriched

    @log_call(logger=logger)
    def _sort_tree(self, node: dict[str, Any]) -> None:
        node["documents"].sort(key=lambda item: (item.get("title") or "").lower())
        children = node["children"]
        children.sort(key=lambda item: item["name"].lower())
        for child in children:
            self._sort_tree(child)
