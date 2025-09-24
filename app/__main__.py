"""Entry point for launching the DataMiner desktop application."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from .logging import setup_logging
from .ingest.service import IngestService
from .services.backup_service import BackupService
from .services.document_hierarchy import DocumentHierarchyService
from .services.export_service import ExportService
from .services.lmstudio_client import LMStudioClient
from .services.progress_service import ProgressService
from .services.project_service import ProjectService
from .services.settings_service import SettingsService
from .ui import MainWindow


def main() -> None:
    """Start the PyQt6 application."""
    logger = setup_logging()
    logger.debug("Starting QApplication")

    app = QApplication(sys.argv)
    app.setApplicationName("DataMiner")

    settings_service = SettingsService()
    progress_service = ProgressService()
    lmstudio_client = LMStudioClient()
    project_service = ProjectService()
    ingest_service = IngestService(project_service.database_manager)
    document_hierarchy = DocumentHierarchyService(project_service.documents)
    export_service = ExportService()
    backup_service = BackupService(project_service)

    window = MainWindow(
        settings_service=settings_service,
        progress_service=progress_service,
        lmstudio_client=lmstudio_client,
        project_service=project_service,
        ingest_service=ingest_service,
        document_hierarchy=document_hierarchy,
        export_service=export_service,
        backup_service=backup_service,
    )
    window.show()

    logger.info("Application started")
    try:
        exit_code = app.exec()
    finally:
        ingest_service.shutdown(wait=False)
        project_service.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
