"""Backup and restore helpers for DataMiner storage."""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from .project_service import ProjectService
from ..storage.database import SCHEMA_VERSION


MANIFEST_NAME = "manifest.json"


class BackupService:
    """Create and restore archive snapshots of application data."""

    def __init__(self, project_service: ProjectService) -> None:
        self._projects = project_service

    def create_backup(self, destination: str | Path) -> Path:
        destination_path = Path(destination)
        if destination_path.is_dir():
            timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M%S")
            destination_path = destination_path / f"dataminer-backup-{timestamp}.zip"
        if not destination_path.parent.exists():
            destination_path.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "created": _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z"),
            "schema_version": SCHEMA_VERSION,
            "active_project": self._projects.active_project_id,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_export = tmp_path / "database.db"
            self._projects.database_manager.export_database(db_export)
            manifest_path = tmp_path / MANIFEST_NAME
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            derived_root = self._projects.storage_root / "projects"

            with zipfile.ZipFile(destination_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(db_export, "database/dataminer.db")
                if derived_root.exists():
                    for path in derived_root.rglob("*"):
                        if path.is_file():
                            arcname = Path("derived") / path.relative_to(self._projects.storage_root)
                            archive.write(path, arcname)
                archive.write(manifest_path, MANIFEST_NAME)
        return destination_path

    def restore_backup(self, archive_path: str | Path) -> None:
        archive = Path(archive_path)
        if not archive.exists():
            raise FileNotFoundError(archive)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with zipfile.ZipFile(archive, "r") as zip_file:
                zip_file.extractall(tmp_path)
            manifest_path = tmp_path / MANIFEST_NAME
            if not manifest_path.exists():
                raise ValueError("Backup archive missing manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            schema_version = int(manifest.get("schema_version", 0))
            if schema_version > SCHEMA_VERSION:
                raise ValueError("Backup schema version is newer than supported")

            database_source = tmp_path / "database" / "dataminer.db"
            if not database_source.exists():
                raise ValueError("Backup archive missing database snapshot")
            self._projects.database_manager.import_database(database_source)

            derived_source = tmp_path / "derived" / "projects"
            target_root = self._projects.storage_root / "projects"
            if target_root.exists():
                shutil.rmtree(target_root)
            if derived_source.exists():
                shutil.copytree(derived_source, target_root)
            else:
                target_root.mkdir(parents=True, exist_ok=True)

        self._projects.reload()


__all__ = ["BackupService", "MANIFEST_NAME"]

