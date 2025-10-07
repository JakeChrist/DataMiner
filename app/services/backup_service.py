"""Backup and restore helpers for DataMiner storage."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

from .project_service import ProjectService
from ..storage.database import SCHEMA_VERSION
from ..logging import log_call


MANIFEST_NAME = "manifest.json"


logger = logging.getLogger(__name__)


class BackupService:
    """Create and restore archive snapshots of application data."""

    @log_call(logger=logger)
    def __init__(self, project_service: ProjectService) -> None:
        self._projects = project_service

    @log_call(logger=logger, include_result=True)
    def create_backup(self, destination: str | Path) -> Path:
        destination_path = Path(destination)

        if destination_path.exists() and destination_path.is_dir():
            target_directory = destination_path
        elif not destination_path.exists() and not destination_path.suffix:
            destination_path.mkdir(parents=True, exist_ok=True)
            target_directory = destination_path
        else:
            target_directory = None

        if target_directory is not None:
            timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M%S")
            destination_path = target_directory / f"dataminer-backup-{timestamp}.zip"
        if not destination_path.parent.exists():
            destination_path.parent.mkdir(parents=True, exist_ok=True)

        storage_locations: dict[str, str] = {}
        for record in self._projects.list_projects():
            location = self._projects.get_project_storage_location(record.id)
            if location is None:
                continue
            storage_locations[str(record.id)] = str(location)

        manifest = {
            "created": _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z"),
            "schema_version": SCHEMA_VERSION,
            "active_project": self._projects.active_project_id,
            "storage_locations": storage_locations,
        }

        logger.info(
            "Creating backup",
            extra={
                "destination": str(destination_path),
                "project_count": len(storage_locations),
                "active_project": manifest["active_project"],
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_export = tmp_path / "database.db"
            self._projects.database_manager.export_database(db_export)
            manifest_path = tmp_path / MANIFEST_NAME
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            with zipfile.ZipFile(destination_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(db_export, "database/dataminer.db")
                for project_id, path_str in storage_locations.items():
                    project_path = Path(path_str)
                    if not project_path.exists():
                        continue
                    for path in project_path.rglob("*"):
                        if path.is_file():
                            arcname = Path("derived") / project_id / path.relative_to(project_path)
                            archive.write(path, arcname)
                archive.write(manifest_path, MANIFEST_NAME)
        logger.info(
            "Backup created",
            extra={"destination": str(destination_path), "bytes": destination_path.stat().st_size},
        )
        return destination_path

    @log_call(logger=logger)
    def restore_backup(self, archive_path: str | Path) -> None:
        archive = Path(archive_path)
        if not archive.exists():
            raise FileNotFoundError(archive)

        logger.info("Restoring backup", extra={"archive": str(archive)})

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

            storage_locations = manifest.get("storage_locations")
            derived_root = tmp_path / "derived"
            if isinstance(storage_locations, dict) and storage_locations:
                for project_id, path_str in storage_locations.items():
                    target = Path(path_str)
                    source = derived_root / str(project_id)
                    if target.exists():
                        shutil.rmtree(target, ignore_errors=True)
                    if source.exists():
                        shutil.copytree(source, target)
                    else:
                        target.mkdir(parents=True, exist_ok=True)
                    try:
                        numeric_id = int(project_id)
                    except (TypeError, ValueError):
                        continue
                    self._projects.set_project_storage_location(numeric_id, target)
            else:
                derived_source = derived_root / "projects"
                target_root = self._projects.storage_root / "projects"
                if target_root.exists():
                    shutil.rmtree(target_root)
                if derived_source.exists():
                    shutil.copytree(derived_source, target_root)
                else:
                    target_root.mkdir(parents=True, exist_ok=True)

        self._projects.reload()
        logger.info("Backup restore complete")


__all__ = ["BackupService", "MANIFEST_NAME"]

