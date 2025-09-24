"""Application configuration management."""

from __future__ import annotations

import json
import os
import sys
from configparser import ConfigParser
from pathlib import Path
from typing import Any, MutableMapping

CONFIG_DIR_NAME = "DataMiner"
DEFAULT_JSON_FILENAME = "settings.json"
DEFAULT_INI_FILENAME = "settings.ini"


def get_user_config_dir(app_name: str = CONFIG_DIR_NAME) -> Path:
    """Return the configuration directory for the current user.

    The directory is created on first use. On Windows the directory is
    under ``%APPDATA%``; otherwise the XDG base directory or ``~/.config``
    is used.
    """
    if sys.platform.startswith("win"):
        base_dir = Path(os.getenv("APPDATA", Path.home()))
    else:
        base_dir = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    config_dir = base_dir / app_name
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


class ConfigManager:
    """Handle loading and saving user configuration settings."""

    def __init__(
        self,
        app_name: str = CONFIG_DIR_NAME,
        *,
        format: str = "json",
        filename: str | None = None,
    ) -> None:
        self.app_name = app_name
        self.format = format.lower()
        if self.format not in {"json", "ini"}:
            raise ValueError("format must be either 'json' or 'ini'")
        if filename is None:
            filename = (
                DEFAULT_JSON_FILENAME if self.format == "json" else DEFAULT_INI_FILENAME
            )
        self.config_dir = get_user_config_dir(app_name)
        self.config_path = self.config_dir / filename

    def load(self) -> dict[str, Any]:
        """Load configuration from disk.

        Returns an empty dictionary if the configuration file is absent.
        """
        if not self.config_path.exists():
            return {}

        if self.format == "json":
            with self.config_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        parser = ConfigParser()
        parser.read(self.config_path, encoding="utf-8")
        return {section: dict(parser.items(section)) for section in parser.sections()}

    def save(self, data: MutableMapping[str, Any]) -> None:
        """Persist configuration data to disk."""
        if self.format == "json":
            with self.config_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
                fh.write("\n")
            return

        parser = ConfigParser()
        for section, values in data.items():
            if not isinstance(values, MutableMapping):
                raise ValueError("INI configuration requires mapping values per section")
            parser[section] = {str(key): str(value) for key, value in values.items()}
        with self.config_path.open("w", encoding="utf-8") as fh:
            parser.write(fh)

    def update(self, data: MutableMapping[str, Any]) -> dict[str, Any]:
        """Update the stored configuration with ``data`` and return the result."""
        current = self.load()
        if self.format == "json":
            current.update(data)
        else:
            for section, values in data.items():
                if not isinstance(values, MutableMapping):
                    raise ValueError(
                        "INI configuration updates require mapping values per section"
                    )
                section_data = current.setdefault(section, {})
                if not isinstance(section_data, dict):
                    raise ValueError(
                        "Existing INI section must be a mapping to apply updates"
                    )
                section_data.update(values)
        self.save(current)
        return current

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"ConfigManager(app_name={self.app_name!r}, format={self.format!r}, path={self.config_path!s})"


__all__ = ["ConfigManager", "get_user_config_dir"]
