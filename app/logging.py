"""Logging utilities for the DataMiner application."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .config import get_user_config_dir

LOG_FILENAME = "dataminer.log"
MAX_BYTES = 1_048_576  # 1 MiB
BACKUP_COUNT = 3


def setup_logging(
    app_name: str = "DataMiner",
    *,
    level: int = logging.INFO,
    log_filename: Optional[str] = None,
) -> logging.Logger:
    """Configure logging for the application and return the root logger.

    Logging output is directed to both the console and a rotating file located
    in the user's configuration directory. This utility avoids any external
    telemetry by keeping all logs on the local filesystem.
    """
    logger = logging.getLogger(app_name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config_dir = get_user_config_dir(app_name)
    log_path = Path(config_dir) / (log_filename or LOG_FILENAME)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.debug("Logging initialised. Writing to %s", log_path)
    return logger


__all__ = ["setup_logging"]
