"""Logging utilities for the DataMiner application."""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .config import get_user_config_dir

LOG_FILENAME = "dataminer.log"
MAX_BYTES = 1_048_576  # 1 MiB
BACKUP_COUNT = 3

_EXCEPTION_HOOK_INSTALLED = False
_HOOK_LOCK = threading.Lock()


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

    setattr(logger, "log_path", log_path)
    logger.debug("Logging initialised. Writing to %s", log_path)
    return logger


def get_log_file_path(logger: logging.Logger) -> Optional[Path]:
    """Return the path of the first file handler attached to ``logger``."""

    log_path = getattr(logger, "log_path", None)
    if isinstance(log_path, Path):
        return log_path

    for handler in logger.handlers:
        filename = getattr(handler, "baseFilename", None)
        if filename:
            return Path(filename)
    return None


def install_exception_hook(logger: logging.Logger) -> None:
    """Install handlers that log and surface unhandled exceptions."""

    global _EXCEPTION_HOOK_INSTALLED
    with _HOOK_LOCK:
        if _EXCEPTION_HOOK_INSTALLED:
            return
        _EXCEPTION_HOOK_INSTALLED = True

    log_path = get_log_file_path(logger)
    default_hook = sys.excepthook

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            default_hook(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback)
        )
        formatted = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        _show_log_dialog(logger, formatted, log_path)
        default_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception

    thread_hook = getattr(threading, "excepthook", None)
    if thread_hook is not None:

        def handle_thread_exception(args):
            if issubclass(args.exc_type, KeyboardInterrupt):
                thread_hook(args)
                return

            logger.critical(
                "Unhandled exception in thread %s",
                args.thread.name,
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            formatted_thread = "".join(
                traceback.format_exception(
                    args.exc_type, args.exc_value, args.exc_traceback
                )
            )
            _show_log_dialog(logger, formatted_thread, log_path)
            thread_hook(args)

        threading.excepthook = handle_thread_exception


def _show_log_dialog(
    logger: logging.Logger, traceback_text: str, log_path: Optional[Path]
) -> None:
    """Display a dialog containing the traceback and recent log output."""

    try:
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Unable to import PyQt6 components for crash dialog: %s", exc)
        return

    is_main_thread = threading.current_thread() is threading.main_thread()
    app = QApplication.instance()
    created_app = False
    if app is None:
        if not is_main_thread:
            logger.error(
                "Cannot display crash dialog on non-main thread without QApplication"
            )
            return
        try:
            app = QApplication(sys.argv or ["DataMiner"])
            created_app = True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to create QApplication for crash dialog: %s", exc)
            return

    def _exec_dialog() -> None:
        try:
            from .ui.log_viewer_dialog import LogViewerDialog

            dialog = LogViewerDialog(
                log_path=log_path,
                traceback_text=traceback_text,
                parent=app.activeWindow(),
            )
            dialog.exec()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to display crash dialog: %s", exc)
        finally:
            if created_app:
                app.quit()

    if is_main_thread:
        _exec_dialog()
    else:
        QTimer.singleShot(0, _exec_dialog)


__all__ = ["setup_logging", "install_exception_hook", "get_log_file_path"]
