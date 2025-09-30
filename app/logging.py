"""Logging utilities for the DataMiner application."""

from __future__ import annotations

import functools
import inspect
import logging
import os
import platform
import sys
import threading
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

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
    logger.info(
        "Logging initialised",
        extra={
            "log_path": str(log_path),
            "level": logging.getLevelName(level),
        },
    )
    logger.debug(
        "Runtime environment",
        extra={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": os.getcwd(),
        },
    )
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


def _safe_repr(value: Any, *, max_length: int = 2000) -> str:
    """Return a truncated ``repr`` suitable for logging."""

    try:
        result = repr(value)
    except Exception:
        result = object.__repr__(value)
    if len(result) > max_length:
        return result[: max_length - 1] + "…"
    return result


def _format_arguments(signature: inspect.Signature, *args: Any, **kwargs: Any) -> str:
    try:
        bound = signature.bind_partial(*args, **kwargs)
    except Exception:
        return "unavailable"
    arguments = []
    for name, value in bound.arguments.items():
        if name in {"self", "cls"}:
            continue
        arguments.append(f"{name}={_safe_repr(value)}")
    return ", ".join(arguments)


def _resolve_logger(target: logging.Logger | str | None, module: str) -> logging.Logger:
    if isinstance(target, logging.Logger):
        return target
    if isinstance(target, str):
        return logging.getLogger(target)
    return logging.getLogger(module)


def log_call(
    _func: Optional[Any] = None,
    *,
    logger: logging.Logger | str | None = None,
    level: int = logging.DEBUG,
    include_args: bool = True,
    include_result: bool = False,
    exc_level: int = logging.ERROR,
) -> Any:
    """Decorator that logs entry, exit, and failures for ``_func``.

    Parameters mirror :mod:`logging` with sensible defaults and can be used both
    with and without arguments::

        @log_call
        def some_function(...):
            ...

        @log_call(level=logging.INFO, include_result=True)
        def another(...):
            ...
    """

    def decorator(func: Any) -> Any:
        signature = inspect.signature(func)
        qualname = getattr(func, "__qualname__", getattr(func, "__name__", "<call>"))
        module = getattr(func, "__module__", "")
        log_identifier = f"{module}.{qualname}" if module else qualname

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            resolved_logger = _resolve_logger(logger, module)
            if include_args:
                arguments = _format_arguments(signature, *args, **kwargs)
                resolved_logger.log(level, "Calling %s(%s)", log_identifier, arguments)
            else:
                resolved_logger.log(level, "Calling %s", log_identifier)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception:
                elapsed = time.perf_counter() - start
                resolved_logger.log(
                    exc_level,
                    "Error in %s after %.3fs",
                    log_identifier,
                    elapsed,
                    exc_info=True,
                )
                raise
            elapsed = time.perf_counter() - start
            if include_result:
                resolved_logger.log(
                    level,
                    "%s returned %s (%.3fs)",
                    log_identifier,
                    _safe_repr(result),
                    elapsed,
                )
            else:
                resolved_logger.log(
                    level,
                    "%s completed in %.3fs",
                    log_identifier,
                    elapsed,
                )
            return result

        return wrapper

    if callable(_func):
        return decorator(_func)
    return decorator


__all__ = [
    "setup_logging",
    "install_exception_hook",
    "get_log_file_path",
    "log_call",
]
