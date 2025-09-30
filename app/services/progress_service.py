"""Centralized progress and toast notification helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from ..logging import log_call


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProgressUpdate:
    """Data object describing the state of a progress update."""

    task_id: str
    message: str
    percent: float | None = None
    indeterminate: bool = False


class ProgressService(QObject):
    """Publish background task progress to interested listeners."""

    progress_started = pyqtSignal(object)
    progress_updated = pyqtSignal(object)
    progress_finished = pyqtSignal(object)
    toast_requested = pyqtSignal(str, str, int)

    @log_call(logger=logger)
    def __init__(self) -> None:
        super().__init__()
        self._subscriptions: list[Callable[[ProgressUpdate], None]] = []

    # ------------------------------------------------------------------
    # Subscription helpers
    @log_call(logger=logger)
    def subscribe(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """Subscribe ``callback`` to raw progress events."""

        if callback not in self._subscriptions:
            self._subscriptions.append(callback)

    @log_call(logger=logger)
    def unsubscribe(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """Remove ``callback`` from the subscription list."""

        if callback in self._subscriptions:
            self._subscriptions.remove(callback)

    @log_call(logger=logger)
    def _dispatch(self, update: ProgressUpdate) -> None:
        for callback in list(self._subscriptions):
            callback(update)

    # ------------------------------------------------------------------
    # Emission helpers
    @log_call(logger=logger)
    def start(self, task_id: str, message: str = "") -> None:
        update = ProgressUpdate(task_id=task_id, message=message, percent=0.0)
        logger.info("Progress started", extra={"task_id": task_id, "message": message})
        self.progress_started.emit(update)
        self._dispatch(update)

    @log_call(logger=logger)
    def update(
        self,
        task_id: str,
        *,
        message: str | None = None,
        percent: float | None = None,
        indeterminate: bool | None = None,
    ) -> None:
        update = ProgressUpdate(
            task_id=task_id,
            message=message or "",
            percent=percent,
            indeterminate=bool(indeterminate) if indeterminate is not None else False,
        )
        logger.info(
            "Progress updated",
            extra={
                "task_id": task_id,
                "message": update.message,
                "percent": percent,
                "indeterminate": update.indeterminate,
            },
        )
        self.progress_updated.emit(update)
        self._dispatch(update)

    @log_call(logger=logger)
    def finish(self, task_id: str, message: str = "") -> None:
        update = ProgressUpdate(task_id=task_id, message=message, percent=100.0)
        logger.info(
            "Progress finished",
            extra={"task_id": task_id, "message": message},
        )
        self.progress_finished.emit(update)
        self._dispatch(update)

    @log_call(logger=logger)
    def notify(self, message: str, *, level: str = "info", duration_ms: int = 4000) -> None:
        """Request a toast notification."""

        logger.info(
            "Toast notification requested",
            extra={"message": message, "level": level, "duration_ms": duration_ms},
        )
        self.toast_requested.emit(message, level, duration_ms)

    # ------------------------------------------------------------------
    # Background task integration
    @log_call(logger=logger)
    def subscribe_to(self, emitter: QObject) -> None:
        """Connect a background task emitter with ``started/progress/finished`` signals."""

        if hasattr(emitter, "started"):
            emitter.started.connect(lambda task_id, message="": self.start(task_id, message))  # type: ignore[arg-type]
        if hasattr(emitter, "progress"):
            emitter.progress.connect(  # type: ignore[attr-defined]
                lambda task_id, percent, message="": self.update(
                    task_id, message=message, percent=percent
                )
            )
        if hasattr(emitter, "finished"):
            emitter.finished.connect(lambda task_id, message="": self.finish(task_id, message))  # type: ignore[arg-type]
        logger.debug(
            "Attached progress emitter",
            extra={"emitter": emitter.__class__.__name__},
        )


__all__ = ["ProgressService", "ProgressUpdate"]

