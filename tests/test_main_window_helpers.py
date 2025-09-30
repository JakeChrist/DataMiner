"""Tests for helper utilities in the main window module."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.storage import DatabaseError

pytest.importorskip("PyQt6")

try:
    from app.ui.main_window import _attempt_document_repository_operation
except ImportError as exc:  # pragma: no cover - environment without GUI deps
    pytest.skip(
        f"PyQt6 dependencies unavailable: {exc}",
        allow_module_level=True,
    )


def test_attempt_document_repository_operation_success_no_lock() -> None:
    result, locked = _attempt_document_repository_operation(lambda: "ok", attempts=1, delay=0)
    assert result == "ok"
    assert locked is False


def test_attempt_document_repository_operation_retries_on_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    def operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return "complete"

    result, locked = _attempt_document_repository_operation(operation, attempts=5, delay=0)
    assert result == "complete"
    assert locked is True
    assert attempts["count"] == 3


def test_attempt_document_repository_operation_handles_non_lock_error() -> None:
    def operation() -> None:
        raise sqlite3.OperationalError("syntax error")

    result, locked = _attempt_document_repository_operation(operation, attempts=2, delay=0)
    assert result is None
    assert locked is True


def test_attempt_document_repository_operation_handles_database_error() -> None:
    def operation() -> None:
        raise DatabaseError("boom")

    result, locked = _attempt_document_repository_operation(operation, attempts=2, delay=0)
    assert result is None
    assert locked is True
