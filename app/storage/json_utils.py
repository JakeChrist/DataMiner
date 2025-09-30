"""Utilities for normalising metadata into JSON-serialisable structures."""

from __future__ import annotations

import base64
import math
from collections.abc import Mapping
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any


def make_json_safe(value: Any) -> Any:
    """Return ``value`` converted into a JSON-serialisable structure.

    The ingestion pipeline collects metadata from a variety of sources. Some
    parsers (and third-party extensions) may yield values such as ``datetime``
    objects, ``Path`` instances, ``set`` collections, or even raw ``bytes``.
    These values cause ``json.dumps`` to raise ``TypeError``, which previously
    prevented documents from being stored or synchronised.  By coercing every
    value into a primitive JSON-friendly representation we guarantee that
    ingestion succeeds regardless of parser metadata shape.
    """

    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return make_json_safe(value.value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, Decimal):
        if value.is_nan() or value.is_infinite():
            return None
        return float(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return "base64:" + base64.b64encode(value).decode("ascii")
    if isinstance(value, Mapping):
        return {str(key): make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, set):
        try:
            iterable = sorted(value)
        except TypeError:
            iterable = list(value)
        return [make_json_safe(item) for item in iterable]
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return make_json_safe(value.to_dict())
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return make_json_safe({key: getattr(value, key) for key in vars(value)})
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


__all__ = ["make_json_safe"]
