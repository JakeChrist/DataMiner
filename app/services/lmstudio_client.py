"""Client helpers for interacting with a local LMStudio HTTP server."""

from __future__ import annotations

import json
import logging
import socket
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib import error, request

from ..logging import log_call


logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "http://127.0.0.1:1234"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
HEALTH_PATH = "/v1/health"


class LMStudioError(RuntimeError):
    """Base exception for LMStudio client failures."""


class LMStudioConnectionError(LMStudioError):
    """Raised when the LMStudio server cannot be reached."""


class LMStudioResponseError(LMStudioError):
    """Raised when the LMStudio server returns an invalid response."""


@dataclass(frozen=True)
class ChatMessage:
    """Structured data for a chat completion response."""

    content: str
    citations: list[Any]
    reasoning: dict[str, Any] | None
    raw_response: dict[str, Any]


class AnswerLength(Enum):
    """Presets that control completion verbosity."""

    BRIEF = "brief"
    NORMAL = "normal"
    DETAILED = "detailed"

    def to_request_params(self) -> dict[str, Any]:
        """Return request parameter overrides for the preset."""

        if self is AnswerLength.BRIEF:
            return {"max_tokens": 256, "temperature": 0.2}
        if self is AnswerLength.DETAILED:
            return {"max_tokens": 1024, "temperature": 0.4}
        return {"max_tokens": 512, "temperature": 0.3}


class LMStudioClient:
    """Simple HTTP client for LMStudio's OpenAI-compatible API."""

    @log_call(logger=logger)
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "lmstudio",
        max_retries: int = 2,
        retry_backoff: float = 0.5,
    ) -> None:
        self._base_url = base_url.rstrip("/") or DEFAULT_BASE_URL
        self._model = model
        self.max_retries = max(max_retries, 0)
        self.retry_backoff = retry_backoff

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    @log_call(logger=logger)
    def configure(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Update connection settings without recreating the client."""

        if base_url is not None:
            normalized = base_url.rstrip("/")
            self._base_url = normalized or DEFAULT_BASE_URL
        if model is not None:
            self._model = model

    @log_call(logger=logger, include_result=True)
    def health_check(self) -> bool:
        """Return ``True`` if the LMStudio server responds to a health probe."""

        try:
            self._request("GET", HEALTH_PATH)
        except LMStudioError:
            return False
        return True

    @log_call(logger=logger, include_result=True)
    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        preset: AnswerLength = AnswerLength.NORMAL,
        extra_options: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Send chat ``messages`` to LMStudio and return the first completion."""

        payload = {
            "model": self._model,
            "messages": list(messages),
        }
        payload.update(preset.to_request_params())
        payload.setdefault("stream", False)
        if extra_options:
            payload.update(extra_options)
        logger.info(
            "Dispatching LMStudio chat request",
            extra={
                "message_count": len(payload["messages"]),
                "preset": preset.value,
                "base_url": self._base_url,
            },
        )
        data = self._request_json("POST", CHAT_COMPLETIONS_PATH, payload)
        return self._parse_chat_response(data)

    @log_call(logger=logger, include_result=True)
    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> bytes:
        url = f"{self._base_url}{path}"
        data: bytes | None = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request_obj = request.Request(url, data=data, headers=headers, method=method)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    "LMStudio request attempt",
                    extra={
                        "method": method,
                        "url": url,
                        "attempt": attempt + 1,
                    },
                )
                with request.urlopen(request_obj) as response:
                    status = response.getcode()
                    body = response.read()
                if status >= 400:
                    message = body.decode("utf-8", errors="replace") if body else ""
                    raise LMStudioResponseError(
                        f"LMStudio returned HTTP {status}: {message.strip()}"
                    )
                return body
            except error.HTTPError as exc:
                body = exc.read() if hasattr(exc, "read") else b""
                message = body.decode("utf-8", errors="replace") if body else str(exc)
                last_error = LMStudioResponseError(message)
                if not self._should_retry(exc.code):
                    break
            except error.URLError as exc:
                if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                    last_error = LMStudioConnectionError("LMStudio request timed out")
                else:
                    last_error = LMStudioConnectionError(str(exc.reason))
            except TimeoutError:
                last_error = LMStudioConnectionError("LMStudio request timed out")
            if attempt < self.max_retries:
                logger.warning(
                    "LMStudio request failed, retrying",
                    extra={
                        "method": method,
                        "url": url,
                        "attempt": attempt + 1,
                        "error": str(last_error) if last_error else None,
                    },
                )
                time.sleep(self.retry_backoff * (2**attempt))
        if last_error is None:
            logger.debug(
                "LMStudio request completed",
                extra={"method": method, "url": url},
            )
        if isinstance(last_error, LMStudioError):
            raise last_error
        if last_error is not None:
            raise LMStudioError(str(last_error))
        raise LMStudioError("Unexpected LMStudio request failure")

    @log_call(logger=logger, include_result=True)
    def _request_json(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = self._request(method, path, payload)
        if not body:
            raise LMStudioResponseError("Empty response from LMStudio")
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise LMStudioResponseError("Invalid JSON from LMStudio") from exc

    @staticmethod
    @log_call(logger=logger, include_result=True)
    def _should_retry(status: int | None) -> bool:
        if status is None:
            return True
        if status in {408, 409, 429, 500, 502, 503, 504}:
            return True
        return False

    @staticmethod
    @log_call(logger=logger, include_result=True)
    def _parse_chat_response(data: dict[str, Any]) -> ChatMessage:
        choices = data.get("choices")
        if not isinstance(choices, Iterable):
            raise LMStudioResponseError("LMStudio response missing choices")
        first = next(iter(choices), None)
        if not isinstance(first, dict):
            raise LMStudioResponseError("LMStudio response missing first choice")
        message = first.get("message")
        if not isinstance(message, dict):
            raise LMStudioResponseError("LMStudio response missing message")
        content_raw = message.get("content")
        content = LMStudioClient._normalize_message_content(content_raw)
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        citations = LMStudioClient._extract_citations(first, message, metadata)
        reasoning = metadata.get("reasoning") if isinstance(metadata.get("reasoning"), dict) else None
        return ChatMessage(
            content=content,
            citations=citations,
            reasoning=reasoning,
            raw_response=data,
        )

    @staticmethod
    def _normalize_message_content(content: Any) -> str:
        """Return a usable string from ``content`` or raise an error."""

        if isinstance(content, str):
            return content
        if isinstance(content, Iterable):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "".join(parts)
        raise LMStudioResponseError("LMStudio message missing content")

    @staticmethod
    def _extract_citations(
        choice: dict[str, Any],
        message: dict[str, Any],
        metadata: dict[str, Any],
    ) -> list[Any]:
        """Return citation payloads from a variety of LMStudio response shapes."""

        def _coerce_list(value: Any, *, seen: set[int] | None = None) -> list[Any] | None:
            if seen is None:
                seen = set()
            if isinstance(value, list):
                return value
            if not isinstance(value, dict):
                return None
            identity = id(value)
            if identity in seen:
                return None
            seen.add(identity)

            candidate_keys = (
                "citations",
                "items",
                "entries",
                "evidence",
                "records",
                "sources",
                "list",
                "results",
                "data",
            )
            for key in candidate_keys:
                if key not in value:
                    continue
                nested = value.get(key)
                result = _coerce_list(nested, seen=seen)
                if result:
                    return result

            for key, nested in value.items():
                if key in {"scope", "include", "exclude", "filters"}:
                    continue
                if isinstance(nested, dict):
                    result = _coerce_list(nested, seen=seen)
                    if result:
                        return result
            return None

        candidates: list[Any] = [
            metadata.get("citations"),
            message.get("citations"),
            choice.get("citations"),
        ]

        context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
        if context:
            candidates.append(context.get("citations"))

        for candidate in candidates:
            resolved = _coerce_list(candidate)
            if resolved:
                return list(resolved)
        return []


__all__ = [
    "AnswerLength",
    "ChatMessage",
    "LMStudioClient",
    "LMStudioConnectionError",
    "LMStudioError",
    "LMStudioResponseError",
]

