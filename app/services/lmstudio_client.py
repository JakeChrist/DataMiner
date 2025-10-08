"""Client helpers for interacting with a local LMStudio HTTP server."""

from __future__ import annotations

import copy
import json
import logging
import re
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
                    message = self._build_http_error_message(status, body)
                    raise LMStudioResponseError(message)
                return body
            except error.HTTPError as exc:
                body = exc.read() if hasattr(exc, "read") else b""
                message = self._build_http_error_message(getattr(exc, "code", None), body)
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
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - defensive
            raise LMStudioResponseError("Invalid JSON from LMStudio") from exc

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            fallback = self._coalesce_streamed_response(text)
            if fallback is not None:
                return fallback
            raise LMStudioResponseError("Invalid JSON from LMStudio") from None

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
    def _build_http_error_message(
        status: int | None, body: bytes | str | None
    ) -> str:
        summary = LMStudioClient._summarize_error_body(body)
        if status is not None:
            if summary:
                return f"LMStudio returned HTTP {status}: {summary}"
            return f"LMStudio returned HTTP {status}"
        return summary or "LMStudio request failed"

    @staticmethod
    def _summarize_error_body(body: bytes | str | None) -> str:
        if body is None:
            return ""
        if isinstance(body, bytes):
            text = body.decode("utf-8", errors="replace")
        else:
            text = str(body)
        text = text.strip()
        if not text:
            return ""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return LMStudioClient._normalize_error_message(text)
        if isinstance(data, dict):
            buckets: list[dict[str, Any]] = []
            error_payload = data.get("error")
            if isinstance(error_payload, dict):
                buckets.append(error_payload)
            buckets.append(data)
            for bucket in buckets:
                if not isinstance(bucket, dict):
                    continue
                message = bucket.get("message") or bucket.get("detail") or ""
                if not isinstance(message, str):
                    message = str(message)
                hints = [
                    str(bucket[key]).strip()
                    for key in ("hint", "help")
                    if isinstance(bucket.get(key), str) and str(bucket[key]).strip()
                ]
                normalized = LMStudioClient._normalize_error_message(
                    message,
                    code=bucket.get("code"),
                    hints=hints,
                )
                if normalized:
                    return normalized
        return LMStudioClient._normalize_error_message(text)

    @staticmethod
    def _normalize_error_message(
        message: str, *, code: Any = None, hints: Sequence[str] | None = None
    ) -> str:
        clean = " ".join(str(message or "").split())
        hint_text = ""
        if hints:
            parts = [" ".join(str(hint).split()) for hint in hints if str(hint).strip()]
            hint_text = " ".join(parts).strip()
        crash_message = LMStudioClient._humanize_model_crash(clean, code=code)
        if crash_message:
            if hint_text:
                return f"{crash_message} Details: {hint_text}"
            return crash_message
        if hint_text:
            if clean:
                return f"{clean} ({hint_text})"
            return hint_text
        return clean

    @staticmethod
    def _humanize_model_crash(message: str, *, code: Any = None) -> str | None:
        crash_hint = False
        numeric_code = LMStudioClient._coerce_int(code)
        if numeric_code is None:
            numeric_code = LMStudioClient._extract_large_number(message)
        elif numeric_code >= 1 << 31:
            crash_hint = True
        normalized_message = message.strip()
        lower = normalized_message.lower()
        if "model crash" in lower or "model crashed" in lower:
            crash_hint = True
        if isinstance(code, str) and code.lower() == "model_crash":
            crash_hint = True
        if numeric_code is not None and numeric_code >= 1 << 31:
            crash_hint = True
        if not crash_hint:
            return None
        formatted_code = None
        if numeric_code is not None and numeric_code >= 1 << 31:
            formatted_code = LMStudioClient._format_crash_code(numeric_code)
        details = normalized_message
        if formatted_code and numeric_code is not None and details:
            digits = str(numeric_code)
            if digits in details:
                details = details.replace(digits, formatted_code)
        details = details.strip(" .")
        friendly = "LMStudio reported that the model crashed"
        if formatted_code:
            friendly += f" (error {formatted_code})"
        friendly += ". Restart the model in LMStudio and try again."
        if details and details.lower() not in {"model crash", "model crash error"}:
            friendly += f" Details: {details}"
        return friendly

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            try:
                return int(value)
            except (TypeError, ValueError, OverflowError):
                return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return int(stripped)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_large_number(text: str) -> int | None:
        for match in re.finditer(r"\d{9,}", text):
            try:
                value = int(match.group(0))
            except ValueError:
                continue
            if value >= 1 << 31:
                return value
        return None

    @staticmethod
    def _format_crash_code(value: int) -> str:
        if value >= 1 << 63:
            signed = value - (1 << 64)
            hex_value = f"0x{value & ((1 << 64) - 1):016X}"
        else:
            signed = value - (1 << 32)
            hex_value = f"0x{value & ((1 << 32) - 1):08X}"
        return f"{signed} / {hex_value}"

    @staticmethod
    def _coalesce_streamed_response(payload: str) -> dict[str, Any] | None:
        """Merge Server-Sent Event chunks into a standard chat payload."""

        events: list[dict[str, Any]] = []
        for block in payload.split("\n\n"):
            if not block.strip():
                continue
            for line in block.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(":"):
                    continue
                if not stripped.startswith("data:"):
                    continue
                data = stripped[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    events.append(json.loads(data))
                except json.JSONDecodeError:  # pragma: no cover - defensive
                    return None

        if not events:
            return None

        return LMStudioClient._merge_sse_events(events)

    @staticmethod
    def _merge_sse_events(events: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        choices: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] | None = None
        meta: dict[str, Any] = {}

        for event in events:
            if not isinstance(event, dict):
                continue
            event_usage = event.get("usage")
            if isinstance(event_usage, dict):
                usage = event_usage
            for key in ("id", "object", "model", "created"):
                value = event.get(key)
                if value is not None:
                    meta[key] = value

            raw_choices = event.get("choices")
            if not isinstance(raw_choices, Iterable):
                continue
            for choice in raw_choices:
                if not isinstance(choice, dict):
                    continue
                index_raw = choice.get("index", 0)
                try:
                    index = int(index_raw)
                except (TypeError, ValueError):
                    index = 0
                entry = choices.setdefault(
                    index,
                    {"index": index, "message": {"role": "assistant", "content": ""}},
                )
                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    entry["finish_reason"] = finish_reason
                logprobs = choice.get("logprobs")
                if isinstance(logprobs, dict):
                    entry["logprobs"] = copy.deepcopy(logprobs)

                delta = choice.get("delta")
                if isinstance(delta, dict):
                    message = entry.setdefault("message", {"role": "assistant", "content": ""})
                    LMStudioClient._merge_choice_delta(message, delta)

                message_block = choice.get("message")
                if isinstance(message_block, dict):
                    message = entry.setdefault("message", {"role": "assistant", "content": ""})
                    LMStudioClient._deep_merge_dicts(message, message_block)

        if not choices:
            return None

        aggregated: dict[str, Any] = dict(meta)
        aggregated["choices"] = [choices[index] for index in sorted(choices)]
        if usage is not None:
            aggregated["usage"] = usage
        return aggregated

    @staticmethod
    def _merge_choice_delta(message: dict[str, Any], delta: dict[str, Any]) -> None:
        for key, value in delta.items():
            if key == "content":
                if isinstance(value, str):
                    existing = message.get("content", "")
                    if not isinstance(existing, str):
                        existing = LMStudioClient._normalize_message_content(existing)
                    message["content"] = f"{existing}{value}"
                elif isinstance(value, Iterable):
                    normalized = LMStudioClient._normalize_message_content(value)
                    existing = message.get("content", "")
                    if not isinstance(existing, str):
                        existing = LMStudioClient._normalize_message_content(existing)
                    message["content"] = f"{existing}{normalized}"
            elif key == "role":
                if isinstance(value, str):
                    message["role"] = value
            elif isinstance(value, dict):
                target = message.get(key)
                if isinstance(target, dict):
                    LMStudioClient._deep_merge_dicts(target, value)
                else:
                    message[key] = copy.deepcopy(value)
            elif isinstance(value, list):
                target_list = message.get(key)
                if isinstance(target_list, list):
                    target_list.extend(copy.deepcopy(value))
                else:
                    message[key] = copy.deepcopy(value)
            else:
                message[key] = value

    @staticmethod
    def _deep_merge_dicts(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict):
                current = target.get(key)
                if isinstance(current, dict):
                    LMStudioClient._deep_merge_dicts(current, value)
                else:
                    target[key] = copy.deepcopy(value)
            elif isinstance(value, list):
                current_list = target.get(key)
                if isinstance(current_list, list):
                    current_list.extend(copy.deepcopy(value))
                else:
                    target[key] = copy.deepcopy(value)
            elif key == "content" and isinstance(value, str):
                existing = target.get(key)
                if isinstance(existing, str) and existing:
                    target[key] = existing + value
                else:
                    target[key] = value
            else:
                target[key] = copy.deepcopy(value)

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

