from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from urllib import error, request

from app.services import (
    AnswerLength,
    ConversationManager,
    LMStudioClient,
    LMStudioConnectionError,
    LMStudioError,
    ReasoningVerbosity,
    ResponseMode,
)


def _default_chat_response(payload: dict[str, object]) -> dict[str, object]:
    messages = payload.get("messages")
    if isinstance(messages, Sequence) and messages:
        raw_question = messages[-1]
        if isinstance(raw_question, dict):
            question = str(raw_question.get("content", ""))
        else:
            question = ""
    else:
        question = ""
    reasoning_options = payload.get("reasoning")
    if not isinstance(reasoning_options, dict):
        reasoning_options = {}
    verbosity = str(reasoning_options.get("verbosity", "brief"))
    include_plan = bool(reasoning_options.get("include_plan", True))
    summary_bullets: list[str]
    if verbosity == "minimal":
        summary_bullets = ["Check context"]
    elif verbosity == "extended":
        summary_bullets = [
            "Check context",
            "Draft response",
            "Verify citations",
        ]
    else:
        summary_bullets = ["Check context", "Draft response"]
    plan_items: list[dict[str, str]] = []
    if include_plan:
        base_plan = [
            {"step": "Gather evidence", "status": "complete"},
            {"step": "Compose answer", "status": "pending"},
            {"step": "Review assumptions", "status": "pending"},
        ]
        max_items = reasoning_options.get("max_plan_items")
        if isinstance(max_items, int):
            plan_items = base_plan[: max(0, max_items)]
        else:
            plan_items = base_plan
    reasoning_metadata = {
        "summary_bullets": summary_bullets,
        "plan": plan_items,
        "assumptions": {
            "used": ["Assuming a concise summary is acceptable."],
            "should_ask": False,
            "rationale": "Request already specifies a summary format.",
        },
        "self_check": {"passed": True, "flags": []},
    }
    content = f"Echo: {question}"
    if payload.get("response_mode") == ResponseMode.SOURCES_ONLY.value:
        content = "Facts:\n- detail one (doc1)\n- detail two (doc2)"
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "metadata": {
                        "citations": [{"source": "doc1", "snippet": "alpha"}],
                        "reasoning": reasoning_metadata,
                    },
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def test_lmstudio_client_parses_structured_content() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        "Part one. ",
                        {"type": "text", "text": "Part two."},
                    ],
                    "metadata": {
                        "citations": [{"source": "doc", "snippet": "text"}],
                    },
                }
            }
        ]
    }

    message = LMStudioClient._parse_chat_response(data)

    assert message.content == "Part one. Part two."
    assert message.citations == [{"source": "doc", "snippet": "text"}]


def test_lmstudio_client_handles_message_level_citations() -> None:
    data = {
        "choices": [
            {
                "citations": [{"source": "root"}],
                "message": {
                    "role": "assistant",
                    "content": "Answer",
                    "citations": [{"source": "message"}],
                    "metadata": {},
                },
            }
        ]
    }

    message = LMStudioClient._parse_chat_response(data)

    assert message.citations == [{"source": "message"}]


def test_lmstudio_client_falls_back_to_choice_level_citations() -> None:
    data = {
        "choices": [
            {
                "citations": [{"source": "choice"}],
                "message": {
                    "role": "assistant",
                    "content": "Answer",
                    "metadata": {},
                },
            }
        ]
    }

    message = LMStudioClient._parse_chat_response(data)

    assert message.citations == [{"source": "choice"}]


def _make_handler(state: dict[str, object]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - signature defined by BaseHTTPRequestHandler
            if self.path == "/v1/health":
                status = int(state.get("health_status", 200))
                body = state.get("health_body", {"status": "ok"})
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                payload = json.dumps(body).encode("utf-8")
                self.wfile.write(payload)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:  # noqa: N802 - signature defined by BaseHTTPRequestHandler
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            try:
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except json.JSONDecodeError:
                payload = {}
            requests = state.setdefault("requests", [])
            if isinstance(requests, list):
                requests.append(payload)

            responses = state.setdefault("responses", [])
            current: dict[str, object]
            if isinstance(responses, list) and responses:
                current_raw = responses.pop(0)
                if isinstance(current_raw, dict):
                    current = current_raw
                else:
                    current = {"status": 200, "body": {}}
            else:
                template = state.get("response_template", _default_chat_response)
                if isinstance(template, Callable):
                    body = template(payload)
                else:
                    body = _default_chat_response(payload)
                current = {"status": 200, "body": body}

            status = int(current.get("status", 200))
            body = current.get("body", {})
            if not isinstance(body, (bytes, bytearray)):
                body = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: D401, N802 - disable noisy logs
            """Silence default request logging during tests."""

    return Handler


@pytest.fixture()
def lmstudio_server() -> tuple[dict[str, object], str]:
    state: dict[str, object] = {
        "requests": [],
        "responses": [],
        "health_status": 200,
        "health_body": {"status": "ok"},
    }
    handler = _make_handler(state)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://{server.server_address[0]}:{server.server_address[1]}"
    try:
        yield state, base_url
    finally:
        server.shutdown()
        thread.join()


def test_lmstudio_client_and_conversation_manager_success(lmstudio_server: tuple[dict[str, object], str]) -> None:
    state, base_url = lmstudio_server
    client = LMStudioClient(base_url=base_url, model="test-model", max_retries=2, retry_backoff=0.01)
    manager = ConversationManager(client, system_prompt="You are helpful.", context_window=2)

    events: list[tuple[bool, str | None]] = []
    manager.add_connection_listener(lambda payload: events.append((payload.connected, payload.message)))

    connection_state = manager.check_connection()
    assert connection_state.connected
    if events:
        assert events[-1][0] is True

    turn = manager.ask(
        "What is the summary?",
        context_snippets=["Document A: important details."],
        preset=AnswerLength.BRIEF,
        reasoning_verbosity=ReasoningVerbosity.BRIEF,
    )

    assert turn.answer.startswith("Echo:")
    assert turn.citations
    assert turn.response_mode is ResponseMode.GENERATIVE
    assert turn.reasoning_artifacts is not None
    assert turn.reasoning_bullets
    assert "Check context" in turn.reasoning_bullets[0]
    assert turn.plan
    assert turn.plan[0].is_complete
    assert turn.assumptions
    assert turn.assumption_decision is not None
    assert turn.assumption_decision.mode == "assume"
    assert turn.assumption_decision.rationale
    assert turn.self_check is not None and turn.self_check.passed is True

    assert state["requests"] and isinstance(state["requests"], list)
    payload = state["requests"][0]
    assert payload["model"] == "test-model"
    assert payload["max_tokens"] == AnswerLength.BRIEF.to_request_params()["max_tokens"]
    assert payload["reasoning"]["verbosity"] == ReasoningVerbosity.BRIEF.value
    assert payload["reasoning"]["include_plan"] is True
    messages = payload["messages"]
    assert isinstance(messages, list)
    # Expect system prompt, plus user/assistant pairs (none yet), and new user message.
    assert messages[0]["role"] == "system"
    assert "Context:" in messages[-1]["content"]


def test_conversation_manager_populates_retrieval_query(
    lmstudio_server: tuple[dict[str, object], str]
) -> None:
    state, base_url = lmstudio_server
    client = LMStudioClient(base_url=base_url, retry_backoff=0.01)
    manager = ConversationManager(client, context_window=0)

    extra = {"retrieval": {"include": ["doc-1"]}}
    manager.ask("Gather context please", extra_options=extra)

    assert state["requests"] and isinstance(state["requests"], list)
    payload = state["requests"][-1]
    assert payload.get("retrieval", {}).get("query") == "Gather context please"
    assert payload.get("retrieval", {}).get("include") == ["doc-1"]
    assert "query" not in extra["retrieval"]


def test_lmstudio_client_disables_streaming_by_default(
    lmstudio_server: tuple[dict[str, object], str]
) -> None:
    state, base_url = lmstudio_server
    client = LMStudioClient(base_url=base_url)

    client.chat([{"role": "user", "content": "Hello"}])

    first_request = state["requests"][0]
    assert isinstance(first_request, dict)
    assert first_request.get("stream") is False


def test_lmstudio_client_allows_stream_override(
    lmstudio_server: tuple[dict[str, object], str]
) -> None:
    state, base_url = lmstudio_server
    client = LMStudioClient(base_url=base_url)

    client.chat(
        [{"role": "user", "content": "Hello"}],
        extra_options={"stream": True},
    )

    first_request = state["requests"][0]
    assert isinstance(first_request, dict)
    assert first_request.get("stream") is True


def test_lmstudio_client_does_not_set_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, request.Request] = {}

    class _Response:
        def __init__(self) -> None:
            self._body = json.dumps(_default_chat_response({})).encode("utf-8")

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def getcode(self) -> int:
            return 200

        def read(self) -> bytes:
            return self._body

    def _fake_urlopen(req: request.Request, *args: object, **kwargs: object) -> _Response:
        if args:
            raise AssertionError(f"Unexpected positional args: {args!r}")
        if "timeout" in kwargs:
            raise AssertionError("timeout should not be provided to urlopen")
        captured["request"] = req
        return _Response()

    monkeypatch.setattr(request, "urlopen", _fake_urlopen)

    client = LMStudioClient(max_retries=0)
    client.chat([{"role": "user", "content": "Hi"}], preset=AnswerLength.DETAILED)

    assert "request" in captured


def test_lmstudio_client_reports_timeout_without_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def _raise_timeout(req: request.Request, *args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise error.URLError(socket.timeout("timed out"))

    monkeypatch.setattr(request, "urlopen", _raise_timeout)

    client = LMStudioClient(max_retries=0)

    with pytest.raises(LMStudioConnectionError) as excinfo:
        client.chat([{"role": "user", "content": "Hello"}], preset=AnswerLength.DETAILED)

    assert calls == 1
    assert "timed out" in str(excinfo.value).lower()


def test_conversation_manager_handles_failures_and_recovers(
    lmstudio_server: tuple[dict[str, object], str]
) -> None:
    state, base_url = lmstudio_server
    client = LMStudioClient(base_url=base_url, max_retries=1, retry_backoff=0.01)
    manager = ConversationManager(client, context_window=1)
    events: list[tuple[bool, str | None]] = []
    manager.add_connection_listener(lambda payload: events.append((payload.connected, payload.message)))

    manager.check_connection()
    state["responses"] = [
        {"status": 500, "body": {"error": "temporary"}},
    ]
    turn = manager.ask("First question?")
    assert turn.answer.startswith("Echo:")
    assert len(state["requests"]) == 2

    # Now exhaust retries with persistent errors.
    state["responses"] = [
        {"status": 503, "body": {"error": "down"}},
        {"status": 503, "body": {"error": "still down"}},
    ]
    with pytest.raises(LMStudioError):
        manager.ask("Second question?")
    assert events[-1][0] is False
    assert events[-1][1]
    assert manager.can_ask() is False

    with pytest.raises(LMStudioConnectionError):
        manager.ask("Should not run")

    # Restore health and verify context window behaviour.
    state["health_status"] = 200
    state["responses"] = []
    recovery_state = manager.check_connection()
    assert recovery_state.connected is True
    assert manager.can_ask() is True
    assert events[-1][0] is True

    state["responses"] = []
    state["requests"] = []
    manager.ask("Follow-up?")
    payload = state["requests"][0]
    messages = payload["messages"]
    assert len(messages) == 3  # previous user+assistant + new user
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].startswith("First question?")
    assert messages[1]["role"] == "assistant"


def test_reasoning_verbosity_controls_request_and_artifacts(
    lmstudio_server: tuple[dict[str, object], str]
) -> None:
    state, base_url = lmstudio_server
    client = LMStudioClient(base_url=base_url, retry_backoff=0.01)
    manager = ConversationManager(client, context_window=0)

    manager.ask("Minimal please", reasoning_verbosity=ReasoningVerbosity.MINIMAL)
    assert state["requests"]
    minimal_payload = state["requests"][0]
    assert minimal_payload["reasoning"]["verbosity"] == ReasoningVerbosity.MINIMAL.value
    assert minimal_payload["reasoning"]["include_plan"] is False
    minimal_turn = manager.turns[-1]
    assert minimal_turn.reasoning_bullets
    assert len(minimal_turn.plan) == 0

    manager.ask("Extended please", reasoning_verbosity=ReasoningVerbosity.EXTENDED)
    extended_payload = state["requests"][-1]
    assert extended_payload["reasoning"]["verbosity"] == ReasoningVerbosity.EXTENDED.value
    assert extended_payload["reasoning"]["include_plan"] is True
    extended_turn = manager.turns[-1]
    assert len(extended_turn.plan) >= 2
    assert len(extended_turn.reasoning_bullets) >= 3


def test_sources_only_mode_tracks_clarification_and_self_check(
    lmstudio_server: tuple[dict[str, object], str]
) -> None:
    state, base_url = lmstudio_server

    def _clarifying_response(payload: dict[str, object]) -> dict[str, object]:
        base = _default_chat_response(payload)
        choice = base["choices"][0]
        message = choice["message"]
        metadata = message["metadata"]
        reasoning = metadata["reasoning"]
        reasoning["assumptions"] = {
            "used": [],
            "should_ask": True,
            "clarifying_question": "Which quarter should I analyse?",
            "rationale": "Ambiguous timeframe for revenue.",
        }
        reasoning["self_check"] = {
            "passed": False,
            "flags": ["Needs timeframe clarification"],
            "notes": "Unable to validate without timeframe.",
        }
        if payload.get("response_mode") == ResponseMode.SOURCES_ONLY.value:
            message["content"] = "Sources only: doc1, doc2"
        return base

    state["response_template"] = _clarifying_response

    client = LMStudioClient(base_url=base_url, retry_backoff=0.01)
    manager = ConversationManager(client)

    turn = manager.ask(
        "Summarise recent revenue performance",
        reasoning_verbosity=ReasoningVerbosity.BRIEF,
        response_mode=ResponseMode.SOURCES_ONLY,
    )

    payload = state["requests"][-1]
    assert payload["response_mode"] == ResponseMode.SOURCES_ONLY.value
    assert turn.response_mode is ResponseMode.SOURCES_ONLY
    assert turn.assumption_decision is not None
    assert turn.assumption_decision.mode == "clarify"
    assert turn.assumption_decision.clarifying_question
    assert not turn.assumptions
    assert turn.self_check is not None
    assert turn.self_check.passed is False
    assert any("timeframe" in flag.lower() for flag in turn.self_check.flags)
