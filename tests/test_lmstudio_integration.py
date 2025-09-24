from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.services import (
    AnswerLength,
    ConversationManager,
    LMStudioClient,
    LMStudioConnectionError,
    LMStudioError,
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
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Echo: {question}",
                    "metadata": {
                        "citations": [{"source": "doc1", "snippet": "alpha"}],
                        "reasoning": {"steps": ["Review context", "Respond"]},
                    },
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


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
    )

    assert turn.answer.startswith("Echo:")
    assert turn.citations
    assert turn.reasoning == {"steps": ["Review context", "Respond"]}

    assert state["requests"] and isinstance(state["requests"], list)
    payload = state["requests"][0]
    assert payload["model"] == "test-model"
    assert payload["max_tokens"] == AnswerLength.BRIEF.to_request_params()["max_tokens"]
    messages = payload["messages"]
    assert isinstance(messages, list)
    # Expect system prompt, plus user/assistant pairs (none yet), and new user message.
    assert messages[0]["role"] == "system"
    assert "Context:" in messages[-1]["content"]


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

