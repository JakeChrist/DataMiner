from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import ChatMessage, ConversationManager, StepContextBatch


class StubLMStudioClient:
    """Deterministic client capturing chat invocations for tests."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def health_check(self) -> bool:
        return True

    def chat(self, messages, *, preset, extra_options=None) -> ChatMessage:  # type: ignore[override]
        index = len(self.requests) + 1
        self.requests.append({"messages": list(messages), "options": extra_options})
        content = f"Step {index} finding"
        citations = [
            {
                "id": f"doc-{index}",
                "source": f"Doc {index}",
                "snippet": f"Detail {index}",
            }
        ]
        return ChatMessage(
            content=content,
            citations=citations,
            reasoning=None,
            raw_response={"choices": []},
        )


def _provider_for_steps(
    texts: list[str],
) -> Iterable[StepContextBatch]:
    batches: list[StepContextBatch] = []
    for text in texts:
        batches.append(
            StepContextBatch(
                snippets=[text],
                documents=[{"id": text.lower().replace(" ", "-"), "source": text, "text": text}],
            )
        )
    return batches


def test_conversation_manager_executes_dynamic_plan() -> None:
    client = StubLMStudioClient()
    manager = ConversationManager(client)

    def provider(_item, step_index: int, _total: int) -> Iterable[StepContextBatch]:
        return _provider_for_steps([f"Evidence {step_index}"])

    turn = manager.ask(
        "Compare dataset A. Summarize insights.",
        context_provider=provider,
    )

    assert len(client.requests) == 2
    assert len(turn.step_results) == 2
    assert all(item.status == "done" for item in turn.plan)
    assert "Step 1 finding" in turn.answer
    assert "[1]" in turn.answer
    assert len(turn.citations) == 2
    assert turn.citations[0].get("steps") == [1]
    assert turn.step_results[0].citation_indexes == [1]


def test_dynamic_plan_notes_missing_citations() -> None:
    class EmptyCitationClient(StubLMStudioClient):
        def chat(self, messages, *, preset, extra_options=None) -> ChatMessage:  # type: ignore[override]
            self.requests.append({"messages": list(messages), "options": extra_options})
            return ChatMessage(
                content="",
                citations=[],
                reasoning=None,
                raw_response={"choices": []},
            )

    client = EmptyCitationClient()
    manager = ConversationManager(client)

    def provider(_item, step_index: int, _total: int) -> Iterable[StepContextBatch]:
        return _provider_for_steps([f"Context {step_index}"])

    turn = manager.ask(
        "Outline requirements. Provide recommendation.",
        context_provider=provider,
    )

    assert len(turn.step_results) == 2
    assert turn.reasoning_artifacts is not None
    assert any("No citations" in text for text in turn.reasoning_artifacts.assumptions)
    assert all(result.citation_indexes == [] for result in turn.step_results)
