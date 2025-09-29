from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import ChatMessage, ConversationManager, PlanItem, StepContextBatch


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

    assert len(client.requests) == 3
    assert len(turn.step_results) == 3
    assert [item.status for item in turn.plan] == ["done"] * 3
    assert turn.plan[0].description.startswith(
        "Input: Corpus context → Action: Collect background references on"
    )
    assert "document citations" in turn.plan[-1].description.lower()
    assert turn.answer.startswith("Context & Background:")
    assert len(turn.citations) == 3
    assert turn.citations[0].get("steps") == [1]
    assert "Context & Background" in turn.citations[0].get("tag_names", [])
    assert turn.step_results[0].citation_indexes == [1]
    assert turn.reasoning is not None
    assert turn.reasoning.get("final_sections") == [
        {
            "title": "Context & Background",
            "sentences": ["Finding. [1][2][3]"],
            "citations": [1, 2, 3],
        }
    ]
    assert not turn.reasoning.get("conflicts")


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

    assert len(turn.step_results) == 3
    assert turn.reasoning_artifacts is not None
    assert any("No direct evidence" in text for text in turn.reasoning_artifacts.assumptions)
    assert all(result.citation_indexes for result in turn.step_results)
    assert all(turn.citations)


def test_plan_critic_rejects_meta_steps() -> None:
    client = StubLMStudioClient()
    manager = ConversationManager(client)
    critic = manager._plan_critic

    plan = [
        PlanItem(
            description=(
                "Input: Corpus context → Action: Write final answer → Output: Paragraph response"
            )
        )
    ]

    approved, reasons = critic.review(plan)

    assert not approved
    assert any("banned" in reason.lower() or "unsupported" in reason.lower() for reason in reasons)


def test_generate_plan_produces_atomic_steps() -> None:
    client = StubLMStudioClient()
    manager = ConversationManager(client)

    plan = manager._generate_plan("Explain the derivation of the path integral.")

    assert plan
    for item in plan:
        assert " → " in item.description
        assert "explain" not in item.description.lower()
        assert "write" not in item.description.lower()
