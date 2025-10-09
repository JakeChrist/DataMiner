from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import ChatMessage, ConversationManager, PlanItem, StepContextBatch
from app.services.conversation_manager import CONSOLIDATION_SYSTEM_PROMPT


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
        "Input: Corpus context (retrieved corpus snippets with chunk IDs) → Action: Collect background references on"
    )
    assert "document citations" in turn.plan[-1].description.lower()
    assert all(item.rationale for item in turn.plan)
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
    assert any(
        "INSUFFICIENT_EVIDENCE" in text for text in turn.reasoning_artifacts.assumptions
    )
    assert any(result.insufficient for result in turn.step_results)
    assert all(
        result.citation_indexes
        for result in turn.step_results
        if not result.insufficient
    )
    assert all(turn.citations)


def test_single_shot_context_enforces_grounding() -> None:
    client = StubLMStudioClient()
    manager = ConversationManager(client)

    manager.ask(
        "Summarize the evidence.",
        context_snippets=["Document A: Evidence snippet."],
    )

    assert len(client.requests) == 1
    user_message = client.requests[0]["messages"][-1]["content"]
    assert "Context:" in user_message
    assert "Document A: Evidence snippet." in user_message
    assert "Use only the provided corpus context snippets" in user_message


def test_dynamic_plan_passes_context_and_retrieval_metadata() -> None:
    client = StubLMStudioClient()
    manager = ConversationManager(client)

    shared_context = ["Shared corpus note"]

    def provider(_item, step_index: int, _total: int) -> Iterable[StepContextBatch]:
        return _provider_for_steps([f"Step {step_index} snippet"])

    manager.ask(
        "Compare dataset A. Summarize insights.",
        context_snippets=shared_context,
        context_provider=provider,
    )

    first_request = client.requests[0]
    user_message = first_request["messages"][-1]["content"]
    assert "Context:" in user_message
    assert shared_context[0] in user_message
    assert "Step 1 snippet" in user_message
    assert "Use only the provided corpus context snippets" in user_message

    retrieval = first_request.get("options", {}).get("retrieval", {})
    documents = retrieval.get("documents", [])
    assert documents and documents[0]["text"] == "Step 1 snippet"

    final_request = client.requests[-1]
    system_prompts = [msg["content"] for msg in final_request["messages"] if msg["role"] == "system"]
    headline = CONSOLIDATION_SYSTEM_PROMPT.strip().splitlines()[0]
    assert any(headline in prompt for prompt in system_prompts)


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
    assert any(reason.startswith("META:") for reason in reasons)


def test_plan_critic_flags_non_atomic_steps() -> None:
    client = StubLMStudioClient()
    manager = ConversationManager(client)
    critic = manager._plan_critic

    plan = [
        PlanItem(
            description=(
                "Input: Corpus context → Action: Collect supplier list and metrics → Output: Bullet list of suppliers with document citations"
            )
        ),
        PlanItem(
            description=(
                "Input: Step 1 output → Action: Compare supplier performance → Output: Comparison table for supplier performance with cited source snippets"
            )
        ),
    ]

    approved, reasons = critic.review(plan)

    assert not approved
    assert any(reason.startswith("NON_ATOMIC:") for reason in reasons)


def test_plan_critic_catches_order_errors() -> None:
    client = StubLMStudioClient()
    manager = ConversationManager(client)
    critic = manager._plan_critic

    plan = [
        PlanItem(
            description=(
                "Input: Step 2 output → Action: Analyze findings → Output: Bullet list of findings with document citations"
            )
        ),
        PlanItem(
            description=(
                "Input: Corpus context → Action: Collect case studies → Output: Bullet list of case studies with document citations"
            )
        ),
    ]

    approved, reasons = critic.review(plan)

    assert not approved
    assert any(reason.startswith("ORDER_ERROR:") for reason in reasons)


def test_generate_plan_produces_atomic_steps() -> None:
    client = StubLMStudioClient()
    manager = ConversationManager(client)

    plan = manager._generate_plan("Explain the derivation of the path integral.")

    assert plan
    for item in plan:
        assert " → " in item.description
        assert "explain" not in item.description.lower()
        assert "write" not in item.description.lower()
        assert item.rationale
