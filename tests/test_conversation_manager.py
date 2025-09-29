import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import ChatMessage
from app.services.conversation_manager import (
    AnswerLength,
    ConversationManager,
    ConsolidatedSection,
    ConflictNote,
    ConsolidationOutput,
    ResponseMode,
    StepResult,
    _EvidenceLedger,
    _AdversarialJudge,
)


def test_compose_final_answer_deduplicates_and_merges_citations():
    step_results = [
        StepResult(
            index=1,
            description="Alpha",
            answer="Insight A. [1]",
            citation_indexes=[1],
        ),
        StepResult(
            index=2,
            description="Alpha duplicate",
            answer="  insight A.  [2]",
            citation_indexes=[2],
        ),
        StepResult(
            index=3,
            description="Beta",
            answer="Follow up detail. [3]",
            citation_indexes=[3],
        ),
        StepResult(
            index=4,
            description="Fallback text.",
            answer="",
            citation_indexes=[4],
        ),
        StepResult(
            index=5,
            description="Tip one",
            answer="Quick tip",
            citation_indexes=[5],
        ),
        StepResult(
            index=6,
            description="Tip two",
            answer="Another tip",
            citation_indexes=[6],
        ),
    ]

    final_answer = ConversationManager._compose_final_answer(step_results)

    assert (
        final_answer.text
        == "Evidence & Findings: Insight A. [1][2] Follow up detail. [3]\n\n"
        "Key Points: Fallback text. [4]\n\n"
        "Practical Guidance: Quick tip. [5] Another tip. [6]"
    )
    assert [section.title for section in final_answer.sections] == [
        "Evidence & Findings",
        "Key Points",
        "Practical Guidance",
    ]
    assert final_answer.sections[0].sentences == [
        "Insight A. [1][2]",
        "Follow up detail. [3]",
    ]
    assert final_answer.sections[1].sentences == ["Fallback text. [4]"]
    assert final_answer.sections[2].sentences == [
        "Quick tip. [5]",
        "Another tip. [6]",
    ]
    assert final_answer.conflicts == []
    assert final_answer.section_usage[1] == {"Evidence & Findings"}
    assert final_answer.section_usage[4] == {"Key Points"}
    assert final_answer.section_usage[5] == {"Practical Guidance"}


def test_compose_final_answer_strips_step_prefixes():
    step_results = [
        StepResult(
            index=1,
            description="First",
            answer="Step 1: Value found. [1]",
            citation_indexes=[1],
        ),
        StepResult(
            index=2,
            description="Second",
            answer="step 2 - Value found. [2]",
            citation_indexes=[2],
        ),
        StepResult(
            index=3,
            description="Third",
            answer="1) Value found. [3]",
            citation_indexes=[3],
        ),
    ]

    final_answer = ConversationManager._compose_final_answer(step_results)

    assert final_answer.text == "Key Points: Value found. [1][2][3]"
    assert len(final_answer.sections) == 1
    assert final_answer.sections[0].sentences == ["Value found. [1][2][3]"]


def test_ledger_merges_duplicate_claims_across_steps():
    ledger = _EvidenceLedger()
    step_results = [
        StepResult(
            index=1,
            description="Alpha",
            answer="Insight A.",
            citation_indexes=[1],
        ),
        StepResult(
            index=2,
            description="Alpha follow",
            answer="Insight A.",
            citation_indexes=[2],
        ),
        StepResult(
            index=3,
            description="Beta",
            answer="Next point.",
            citation_indexes=[3],
        ),
    ]

    for result in step_results:
        ledger.record_step(turn_id=1, result=result)

    final_answer = ConversationManager._compose_final_answer(
        step_results, ledger=ledger, turn_id=1
    )

    assert final_answer.text == "Evidence & Findings: Insight A. [1][2]\n\nKey Points: Next point. [3]"
    assert ledger.snapshot_for_turn(1) == [
        {
            "text": "Insight A.",
            "normalized": "insight a",
            "citations": [1, 2],
            "section": "Evidence & Findings",
            "steps": [1, 2],
        },
        {
            "text": "Next point.",
            "normalized": "next point",
            "citations": [3],
            "section": "Key Points",
            "steps": [3],
        },
    ]


def test_ledger_retains_claims_for_follow_up_turns():
    ledger = _EvidenceLedger()

    turn_one = StepResult(
        index=1,
        description="Alpha",
        answer="Insight A.",
        citation_indexes=[1],
    )
    ledger.record_step(turn_id=1, result=turn_one)

    turn_two = StepResult(
        index=1,
        description="Follow up",
        answer="Insight A.",
        citation_indexes=[4],
    )
    ledger.record_step(turn_id=2, result=turn_two)

    claims_turn_one = ledger.claims_for_turn(1)
    claims_turn_two = ledger.claims_for_turn(2)

    assert len(claims_turn_one) == 1
    assert claims_turn_one[0].citations == {1}
    assert claims_turn_one[0].steps == [1]

    assert len(claims_turn_two) == 1
    assert claims_turn_two[0].citations == {4}
    assert claims_turn_two[0].steps == [1]


def test_adversarial_judge_detects_duplicate_claims():
    judge = _AdversarialJudge()
    sections = [
        ConsolidatedSection(
            title="Key Points",
            sentences=["Insight A. [1]"],
            citation_indexes=[1],
        ),
        ConsolidatedSection(
            title="Key Points",
            sentences=["Insight A. [2]"],
            citation_indexes=[2],
        ),
    ]
    conflicts: list[ConflictNote] = []
    consolidation = ConsolidationOutput(
        text=ConversationManager._assemble_answer_text(sections, conflicts),
        sections=sections,
        conflicts=conflicts,
        section_usage={1: {"Key Points"}, 2: {"Key Points"}},
    )
    citations = [{"id": 1}, {"id": 2}]
    verdict = judge.review(
        consolidation,
        citations,
        ledger_snapshot=[{"normalized": "insight a", "citations": [1, 2]}],
        scope=None,
        answer_length=AnswerLength.NORMAL,
        response_mode=ResponseMode.GENERATIVE,
    )

    assert verdict.decision == "repair"
    assert "duplicate_claim" in verdict.reason_codes


def test_adversarial_judge_trims_unused_evidence():
    judge = _AdversarialJudge()
    sections = [
        ConsolidatedSection(
            title="Key Points",
            sentences=["Fact. [1]"],
            citation_indexes=[1],
        )
    ]
    conflicts: list[ConflictNote] = []
    consolidation = ConsolidationOutput(
        text=ConversationManager._assemble_answer_text(sections, conflicts),
        sections=sections,
        conflicts=conflicts,
        section_usage={1: {"Key Points"}},
    )
    citations = [{"id": 1}, {"id": 2}]
    verdict = judge.review(
        consolidation,
        citations,
        ledger_snapshot=[{"normalized": "fact", "citations": [1]}],
        scope=None,
        answer_length=AnswerLength.NORMAL,
        response_mode=ResponseMode.GENERATIVE,
    )

    assert verdict.decision == "repair"
    assert "unused_evidence" in verdict.reason_codes

    fix = judge.apply_fixes(
        consolidation,
        citations,
        verdict=verdict,
        ledger_snapshot=[{"normalized": "fact", "citations": [1]}],
    )

    assert len(fix.citations) == 1
    assert fix.citation_mapping == {1: 1}
    assert fix.consolidation.sections[0].sentences == ["Fact. [1]"]


def test_adversarial_judge_requires_claim_coverage():
    judge = _AdversarialJudge()
    sections = [
        ConsolidatedSection(
            title="Key Points",
            sentences=["Insight A. [1]"],
            citation_indexes=[1],
        )
    ]
    consolidation = ConsolidationOutput(
        text=ConversationManager._assemble_answer_text(sections, []),
        sections=sections,
        conflicts=[],
        section_usage={1: {"Key Points"}},
    )
    citations = [{"id": 1}, {"id": 2}]
    ledger_snapshot = [
        {"normalized": "insight a", "text": "Insight A", "citations": [1]},
        {"normalized": "insight b", "text": "Insight B", "citations": [2]},
    ]

    verdict = judge.review(
        consolidation,
        citations,
        ledger_snapshot=ledger_snapshot,
        scope=None,
        answer_length=AnswerLength.NORMAL,
        response_mode=ResponseMode.GENERATIVE,
    )

    assert verdict.decision == "replan"
    assert "missing_claim" in verdict.reason_codes


class _NoMarkerClient:
    def health_check(self) -> bool:
        return True

    def chat(self, messages, *, preset, extra_options=None):  # type: ignore[override]
        return ChatMessage(
            content="Summary without markers",
            citations=[{"source": "Doc", "snippet": "Detail"}],
            reasoning=None,
            raw_response={"choices": []},
        )


class _ExistingMarkerClient(_NoMarkerClient):
    def chat(self, messages, *, preset, extra_options=None):  # type: ignore[override]
        return ChatMessage(
            content="Summary already cited [1].",
            citations=[{"source": "Doc", "snippet": "Detail"}],
            reasoning=None,
            raw_response={"choices": []},
        )


def test_register_turn_appends_markers_when_missing() -> None:
    manager = ConversationManager(_NoMarkerClient())

    turn = manager.ask("What happened?")

    assert "[1]" in turn.answer
    assert turn.answer.rstrip().endswith("[1]")


def test_register_turn_preserves_existing_markers() -> None:
    manager = ConversationManager(_ExistingMarkerClient())

    turn = manager.ask("What happened?")

    assert turn.answer == "Summary already cited [1]."
