import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.conversation_manager import ConversationManager, StepResult


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
