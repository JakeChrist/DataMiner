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

    expected = (
        "Insight A. [1][2]\n\n"
        "Follow up detail. [3]\n\n"
        "Fallback text. [4]\n\n"
        "Quick tip Another tip [5][6]"
    )
    assert final_answer == expected
