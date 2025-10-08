"""Service for persisting and retrieving conversational working memory."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from ..logging import log_call
from ..storage import WorkingMemoryRepository

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers
    from .conversation_manager import (
        EvidenceRecord,
        PlanItem,
        StateDigestEntry,
        StepResult,
    )


logger = logging.getLogger(__name__)


class WorkingMemoryService:
    """Coordinate persistence and retrieval of working memory entries."""

    def __init__(self, repository: WorkingMemoryRepository) -> None:
        self._repository = repository

    @log_call(logger=logger, level=logging.DEBUG, include_args=False)
    def store_turn_memory(
        self,
        *,
        project_id: int,
        turn_index: int,
        question: str,
        answer: str,
        plan: Sequence["PlanItem"] | None = None,
        step_results: Sequence["StepResult"] | None = None,
        state_digest: Sequence["StateDigestEntry"] | None = None,
        evidence_log: Sequence["EvidenceRecord"] | None = None,
    ) -> None:
        """Persist structured working memory for a completed turn."""

        self._repository.clear_turn(project_id, turn_index)
        sanitized_question = question.strip()
        sanitized_answer = answer.strip()

        if sanitized_question:
            self._repository.add_entry(
                project_id,
                turn_index,
                kind="question",
                title="Question",
                summary=sanitized_question,
                content=f"question {sanitized_question}",
                metadata={"question": sanitized_question},
            )

        if sanitized_answer:
            self._repository.add_entry(
                project_id,
                turn_index,
                kind="final_answer",
                title="Final Answer",
                summary=sanitized_answer,
                content=f"answer {sanitized_answer}",
                metadata={"question": sanitized_question, "turn_index": turn_index},
            )

        for index, item in enumerate(plan or [], start=1):
            description = (item.description or "").strip()
            status = (item.status or "pending").strip()
            if not description:
                continue
            metadata = {
                "status": status,
                "step_index": index,
                "question": sanitized_question,
            }
            content = " ".join(part for part in (description, status, sanitized_question) if part)
            self._repository.add_entry(
                project_id,
                turn_index,
                step_index=index,
                kind="plan",
                title=f"Plan Step {index}",
                summary=description,
                content=content,
                metadata=metadata,
            )

        for result in step_results or []:
            summary = (result.answer or "").strip() or (result.description or "").strip()
            if not summary:
                continue
            contexts: list[str] = []
            for batch in result.contexts:
                snippets = getattr(batch, "snippets", None)
                if isinstance(snippets, Iterable):
                    contexts.extend(str(snippet) for snippet in snippets if snippet)
            metadata: dict[str, Any] = {
                "description": result.description,
                "citation_indexes": list(result.citation_indexes),
                "insufficient": bool(result.insufficient),
                "question": sanitized_question,
                "contexts": contexts,
            }
            self._repository.add_entry(
                project_id,
                turn_index,
                step_index=result.index,
                kind="step_result",
                title=f"Step Result {result.index}",
                summary=summary,
                content=" ".join(part for part in (result.description, summary, sanitized_question) if part),
                metadata=metadata,
            )

        for entry in state_digest or []:
            summary = (entry.summary or "").strip()
            if not summary:
                continue
            metadata = {
                "citation_indexes": list(entry.citation_indexes),
                "pending_questions": list(entry.pending_questions),
                "decisions": list(entry.decisions),
                "question": sanitized_question,
                "step_index": entry.step_index,
            }
            self._repository.add_entry(
                project_id,
                turn_index,
                step_index=entry.step_index,
                kind="state_digest",
                title=f"State Digest {entry.step_index}",
                summary=summary,
                content=" ".join(
                    part
                    for part in (
                        summary,
                        " ".join(entry.pending_questions),
                        " ".join(entry.decisions),
                        sanitized_question,
                    )
                    if part
                ),
                metadata=metadata,
            )

        for record in evidence_log or []:
            intent = (record.intent or "").strip()
            snippets = list(record.snippets)
            documents = [dict(doc) for doc in record.documents]
            summary = intent or "Evidence" if snippets else intent
            metadata = {
                "intent": intent,
                "snippets": snippets,
                "documents": documents,
                "question": sanitized_question,
                "step_index": record.step_index,
            }
            content = " ".join(part for part in (intent, " ".join(snippets), sanitized_question) if part)
            self._repository.add_entry(
                project_id,
                turn_index,
                step_index=record.step_index,
                kind="evidence",
                title=f"Evidence {record.step_index}",
                summary=summary.strip() or "Evidence",
                content=content,
                metadata=metadata,
            )

    def collect_context_records(
        self,
        query: str,
        *,
        project_id: int,
        limit: int = 5,
        kinds: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return working-memory context records matching ``query``."""

        entries = self._repository.search(project_id, query, limit=limit, kinds=kinds)
        records: list[dict[str, Any]] = []
        for entry in entries:
            metadata = entry.get("metadata") or {}
            title = self._entry_title(entry)
            context_text = self._build_context(entry, metadata)
            records.append(
                {
                    "document": {
                        "id": f"memory-{entry['id']}",
                        "title": title,
                        "source_path": self._memory_source_path(project_id, entry),
                        "metadata": metadata,
                        "project_id": project_id,
                        "working_memory": True,
                        "kind": entry.get("kind"),
                    },
                    "chunk": {
                        "id": entry["id"],
                        "index": entry.get("step_index") or 0,
                        "text": entry.get("summary") or title,
                    },
                    "context": context_text,
                    "highlight": entry.get("highlight"),
                    "score": entry.get("score"),
                    "memory_entry": entry,
                }
            )
        return records

    @staticmethod
    def _memory_source_path(project_id: int, entry: dict[str, Any]) -> str:
        turn_index = entry.get("turn_index") or 0
        return f"memory://project/{project_id}/turn/{turn_index}/entry/{entry['id']}"

    @staticmethod
    def _entry_title(entry: dict[str, Any]) -> str:
        kind = str(entry.get("kind") or "memory").replace("_", " ").title()
        turn_index = entry.get("turn_index")
        step_index = entry.get("step_index")
        if step_index:
            return f"Turn {turn_index} Step {step_index} {kind}"
        if turn_index:
            return f"Turn {turn_index} {kind}"
        return kind

    @staticmethod
    def _build_context(entry: dict[str, Any], metadata: dict[str, Any]) -> str:
        lines: list[str] = []
        summary = str(entry.get("summary") or "").strip()
        if summary:
            lines.append(summary)
        kind = entry.get("kind")
        if kind == "plan":
            status = metadata.get("status")
            if status:
                lines.append(f"Status: {status}")
        elif kind == "step_result":
            description = metadata.get("description")
            if description:
                lines.insert(0, str(description))
        elif kind == "state_digest":
            pending = metadata.get("pending_questions") or []
            if pending:
                lines.append("Pending questions: " + "; ".join(str(item) for item in pending if item))
            decisions = metadata.get("decisions") or []
            if decisions:
                lines.append("Decisions: " + "; ".join(str(item) for item in decisions if item))
        elif kind == "evidence":
            intent = metadata.get("intent")
            if intent:
                lines.insert(0, str(intent))
            snippets = metadata.get("snippets") or []
            if snippets:
                lines.append("Snippets: " + " ".join(str(item) for item in snippets if item))
        elif kind == "final_answer":
            lines.insert(0, "Final Answer")
        elif kind == "question":
            lines.insert(0, "Question")
        question = metadata.get("question")
        if question and str(question).strip():
            lines.append(f"Question: {question}")
        return "\n".join(line for line in lines if str(line).strip())

