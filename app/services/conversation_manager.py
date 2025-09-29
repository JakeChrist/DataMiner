"""Stateful helpers for coordinating LMStudio conversations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import difflib
import copy
import html
import json
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from .lmstudio_client import (
    AnswerLength,
    ChatMessage,
    LMStudioClient,
    LMStudioConnectionError,
    LMStudioError,
)


CONSOLIDATION_SYSTEM_PROMPT = """You are a corpus-grounded answering engine. You only answer using information from the provided corpus context. If evidence is insufficient, say so briefly and suggest widening scope. Do not ask for websites, repos, or external info.

Operating mode

Plan briefly (internal): Form a short list of single-action steps needed to answer the user’s question.

Execute per step: Use the corpus context to produce step results tied to sources.

Consolidate (final pass): Combine all usable results into one cohesive answer.

Consolidation rules (non-negotiable)

One-claim-once: Each distinct claim appears once. If a sentence repeats an earlier claim without adding new, specific information, remove or merge it.

Eliminate repetition: Do not keep multiple paraphrases of the same point. Keep the clearest single phrasing.

Minimal citations: Attach the fewest citations needed for each unique claim. Merge duplicate citations for the same claim.

Single conflict note: If sources disagree, include one concise conflict note with both citations."""


class ResponseMode(Enum):
    """Requested shape of the assistant response."""

    GENERATIVE = "generative"
    SOURCES_ONLY = "sources_only"


class ReasoningVerbosity(Enum):
    """How much structured reasoning the assistant should emit."""

    MINIMAL = "minimal"
    BRIEF = "brief"
    EXTENDED = "extended"

    def to_request_options(self) -> dict[str, Any]:
        """Translate the verbosity preset into LMStudio request options."""

        payload: dict[str, Any] = {
            "reasoning": {
                "verbosity": self.value,
                "include_summary": True,
                "include_assumptions": True,
                "include_self_check": True,
            }
        }
        if self is ReasoningVerbosity.MINIMAL:
            payload["reasoning"].update({
                "include_plan": False,
                "max_bullets": 2,
            })
        elif self is ReasoningVerbosity.BRIEF:
            payload["reasoning"].update({
                "include_plan": True,
                "max_bullets": 4,
                "max_plan_items": 3,
            })
        else:  # EXTENDED
            payload["reasoning"].update({
                "include_plan": True,
                "max_bullets": 8,
                "max_plan_items": 6,
            })
        return payload


@dataclass
class PlanItem:
    """A single plan entry returned by the model."""

    description: str
    status: str = "queued"

    @property
    def is_complete(self) -> bool:
        return self.status.lower() in {"complete", "completed", "done"}


@dataclass
class SelfCheckResult:
    """Outcome of the model's self-check routine."""

    passed: bool
    flags: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class AssumptionDecision:
    """Record how ambiguity was handled for a turn."""

    mode: Literal["clarify", "assume", "unspecified"]
    rationale: str | None = None
    clarifying_question: str | None = None


@dataclass
class ReasoningArtifacts:
    """Structured reasoning data parsed from LMStudio metadata."""

    summary_bullets: list[str] = field(default_factory=list)
    plan_items: list[PlanItem] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    assumption_decision: AssumptionDecision | None = None
    self_check: SelfCheckResult | None = None


@dataclass
class ConversationTurn:
    """Record of a single question/answer exchange."""

    question: str
    answer: str
    citations: list[Any] = field(default_factory=list)
    reasoning: dict[str, Any] | None = None
    reasoning_artifacts: ReasoningArtifacts | None = None
    response_mode: ResponseMode = ResponseMode.GENERATIVE
    answer_length: AnswerLength = AnswerLength.NORMAL
    model_name: str | None = None
    asked_at: datetime | None = None
    answered_at: datetime | None = None
    latency_ms: int | None = None
    token_usage: dict[str, int] | None = None
    raw_response: dict[str, Any] | None = None
    step_results: list["StepResult"] = field(default_factory=list)
    ledger_claims: list[dict[str, Any]] = field(default_factory=list)

    @property
    def reasoning_bullets(self) -> list[str]:
        if self.reasoning_artifacts is None:
            return []
        return list(self.reasoning_artifacts.summary_bullets)

    @property
    def plan(self) -> list[PlanItem]:
        if self.reasoning_artifacts is None:
            return []
        return list(self.reasoning_artifacts.plan_items)

    @property
    def assumptions(self) -> list[str]:
        if self.reasoning_artifacts is None:
            return []
        return list(self.reasoning_artifacts.assumptions)

    @property
    def assumption_decision(self) -> AssumptionDecision | None:
        if self.reasoning_artifacts is None:
            return None
        return self.reasoning_artifacts.assumption_decision

    @property
    def self_check(self) -> SelfCheckResult | None:
        if self.reasoning_artifacts is None:
            return None
        return self.reasoning_artifacts.self_check


@dataclass(frozen=True)
class ConnectionState:
    """Connection status event payload."""

    connected: bool
    message: str | None = None


@dataclass
class StepContextBatch:
    """Context payload for a single execution pass of a plan item."""

    snippets: list[str]
    documents: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StepResult:
    """Aggregate response for a single plan step."""

    index: int
    description: str
    answer: str
    citations: list[Any] = field(default_factory=list)
    contexts: list[StepContextBatch] = field(default_factory=list)
    citation_indexes: list[int] = field(default_factory=list)


@dataclass
class ConsolidatedSection:
    """Composition-ready section derived from dynamic plan steps."""

    title: str
    sentences: list[str]
    citation_indexes: list[int]


@dataclass
class ConflictNote:
    """Summary of conflicting claims surfaced during consolidation."""

    summary: str
    citation_indexes: list[int]
    variants: list[dict[str, Any]]


@dataclass
class ConsolidationOutput:
    """Container for the final composed answer and supporting metadata."""

    text: str
    sections: list[ConsolidatedSection]
    conflicts: list[ConflictNote]
    section_usage: dict[int, set[str]]


@dataclass
class _Claim:
    """Internal representation of a unique claim extracted from step outputs."""

    text: str
    normalized: str
    citations: set[int]
    section: str
    steps: list[int]
    has_negation: bool
    tokens: set[str] = field(default_factory=set)
    numbers: set[str] = field(default_factory=set)
    turn_ids: set[int] = field(default_factory=set)


@dataclass
class _LedgerEntry:
    """Persistent record for a claim stored in the evidence ledger."""

    normalized: str
    texts_by_turn: dict[int, str] = field(default_factory=dict)
    citations_by_turn: dict[int, set[int]] = field(default_factory=dict)
    sections_by_turn: dict[int, str] = field(default_factory=dict)
    steps_by_turn: dict[int, set[int]] = field(default_factory=dict)
    tokens_by_turn: dict[int, set[str]] = field(default_factory=dict)
    numbers_by_turn: dict[int, set[str]] = field(default_factory=dict)
    has_negation_by_turn: dict[int, bool] = field(default_factory=dict)
    insertion_order: list[tuple[int, int]] = field(default_factory=list)

    def record(
        self,
        *,
        turn_id: int,
        step_index: int,
        text: str,
        citations: set[int],
        section: str,
        tokens: set[str],
        numbers: set[str],
        has_negation: bool,
    ) -> None:
        steps = self.steps_by_turn.setdefault(turn_id, set())
        steps.add(step_index)
        if (turn_id, step_index) not in self.insertion_order:
            self.insertion_order.append((turn_id, step_index))
        citations_set = self.citations_by_turn.setdefault(turn_id, set())
        citations_set.update(citations)
        tokens_set = self.tokens_by_turn.setdefault(turn_id, set())
        tokens_set.update(tokens)
        numbers_set = self.numbers_by_turn.setdefault(turn_id, set())
        numbers_set.update(numbers)
        previous_text = self.texts_by_turn.get(turn_id)
        if not previous_text or len(text) > len(previous_text):
            self.texts_by_turn[turn_id] = text
        self.sections_by_turn.setdefault(turn_id, section)
        if has_negation:
            self.has_negation_by_turn[turn_id] = True
        else:
            self.has_negation_by_turn.setdefault(turn_id, False)


class _EvidenceLedger:
    """Track claims across turns so consolidation can reuse prior findings."""

    def __init__(self) -> None:
        self._entries: dict[str, _LedgerEntry] = {}
        self._order: list[str] = []

    def clear_turn(self, turn_id: int) -> None:
        """Remove any existing records for ``turn_id``."""

        for key, entry in list(self._entries.items()):
            entry.texts_by_turn.pop(turn_id, None)
            entry.citations_by_turn.pop(turn_id, None)
            entry.sections_by_turn.pop(turn_id, None)
            entry.steps_by_turn.pop(turn_id, None)
            entry.tokens_by_turn.pop(turn_id, None)
            entry.numbers_by_turn.pop(turn_id, None)
            entry.has_negation_by_turn.pop(turn_id, None)
            entry.insertion_order = [
                item for item in entry.insertion_order if item[0] != turn_id
            ]
            if not entry.texts_by_turn:
                self._entries.pop(key, None)
                self._order = [candidate for candidate in self._order if candidate != key]

    def record_step(self, turn_id: int, result: "StepResult") -> None:
        """Store the findings from ``result`` in the ledger."""

        raw_text = (result.answer or "").strip()
        fallback = result.description.strip()
        base_text = ConversationManager._strip_citation_markers(raw_text).strip()
        if not base_text:
            base_text = (
                ConversationManager._strip_citation_markers(fallback).strip() or fallback
            )
        if not base_text:
            return

        clean_text = ConversationManager._remove_step_prefix(base_text).strip()
        if not clean_text:
            clean_text = base_text.strip()
        polished_text = ConversationManager._polish_sentence(clean_text)
        normalized = ConversationManager._normalize_answer_text(polished_text)
        if not normalized:
            return

        citation_indexes: set[int] = set()
        for index in result.citation_indexes:
            try:
                value = int(index)
            except (TypeError, ValueError):
                continue
            if value > 0:
                citation_indexes.add(value)

        section = ConversationManager._categorize_claim(result.description, polished_text)
        tokens = ConversationManager._claim_tokens(polished_text)
        numbers = ConversationManager._extract_numbers(polished_text)
        has_negation = ConversationManager._has_negation(polished_text)

        entry = self._entries.get(normalized)
        if entry is None:
            entry = _LedgerEntry(normalized=normalized)
            self._entries[normalized] = entry
            self._order.append(normalized)

        entry.record(
            turn_id=turn_id,
            step_index=result.index,
            text=polished_text,
            citations=citation_indexes,
            section=section,
            tokens=tokens,
            numbers=numbers,
            has_negation=has_negation,
        )

    def claims_for_turn(self, turn_id: int) -> list[_Claim]:
        """Return all claims associated with ``turn_id`` in step order."""

        claims: list[_Claim] = []
        for key in self._order:
            entry = self._entries[key]
            steps = sorted(entry.steps_by_turn.get(turn_id, set()))
            if not steps:
                continue
            text = entry.texts_by_turn.get(turn_id)
            if not text:
                text = next(iter(entry.texts_by_turn.values()), "")
            section = entry.sections_by_turn.get(turn_id)
            if not section:
                section = next(iter(entry.sections_by_turn.values()), "Key Points")
            citations = set(entry.citations_by_turn.get(turn_id, set()))
            tokens = set(entry.tokens_by_turn.get(turn_id, set()))
            numbers = set(entry.numbers_by_turn.get(turn_id, set()))
            has_negation = bool(entry.has_negation_by_turn.get(turn_id, False))
            claim = _Claim(
                text=text,
                normalized=entry.normalized,
                citations=citations,
                section=section,
                steps=steps,
                has_negation=has_negation,
                tokens=tokens,
                numbers=numbers,
                turn_ids={turn_id},
            )
            claims.append(claim)
        claims.sort(key=lambda claim: min(claim.steps))
        return claims

    def snapshot_for_turn(self, turn_id: int) -> list[dict[str, Any]]:
        """Return a serialisable ledger view for diagnostics."""

        snapshot: list[dict[str, Any]] = []
        for claim in self.claims_for_turn(turn_id):
            snapshot.append(
                {
                    "text": claim.text,
                    "normalized": claim.normalized,
                    "citations": sorted(claim.citations),
                    "section": claim.section,
                    "steps": list(claim.steps),
                }
            )
        return snapshot


class DynamicPlanningError(RuntimeError):
    """Raised when dynamic planning cannot be completed."""


class ConversationManager:
    """Track conversation history and orchestrate LMStudio requests."""

    def __init__(
        self,
        client: LMStudioClient,
        *,
        system_prompt: str | None = None,
        context_window: int = 4,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.context_window = max(0, context_window)
        self.turns: list[ConversationTurn] = []
        self._connected = True
        self._connection_error: str | None = None
        self._listeners: list[Callable[[ConnectionState], None]] = []
        self._ledger = _EvidenceLedger()

    def add_connection_listener(
        self, listener: Callable[[ConnectionState], None]
    ) -> Callable[[], None]:
        """Subscribe to connection changes and return an unsubscribe callable."""

        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    @property
    def connection_state(self) -> ConnectionState:
        return ConnectionState(self._connected, self._connection_error)

    def check_connection(self) -> ConnectionState:
        """Probe LMStudio and update connection state."""

        healthy = self.client.health_check()
        if healthy:
            self._update_connection(True, None)
        else:
            self._update_connection(False, "LMStudio is unreachable. Check the base URL.")
        return self.connection_state

    def can_ask(self) -> bool:
        """Return ``True`` when it is safe to issue a new query."""

        return self._connected

    def ask(
        self,
        question: str,
        *,
        context_snippets: Sequence[str] | None = None,
        preset: AnswerLength = AnswerLength.NORMAL,
        reasoning_verbosity: ReasoningVerbosity | None = ReasoningVerbosity.BRIEF,
        response_mode: ResponseMode = ResponseMode.GENERATIVE,
        extra_options: dict[str, Any] | None = None,
        context_provider: Callable[[PlanItem, int, int], Iterable[StepContextBatch]] | None = None,
    ) -> ConversationTurn:
        """Send ``question`` to LMStudio and append the resulting turn."""

        if not self._connected:
            message = self._connection_error or "LMStudio is disconnected."
            raise LMStudioConnectionError(message)

        if context_provider is not None:
            try:
                turn = self._ask_with_plan(
                    question,
                    context_snippets=context_snippets,
                    preset=preset,
                    reasoning_verbosity=reasoning_verbosity,
                    response_mode=response_mode,
                    extra_options=extra_options,
                    context_provider=context_provider,
                )
            except DynamicPlanningError:
                turn = self._ask_single_shot(
                    question,
                    context_snippets=context_snippets,
                    preset=preset,
                    reasoning_verbosity=reasoning_verbosity,
                    response_mode=response_mode,
                    extra_options=extra_options,
                )
        else:
            turn = self._ask_single_shot(
                question,
                context_snippets=context_snippets,
                preset=preset,
                reasoning_verbosity=reasoning_verbosity,
                response_mode=response_mode,
                extra_options=extra_options,
            )
        return turn

    def _ask_single_shot(
        self,
        question: str,
        *,
        context_snippets: Sequence[str] | None,
        preset: AnswerLength,
        reasoning_verbosity: ReasoningVerbosity | None,
        response_mode: ResponseMode,
        extra_options: dict[str, Any] | None,
    ) -> ConversationTurn:
        messages = self._build_messages(question, context_snippets)
        request_options = self._build_request_options(
            question,
            reasoning_verbosity,
            response_mode,
            extra_options,
        )
        try:
            response = self.client.chat(
                messages,
                preset=preset,
                extra_options=request_options or None,
            )
        except LMStudioError as exc:
            self._update_connection(False, str(exc) or "Unable to reach LMStudio.")
            raise

        self._update_connection(True, None)
        turn = self._register_turn(question, response, response_mode, preset)
        return turn

    _PLAN_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "please",
        "provide",
        "should",
        "tell",
        "that",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
    }

    _PLAN_ACTION_STARTERS = {
        "analyze",
        "assess",
        "build",
        "calculate",
        "check",
        "collect",
        "compare",
        "compile",
        "create",
        "determine",
        "develop",
        "evaluate",
        "examine",
        "explain",
        "gather",
        "highlight",
        "identify",
        "list",
        "map",
        "outline",
        "prepare",
        "propose",
        "recommend",
        "research",
        "review",
        "summarize",
        "synthesize",
    }

    _PLAN_CONNECTOR_PATTERN = re.compile(
        r"\b(?:and then|and|then|after that|next|finally)\b",
        flags=re.IGNORECASE,
    )

    _CLAIM_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "was",
        "were",
        "which",
        "with",
    }

    _WORD_PATTERN = re.compile(r"[A-Za-z0-9%$€£¥]+")
    _NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")

    def _ask_with_plan(
        self,
        question: str,
        *,
        context_snippets: Sequence[str] | None,
        preset: AnswerLength,
        reasoning_verbosity: ReasoningVerbosity | None,
        response_mode: ResponseMode,
        extra_options: dict[str, Any] | None,
        context_provider: Callable[[PlanItem, int, int], Iterable[StepContextBatch]],
    ) -> ConversationTurn:
        plan_items = self._generate_plan(question)
        if not plan_items:
            raise DynamicPlanningError("No plan items generated")

        total_steps = len(plan_items)
        turn_id = len(self.turns) + 1
        self._ledger.clear_turn(turn_id)
        shared_context = "\n\n".join(context_snippets or [])
        base_options = copy.deepcopy(extra_options) if extra_options else {}
        executed_plan: list[PlanItem] = [
            PlanItem(description=item.description, status="queued") for item in plan_items
        ]
        step_results: list[StepResult] = []
        assumptions: list[str] = []

        for index, plan_item in enumerate(executed_plan, start=1):
            plan_item.status = "running"
            try:
                batches = list(context_provider(plan_item, index, total_steps))
            except Exception as exc:  # pragma: no cover - fallback to single shot
                raise DynamicPlanningError("Context provider failed") from exc
            if not batches:
                batches = [StepContextBatch(snippets=[], documents=[])]

            answer_parts: list[str] = []
            citations: list[Any] = []
            used_contexts: list[StepContextBatch] = []

            for pass_index, batch in enumerate(batches, start=1):
                used_contexts.append(batch)
                combined_snippets: list[str] = []
                if shared_context:
                    combined_snippets.append(shared_context)
                combined_snippets.extend(batch.snippets)
                prompt = self._build_step_prompt(
                    question,
                    plan_item.description,
                    index,
                    total_steps,
                    pass_index,
                )
                extra_system_prompts = None
                if index == total_steps:
                    extra_system_prompts = [CONSOLIDATION_SYSTEM_PROMPT]
                messages = self._build_messages(
                    prompt,
                    combined_snippets if combined_snippets else None,
                    extra_system_prompts=extra_system_prompts,
                )
                merged_options = self._merge_step_options(
                    base_options,
                    batch.documents,
                    plan_item.description,
                    reasoning_verbosity,
                    response_mode,
                )
                try:
                    response = self.client.chat(
                        messages,
                        preset=preset,
                        extra_options=merged_options or None,
                    )
                except LMStudioError as exc:
                    self._update_connection(False, str(exc) or "Unable to reach LMStudio.")
                    raise

                self._update_connection(True, None)
                text = response.content.strip()
                if text:
                    answer_parts.append(text)
                if response.citations:
                    citations.extend(copy.deepcopy(response.citations))

            if not answer_parts:
                message = (
                    f"No direct evidence located for step {index}: {plan_item.description}"
                )
                answer_parts.append(message)
                assumptions.append(message)
            plan_item.status = "done"
            result = StepResult(
                index=index,
                description=plan_item.description,
                answer="\n\n".join(answer_parts).strip(),
                citations=self._deduplicate_citations(citations),
                contexts=used_contexts,
            )
            if not result.citations:
                inferred = self._fallback_citations_from_contexts(used_contexts)
                if inferred:
                    result.citations = inferred
                else:
                    assumptions.append(
                        f"No citations available for step {index}: {plan_item.description}"
                    )
            step_results.append(result)

        aggregated, citation_index_map = self._aggregate_citations(step_results)
        for result in step_results:
            indexes = self._collect_citation_indexes(result.citations, citation_index_map)
            result.citation_indexes = indexes
            self._ledger.record_step(turn_id, result)

        consolidation = self._compose_final_answer(
            step_results, ledger=self._ledger, turn_id=turn_id
        )
        answer = consolidation.text
        summary = f"Executed {total_steps} dynamic step{'s' if total_steps != 1 else ''}."
        artifacts = ReasoningArtifacts(
            summary_bullets=[summary],
            plan_items=executed_plan,
            assumptions=assumptions,
        )
        reasoning_payload = {
            "plan": [
                {"description": item.description, "status": item.status}
                for item in executed_plan
            ],
            "assumptions": assumptions,
            "steps": [
                {
                    "index": result.index,
                    "description": result.description,
                    "answer": result.answer,
                    "citations": result.citation_indexes,
                }
                for result in step_results
            ],
        }
        ledger_snapshot = self._ledger.snapshot_for_turn(turn_id)
        if ledger_snapshot:
            reasoning_payload["ledger"] = ledger_snapshot
        if consolidation.sections:
            reasoning_payload["final_sections"] = [
                {
                    "title": section.title,
                    "sentences": list(section.sentences),
                    "citations": section.citation_indexes,
                }
                for section in consolidation.sections
            ]
        if consolidation.conflicts:
            reasoning_payload["conflicts"] = [
                {
                    "summary": note.summary,
                    "citations": note.citation_indexes,
                    "variants": note.variants,
                }
                for note in consolidation.conflicts
            ]
        if consolidation.section_usage:
            for citation_index, section_names in consolidation.section_usage.items():
                if citation_index <= 0 or citation_index > len(aggregated):
                    continue
                citation = aggregated[citation_index - 1]
                existing_tags: list[str] = []
                raw_tags = citation.get("tag_names") if isinstance(citation, dict) else None
                if isinstance(raw_tags, list):
                    existing_tags = [str(tag) for tag in raw_tags if str(tag).strip()]
                merged = list(dict.fromkeys(existing_tags + sorted(section_names)))
                if isinstance(citation, dict):
                    citation["tag_names"] = merged
        if consolidation.conflicts:
            for note in consolidation.conflicts:
                for citation_index in note.citation_indexes:
                    if citation_index <= 0 or citation_index > len(aggregated):
                        continue
                    citation = aggregated[citation_index - 1]
                    if isinstance(citation, dict):
                        citation["conflict_summary"] = note.summary
                        citation["conflicts"] = [
                            {
                                "text": variant["text"],
                                "citations": variant["citations"],
                            }
                            for variant in note.variants
                        ]
        raw_response = {
            "dynamic_plan": {
                "steps": reasoning_payload["steps"],
                "citations": aggregated,
            }
        }
        if ledger_snapshot:
            raw_response["dynamic_plan"]["ledger"] = ledger_snapshot
        turn = ConversationTurn(
            question=question,
            answer=answer,
            citations=aggregated,
            reasoning=reasoning_payload,
            reasoning_artifacts=artifacts,
            response_mode=response_mode,
            answer_length=preset,
            model_name=getattr(self.client, "model", None),
            raw_response=raw_response,
            step_results=step_results,
            ledger_claims=ledger_snapshot,
        )
        self.turns.append(turn)
        return turn

    def _generate_plan(self, question: str) -> list[PlanItem]:
        normalized = question.strip()
        if not normalized:
            return []

        actions = self._split_question_into_actions(normalized)
        plan: list[PlanItem] = []

        keyword_list = self._extract_plan_keywords(normalized)
        if keyword_list:
            keyword_text = ", ".join(keyword_list)
            plan.append(
                PlanItem(
                    description=f"Scan corpus for background on {keyword_text}.",
                    status="queued",
                )
            )

        for action in actions:
            formatted = self._format_plan_action(action)
            if formatted:
                plan.append(PlanItem(description=formatted, status="queued"))

        plan.append(
            PlanItem(
                description="Synthesize findings into a final answer with citations.",
                status="queued",
            )
        )

        if len(plan) > 8:
            trimmed = plan[:7] + [plan[-1]]
        else:
            trimmed = list(plan)
        if not trimmed:
            trimmed = [PlanItem(description=normalized, status="queued")]
        return trimmed

    def _split_question_into_actions(self, question: str) -> list[str]:
        """Break a question into granular, executable action strings."""

        segments = re.split(r"[\n\.!?]+", question)
        actions: list[str] = []
        for segment in segments:
            text = segment.strip()
            if len(text) < 3:
                continue
            actions.extend(self._split_segment_on_connectors(text))
        return [action for action in actions if action]

    def _split_segment_on_connectors(self, segment: str) -> list[str]:
        """Split a sentence on conjunctions that likely denote separate actions."""

        if not segment:
            return []

        parts: list[str] = []
        queue = [segment]

        while queue:
            current = queue.pop(0).strip()
            if not current:
                continue

            match = self._PLAN_CONNECTOR_PATTERN.search(current)
            if not match:
                parts.append(current)
                continue

            prefix = current[: match.start()].strip()
            suffix = current[match.end() :].strip()

            if not prefix or not suffix:
                parts.append(current)
                continue

            if self._looks_like_action(suffix):
                queue.append(prefix)
                queue.append(suffix)
                continue

            parts.append(current)

        normalized_parts: list[str] = []
        for part in parts:
            stripped = part.strip(" ,;:-")
            if stripped:
                normalized_parts.append(stripped)
        return normalized_parts

    def _looks_like_action(self, text: str) -> bool:
        if not text:
            return False
        word = text.split()[0].lower()
        return word in self._PLAN_ACTION_STARTERS

    def _extract_plan_keywords(self, question: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", question)
        keywords: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            lower = token.lower()
            if lower in self._PLAN_STOPWORDS or lower in seen:
                continue
            seen.add(lower)
            keywords.append(token)
            if len(keywords) >= 3:
                break
        return keywords

    @classmethod
    def _format_plan_action(cls, action: str) -> str:
        text = action.strip()
        if not text:
            return ""

        text = re.sub(
            r"^(?:please|kindly|could you|would you|let's|lets|we need to|need to|i need to|i want to|should)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.strip()
        if not text:
            return ""
        if text[-1] in "?!.":
            text = text[:-1]
        text = text.strip()
        if not text:
            return ""
        if not text[0].isupper():
            text = text[0].upper() + text[1:]
        if not text.lower().startswith(tuple(cls._PLAN_ACTION_STARTERS)):
            text = f"Investigate {text}" if not text.lower().startswith("investigate") else text
        if not text.endswith("."):
            text = f"{text}."
        return text

    def _register_turn(
        self,
        question: str,
        response: ChatMessage,
        response_mode: ResponseMode,
        preset: AnswerLength,
    ) -> ConversationTurn:
        artifacts = self._parse_reasoning_artifacts(response.reasoning)
        turn = ConversationTurn(
            question=question,
            answer=response.content,
            citations=response.citations,
            reasoning=response.reasoning,
            reasoning_artifacts=artifacts,
            response_mode=response_mode,
            answer_length=preset,
            model_name=getattr(self.client, "model", None),
            raw_response=response.raw_response,
        )
        self.turns.append(turn)
        return turn

    @staticmethod
    def _build_request_options(
        question: str,
        reasoning_verbosity: ReasoningVerbosity | None,
        response_mode: ResponseMode,
        extra_options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if reasoning_verbosity is not None:
            options.update(reasoning_verbosity.to_request_options())
        if response_mode is ResponseMode.SOURCES_ONLY:
            options["response_mode"] = response_mode.value
        if extra_options:
            for key, value in extra_options.items():
                if (
                    key == "reasoning"
                    and isinstance(value, dict)
                    and isinstance(options.get("reasoning"), dict)
                ):
                    options["reasoning"].update(value)
                else:
                    options[key] = copy.deepcopy(value)
        retrieval = options.get("retrieval")
        if isinstance(retrieval, dict):
            normalized_question = question.strip()
            query = retrieval.get("query")
            if normalized_question and (query is None or not str(query).strip()):
                retrieval["query"] = normalized_question
        return options

    def _merge_step_options(
        self,
        base_options: dict[str, Any],
        documents: Sequence[dict[str, Any]],
        question: str,
        reasoning_verbosity: ReasoningVerbosity | None,
        response_mode: ResponseMode,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = copy.deepcopy(base_options) if base_options else {}
        if documents:
            retrieval = extra.setdefault("retrieval", {})
            payload_docs = retrieval.setdefault("documents", [])
            payload_docs.extend(copy.deepcopy(list(documents)))
        merged = self._build_request_options(
            question,
            reasoning_verbosity,
            response_mode,
            extra,
        )
        return merged

    def _build_step_prompt(
        self,
        question: str,
        step_description: str,
        index: int,
        total_steps: int,
        pass_index: int,
    ) -> str:
        prefix = (
            f"Step {index} of {total_steps} (pass {pass_index}): {step_description}."
        )
        instructions = (
            "Respond with a concise factual finding for this step. "
            "Do not include bracketed citation markers; they will be added later."
        )
        return f"{prefix}\nOriginal question: {question}\n{instructions}"

    @staticmethod
    def _deduplicate_citations(citations: Sequence[Any]) -> list[Any]:
        unique: list[Any] = []
        seen: set[str] = set()
        for citation in citations:
            key = ConversationManager._citation_key(citation)
            if key in seen:
                continue
            seen.add(key)
            unique.append(copy.deepcopy(citation))
        return unique

    @staticmethod
    def _citation_key(citation: Any) -> str:
        try:
            return json.dumps(citation, sort_keys=True, default=str)
        except TypeError:
            return str(citation)

    def _aggregate_citations(
        self, step_results: Sequence[StepResult]
    ) -> tuple[list[Any], dict[str, int]]:
        aggregated: list[Any] = []
        index_map: dict[str, int] = {}
        for result in step_results:
            for citation in result.citations:
                key = self._citation_key(citation)
                if key not in index_map:
                    entry = copy.deepcopy(citation)
                    aggregated.append(entry)
                    index_map[key] = len(aggregated)
        for citation in aggregated:
            key = self._citation_key(citation)
            steps = []
            for result in step_results:
                if any(self._citation_key(item) == key for item in result.citations):
                    steps.append(result.index)
            if steps:
                try:
                    citation["steps"] = sorted(set(steps))  # type: ignore[index]
                except TypeError:
                    pass
        return aggregated, index_map

    @staticmethod
    def _collect_citation_indexes(
        citations: Sequence[Any], index_map: dict[str, int]
    ) -> list[int]:
        indexes = {
            index_map[key]
            for key in (ConversationManager._citation_key(citation) for citation in citations)
            if key in index_map
        }
        return sorted(indexes)

    _CITATION_PATTERN = re.compile(r"\[(?:\d+|[a-zA-Z]+)\]")
    _STEP_PREFIX_PATTERN = re.compile(
        r"""
        ^\s*
        (
            (?:(?:step|pass|plan\s+item)\s+\d+(?:\s*(?:of|/)\s*\d+)?[:\.\)\-\]]*)
            |
            (?:\(?\d{1,3}\)?[:\.\)\-])
            |
            (?:[A-Za-z][:\.\)\-])
            |
            (?:[ivxlcdm]{1,4}[:\.\)\-])
        )
        \s*
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    @staticmethod
    def _strip_citation_markers(text: str) -> str:
        return ConversationManager._CITATION_PATTERN.sub("", text)

    @staticmethod
    def _remove_step_prefix(text: str) -> str:
        """Remove leading step/number markers used in intermediate outputs."""

        cleaned = ConversationManager._STEP_PREFIX_PATTERN.sub("", text, count=1)
        cleaned = cleaned.lstrip(" \t-–—:.)]")
        return cleaned.lstrip()

    @staticmethod
    def _normalize_answer_text(text: str) -> str:
        stripped = ConversationManager._strip_citation_markers(text)
        stripped = ConversationManager._remove_step_prefix(stripped)
        collapsed = " ".join(stripped.split())
        collapsed = collapsed.strip(".,;:!?")
        return collapsed.lower()

    @staticmethod
    def _polish_sentence(text: str) -> str:
        compact = " ".join((text or "").split())
        if not compact:
            return ""
        if compact[0].islower():
            compact = compact[0].upper() + compact[1:]
        return compact

    @staticmethod
    def _claim_tokens(text: str) -> set[str]:
        tokens: set[str] = set()
        for match in ConversationManager._WORD_PATTERN.findall(text.lower()):
            token = match.strip()
            if not token or token in ConversationManager._CLAIM_STOPWORDS:
                continue
            tokens.add(token)
        return tokens

    @staticmethod
    def _extract_numbers(text: str) -> set[str]:
        numbers: set[str] = set()
        for match in ConversationManager._NUMBER_PATTERN.findall(text):
            cleaned = match.strip()
            if not cleaned:
                continue
            normalized = cleaned.replace(",", "")
            numbers.add(normalized)
        return numbers

    @staticmethod
    def _claims_redundant(primary: _Claim, candidate: _Claim) -> bool:
        if not candidate.tokens:
            return False
        shared = primary.tokens & candidate.tokens
        if not shared:
            return False
        candidate_size = max(len(candidate.tokens), 1)
        coverage = len(shared) / candidate_size
        union = primary.tokens | candidate.tokens
        jaccard = len(shared) / max(len(union), 1)
        if candidate.tokens <= primary.tokens and candidate.numbers <= primary.numbers:
            return True
        if coverage >= 0.85 and candidate.numbers <= primary.numbers:
            return True
        if jaccard >= 0.9 and candidate.numbers <= primary.numbers:
            return True
        ratio = difflib.SequenceMatcher(
            None, primary.normalized, candidate.normalized
        ).ratio()
        if ratio >= 0.92 and candidate.numbers <= primary.numbers:
            return True
        return False

    @staticmethod
    def _filter_redundant_claims(claims: Sequence[_Claim]) -> list[_Claim]:
        filtered: list[_Claim] = []
        for claim in claims:
            merged = False
            for existing in filtered:
                if ConversationManager._claims_redundant(existing, claim):
                    existing.citations.update(claim.citations)
                    existing.steps.extend(claim.steps)
                    existing.tokens.update(claim.tokens)
                    existing.numbers.update(claim.numbers)
                    existing.steps = sorted(set(existing.steps))
                    merged = True
                    break
                if ConversationManager._claims_redundant(claim, existing):
                    existing.text = claim.text
                    existing.normalized = claim.normalized
                    existing.has_negation = claim.has_negation
                    existing.citations.update(claim.citations)
                    existing.steps.extend(claim.steps)
                    existing.tokens.update(claim.tokens)
                    existing.numbers.update(claim.numbers)
                    existing.steps = sorted(set(existing.steps))
                    merged = True
                    break
            if not merged:
                filtered.append(claim)
        return filtered

    @staticmethod
    def _categorize_claim(description: str, answer_text: str) -> str:
        lookup = [
            ("Context & Background", ("background", "context", "overview", "history", "landscape", "scan")),
            ("Comparisons & Trends", ("compare", "comparison", "contrast", "versus", "trend", "change")),
            (
                "Evidence & Findings",
                ("finding", "result", "evidence", "analysis", "insight", "detail", "data"),
            ),
            (
                "Risks & Limitations",
                ("risk", "concern", "issue", "challenge", "gap", "limitation", "problem"),
            ),
            (
                "Recommendations & Next Steps",
                ("recommend", "should", "plan", "propose", "next step", "action", "improve", "need"),
            ),
            (
                "Practical Guidance",
                ("tip", "guide", "how to", "process", "step", "implementation", "instruction"),
            ),
            (
                "Synthesis & Conclusion",
                ("synthes", "conclusion", "summary", "overall", "final", "wrap"),
            ),
        ]
        source = f"{description}\n{answer_text}".lower()
        for title, keywords in lookup:
            if any(keyword in source for keyword in keywords):
                return title
        return "Key Points"

    @staticmethod
    def _format_sentence(text: str, citations: set[int]) -> str:
        polished = ConversationManager._polish_sentence(text)
        if not polished:
            return ""
        if polished[-1] not in ".!?":
            polished = f"{polished}."
        markers = "".join(f"[{index}]" for index in sorted(index for index in citations if index > 0))
        if markers:
            return f"{polished} {markers}".strip()
        return polished

    @staticmethod
    def _conflict_key(normalized: str) -> str:
        base = re.sub(
            r"\b(?:not|no|never|without|lack|lacks|lacking|failed|fails|failing|isn't|aren't|wasn't|weren't|can't|cannot|won't|doesn't|didn't|don't|negative)\b",
            "",
            normalized,
        )
        return " ".join(base.split())

    @staticmethod
    def _has_negation(text: str) -> bool:
        lowered = text.lower()
        return bool(
            re.search(
                r"\b(?:not|no|never|without|lack|lacks|lacking|failed|fails|failing|isn't|aren't|wasn't|weren't|can't|cannot|won't|doesn't|didn't|don't|negative)\b",
                lowered,
            )
        )

    @staticmethod
    def _build_conflict_summary_text(variants: Sequence[str]) -> str:
        texts = [text.strip().rstrip(".") for text in variants if text and text.strip()]
        if not texts:
            return "Conflicting evidence from cited sources."
        if len(texts) == 1:
            return f"Conflicting evidence: {texts[0]}."
        if len(texts) == 2:
            return f"Conflicting evidence: {texts[0]} vs {texts[1]}."
        head = ", ".join(texts[:-1])
        tail = texts[-1]
        return f"Conflicting evidence: {head}, and {tail}."

    @staticmethod
    def _claims_from_step_results(
        step_results: Sequence[StepResult],
    ) -> tuple[list[str], dict[str, list[_Claim]], dict[str, list[_Claim]]]:
        sections_in_order: list[str] = []
        section_claims: dict[str, list[_Claim]] = {}
        seen_by_normalized: dict[str, _Claim] = {}
        conflict_groups: dict[str, list[_Claim]] = {}

        for result in step_results:
            raw_text = (result.answer or "").strip()
            fallback = result.description.strip()
            base_text = ConversationManager._strip_citation_markers(raw_text).strip()
            if not base_text:
                base_text = (
                    ConversationManager._strip_citation_markers(fallback).strip() or fallback
                )
            if not base_text:
                continue

            clean_text = ConversationManager._remove_step_prefix(base_text).strip()
            if not clean_text:
                clean_text = base_text.strip()
            polished_text = ConversationManager._polish_sentence(clean_text)
            normalized = ConversationManager._normalize_answer_text(polished_text)
            if not normalized:
                continue

            citation_indexes: set[int] = set()
            for index in result.citation_indexes:
                try:
                    value = int(index)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    citation_indexes.add(value)

            section = ConversationManager._categorize_claim(result.description, polished_text)
            has_negation = ConversationManager._has_negation(polished_text)
            tokens = ConversationManager._claim_tokens(polished_text)
            numbers = ConversationManager._extract_numbers(polished_text)

            if normalized in seen_by_normalized:
                claim = seen_by_normalized[normalized]
                claim.citations.update(citation_indexes)
                claim.steps.append(result.index)
                claim.tokens.update(tokens)
                claim.numbers.update(numbers)
                claim.turn_ids.add(0)
            else:
                claim = _Claim(
                    text=polished_text,
                    normalized=normalized,
                    citations=set(citation_indexes),
                    section=section,
                    steps=[result.index],
                    has_negation=has_negation,
                    tokens=set(tokens),
                    numbers=set(numbers),
                    turn_ids={0},
                )
                seen_by_normalized[normalized] = claim
                section_claims.setdefault(section, []).append(claim)
                if section not in sections_in_order:
                    sections_in_order.append(section)

            conflict_key = ConversationManager._conflict_key(normalized)
            conflict_groups.setdefault(conflict_key, []).append(claim)

        return sections_in_order, section_claims, conflict_groups

    @staticmethod
    def _claims_from_ledger(
        ledger: _EvidenceLedger, turn_id: int
    ) -> tuple[list[str], dict[str, list[_Claim]], dict[str, list[_Claim]]]:
        sections_in_order: list[str] = []
        section_claims: dict[str, list[_Claim]] = {}
        conflict_groups: dict[str, list[_Claim]] = {}

        claims = ledger.claims_for_turn(turn_id)
        for claim in claims:
            if claim.section not in sections_in_order:
                sections_in_order.append(claim.section)
            section_claims.setdefault(claim.section, []).append(claim)
            conflict_key = ConversationManager._conflict_key(claim.normalized)
            conflict_groups.setdefault(conflict_key, []).append(claim)

        return sections_in_order, section_claims, conflict_groups

    @staticmethod
    def _compose_final_answer(
        step_results: Sequence[StepResult],
        *,
        ledger: _EvidenceLedger | None = None,
        turn_id: int | None = None,
    ) -> ConsolidationOutput:
        if ledger is not None and turn_id is not None:
            sections_in_order, section_claims, conflict_groups = (
                ConversationManager._claims_from_ledger(ledger, turn_id)
            )
        else:
            sections_in_order, section_claims, conflict_groups = (
                ConversationManager._claims_from_step_results(step_results)
            )

        sections: list[ConsolidatedSection] = []
        section_usage: dict[int, set[str]] = {}

        for section_name in sections_in_order:
            claims = section_claims.get(section_name, [])
            if not claims:
                continue
            claims.sort(key=lambda claim: min(claim.steps))
            filtered_claims = ConversationManager._filter_redundant_claims(claims)
            sentences: list[str] = []
            section_citations: set[int] = set()
            for claim in filtered_claims:
                sentence = ConversationManager._format_sentence(claim.text, claim.citations)
                if not sentence:
                    continue
                sentences.append(sentence)
                section_citations.update(index for index in claim.citations if index > 0)
                for index in claim.citations:
                    if index > 0:
                        section_usage.setdefault(index, set()).add(section_name)
            if sentences:
                sections.append(
                    ConsolidatedSection(
                        title=section_name,
                        sentences=sentences,
                        citation_indexes=sorted(section_citations),
                    )
                )

        conflict_notes: list[ConflictNote] = []
        for group in conflict_groups.values():
            unique_claims = {claim.normalized for claim in group}
            negation_mix = {claim.has_negation for claim in group}
            if len(group) < 2 or len(unique_claims) < 2 or len(negation_mix) < 2:
                continue

            variants: list[dict[str, Any]] = []
            citation_union: set[int] = set()
            for claim in group:
                citations_sorted = sorted(index for index in claim.citations if index > 0)
                if not citations_sorted:
                    continue
                variant_text = ConversationManager._polish_sentence(claim.text)
                variants.append({"text": variant_text.rstrip(), "citations": citations_sorted})
                citation_union.update(citations_sorted)
            if len(variants) < 2 or not citation_union:
                continue

            summary = ConversationManager._build_conflict_summary_text(
                [variant["text"] for variant in variants]
            )
            conflict_notes.append(
                ConflictNote(
                    summary=summary,
                    citation_indexes=sorted(citation_union),
                    variants=variants,
                )
            )

        sections_text: list[str] = []
        for section in sections:
            body = " ".join(section.sentences).strip()
            if body:
                sections_text.append(f"{section.title}: {body}")

        conflict_note_text: list[str] = []
        if conflict_notes:
            summaries: list[str] = []
            citation_union: set[int] = set()
            for note in conflict_notes:
                summaries.append(note.summary.rstrip("."))
                citation_union.update(index for index in note.citation_indexes if index > 0)
            summary_body = " / ".join(part for part in summaries if part)
            if not summary_body:
                summary_body = "Conflicting evidence surfaced."
            markers = "".join(f"[{index}]" for index in sorted(citation_union))
            conflict_statement = f"Conflict: {summary_body}."
            if markers:
                conflict_statement = f"{conflict_statement.rstrip()} {markers}".strip()
            conflict_note_text.append(conflict_statement)

        parts = [part for part in sections_text + conflict_note_text if part]
        final_text = "\n\n".join(parts).strip()

        return ConsolidationOutput(
            text=final_text,
            sections=sections,
            conflicts=conflict_notes,
            section_usage=section_usage,
        )

    @staticmethod
    def _fallback_citations_from_contexts(
        contexts: Sequence[StepContextBatch],
    ) -> list[dict[str, Any]]:
        """Build citation dictionaries from retrieval contexts when the model omits them."""

        candidates: list[dict[str, Any]] = []
        for batch in contexts:
            for document in getattr(batch, "documents", []) or []:
                if not isinstance(document, dict):
                    continue
                citation = ConversationManager._context_document_to_citation(document)
                if citation:
                    candidates.append(citation)
        if not candidates:
            return []
        return ConversationManager._deduplicate_citations(candidates)

    @staticmethod
    def _context_document_to_citation(document: dict[str, Any]) -> dict[str, Any] | None:
        source = str(
            document.get("source")
            or document.get("title")
            or document.get("path")
            or document.get("id")
            or ""
        ).strip()
        if not source:
            source = "Document"

        snippet_raw = (
            document.get("snippet")
            or document.get("highlight")
            or document.get("text")
            or ""
        )
        snippet_text = str(snippet_raw).strip()
        if not snippet_text:
            return None
        snippet_text = ConversationManager._trim_snippet(snippet_text)
        if "<" in snippet_text:
            snippet = snippet_text
        else:
            snippet = html.escape(snippet_text)
        if "<mark" not in snippet:
            snippet = f"<mark>{snippet}</mark>"

        citation: dict[str, Any] = {
            "id": document.get("id") or document.get("document_id") or source,
            "source": source,
            "snippet": snippet,
            "path": document.get("path"),
        }

        document_id = document.get("document_id")
        if document_id is not None:
            try:
                citation["document_id"] = int(document_id)
            except (TypeError, ValueError):
                pass

        chunk_id = document.get("chunk_id") or document.get("passage_id")
        if chunk_id is not None:
            citation["passage_id"] = str(chunk_id)

        page = document.get("page") or document.get("page_number")
        if page is not None:
            try:
                citation["page"] = int(page)
            except (TypeError, ValueError):
                citation["page"] = page

        section = document.get("section") or document.get("heading")
        if section:
            citation["section"] = str(section)
        elif document.get("chunk_index") is not None:
            try:
                citation["section"] = f"Chunk {int(document['chunk_index']) + 1}"
            except (TypeError, ValueError, KeyError):
                pass

        score = document.get("score")
        if score is not None:
            try:
                citation["score"] = float(score)
            except (TypeError, ValueError):
                pass

        identifiers = document.get("identifiers")
        if isinstance(identifiers, list) and identifiers:
            citation["tag_names"] = [str(identifier) for identifier in identifiers[:3]]

        citation.setdefault("tag_names", []).append("Context")

        return citation

    @staticmethod
    def _trim_snippet(snippet: str, *, max_chars: int = 320) -> str:
        compact = " ".join(snippet.split())
        if len(compact) <= max_chars:
            return compact
        return textwrap.shorten(compact, width=max_chars, placeholder="…")

    @staticmethod
    def _parse_reasoning_artifacts(
        reasoning: dict[str, Any] | None,
    ) -> ReasoningArtifacts | None:
        if not reasoning:
            return None

        summary_candidates: list[str] = []
        for key in ("summary_bullets", "bullets", "summary", "points"):
            summary_candidates.extend(
                ConversationManager._coerce_str_list(reasoning.get(key))
            )
        summary_bullets = ConversationManager._deduplicate_preserve_order(summary_candidates)

        plan_items: list[PlanItem] = []
        plan_raw = reasoning.get("plan") or reasoning.get("plan_items")
        if isinstance(plan_raw, Sequence) and not isinstance(plan_raw, (str, bytes, dict)):
            for entry in plan_raw:
                if isinstance(entry, dict):
                    description = str(
                        entry.get("description")
                        or entry.get("step")
                        or entry.get("text")
                        or ""
                    ).strip()
                    status_raw = entry.get("status") or entry.get("state")
                    status = (
                        str(status_raw).strip().lower() if status_raw is not None else ""
                    )
                    if not status:
                        status = "pending"
                elif isinstance(entry, str):
                    description = entry.strip()
                    status = "pending"
                else:
                    continue
                if description:
                    plan_items.append(PlanItem(description=description, status=status))

        assumptions_list: list[str] = []
        decision: AssumptionDecision | None = None
        assumptions_raw = reasoning.get("assumptions")
        if isinstance(assumptions_raw, dict):
            assumptions_list = ConversationManager._coerce_str_list(
                assumptions_raw.get("used")
                or assumptions_raw.get("list")
                or assumptions_raw.get("items")
                or assumptions_raw.get("assumptions")
            )
            question_raw = (
                assumptions_raw.get("clarifying_question")
                or assumptions_raw.get("question")
            )
            clarifying_question = (
                str(question_raw).strip() if isinstance(question_raw, str) else None
            )
            rationale_raw = assumptions_raw.get("rationale") or assumptions_raw.get("reason")
            rationale = (
                str(rationale_raw).strip() if isinstance(rationale_raw, str) else None
            )
            decision_raw = assumptions_raw.get("decision")
            if isinstance(decision_raw, str):
                mode = decision_raw.strip().lower() or "unspecified"
            else:
                mode = "unspecified"
            should_ask = assumptions_raw.get("should_ask")
            if isinstance(should_ask, bool) and should_ask:
                mode = "clarify"
            if mode not in {"clarify", "assume"}:
                if assumptions_list:
                    mode = "assume"
                elif clarifying_question:
                    mode = "clarify"
                else:
                    mode = "unspecified"
            decision = AssumptionDecision(
                mode=mode if mode in {"clarify", "assume", "unspecified"} else "unspecified",
                rationale=rationale,
                clarifying_question=clarifying_question,
            )
        elif isinstance(assumptions_raw, Sequence) and not isinstance(
            assumptions_raw, (str, bytes)
        ):
            assumptions_list = ConversationManager._coerce_str_list(assumptions_raw)
            mode = "assume" if assumptions_list else "unspecified"
            decision = AssumptionDecision(mode=mode, rationale=None, clarifying_question=None)

        self_check_data = reasoning.get("self_check") or reasoning.get("selfCheck")
        self_check: SelfCheckResult | None = None
        if isinstance(self_check_data, dict):
            passed = bool(self_check_data.get("passed"))
            flags = ConversationManager._coerce_str_list(
                self_check_data.get("flags")
                or self_check_data.get("issues")
                or self_check_data.get("notes")
            )
            notes_raw = self_check_data.get("notes") or self_check_data.get("explanation")
            notes = str(notes_raw).strip() if isinstance(notes_raw, str) else None
            self_check = SelfCheckResult(passed=passed, flags=flags, notes=notes)

        if not any([summary_bullets, plan_items, assumptions_list, self_check, decision]):
            return None

        return ReasoningArtifacts(
            summary_bullets=summary_bullets,
            plan_items=plan_items,
            assumptions=assumptions_list,
            assumption_decision=decision,
            self_check=self_check,
        )

    @staticmethod
    def _coerce_str_list(value: Any) -> list[str]:
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, dict)):
            results: list[str] = []
            for item in value:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        results.append(text)
            return results
        return []

    @staticmethod
    def _deduplicate_preserve_order(values: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def _build_messages(
        self,
        question: str,
        context_snippets: Sequence[str] | None,
        *,
        extra_system_prompts: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if extra_system_prompts:
            for prompt in extra_system_prompts:
                text = str(prompt).strip()
                if text:
                    messages.append({"role": "system", "content": text})
        history = self.turns[-self.context_window :] if self.context_window else []
        for turn in history:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})

        user_content = question
        if context_snippets:
            formatted_context = "\n\n".join(context_snippets)
            user_content = f"{question}\n\nContext:\n{formatted_context}"
        messages.append({"role": "user", "content": user_content})
        return messages

    def _update_connection(self, connected: bool, message: str | None) -> None:
        state_changed = connected != self._connected or message != self._connection_error
        self._connected = connected
        self._connection_error = message
        if state_changed:
            self._emit_connection_state()

    def _emit_connection_state(self) -> None:
        state = self.connection_state
        for listener in list(self._listeners):
            try:
                listener(state)
            except Exception:  # pragma: no cover - defensive guard
                continue


__all__ = [
    "AnswerLength",
    "AssumptionDecision",
    "ConnectionState",
    "ConversationManager",
    "ConversationTurn",
    "ConsolidatedSection",
    "ConflictNote",
    "ConsolidationOutput",
    "DynamicPlanningError",
    "PlanItem",
    "ReasoningArtifacts",
    "ReasoningVerbosity",
    "ResponseMode",
    "SelfCheckResult",
    "StepContextBatch",
    "StepResult",
]

