"""Stateful helpers for coordinating LMStudio conversations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import copy
import json
import re
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
    status: str = "pending"

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
        shared_context = "\n\n".join(context_snippets or [])
        base_options = copy.deepcopy(extra_options) if extra_options else {}
        executed_plan: list[PlanItem] = [
            PlanItem(description=item.description, status="not_started")
            for item in plan_items
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
                messages = self._build_messages(
                    prompt,
                    combined_snippets if combined_snippets else None,
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
                assumptions.append(
                    f"No citations available for step {index}: {plan_item.description}"
                )
            step_results.append(result)

        aggregated, citation_index_map = self._aggregate_citations(step_results)
        for result in step_results:
            indexes = self._collect_citation_indexes(result.citations, citation_index_map)
            result.citation_indexes = indexes

        answer = self._compose_final_answer(step_results)
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
        raw_response = {
            "dynamic_plan": {
                "steps": reasoning_payload["steps"],
                "citations": aggregated,
            }
        }
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
        )
        self.turns.append(turn)
        return turn

    def _generate_plan(self, question: str) -> list[PlanItem]:
        normalized = question.strip()
        if not normalized:
            return []
        segments = re.split(r"[\n\.!?]+", normalized)
        plan: list[PlanItem] = []
        for segment in segments:
            text = segment.strip()
            if len(text) < 4:
                continue
            plan.append(PlanItem(description=text, status="not_started"))
        if not plan:
            plan = [PlanItem(description=normalized, status="not_started")]
        return plan[:6]

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

    @staticmethod
    def _compose_final_answer(step_results: Sequence[StepResult]) -> str:
        lines: list[str] = []
        for result in step_results:
            text = result.answer.strip() or result.description
            markers = "".join(f"[{index}]" for index in result.citation_indexes)
            if markers:
                text = f"{text} {markers}".strip()
            lines.append(text)
        return "\n\n".join(lines)

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
        self, question: str, context_snippets: Sequence[str] | None
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
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
    "DynamicPlanningError",
    "PlanItem",
    "ReasoningArtifacts",
    "ReasoningVerbosity",
    "ResponseMode",
    "SelfCheckResult",
    "StepContextBatch",
    "StepResult",
]

