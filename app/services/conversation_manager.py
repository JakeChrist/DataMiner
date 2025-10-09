"""Stateful helpers for coordinating LMStudio conversations."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
import difflib
import copy
import html
import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
from html.parser import HTMLParser
from typing import Any, Literal

from ..logging import log_call


logger = logging.getLogger(__name__)


class _SnippetMarkParser(HTMLParser):
    """Parse HTML snippets and remember original highlight spans."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._ranges: list[tuple[int, int]] = []
        self._position = 0
        self._mark_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: D401
        if tag.lower() == "mark":
            self._mark_depth += 1

    def handle_endtag(self, tag: str) -> None:  # noqa: D401
        if tag.lower() == "mark" and self._mark_depth:
            self._mark_depth -= 1

    def handle_data(self, data: str) -> None:  # noqa: D401
        self._append_text(data)

    def handle_entityref(self, name: str) -> None:  # noqa: D401
        self._append_text(html.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:  # noqa: D401
        self._append_text(html.unescape(f"&#{name};"))

    def _append_text(self, data: str) -> None:
        if not data:
            return
        self._parts.append(data)
        length = len(data)
        if self._mark_depth > 0:
            self._ranges.append((self._position, self._position + length))
        self._position += length

    @property
    def text(self) -> str:
        return "".join(self._parts)

    @property
    def mark_ranges(self) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in sorted(self._ranges):
            if not merged:
                merged.append((start, end))
                continue
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged


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

Single conflict note: If sources disagree, include one concise conflict note with both citations.

Grounding enforcement

- Ignore any step outputs flagged as "INSUFFICIENT_EVIDENCE".
- Drop sentences that lack citations or whose citations are not present in the retrieved corpus.
- If supporting snippets are missing for a major claim, request a replan or state that the evidence is insufficient instead of guessing."""


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
    rationale: str | None = None

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
    adversarial_review: JudgeReport | None = None

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
    insufficient: bool = False


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
class JudgeReport:
    """Outcome of the adversarial judge review for a turn."""

    decision: Literal["publish", "insufficient_evidence", "replan"]
    revised: bool
    reasons: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    cycles: int = 0


@dataclass
class _JudgeVerdict:
    """Internal representation of the judge's decision for a draft answer."""

    decision: Literal["publish", "repair", "replan", "insufficient_evidence"]
    reasons: list[str]
    reason_codes: list[str]
    metrics: dict[str, Any]


@dataclass
class _JudgeFixResult:
    """Outcome of automatic repairs applied after a failed review."""

    consolidation: ConsolidationOutput
    citations: list[Any]
    citation_mapping: dict[int, int]
    duplicates_removed: int
    sentences_removed: int


class _AdversarialJudge:
    """Validate consolidated answers against strict quality rules."""

    _CITATION_PATTERN = re.compile(r"\[(\d+)\]")
    _BANNED_HEADINGS = {"recommendations", "next steps", "context"}

    def __init__(self, *, max_cycles: int = 3) -> None:
        self.max_cycles = max(1, max_cycles)

    def review(
        self,
        consolidation: ConsolidationOutput,
        citations: Sequence[Any],
        *,
        ledger_snapshot: Sequence[dict[str, Any]] | None,
        scope: dict[str, Any] | None,
        answer_length: AnswerLength,
        response_mode: ResponseMode,
    ) -> _JudgeVerdict:
        ledger_map = self._build_ledger_map(ledger_snapshot)
        expected_claims = self._expected_claims(ledger_snapshot)
        claim_records: list[dict[str, Any]] = []
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        missing_citations: list[str] = []
        invalid_citations: list[str] = []
        unsupported_claims: list[str] = []
        missing_required_claims: list[dict[str, Any]] = []
        reason_codes: list[str] = []
        reasons: list[str] = []

        for section in consolidation.sections:
            title_key = (section.title or "").strip().lower()
            if title_key in self._BANNED_HEADINGS:
                reasons.append(
                    f"Banned heading '{section.title}' detected; rename or remove section."
                )
                reason_codes.append("banned_heading")
            for sentence in section.sentences:
                normalized, indexes = self._parse_sentence(sentence)
                if not normalized:
                    continue
                claim_records.append(
                    {
                        "normalized": normalized,
                        "sentence": sentence,
                        "indexes": indexes,
                    }
                )
                if normalized in seen:
                    snippet = self._sentence_snippet(sentence)
                    reasons.append(f"Claim '{snippet}' duplicates earlier content.")
                    if "duplicate_claim" not in reason_codes:
                        reason_codes.append("duplicate_claim")
                    duplicates.append(normalized)
                else:
                    seen[normalized] = sentence
                if not indexes:
                    missing_citations.append(normalized)
                else:
                    invalid = [index for index in indexes if not self._index_valid(index, citations)]
                    if invalid:
                        invalid_citations.append(normalized)
                    elif not self._claim_supported(normalized, indexes, ledger_map):
                        unsupported_claims.append(normalized)

        for normalized, record in expected_claims.items():
            if normalized in seen:
                continue
            missing_required_claims.append(record)

        citations_total = sum(len(record["indexes"]) for record in claim_records)
        citations_verified_ok = 0
        for record in claim_records:
            indexes = [
                index
                for index in record["indexes"]
                if self._index_valid(index, citations)
                and (not ledger_map or self._claim_supported(record["normalized"], [index], ledger_map))
            ]
            citations_verified_ok += len(indexes)

        if missing_citations:
            if "missing_citation" not in reason_codes:
                reason_codes.append("missing_citation")
            for normalized in missing_citations:
                snippet = self._sentence_snippet(seen.get(normalized, normalized))
                reasons.append(f"Claim '{snippet}' is missing supporting citations.")

        if invalid_citations:
            if "invalid_citation" not in reason_codes:
                reason_codes.append("invalid_citation")
            for normalized in invalid_citations:
                snippet = self._sentence_snippet(seen.get(normalized, normalized))
                reasons.append(f"Claim '{snippet}' cites unavailable evidence.")

        if unsupported_claims:
            if "unsupported_claim" not in reason_codes:
                reason_codes.append("unsupported_claim")
            for normalized in unsupported_claims:
                snippet = self._sentence_snippet(seen.get(normalized, normalized))
                reasons.append(f"Claim '{snippet}' is not backed by the evidence set.")

        if missing_required_claims:
            if "missing_claim" not in reason_codes:
                reason_codes.append("missing_claim")
            for record in missing_required_claims:
                text = record.get("text") or record.get("normalized") or ""
                snippet = self._sentence_snippet(text)
                reasons.append(f"Claim '{snippet}' is absent from the draft answer.")

        used_indexes = self._collect_used_indexes(consolidation, claim_records)
        unused = [index for index in range(1, len(citations) + 1) if index not in used_indexes]
        if unused:
            if "unused_evidence" not in reason_codes:
                reason_codes.append("unused_evidence")
            markers = ", ".join(f"[{index}]" for index in unused)
            reasons.append(f"Evidence entries {markers} are not referenced in the answer.")

        total_sentences = sum(len(section.sentences) for section in consolidation.sections)
        length_cap = self._length_cap(answer_length)
        if length_cap and total_sentences > length_cap:
            if "length_exceeded" not in reason_codes:
                reason_codes.append("length_exceeded")
            reasons.append(
                "Draft is too long for the requested length preset; remove redundant sentences."
            )

        metrics = {
            "retrieved_k": len(citations),
            "unique_claims": len(seen),
            "duplicates_identified": len(duplicates),
            "citations_total": citations_total,
            "citations_verified_ok": citations_verified_ok,
            "conflict_notes": len(consolidation.conflicts),
            "expected_claims": len(expected_claims),
            "covered_claims": len(expected_claims) - len(missing_required_claims),
            "missing_claims": len(missing_required_claims),
            "sentence_count": total_sentences,
            "length_cap": length_cap,
        }

        no_claims = not claim_records and not consolidation.conflicts
        no_citations = len(citations) == 0
        if no_claims and no_citations:
            scope_reason = "No supporting evidence in current scope"
            reason_codes = ["insufficient_evidence"]
            if self._scope_active(scope):
                scope_reason += "; active filters returned no results"
                reason_codes.append("overscoped_scope")
            return _JudgeVerdict("insufficient_evidence", [scope_reason], reason_codes, metrics)

        if no_citations:
            scope_reason = "No supporting evidence in current scope"
            reason_codes = ["insufficient_evidence"]
            if self._scope_active(scope):
                scope_reason += "; active filters returned no results"
                reason_codes.append("overscoped_scope")
            return _JudgeVerdict("insufficient_evidence", [scope_reason], reason_codes, metrics)

        if reasons:
            decision = "repair"
            if "missing_claim" in reason_codes and not missing_citations:
                decision = "replan"
            return _JudgeVerdict(decision, reasons, reason_codes, metrics)

        return _JudgeVerdict("publish", [], [], metrics)

    def apply_fixes(
        self,
        consolidation: ConsolidationOutput,
        citations: Sequence[Any],
        *,
        verdict: _JudgeVerdict,
        ledger_snapshot: Sequence[dict[str, Any]] | None,
    ) -> _JudgeFixResult:
        sections = [
            ConsolidatedSection(
                title=section.title,
                sentences=list(section.sentences),
                citation_indexes=list(section.citation_indexes),
            )
            for section in consolidation.sections
        ]
        conflicts = [
            ConflictNote(
                summary=note.summary,
                citation_indexes=list(note.citation_indexes),
                variants=[
                    {"text": variant.get("text", ""), "citations": list(variant.get("citations", []))}
                    for variant in note.variants
                ],
            )
            for note in consolidation.conflicts
        ]

        duplicates_removed = 0
        sentences_removed = 0
        ledger_map = self._build_ledger_map(ledger_snapshot)

        if "banned_heading" in verdict.reason_codes:
            sections = self._rename_banned_sections(sections)

        if "duplicate_claim" in verdict.reason_codes:
            sections, removed = self._drop_duplicate_sentences(sections)
            duplicates_removed += removed

        if {"missing_citation", "invalid_citation", "unsupported_claim"}.intersection(
            verdict.reason_codes
        ):
            sections, removed = self._drop_unsupported_sentences(
                sections, len(citations), ledger_map
            )
            sentences_removed += removed

        result = self._finalize_output(sections, conflicts, citations)
        sentences_removed += result.sentences_removed
        return _JudgeFixResult(
            consolidation=result.consolidation,
            citations=result.citations,
            citation_mapping=result.citation_mapping,
            duplicates_removed=duplicates_removed,
            sentences_removed=sentences_removed,
        )

    @staticmethod
    def _sentence_snippet(sentence: str | None, *, limit: int = 80) -> str:
        if not sentence:
            return ""
        clean = ConversationManager._strip_citation_markers(sentence)
        clean = ConversationManager._polish_sentence(clean)
        if len(clean) <= limit:
            return clean
        return clean[: limit - 1].rstrip() + "…"

    @classmethod
    def _parse_sentence(cls, sentence: str) -> tuple[str, list[int]]:
        normalized = ConversationManager._normalize_answer_text(sentence)
        indexes = []
        for match in cls._CITATION_PATTERN.finditer(sentence):
            try:
                indexes.append(int(match.group(1)))
            except (TypeError, ValueError):
                continue
        return normalized, indexes

    @staticmethod
    def _index_valid(index: int, citations: Sequence[Any]) -> bool:
        return 1 <= index <= len(citations)

    @staticmethod
    def _build_ledger_map(
        ledger_snapshot: Sequence[dict[str, Any]] | None,
    ) -> dict[str, set[int]]:
        ledger_map: dict[str, set[int]] = {}
        if not ledger_snapshot:
            return ledger_map
        for entry in ledger_snapshot:
            normalized = str(entry.get("normalized") or "").strip().lower()
            if not normalized:
                continue
            citations = entry.get("citations")
            if isinstance(citations, Iterable) and not isinstance(citations, (str, bytes)):
                indexes: set[int] = set()
                for value in citations:
                    try:
                        index = int(value)
                    except (TypeError, ValueError):
                        continue
                    if index > 0:
                        indexes.add(index)
                if indexes:
                    ledger_map[normalized] = indexes
        return ledger_map

    @staticmethod
    def _expected_claims(
        ledger_snapshot: Sequence[dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        expected: dict[str, dict[str, Any]] = {}
        if not ledger_snapshot:
            return expected
        for entry in ledger_snapshot:
            normalized = str(entry.get("normalized") or "").strip().lower()
            if not normalized:
                continue
            text_raw = str(entry.get("text") or "").strip()
            text = ConversationManager._strip_citation_markers(text_raw)
            citations = []
            raw_citations = entry.get("citations")
            if isinstance(raw_citations, Iterable) and not isinstance(
                raw_citations, (str, bytes)
            ):
                for value in raw_citations:
                    try:
                        index = int(value)
                    except (TypeError, ValueError):
                        continue
                    if index > 0:
                        citations.append(index)
            expected[normalized] = {
                "text": text,
                "citations": citations,
            }
        return expected

    @staticmethod
    def _claim_supported(
        normalized: str, indexes: Sequence[int], ledger_map: dict[str, set[int]]
    ) -> bool:
        if not ledger_map:
            return True
        citations = ledger_map.get(normalized)
        if not citations:
            return False
        return any(index in citations for index in indexes)

    @classmethod
    def _collect_used_indexes(
        cls, consolidation: ConsolidationOutput, claims: Sequence[dict[str, Any]]
    ) -> set[int]:
        used: set[int] = set()
        for record in claims:
            for index in record.get("indexes", []):
                if index > 0:
                    used.add(index)
        for section in consolidation.sections:
            for index in section.citation_indexes:
                if index > 0:
                    used.add(index)
        for note in consolidation.conflicts:
            for index in note.citation_indexes:
                if index > 0:
                    used.add(index)
            for variant in note.variants:
                citations = variant.get("citations")
                if isinstance(citations, Iterable) and not isinstance(
                    citations, (str, bytes)
                ):
                    for index in citations:
                        try:
                            value = int(index)
                        except (TypeError, ValueError):
                            continue
                        if value > 0:
                            used.add(value)
        return used

    @staticmethod
    def _length_cap(answer_length: AnswerLength) -> int:
        if answer_length is AnswerLength.BRIEF:
            return 6
        if answer_length is AnswerLength.NORMAL:
            return 12
        if answer_length is AnswerLength.DETAILED:
            return 20
        return 0

    @staticmethod
    def _scope_active(scope: dict[str, Any] | None) -> bool:
        if not scope:
            return False
        for key in ("include", "exclude", "tags", "folders", "date"):
            value = scope.get(key)
            if isinstance(value, dict):
                if any(value.values()):
                    return True
            elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                if any(str(item).strip() for item in value):
                    return True
            elif value:
                return True
        return False

    @staticmethod
    def _rename_banned_sections(
        sections: Sequence[ConsolidatedSection],
    ) -> list[ConsolidatedSection]:
        renamed: list[ConsolidatedSection] = []
        for section in sections:
            title_key = (section.title or "").strip().lower()
            if title_key in _AdversarialJudge._BANNED_HEADINGS:
                title = "Key Points"
            else:
                title = section.title
            renamed.append(
                ConsolidatedSection(
                    title=title,
                    sentences=list(section.sentences),
                    citation_indexes=list(section.citation_indexes),
                )
            )
        return renamed

    @classmethod
    def _drop_duplicate_sentences(
        cls, sections: Sequence[ConsolidatedSection]
    ) -> tuple[list[ConsolidatedSection], int]:
        seen: set[str] = set()
        updated: list[ConsolidatedSection] = []
        removed = 0
        for section in sections:
            sentences: list[str] = []
            citation_indexes: set[int] = set()
            for sentence in section.sentences:
                normalized = ConversationManager._normalize_answer_text(sentence)
                if normalized and normalized in seen:
                    removed += 1
                    continue
                if normalized:
                    seen.add(normalized)
                sentences.append(sentence)
                citation_indexes.update(
                    int(match.group(1))
                    for match in cls._CITATION_PATTERN.finditer(sentence)
                    if match.group(1).isdigit()
                )
            if sentences:
                updated.append(
                    ConsolidatedSection(
                        title=section.title,
                        sentences=sentences,
                        citation_indexes=sorted(index for index in citation_indexes if index > 0),
                    )
                )
        return updated, removed

    @classmethod
    def _drop_unsupported_sentences(
        cls,
        sections: Sequence[ConsolidatedSection],
        citation_count: int,
        ledger_map: dict[str, set[int]],
    ) -> tuple[list[ConsolidatedSection], int]:
        updated: list[ConsolidatedSection] = []
        removed = 0
        for section in sections:
            sentences: list[str] = []
            section_indexes: set[int] = set()
            for sentence in section.sentences:
                normalized = ConversationManager._normalize_answer_text(sentence)
                indexes = [
                    int(match.group(1))
                    for match in cls._CITATION_PATTERN.finditer(sentence)
                    if match.group(1).isdigit()
                ]
                if not indexes:
                    removed += 1
                    continue
                if any(index <= 0 or index > citation_count for index in indexes):
                    removed += 1
                    continue
                if ledger_map and not cls._claim_supported(normalized, indexes, ledger_map):
                    removed += 1
                    continue
                sentences.append(sentence)
                section_indexes.update(indexes)
            if sentences:
                updated.append(
                    ConsolidatedSection(
                        title=section.title,
                        sentences=sentences,
                        citation_indexes=sorted(section_indexes),
                    )
                )
        return updated, removed

    @classmethod
    def _finalize_output(
        cls,
        sections: Sequence[ConsolidatedSection],
        conflicts: Sequence[ConflictNote],
        citations: Sequence[Any],
    ) -> _JudgeFixResult:
        used_indexes: list[int] = []
        for section in sections:
            for sentence in section.sentences:
                for match in cls._CITATION_PATTERN.finditer(sentence):
                    if match.group(1).isdigit():
                        value = int(match.group(1))
                        if value > 0:
                            used_indexes.append(value)
        for note in conflicts:
            for index in note.citation_indexes:
                if index > 0:
                    used_indexes.append(index)
            for variant in note.variants:
                citations_list = variant.get("citations")
                if isinstance(citations_list, Iterable) and not isinstance(
                    citations_list, (str, bytes)
                ):
                    for index in citations_list:
                        try:
                            value = int(index)
                        except (TypeError, ValueError):
                            continue
                        if value > 0:
                            used_indexes.append(value)

        ordered = sorted({index for index in used_indexes if 0 < index <= len(citations)})
        citation_mapping: dict[int, int] = {
            old: new for new, old in enumerate(ordered, start=1)
        }
        new_citations = [copy.deepcopy(citations[old - 1]) for old in ordered]

        remapped_sections: list[ConsolidatedSection] = []
        sentences_removed = 0
        for section in sections:
            sentences: list[str] = []
            section_indexes: set[int] = set()
            for sentence in section.sentences:
                indexes = [
                    citation_mapping.get(int(match.group(1)))
                    for match in cls._CITATION_PATTERN.finditer(sentence)
                    if match.group(1).isdigit()
                ]
                indexes = [index for index in indexes if index]
                if not indexes:
                    sentences_removed += 1
                    continue
                sentences.append(cls._remap_sentence(sentence, citation_mapping))
                section_indexes.update(indexes)
            if sentences:
                remapped_sections.append(
                    ConsolidatedSection(
                        title=section.title,
                        sentences=sentences,
                        citation_indexes=sorted(section_indexes),
                    )
                )

        remapped_conflicts: list[ConflictNote] = []
        for note in conflicts:
            mapped_note_indexes = sorted(
                {
                    citation_mapping[index]
                    for index in note.citation_indexes
                    if index in citation_mapping
                }
            )
            variants: list[dict[str, Any]] = []
            for variant in note.variants:
                mapped_variant_indexes = sorted(
                    {
                        citation_mapping[index]
                        for index in variant.get("citations", [])
                        if index in citation_mapping
                    }
                )
                if mapped_variant_indexes:
                    variants.append(
                        {
                            "text": variant.get("text", ""),
                            "citations": mapped_variant_indexes,
                        }
                    )
            if len(variants) >= 2 and mapped_note_indexes:
                remapped_conflicts.append(
                    ConflictNote(
                        summary=note.summary,
                        citation_indexes=mapped_note_indexes,
                        variants=variants,
                    )
                )

        section_usage: dict[int, set[str]] = {}
        for section in remapped_sections:
            for index in section.citation_indexes:
                section_usage.setdefault(index, set()).add(section.title)

        text = ConversationManager._assemble_answer_text(remapped_sections, remapped_conflicts)
        consolidation = ConsolidationOutput(
            text=text,
            sections=remapped_sections,
            conflicts=remapped_conflicts,
            section_usage=section_usage,
        )

        return _JudgeFixResult(
            consolidation=consolidation,
            citations=new_citations,
            citation_mapping=citation_mapping,
            duplicates_removed=0,
            sentences_removed=sentences_removed,
        )

    @staticmethod
    def _sanitize_consolidation_without_citations(
        consolidation: ConsolidationOutput,
    ) -> ConsolidationOutput:
        sections: list[ConsolidatedSection] = []
        for section in consolidation.sections:
            cleaned_sentences: list[str] = []
            for sentence in section.sentences:
                cleaned = ConversationManager._strip_citation_markers(sentence).strip()
                if cleaned:
                    cleaned_sentences.append(cleaned)
            if cleaned_sentences:
                sections.append(
                    ConsolidatedSection(
                        title=section.title,
                        sentences=cleaned_sentences,
                        citation_indexes=[],
                    )
                )

        text = ConversationManager._assemble_answer_text(sections, [])
        return ConsolidationOutput(
            text=text,
            sections=sections,
            conflicts=[],
            section_usage={},
        )

    @classmethod
    def _remap_sentence(
        cls, sentence: str, citation_mapping: dict[int, int]
    ) -> str:
        def replace(match: re.Match[str]) -> str:
            value = int(match.group(1)) if match.group(1).isdigit() else 0
            mapped = citation_mapping.get(value)
            return f"[{mapped}]" if mapped else ""

        updated = cls._CITATION_PATTERN.sub(replace, sentence)
        updated = " ".join(updated.split())
        return updated


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

        if getattr(result, "insufficient", False):
            return

        raw_text = (result.answer or "").strip()
        if ConversationManager._text_declares_insufficient_evidence(raw_text):
            return
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


class _PlanCritic:
    """Reject plan steps that are vague, overlapping, or non-executable."""

    _STEP_PATTERN = re.compile(
        r"^Input: (?P<input>.+?) → Action: (?P<action>.+?) → Output: (?P<output>.+)$"
    )
    _APPROVED_VERBS = {
        "analyze",
        "assess",
        "check",
        "collect",
        "compare",
        "determine",
        "evaluate",
        "extract",
        "gather",
        "identify",
        "list",
        "map",
        "profile",
        "trace",
        "verify",
    }
    _BANNED_TOKENS = {
        "answer",
        "compose",
        "explain",
        "execute",
        "final",
        "plan",
        "respond",
        "solution",
        "summarize",
        "synthesize",
        "write",
    }
    _ARTIFACT_KEYWORDS = {
        "list",
        "table",
        "matrix",
        "map",
        "note",
        "profile",
        "summary",
        "timeline",
        "catalog",
        "finding",
    }
    _EVIDENCE_KEYWORDS = {"citation", "source", "document", "evidence", "snippet"}

    def ensure(self, plan: Sequence[PlanItem]) -> None:
        approved, reasons = self.review(plan)
        if not approved:
            logger.debug(
                "Plan critic rejection",
                extra={
                    "step_count": len(plan),
                    "issues": reasons,
                },
            )
            reason_text = ", ".join(reasons)
            raise DynamicPlanningError(f"Plan critic rejection: {reason_text}")

    def review(self, plan: Sequence[PlanItem]) -> tuple[bool, list[str]]:
        logger.debug(
            "Reviewing plan for quality",
            extra={"step_count": len(plan)},
        )
        issues: list[str] = []
        if not plan:
            logger.debug("Plan review failed: empty plan")
            return False, [self._format_issue("PLAN_EMPTY", "Plan has no steps.", "Generate 2–5 atomic plan steps before execution.")]

        if len(plan) < 2:
            issues.append(
                self._format_issue(
                    "PLAN_SIZE",
                    f"Plan only contains {len(plan)} step.",
                    "Create at least two steps covering evidence gathering and synthesis inputs.",
                )
            )
        if len(plan) > 10:
            issues.append(
                self._format_issue(
                    "PLAN_SIZE",
                    f"Plan contains {len(plan)} steps which exceeds reviewer allowance.",
                    "Trim to the highest value 2–10 atomic steps or group related subtasks upstream.",
                )
            )

        seen_artifacts: set[str] = set()

        for index, item in enumerate(plan, start=1):
            description = (item.description or "").strip()
            if not description:
                issues.append(
                    self._format_issue(
                        "NO_DESCRIPTION",
                        f"Step {index} is blank.",
                        "Provide a full 'Input → Action → Output' description.",
                    )
                )
                continue

            match = self._STEP_PATTERN.match(description)
            if not match:
                issues.append(
                    self._format_issue(
                        "STRUCTURE",
                        f"Step {index} is not structured as 'Input → Action → Output'.",
                        "Reformat as 'Input: <source> → Action: <verb target> → Output: <artifact>'.",
                    )
                )
                continue

            input_section = match.group("input").strip()
            action = match.group("action").strip()
            output = match.group("output").strip()

            if not input_section:
                issues.append(
                    self._format_issue(
                        "NO_INPUT",
                        f"Step {index} does not declare an input.",
                        "Specify the evidence scope or prior artifact feeding the step.",
                    )
                )

            verb, _, remainder = action.partition(" ")
            verb_lower = verb.lower()
            remainder_lower = remainder.strip().lower()

            if verb_lower not in self._APPROVED_VERBS:
                issues.append(
                    self._format_issue(
                        "UNSUPPORTED_VERB",
                        f"Step {index} uses unsupported action verb '{verb_lower}'.",
                        f"Start the action with one approved verb such as {sorted(self._APPROVED_VERBS)}.",
                    )
                )

            lowered_action = action.lower()
            if any(token in lowered_action for token in self._BANNED_TOKENS):
                issues.append(
                    self._format_issue(
                        "META",
                        f"Step {index} contains banned meta language.",
                        "Remove response-oriented verbs such as 'write' or 'summarize'.",
                    )
                )

            if remainder_lower:
                connector_issue = self._detect_connector_issue(verb_lower, remainder_lower)
                if connector_issue is not None:
                    issues.append(
                        self._format_issue(
                            "NON_ATOMIC",
                            f"Step {index} bundles multiple actions ({connector_issue}).",
                            "Split into separate steps so each action yields one artifact.",
                        )
                    )
                if remainder_lower.startswith("the question") or remainder_lower == "question":
                    issues.append(
                        self._format_issue(
                            "RESTATES_QUESTION",
                            f"Step {index} simply restates the user question.",
                            "Target a concrete concept or evidence set instead of repeating the ask.",
                        )
                    )
            else:
                issues.append(
                    self._format_issue(
                        "NO_TARGET",
                        f"Step {index} is missing a concrete target.",
                        "Describe the focus (e.g., metrics, claims, sections) after the verb.",
                    )
                )

            artifact_key = remainder_lower or lowered_action
            if artifact_key in seen_artifacts:
                issues.append(
                    self._format_issue(
                        "OVERLAP",
                        f"Step {index} duplicates an earlier artifact.",
                        "Merge overlapping steps or adjust the target to cover new ground.",
                    )
                )
            else:
                seen_artifacts.add(artifact_key)

            normalized_output = output.lower()
            if not normalized_output:
                issues.append(
                    self._format_issue(
                        "NO_ARTIFACT",
                        f"Step {index} output is empty.",
                        "Declare a tangible artifact such as a list, table, or mapping.",
                    )
                )
            elif not any(
                keyword in normalized_output for keyword in self._ARTIFACT_KEYWORDS
            ):
                issues.append(
                    self._format_issue(
                        "NO_ARTIFACT",
                        f"Step {index} output lacks a concrete artifact description.",
                        "Describe the deliverable (list, table, map, profile, etc.).",
                    )
                )

            if not normalized_output or not any(
                keyword in normalized_output for keyword in self._EVIDENCE_KEYWORDS
            ):
                issues.append(
                    self._format_issue(
                        "NO_EVIDENCE",
                        f"Step {index} output does not specify how evidence or citations will be recorded.",
                        "Include a note that the artifact stores citations or document references.",
                    )
                )

            referenced_steps = self._extract_referenced_steps(input_section)
            for ref in referenced_steps:
                if ref >= index:
                    issues.append(
                        self._format_issue(
                            "ORDER_ERROR",
                            f"Step {index} references Step {ref}, which is not yet available.",
                            "Reorder steps or adjust inputs so dependencies flow forward.",
                        )
                    )

        if issues:
            logger.debug(
                "Plan review identified issues",
                extra={
                    "step_count": len(plan),
                    "issue_count": len(issues),
                },
            )
        else:
            logger.debug(
                "Plan approved by critic",
                extra={"step_count": len(plan)},
            )
        return len(issues) == 0, issues

    @staticmethod
    def _format_issue(code: str, message: str, fix: str) -> str:
        return f"{code}: {message} (Fix: {fix})"

    @staticmethod
    def _detect_connector_issue(verb: str, remainder_lower: str) -> str | None:
        if any(token in remainder_lower for token in {";", " then ", " after ", " before "}):
            return "sequence connector detected"
        if " and " in remainder_lower and verb not in {"compare", "contrast"}:
            return "contains conjunction 'and'"
        if "," in remainder_lower and verb not in {"compare", "contrast"}:
            return "comma-separated targets"
        return None

    @staticmethod
    def _extract_referenced_steps(input_section: str) -> set[int]:
        refs: set[int] = set()
        for match in re.finditer(r"step\s*(\d+)", input_section.lower()):
            try:
                refs.add(int(match.group(1)))
            except ValueError:
                continue
        return refs


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
        self._judge = _AdversarialJudge()
        self._plan_critic = _PlanCritic()
        self._judge_log: deque[dict[str, Any]] = deque(maxlen=50)

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

    @log_call(logger=logger, level=logging.DEBUG, include_args=False)
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

        sanitized_question = question.strip()
        preview = sanitized_question[:120]
        logger.info(
            "Received question",
            extra={
                "question_preview": preview,
                "preset": preset.value,
                "response_mode": response_mode.value,
                "reasoning_verbosity": getattr(reasoning_verbosity, "value", None),
                "context_snippet_count": len(context_snippets or []),
                "planning_enabled": context_provider is not None,
            },
        )
        logger.debug(
            "Prepared conversation context",
            extra={
                "turn_index": len(self.turns) + 1,
                "question_length": len(sanitized_question),
                "context_lengths": [
                    len(snippet or "") for snippet in (context_snippets or [])
                ],
                "extra_option_keys": sorted((extra_options or {}).keys()),
            },
        )

        if not self._connected:
            message = self._connection_error or "LMStudio is disconnected."
            raise LMStudioConnectionError(message)

        if context_provider is not None:
            logger.debug(
                "Invoking dynamic planning pipeline",
                extra={
                    "question_preview": preview,
                    "context_provider": getattr(context_provider, "__name__", None),
                },
            )
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
            except DynamicPlanningError as exc:
                logger.warning(
                    "Dynamic planning failed, falling back to single-shot",
                    extra={
                        "question_preview": preview,
                        "error": str(exc),
                    },
                )
                turn = self._ask_single_shot(
                    question,
                    context_snippets=context_snippets,
                    preset=preset,
                    reasoning_verbosity=reasoning_verbosity,
                    response_mode=response_mode,
                    extra_options=extra_options,
                )
        else:
            logger.debug(
                "Dynamic planning disabled; using single-shot mode",
                extra={"question_preview": preview},
            )
            turn = self._ask_single_shot(
                question,
                context_snippets=context_snippets,
                preset=preset,
                reasoning_verbosity=reasoning_verbosity,
                response_mode=response_mode,
                extra_options=extra_options,
            )
        logger.info(
            "Completed question",
            extra={
                "question_preview": preview,
                "response_mode": response_mode.value,
                "plan_step_count": len(turn.plan),
                "step_result_count": len(getattr(turn, "step_results", [])),
                "citation_count": len(turn.citations),
            },
        )
        return turn

    @log_call(logger=logger, level=logging.DEBUG, include_args=False)
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
        question_preview = question.strip()[:120]
        messages = self._build_messages(question, context_snippets)
        logger.debug(
            "Prepared single-shot messages",
            extra={
                "question_preview": question_preview,
                "message_roles": [message.get("role") for message in messages],
                "context_included": bool(context_snippets),
                "context_lengths": [len(snippet or "") for snippet in (context_snippets or [])],
            },
        )
        logger.info(
            "Dispatching single-shot query",
            extra={
                "question_preview": question_preview,
                "message_count": len(messages),
                "context_snippet_count": len(context_snippets or []),
                "preset": preset.value,
                "response_mode": response_mode.value,
            },
        )
        request_options = self._build_request_options(
            question,
            reasoning_verbosity,
            response_mode,
            extra_options,
        )
        logger.debug(
            "Prepared single-shot request options",
            extra={
                "question_preview": question_preview,
                "option_keys": sorted((request_options or {}).keys()),
                "reasoning_verbosity": getattr(reasoning_verbosity, "value", None),
            },
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

        logger.debug(
            "Received single-shot response",
            extra={
                "question_preview": question_preview,
                "answer_length": len(response.content or ""),
                "citation_count": len(response.citations or []),
                "reasoning_keys": sorted((response.reasoning or {}).keys())
                if isinstance(response.reasoning, dict)
                else None,
            },
        )
        self._update_connection(True, None)
        turn = self._register_turn(question, response, response_mode, preset)
        logger.info(
            "Single-shot query completed",
            extra={
                "question_preview": question_preview,
                "citation_count": len(turn.citations),
                "answer_length": len(turn.answer),
            },
        )
        return turn

    def _apply_adversarial_review(
        self,
        consolidation: ConsolidationOutput,
        citations: list[Any],
        *,
        ledger_snapshot: Sequence[dict[str, Any]] | None,
        scope: dict[str, Any] | None,
        preset: AnswerLength,
        response_mode: ResponseMode,
    ) -> tuple[ConsolidationOutput, list[Any], dict[int, int], JudgeReport]:
        original_consolidation = copy.deepcopy(consolidation)
        cycles = 0
        revisions = False
        duplicates_removed_total = 0
        sentences_removed_total = 0
        mapping_overall: dict[int, int] = {
            index: index for index in range(1, len(citations) + 1)
        }
        last_verdict: _JudgeVerdict | None = None

        while True:
            cycles += 1
            verdict = self._judge.review(
                consolidation,
                citations,
                ledger_snapshot=ledger_snapshot,
                scope=scope,
                answer_length=preset,
                response_mode=response_mode,
            )
            last_verdict = verdict
            logger.info(
                "Adversarial review cycle",
                extra={
                    "cycle": cycles,
                    "decision": verdict.decision,
                    "reason_codes": verdict.reason_codes,
                },
            )
            if verdict.decision == "publish":
                break
            if verdict.decision == "insufficient_evidence":
                sanitized = self._judge._sanitize_consolidation_without_citations(
                    original_consolidation
                )
                if sanitized.text:
                    consolidation = sanitized
                else:
                    consolidation = ConsolidationOutput(
                        text="No supporting evidence in current scope",
                        sections=[],
                        conflicts=[],
                        section_usage={},
                    )
                citations = []
                mapping_overall = {}
                revisions = True
                break
            if verdict.decision == "replan":
                revisions = True
                break
            if cycles >= self._judge.max_cycles:
                metrics = verdict.metrics if verdict else {}
                sanitized = self._judge._sanitize_consolidation_without_citations(
                    original_consolidation
                )
                if sanitized.text:
                    consolidation = sanitized
                else:
                    consolidation = ConsolidationOutput(
                        text="No supporting evidence in current scope",
                        sections=[],
                        conflicts=[],
                        section_usage={},
                    )
                citations = []
                mapping_overall = {}
                last_verdict = _JudgeVerdict(
                    decision="insufficient_evidence",
                    reasons=["No supporting evidence in current scope"],
                    reason_codes=["insufficient_evidence"],
                    metrics=metrics,
                )
                revisions = True
                break

            previous_citation_count = len(citations)
            fix = self._judge.apply_fixes(
                consolidation,
                citations,
                verdict=verdict,
                ledger_snapshot=ledger_snapshot,
            )
            consolidation = fix.consolidation
            citations = fix.citations
            duplicates_removed_total += fix.duplicates_removed
            sentences_removed_total += fix.sentences_removed
            logger.info(
                "Applied adversarial fixes",
                extra={
                    "cycle": cycles,
                    "duplicates_removed": fix.duplicates_removed,
                    "sentences_removed": fix.sentences_removed,
                    "citation_count": len(citations),
                },
            )
            if (
                fix.duplicates_removed > 0
                or fix.sentences_removed > 0
                or len(citations) != previous_citation_count
            ):
                revisions = True

            if mapping_overall:
                composed: dict[int, int] = {}
                for original, current in mapping_overall.items():
                    if current in fix.citation_mapping:
                        composed[original] = fix.citation_mapping[current]
                mapping_overall = composed
            else:
                mapping_overall = {
                    index: value
                    for index, value in fix.citation_mapping.items()
                    if value is not None
                }

        if last_verdict is None:
            last_verdict = _JudgeVerdict(
                decision="publish",
                reasons=[],
                reason_codes=[],
                metrics={
                    "retrieved_k": len(citations),
                    "unique_claims": 0,
                    "duplicates_identified": 0,
                    "citations_total": 0,
                    "citations_verified_ok": 0,
                    "conflict_notes": 0,
                },
            )

        metrics = dict(last_verdict.metrics)
        metrics["dup_claims_removed"] = duplicates_removed_total
        metrics["sentences_removed"] = sentences_removed_total

        log_entry = {
            "retrieved_k": metrics.get("retrieved_k", len(citations)),
            "unique_claims": metrics.get("unique_claims", 0),
            "dup_claims_removed": duplicates_removed_total,
            "citations_verified_ok": metrics.get("citations_verified_ok", 0),
            "citations_total": metrics.get("citations_total", 0),
            "conflict_notes_count": metrics.get("conflict_notes", 0),
            "reason_codes": list(last_verdict.reason_codes),
            "cycles_to_approve": cycles,
        }
        self._judge_log.append(log_entry)

        final_decision = last_verdict.decision
        if final_decision not in {"publish", "insufficient_evidence", "replan"}:
            final_decision = "publish"
        report = JudgeReport(
            decision=final_decision,
            revised=revisions,
            reasons=list(last_verdict.reasons),
            reason_codes=list(last_verdict.reason_codes),
            metrics=metrics,
            cycles=cycles,
        )

        return consolidation, citations, mapping_overall, report

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

    @log_call(logger=logger, level=logging.DEBUG, include_args=False)
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
        normalized_question = question.strip()
        question_preview = normalized_question[:120]
        plan_items = self._generate_plan(question)
        if not plan_items:
            raise DynamicPlanningError("No plan items generated")

        total_steps = len(plan_items)
        turn_id = len(self.turns) + 1
        self._ledger.clear_turn(turn_id)
        shared_context = "\n\n".join(context_snippets or [])
        base_options = copy.deepcopy(extra_options) if extra_options else {}
        logger.debug(
            "Dynamic plan details",
            extra={
                "question_preview": question_preview,
                "plan_step_count": total_steps,
                "plan_descriptions": [item.description for item in plan_items],
                "plan_has_rationales": [bool(item.rationale) for item in plan_items],
                "base_option_keys": sorted(base_options.keys()),
                "shared_context_length": len(shared_context),
            },
        )
        executed_plan: list[PlanItem] = [
            PlanItem(
                description=item.description,
                status="queued",
                rationale=item.rationale,
            )
            for item in plan_items
        ]
        step_results: list[StepResult] = []
        assumptions: list[str] = []

        logger.info(
            "Executing dynamic plan",
            extra={
                "question_preview": question_preview,
                "plan_step_count": total_steps,
                "plan_descriptions": [item.description for item in plan_items],
            },
        )

        for index, plan_item in enumerate(executed_plan, start=1):
            plan_item.status = "running"
            try:
                batches = list(context_provider(plan_item, index, total_steps))
            except Exception as exc:  # pragma: no cover - fallback to single shot
                raise DynamicPlanningError("Context provider failed") from exc
            if not batches:
                batches = [StepContextBatch(snippets=[], documents=[])]
            logger.debug(
                "Prepared plan step batches",
                extra={
                    "step_index": index,
                    "batch_count": len(batches),
                    "snippet_totals": [len(batch.snippets) for batch in batches],
                    "document_totals": [len(batch.documents) for batch in batches],
                },
            )

            answer_parts: list[str] = []
            citations: list[Any] = []
            used_contexts: list[StepContextBatch] = []

            for pass_index, batch in enumerate(batches, start=1):
                used_contexts.append(batch)
                combined_snippets: list[str] = []
                if shared_context:
                    combined_snippets.append(shared_context)
                combined_snippets.extend(batch.snippets)
                logger.debug(
                    "Compiled plan step context",
                    extra={
                        "step_index": index,
                        "pass_index": pass_index,
                        "combined_snippet_count": len(combined_snippets),
                        "snippet_lengths": [len(snippet or "") for snippet in combined_snippets],
                        "document_ids": [
                            doc.get("id")
                            for doc in (batch.documents or [])
                            if isinstance(doc, dict)
                        ],
                    },
                )
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
                logger.debug(
                    "Prepared plan step request",
                    extra={
                        "step_index": index,
                        "pass_index": pass_index,
                        "message_roles": [message.get("role") for message in messages],
                        "option_keys": sorted((merged_options or {}).keys()),
                        "reasoning_verbosity": getattr(reasoning_verbosity, "value", None),
                    },
                )
                logger.info(
                    "Dispatching plan step",
                    extra={
                        "step_index": index,
                        "pass_index": pass_index,
                        "total_steps": total_steps,
                        "question_preview": question_preview,
                        "description": plan_item.description,
                        "context_snippet_count": len(combined_snippets),
                        "document_count": len(batch.documents),
                    },
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
                logger.debug(
                    "Received plan step response payload",
                    extra={
                        "step_index": index,
                        "pass_index": pass_index,
                        "answer_length": len(text),
                        "citation_count": len(response.citations or []),
                        "reasoning_keys": sorted((response.reasoning or {}).keys())
                        if isinstance(response.reasoning, dict)
                        else None,
                    },
                )
                logger.info(
                    "Received plan step response",
                    extra={
                        "step_index": index,
                        "pass_index": pass_index,
                        "answer_length": len(text),
                        "citation_count": len(response.citations or []),
                    },
                )

            if not answer_parts:
                message = (
                    "INSUFFICIENT_EVIDENCE: No corpus snippet supported "
                    f"step {index} ({plan_item.description})."
                )
                answer_parts.append(message)
                assumptions.append(message)
                logger.info(
                    "Plan step marked insufficient",
                    extra={
                        "step_index": index,
                        "description": plan_item.description,
                    },
                )
            plan_item.status = "done"
            combined_answer = "\n\n".join(answer_parts).strip()
            insufficient = self._text_declares_insufficient_evidence(combined_answer)
            result = StepResult(
                index=index,
                description=plan_item.description,
                answer=combined_answer,
                citations=self._deduplicate_citations(citations),
                contexts=used_contexts,
                insufficient=insufficient,
            )
            logger.debug(
                "Compiled plan step result",
                extra={
                    "step_index": index,
                    "answer_length": len(combined_answer),
                    "insufficient": insufficient,
                    "citation_count": len(result.citations),
                },
            )
            if result.insufficient:
                if combined_answer not in assumptions:
                    assumptions.append(combined_answer)
                result.citations = []
                logger.info(
                    "Removed citations from insufficient step",
                    extra={
                        "step_index": index,
                        "assumption_count": len(assumptions),
                    },
                )
            elif not result.citations:
                inferred = self._fallback_citations_from_contexts(used_contexts)
                if inferred:
                    result.citations = inferred
                else:
                    assumptions.append(
                        f"No citations available for step {index}: {plan_item.description}"
                    )
                    logger.info(
                        "Recorded citation assumption",
                        extra={
                            "step_index": index,
                            "description": plan_item.description,
                        },
                    )
            step_results.append(result)

            logger.info(
                "Plan step completed",
                extra={
                    "step_index": index,
                    "description": plan_item.description,
                    "insufficient": result.insufficient,
                    "citation_count": len(result.citations),
                },
            )
            logger.debug(
                "Plan step ledger snapshot",
                extra={
                    "step_index": index,
                    "assumption_count": len(assumptions),
                    "contexts_recorded": len(used_contexts),
                },
            )

        aggregated, citation_index_map = self._aggregate_citations(step_results)
        logger.debug(
            "Aggregated plan step citations",
            extra={
                "question_preview": question_preview,
                "unique_citations": len(aggregated),
                "citation_index_map_size": len(citation_index_map),
            },
        )
        logger.info(
            "Aggregated dynamic plan citations",
            extra={
                "unique_citation_count": len(aggregated),
                "step_count": len(step_results),
                "assumption_count": len(assumptions),
            },
        )
        for result in step_results:
            indexes = self._collect_citation_indexes(result.citations, citation_index_map)
            result.citation_indexes = indexes
            if not result.insufficient:
                self._ledger.record_step(turn_id, result)

        consolidation = self._compose_final_answer(
            step_results, ledger=self._ledger, turn_id=turn_id
        )
        retrieval_scope = None
        if extra_options and isinstance(extra_options.get("retrieval"), dict):
            retrieval_scope = copy.deepcopy(extra_options["retrieval"])
        ledger_snapshot_initial = self._ledger.snapshot_for_turn(turn_id)
        logger.debug(
            "Invoking adversarial review",
            extra={
                "question_preview": question_preview,
                "initial_citation_count": len(aggregated),
                "step_count": len(step_results),
                "retrieval_scope_keys": sorted((retrieval_scope or {}).keys())
                if isinstance(retrieval_scope, dict)
                else None,
            },
        )
        (
            consolidation,
            aggregated,
            citation_mapping,
            review_report,
        ) = self._apply_adversarial_review(
            consolidation,
            aggregated,
            ledger_snapshot=ledger_snapshot_initial,
            scope=retrieval_scope,
            preset=preset,
            response_mode=response_mode,
        )
        logger.debug(
            "Adversarial review outcome",
            extra={
                "question_preview": question_preview,
                "revised_citation_count": len(aggregated),
                "citation_mapping_size": len(citation_mapping),
                "review_cycles": review_report.cycles,
                "revised": review_report.revised,
            },
        )
        logger.info(
            "Adversarial review completed",
            extra={
                "decision": review_report.decision,
                "revised": review_report.revised,
                "cycles": review_report.cycles,
                "reason_codes": review_report.reason_codes,
            },
        )

        if citation_mapping:
            for result in step_results:
                remapped = [
                    citation_mapping[index]
                    for index in result.citation_indexes
                    if index in citation_mapping and citation_mapping[index] > 0
                ]
                result.citation_indexes = sorted(dict.fromkeys(remapped))
        else:
            for result in step_results:
                result.citation_indexes = []

        self._ledger.clear_turn(turn_id)
        for result in step_results:
            self._ledger.record_step(turn_id, result)

        ledger_snapshot = self._ledger.snapshot_for_turn(turn_id)
        answer = consolidation.text
        summary = f"Executed {total_steps} dynamic step{'s' if total_steps != 1 else ''}."
        artifacts = ReasoningArtifacts(
            summary_bullets=[summary],
            plan_items=executed_plan,
            assumptions=assumptions,
        )
        reasoning_payload = {
            "plan": [
                {
                    "description": item.description,
                    "status": item.status,
                    **({"rationale": item.rationale} if item.rationale else {}),
                }
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
            adversarial_review=review_report,
        )
        self.turns.append(turn)
        logger.info(
            "Dynamic plan completed",
            extra={
                "question_preview": question_preview,
                "steps": total_steps,
                "final_citation_count": len(turn.citations),
                "assumption_count": len(assumptions),
                "conflict_count": len(consolidation.conflicts),
            },
        )
        return turn

    @log_call(logger=logger, level=logging.DEBUG, include_args=False)
    def _generate_plan(self, question: str) -> list[PlanItem]:
        normalized = question.strip()
        if not normalized:
            return []

        logger.debug(
            "Generating dynamic plan",
            extra={
                "question_preview": normalized[:120],
                "question_length": len(normalized),
            },
        )
        actions = self._split_question_into_actions(normalized)
        plan: list[PlanItem] = []

        keyword_list = self._extract_plan_keywords(normalized)
        logger.debug(
            "Plan keyword extraction",
            extra={
                "question_preview": normalized[:120],
                "keyword_count": len(keyword_list),
                "actions_detected": len(actions),
            },
        )
        if keyword_list:
            keyword_text = " ".join(keyword_list)
            plan.append(
                self._build_structured_plan_item(
                    input_hint="Corpus context",
                    verb="collect",
                    target=f"background references on {keyword_text}",
                    output_hint="Background reference list (up to 5 entries) with source titles and citation-ready locations",
                    rationale=self._compose_background_rationale(keyword_list),
                )
            )

        for action in actions:
            normalized_action = self._normalize_plan_action(action)
            if normalized_action is None:
                logger.debug(
                    "Discarded unstructured action",
                    extra={"action": action},
                )
                continue
            verb, target = normalized_action
            input_hint = "Corpus context" if not plan else "Targeted corpus snippets for this step"
            plan.append(
                self._build_structured_plan_item(
                    input_hint=input_hint,
                    verb=verb,
                    target=target,
                    output_hint=self._describe_output_artifact(verb, target),
                )
            )

        if len(plan) > 8:
            plan = plan[:8]
        if not plan:
            plan = [
                PlanItem(
                    description=normalized,
                    status="queued",
                    rationale="Address the user's question directly using the available corpus context.",
                )
            ]

        self._plan_critic.ensure(plan)
        logger.info(
            "Generated plan",
            extra={
                "question_preview": normalized[:120],
                "plan_step_count": len(plan),
                "plan_descriptions": [item.description for item in plan],
            },
        )
        logger.debug(
            "Dynamic plan construction complete",
            extra={
                "plan_step_count": len(plan),
                "plan_has_rationales": [bool(item.rationale) for item in plan],
                "actions_considered": len(actions),
            },
        )
        return plan

    @staticmethod
    def _compose_action(verb: str, target: str) -> str:
        cleaned_target = re.sub(r"\s+", " ", target).strip().rstrip("?.!")
        if not cleaned_target:
            return verb.capitalize()
        return f"{verb.capitalize()} {cleaned_target}"

    def _normalize_plan_action(self, action: str) -> tuple[str, str] | None:
        text = action.strip()
        if not text:
            return None

        text = re.sub(
            r"^(?:please|kindly|could you|would you|let's|lets|we need to|need to|i need to|i want to|should)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        if not text:
            return None

        words = text.split()
        if not words:
            return None

        verb_raw = words[0].lower()
        remainder_words = words[1:]

        verb = self._canonicalize_action_verb(verb_raw)
        if verb is None:
            verb = "identify"
            remainder_words = words[1:] if len(words) > 1 else []

        target = " ".join(remainder_words).strip()
        target = self._clean_action_target(target)
        if not target:
            return None
        return verb, target

    _PLAN_VERB_SYNONYMS = {
        "analyze": "analyze",
        "assess": "assess",
        "audit": "evaluate",
        "categorize": "map",
        "check": "check",
        "collect": "collect",
        "compare": "compare",
        "compile": "collect",
        "contrast": "compare",
        "determine": "determine",
        "develop": "determine",
        "document": "list",
        "explain": "identify",
        "evaluate": "evaluate",
        "examine": "analyze",
        "extract": "extract",
        "gather": "gather",
        "highlight": "identify",
        "identify": "identify",
        "investigate": "collect",
        "list": "list",
        "map": "map",
        "outline": "identify",
        "profile": "profile",
        "recommend": "determine",
        "research": "collect",
        "review": "gather",
        "study": "analyze",
        "summarize": "identify",
        "trace": "trace",
        "verify": "verify",
    }

    def _canonicalize_action_verb(self, verb: str) -> str | None:
        canonical = self._PLAN_VERB_SYNONYMS.get(verb)
        if canonical in _PlanCritic._APPROVED_VERBS:
            return canonical
        return None

    @staticmethod
    def _clean_action_target(target: str) -> str:
        cleaned = re.sub(r"\s+", " ", target or "").strip()
        cleaned = cleaned.strip(" ,;:-")
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _describe_output_artifact(self, verb: str, target: str) -> str:
        normalized_target = re.sub(r"\s+", " ", target).strip()
        lower_target = normalized_target.lower()
        if verb in {"compare", "analyze", "evaluate", "assess"}:
            return (
                f"Comparison table for {normalized_target} with cited source snippets"
            )
        if verb in {"check", "determine", "verify"}:
            return (
                f"Verification note on {normalized_target} referencing supporting evidence"
            )
        if verb in {"map", "trace"}:
            return (
                f"Mapping of {normalized_target} paired with citation references"
            )
        if verb == "profile":
            return f"Profile of {normalized_target} annotated with source citations"
        if verb == "list" and "timeline" in lower_target:
            return f"Timeline list covering {normalized_target} with citation references"
        return f"Bullet list of {normalized_target} with document citations"

    def _split_question_into_actions(self, question: str) -> list[str]:
        """Break a question into granular, executable action strings."""

        segments = re.split(r"[\n\.!?]+", question)
        actions: list[str] = []
        for segment in segments:
            text = segment.strip()
            if len(text) < 3:
                continue
            actions.extend(self._split_segment_on_connectors(text))
        filtered = [action for action in actions if action]
        logger.debug(
            "Split question into actions",
            extra={
                "segment_count": len(segments),
                "action_count": len(filtered),
                "actions": filtered,
            },
        )
        return filtered

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
        logger.debug(
            "Split segment on connectors",
            extra={
                "original_segment": segment,
                "part_count": len(normalized_parts),
                "parts": normalized_parts,
            },
        )
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
            if (
                lower in self._PLAN_STOPWORDS
                or lower in _PlanCritic._APPROVED_VERBS
                or lower in self._PLAN_VERB_SYNONYMS
                or lower in seen
            ):
                continue
            seen.add(lower)
            keywords.append(token)
            if len(keywords) >= 3:
                break
        return keywords

    def _build_structured_plan_item(
        self,
        *,
        input_hint: str,
        verb: str,
        target: str,
        output_hint: str,
        rationale: str | None = None,
    ) -> PlanItem:
        action = self._compose_action(verb, target)
        description = (
            f"Input: {input_hint.strip()} (retrieved corpus snippets with chunk IDs) → "
            f"Action: {action.strip()} grounded strictly in those snippets → "
            f"Output: {output_hint.strip()} (include chunk IDs used)"
        )
        rationale_text = rationale or self._compose_plan_rationale(
            input_hint=input_hint,
            verb=verb,
            target=target,
            output_hint=output_hint,
        )
        return PlanItem(description=description, status="queued", rationale=rationale_text)

    def _compose_plan_rationale(
        self,
        *,
        input_hint: str,
        verb: str,
        target: str,
        output_hint: str,
    ) -> str:
        action_clause = self._normalize_plan_action_clause(verb, target)
        input_clause = self._normalize_plan_input_hint(input_hint)
        output_clause = self._normalize_plan_output_hint(output_hint)
        return f"Use {input_clause} to {action_clause} so the final answer can deliver {output_clause}."

    @staticmethod
    def _compose_background_rationale(keywords: Sequence[str]) -> str:
        cleaned = [token.strip() for token in keywords if token.strip()]
        if not cleaned:
            return (
                "Collect background references so subsequent steps stay grounded in the corpus."
            )
        if len(cleaned) == 1:
            topics = cleaned[0]
        elif len(cleaned) == 2:
            topics = " and ".join(cleaned)
        else:
            topics = ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"
        return (
            f"Collect background references on {topics} so later steps stay anchored to the key topics."
        )

    @staticmethod
    def _normalize_plan_action_clause(verb: str, target: str) -> str:
        verb_clean = re.sub(r"\s+", " ", (verb or "").strip()).lower()
        target_clean = re.sub(r"\s+", " ", (target or "").strip())
        if not verb_clean:
            return target_clean.lower() or "address the request"
        if not target_clean:
            return f"{verb_clean} the requested details"
        return f"{verb_clean} {target_clean}".strip()

    @staticmethod
    def _normalize_plan_input_hint(text: str) -> str:
        cleaned = " ".join((text or "").strip().split())
        if not cleaned:
            return "the available corpus context"
        lowered = cleaned[0].lower() + cleaned[1:]
        if re.match(r"^(?:the|a|an|any|all)\b", lowered):
            return lowered
        return f"the {lowered}"

    @staticmethod
    def _normalize_plan_output_hint(text: str) -> str:
        cleaned = " ".join((text or "").strip().split())
        if not cleaned:
            return "a grounded response"
        lowered = cleaned[0].lower() + cleaned[1:]
        if re.match(r"^(?:the|a|an)\b", lowered):
            return lowered
        article = "an" if re.match(r"^[aeiou]", lowered) else "a"
        return f"{article} {lowered}"

    def _register_turn(
        self,
        question: str,
        response: ChatMessage,
        response_mode: ResponseMode,
        preset: AnswerLength,
    ) -> ConversationTurn:
        artifacts = self._parse_reasoning_artifacts(response.reasoning)
        answer = self._ensure_answer_citation_markers(
            response.content, response.citations
        )

        turn = ConversationTurn(
            question=question,
            answer=answer,
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
    def _ensure_answer_citation_markers(
        answer: str, citations: Sequence[Any]
    ) -> str:
        """Ensure single-shot answers reference returned citations."""

        if not answer or not citations:
            return answer

        if ConversationManager._NUMERIC_CITATION_PATTERN.search(answer):
            return answer

        fragments: list[dict[str, Any]] = []
        for match in ConversationManager._SENTENCE_FRAGMENT_PATTERN.finditer(answer):
            raw = match.group(0)
            stripped = raw.strip()
            if not stripped:
                continue
            leading_len = len(raw) - len(raw.lstrip())
            trailing_len = len(raw) - len(raw.rstrip())
            leading = raw[:leading_len]
            trailing = raw[len(raw) - trailing_len :] if trailing_len else ""
            core = (
                raw[leading_len:-trailing_len]
                if trailing_len
                else raw[leading_len:]
            )
            fragments.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "leading": leading,
                    "core": core,
                    "trailing": trailing,
                    "text": stripped,
                }
            )

        if not fragments:
            return answer

        token_pattern = re.compile(r"[A-Za-z0-9']+")

        def tokenize(text: str) -> set[str]:
            return {
                token
                for token in token_pattern.findall(text.lower())
                if len(token) > 2
            }

        def citation_text(citation: Any) -> str:
            if isinstance(citation, dict):
                snippet = citation.get("snippet") or citation.get("text")
                if isinstance(snippet, str) and snippet.strip():
                    snippet_plain = html.unescape(snippet)
                    snippet_plain = re.sub(r"<[^>]+>", " ", snippet_plain)
                    snippet_plain = " ".join(snippet_plain.split())
                    if snippet_plain:
                        return snippet_plain
                label = citation.get("label") or citation.get("source")
                if isinstance(label, str) and label.strip():
                    return label.strip()
                path = citation.get("path")
                if isinstance(path, str) and path.strip():
                    return path.strip()
            return str(citation)

        fragment_tokens: list[set[str]] = []
        fragment_lower: list[str] = []
        for fragment in fragments:
            fragment_lower.append(fragment["text"].lower())
            fragment_tokens.append(tokenize(fragment["text"]))

        citation_infos: list[dict[str, Any]] = []
        for index, citation in enumerate(citations, start=1):
            text = citation_text(citation).strip()
            citation_infos.append(
                {
                    "index": index,
                    "text": text,
                    "lower": text.lower(),
                    "tokens": tokenize(text),
                }
            )

        if not citation_infos:
            return answer

        scores: list[list[float]] = []
        for fragment_index, fragment in enumerate(fragments):
            fragment_scores: list[float] = []
            fragment_token_set = fragment_tokens[fragment_index]
            fragment_lower_text = fragment_lower[fragment_index]
            for info in citation_infos:
                tokens = info["tokens"]
                overlap = len(fragment_token_set & tokens)
                ratio = overlap / max(len(tokens), 1)
                snippet_match = 0.0
                snippet = info["lower"]
                if snippet and snippet in fragment_lower_text:
                    snippet_match = max(len(tokens), 1) * 2.0
                elif tokens:
                    partial_matches = sum(
                        1 for token in tokens if token in fragment_lower_text
                    )
                    snippet_match = partial_matches * 0.5
                semantic_similarity = 0.0
                if snippet:
                    semantic_similarity = (
                        SequenceMatcher(None, fragment_lower_text, snippet).ratio()
                        * max(len(tokens), 1)
                    )
                fragment_scores.append(overlap + ratio + snippet_match + semantic_similarity)
            scores.append(fragment_scores)

        assignments: list[list[int]] = [[] for _ in fragments]
        citation_used = [False] * len(citation_infos)

        for citation_index, info in enumerate(citation_infos):
            best_fragment = max(
                range(len(fragments)),
                key=lambda idx: scores[idx][citation_index],
                default=None,
            )
            if best_fragment is None:
                continue
            best_score = scores[best_fragment][citation_index]
            if best_score > 0:
                assignments[best_fragment].append(info["index"])
                citation_used[citation_index] = True

        for fragment_index in range(len(fragments)):
            if assignments[fragment_index]:
                continue
            best_citation = max(
                range(len(citation_infos)),
                key=lambda idx: scores[fragment_index][idx],
                default=None,
            )
            if best_citation is None:
                continue
            assignments[fragment_index].append(
                citation_infos[best_citation]["index"]
            )
            citation_used[best_citation] = True

        for citation_index, used in enumerate(citation_used):
            if used:
                continue
            best_fragment = max(
                range(len(fragments)),
                key=lambda idx: scores[idx][citation_index],
                default=None,
            )
            if best_fragment is None:
                continue
            citation_number = citation_infos[citation_index]["index"]
            if citation_number not in assignments[best_fragment]:
                assignments[best_fragment].append(citation_number)

        rebuilt: list[str] = []
        last_end = 0
        for fragment_index, fragment in enumerate(fragments):
            start = fragment["start"]
            end = fragment["end"]
            rebuilt.append(answer[last_end:start])
            markers = "".join(
                f"[{number}]" for number in sorted(dict.fromkeys(assignments[fragment_index]))
            )
            core = fragment["core"]
            if markers:
                base_end = len(core)
                while base_end > 0 and core[base_end - 1] in "\"'”’)]":
                    base_end -= 1
                base = core[:base_end]
                tail = core[base_end:]
                if base and base[-1].isalnum():
                    core_with_markers = f"{base} {markers}{tail}"
                else:
                    core_with_markers = f"{base}{markers}{tail}"
            else:
                core_with_markers = core
            rebuilt.append(fragment["leading"] + core_with_markers + fragment["trailing"])
            last_end = end
        rebuilt.append(answer[last_end:])

        return "".join(rebuilt)

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
        instructions = textwrap.dedent(
            """Instructions:
- Use only the corpus snippets provided in the Context section.
- Tie every claim to the supporting snippet identifiers (e.g., "DOC17 chunk 3") using plain parentheses.
- Do not rely on outside knowledge or speculate beyond what the snippets state.
- If the snippets cannot answer the step, reply exactly with "INSUFFICIENT_EVIDENCE: <brief reason>".
- Limit the response to at most two sentences and do not include bracketed citation markers; they will be added later."""
        ).strip()
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
        logger.debug(
            "Starting citation aggregation",
            extra={
                "step_count": len(step_results),
                "total_citation_candidates": sum(
                    len(result.citations) for result in step_results
                ),
            },
        )
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
        logger.debug(
            "Completed citation aggregation",
            extra={
                "unique_citations": len(aggregated),
                "index_map_size": len(index_map),
            },
        )
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
        sorted_indexes = sorted(indexes)
        logger.debug(
            "Resolved citation indexes",
            extra={
                "citation_count": len(citations),
                "resolved_indexes": sorted_indexes,
            },
        )
        return sorted_indexes

    _NUMERIC_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
    _CITATION_PATTERN = re.compile(r"\[(?:\d+|[a-zA-Z]+)\]")
    _DOC_REFERENCE_PAREN_PATTERN = re.compile(r"\(([^()]*)\)")
    _SENTENCE_FRAGMENT_PATTERN = re.compile(r"[^.!?\n]+[.!?]?")
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
        without_brackets = ConversationManager._CITATION_PATTERN.sub("", text)

        def _maybe_strip_parenthetical(match: re.Match[str]) -> str:
            content = match.group(1).strip()
            if not content:
                return ""
            lowered = content.lower()
            lowered = re.sub(
                r"(?:doc\s*-?\d+|doc\d+|chunk\s*\d+|section\s*\d+|snippet\s*\d+|source\s*\d+|r(?:eq)?\s*-?(?:\d+[\da-z]*))",
                "",
                lowered,
            )
            lowered = re.sub(r"[\s,;:&/\\-]+", "", lowered)
            if not lowered:
                return ""
            return match.group(0)

        return ConversationManager._DOC_REFERENCE_PAREN_PATTERN.sub(
            _maybe_strip_parenthetical, without_brackets
        )

    @staticmethod
    def _remove_step_prefix(text: str) -> str:
        """Remove leading step/number markers used in intermediate outputs."""

        cleaned = ConversationManager._STEP_PREFIX_PATTERN.sub("", text, count=1)
        cleaned = cleaned.lstrip(" \t-–—:.)]")
        return cleaned.lstrip()

    @staticmethod
    def _text_declares_insufficient_evidence(text: str) -> bool:
        if not text:
            return False
        stripped = ConversationManager._strip_citation_markers(text).strip()
        normalized = ConversationManager._remove_step_prefix(stripped)
        return normalized.upper().startswith("INSUFFICIENT_EVIDENCE")

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
            if getattr(result, "insufficient", False):
                continue
            raw_text = (result.answer or "").strip()
            if ConversationManager._text_declares_insufficient_evidence(raw_text):
                continue
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

        final_text = ConversationManager._assemble_answer_text(sections, conflict_notes)

        return ConsolidationOutput(
            text=final_text,
            sections=sections,
            conflicts=conflict_notes,
            section_usage=section_usage,
        )

    @staticmethod
    def _assemble_answer_text(
        sections: Sequence[ConsolidatedSection],
        conflicts: Sequence[ConflictNote],
    ) -> str:
        sections_text: list[str] = []
        for section in sections:
            body = " ".join(section.sentences).strip()
            if body:
                sections_text.append(f"{section.title}: {body}")

        conflict_note_text: list[str] = []
        if conflicts:
            summaries: list[str] = []
            citation_union: set[int] = set()
            for note in conflicts:
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
        return "\n\n".join(parts).strip()

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
        snippet = ConversationManager._build_snippet_html(snippet_text)
        if not snippet:
            return None

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
    def _build_snippet_html(snippet: str, *, max_chars: int = 320) -> str:
        parser = _SnippetMarkParser()
        parser.feed(snippet)
        parser.close()
        plain_text = parser.text
        if not plain_text.strip():
            return ""

        sentences = ConversationManager._sentence_spans_for_snippet(plain_text)
        if not sentences:
            normalized = " ".join(plain_text.split())
            if not normalized:
                return ""
            escaped = html.escape(normalized)
            if parser.mark_ranges:
                return f"<mark>{escaped}</mark>"
            return escaped

        highlight_indices: set[int] = set()
        for index, (start, end, _text) in enumerate(sentences):
            for mark_start, mark_end in parser.mark_ranges:
                if mark_start < end and mark_end > start:
                    highlight_indices.add(index)
                    break
        if parser.mark_ranges and not highlight_indices:
            highlight_indices = {idx for idx in range(len(sentences))}

        selected: set[int] = set(highlight_indices)
        if not selected:
            selected.add(0)

        def snippet_length(indices: set[int]) -> int:
            if not indices:
                return 0
            parts: list[str] = []
            previous_index: int | None = None
            for idx in sorted(indices):
                if previous_index is not None and idx - previous_index > 1:
                    parts.append("…")
                parts.append(sentences[idx][2])
                previous_index = idx
            return len(" ".join(parts))

        left_index = min(selected) - 1
        right_index = max(selected) + 1
        while True:
            expanded = False
            if left_index >= 0:
                candidate = set(selected)
                candidate.add(left_index)
                if snippet_length(candidate) <= max_chars:
                    selected = candidate
                    left_index -= 1
                    expanded = True
                else:
                    left_index = -1
            if right_index < len(sentences):
                candidate = set(selected)
                candidate.add(right_index)
                if snippet_length(candidate) <= max_chars:
                    selected = candidate
                    right_index += 1
                    expanded = True
                else:
                    right_index = len(sentences)
            if not expanded:
                break

        parts: list[str] = []
        previous_index: int | None = None
        mark_sentences = bool(parser.mark_ranges)
        for idx in sorted(selected):
            if previous_index is not None and idx - previous_index > 1:
                parts.append("…")
            text = sentences[idx][2]
            escaped = html.escape(text)
            if mark_sentences and idx in highlight_indices:
                parts.append(f"<mark>{escaped}</mark>")
            else:
                parts.append(escaped)
            previous_index = idx

        return " ".join(parts).strip()

    @staticmethod
    def _sentence_spans_for_snippet(text: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        length = len(text)
        start = 0
        position = 0
        while position < length:
            character = text[position]
            if character in ".!?":
                end = position + 1
                while end < length and text[end] in " \t":
                    end += 1
                trimmed_start = start
                while trimmed_start < end and text[trimmed_start].isspace():
                    trimmed_start += 1
                trimmed_end = end
                while trimmed_end > trimmed_start and text[trimmed_end - 1].isspace():
                    trimmed_end -= 1
                segment = text[trimmed_start:trimmed_end]
                normalized = " ".join(segment.split())
                if normalized:
                    spans.append((trimmed_start, trimmed_end, normalized))
                start = end
                position = end
                continue
            if character == "\n":
                trimmed_start = start
                trimmed_end = position
                while trimmed_start < trimmed_end and text[trimmed_start].isspace():
                    trimmed_start += 1
                while trimmed_end > trimmed_start and text[trimmed_end - 1].isspace():
                    trimmed_end -= 1
                segment = text[trimmed_start:trimmed_end]
                normalized = " ".join(segment.split())
                if normalized:
                    spans.append((trimmed_start, trimmed_end, normalized))
                start = position + 1
            position += 1

        if start < length:
            trimmed_start = start
            trimmed_end = length
            while trimmed_start < trimmed_end and text[trimmed_start].isspace():
                trimmed_start += 1
            while trimmed_end > trimmed_start and text[trimmed_end - 1].isspace():
                trimmed_end -= 1
            segment = text[trimmed_start:trimmed_end]
            normalized = " ".join(segment.split())
            if normalized:
                spans.append((trimmed_start, trimmed_end, normalized))

        return spans

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
                    rationale_list: list[str] = []
                    for key in ("rationale", "reason", "thought", "explanation", "why"):
                        rationale_list.extend(
                            ConversationManager._coerce_str_list(entry.get(key))
                        )
                    rationale = rationale_list[0] if rationale_list else None
                elif isinstance(entry, str):
                    description = entry.strip()
                    status = "pending"
                    rationale = None
                else:
                    continue
                if description:
                    plan_items.append(
                        PlanItem(description=description, status=status, rationale=rationale)
                    )

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
            instructions = "\n".join(
                [
                    "Instructions:",
                    "- Use only the provided corpus context snippets to answer.",
                    "- If the context lacks the necessary details, reply exactly with \"INSUFFICIENT_EVIDENCE: <brief reason>\".",
                    "- Do not rely on outside knowledge or speculate beyond the snippets.",
                ]
            )
            user_content = (
                f"{question}\n\nContext:\n{formatted_context}\n\n{instructions}"
            )
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

