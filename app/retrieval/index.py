"""Vector-based retrieval helpers for semantic passage search."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import math


def _normalize_vector(values: Sequence[float]) -> tuple[float, ...]:
    """Return a unit-length vector derived from ``values``."""

    if not values:
        raise ValueError("Embedding vectors must contain at least one value")
    floats = tuple(float(value) for value in values)
    length = math.sqrt(sum(component * component for component in floats))
    if length == 0:
        raise ValueError("Embedding vectors must have non-zero magnitude")
    return tuple(component / length for component in floats)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity for pre-normalised vectors."""

    return float(sum(a * b for a, b in zip(left, right)))


def _normalise_folder(folder: str | Path | None) -> str | None:
    if folder in (None, ""):
        return None
    return str(Path(folder).resolve())


@dataclass(slots=True)
class Passage:
    """Container for retrievable passages and their metadata."""

    passage_id: str
    document_id: int
    text: str
    embedding: Sequence[float]
    tags: Iterable[int] = ()
    folder: str | Path | None = None
    created_at: datetime | None = None
    language: str | None = None
    page: int | None = None
    section: str | None = None
    metadata: Mapping[str, Any] | None = None
    _normalized_embedding: tuple[float, ...] = field(init=False, repr=False)
    _normalized_text: str = field(init=False, repr=False)
    _normalized_folder: str | None = field(init=False, repr=False)
    _language: str | None = field(init=False, repr=False)
    _tags: frozenset[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:  # noqa: D401 - documented on class
        self._normalized_embedding = _normalize_vector(self.embedding)
        self._normalized_text = " ".join(self.text.split())
        self._normalized_folder = _normalise_folder(self.folder)
        self._language = self.language.lower() if self.language else None
        self._tags = frozenset(int(tag) for tag in self.tags)
        self.metadata = dict(self.metadata or {})

    @property
    def normalized_embedding(self) -> tuple[float, ...]:
        return self._normalized_embedding

    @property
    def normalized_folder(self) -> str | None:
        return self._normalized_folder

    @property
    def normalized_language(self) -> str | None:
        return self._language

    @property
    def tag_set(self) -> frozenset[int]:
        return self._tags

    @property
    def deduplication_key(self) -> tuple[int, str | None, int | None, str]:
        return (
            self.document_id,
            self.section,
            self.page,
            self._normalized_text.lower(),
        )


@dataclass(slots=True)
class RetrievalScope:
    """Describe constraints applied to a retrieval query."""

    tags: Iterable[int] | None = None
    folder: str | Path | None = None
    recursive: bool = True
    start_time: datetime | None = None
    end_time: datetime | None = None
    languages: Iterable[str] | None = None
    _tags: frozenset[int] = field(init=False, repr=False)
    _folder: str | None = field(init=False, repr=False)
    _languages: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:  # noqa: D401 - documented on class
        self.tags = tuple(self.tags or [])
        self.languages = tuple(self.languages or [])
        self._tags = frozenset(int(tag) for tag in self.tags)
        self._folder = _normalise_folder(self.folder)
        self._languages = frozenset(lang.lower() for lang in self.languages)

    def matches(self, passage: Passage) -> bool:
        if self._tags and not self._tags.issubset(passage.tag_set):
            return False
        if self._folder is not None:
            folder = passage.normalized_folder
            if folder is None:
                return False
            if self.recursive:
                if not folder.startswith(self._folder):
                    return False
            else:
                if folder != self._folder:
                    return False
        if self.start_time and (
            passage.created_at is None or passage.created_at < self.start_time
        ):
            return False
        if self.end_time and (
            passage.created_at is None or passage.created_at > self.end_time
        ):
            return False
        if self._languages and (
            passage.normalized_language is None
            or passage.normalized_language not in self._languages
        ):
            return False
        return True


@dataclass(slots=True)
class Citation:
    """Map a passage back to its original document location."""

    document_id: int
    page: int | None = None
    section: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "page": self.page,
            "section": self.section,
        }


@dataclass(slots=True)
class RetrievedPassage:
    """Result payload returned from the retrieval engine."""

    passage: Passage
    score: float
    preview: str
    citation: Citation
    conflicts: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        record = {
            "passage_id": self.passage.passage_id,
            "document_id": self.passage.document_id,
            "text": self.passage.text,
            "score": self.score,
            "preview": self.preview,
            "citation": self.citation.to_dict(),
            "metadata": dict(self.passage.metadata),
            "conflicts": self.conflicts,
        }
        if self.passage.normalized_language is not None:
            record["language"] = self.passage.normalized_language
        if self.passage.tag_set:
            record["tags"] = sorted(self.passage.tag_set)
        if self.passage.normalized_folder is not None:
            record["folder"] = self.passage.normalized_folder
        if self.passage.created_at is not None:
            record["created_at"] = self.passage.created_at
        return record


class RetrievalIndex:
    """Perform semantic retrieval across stored passages."""

    def __init__(self) -> None:
        self._passages: dict[str, Passage] = {}

    def add_passage(self, passage: Passage) -> None:
        """Store or replace a passage in the vector index."""

        self._passages[passage.passage_id] = passage

    def remove_passage(self, passage_id: str) -> None:
        self._passages.pop(passage_id, None)

    def clear(self) -> None:
        self._passages.clear()

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        diversity: float = 0.3,
        scope: RetrievalScope | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top ``k`` passages respecting ``scope`` constraints."""

        if top_k <= 0:
            return []
        normalized_query = _normalize_vector(query_embedding)
        scope = scope or RetrievalScope()
        filtered = [
            passage
            for passage in self._passages.values()
            if scope.matches(passage)
        ]
        if not filtered:
            return []

        scored = self._score_passages(normalized_query, filtered)
        if not scored:
            return []

        candidates = sorted(scored, key=lambda item: item.score, reverse=True)
        selected = self._apply_diversity_filter(candidates, top_k, diversity)
        conflicts = self._detect_conflicts([item.passage for item in selected])
        results = [
            RetrievedPassage(
                passage=item.passage,
                score=item.score,
                preview=item.preview,
                citation=Citation(
                    document_id=item.passage.document_id,
                    page=item.passage.page,
                    section=item.passage.section,
                ),
                conflicts=conflicts.get(item.passage.passage_id, []),
            ).to_dict()
            for item in selected
        ]
        return results

    def _score_passages(
        self,
        normalized_query: Sequence[float],
        passages: Iterable[Passage],
    ) -> list[_ScoredPassage]:
        deduplicated: dict[tuple[int, str | None, int | None, str], _ScoredPassage] = {}
        for passage in passages:
            score = _cosine_similarity(normalized_query, passage.normalized_embedding)
            preview = self._build_preview(passage.text)
            key = passage.deduplication_key
            current = deduplicated.get(key)
            candidate = _ScoredPassage(passage=passage, score=score, preview=preview)
            if current is None or candidate.score > current.score:
                deduplicated[key] = candidate
        return list(deduplicated.values())

    def _apply_diversity_filter(
        self,
        candidates: list[_ScoredPassage],
        top_k: int,
        diversity: float,
    ) -> list[_ScoredPassage]:
        if not candidates:
            return []
        if diversity <= 0:
            return candidates[:top_k]
        diversity = min(max(diversity, 0.0), 1.0)
        selected: list[_ScoredPassage] = []
        pool = candidates.copy()
        while pool and len(selected) < top_k:
            if not selected:
                selected.append(pool.pop(0))
                continue
            best_index = 0
            best_value = float("-inf")
            for index, candidate in enumerate(pool):
                similarity = max(
                    _cosine_similarity(
                        candidate.passage.normalized_embedding,
                        chosen.passage.normalized_embedding,
                    )
                    for chosen in selected
                )
                value = (1 - diversity) * candidate.score - diversity * similarity
                if value > best_value:
                    best_value = value
                    best_index = index
            selected.append(pool.pop(best_index))
        return selected

    def _detect_conflicts(
        self, passages: Iterable[Passage]
    ) -> dict[str, list[dict[str, Any]]]:
        buckets: dict[str, dict[str, list[Passage]]] = {}
        for passage in passages:
            metadata = passage.metadata or {}
            statement = str(metadata.get("statement", "")).strip().lower()
            stance = str(metadata.get("stance", "")).strip().lower()
            if not statement or not stance:
                continue
            stances = buckets.setdefault(statement, {})
            stances.setdefault(stance, []).append(passage)

        conflicts: dict[str, list[dict[str, Any]]] = {}
        for stances in buckets.values():
            if len(stances) <= 1:
                continue
            for stance, passages_for_stance in stances.items():
                opposing = [
                    {
                        "passage_id": other.passage_id,
                        "stance": other.metadata.get("stance"),
                        "document_id": other.document_id,
                    }
                    for other_stance, others in stances.items()
                    if other_stance != stance
                    for other in others
                ]
                if not opposing:
                    continue
                for passage in passages_for_stance:
                    conflicts.setdefault(passage.passage_id, []).extend(opposing)
        return conflicts

    @staticmethod
    def _build_preview(text: str, *, limit: int = 280) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
            return cleaned
        cutoff = cleaned.rfind(" ", 0, limit)
        if cutoff == -1:
            cutoff = limit
        return cleaned[:cutoff].rstrip() + "…"


@dataclass(slots=True)
class _ScoredPassage:
    passage: Passage
    score: float
    preview: str

