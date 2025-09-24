from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval import Passage, RetrievalIndex, RetrievalScope


def _vec(*values: float) -> tuple[float, ...]:
    return tuple(values)


def test_multi_source_retrieval_with_conflict_detection() -> None:
    index = RetrievalIndex()
    now = datetime.now(UTC)
    base_folder = Path("reports").resolve()

    index.add_passage(
        Passage(
            passage_id="alpha",
            document_id=1,
            text="Alpha report highlights the rapid adoption of solar storage systems.",
            embedding=_vec(1.0, 0.0),
            tags={1, 2},
            folder=base_folder / "alpha",
            created_at=now,
            language="EN",
            page=3,
            section="overview",
            metadata={"statement": "solar storage is viable", "stance": "support"},
        )
    )
    index.add_passage(
        Passage(
            passage_id="beta",
            document_id=2,
            text="Beta dossier questions the economic viability of solar storage deployments.",
            embedding=_vec(0.9, 0.1),
            tags={2},
            folder=base_folder / "beta",
            created_at=now - timedelta(minutes=5),
            language="en",
            page=7,
            section="analysis",
            metadata={"statement": "solar storage is viable", "stance": "challenge"},
        )
    )
    index.add_passage(
        Passage(
            passage_id="gamma",
            document_id=3,
            text="Gamma memo discusses fossil fuel subsidies with no mention of renewables.",
            embedding=_vec(-1.0, 0.0),
            tags={3},
            folder=base_folder / "gamma",
            created_at=now,
            language="en",
            page=2,
            section="context",
        )
    )

    results = index.search(_vec(1.0, 0.0), top_k=2, diversity=0.2)

    assert [entry["document_id"] for entry in results] == [1, 2]
    assert results[0]["citation"]["page"] == 3
    conflict_ids = {
        conflict["passage_id"] for conflict in results[0]["conflicts"]
    }
    assert "beta" in conflict_ids


def test_scope_filters_trim_results(tmp_path: Path) -> None:
    index = RetrievalIndex()
    now = datetime.now(UTC)
    scope_folder = tmp_path / "reports"
    deep_folder = scope_folder / "deep"

    index.add_passage(
        Passage(
            passage_id="scoped",
            document_id=10,
            text="Scoped insight on market forecasts for solar deployments.",
            embedding=_vec(0.8, 0.2),
            tags={5},
            folder=scope_folder,
            created_at=now,
            language="en",
            page=1,
            section="summary",
        )
    )
    index.add_passage(
        Passage(
            passage_id="too_old",
            document_id=11,
            text="Outdated analysis that should be removed by the time range filter.",
            embedding=_vec(0.7, 0.3),
            tags={5},
            folder=deep_folder,
            created_at=now - timedelta(days=5),
            language="en",
            page=4,
            section="history",
        )
    )
    index.add_passage(
        Passage(
            passage_id="wrong_language",
            document_id=12,
            text="Analyse du marché solaire écrite en français uniquement.",
            embedding=_vec(0.75, 0.25),
            tags={5},
            folder=scope_folder,
            created_at=now,
            language="fr",
            page=2,
            section="fr",
        )
    )

    scope = RetrievalScope(
        tags={5},
        folder=scope_folder,
        recursive=False,
        start_time=now - timedelta(days=1),
        end_time=now + timedelta(days=1),
        languages={"en"},
    )

    results = index.search(_vec(0.8, 0.2), top_k=3, scope=scope)

    assert [entry["passage_id"] for entry in results] == ["scoped"]


def test_duplicate_passages_are_suppressed() -> None:
    index = RetrievalIndex()
    now = datetime.now(UTC)

    index.add_passage(
        Passage(
            passage_id="dup_a",
            document_id=20,
            text="Repeated finding about long-duration storage performance.",
            embedding=_vec(0.6, 0.4),
            tags=set(),
            folder="notes",
            created_at=now,
            language="en",
            page=1,
            section="insight",
        )
    )
    index.add_passage(
        Passage(
            passage_id="dup_b",
            document_id=20,
            text="Repeated finding about long-duration storage performance.",
            embedding=_vec(0.62, 0.38),
            tags=set(),
            folder="notes",
            created_at=now,
            language="en",
            page=1,
            section="insight",
        )
    )

    results = index.search(_vec(0.61, 0.39), top_k=5, diversity=0.0)

    returned_ids = [entry["passage_id"] for entry in results]
    assert returned_ids == ["dup_b"]

