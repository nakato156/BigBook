"""The ranker: scoring/diversify/explore config and logic (the "score -> diversify"
half of ``retrieve -> score -> diversify -> explain``). Candidate retrieval lives in
:mod:`src.reduction.retrieval`; the orchestrating :class:`Recommender` lives in
:mod:`src.reduction.recommend`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankingConfig:
    """Tunables for the four fixes. Defaults are the v1 decisions in the alcance doc."""

    # A1: tabular early axes excluded from the interest cosine (popularity/lang/missingness).
    tabular_pcs: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    k: int = 10
    # A3: slots reserved for relevant tail/mid books outside the retrieved macros.
    explore_slots: int = 2
    popularity_tail_quantile: float = 0.25
    popularity_head_quantile: float = 0.90
    # Exploration candidates must retain this fraction of the best interest similarity.
    explore_min_relevance_ratio: float = 0.75
    # Diversity/relevance trade-off for MMR (1.0 = pure relevance, 0.0 = pure diversity).
    mmr_lambda: float = 0.7
    # Explicit penalty when a candidate repeats genres already present in the list.
    genre_diversity_weight: float = 0.15
    # Soft tie-break toward shorter eligible books in the normal interest ranking.
    accessibility_weight: float = 0.05
    # How many nearest fine clusters to pull candidates from (retrieve breadth).
    n_clusters_retrieve: int = 5
    # Opt-in per-mode retrieve (E1/E6 fix): each mode keeps its own top clusters instead of
    # max-pooling. None preserves today's exact behaviour (retrieve_top_clusters).
    clusters_per_mode: int | None = None
    # Final cap on the round-robin union when clusters_per_mode is set; None = no cap.
    retrieve_budget: int | None = None
    # A4: accessibility floor so the cold-start sampler does not surface pamphlets.
    min_pages_accessible: int = 50
    # Users with 1-2 positives keep their profile, shrunk toward the nearest catalog cluster.
    sparse_profile_weight: float = 0.5


@dataclass(frozen=True)
class HybridV12Weights:
    """Percentile-calibrated weights for the V1.2 union ranker."""

    content: float = 0.35
    global_popularity: float = 0.30
    genre_popularity: float = 0.20
    cooccurrence: float = 0.10
    user_knn: float = 0.05
    duplicate_title_penalty: float = 0.05

    @classmethod
    def from_mapping(cls, values: dict[str, float] | None) -> "HybridV12Weights":
        if values is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown hybrid_v12 weight keys: {sorted(unknown)}")
        return cls(**{key: float(value) for key, value in values.items()})


def mmr_select(
    cand_norm: np.ndarray,
    relevance: np.ndarray,
    k: int,
    lam: float,
    candidate_genres: np.ndarray | None = None,
    genre_weight: float = 0.0,
) -> list[int]:
    """Maximal Marginal Relevance over the candidate set (diversity vs. relevance).

    ``cand_norm`` are L2-normalized candidate vectors in the **taste subspace**, so the
    redundancy penalty is itself popularity-free (A1). Greedy, ``O(k·n·d)``; no popularity
    enters here, only interest similarity, pairwise redundancy and optional genre overlap.
    """
    n = len(relevance)
    if n == 0 or k <= 0:
        return []
    selected: list[int] = []
    remaining = list(range(n))
    max_sim = np.zeros(n, dtype=np.float64)
    while remaining and len(selected) < k:
        if not selected:
            pick = remaining[int(np.argmax(relevance[remaining]))]
        else:
            mmr = lam * relevance[remaining] - (1.0 - lam) * max_sim[remaining]
            if candidate_genres is not None and genre_weight > 0:
                selected_genres = candidate_genres[selected].max(axis=0)
                overlap = candidate_genres[remaining] @ selected_genres
                denom = np.maximum(candidate_genres[remaining].sum(axis=1), 1.0)
                mmr -= genre_weight * (overlap / denom)
            pick = remaining[int(np.argmax(mmr))]
        selected.append(pick)
        remaining.remove(pick)
        sims = cand_norm @ cand_norm[pick]
        max_sim = np.maximum(max_sim, sims)
    return selected


def accessibility_scores(num_pages: np.ndarray, min_pages: int) -> np.ndarray:
    """Soft accessibility score: shorter valid books rank higher; missing/too short get zero."""
    pages = np.asarray(num_pages, dtype=np.float64)
    valid = np.isfinite(pages) & (pages >= min_pages)
    scores = np.zeros(len(pages), dtype=np.float64)
    if not valid.any():
        return scores
    log_pages = np.log1p(pages[valid])
    low, high = float(log_pages.min()), float(log_pages.max())
    scores[valid] = 1.0 if high == low else 1.0 - ((log_pages - low) / (high - low))
    return scores


def select_exploration_rows(
    rows: np.ndarray,
    relevance: np.ndarray,
    popularity_segment: np.ndarray,
    k: int,
    best_relevance: float,
    min_relevance_ratio: float,
) -> np.ndarray:
    """A3: select relevant exploration books from tail/mid only.

    Head books are never eligible for exploration. Tail is preferred over mid among
    candidates that pass the relevance floor. Returned values are catalog row ids.
    """
    if not len(rows) or k <= 0:
        return np.array([], dtype=np.int64)
    floor = best_relevance * min_relevance_ratio if best_relevance > 0 else best_relevance
    segments = popularity_segment[rows]
    keep = (relevance >= floor) & np.isin(segments, ["tail", "mid"])
    if not keep.any():
        return np.array([], dtype=np.int64)
    eligible_rows = rows[keep]
    eligible_rel = relevance[keep]
    priority = {"tail": 0, "mid": 1}
    segment_priority = np.array(
        [priority.get(str(popularity_segment[row]), 3) for row in eligible_rows]
    )
    order = np.lexsort((-eligible_rel, segment_priority))
    return eligible_rows[order[:k]].astype(np.int64)
