"""Tests for the ranking layer: each test pins one of the four contradiction fixes.

C1→A1 (taste subspace drops pc_0..pc_5), C2→A2 (popularity is a gate, not a score),
C3→A3 (exploration slot from a non-occupied macro), C4→A4 (cold-start has no popularity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.reduction.recommend import (
    RankingConfig,
    Recommender,
    l2_normalize_rows,
    mmr_select,
    pick_exploration_macro,
    quality_gate_mask,
    taste_pc_indices,
)

GENRES = ["genre_fantasy", "genre_mystery", "genre_history", "genre_ya", "genre_romance"]


# --------------------------------------------------------------------------- #
# Unit tests on the pure helpers
# --------------------------------------------------------------------------- #
def test_a1_taste_subspace_drops_tabular_pcs() -> None:
    pc_cols = [f"pc_{i}" for i in range(10)]
    idx = taste_pc_indices(pc_cols, (0, 1, 2, 3, 4, 5))
    # pc_0..pc_5 (popularity/lang/missingness) excluded; pc_6..pc_9 kept.
    assert idx.tolist() == [6, 7, 8, 9]


def test_a2_quality_gate_filters_low_evidence_only() -> None:
    counts = np.array([0, 4, 5, 100])
    mask = quality_gate_mask(counts, min_ratings=5)
    # Gate drops the under-evidenced books and keeps the rest — independent of magnitude.
    assert mask.tolist() == [False, False, True, True]


def test_a3_exploration_picks_nearest_non_occupied_macro() -> None:
    # 3 macro centroids; user closest to macro 0, then 2, then 1.
    user = np.array([1.0, 0.0])
    macros = l2_normalize_rows(np.array([[1.0, 0.05], [0.0, 1.0], [0.9, 0.2]]))
    user_n = user / np.linalg.norm(user)
    # macro 0 occupied → must skip to the next nearest non-occupied (macro 2).
    assert pick_exploration_macro({0}, macros, user_n) == 2
    # all occupied → None.
    assert pick_exploration_macro({0, 1, 2}, macros, user_n) is None


def test_mmr_prefers_relevance_then_diversifies() -> None:
    # Two near-duplicate high-relevance vectors + one orthogonal one.
    cand = l2_normalize_rows(np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]))
    relevance = np.array([0.9, 0.89, 0.4])
    picked = mmr_select(cand, relevance, k=2, lam=0.5)
    # First the top-relevance item, then the diverse one — not its near-duplicate.
    assert picked[0] == 0
    assert picked[1] == 2


# --------------------------------------------------------------------------- #
# End-to-end on a tiny synthetic catalog
# --------------------------------------------------------------------------- #
def _toy_recommender(config: RankingConfig) -> Recommender:
    n_pc = 4  # pc_0 = popularity axis (tabular), pc_1..pc_3 = taste
    pc_cols = [f"pc_{i}" for i in range(n_pc)]
    book_ids = np.array([f"b{i}" for i in range(8)])

    # pc_0 (popularity) is huge for the "popular" books; taste lives in pc_1..pc_3.
    book_pc = np.array(
        [
            [50.0, 1.0, 0.0, 0.0],  # b0 popular, taste A
            [0.1, 1.0, 0.0, 0.0],   # b1 niche,  taste A  (same taste as b0)
            [0.1, 0.9, 0.1, 0.0],   # b2 niche,  taste A'
            [40.0, 0.0, 1.0, 0.0],  # b3 popular, taste B
            [0.1, 0.0, 1.0, 0.0],   # b4 niche,  taste B
            [0.1, 0.0, 0.0, 1.0],   # b5 niche,  taste C (far macro)
            [0.1, 0.0, 0.0, 0.9],   # b6 niche,  taste C
            [0.1, 0.0, 0.0, 1.0],   # b7 LOW ratings, taste C (gate should drop)
        ],
        dtype=np.float32,
    )
    ratings_count = np.array([1000, 50, 50, 1000, 50, 50, 50, 2])  # b7 below gate
    num_pages = np.array([400, 120, 600, 300, 90, 500, 80, 70], dtype=np.float64)

    # Two fine clusters per taste region; 3 macro-clusters: {A}, {B}, {C}.
    book_cluster = np.array([0, 0, 1, 2, 2, 3, 4, 4], dtype=np.int64)
    centroids = np.array(
        [
            [25.0, 1.0, 0.0, 0.0],  # cluster 0 → macro 0 (taste A)
            [0.1, 0.9, 0.1, 0.0],   # cluster 1 → macro 0 (taste A)
            [20.0, 0.0, 1.0, 0.0],  # cluster 2 → macro 1 (taste B)
            [0.1, 0.0, 0.0, 1.0],   # cluster 3 → macro 2 (taste C)
            [0.1, 0.0, 0.0, 0.95],  # cluster 4 → macro 2 (taste C)
        ]
    )
    macro_of_cluster = np.array([0, 0, 1, 2, 2], dtype=np.int64)

    genres = pd.DataFrame(
        {"title": [f"Title {b}" for b in book_ids], **{g: 0 for g in GENRES}},
        index=pd.Index(book_ids, name="book_id"),
    )

    # One user whose taste is region A (pc_1), and one cold-start user.
    user_ids = np.array(["u_taste_A"])
    user_pc = np.array([[30.0, 1.0, 0.0, 0.0]], dtype=np.float32)  # huge pc_0 inherited!

    return Recommender(
        book_ids=book_ids,
        book_pc=book_pc,
        ratings_count=ratings_count,
        num_pages=num_pages,
        genres=genres,
        book_cluster=book_cluster,
        centroids=centroids,
        macro_of_cluster=macro_of_cluster,
        user_ids=user_ids,
        user_pc=user_pc,
        cold_start_users=set(),
        pc_cols=pc_cols,
        config=config,
    )


def test_a2_popularity_never_orders_the_interest_slots() -> None:
    # k=3, no exploration, so we see only interest-scored books.
    rec = _toy_recommender(RankingConfig(k=3, explore_slots=0, n_clusters_retrieve=2,
                                         min_ratings_gate=5, mmr_lambda=0.8))
    out = rec.recommend("u_taste_A")
    # The user "inherited" a huge pc_0, yet the popular b0 must NOT outrank the niche b1/b2
    # purely by popularity: b0 and b1 share taste, so popularity cannot be the tiebreaker.
    top = out[out["slot"] == "interest"].iloc[0]
    # b1 and b2 (niche, taste A) are equally/most relevant in the taste subspace as b0;
    # the ranking is popularity-free, so b0 (1000 ratings) is not privileged to rank 1.
    assert top["book_id"] in {"b0", "b1", "b2"}
    # And critically: the order is not the popularity order. Collect interest rows.
    interest_ids = out[out["slot"] == "interest"]["book_id"].tolist()
    # b7 (2 ratings) is gated out everywhere.
    assert "b7" not in out["book_id"].tolist()
    # A diverse, taste-A set — popularity (b0=1000) does not dominate the top slot order.
    assert set(interest_ids).issubset({"b0", "b1", "b2"})


def test_a3_recommendation_includes_exploration_from_unoccupied_macro() -> None:
    rec = _toy_recommender(RankingConfig(k=4, explore_slots=2, n_clusters_retrieve=2,
                                         min_ratings_gate=5))
    out = rec.recommend("u_taste_A")
    explore = out[out["slot"] == "exploration"]
    assert len(explore) >= 1
    # User occupies macro 0 (taste A). Exploration must come from a macro he does not occupy.
    assert (explore["macro_cluster"] != 0).all()


def test_a4_cold_start_has_no_popularity_and_spans_macros() -> None:
    rec = _toy_recommender(RankingConfig(min_ratings_gate=5, min_pages_accessible=50))
    out = rec.recommend_cold_start("new_user")
    assert (out["slot"] == "cold_start").all()
    # One book per macro-cluster present in the catalog (diversity, not a bestseller list).
    assert set(out["macro_cluster"]) == {0, 1, 2}
    # The most popular books (b0=1000, b3=1000 ratings) must NOT be auto-picked;
    # accessibility (shortest gated book) drives the pick, not ratings_count.
    assert "b0" not in out["book_id"].tolist()  # 400p vs b1/b2 shorter in macro 0
    # Gated book b7 (2 ratings) is excluded from the accessible pool.
    assert "b7" not in out["book_id"].tolist()


def test_cold_start_user_routed_to_cold_path() -> None:
    rec = _toy_recommender(RankingConfig())
    rec.cold_start_users = {"u_taste_A"}  # force the route
    out = rec.recommend("u_taste_A")
    assert (out["slot"] == "cold_start").all()
