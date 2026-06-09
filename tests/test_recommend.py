"""Tests for the ranking layer: each test pins one of the four contradiction fixes.

C1→A1 (taste subspace drops pc_0..pc_5), C2→A2 (technical eligibility, no popularity
gate/score), C3→A3 (relevant tail/mid exploration), C4→A4 (cold-start has no popularity).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.reduction.evaluate_recommender import (
    baseline_recommendations,
    evaluate_temporal,
    temporal_split,
)
from src.reduction.recommend import (
    RankingConfig,
    Recommender,
    accessibility_scores,
    eligibility_mask,
    l2_normalize_rows,
    mmr_select,
    popularity_segments,
    select_exploration_rows,
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


def test_a2_eligibility_uses_artifact_integrity_not_popularity() -> None:
    mask = eligibility_mask(
        book_ids=np.array(["low", "high", "", "bad-vector", "bad-cluster"]),
        titles=np.array(["Low evidence", "Bestseller", "Missing id", "Bad", "Bad"]),
        book_pc=np.array([[1, 0], [1, 0], [1, 0], [np.nan, 0], [1, 0]]),
        book_cluster=np.array([0, 0, 0, 0, -1]),
        n_clusters=1,
    )
    assert mask.tolist() == [True, True, False, False, False]


def test_a3_popularity_segments_use_catalog_quantiles() -> None:
    labels, tail_cut, head_cut = popularity_segments(
        np.array([10, 20, 30, 40, 50]), tail_quantile=0.25, head_quantile=0.80
    )
    assert tail_cut == 20
    assert head_cut == 42
    assert labels.tolist() == ["tail", "tail", "mid", "mid", "head"]


def test_a3_exploration_prefers_tail_after_relevance_floor() -> None:
    rows = np.array([1, 2, 3])
    relevance = np.array([0.80, 0.78, 0.95])
    segments = np.array(["unknown", "tail", "mid", "head"])
    picked = select_exploration_rows(
        rows, relevance, segments, k=2, best_relevance=1.0, min_relevance_ratio=0.75
    )
    assert picked.tolist() == [1, 2]


def test_a3_exploration_never_uses_head_as_fallback() -> None:
    rows = np.array([0, 1])
    picked = select_exploration_rows(
        rows,
        relevance=np.array([0.99, 0.95]),
        popularity_segment=np.array(["head", "head"]),
        k=2,
        best_relevance=1.0,
        min_relevance_ratio=0.75,
    )
    assert picked.tolist() == []


def test_mmr_prefers_relevance_then_diversifies() -> None:
    # Two near-duplicate high-relevance vectors + one orthogonal one.
    cand = l2_normalize_rows(np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]))
    relevance = np.array([0.9, 0.89, 0.4])
    picked = mmr_select(cand, relevance, k=2, lam=0.5)
    # First the top-relevance item, then the diverse one — not its near-duplicate.
    assert picked[0] == 0
    assert picked[1] == 2


def test_mmr_penalizes_repeated_genres() -> None:
    cand = l2_normalize_rows(np.array([[1.0, 0.0], [0.99, 0.01], [0.8, 0.2]]))
    relevance = np.array([0.9, 0.89, 0.88])
    genres = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float64)
    picked = mmr_select(
        cand,
        relevance,
        k=2,
        lam=0.9,
        candidate_genres=genres,
        genre_weight=0.2,
    )
    assert picked == [0, 2]


def test_accessibility_scores_favor_shorter_valid_books() -> None:
    scores = accessibility_scores(np.array([np.nan, 30, 100, 500]), min_pages=50)
    assert scores[0] == 0
    assert scores[1] == 0
    assert scores[2] > scores[3]


# --------------------------------------------------------------------------- #
# End-to-end on a tiny synthetic catalog
# --------------------------------------------------------------------------- #
def _toy_recommender(config: RankingConfig) -> Recommender:
    n_pc = 4  # pc_0 = popularity axis (tabular), pc_1..pc_3 = taste
    if config.tabular_pcs == (0, 1, 2, 3, 4, 5):
        config = replace(config, tabular_pcs=(0,))
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
            [0.1, 0.0, 0.0, 0.8],   # b7 very low exposure, taste C
        ],
        dtype=np.float32,
    )
    ratings_count = np.array([1000, 50, 60, 1000, 70, 80, 90, 2])
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
    genres.loc[["b0", "b1"], "genre_fantasy"] = 1
    genres.loc["b2", "genre_romance"] = 1
    genres.loc[["b3", "b4"], "genre_mystery"] = 1
    genres.loc[["b5", "b6", "b7"], "genre_history"] = 1

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
        positive_count_by_user={"u_taste_A": 3},
        centroid_user_ids=np.array([], dtype=str),
        user_centroid_pc=np.empty((0, n_pc), dtype=np.float32),
        user_centroid_weight=np.array([], dtype=np.float32),
        pc_cols=pc_cols,
        config=config,
    )


def test_a2_popularity_never_orders_the_interest_slots() -> None:
    # k=3, no exploration, so we see only interest-scored books.
    rec = _toy_recommender(RankingConfig(k=3, explore_slots=0, n_clusters_retrieve=2,
                                         mmr_lambda=0.8))
    out = rec.recommend("u_taste_A", set())
    # The user "inherited" a huge pc_0, yet the popular b0 must NOT outrank the niche b1/b2
    # purely by popularity: b0 and b1 share taste, so popularity cannot be the tiebreaker.
    top = out[out["slot"] == "interest"].iloc[0]
    # b1 and b2 (niche, taste A) are equally/most relevant in the taste subspace as b0;
    # the ranking is popularity-free, so b0 (1000 ratings) is not privileged to rank 1.
    assert top["book_id"] in {"b0", "b1", "b2"}
    # And critically: the order is not the popularity order. Collect interest rows.
    interest_ids = out[out["slot"] == "interest"]["book_id"].tolist()
    # A diverse, taste-A set — popularity (b0=1000) does not dominate the top slot order.
    assert set(interest_ids).issubset({"b0", "b1", "b2"})


def test_a3_recommendation_includes_exploration_from_unoccupied_macro() -> None:
    rec = _toy_recommender(RankingConfig(k=4, explore_slots=2, n_clusters_retrieve=2,
                                         explore_min_relevance_ratio=0.0))
    out = rec.recommend("u_taste_A", set())
    explore = out[out["slot"] == "exploration"]
    assert len(explore) >= 1
    # User occupies macro 0 (taste A). Exploration must come from a macro he does not occupy.
    assert (explore["macro_cluster"] != 0).all()
    assert set(explore["popularity_segment"]).issubset({"tail", "mid"})


def test_normal_ranking_uses_accessibility_as_soft_tiebreak() -> None:
    rec = _toy_recommender(
        RankingConfig(
            k=1,
            explore_slots=0,
            n_clusters_retrieve=1,
            mmr_lambda=1.0,
            accessibility_weight=0.2,
        )
    )
    out = rec.recommend("u_taste_A", set())
    assert out.iloc[0]["book_id"] == "b1"


def test_a4_cold_start_has_no_popularity_and_spans_macros() -> None:
    rec = _toy_recommender(RankingConfig(min_pages_accessible=50))
    out = rec.recommend_cold_start("new_user", set())
    assert (out["slot"] == "cold_start").all()
    # One book per macro-cluster present in the catalog (diversity, not a bestseller list).
    assert set(out["macro_cluster"]) == {0, 1, 2}
    # The most popular books (b0=1000, b3=1000 ratings) must NOT be auto-picked;
    # accessibility (shortest eligible book) drives the pick, not ratings_count.
    assert "b0" not in out["book_id"].tolist()  # 400p vs b1/b2 shorter in macro 0


def test_cold_start_user_routed_to_cold_path() -> None:
    rec = _toy_recommender(RankingConfig())
    rec.positive_count_by_user["u_taste_A"] = 0
    del rec._user_row["u_taste_A"]
    out = rec.recommend("u_taste_A", set())
    assert (out["slot"] == "cold_start").all()


def test_consumed_books_are_excluded_from_profile_and_cold_start() -> None:
    rec = _toy_recommender(RankingConfig(k=4, explore_slots=0, n_clusters_retrieve=5))
    out = rec.recommend("u_taste_A", {"b0", "b1"})
    assert not {"b0", "b1"}.intersection(out["book_id"])

    cold = rec.recommend("new_user", {"b1", "b4", "b6"})
    assert not {"b1", "b4", "b6"}.intersection(cold["book_id"])


def test_seed_books_build_a_cold_start_profile() -> None:
    rec = _toy_recommender(RankingConfig(k=3, explore_slots=0, n_clusters_retrieve=2))
    out = rec.recommend("new_user", set(), seed_book_ids=["b5", "b6"])
    assert (out["slot"] == "interest").all()
    assert not {"b5", "b6"}.intersection(out["book_id"])
    assert set(out["fine_cluster"]).issubset({3, 4})


def test_sparse_profile_uses_shrinkage_instead_of_global_fallback() -> None:
    rec = _toy_recommender(RankingConfig(k=3, explore_slots=0, n_clusters_retrieve=2))
    rec.positive_count_by_user["u_taste_A"] = 1
    out = rec.recommend("u_taste_A", {"b0"})
    assert (out["slot"] == "interest").all()
    assert "b0" not in out["book_id"].tolist()


def test_multi_centroid_modes_drive_ranking() -> None:
    rec = _toy_recommender(RankingConfig(k=2, explore_slots=0, n_clusters_retrieve=2))
    rec.centroid_user_ids = np.array(["u_taste_A", "u_taste_A"])
    rec.user_centroid_pc = np.array(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32
    )
    rec.user_centroid_weight = np.array([0.1, 0.9], dtype=np.float32)
    rec._centroid_rows = {"u_taste_A": np.array([0, 1], dtype=np.int64)}

    out = rec.recommend("u_taste_A", set())
    assert out.iloc[0]["book_id"] in {"b3", "b4"}


def test_temporal_split_is_chronological_per_user() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["u", "u", "u"],
            "book_id": ["late", "early", "future"],
            "date_added": pd.to_datetime(["2020-02-01", "2020-01-01", "2020-03-01"]),
        }
    )
    train, future = temporal_split(interactions, train_fraction=0.67)
    assert train["book_id"].tolist() == ["early", "late"]
    assert future["book_id"].tolist() == ["future"]


def test_baselines_always_exclude_consumed_books() -> None:
    rec = _toy_recommender(RankingConfig(k=3, explore_slots=0, n_clusters_retrieve=2))
    baselines = baseline_recommendations(
        rec,
        average_rating=np.ones(len(rec.book_ids)),
        consumed={"b0", "b1"},
        train_genres=np.zeros(len(GENRES), dtype=int),
        user_id="u",
        k=3,
    )
    for recommended in baselines.values():
        assert not {"b0", "b1"}.intersection(recommended)


def test_temporal_evaluation_runs_model_and_three_baselines() -> None:
    rec = _toy_recommender(RankingConfig(k=3, explore_slots=0, n_clusters_retrieve=3))
    interactions = pd.DataFrame(
        [
            ("u1", "b0", True, 5.0, False, np.nan, "2020-01-01"),
            ("u1", "b1", True, 4.0, False, np.nan, "2020-01-02"),
            ("u1", "b2", True, 5.0, False, np.nan, "2020-02-01"),
            ("u2", "b3", True, 5.0, False, np.nan, "2020-01-01"),
            ("u2", "b4", True, 4.0, False, np.nan, "2020-01-02"),
            ("u2", "b5", True, 5.0, False, np.nan, "2020-02-01"),
        ],
        columns=[
            "user_id",
            "book_id",
            "is_read",
            "rating_clean",
            "has_review_text",
            "reading_duration_days",
            "date_added",
        ],
    )
    interactions["date_added"] = pd.to_datetime(interactions["date_added"])
    summary, per_user = evaluate_temporal(
        interactions, rec, average_rating=np.ones(len(rec.book_ids)), train_fraction=0.67
    )
    assert set(summary["system"]) == {
        "model",
        "B0_random",
        "B1_popularity",
        "B2_genre_popularity",
    }
    assert len(per_user) == 8
    assert {
        "recall",
        "precision",
        "ndcg",
        "catalog_coverage",
        "long_tail_coverage",
        "novelty",
    }.issubset(summary.columns)
