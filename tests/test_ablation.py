"""Tests for the ablation harness (Fase 2): pure run_ablation() over a toy Recommender.

No tests/__init__.py and no cross-test imports exist in this repo, so this file builds its
own self-contained toy Recommender fixture (inspired by, but not imported from,
tests/test_recommend.py::_toy_recommender). scripts/ is an implicit namespace package from
the repo root (precedent: tests/test_clustering_script.py), so `from scripts.run_ablation
import ...` works without scripts/__init__.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_ablation import ABLATION_CONFIGS, AblationConfig, run_ablation
from src.reduction.recommend import RankingConfig, Recommender

GENRES = ["genre_fantasy", "genre_mystery", "genre_history", "genre_ya", "genre_romance"]


def _toy_recommender(config: RankingConfig) -> Recommender:
    """Same taste-region layout as tests/test_recommend.py::_toy_recommender (self-contained)."""
    n_pc = 4  # pc_0 = popularity axis (tabular), pc_1..pc_3 = taste
    pc_cols = [f"pc_{i}" for i in range(n_pc)]
    book_ids = np.array([f"b{i}" for i in range(8)])

    book_pc = np.array(
        [
            [50.0, 1.0, 0.0, 0.0],  # b0 popular, taste A
            [0.1, 1.0, 0.0, 0.0],  # b1 niche,  taste A  (same taste as b0)
            [0.1, 0.9, 0.1, 0.0],  # b2 niche,  taste A'
            [40.0, 0.0, 1.0, 0.0],  # b3 popular, taste B
            [0.1, 0.0, 1.0, 0.0],  # b4 niche,  taste B
            [0.1, 0.0, 0.0, 1.0],  # b5 niche,  taste C
            [0.1, 0.0, 0.0, 0.9],  # b6 niche,  taste C
            [0.1, 0.0, 0.0, 0.8],  # b7 very low exposure, taste C
        ],
        dtype=np.float32,
    )
    ratings_count = np.array([1000, 50, 60, 1000, 70, 80, 90, 2])
    num_pages = np.array([400, 120, 600, 300, 90, 500, 80, 70], dtype=np.float64)

    # Two fine clusters per taste region; 3 macro-clusters: {A}, {B}, {C}.
    book_cluster = np.array([0, 0, 1, 2, 2, 3, 4, 4], dtype=np.int64)
    centroids = np.array(
        [
            [25.0, 1.0, 0.0, 0.0],  # cluster 0 -> macro 0 (taste A)
            [0.1, 0.9, 0.1, 0.0],  # cluster 1 -> macro 0 (taste A')
            [20.0, 0.0, 1.0, 0.0],  # cluster 2 -> macro 1 (taste B)
            [0.1, 0.0, 0.0, 1.0],  # cluster 3 -> macro 2 (taste C)
            [0.1, 0.0, 0.0, 0.95],  # cluster 4 -> macro 2 (taste C)
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

    user_ids = np.array(["u1", "u2"])
    user_pc = np.array(
        [
            [30.0, 1.0, 0.0, 0.0],  # u1: taste A
            [20.0, 0.0, 1.0, 0.0],  # u2: taste B
        ],
        dtype=np.float32,
    )

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
        positive_count_by_user={"u1": 3, "u2": 3},
        centroid_user_ids=np.array([], dtype=str),
        user_centroid_pc=np.empty((0, n_pc), dtype=np.float32),
        user_centroid_weight=np.array([], dtype=np.float32),
        pc_cols=pc_cols,
        config=config,
    )


def _toy_interactions() -> pd.DataFrame:
    """Two users, each with exactly one train positive and one future holdout book.

    Single train positive (< 3) routes modes_from_history through the sparse "shrink toward
    nearest centroid" path, deliberately: this keeps the retrieved cluster set small and
    predictable, which is what the candidate_recall test below depends on. u1 trains on b0
    (taste A, nearest cluster 0 = {b0, b1}) and holds out b2 (taste A', cluster 1) — a
    different fine cluster in the same macro, so a low n_clusters_retrieve excludes it from
    candidates entirely while a higher one includes it. u2 trains on b3 (taste B, cluster 2)
    and holds out b5 (taste C, cluster 3), keeping the two users' candidate pools disjoint.
    """
    return pd.DataFrame(
        [
            ("u1", "b0", True, 5.0, False, np.nan, "2020-01-01"),
            ("u1", "b2", True, 5.0, False, np.nan, "2020-02-01"),
            ("u2", "b3", True, 5.0, False, np.nan, "2020-01-01"),
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


def test_run_ablation_produces_one_row_per_config_system_k() -> None:
    rec = _toy_recommender(RankingConfig(k=3, explore_slots=0, n_clusters_retrieve=3))
    interactions = _toy_interactions()
    configs = [
        AblationConfig("baseline", {}),
        AblationConfig("n_clusters_retrieve=1", {"n_clusters_retrieve": 1}),
    ]

    results = run_ablation(
        rec,
        interactions,
        configs,
        popularity_count=rec.ratings_count,
        average_rating=np.ones(len(rec.book_ids)),
        train_fraction=0.67,
    )

    # evaluate_temporal's summary has one row per (system, k); systems are
    # model + B0_random + B1_popularity + B2_genre_popularity = 4, k has a single value here
    # (recommender.config.k=3, ks=None default), so each config contributes 4 rows.
    n_systems = 4
    n_k_values = 1
    assert len(results) == len(configs) * n_systems * n_k_values
    assert set(results["config_label"]) == {"baseline", "n_clusters_retrieve=1"}
    assert set(results["system"]) == {"model", "B0_random", "B1_popularity", "B2_genre_popularity"}

    # The override column is recorded per row, NaN-filled for configs that don't touch it
    # (baseline has no overrides dict entry, so the column reflects only the swept configs).
    swept = results.loc[results["config_label"] == "n_clusters_retrieve=1", "n_clusters_retrieve"]
    assert (swept == 1).all()


def test_run_ablation_restores_config_even_when_evaluate_temporal_raises(monkeypatch) -> None:
    rec = _toy_recommender(RankingConfig(k=3, explore_slots=0, n_clusters_retrieve=3))
    interactions = _toy_interactions()
    original_config = rec.config

    import scripts.run_ablation as run_ablation_module

    def fail(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(run_ablation_module, "evaluate_temporal", fail)

    with np.testing.assert_raises(RuntimeError):
        run_ablation(
            rec,
            interactions,
            [AblationConfig("broken", {"n_clusters_retrieve": 1})],
            popularity_count=rec.ratings_count,
            average_rating=np.ones(len(rec.book_ids)),
        )

    assert rec.config == original_config


def test_n_clusters_retrieve_changes_candidate_recall_for_model_system() -> None:
    # Same scenario as test_recommend.py::test_candidate_recall_separates_retrieval_from_
    # ranking_failure: u1's profile (taste A) sits in cluster 0; the holdout target b2 lives
    # in cluster 1 (taste A', not retrieved when n_clusters_retrieve=1). With
    # n_clusters_retrieve=3 all 5 clusters... well, 3 of 5 are retrieved, which is enough to
    # include cluster 1 (b2's cluster) since cluster 0 and cluster 1 are both taste-A nearest
    # neighbors of u1's profile.
    rec = _toy_recommender(RankingConfig(k=3, explore_slots=0, n_clusters_retrieve=1))
    interactions = _toy_interactions()
    configs = [
        AblationConfig("retrieve=1", {"n_clusters_retrieve": 1}),
        AblationConfig("retrieve=3", {"n_clusters_retrieve": 3}),
    ]

    results = run_ablation(
        rec,
        interactions,
        configs,
        popularity_count=rec.ratings_count,
        average_rating=np.ones(len(rec.book_ids)),
        train_fraction=0.67,
    )

    model_rows = results.loc[results["system"] == "model"].set_index("config_label")
    low_recall = model_rows.loc["retrieve=1", "candidate_recall"]
    high_recall = model_rows.loc["retrieve=3", "candidate_recall"]
    assert low_recall < high_recall


def test_ablation_configs_constant_has_expected_labels() -> None:
    labels = {cfg.label for cfg in ABLATION_CONFIGS}
    assert "baseline" in labels
    assert any("n_clusters_retrieve" in label for label in labels)
    assert any("mmr_lambda" in label for label in labels)
    assert any("explore_slots" in label for label in labels)
