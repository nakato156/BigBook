from __future__ import annotations

import numpy as np
import pandas as pd

from src.reduction.build_user_centroids import (
    M_CAP,
    build_user_centroids,
    choose_m,
    compute_engagement_weight,
)


def _item_matrix() -> pd.DataFrame:
    # Two clear PCA groups: A near the origin, B near [10, 10].
    return pd.DataFrame(
        {
            "book_id": ["a1", "a2", "a3", "b1", "b2", "b3"],
            "pc_0": [0.0, 0.0, 1.0, 10.0, 10.0, 11.0],
            "pc_1": [0.0, 1.0, 0.0, 10.0, 11.0, 10.0],
        }
    )


def _interactions() -> pd.DataFrame:
    rows = [
        # uA: 6 positives split across the two groups; group A reviewed + rating 5
        ("uA", "a1", True, 5.0, True, np.nan),
        ("uA", "a2", True, 5.0, True, np.nan),
        ("uA", "a3", True, 5.0, True, np.nan),
        ("uA", "b1", True, 4.0, False, np.nan),
        ("uA", "b2", True, 4.0, False, np.nan),
        ("uA", "b3", True, 4.0, False, np.nan),
        # uB: 3 positives -> below the multi-centroid floor -> m=1
        ("uB", "a1", True, 4.0, False, np.nan),
        ("uB", "a2", True, 5.0, False, np.nan),
        ("uB", "a3", True, 4.0, False, np.nan),
        # uC: no surviving positives (not-read / sub-threshold / absent book)
        ("uC", "a1", False, 5.0, False, np.nan),
        ("uC", "a2", True, 3.0, False, np.nan),
        ("uC", "zzz", True, 5.0, False, np.nan),
    ]
    return pd.DataFrame(
        rows,
        columns=["user_id", "book_id", "is_read", "rating_clean", "has_review_text", "reading_duration_days"],
    )


def _build():
    return build_user_centroids(_item_matrix(), [_interactions()])


def test_choose_m_cap_and_floor() -> None:
    assert choose_m(5) == 1  # below MULTI_CENTROID_MIN_POSITIVES
    assert choose_m(6) == 2  # 6 // 3
    assert choose_m(9) == 3
    assert choose_m(12) == M_CAP
    assert choose_m(100) == M_CAP  # capped


def test_fallback_m1_is_mean_of_positives() -> None:
    centroids, _diag = _build()
    ub = centroids[centroids["user_id"] == "uB"]
    assert len(ub) == 1
    row = ub.iloc[0]
    assert row["n_books"] == 3
    assert np.isclose(row["weight"], 1.0)
    assert np.isclose(row["centroid_weight"], 1.0)
    # mean of a1=[0,0], a2=[0,1], a3=[1,0]
    assert np.allclose([row["pc_0"], row["pc_1"]], [1.0 / 3.0, 1.0 / 3.0])


def test_adaptive_m_splits_into_groups() -> None:
    centroids, _diag = _build()
    ua = centroids[centroids["user_id"] == "uA"].reset_index(drop=True)
    assert len(ua) == 2  # n=6 -> m=2
    centers = ua[["pc_0", "pc_1"]].to_numpy()
    centers = centers[np.argsort(centers[:, 0])]  # order by pc_0: group A first
    assert np.allclose(centers[0], [1.0 / 3.0, 1.0 / 3.0])
    assert np.allclose(centers[1], [31.0 / 3.0, 31.0 / 3.0])


def test_weight_sums_to_one_and_n_books_match() -> None:
    centroids, _diag = _build()
    ua = centroids[centroids["user_id"] == "uA"]
    assert np.isclose(ua["weight"].sum(), 1.0)
    assert set(ua["n_books"]) == {3}  # both clusters have 3 books
    assert ua["n_books"].sum() == 6


def test_centroid_weight_higher_for_reviewed_high_rating_cluster() -> None:
    centroids, _diag = _build()
    ua = centroids[centroids["user_id"] == "uA"].reset_index(drop=True)
    assert np.isclose(ua["centroid_weight"].sum(), 1.0)
    # group A (near origin) is reviewed + rating 5 -> heavier centroid_weight
    group_a = ua.loc[ua["pc_0"] < 5.0, "centroid_weight"].iloc[0]
    group_b = ua.loc[ua["pc_0"] >= 5.0, "centroid_weight"].iloc[0]
    assert group_a > group_b


def test_positive_definition_unchanged() -> None:
    centroids, _diag = _build()
    # uC has no read+>=4 positive on an in-universe book
    assert "uC" not in set(centroids["user_id"])


def test_absent_book_id_is_dropped() -> None:
    _centroids, diag = _build()
    assert diag["dropped_positive_rows"] == 1  # uC's "zzz"


def test_pc_columns_aligned_with_item_matrix() -> None:
    centroids, diag = _build()
    pc_cols = [c for c in centroids.columns if c.startswith("pc_")]
    assert pc_cols == ["pc_0", "pc_1"]
    assert diag["pc_columns"] == ["pc_0", "pc_1"]


def test_compute_engagement_weight() -> None:
    rating = np.array([5.0, 4.0, 5.0])
    review = np.array([True, False, False])
    duration = np.array([10.0, np.nan, 500.0])
    w = compute_engagement_weight(rating, review, duration)
    assert np.isclose(w[0], (5 - 3) * 1.3 * 1.2)  # reviewed + in-window
    assert np.isclose(w[1], (4 - 3) * 1.0 * 1.0)  # nan duration -> no boost
    assert np.isclose(w[2], (5 - 3) * 1.0 * 1.0)  # duration out of window
