from __future__ import annotations

import numpy as np
import pandas as pd

from src.reduction.build_user_matrix import build_user_artifacts, pc_columns


def _item_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "book_id": ["b1", "b2", "b3", "b4", "b5"],
            "pc_0": [1.0, 0.0, 1.0, 2.0, 10.0],
            "pc_1": [0.0, 1.0, 1.0, 2.0, 10.0],
        }
    )


def _books_master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "book_id": ["b1", "b2", "b3", "b4", "b5"],
            "genre_fantasy": [1, 0, 1, 0, 0],
            "genre_mystery": [0, 1, 1, 0, 0],
            "genre_history": [0, 0, 0, 1, 0],
            "genre_ya": [0, 0, 0, 0, 0],
            "genre_romance": [0, 0, 0, 0, 1],
        }
    )


def _user_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "user_rating_bias": [0.5, -0.2, 0.1, 0.0, 0.3],
        }
    )


def _interactions() -> pd.DataFrame:
    rows = [
        # u1: two positives (b1, b2), two genres
        ("u1", "b1", True, 5.0, False, False),
        ("u1", "b2", True, 4.0, True, False),
        # u2: no positives -> rating<4, and an is_read=False high rating, plus want_to_read
        ("u2", "b1", True, 3.0, False, False),
        ("u2", "b2", False, 5.0, False, False),
        ("u2", "b3", False, np.nan, False, True),
        # u3: single positive
        ("u3", "b4", True, 5.0, False, False),
        # u4: three positives (b1, b3, b4), an absent book, and a want_to_read
        ("u4", "b1", True, 4.0, False, False),
        ("u4", "b3", True, 5.0, True, False),
        ("u4", "b4", True, 4.0, False, False),
        ("u4", "bX", True, 5.0, False, False),
        ("u4", "b2", False, np.nan, False, True),
    ]
    df = pd.DataFrame(
        rows,
        columns=["user_id", "book_id", "is_read", "rating_clean", "has_review_text", "is_want_to_read"],
    )
    df["date_added"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(len(df)), unit="D")
    return df


def _build():
    return build_user_artifacts(
        _item_matrix(), _books_master(), _user_features(), [_interactions()]
    )


def test_user_vector_is_mean_of_positive_books_and_pc_aligned() -> None:
    matrix, _meta, diag = _build()
    assert pc_columns(matrix) == ["pc_0", "pc_1"]
    assert diag["pc_columns"] == ["pc_0", "pc_1"]
    u1 = matrix.set_index("user_id").loc["u1"]
    # mean of b1=[1,0] and b2=[0,1]
    assert np.allclose([u1["pc_0"], u1["pc_1"]], [0.5, 0.5])


def test_rating_below_threshold_does_not_contribute() -> None:
    matrix, meta, _diag = _build()
    # u2 has only sub-threshold / non-read / want_to_read rows -> no positives
    assert "u2" not in set(matrix["user_id"])
    u2 = meta.set_index("user_id").loc["u2"]
    assert u2["positive_count"] == 0


def test_zero_positive_user_absent_in_matrix_present_in_meta() -> None:
    matrix, meta, _diag = _build()
    assert "u2" not in set(matrix["user_id"])
    u2 = meta.set_index("user_id").loc["u2"]
    assert u2["positive_count"] == 0
    assert bool(u2["is_cold_start"]) is True


def test_cold_start_flag_threshold() -> None:
    _matrix, meta, _diag = _build()
    meta = meta.set_index("user_id")
    assert bool(meta.loc["u3", "is_cold_start"]) is True  # 1 positive < 3
    assert bool(meta.loc["u4", "is_cold_start"]) is False  # 3 positives >= 3


def test_is_read_false_high_rating_is_not_positive() -> None:
    _matrix, meta, _diag = _build()
    # u2's b2 row is is_read=False rating 5 -> not positive
    assert meta.set_index("user_id").loc["u2", "positive_count"] == 0


def test_want_to_read_excluded_from_vector_but_counted_in_meta() -> None:
    _matrix, meta, _diag = _build()
    meta = meta.set_index("user_id")
    assert meta.loc["u4", "want_to_read_count"] == 1
    assert meta.loc["u4", "positive_count"] == 3  # want_to_read does not count


def test_absent_book_id_is_dropped() -> None:
    matrix, meta, diag = _build()
    assert diag["dropped_positive_rows"] == 1  # u4's "bX"
    u4 = matrix.set_index("user_id").loc["u4"]
    # mean of b1=[1,0], b3=[1,1], b4=[2,2] -> [1.333.., 1.0]
    assert np.allclose([u4["pc_0"], u4["pc_1"]], [4.0 / 3.0, 1.0])


def test_multi_genre_category_count() -> None:
    _matrix, meta, _diag = _build()
    meta = meta.set_index("user_id")
    assert meta.loc["u1", "category_count"] == 2  # fantasy + mystery
    assert meta.loc["u3", "category_count"] == 1  # history only


def test_user_without_interactions_absent_from_meta() -> None:
    _matrix, meta, _diag = _build()
    assert "u5" not in set(meta["user_id"])  # in census but no canonical rows


def test_bias_taken_from_user_features() -> None:
    _matrix, meta, _diag = _build()
    assert np.isclose(meta.set_index("user_id").loc["u1", "user_rating_bias"], 0.5)
