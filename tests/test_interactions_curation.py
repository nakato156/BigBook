from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.curation.interactions import (
    ENGAGEMENT_RANK,
    INTERACTION_WEIGHTS,
    K_USER_MIN,
    add_interaction_key,
    build_global_interactions,
    clean_interaction_chunk,
    dedup_keep_best,
    finalize_user_features,
)

DATE = "Wed Aug 30 00:00:26 -0700 2017"
LATER_DATE = "Thu Aug 31 00:00:26 -0700 2017"

RAW_COLUMNS = [
    "user_id",
    "book_id",
    "review_id",
    "is_read",
    "rating",
    "review_text_incomplete",
    "date_added",
    "date_updated",
    "read_at",
    "started_at",
]


def _record(
    user_id: str,
    book_id: str,
    *,
    review_id: str = "",
    is_read: bool = False,
    rating: int = 0,
    text: str = "",
    date_updated: str = DATE,
) -> dict:
    return {
        "user_id": user_id,
        "book_id": book_id,
        "review_id": review_id,
        "is_read": is_read,
        "rating": rating,
        "review_text_incomplete": text,
        "date_added": DATE,
        "date_updated": date_updated,
        "read_at": "",
        "started_at": "",
    }


def _raw_df(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=RAW_COLUMNS)


def _write_gz(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
# Cleaning / engagement
# --------------------------------------------------------------------------- #
def test_clean_recovers_implicit_layer_and_modes() -> None:
    raw = _raw_df(
        [
            _record("u", "b1", rating=0, is_read=False),  # want_to_read
            _record("u", "b2", rating=0, is_read=True),  # read_no_rating
            _record("u", "b3", rating=4, is_read=True),  # rating_only
            _record("u", "b4", rating=5, is_read=True, text="loved it"),  # review
        ]
    )
    cleaned = clean_interaction_chunk(raw)

    # rating == 0 -> NA rating_clean + rating_missing
    assert cleaned["rating_clean"].isna().tolist() == [True, True, False, False]
    assert cleaned["rating_missing"].tolist() == [True, True, False, False]

    assert cleaned["engagement_mode"].tolist() == [
        "want_to_read",
        "read_no_rating",
        "rating_only",
        "review",
    ]
    # is_want_to_read only when not read, no rating, no review
    assert cleaned["is_want_to_read"].tolist() == [True, False, False, False]
    assert cleaned["interaction_weight"].tolist() == pytest.approx(
        [
            INTERACTION_WEIGHTS["want_to_read"],
            INTERACTION_WEIGHTS["read_no_rating"],
            INTERACTION_WEIGHTS["rating_only"],
            INTERACTION_WEIGHTS["review"],
        ]
    )


# --------------------------------------------------------------------------- #
# Dedup
# --------------------------------------------------------------------------- #
def test_dedup_by_review_id_collapses() -> None:
    raw = _raw_df(
        [
            _record("u", "b", review_id="rev1", rating=4, is_read=True),
            _record("u", "b", review_id="rev1", rating=5, is_read=True, text="updated"),
        ]
    )
    keyed = add_interaction_key(clean_interaction_chunk(raw))
    assert keyed["interaction_key"].nunique() == 1
    out = dedup_keep_best(keyed)
    assert len(out) == 1


def test_dedup_identical_review_id_no_error() -> None:
    record = _record("u", "b", review_id="rev1", rating=4, is_read=True)
    keyed = add_interaction_key(clean_interaction_chunk(_raw_df([record, dict(record)])))
    out = dedup_keep_best(keyed)
    assert len(out) == 1


def test_dedup_keep_best_picks_strongest_signal() -> None:
    # Same (user, book), null review_id -> fallback key. want_to_read then rating_only.
    raw = _raw_df(
        [
            _record("u", "b", review_id="", rating=0, is_read=False, date_updated=DATE),
            _record("u", "b", review_id="", rating=4, is_read=True, date_updated=LATER_DATE),
        ]
    )
    keyed = add_interaction_key(clean_interaction_chunk(raw))
    assert keyed["interaction_key"].nunique() == 1
    out = dedup_keep_best(keyed)
    assert len(out) == 1
    # rating_only (priority 2) survives over want_to_read (priority 0), not keep-first
    assert out.iloc[0]["engagement_mode"] == "rating_only"
    assert ENGAGEMENT_RANK["rating_only"] > ENGAGEMENT_RANK["want_to_read"]


# --------------------------------------------------------------------------- #
# Global user features (pure)
# --------------------------------------------------------------------------- #
def test_finalize_user_features_global_bias_and_kcore() -> None:
    partials = pd.DataFrame(
        {
            "read_or_rated_count": [4, 2],
            "rating_sum": [16.0, 6.0],
            "rating_sq_sum": [66.0, 18.0],
            "rating_count": [4, 2],
        },
        index=pd.Index(["valid_user", "thin_user"], name="user_id"),
    )
    global_mean = 3.5
    features = finalize_user_features(partials, global_mean).set_index("user_id")

    assert features.loc["valid_user", "user_mean_rating"] == pytest.approx(4.0)
    assert features.loc["valid_user", "user_rating_bias"] == pytest.approx(4.0 - global_mean)
    assert features.loc["thin_user", "user_rating_bias"] == pytest.approx(3.0 - global_mean)
    assert features["user_rating_bias"].notna().all()
    # K-core is global
    assert bool(features.loc["valid_user", "valid"]) is True
    assert bool(features.loc["thin_user", "valid"]) is False
    assert K_USER_MIN == 3


# --------------------------------------------------------------------------- #
# Full streaming build (cross-category, K-core, text split, book universe)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def built(tmp_path: Path) -> dict:
    valid_books = {"b1", "b2", "b3", "b4", "b5", "bw", "br"}

    cat_a = [
        # u1: cross-category book b1 (shared review_id), + b2, b3 -> 3 engaged
        _record("u1", "b1", review_id="u1_b1", rating=4, is_read=True),
        _record("u1", "b2", review_id="u1_b2", rating=5, is_read=True),
        _record("u1", "b3", review_id="u1_b3", rating=3, is_read=True),
        # uG: only 2 in cat A (would fail per-category K-core)
        _record("uG", "b2", review_id="uG_b2", rating=4, is_read=True),
        _record("uG", "b3", review_id="uG_b3", rating=5, is_read=True),
        # uS: 2 engaged total -> invalid
        _record("uS", "b1", review_id="uS_b1", rating=4, is_read=True),
        _record("uS", "b2", review_id="uS_b2", rating=5, is_read=True),
        # uW: 3 ratings + 1 review (engaged=4) + 1 want_to_read (diversity)
        _record("uW", "b1", review_id="uW_b1", rating=4, is_read=True),
        _record("uW", "b2", review_id="uW_b2", rating=5, is_read=True),
        _record("uW", "b3", review_id="uW_b3", rating=3, is_read=True),
        _record("uW", "br", review_id="uW_br", rating=5, is_read=True, text="a great book"),
        _record("uW", "bw", review_id="uW_bw", rating=0, is_read=False),  # want_to_read
        # uX: invalid book bX must be dropped; valid via b1,b2,b3
        _record("uX", "bX", review_id="uX_bX", rating=5, is_read=True),
        _record("uX", "b1", review_id="uX_b1", rating=4, is_read=True),
        _record("uX", "b2", review_id="uX_b2", rating=4, is_read=True),
        _record("uX", "b3", review_id="uX_b3", rating=4, is_read=True),
    ]
    cat_b = [
        # cross-category duplicate of u1/b1 (same review_id) -> single interaction, scc=2
        _record("u1", "b1", review_id="u1_b1", rating=4, is_read=True),
        # uG: 2 more distinct books -> 4 engaged globally -> valid
        _record("uG", "b4", review_id="uG_b4", rating=4, is_read=True),
        _record("uG", "b5", review_id="uG_b5", rating=3, is_read=True),
    ]

    path_a = tmp_path / "cat_a.json.gz"
    path_b = tmp_path / "cat_b.json.gz"
    _write_gz(path_a, cat_a)
    _write_gz(path_b, cat_b)

    out_interactions = tmp_path / "interactions_curated.parquet"
    out_review_texts = tmp_path / "review_texts.parquet"
    out_user_features = tmp_path / "user_features_global.parquet"

    summary = build_global_interactions(
        category_files={"cat_a": path_a, "cat_b": path_b},
        valid_books=valid_books,
        out_interactions=out_interactions,
        out_review_texts=out_review_texts,
        out_user_features=out_user_features,
        with_source_category_count=True,
        force=True,
        progress=False,
    )
    return {
        "summary": summary,
        "canonical": pd.read_parquet(out_interactions),
        "review_texts": pd.read_parquet(out_review_texts),
        "user_features": pd.read_parquet(out_user_features),
    }


def test_cross_category_dedup_and_source_count(built: dict) -> None:
    canonical = built["canonical"]
    # u1/b1 appears once despite two source dumps
    u1_b1 = canonical[(canonical["user_id"] == "u1") & (canonical["book_id"] == "b1")]
    assert len(u1_b1) == 1
    assert int(u1_b1.iloc[0]["source_category_count"]) == 2
    # a single-category book counts once
    u1_b2 = canonical[(canonical["user_id"] == "u1") & (canonical["book_id"] == "b2")]
    assert int(u1_b2.iloc[0]["source_category_count"]) == 1
    # interaction_key unique in canonical
    assert canonical["interaction_key"].is_unique


def test_kcore_is_global(built: dict) -> None:
    canonical = built["canonical"]
    features = built["user_features"].set_index("user_id")
    # uG: 2 in cat A + 2 in cat B = 4 -> valid (per-category would drop it)
    assert bool(features.loc["uG", "valid"]) is True
    assert "uG" in set(canonical["user_id"])
    # uS: 2 engaged total -> excluded from canonical
    assert bool(features.loc["uS", "valid"]) is False
    assert "uS" not in set(canonical["user_id"])


def test_global_bias_is_consistent(built: dict) -> None:
    features = built["user_features"].set_index("user_id")
    assert features["user_rating_bias"].notna().all()
    # global term cancels in the difference of biases
    bias_diff = features.loc["u1", "user_rating_bias"] - features.loc["uG", "user_rating_bias"]
    mean_diff = features.loc["u1", "user_mean_rating"] - features.loc["uG", "user_mean_rating"]
    assert bias_diff == pytest.approx(mean_diff, abs=1e-5)


def test_book_universe_filter(built: dict) -> None:
    canonical = built["canonical"]
    assert "bX" not in set(canonical["book_id"])
    assert set(canonical["book_id"]).issubset({"b1", "b2", "b3", "b4", "b5", "bw", "br"})


def test_want_to_read_survives_as_diversity(built: dict) -> None:
    canonical = built["canonical"]
    want = canonical[(canonical["user_id"] == "uW") & (canonical["book_id"] == "bw")]
    assert len(want) == 1
    assert want.iloc[0]["engagement_mode"] == "want_to_read"
    assert want.iloc[0]["interaction_weight"] == pytest.approx(INTERACTION_WEIGHTS["want_to_read"])
    assert bool(want.iloc[0]["is_want_to_read"]) is True


def test_review_text_is_separated(built: dict) -> None:
    canonical = built["canonical"]
    review_texts = built["review_texts"]
    # canonical carries no review text
    assert "review_text_clean" not in canonical.columns
    # only rows with review text appear, keyed and deduplicated
    assert (review_texts["review_text_length"] > 0).all()
    assert review_texts["interaction_key"].is_unique
    review_keys = set(review_texts["interaction_key"])
    no_text_rows = canonical[~canonical["has_review_text"].fillna(False)]
    assert review_keys.isdisjoint(set(no_text_rows["interaction_key"]))
    # the one review (uW/br) is present
    canonical_br = canonical[(canonical["user_id"] == "uW") & (canonical["book_id"] == "br")]
    assert canonical_br.iloc[0]["interaction_key"] in review_keys
