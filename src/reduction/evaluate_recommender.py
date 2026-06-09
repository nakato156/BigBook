"""Temporal offline evaluation for the recommender and B0/B1/B2 baselines."""

from __future__ import annotations

import argparse
import math
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    BOOKS_MASTER_PATH,
    INTERACTIONS_CURATED_PATH,
    PROJECT_ROOT,
    USER_FEATURES_GLOBAL_PATH,
)
from src.reduction.build_user_centroids import compute_engagement_weight
from src.reduction.recommend import GENRE_COLUMNS, Recommender, popularity_segments

RANDOM_STATE = 42
MIN_VALID_DATE = pd.Timestamp("2006-01-01", tz="UTC")
EVALUATION_MODE = "global_historical_snapshot_frozen_representation"
EVALUATION_OUTPUT = PROJECT_ROOT / "data" / "outputs" / "recommendations" / "temporal_evaluation.csv"
INTERACTION_COLUMNS = [
    "user_id",
    "book_id",
    "is_read",
    "rating_clean",
    "has_review_text",
    "reading_duration_days",
    "date_added",
]
POPULARITY_COLUMNS = ["book_id", "rating_clean", "date_added"]


@dataclass(frozen=True)
class HistoricalSnapshot:
    """Catalog evidence observed in the canonical interaction log."""

    rating_count: np.ndarray
    average_rating: np.ndarray
    first_observed: np.ndarray
    invalid_date_count: int


def _utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _valid_dates(values: pd.Series) -> tuple[pd.Series, np.ndarray]:
    dates = pd.to_datetime(values, errors="coerce", utc=True)
    valid = dates.notna() & (dates >= MIN_VALID_DATE)
    return dates, valid.to_numpy(dtype=bool)


def temporal_split(
    interactions: pd.DataFrame, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological per-user split with at least one row on each side."""
    data = interactions.copy()
    data["user_id"] = data["user_id"].astype(str)
    data["book_id"] = data["book_id"].astype(str)
    dates, valid = _valid_dates(data["date_added"])
    data["date_added"] = dates
    data = data.loc[valid].sort_values(["user_id", "date_added"], kind="stable")
    if data.empty:
        return data.copy(), data.copy()
    position = data.groupby("user_id").cumcount()
    size = data.groupby("user_id")["user_id"].transform("size")
    cutoff = np.floor(size * train_fraction).astype(int).clip(lower=1)
    cutoff = np.minimum(cutoff, size - 1)
    train_mask = (size > 1) & (position < cutoff)
    return data.loc[train_mask].copy(), data.loc[~train_mask].copy()


def global_temporal_split(
    interactions: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split every user at one shared timestamp; invalid dates are not evaluable."""
    data = interactions.copy()
    data["user_id"] = data["user_id"].astype(str)
    data["book_id"] = data["book_id"].astype(str)
    dates, valid = _valid_dates(data["date_added"])
    data["date_added"] = dates
    cutoff = _utc_timestamp(cutoff)
    return (
        data.loc[valid & (data["date_added"] <= cutoff)].copy(),
        data.loc[valid & (data["date_added"] > cutoff)].copy(),
    )


def choose_global_cutoff(interactions: pd.DataFrame, train_fraction: float) -> pd.Timestamp:
    """Choose one chronological cutoff from the bounded evaluation cohort."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    dates, valid = _valid_dates(interactions["date_added"])
    valid_dates = dates.loc[valid]
    if valid_dates.empty:
        raise ValueError("Cannot choose a temporal cutoff without valid date_added values.")
    return pd.Timestamp(valid_dates.quantile(train_fraction))


def popularity_from_training(
    train: pd.DataFrame,
    book_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build rating count and mean using training rows only."""
    frame = train[["book_id", "rating_clean"]].copy()
    frame["book_id"] = frame["book_id"].astype(str)
    frame["rating_clean"] = pd.to_numeric(frame["rating_clean"], errors="coerce")
    rated = frame.dropna(subset=["rating_clean"])
    grouped = rated.groupby("book_id", sort=False)["rating_clean"].agg(["count", "mean"])
    aligned = grouped.reindex(pd.Index(book_ids.astype(str), name="book_id"))
    return (
        aligned["count"].fillna(0).to_numpy(dtype=np.float64),
        aligned["mean"].fillna(0.0).to_numpy(dtype=np.float64),
    )


def historical_popularity_snapshot(
    interactions_path: Path,
    book_ids: np.ndarray,
    cutoff: pd.Timestamp,
    batch_size: int = 500_000,
) -> HistoricalSnapshot:
    """Aggregate rating evidence and first observation from one canonical scan."""
    cutoff = _utc_timestamp(cutoff)
    if cutoff < MIN_VALID_DATE:
        raise ValueError(f"cutoff must be on or after {MIN_VALID_DATE.date()}.")
    book_row = {str(book_id): row for row, book_id in enumerate(book_ids)}
    counts = np.zeros(len(book_ids), dtype=np.int64)
    sums = np.zeros(len(book_ids), dtype=np.float64)
    first_observed_ns = np.full(len(book_ids), np.iinfo(np.int64).max, dtype=np.int64)
    invalid_date_count = 0

    parquet = pq.ParquetFile(interactions_path)
    for batch in parquet.iter_batches(columns=POPULARITY_COLUMNS, batch_size=batch_size):
        frame = batch.to_pandas()
        dates, valid_dates = _valid_dates(frame["date_added"])
        invalid_date_count += int((~valid_dates).sum())
        mapped = frame["book_id"].astype(str).map(book_row)
        observed = valid_dates & mapped.notna().to_numpy()
        if observed.any():
            observed_rows = mapped.loc[observed].to_numpy(dtype=np.int64)
            observed_ns = (
                dates.loc[observed]
                .dt.tz_localize(None)
                .to_numpy(dtype="datetime64[ns]")
                .view(np.int64)
            )
            np.minimum.at(first_observed_ns, observed_rows, observed_ns)

        ratings = pd.to_numeric(frame["rating_clean"], errors="coerce")
        keep = valid_dates & (dates <= cutoff).to_numpy() & ratings.notna().to_numpy()
        if not keep.any():
            continue
        rated_rows = mapped.loc[keep]
        present = rated_rows.notna()
        rows = rated_rows.loc[present].to_numpy(dtype=np.int64)
        values = ratings.loc[keep].loc[present].to_numpy(dtype=np.float64)
        np.add.at(counts, rows, 1)
        np.add.at(sums, rows, values)

    averages = np.divide(
        sums,
        counts,
        out=np.zeros(len(book_ids), dtype=np.float64),
        where=counts > 0,
    )
    first_observed = np.full(len(book_ids), np.datetime64("NaT"), dtype="datetime64[ns]")
    observed = first_observed_ns != np.iinfo(np.int64).max
    first_observed[observed] = first_observed_ns[observed].astype("datetime64[ns]")
    return HistoricalSnapshot(
        rating_count=counts.astype(np.float64),
        average_rating=averages,
        first_observed=first_observed,
        invalid_date_count=invalid_date_count,
    )


def historical_catalog_mask(
    books_path: Path,
    book_ids: np.ndarray,
    cutoff: pd.Timestamp,
    first_observed: np.ndarray,
) -> np.ndarray:
    """Catalog availability from publication year or observed interaction evidence."""
    books = pd.read_parquet(books_path, columns=["book_id", "publication_year"])
    books["book_id"] = books["book_id"].astype(str)
    years = pd.to_numeric(
        books.set_index("book_id").reindex(book_ids.astype(str))["publication_year"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    observed = np.asarray(first_observed, dtype="datetime64[ns]")
    if len(observed) != len(book_ids):
        raise ValueError("first_observed must align 1:1 with book_ids.")
    cutoff = _utc_timestamp(cutoff)
    known_year = np.isfinite(years)
    observed_before_cutoff = ~np.isnat(observed) & (
        observed <= cutoff.tz_localize(None).to_datetime64()
    )
    return (known_year & (years <= cutoff.year)) | (~known_year & observed_before_cutoff)


def _positive_mask(frame: pd.DataFrame) -> np.ndarray:
    is_read = frame["is_read"].fillna(False).to_numpy(dtype=bool)
    rating = pd.to_numeric(frame["rating_clean"], errors="coerce").to_numpy(dtype=np.float64)
    return is_read & (rating >= 4.0)


def _consumed_books(frame: pd.DataFrame) -> set[str]:
    read = frame["is_read"].fillna(False).to_numpy(dtype=bool)
    return set(frame.loc[read, "book_id"].astype(str))


def _binary_metrics(recommended: list[str], relevant: set[str], k: int) -> dict[str, float]:
    top = recommended[:k]
    hits = np.array([book_id in relevant for book_id in top], dtype=np.float64)
    recall = float(hits.sum() / len(relevant)) if relevant else 0.0
    precision = float(hits.sum() / k) if k > 0 else 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
    dcg = float((hits * discounts).sum())
    ideal_n = min(len(relevant), k)
    idcg = float((1.0 / np.log2(np.arange(2, ideal_n + 2))).sum()) if ideal_n else 0.0
    precision_at_hits = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    average_precision = (
        float((precision_at_hits * hits).sum() / ideal_n) if ideal_n else 0.0
    )
    return {
        "recall": recall,
        "precision": precision,
        "ndcg": dcg / idcg if idcg else 0.0,
        "average_precision": average_precision,
    }


def _intra_list_diversity(recommender: Recommender, recommended: list[str]) -> float:
    """Mean pairwise cosine distance in the ranking taste subspace."""
    rows = [
        recommender._book_row[book_id]
        for book_id in recommended
        if book_id in recommender._book_row
    ]
    if len(rows) < 2:
        return 0.0
    vectors = recommender.book_taste_norm[np.asarray(rows, dtype=np.int64)]
    similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)
    upper = similarities[np.triu_indices(len(rows), k=1)]
    return float(np.mean(1.0 - upper))


def _slot_metrics(model: pd.DataFrame, relevant: set[str], slot: str) -> dict[str, float]:
    """Precision and user-level hit rate for one model slot type."""
    selected = model.loc[model["slot"] == slot, "book_id"].astype(str).tolist()
    if not selected:
        return {f"{slot}_precision": np.nan, f"{slot}_hit_rate": np.nan}
    hits = sum(book_id in relevant for book_id in selected)
    return {
        f"{slot}_precision": hits / len(selected),
        f"{slot}_hit_rate": float(hits > 0),
    }


def _ranked_candidates(
    candidate_rows: np.ndarray,
    scores: np.ndarray,
    book_ids: np.ndarray,
    k: int,
) -> np.ndarray:
    if not len(candidate_rows):
        return np.array([], dtype=np.int64)
    ids = np.asarray(book_ids, dtype=str)[candidate_rows]
    order = np.lexsort((ids, -np.asarray(scores, dtype=np.float64)))
    return candidate_rows[order[:k]]


def baseline_recommendations(
    recommender: Recommender,
    popularity_count: np.ndarray,
    average_rating: np.ndarray,
    consumed: set[str],
    train_genres: np.ndarray,
    user_id: str,
    k: int,
) -> dict[str, list[str]]:
    """Return B0 random, B1 global popularity and B2 genre popularity."""
    rows = np.nonzero(recommender.eligible_mask)[0]
    if consumed:
        rows = np.asarray(
            [row for row in rows if recommender.book_ids[row] not in consumed], dtype=np.int64
        )
    pop_score = np.log1p(popularity_count[rows]) * average_rating[rows]

    seed = RANDOM_STATE + zlib.crc32(user_id.encode("utf-8"))
    rng = np.random.default_rng(seed)
    random_rows = rng.choice(rows, size=min(k, len(rows)), replace=False) if len(rows) else rows
    popular_rows = _ranked_candidates(rows, pop_score, recommender.book_ids, k)

    genre_flags = recommender.genres.reindex(recommender.book_ids)[GENRE_COLUMNS].fillna(0).to_numpy()
    if train_genres.any():
        genre_rows = rows[(genre_flags[rows] @ train_genres.astype(int)) > 0]
    else:
        genre_rows = rows
    genre_score = (
        np.log1p(popularity_count[genre_rows]) * average_rating[genre_rows]
        if len(genre_rows)
        else np.array([])
    )
    genre_popular_rows = _ranked_candidates(
        genre_rows,
        genre_score,
        recommender.book_ids,
        k,
    )

    return {
        "B0_random": recommender.book_ids[random_rows].tolist(),
        "B1_popularity": recommender.book_ids[popular_rows].tolist(),
        "B2_genre_popularity": recommender.book_ids[genre_popular_rows].tolist(),
    }


def evaluate_temporal(
    interactions: pd.DataFrame,
    recommender: Recommender,
    popularity_count: np.ndarray | None = None,
    average_rating: np.ndarray | None = None,
    train_fraction: float = 0.8,
    ks: Sequence[int] | None = None,
    temporal_cutoff: pd.Timestamp | None = None,
    catalog_available: np.ndarray | None = None,
    invalid_date_count: int = 0,
    evaluation_mode: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate with fold-local or historical popularity, never full-catalog future aggregates."""
    cutoffs = sorted({int(k) for k in (ks or (recommender.config.k,)) if int(k) > 0})
    if not cutoffs:
        raise ValueError("At least one positive k is required.")

    if temporal_cutoff is None:
        train, future = temporal_split(interactions, train_fraction)
    else:
        train, future = global_temporal_split(interactions, temporal_cutoff)
    if popularity_count is None or average_rating is None:
        popularity_count, average_rating = popularity_from_training(train, recommender.book_ids)
    popularity_count = np.asarray(popularity_count, dtype=np.float64)
    average_rating = np.asarray(average_rating, dtype=np.float64)
    if len(popularity_count) != len(recommender.book_ids) or len(average_rating) != len(
        recommender.book_ids
    ):
        raise ValueError("Historical popularity arrays must align 1:1 with recommender.book_ids.")
    evaluation_eligible_mask = recommender.eligible_mask.copy()
    if catalog_available is not None:
        available = np.asarray(catalog_available, dtype=bool)
        if len(available) != len(recommender.book_ids):
            raise ValueError("catalog_available must align 1:1 with recommender.book_ids.")
        evaluation_eligible_mask &= available
    available_book_ids = set(recommender.book_ids[evaluation_eligible_mask])

    train_groups = {uid: group for uid, group in train.groupby("user_id", sort=False)}
    future_groups = {uid: group for uid, group in future.groupby("user_id", sort=False)}
    genre_flags = recommender.genres.reindex(recommender.book_ids)[GENRE_COLUMNS].fillna(0).to_numpy()

    rows: list[dict] = []
    recommended_by_system: dict[tuple[str, int], set[str]] = {}
    popularity_by_book = dict(zip(recommender.book_ids, popularity_count, strict=False))
    historical_segments = np.full(len(popularity_count), "unknown", dtype=object)
    eligible_segments, historical_tail_cut, historical_head_cut = popularity_segments(
        popularity_count[evaluation_eligible_mask],
        recommender.config.popularity_tail_quantile,
        recommender.config.popularity_head_quantile,
    )
    historical_segments[evaluation_eligible_mask] = eligible_segments
    segment_by_book = dict(
        zip(recommender.book_ids, historical_segments, strict=False)
    )
    total_popularity = float(popularity_count.sum())
    popularity_denominator = total_popularity + len(popularity_count)
    original_segments = recommender.popularity_segment
    original_tail_cut = recommender.popularity_tail_cut
    original_head_cut = recommender.popularity_head_cut
    original_eligible_mask = recommender.eligible_mask
    original_ratings_count = recommender.ratings_count
    recommender.popularity_segment = historical_segments
    recommender.popularity_tail_cut = historical_tail_cut
    recommender.popularity_head_cut = historical_head_cut
    recommender.eligible_mask = evaluation_eligible_mask
    recommender.ratings_count = popularity_count

    try:
        for user_id, train_user in train_groups.items():
            future_user = future_groups.get(user_id)
            if future_user is None:
                continue
            train_positive = train_user.loc[_positive_mask(train_user)]
            future_positive = future_user.loc[_positive_mask(future_user)]
            relevant = set(future_positive["book_id"].astype(str)) & available_book_ids
            if train_positive.empty or not relevant:
                continue

            consumed = _consumed_books(train_user)
            positive_ids = train_positive["book_id"].astype(str).tolist()
            engagement = compute_engagement_weight(
                pd.to_numeric(train_positive["rating_clean"], errors="coerce").to_numpy(dtype=float),
                train_positive["has_review_text"].fillna(False).to_numpy(dtype=bool),
                pd.to_numeric(
                    train_positive["reading_duration_days"], errors="coerce"
                ).to_numpy(dtype=float),
            )
            modes = recommender.modes_from_history(positive_ids, engagement)
            if modes is None:
                continue

            positive_rows = [
                recommender._book_row[book_id]
                for book_id in positive_ids
                if book_id in recommender._book_row
            ]
            train_genres = (
                genre_flags[np.asarray(positive_rows)].max(axis=0)
                if positive_rows
                else np.zeros(len(GENRE_COLUMNS), dtype=int)
            )
            for k in cutoffs:
                original_config = recommender.config
                recommender.config = replace(
                    original_config,
                    k=k,
                    explore_slots=min(original_config.explore_slots, k),
                )
                try:
                    model = recommender.recommend_from_modes(
                        str(user_id), modes[0], modes[1], consumed
                    )
                finally:
                    recommender.config = original_config

                systems = {"model": model["book_id"].astype(str).tolist()}
                systems.update(
                    baseline_recommendations(
                        recommender,
                        popularity_count,
                        average_rating,
                        consumed,
                        train_genres,
                        str(user_id),
                        k,
                    )
                )

                model_slot_metrics = {
                    **_slot_metrics(model, relevant, "interest"),
                    **_slot_metrics(model, relevant, "exploration"),
                }
                for system, recommended in systems.items():
                    metrics = _binary_metrics(recommended, relevant, k)
                    segments = [segment_by_book[book_id] for book_id in recommended]
                    popularity = [math.log1p(popularity_by_book[book_id]) for book_id in recommended]
                    novelty = [
                        -math.log2((popularity_by_book[book_id] + 1.0) / popularity_denominator)
                        for book_id in recommended
                    ]
                    recommended_by_system.setdefault((system, k), set()).update(recommended)
                    rows.append(
                        {
                            "user_id": str(user_id),
                            "system": system,
                            "k": k,
                            "relevant_count": len(relevant),
                            **metrics,
                            "diversity": _intra_list_diversity(recommender, recommended),
                            "avg_recommendation_popularity": (
                                float(np.mean(popularity)) if popularity else 0.0
                            ),
                            "novelty": float(np.mean(novelty)) if novelty else 0.0,
                            "tail_share": segments.count("tail") / len(segments) if segments else 0.0,
                            "mid_share": segments.count("mid") / len(segments) if segments else 0.0,
                            "head_share": segments.count("head") / len(segments) if segments else 0.0,
                            **(
                                model_slot_metrics
                                if system == "model"
                                else {
                                    "interest_precision": np.nan,
                                    "interest_hit_rate": np.nan,
                                    "exploration_precision": np.nan,
                                    "exploration_hit_rate": np.nan,
                                }
                            ),
                        }
                    )
    finally:
        recommender.popularity_segment = original_segments
        recommender.popularity_tail_cut = original_tail_cut
        recommender.popularity_head_cut = original_head_cut
        recommender.eligible_mask = original_eligible_mask
        recommender.ratings_count = original_ratings_count

    per_user = pd.DataFrame(rows)
    if per_user.empty:
        return pd.DataFrame(), per_user

    eligible_books = available_book_ids
    eligible_tail = {
        book_id
        for book_id in eligible_books
        if segment_by_book.get(book_id) == "tail"
    }
    cutoff_text = _utc_timestamp(temporal_cutoff).isoformat() if temporal_cutoff is not None else ""
    resolved_evaluation_mode = evaluation_mode or (
        EVALUATION_MODE
        if temporal_cutoff is not None
        else "per_user_temporal_split_training_snapshot"
    )
    users_evaluable = int(per_user["user_id"].nunique())
    metadata = {
        "temporal_cutoff": cutoff_text,
        "evaluation_mode": resolved_evaluation_mode,
        "books_available": int(evaluation_eligible_mask.sum()),
        "users_evaluable": users_evaluable,
        "invalid_dates_discarded": int(invalid_date_count),
    }
    for key, value in metadata.items():
        per_user[key] = value

    summary_rows = []
    for (system, k), group in per_user.groupby(["system", "k"], sort=False):
        exposed = recommended_by_system[(system, int(k))]
        summary_rows.append(
            {
                "system": system,
                "k": int(k),
                "users": int(group["user_id"].nunique()),
                "recall": float(group["recall"].mean()),
                "precision": float(group["precision"].mean()),
                "ndcg": float(group["ndcg"].mean()),
                "map": float(group["average_precision"].mean()),
                "diversity": float(group["diversity"].mean()),
                "catalog_coverage": len(exposed) / len(eligible_books) if eligible_books else 0.0,
                "long_tail_coverage": (
                    len(exposed & eligible_tail) / len(eligible_tail) if eligible_tail else 0.0
                ),
                "avg_recommendation_popularity": float(
                    group["avg_recommendation_popularity"].mean()
                ),
                "novelty": float(group["novelty"].mean()),
                "tail_share": float(group["tail_share"].mean()),
                "mid_share": float(group["mid_share"].mean()),
                "head_share": float(group["head_share"].mean()),
                "interest_precision": float(group["interest_precision"].mean()),
                "interest_hit_rate": float(group["interest_hit_rate"].mean()),
                "exploration_precision": float(group["exploration_precision"].mean()),
                "exploration_hit_rate": float(group["exploration_hit_rate"].mean()),
                **metadata,
            }
        )
    return pd.DataFrame(summary_rows), per_user


def collect_valid_user_ids(path: Path, max_users: int) -> list[str]:
    """Return a deterministic bounded cohort from the global K-core valid users."""
    valid = pd.read_parquet(path, columns=["user_id", "valid"])
    return (
        valid.loc[valid["valid"].fillna(False), "user_id"]
        .astype(str)
        .sort_values(kind="stable")
        .head(max_users)
        .tolist()
    )


def collect_users(path: Path, user_ids: Sequence[str]) -> pd.DataFrame:
    """Scan the canonical parquet while retaining complete histories for selected users."""
    selected = {str(user_id) for user_id in user_ids}
    if not selected:
        return pd.DataFrame(columns=INTERACTION_COLUMNS)
    chunks: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=INTERACTION_COLUMNS, batch_size=250_000):
        frame = batch.to_pandas()
        frame["user_id"] = frame["user_id"].astype(str)
        kept = frame[frame["user_id"].isin(selected)]
        if len(kept):
            chunks.append(kept)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=INTERACTION_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=1_000)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--cutoff", type=str)
    parser.add_argument("--output", type=Path, default=EVALUATION_OUTPUT)
    args = parser.parse_args()

    recommender = Recommender.from_artifacts()
    valid_user_ids = collect_valid_user_ids(USER_FEATURES_GLOBAL_PATH, args.max_users)
    interactions = collect_users(INTERACTIONS_CURATED_PATH, valid_user_ids)
    cutoff = (
        _utc_timestamp(args.cutoff)
        if args.cutoff
        else choose_global_cutoff(interactions, args.train_fraction)
    )
    if cutoff < MIN_VALID_DATE:
        raise ValueError(f"--cutoff must be on or after {MIN_VALID_DATE.date()}.")
    snapshot = historical_popularity_snapshot(
        INTERACTIONS_CURATED_PATH,
        recommender.book_ids,
        cutoff,
    )
    catalog_available = historical_catalog_mask(
        BOOKS_MASTER_PATH,
        recommender.book_ids,
        cutoff,
        snapshot.first_observed,
    )
    summary, _ = evaluate_temporal(
        interactions,
        recommender,
        snapshot.rating_count,
        snapshot.average_rating,
        args.train_fraction,
        ks=args.k,
        temporal_cutoff=cutoff,
        catalog_available=catalog_available,
        invalid_date_count=snapshot.invalid_date_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Temporal cutoff: {cutoff}")
    print(summary.to_string(index=False))
    print(f"Wrote temporal evaluation to {args.output}")


if __name__ == "__main__":
    main()
