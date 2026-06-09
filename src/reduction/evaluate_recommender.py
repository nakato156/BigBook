"""Temporal offline evaluation for the recommender and B0/B1/B2 baselines."""

from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import BOOKS_MASTER_PATH, INTERACTIONS_CURATED_PATH, PROJECT_ROOT
from src.reduction.build_user_centroids import compute_engagement_weight
from src.reduction.recommend import GENRE_COLUMNS, Recommender

RANDOM_STATE = 42
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


def temporal_split(
    interactions: pd.DataFrame, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological per-user split with at least one row on each side."""
    data = interactions.copy()
    data["user_id"] = data["user_id"].astype(str)
    data["book_id"] = data["book_id"].astype(str)
    data["date_added"] = pd.to_datetime(data["date_added"], errors="coerce")
    data = data.sort_values(["user_id", "date_added"], kind="stable", na_position="last")
    position = data.groupby("user_id").cumcount()
    size = data.groupby("user_id")["user_id"].transform("size")
    cutoff = np.floor(size * train_fraction).astype(int).clip(lower=1)
    cutoff = np.minimum(cutoff, size - 1)
    train_mask = position < cutoff
    return data.loc[train_mask].copy(), data.loc[~train_mask].copy()


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
    return {"recall": recall, "precision": precision, "ndcg": dcg / idcg if idcg else 0.0}


def _ranked_candidates(
    candidate_rows: np.ndarray, scores: np.ndarray, k: int
) -> np.ndarray:
    if not len(candidate_rows):
        return np.array([], dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    return candidate_rows[order[:k]]


def baseline_recommendations(
    recommender: Recommender,
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
    pop_score = np.log1p(recommender.ratings_count[rows]) * average_rating[rows]

    seed = RANDOM_STATE + zlib.crc32(user_id.encode("utf-8"))
    rng = np.random.default_rng(seed)
    random_rows = rng.choice(rows, size=min(k, len(rows)), replace=False) if len(rows) else rows
    popular_rows = _ranked_candidates(rows, pop_score, k)

    genre_flags = recommender.genres.reindex(recommender.book_ids)[GENRE_COLUMNS].fillna(0).to_numpy()
    if train_genres.any():
        genre_rows = rows[(genre_flags[rows] @ train_genres.astype(int)) > 0]
    else:
        genre_rows = rows
    genre_score = (
        np.log1p(recommender.ratings_count[genre_rows]) * average_rating[genre_rows]
        if len(genre_rows)
        else np.array([])
    )
    genre_popular_rows = _ranked_candidates(genre_rows, genre_score, k)

    return {
        "B0_random": recommender.book_ids[random_rows].tolist(),
        "B1_popularity": recommender.book_ids[popular_rows].tolist(),
        "B2_genre_popularity": recommender.book_ids[genre_popular_rows].tolist(),
    }


def evaluate_temporal(
    interactions: pd.DataFrame,
    recommender: Recommender,
    average_rating: np.ndarray,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate model and baselines under one chronological protocol."""
    train, future = temporal_split(interactions, train_fraction)
    train_groups = {uid: group for uid, group in train.groupby("user_id", sort=False)}
    future_groups = {uid: group for uid, group in future.groupby("user_id", sort=False)}
    genre_flags = recommender.genres.reindex(recommender.book_ids)[GENRE_COLUMNS].fillna(0).to_numpy()

    rows: list[dict] = []
    recommended_by_system: dict[str, set[str]] = {}
    popularity_by_book = dict(
        zip(recommender.book_ids, recommender.ratings_count, strict=False)
    )
    segment_by_book = dict(
        zip(recommender.book_ids, recommender.popularity_segment, strict=False)
    )
    total_popularity = float(np.asarray(recommender.ratings_count, dtype=np.float64).sum())
    popularity_denominator = total_popularity + len(recommender.ratings_count)

    for user_id, train_user in train_groups.items():
        future_user = future_groups.get(user_id)
        if future_user is None:
            continue
        train_positive = train_user.loc[_positive_mask(train_user)]
        future_positive = future_user.loc[_positive_mask(future_user)]
        relevant = set(future_positive["book_id"].astype(str))
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

        model = recommender.recommend_from_modes(
            str(user_id), modes[0], modes[1], consumed
        )
        systems = {"model": model["book_id"].astype(str).tolist()}

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
        systems.update(
            baseline_recommendations(
                recommender,
                average_rating,
                consumed,
                train_genres,
                str(user_id),
                recommender.config.k,
            )
        )

        for system, recommended in systems.items():
            metrics = _binary_metrics(recommended, relevant, recommender.config.k)
            segments = [segment_by_book[book_id] for book_id in recommended]
            popularity = [math.log1p(popularity_by_book[book_id]) for book_id in recommended]
            novelty = [
                -math.log2((popularity_by_book[book_id] + 1.0) / popularity_denominator)
                for book_id in recommended
            ]
            recommended_by_system.setdefault(system, set()).update(recommended)
            rows.append(
                {
                    "user_id": str(user_id),
                    "system": system,
                    **metrics,
                    "avg_recommendation_popularity": float(np.mean(popularity)) if popularity else 0.0,
                    "novelty": float(np.mean(novelty)) if novelty else 0.0,
                    "tail_share": segments.count("tail") / len(segments) if segments else 0.0,
                    "mid_share": segments.count("mid") / len(segments) if segments else 0.0,
                    "head_share": segments.count("head") / len(segments) if segments else 0.0,
                }
            )

    per_user = pd.DataFrame(rows)
    if per_user.empty:
        return pd.DataFrame(), per_user

    eligible_books = set(recommender.book_ids[recommender.eligible_mask])
    eligible_tail = {
        book_id
        for book_id in eligible_books
        if segment_by_book.get(book_id) == "tail"
    }
    summary_rows = []
    for system, group in per_user.groupby("system", sort=False):
        exposed = recommended_by_system[system]
        summary_rows.append(
            {
                "system": system,
                "users": int(group["user_id"].nunique()),
                "recall": float(group["recall"].mean()),
                "precision": float(group["precision"].mean()),
                "ndcg": float(group["ndcg"].mean()),
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
            }
        )
    return pd.DataFrame(summary_rows), per_user


def collect_users(path: Path, max_users: int) -> pd.DataFrame:
    """Scan the canonical parquet while retaining complete histories for a bounded user set."""
    selected: set[str] = set()
    chunks: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=INTERACTION_COLUMNS, batch_size=250_000):
        frame = batch.to_pandas()
        frame["user_id"] = frame["user_id"].astype(str)
        if len(selected) < max_users:
            for user_id in frame["user_id"].drop_duplicates():
                selected.add(user_id)
                if len(selected) >= max_users:
                    break
        kept = frame[frame["user_id"].isin(selected)]
        if len(kept):
            chunks.append(kept)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=INTERACTION_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=1_000)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=EVALUATION_OUTPUT)
    args = parser.parse_args()

    recommender = Recommender.from_artifacts()
    interactions = collect_users(INTERACTIONS_CURATED_PATH, args.max_users)
    books = pd.read_parquet(BOOKS_MASTER_PATH, columns=["book_id", "average_rating"])
    books["book_id"] = books["book_id"].astype(str)
    average_rating = (
        books.set_index("book_id")
        .reindex(recommender.book_ids)["average_rating"]
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    summary, _ = evaluate_temporal(
        interactions, recommender, average_rating, args.train_fraction
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote temporal evaluation to {args.output}")


if __name__ == "__main__":
    main()
