"""B0/B1/B2 baseline rankers and the historical popularity/catalog snapshots they're
built from. Ranking metrics live in :mod:`src.reduction.metrics`; the evaluation driver
that calls these lives in :mod:`src.reduction.evaluate_recommender`.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.reduction.metrics import _ranked_candidates
from src.reduction.recommend import GENRE_COLUMNS, Recommender
from src.reduction.temporal_split import MIN_VALID_DATE, RANDOM_STATE, _utc_timestamp, _valid_dates

POPULARITY_COLUMNS = ["book_id", "rating_clean", "has_review_text", "date_added"]


@dataclass(frozen=True)
class HistoricalSnapshot:
    """Catalog evidence observed in the canonical interaction log."""

    rating_count: np.ndarray
    average_rating: np.ndarray
    review_count: np.ndarray
    first_observed: np.ndarray
    invalid_date_count: int


@dataclass(frozen=True)
class BaselineRankings:
    """Precomputed historical rankings shared by every evaluated user."""

    eligible_rows: np.ndarray
    global_popularity_rows: np.ndarray
    genre_popularity_rows: dict[int, np.ndarray]


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
    progress_every_batches: int = 0,
) -> HistoricalSnapshot:
    """Aggregate rating evidence and first observation from one canonical scan."""
    cutoff = _utc_timestamp(cutoff)
    if cutoff < MIN_VALID_DATE:
        raise ValueError(f"cutoff must be on or after {MIN_VALID_DATE.date()}.")
    book_row = {str(book_id): row for row, book_id in enumerate(book_ids)}
    counts = np.zeros(len(book_ids), dtype=np.int64)
    sums = np.zeros(len(book_ids), dtype=np.float64)
    review_counts = np.zeros(len(book_ids), dtype=np.int64)
    first_observed_ns = np.full(len(book_ids), np.iinfo(np.int64).max, dtype=np.int64)
    invalid_date_count = 0

    parquet = pq.ParquetFile(interactions_path)
    for batch_idx, batch in enumerate(
        parquet.iter_batches(columns=POPULARITY_COLUMNS, batch_size=batch_size),
        start=1,
    ):
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
        reviews = frame["has_review_text"].fillna(False).to_numpy(dtype=bool)
        review_keep = valid_dates & (dates <= cutoff).to_numpy() & reviews
        if review_keep.any():
            review_rows = mapped.loc[review_keep]
            review_rows = review_rows.loc[review_rows.notna()].to_numpy(dtype=np.int64)
            np.add.at(review_counts, review_rows, 1)
        keep = valid_dates & (dates <= cutoff).to_numpy() & ratings.notna().to_numpy()
        if not keep.any():
            if progress_every_batches and batch_idx % progress_every_batches == 0:
                print(
                    "[ranker_grid] historical snapshot "
                    f"batch={batch_idx:,}, rated_rows_seen={int(counts.sum()):,}, "
                    f"invalid_dates={invalid_date_count:,}",
                    flush=True,
                )
            continue
        rated_rows = mapped.loc[keep]
        present = rated_rows.notna()
        rows = rated_rows.loc[present].to_numpy(dtype=np.int64)
        values = ratings.loc[keep].loc[present].to_numpy(dtype=np.float64)
        np.add.at(counts, rows, 1)
        np.add.at(sums, rows, values)
        if progress_every_batches and batch_idx % progress_every_batches == 0:
            print(
                "[ranker_grid] historical snapshot "
                f"batch={batch_idx:,}, rated_rows_seen={int(counts.sum()):,}, "
                f"invalid_dates={invalid_date_count:,}",
                flush=True,
            )

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
        review_count=review_counts.astype(np.float64),
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


def prepare_baseline_rankings(
    recommender: Recommender,
    popularity_count: np.ndarray,
    average_rating: np.ndarray,
) -> BaselineRankings:
    """Sort B1 and every B2 genre combination once per evaluation run."""
    rows = np.nonzero(recommender.eligible_mask)[0]
    pop_score = np.log1p(popularity_count[rows]) * average_rating[rows]
    global_rows = _ranked_candidates(
        rows,
        pop_score,
        recommender.book_ids,
        len(rows),
    )
    genre_flags = (
        recommender.genres.reindex(recommender.book_ids)[GENRE_COLUMNS]
        .fillna(0)
        .to_numpy(dtype=np.int8)
    )
    genre_orders: dict[int, np.ndarray] = {0: global_rows}
    for mask in range(1, 1 << len(GENRE_COLUMNS)):
        selected = np.array(
            [(mask >> bit) & 1 for bit in range(len(GENRE_COLUMNS))],
            dtype=np.int8,
        )
        genre_rows = rows[(genre_flags[rows] @ selected) > 0]
        scores = np.log1p(popularity_count[genre_rows]) * average_rating[genre_rows]
        genre_orders[mask] = _ranked_candidates(
            genre_rows,
            scores,
            recommender.book_ids,
            len(genre_rows),
        )
    return BaselineRankings(rows, global_rows, genre_orders)


def _take_unconsumed(
    ordered_rows: np.ndarray,
    book_ids: np.ndarray,
    consumed: set[str],
    k: int,
) -> list[str]:
    selected: list[str] = []
    for row in ordered_rows:
        book_id = str(book_ids[int(row)])
        if book_id not in consumed:
            selected.append(book_id)
            if len(selected) >= k:
                break
    return selected


def _random_unconsumed(
    rows: np.ndarray,
    book_ids: np.ndarray,
    consumed: set[str],
    user_id: str,
    k: int,
) -> list[str]:
    seed = RANDOM_STATE + zlib.crc32(user_id.encode("utf-8"))
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    seen: set[int] = set()
    while len(selected) < k and len(seen) < len(rows):
        position = int(rng.integers(0, len(rows)))
        if position in seen:
            continue
        seen.add(position)
        book_id = str(book_ids[int(rows[position])])
        if book_id not in consumed:
            selected.append(book_id)
    return selected


def baseline_recommendations(
    recommender: Recommender,
    popularity_count: np.ndarray,
    average_rating: np.ndarray,
    consumed: set[str],
    train_genres: np.ndarray,
    user_id: str,
    k: int,
    rankings: BaselineRankings | None = None,
) -> dict[str, list[str]]:
    """Return B0 random, B1 global popularity and B2 genre popularity."""
    rankings = rankings or prepare_baseline_rankings(
        recommender,
        popularity_count,
        average_rating,
    )
    genre_mask = sum(
        (1 << bit) for bit, enabled in enumerate(train_genres) if bool(enabled)
    )

    return {
        "B0_random": _random_unconsumed(
            rankings.eligible_rows,
            recommender.book_ids,
            consumed,
            user_id,
            k,
        ),
        "B1_popularity": _take_unconsumed(
            rankings.global_popularity_rows,
            recommender.book_ids,
            consumed,
            k,
        ),
        "B2_genre_popularity": _take_unconsumed(
            rankings.genre_popularity_rows[genre_mask],
            recommender.book_ids,
            consumed,
            k,
        ),
    }
