"""N1 descriptive reading-habit proxies and prior-activity segmentation.

Correlational only — see ``docs/decisiones_negocio.md`` for why these are reported
alongside (not as a substitute for) the N0 ranking metrics in
:mod:`src.reduction.metrics`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.reduction.recommend import GENRE_COLUMNS
from src.reduction.temporal_split import MIN_VALID_DATE, _utc_timestamp, _valid_dates


def _positive_mask(frame: pd.DataFrame) -> np.ndarray:
    is_read = frame["is_read"].fillna(False).to_numpy(dtype=bool)
    rating = pd.to_numeric(frame["rating_clean"], errors="coerce").to_numpy(dtype=np.float64)
    return is_read & (rating >= 4.0)


def _consumed_books(frame: pd.DataFrame) -> set[str]:
    read = frame["is_read"].fillna(False).to_numpy(dtype=bool)
    return set(frame.loc[read, "book_id"].astype(str))


def habit_proxy_features(
    interactions: pd.DataFrame,
    genres: pd.DataFrame,
    prefix: str,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute descriptive reading-habit proxies for one temporal window."""
    columns = [
        "user_id",
        f"{prefix}_interaction_count",
        f"{prefix}_completed_reads",
        f"{prefix}_active_span_days",
        f"{prefix}_reading_frequency_monthly",
        f"{prefix}_activity_recency_days",
        f"{prefix}_completion_rate",
        f"{prefix}_reading_breadth",
    ]
    if interactions.empty:
        return pd.DataFrame(columns=columns)

    data = interactions.copy()
    data["user_id"] = data["user_id"].astype(str)
    data["book_id"] = data["book_id"].astype(str)
    dates, valid = _valid_dates(data["date_added"])
    data["date_added"] = dates
    data = data.loc[valid].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)

    data["is_read"] = data["is_read"].fillna(False).astype(bool)
    grouped = data.groupby("user_id", sort=False)
    stats = grouped.agg(
        interaction_count=("book_id", "size"),
        completed_reads=("is_read", "sum"),
        first_interaction=("date_added", "min"),
        last_interaction=("date_added", "max"),
    )
    stats["active_span_days"] = (
        stats["last_interaction"] - stats["first_interaction"]
    ).dt.total_seconds() / 86_400.0
    exposure_days = stats["active_span_days"].clip(lower=1.0)
    stats["reading_frequency_monthly"] = (
        stats["completed_reads"] * 30.4375 / exposure_days
    )
    reference = _utc_timestamp(reference_date)
    stats["activity_recency_days"] = (
        reference - stats["last_interaction"]
    ).dt.total_seconds().div(86_400.0).clip(lower=0.0)
    stats["completion_rate"] = stats["completed_reads"] / stats["interaction_count"]

    read_rows = data.loc[data["is_read"], ["user_id", "book_id"]]
    if read_rows.empty:
        breadth = pd.Series(dtype=np.int64, name="reading_breadth")
    else:
        genre_flags = genres.reindex(read_rows["book_id"])[GENRE_COLUMNS].fillna(0).to_numpy()
        read_genres = pd.DataFrame(genre_flags, columns=GENRE_COLUMNS)
        read_genres["user_id"] = read_rows["user_id"].to_numpy()
        breadth = (
            read_genres.groupby("user_id", sort=False)[GENRE_COLUMNS]
            .max()
            .sum(axis=1)
            .astype(np.int64)
            .rename("reading_breadth")
        )
    stats = stats.join(breadth, how="left")
    stats["reading_breadth"] = stats["reading_breadth"].fillna(0).astype(np.int64)
    stats = stats.reset_index()

    rename = {
        column: f"{prefix}_{column}"
        for column in [
            "interaction_count",
            "completed_reads",
            "active_span_days",
            "reading_frequency_monthly",
            "activity_recency_days",
            "completion_rate",
            "reading_breadth",
        ]
    }
    return stats.rename(columns=rename)[columns]


def assign_activity_segments(completed_reads: pd.Series) -> pd.Series:
    """Assign low/mid/high by p33/p67 while keeping equal values together."""
    values = pd.to_numeric(completed_reads, errors="coerce").fillna(0.0)
    if values.empty:
        return pd.Series(dtype="string", index=values.index, name="activity_segment")
    low_cut, high_cut = values.quantile([1 / 3, 2 / 3]).tolist()
    labels = np.full(len(values), "mid", dtype=object)
    labels[values.to_numpy() <= low_cut] = "low"
    labels[values.to_numpy() > high_cut] = "high"
    return pd.Series(labels, index=values.index, dtype="string", name="activity_segment")


def build_habit_proxy_table(
    train: pd.DataFrame,
    future: pd.DataFrame,
    genres: pd.DataFrame,
    temporal_cutoff: pd.Timestamp | None,
) -> pd.DataFrame:
    """Build train/future N1 proxies and prior-activity segments per user."""
    if temporal_cutoff is not None:
        train_reference = _utc_timestamp(temporal_cutoff)
    else:
        train_reference = pd.to_datetime(train["date_added"], utc=True).max()
    future_reference = pd.to_datetime(future["date_added"], utc=True).max()
    if pd.isna(train_reference):
        train_reference = MIN_VALID_DATE
    if pd.isna(future_reference):
        future_reference = train_reference

    train_features = habit_proxy_features(train, genres, "train", train_reference)
    future_features = habit_proxy_features(future, genres, "future", future_reference)
    proxies = train_features.merge(future_features, on="user_id", how="outer")
    numeric = [column for column in proxies.columns if column != "user_id"]
    proxies[numeric] = proxies[numeric].fillna(0.0)
    proxies["activity_segment"] = assign_activity_segments(
        proxies["train_completed_reads"]
    )
    return proxies


def summarize_by_activity(per_user: pd.DataFrame) -> pd.DataFrame:
    """Summarize N0 and descriptive N1 outcomes by prior activity segment."""
    if per_user.empty:
        return pd.DataFrame()
    metrics = [
        "recall",
        "precision",
        "ndcg",
        "average_precision",
        "diversity",
        "novelty",
        "tail_share",
        "mid_share",
        "head_share",
        "train_completed_reads",
        "train_active_span_days",
        "train_reading_frequency_monthly",
        "train_activity_recency_days",
        "train_completion_rate",
        "train_reading_breadth",
        "future_completed_reads",
        "future_active_span_days",
        "future_reading_frequency_monthly",
        "future_activity_recency_days",
        "future_completion_rate",
        "future_reading_breadth",
    ]
    available = [column for column in metrics if column in per_user.columns]
    summary = (
        per_user.groupby(["activity_segment", "system", "k"], observed=True, sort=True)
        .agg(users=("user_id", "nunique"), **{column: (column, "mean") for column in available})
        .reset_index()
    )
    return summary.rename(columns={"average_precision": "map"})
