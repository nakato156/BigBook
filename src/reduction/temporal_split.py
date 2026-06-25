"""Chronological train/future splitting for temporal offline evaluation.

Shared by :mod:`src.reduction.baselines`, :mod:`src.reduction.habit_proxies` and
:mod:`src.reduction.evaluate_recommender` — kept dependency-free (no ``Recommender``
import) so it has no cycle risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 42
MIN_VALID_DATE = pd.Timestamp("2006-01-01", tz="UTC")


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
