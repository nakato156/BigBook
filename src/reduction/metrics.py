"""N0 ranking metrics (recall/precision/NDCG/MAP, candidate recall, diversity) and
bootstrap confidence intervals for temporal evaluation. Baselines live in
:mod:`src.reduction.baselines`; habit-proxy (N1) metrics live in
:mod:`src.reduction.habit_proxies`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.reduction.recommend import Recommender
from src.reduction.temporal_split import RANDOM_STATE


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


def _candidate_recall(candidate_ids: set[str], relevant: set[str]) -> float:
    """Fraction of the relevant holdout that retrieval actually surfaced as candidates."""
    if not relevant:
        return 0.0
    return len(candidate_ids & relevant) / len(relevant)


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


def bootstrap_confidence_intervals(
    per_user: pd.DataFrame,
    metrics: Sequence[str] = ("recall", "precision", "ndcg", "average_precision"),
    n_resamples: int = 1_000,
    confidence: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Bootstrap user-level metric means by ``(system, k)``."""
    if per_user.empty:
        return pd.DataFrame(
            columns=["system", "k", "metric", "users", "mean", "ci_low", "ci_high"]
        )
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    available = [metric for metric in metrics if metric in per_user.columns]
    if not available:
        raise ValueError("None of the requested bootstrap metrics exist in per_user.")

    rng = np.random.default_rng(random_state)
    alpha = (1.0 - confidence) / 2.0
    rows: list[dict] = []
    for (system, k), group in per_user.groupby(["system", "k"], sort=True):
        users = group["user_id"].astype(str).to_numpy()
        if pd.Index(users).duplicated().any():
            raise ValueError("per_user must contain at most one row per user/system/k.")
        n_users = len(group)
        sampled = rng.integers(0, n_users, size=(n_resamples, n_users))
        for metric in available:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values) != n_users:
                continue
            bootstrap_means = values[sampled].mean(axis=1)
            rows.append(
                {
                    "system": str(system),
                    "k": int(k),
                    "metric": metric,
                    "users": n_users,
                    "mean": float(values.mean()),
                    "ci_low": float(np.quantile(bootstrap_means, alpha)),
                    "ci_high": float(np.quantile(bootstrap_means, 1.0 - alpha)),
                    "confidence": float(confidence),
                    "n_resamples": int(n_resamples),
                    "random_state": int(random_state),
                }
            )
    return pd.DataFrame(rows)
