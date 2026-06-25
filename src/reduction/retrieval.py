"""Candidate generation primitives: eligibility, popularity segmentation and cluster
retrieve. Pure, testable helpers with no scoring/diversification logic (see
:mod:`src.reduction.ranking` for that) and no orchestration (see
:mod:`src.reduction.recommend` for the :class:`Recommender` that calls these).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def pc_columns(frame: pd.DataFrame) -> list[str]:
    """``pc_*`` columns sorted by integer suffix (matches the item/user matrices)."""
    cols = [c for c in frame.columns if c.startswith("pc_")]
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))


def taste_pc_indices(pc_cols: Sequence[str], tabular_pcs: Iterable[int]) -> np.ndarray:
    """A1: positional indices of the **taste subspace** (``pc_*`` minus the tabular axes).

    Returned indices address columns of a ``pc``-ordered array, so they can slice both the
    book matrix and the user matrix (same column order by construction).
    """
    excluded = set(int(i) for i in tabular_pcs)
    return np.array(
        [i for i, c in enumerate(pc_cols) if int(c.split("_", 1)[1]) not in excluded],
        dtype=np.int64,
    )


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows stay zero (cosine with them is 0, not NaN)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def eligibility_mask(
    book_ids: np.ndarray,
    titles: np.ndarray,
    book_pc: np.ndarray,
    book_cluster: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """A2: technical catalog eligibility without using popularity.

    Eligible books have a non-empty id/title, finite PCA coordinates and a valid cluster.
    """
    ids = np.char.strip(np.asarray(book_ids, dtype=str))
    names = np.asarray(titles, dtype=object)
    clean_names = np.char.strip(np.asarray(names, dtype=str))
    title_ok = pd.notna(names) & (np.char.str_len(clean_names) > 0)
    vector_ok = np.isfinite(np.asarray(book_pc)).all(axis=1)
    clusters = np.asarray(book_cluster)
    cluster_ok = np.isfinite(clusters) & (clusters >= 0) & (clusters < n_clusters)
    return (np.char.str_len(ids) > 0) & title_ok & vector_ok & cluster_ok


def popularity_segments(
    ratings_count: np.ndarray,
    tail_quantile: float = 0.25,
    head_quantile: float = 0.90,
) -> tuple[np.ndarray, float, float]:
    """Label books ``tail``/``mid``/``head`` using current-catalog quantiles."""
    counts = np.asarray(ratings_count, dtype=np.float64)
    finite = counts[np.isfinite(counts)]
    if not len(finite):
        return np.full(len(counts), "unknown", dtype=object), np.nan, np.nan
    tail_cut, head_cut = np.quantile(finite, [tail_quantile, head_quantile])
    labels = np.full(len(counts), "mid", dtype=object)
    labels[counts <= tail_cut] = "tail"
    labels[counts >= head_cut] = "head"
    labels[~np.isfinite(counts)] = "unknown"
    return labels, float(tail_cut), float(head_cut)


def normalized_title_key(title: object) -> str:
    """Small edition-level duplicate key; intentionally conservative and dependency-free."""
    if pd.isna(title):
        return ""
    return " ".join(str(title).lower().strip().split())


def nearest_clusters(user_taste_norm: np.ndarray, centroids_taste_norm: np.ndarray) -> np.ndarray:
    """Cluster ids ordered by cosine closeness of their centroid to the user (retrieve)."""
    sims = centroids_taste_norm @ user_taste_norm
    return np.argsort(-sims)


def retrieve_top_clusters(
    modes_taste_norm: np.ndarray,
    weights: np.ndarray,
    centroids_taste_norm: np.ndarray,
    n_clusters_retrieve: int,
) -> np.ndarray:
    """Top-N cluster ids by max-pooled weighted cosine across taste modes (today's retrieve)."""
    cluster_mode_sim = modes_taste_norm @ centroids_taste_norm.T
    cluster_relevance = (weights[:, None] * cluster_mode_sim).max(axis=0)
    ranked_clusters = np.argsort(-cluster_relevance)
    return ranked_clusters[:n_clusters_retrieve]


def retrieve_clusters_per_mode(
    modes_taste_norm: np.ndarray,
    centroids_taste_norm: np.ndarray,
    clusters_per_mode: int,
    total_budget: int | None = None,
) -> np.ndarray:
    """Each taste mode retrieves its own top clusters; union via round-robin by rank.

    Unlike :func:`retrieve_top_clusters` (max-pooling across modes), no mode is crowded out
    by a stronger one: every mode contributes its own nearest clusters, interleaved
    rank-1-of-each-mode-first so no single mode monopolizes the pool under a shared budget.

    The candidate pool grows with ``n_modes x clusters_per_mode`` (downstream ``mmr_select``
    is ``O(k*n*d)`` over that pool); for this project's expected ranges (<=4 modes,
    ``clusters_per_mode`` ~5-8) this stays small, but it is not a fixed-size pool like
    :func:`retrieve_top_clusters`.
    """
    sims = modes_taste_norm @ centroids_taste_norm.T
    per_mode_ranked = np.argsort(-sims, axis=1)[:, :clusters_per_mode]
    n_modes = per_mode_ranked.shape[0]
    union: list[int] = []
    seen: set[int] = set()
    for rank in range(per_mode_ranked.shape[1]):
        for mode in range(n_modes):
            cluster_id = int(per_mode_ranked[mode, rank])
            if cluster_id not in seen:
                seen.add(cluster_id)
                union.append(cluster_id)
    result = np.array(union, dtype=np.int64)
    if total_budget is not None:
        result = result[:total_budget]
    return result


def consumed_books_for_users(
    interactions_path: Path,
    user_ids: Sequence[str],
    cutoff: pd.Timestamp | None = None,
) -> dict[str, set[str]]:
    """Load completed books for a small user set using parquet predicate pushdown.

    When ``cutoff`` is provided, rows after that timestamp are excluded so collaborative
    evaluation cannot inspect a neighbour's future reading history.
    """
    users = [str(user_id) for user_id in user_ids]
    consumed = {user_id: set() for user_id in users}
    if not users:
        return consumed

    columns = ["user_id", "book_id", "is_read"]
    if cutoff is not None:
        columns.append("date_added")
    table = pq.read_table(
        interactions_path,
        columns=columns,
        filters=[("user_id", "in", users), ("is_read", "=", True)],
    )
    frame = table.to_pandas()
    if frame.empty:
        return consumed
    if cutoff is not None:
        dates = pd.to_datetime(frame["date_added"], errors="coerce", utc=True)
        resolved_cutoff = pd.Timestamp(cutoff)
        resolved_cutoff = (
            resolved_cutoff.tz_localize("UTC")
            if resolved_cutoff.tzinfo is None
            else resolved_cutoff.tz_convert("UTC")
        )
        frame = frame.loc[dates.notna() & (dates <= resolved_cutoff)]

    frame["user_id"] = frame["user_id"].astype(str)
    frame["book_id"] = frame["book_id"].astype(str)
    for user_id, group in frame.groupby("user_id", sort=False):
        consumed[user_id] = set(group["book_id"])
    return consumed
