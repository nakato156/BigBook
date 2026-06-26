"""Collaborative scoring helpers shared by offline experiments and the ranker."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


def percentile_scores(values: np.ndarray) -> np.ndarray:
    """Map one candidate-score vector to deterministic percentiles in ``[0, 1]``.

    Ties receive their average rank. A constant non-empty vector is neutral (0.5) so
    an uninformative signal cannot reorder the content ranking.
    """
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("Collaborative/content scores must be a one-dimensional vector.")
    if not np.isfinite(scores).all():
        raise ValueError("Collaborative/content scores must contain only finite values.")
    if len(scores) == 0:
        return scores.copy()
    if np.all(scores == scores[0]):
        return np.full(len(scores), 0.5, dtype=np.float64)
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    return (ranks - 1.0) / (len(scores) - 1.0)


def blend_percentile_scores(
    content_scores: np.ndarray,
    collaborative_scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend content and collaborative evidence after per-pool percentile calibration."""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("blend_alpha must be between 0.0 and 1.0.")
    content = np.asarray(content_scores, dtype=np.float64)
    collaborative = np.asarray(collaborative_scores, dtype=np.float64)
    if content.shape != collaborative.shape:
        raise ValueError(
            "additional_score_fn must return exactly one score per retrieved candidate."
        )
    return (
        float(alpha) * percentile_scores(content)
        + (1.0 - float(alpha)) * percentile_scores(collaborative)
    )


class CooccurrenceIndex:
    """Sparse pair lookup for max-PMI scoring against a user's positive history."""

    def __init__(self, pairs: pd.DataFrame) -> None:
        required = {"book_id_a", "book_id_b", "pmi"}
        missing = required - set(pairs.columns)
        if missing:
            raise ValueError(f"Co-occurrence artifact is missing columns: {sorted(missing)}")
        self.neighbors: dict[str, dict[str, float]] = {}
        for row in pairs.loc[:, ["book_id_a", "book_id_b", "pmi"]].itertuples(index=False):
            left, right, pmi = str(row.book_id_a), str(row.book_id_b), float(row.pmi)
            if not np.isfinite(pmi) or pmi < 0.0:
                raise ValueError("Co-occurrence PMI values must be finite and non-negative.")
            self.neighbors.setdefault(left, {})[right] = pmi
            self.neighbors.setdefault(right, {})[left] = pmi

    def score(self, candidate_ids: Sequence[str], positive_book_ids: Sequence[str]) -> np.ndarray:
        positives = {str(book_id) for book_id in positive_book_ids}
        scores = np.zeros(len(candidate_ids), dtype=np.float64)
        for idx, candidate in enumerate(candidate_ids):
            candidate_neighbors = self.neighbors.get(str(candidate), {})
            scores[idx] = max(
                (candidate_neighbors.get(positive, 0.0) for positive in positives),
                default=0.0,
            )
        return scores


def score_map_callback(
    book_ids: np.ndarray,
    scores_by_user: Mapping[str, Mapping[str, float]],
) -> Callable[[np.ndarray, str], np.ndarray]:
    """Adapt ``{user: {book: score}}`` into the ranker's row-based callback contract."""
    aligned_book_ids = np.asarray(book_ids, dtype=str)

    def callback(rows: np.ndarray, user_id: str) -> np.ndarray:
        user_scores = scores_by_user.get(str(user_id), {})
        return np.fromiter(
            (float(user_scores.get(str(aligned_book_ids[int(row)]), 0.0)) for row in rows),
            dtype=np.float64,
            count=len(rows),
        )

    return callback

