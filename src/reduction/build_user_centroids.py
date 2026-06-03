"""Build ``user_centroids`` — the **m-vector** shape of the same user profile.

This is the comparison form of the profile produced by ``build_user_matrix.py`` (the
baseline 1-vector). It reads the **same** positives — ``(is_read == True) AND
(rating_clean >= 4)`` — and only changes the **aggregation**: one global mean becomes
``m`` sub-means (sub-centroids), each a mean of the PCA vectors of a cluster of the
user's positive books. ``m = 1`` reproduces the baseline exactly.

Why multi-centroid (see plan business decisions): a single average dilutes distinct
tastes for >90% of users (most span >=2 genres). Splitting into a few sub-centroids
captures real intra-user multimodality and enables cross-genre discovery / "reading
modes" instead of "more of the same".

Engagement is **not** profile geometry (weighting book-by-book does not move the taste
direction — averaging washes per-item weights out). It is reused only as
``centroid_weight``: which taste-mode is the more committed one, where the signal does
**not** wash out because it aggregates per centroid.

Invoke as a module::

    env/bin/python -m src.reduction.build_user_centroids
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import (
    INTERACTIONS_CURATED_PATH,
    MASTER_FEATURE_MATRIX_PATH,
    USER_CENTROIDS_PATH,
)
from src.reduction.build_user_matrix import ItemSpace, pc_columns
from src.utils.io import read_parquet_chunks, safe_write_parquet


POSITIVE_RATING_THRESHOLD = 4.0
COLD_START_MIN = 3
MULTI_CENTROID_MIN_POSITIVES = 6  # positive_count < 6 => m = 1 (graceful fallback)
M_CAP = 4
MIN_BOOKS_PER_CENTROID = 3
RANDOM_STATE = 42
CHUNK_SIZE = 1_000_000

INTERACTION_COLUMNS = [
    "user_id",
    "book_id",
    "is_read",
    "rating_clean",
    "has_review_text",
    "reading_duration_days",
]


# --------------------------------------------------------------------------- #
# Pure, testable helpers
# --------------------------------------------------------------------------- #
def compute_engagement_weight(
    rating_clean: np.ndarray, has_review_text: np.ndarray, reading_duration_days: np.ndarray
) -> np.ndarray:
    """Per-positive engagement weight (>= 1). Used **only** for ``centroid_weight``.

    ``(rating - 3) * (1.3 if reviewed) * (1.2 if 1 <= duration <= 180)``. Never used
    to position a centroid.
    """
    weight = (rating_clean - 3.0).astype(np.float64)
    weight *= np.where(has_review_text, 1.3, 1.0)
    in_window = (reading_duration_days >= 1.0) & (reading_duration_days <= 180.0)
    weight *= np.where(in_window, 1.2, 1.0)
    return weight.astype(np.float32)


def choose_m(n_positive: int) -> int:
    """Adaptive centroid count: ``1`` below the floor, else capped by books-per-centroid."""
    if n_positive < MULTI_CENTROID_MIN_POSITIVES:
        return 1
    return int(min(M_CAP, n_positive // MIN_BOOKS_PER_CENTROID))


def _user_centroids(
    vecs: np.ndarray, eng_w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sub-centroids for one user. Returns ``(pc, n_books, weight, centroid_weight)``."""
    n = vecs.shape[0]
    m = choose_m(n)
    eng_total = float(eng_w.sum())
    if m == 1:
        pc = vecs.mean(axis=0, keepdims=True).astype(np.float32)
        return pc, np.array([n]), np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)

    labels = KMeans(n_clusters=m, random_state=RANDOM_STATE, n_init=10).fit_predict(vecs)
    pc = np.empty((m, vecs.shape[1]), dtype=np.float32)
    n_books = np.empty(m, dtype=np.int64)
    weight = np.empty(m, dtype=np.float32)
    centroid_weight = np.empty(m, dtype=np.float32)
    for c in range(m):
        members = labels == c
        pc[c] = vecs[members].mean(axis=0)
        n_books[c] = int(members.sum())
        weight[c] = n_books[c] / n
        centroid_weight[c] = float(eng_w[members].sum()) / eng_total if eng_total > 0 else weight[c]
    return pc, n_books, weight, centroid_weight


def _variance_captured(vecs: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Fraction of intra-user variance captured by the clustering (1 - within/total)."""
    total = float(((vecs - vecs.mean(axis=0)) ** 2).sum())
    if total == 0.0:
        return 0.0
    within = float(((vecs - centroids[labels]) ** 2).sum())
    return 1.0 - within / total


# --------------------------------------------------------------------------- #
# Collection + aggregation
# --------------------------------------------------------------------------- #
class PositiveCollector:
    """Stream the canonical once, collecting positives into flat in-memory arrays.

    ~38M positives -> roughly (int32 + int32 + float32) per row ≈ 460 MB, well within
    RAM. The KMeans clustering runs **after** collection (never inside the stream).
    """

    def __init__(self) -> None:
        self.code_of: dict[str, int] = {}
        self.user_ids: list[str] = []
        self._codes: list[np.ndarray] = []
        self._rows: list[np.ndarray] = []
        self._eng: list[np.ndarray] = []
        self.dropped_positive_rows = 0

    def _codes_for(self, uids: np.ndarray) -> np.ndarray:
        cats = pd.Categorical(uids)
        global_for_local = np.empty(len(cats.categories), dtype=np.int64)
        for i, value in enumerate(cats.categories):
            code = self.code_of.get(value)
            if code is None:
                code = len(self.user_ids)
                self.code_of[value] = code
                self.user_ids.append(value)
            global_for_local[i] = code
        return global_for_local[cats.codes]

    def update(self, chunk: pd.DataFrame, space: ItemSpace) -> None:
        is_read = chunk["is_read"].fillna(False).to_numpy(dtype=bool)
        rating = pd.to_numeric(chunk["rating_clean"], errors="coerce").to_numpy(dtype=np.float64)
        positive = is_read & (rating >= POSITIVE_RATING_THRESHOLD)
        if not positive.any():
            return
        pos = chunk.loc[positive]
        books = pos["book_id"].astype(str).to_numpy()
        rows = np.fromiter((space.book_to_row.get(b, -1) for b in books), dtype=np.int64, count=len(books))
        present = rows >= 0
        self.dropped_positive_rows += int((~present).sum())
        if not present.any():
            return

        rating_pos = rating[positive][present]
        review = pos["has_review_text"].fillna(False).to_numpy(dtype=bool)[present]
        duration = pd.to_numeric(pos["reading_duration_days"], errors="coerce").to_numpy(dtype=np.float64)[present]
        eng_w = compute_engagement_weight(rating_pos, review, duration)

        self._codes.append(self._codes_for(pos["user_id"].astype(str).to_numpy()[present]).astype(np.int32))
        self._rows.append(rows[present].astype(np.int32))
        self._eng.append(eng_w)

    def collected(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._codes:
            empty = np.array([], dtype=np.int32)
            return empty, empty, np.array([], dtype=np.float32)
        return (
            np.concatenate(self._codes),
            np.concatenate(self._rows),
            np.concatenate(self._eng),
        )


def build_user_centroids(
    feature_matrix: pd.DataFrame,
    interaction_chunks: Iterable[pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collect positives, then aggregate each user into ``m`` sub-centroids."""
    space = ItemSpace(feature_matrix)  # genre flags unused here
    collector = PositiveCollector()
    for chunk in interaction_chunks:
        collector.update(chunk, space)
    codes, rows, eng = collector.collected()

    out_user: list[str] = []
    out_centroid_id: list[int] = []
    out_n_books: list[int] = []
    out_weight: list[float] = []
    out_centroid_weight: list[float] = []
    out_pc: list[np.ndarray] = []
    m_counter: dict[int, int] = {}

    if len(codes):
        order = np.argsort(codes, kind="stable")
        codes_s = codes[order]
        rows_s = rows[order]
        eng_s = eng[order]
        uniq, starts = np.unique(codes_s, return_index=True)
        bounds = np.append(starts, len(codes_s))
        for i, code in enumerate(uniq):
            sl = slice(bounds[i], bounds[i + 1])
            vecs = space.pc[rows_s[sl]]
            pc, n_books, weight, centroid_weight = _user_centroids(vecs, eng_s[sl])
            m = len(pc)
            m_counter[m] = m_counter.get(m, 0) + 1
            user_id = collector.user_ids[code]
            for c in range(m):
                out_user.append(user_id)
                out_centroid_id.append(c)
                out_n_books.append(int(n_books[c]))
                out_weight.append(float(weight[c]))
                out_centroid_weight.append(float(centroid_weight[c]))
                out_pc.append(pc[c])

    pc_cols = space.pc_cols
    if out_pc:
        pc_array = np.vstack(out_pc).astype(np.float32)
    else:
        pc_array = np.empty((0, len(pc_cols)), dtype=np.float32)
    centroids = pd.DataFrame(pc_array, columns=pc_cols)
    centroids.insert(0, "centroid_weight", np.asarray(out_centroid_weight, dtype=np.float32))
    centroids.insert(0, "weight", np.asarray(out_weight, dtype=np.float32))
    centroids.insert(0, "n_books", np.asarray(out_n_books, dtype=np.int64))
    centroids.insert(0, "centroid_id", np.asarray(out_centroid_id, dtype=np.int64))
    centroids.insert(0, "user_id", np.asarray(out_user, dtype=object))

    n_users = sum(m_counter.values())
    n_multi = sum(count for m, count in m_counter.items() if m > 1)
    diagnostics = {
        "n_users": int(n_users),
        "n_rows": int(len(centroids)),
        "m_distribution": dict(sorted(m_counter.items())),
        "multi_centroid_pct": (100.0 * n_multi / n_users if n_users else 0.0),
        "dropped_positive_rows": collector.dropped_positive_rows,
        "pc_columns": pc_cols,
    }
    return centroids, diagnostics


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _interaction_chunks(path: Path, chunksize: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    yield from read_parquet_chunks(path, chunksize, columns=INTERACTION_COLUMNS)


def print_validations(centroids: pd.DataFrame, diagnostics: dict[str, Any]) -> None:
    print("\nUser centroids build complete")
    print(f"Users: {diagnostics['n_users']:,} | centroid rows: {diagnostics['n_rows']:,}")
    print(f"m distribution: {diagnostics['m_distribution']}")
    print(f"Users with m>1: {diagnostics['multi_centroid_pct']:.2f}%")
    print(f"Positive rows dropped (book_id absent): {diagnostics['dropped_positive_rows']:,}")

    pc_cols = [c for c in centroids.columns if c.startswith("pc_")]
    if pc_cols != diagnostics["pc_columns"]:
        raise ValueError("user_centroids pc_* columns are not aligned 1:1 with the item matrix.")
    if len(centroids):
        weight_sums = centroids.groupby("user_id")["weight"].sum().to_numpy()
        if not np.allclose(weight_sums, 1.0, atol=1e-4):
            raise ValueError("weight does not sum to 1 per user.")
        cw_sums = centroids.groupby("user_id")["centroid_weight"].sum().to_numpy()
        if not np.allclose(cw_sums, 1.0, atol=1e-4):
            raise ValueError("centroid_weight is not normalized to 1 per user.")
        values = centroids[pc_cols].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("user_centroids contains non-finite values.")


def main() -> None:
    if not MASTER_FEATURE_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"{MASTER_FEATURE_MATRIX_PATH} does not exist. "
            "Run `python -m src.reduction.build_master_feature_matrix` first."
        )
    if not INTERACTIONS_CURATED_PATH.exists():
        raise FileNotFoundError(
            f"{INTERACTIONS_CURATED_PATH} does not exist. "
            "Run `python -m src.curation.interactions` first."
        )

    feature_matrix = pd.read_parquet(MASTER_FEATURE_MATRIX_PATH)
    centroids, diagnostics = build_user_centroids(
        feature_matrix, _interaction_chunks(INTERACTIONS_CURATED_PATH)
    )
    safe_write_parquet(centroids, USER_CENTROIDS_PATH)
    print_validations(centroids, diagnostics)
    print(f"User centroids written to: {USER_CENTROIDS_PATH}")


if __name__ == "__main__":
    main()
