"""Build the baseline ``user_matrix`` — one taste vector per user, in book PCA space.

The recommender needs a user representation **in the same PCA space as the books**
(``pc_0..pc_N``) so user<->book similarity is a single metric. This module rebuilds
that artifact from scratch as the **centroid of the PCA vectors of the books the user
read and rated positively**. Being a mean of vectors that already live in the item PCA
space, it is aligned by construction — PCA is **not** re-fit and the user is **not**
pushed through ``master_pca_model.joblib``.

Positive (closed decision, see plan): ``(is_read == True) AND (rating_clean >= 4)``.
``is_want_to_read`` is intention, not taste, and is excluded from the vector (it never
overlaps the positive set anyway). Aggregation is a simple mean (``w = 1``).

Scope is **only** the build of the artifacts; split/cohorts/scoring/eval live elsewhere.

Invoke as a module::

    env/bin/python -m src.reduction.build_user_matrix
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from src.config import (
    BOOKS_MASTER_PATH,
    INTERACTIONS_CURATED_PATH,
    MASTER_FEATURE_MATRIX_PATH,
    USER_FEATURES_GLOBAL_PATH,
    USER_MATRIX_PATH,
    USER_META_PATH,
)
from src.utils.io import read_parquet_chunks, safe_write_parquet


POSITIVE_RATING_THRESHOLD = 4.0
COLD_START_MIN = 3
CHUNK_SIZE = 1_000_000

GENRE_COLUMNS = [
    "genre_fantasy",
    "genre_mystery",
    "genre_history",
    "genre_ya",
    "genre_romance",
]

# Columns streamed from the canonical global interactions artifact.
INTERACTION_COLUMNS = [
    "user_id",
    "book_id",
    "is_read",
    "rating_clean",
    "has_review_text",
    "is_want_to_read",
    "date_added",
]


# --------------------------------------------------------------------------- #
# Pure, testable helpers
# --------------------------------------------------------------------------- #
def pc_columns(feature_matrix: pd.DataFrame) -> list[str]:
    """``pc_*`` column names sorted by their integer suffix (item-matrix order)."""
    cols = [c for c in feature_matrix.columns if c.startswith("pc_")]
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))


class ItemSpace:
    """In-memory item PCA space: ``book_id -> row`` plus aligned pc / genre arrays."""

    def __init__(self, feature_matrix: pd.DataFrame, books_master: pd.DataFrame | None = None) -> None:
        self.pc_cols = pc_columns(feature_matrix)
        book_ids = feature_matrix["book_id"].astype(str).to_numpy()
        self.book_to_row: dict[str, int] = {bid: i for i, bid in enumerate(book_ids)}
        self.pc = feature_matrix[self.pc_cols].to_numpy(dtype=np.float32)

        # Genre flags are only needed for the baseline's category_count; the
        # centroid build skips them by passing books_master=None.
        if books_master is not None and set(GENRE_COLUMNS).issubset(books_master.columns):
            genres = books_master.set_index(books_master["book_id"].astype(str))
            genres = genres.reindex(book_ids)[GENRE_COLUMNS].fillna(0)
            self.genre_flags = genres.to_numpy(dtype=np.uint8)
        else:
            self.genre_flags = np.zeros((len(book_ids), len(GENRE_COLUMNS)), dtype=np.uint8)

    @property
    def n_books(self) -> int:
        return self.pc.shape[0]

    @property
    def n_pc(self) -> int:
        return self.pc.shape[1]


class UserAccumulator:
    """Dense per-user accumulators keyed by an integer ``user_id`` code.

    The census of codes comes from ``user_features_global`` (small, materialized),
    which removes the need for a first pass over the ~110M-row canonical to discover
    users. Only users with positives populate the taste vector; everyone else stays
    at ``positive_count == 0``.
    """

    def __init__(self, user_ids: np.ndarray, n_pc: int, n_genres: int) -> None:
        self.user_ids = user_ids.astype(str)
        self.code_of: dict[str, int] = {uid: i for i, uid in enumerate(self.user_ids)}
        n = len(self.user_ids)
        self.sum_pc = np.zeros((n, n_pc), dtype=np.float64)  # float64 for stable summation
        self.pos_count = np.zeros(n, dtype=np.int64)
        self.interaction_count = np.zeros(n, dtype=np.int64)
        self.review_count = np.zeros(n, dtype=np.int64)
        self.want_to_read_count = np.zeros(n, dtype=np.int64)
        self.cat_or = np.zeros((n, n_genres), dtype=np.uint8)
        self.last_date = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)  # epoch ns; min == NaT
        self.dropped_positive_rows = 0

    def update(self, chunk: pd.DataFrame, space: ItemSpace) -> None:
        codes = chunk["user_id"].astype(str).map(self.code_of)
        in_census = codes.notna().to_numpy()
        if not in_census.any():
            return
        chunk = chunk.loc[in_census]
        codes = codes[in_census].to_numpy(dtype=np.int64)
        n = len(self.pos_count)

        # --- interaction-level aggregates over ALL of the user's rows ---
        self.interaction_count += np.bincount(codes, minlength=n)
        review = chunk["has_review_text"].fillna(False).to_numpy(dtype=bool)
        self.review_count += np.bincount(codes, weights=review, minlength=n).astype(np.int64)
        wtr = chunk["is_want_to_read"].fillna(False).to_numpy(dtype=bool)
        self.want_to_read_count += np.bincount(codes, weights=wtr, minlength=n).astype(np.int64)

        date_ns = chunk["date_added"].to_numpy(dtype="datetime64[ns]").view(np.int64)
        np.maximum.at(self.last_date, codes, date_ns)

        # --- positive taste vector ---
        is_read = chunk["is_read"].fillna(False).to_numpy(dtype=bool)
        rating = pd.to_numeric(chunk["rating_clean"], errors="coerce").to_numpy(dtype=np.float64)
        positive = is_read & (rating >= POSITIVE_RATING_THRESHOLD)
        if not positive.any():
            return

        pos_codes = codes[positive]
        pos_books = chunk["book_id"].astype(str).to_numpy()[positive]
        rows = np.fromiter((space.book_to_row.get(b, -1) for b in pos_books), dtype=np.int64, count=len(pos_books))
        present = rows >= 0
        self.dropped_positive_rows += int((~present).sum())
        pos_codes = pos_codes[present]
        rows = rows[present]
        if len(pos_codes) == 0:
            return

        self.pos_count += np.bincount(pos_codes, minlength=n)

        # Compress to one entry per (code) within the chunk via a stable sort, then
        # scatter the group sums into the global accumulators (never a flat join).
        order = np.argsort(pos_codes, kind="stable")
        sorted_codes = pos_codes[order]
        uniq, starts = np.unique(sorted_codes, return_index=True)
        sum_pc = np.add.reduceat(space.pc[rows][order], starts, axis=0)
        self.sum_pc[uniq] += sum_pc.astype(np.float64)
        genre_max = np.maximum.reduceat(space.genre_flags[rows][order], starts, axis=0)
        self.cat_or[uniq] = np.maximum(self.cat_or[uniq], genre_max)


def _last_date_series(last_date: np.ndarray) -> pd.Series:
    out = pd.Series(pd.to_datetime(last_date, unit="ns"))
    out[last_date == np.iinfo(np.int64).min] = pd.NaT
    return out


def build_user_artifacts(
    feature_matrix: pd.DataFrame,
    books_master: pd.DataFrame,
    user_features: pd.DataFrame,
    interaction_chunks: Iterable[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Accumulate positives into per-user centroids; return ``(matrix, meta, diag)``.

    ``user_matrix`` holds one row per user with ``positive_count > 0`` (``user_id`` +
    ``pc_*`` aligned 1:1 to the item matrix). ``user_meta`` holds one row per user that
    actually appears in the canonical (``interaction_count > 0``).
    """
    space = ItemSpace(feature_matrix, books_master)
    acc = UserAccumulator(
        user_features["user_id"].astype(str).to_numpy(), space.n_pc, len(GENRE_COLUMNS)
    )

    for chunk in interaction_chunks:
        acc.update(chunk, space)

    bias = user_features.set_index(user_features["user_id"].astype(str))["user_rating_bias"]
    bias = bias.reindex(acc.user_ids).fillna(0.0).to_numpy(dtype=np.float32)

    has_positive = acc.pos_count > 0
    user_vec = acc.sum_pc[has_positive] / acc.pos_count[has_positive][:, None]
    user_vec = user_vec.astype(np.float32)
    matrix = pd.DataFrame(user_vec, columns=space.pc_cols)
    matrix.insert(0, "user_id", acc.user_ids[has_positive])

    present = acc.interaction_count > 0
    meta = pd.DataFrame(
        {
            "user_id": acc.user_ids[present],
            "positive_count": acc.pos_count[present].astype(np.int64),
            "interaction_count": acc.interaction_count[present].astype(np.int64),
            "review_count": acc.review_count[present].astype(np.int64),
            "want_to_read_count": acc.want_to_read_count[present].astype(np.int64),
            "user_rating_bias": bias[present],
            "category_count": acc.cat_or[present].sum(axis=1).astype(np.int64),
            "last_date_added": _last_date_series(acc.last_date[present]).to_numpy(),
            "is_cold_start": acc.pos_count[present] < COLD_START_MIN,
        }
    )

    diagnostics = {
        "n_users_matrix": int(has_positive.sum()),
        "n_users_meta": int(present.sum()),
        "n_users_census": int(len(acc.user_ids)),
        "cold_start_pct": (
            100.0 * float((meta["is_cold_start"]).sum()) / len(meta) if len(meta) else 0.0
        ),
        "dropped_positive_rows": acc.dropped_positive_rows,
        "pc_columns": space.pc_cols,
    }
    return matrix, meta, diagnostics


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _interaction_chunks(path: Path, chunksize: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    yield from read_parquet_chunks(path, chunksize, columns=INTERACTION_COLUMNS)


def print_validations(
    matrix: pd.DataFrame, meta: pd.DataFrame, diagnostics: dict[str, Any], space_pc_cols: list[str]
) -> None:
    print("\nUser matrix build complete")
    print(f"Users in matrix (positive_count>0): {diagnostics['n_users_matrix']:,}")
    print(f"Users in meta (present in canonical): {diagnostics['n_users_meta']:,}")
    print(f"User census (user_features_global): {diagnostics['n_users_census']:,}")
    print(f"Cold-start share: {diagnostics['cold_start_pct']:.2f}%")
    print(f"Positive rows dropped (book_id absent): {diagnostics['dropped_positive_rows']:,}")

    matrix_pc = [c for c in matrix.columns if c.startswith("pc_")]
    if matrix_pc != space_pc_cols:
        raise ValueError("user_matrix pc_* columns are not aligned 1:1 with the item matrix.")
    values = matrix[matrix_pc].to_numpy()
    if values.size and not np.isfinite(values).all():
        raise ValueError("user_matrix contains non-finite values.")
    if len(matrix) > diagnostics["n_users_meta"]:
        raise ValueError("user_matrix cannot have more users than user_meta.")


def main() -> None:
    if not MASTER_FEATURE_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"{MASTER_FEATURE_MATRIX_PATH} does not exist. "
            "Run `python -m src.reduction.build_master_feature_matrix` first."
        )
    for path in (INTERACTIONS_CURATED_PATH, USER_FEATURES_GLOBAL_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Run `python -m src.curation.interactions` first."
            )

    feature_matrix = pd.read_parquet(MASTER_FEATURE_MATRIX_PATH)
    books_master = pd.read_parquet(BOOKS_MASTER_PATH, columns=["book_id", *GENRE_COLUMNS])
    user_features = pd.read_parquet(
        USER_FEATURES_GLOBAL_PATH, columns=["user_id", "user_rating_bias"]
    )

    matrix, meta, diagnostics = build_user_artifacts(
        feature_matrix,
        books_master,
        user_features,
        _interaction_chunks(INTERACTIONS_CURATED_PATH),
    )

    safe_write_parquet(matrix, USER_MATRIX_PATH)
    safe_write_parquet(meta, USER_META_PATH)
    print_validations(matrix, meta, diagnostics, diagnostics["pc_columns"])
    print(f"User matrix written to: {USER_MATRIX_PATH}")
    print(f"User meta written to:   {USER_META_PATH}")


if __name__ == "__main__":
    main()
