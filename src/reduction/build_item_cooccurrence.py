"""Build ``book_cooccurrence`` — collaborative item-item signal via PMI.

Content-based similarity (the PCA ranker) says two books are close because their
*descriptions/metadata* look alike. This module builds an independent, **collaborative**
signal: two books are close because many distinct users marked **both** positive, more
often than expected by chance given each book's individual popularity in user-space.

Positive (same closed decision as the rest of the pipeline, see ``build_user_matrix.py`` /
``build_user_centroids.py``): ``(is_read == True) AND (rating_clean >= 4)``.

PMI, not raw co-occurrence count, is the persisted signal:

    PMI(i, j) = log( co_count(i, j) * N / (count(i) * count(j)) )

where ``count(i)`` is the number of distinct users who positived book ``i`` and ``N`` is
the number of distinct users who positived at least one book. Raw co-occurrence alone
would just reward popular books for showing up together (the same popularity bias A2
already avoids on the content side) — PMI normalizes that out, and the positive floor
(``max(0, PMI)``) keeps only pairs that co-occur *more* than chance, not less.

This module only builds and persists the artifact (``data/features/book_cooccurrence.parquet``);
ranker integration is a later phase (Fase 4c/4d) and is explicitly out of scope here.

Invoke as a module::

    env/bin/python -m src.reduction.build_item_cooccurrence
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
from scipy import sparse

from src.config import (
    BOOK_COOCCURRENCE_PATH,
    INTERACTIONS_CURATED_PATH,
    MASTER_FEATURE_MATRIX_PATH,
)
from src.reduction.build_user_matrix import ItemSpace
from src.utils.io import read_parquet_chunks, safe_write_parquet


POSITIVE_RATING_THRESHOLD = 4.0
MAX_POSITIVES_PER_USER = 200  # caps pair generation only, never the marginal count(i)
MIN_CO_COUNT = 3
CHUNK_SIZE = 1_000_000

INTERACTION_COLUMNS = ["user_id", "book_id", "is_read", "rating_clean"]
DATED_INTERACTION_COLUMNS = [*INTERACTION_COLUMNS, "date_added"]


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
class PositiveCollector:
    """Stream the canonical once, collecting positives into flat in-memory arrays.

    Simpler than ``build_user_centroids.PositiveCollector``: no engagement weight, just
    ``(user_code, book_row)`` pairs for every positive whose book is in the item universe.
    """

    def __init__(self) -> None:
        self.code_of: dict[str, int] = {}
        self.user_ids: list[str] = []
        self._codes: list[np.ndarray] = []
        self._rows: list[np.ndarray] = []
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

        self._codes.append(self._codes_for(pos["user_id"].astype(str).to_numpy()[present]).astype(np.int32))
        self._rows.append(rows[present].astype(np.int32))

    def collected(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._codes:
            empty = np.array([], dtype=np.int32)
            return empty, empty
        return np.concatenate(self._codes), np.concatenate(self._rows)


# --------------------------------------------------------------------------- #
# Aggregation + PMI
# --------------------------------------------------------------------------- #
def build_item_cooccurrence(
    feature_matrix: pd.DataFrame,
    interaction_chunks: Iterable[pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collect positives, aggregate per-user, and normalize co-occurrence into PMI."""
    space = ItemSpace(feature_matrix)
    collector = PositiveCollector()
    for chunk in interaction_chunks:
        collector.update(chunk, space)
    codes, rows = collector.collected()

    n_books = space.n_books
    counts = np.zeros(n_books, dtype=np.int64)
    pair_rows: list[np.ndarray] = []
    pair_cols: list[np.ndarray] = []
    n_users_with_positive = 0
    n_users_truncated = 0

    if len(codes):
        order = np.argsort(codes, kind="stable")
        codes_s = codes[order]
        rows_s = rows[order]
        uniq, starts = np.unique(codes_s, return_index=True)
        bounds = np.append(starts, len(codes_s))
        n_users_with_positive = len(uniq)
        for i in range(len(uniq)):
            sl = slice(bounds[i], bounds[i + 1])
            user_rows = np.unique(rows_s[sl])  # de-dup defensively; canonical is already deduped

            # Marginal count(i): the FULL (uncapped) positive set of this user.
            counts[user_rows] += 1

            # Pair generation: capped, deterministic truncation (sorted, no RNG).
            if len(user_rows) > MAX_POSITIVES_PER_USER:
                n_users_truncated += 1
                pair_source = np.sort(user_rows)[:MAX_POSITIVES_PER_USER]
            else:
                pair_source = user_rows

            n = len(pair_source)
            if n >= 2:
                iu, ju = np.triu_indices(n, k=1)
                pair_rows.append(pair_source[iu])
                pair_cols.append(pair_source[ju])

    n_total_users = len(collector.user_ids)

    if pair_rows:
        all_rows = np.concatenate(pair_rows)
        all_cols = np.concatenate(pair_cols)
    else:
        all_rows = np.array([], dtype=np.int64)
        all_cols = np.array([], dtype=np.int64)

    co_matrix = sparse.coo_matrix(
        (np.ones(len(all_rows), dtype=np.int64), (all_rows, all_cols)),
        shape=(n_books, n_books),
    ).tocsr()
    co_matrix.sum_duplicates()

    co_rows, co_cols = co_matrix.nonzero()
    if len(co_rows):
        co_counts = np.asarray(co_matrix[co_rows, co_cols]).ravel()
    else:
        co_counts = np.array([], dtype=np.int64)

    n_pairs_total = len(co_rows)

    min_count_mask = co_counts >= MIN_CO_COUNT
    co_rows = co_rows[min_count_mask]
    co_cols = co_cols[min_count_mask]
    co_counts = co_counts[min_count_mask]
    n_pairs_after_min_count = len(co_rows)

    if len(co_rows) and n_users_with_positive > 0:
        count_i = counts[co_rows].astype(np.float64)
        count_j = counts[co_cols].astype(np.float64)
        N = float(n_users_with_positive)
        raw_pmi = np.log((co_counts.astype(np.float64) * N) / (count_i * count_j))
        pmi = np.maximum(0.0, raw_pmi)
    else:
        pmi = np.array([], dtype=np.float64)

    book_ids = feature_matrix["book_id"].astype(str).to_numpy()
    id_a = book_ids[co_rows]
    id_b = book_ids[co_cols]

    # Row-index canonicalization (row_i < row_j) does not imply book_id_a < book_id_b as
    # strings — re-canonicalize on the final string ids before writing.
    swap = id_a > id_b
    final_a = np.where(swap, id_b, id_a)
    final_b = np.where(swap, id_a, id_b)

    result = pd.DataFrame(
        {
            "book_id_a": final_a,
            "book_id_b": final_b,
            "pmi": pmi.astype(np.float32),
            "co_count": co_counts.astype(np.int64),
        }
    )

    diagnostics = {
        "n_users_with_positive": int(n_users_with_positive),
        "n_total_users": int(n_total_users),
        "n_users_truncated": int(n_users_truncated),
        "n_books_with_positive": int((counts > 0).sum()),
        "dropped_positive_rows": collector.dropped_positive_rows,
        "n_pairs_total": int(n_pairs_total),
        "n_pairs_after_min_count": int(n_pairs_after_min_count),
        "pmi_min": float(result["pmi"].min()) if len(result) else None,
        "pmi_max": float(result["pmi"].max()) if len(result) else None,
        "pmi_mean": float(result["pmi"].mean()) if len(result) else None,
    }
    return result, diagnostics


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _interaction_chunks(path: Path, chunksize: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    yield from read_parquet_chunks(path, chunksize, columns=INTERACTION_COLUMNS)


def interaction_chunks_before(
    path: Path,
    cutoff: pd.Timestamp,
    chunksize: int = CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield canonical interaction chunks observed on or before one UTC cutoff."""
    resolved_cutoff = pd.Timestamp(cutoff)
    resolved_cutoff = (
        resolved_cutoff.tz_localize("UTC")
        if resolved_cutoff.tzinfo is None
        else resolved_cutoff.tz_convert("UTC")
    )
    for chunk in read_parquet_chunks(path, chunksize, columns=DATED_INTERACTION_COLUMNS):
        dates = pd.to_datetime(chunk["date_added"], errors="coerce", utc=True)
        keep = dates.notna() & (dates <= resolved_cutoff)
        if keep.any():
            yield chunk.loc[keep, INTERACTION_COLUMNS].copy()


def print_validations(result: pd.DataFrame, diagnostics: dict[str, Any]) -> None:
    print("\nItem co-occurrence build complete")
    print(f"Users with >=1 positive: {diagnostics['n_users_with_positive']:,} / {diagnostics['n_total_users']:,}")
    print(f"Users truncated by MAX_POSITIVES_PER_USER cap: {diagnostics['n_users_truncated']:,}")
    print(f"Books with >=1 positive: {diagnostics['n_books_with_positive']:,}")
    print(f"Positive rows dropped (book_id absent): {diagnostics['dropped_positive_rows']:,}")
    print(f"Pairs before MIN_CO_COUNT filter: {diagnostics['n_pairs_total']:,}")
    print(f"Pairs after MIN_CO_COUNT filter:  {diagnostics['n_pairs_after_min_count']:,}")
    if diagnostics["pmi_min"] is not None:
        print(
            f"pmi range: [{diagnostics['pmi_min']:.4f}, {diagnostics['pmi_max']:.4f}], "
            f"mean={diagnostics['pmi_mean']:.4f}"
        )

    if len(result):
        if not (result["book_id_a"] < result["book_id_b"]).all():
            raise ValueError("book_cooccurrence rows are not canonicalized (book_id_a < book_id_b).")
        if (result["pmi"] < 0).any():
            raise ValueError("book_cooccurrence contains negative pmi values.")
        if (result["co_count"] < MIN_CO_COUNT).any():
            raise ValueError("book_cooccurrence contains pairs below MIN_CO_COUNT.")
        if not np.isfinite(result["pmi"].to_numpy()).all():
            raise ValueError("book_cooccurrence contains non-finite pmi values.")


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
    result, diagnostics = build_item_cooccurrence(
        feature_matrix, _interaction_chunks(INTERACTIONS_CURATED_PATH)
    )
    safe_write_parquet(result, BOOK_COOCCURRENCE_PATH)
    print_validations(result, diagnostics)
    print(f"Book co-occurrence written to: {BOOK_COOCCURRENCE_PATH}")


if __name__ == "__main__":
    main()
