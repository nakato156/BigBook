"""Build the canonical, deduplicated, **global** interactions artifact.

Why global and deduplicated (see plan/CLAUDE.md):

* The legacy per-genre ``interactions_curated.parquet`` files dropped the whole
  implicit layer (``want-to-read`` / read-without-rating) because an upstream
  notebook filter kept only ``rating.notna()``. They are also wrongly
  partitioned: ~71% of users appear in >=2 genres, so K-core and rating bias
  computed per category are wrong; and ~19% of ``(user, book)`` pairs are
  duplicated across dumps (the same interaction, because the book belongs to
  several genres). The multi-genre nature already lives on the **book**
  (``genre_*`` in ``books_master``/PCA), so each interaction must count once.

This module streams the five raw interaction dumps and produces a single
canonical artifact (plus a separate review-text artifact and a global
user-features table). K-core and ``user_rating_bias`` are **global**.

Item-side artifacts (``books_master`` / PCA / clusters) are treated as stable:
the valid book universe is read from ``books_master.parquet`` and not recomputed.

Invoke as a module::

    env/bin/python -m src.curation.interactions --max-rows-per-file 200000 --skip-views  # dry-run
    env/bin/python -m src.curation.interactions                                          # full build + views
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.util import hash_pandas_object

from src.config import (
    CATEGORIES,
    INTERACTIONS_CURATED_GLOBAL_PATH,
    PROCESSED_DIR,
    REVIEW_TEXTS_PATH,
    USER_FEATURES_GLOBAL_PATH,
)
from src.utils.cleaning import (
    empty_strings_to_na,
    normalize_review_text,
    parse_bool_series,
    parse_goodreads_dates,
)
from src.utils.io import read_jsonl_chunks, read_parquet_chunks, remove_path, safe_write_parquet

try:  # optional progress bars; the build runs fine without tqdm installed
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


MASTER_PATH = PROCESSED_DIR / "books_master.parquet"

RATING_MIN = 1
RATING_MAX = 5
K_USER_MIN = 3
INTERACTION_CHUNKSIZE = 500_000

# Engagement strength (also the dedup priority order). Higher = stronger signal.
ENGAGEMENT_RANK: dict[str, int] = {
    "review": 3,
    "rating_only": 2,
    "read_no_rating": 1,
    "want_to_read": 0,
}
INTERACTION_WEIGHTS: dict[str, float] = {
    "review": 1.0,
    "rating_only": 0.9,
    "read_no_rating": 0.5,
    "want_to_read": 0.3,
}

# category key -> books_master genre flag column (for the EDA-only category views)
CATEGORY_TO_GENRE_COLUMN: dict[str, str] = {
    "fantasy_paranormal": "genre_fantasy",
    "history_biography": "genre_history",
    "mystery_thriller_crime": "genre_mystery",
    "romance": "genre_romance",
    "young_adult": "genre_ya",
}

# Canonical artifact: one row per deduplicated interaction, **no review text**.
CANONICAL_COLUMNS: list[str] = [
    "interaction_key",
    "user_id",
    "book_id",
    "review_id",
    "is_read",
    "rating_clean",
    "rating_missing",
    "has_review_text",
    "review_text_length",
    "reading_duration_days",
    "engagement_mode",
    "is_want_to_read",
    "interaction_weight",
    "user_rating_bias",
    "date_added",
    "date_updated",
    "read_at",
    "started_at",
]
REVIEW_TEXT_COLUMNS: list[str] = [
    "interaction_key",
    "review_id",
    "review_text_clean",
    "review_text_length",
]
DATE_COLUMNS = ["date_added", "date_updated", "read_at", "started_at"]

# Temp (deduped, pre user-filter) carries everything the post-pass needs.
_TEMP_DTYPES: dict[str, str] = {
    "interaction_key": "uint64",
    "user_id": "string",
    "book_id": "string",
    "review_id": "string",
    "is_read": "boolean",
    "rating_clean": "float32",
    "rating_missing": "boolean",
    "has_review_text": "boolean",
    "review_text_length": "int32",
    "reading_duration_days": "float32",
    "engagement_mode": "string",
    "is_want_to_read": "boolean",
    "interaction_weight": "float32",
    "review_text_clean": "string",
}


# --------------------------------------------------------------------------- #
# Pure, testable helpers
# --------------------------------------------------------------------------- #
def valid_book_ids() -> set[str]:
    """Canonical book universe, aligned to PCA: ``book_id`` of ``books_master``.

    Falls back to the union of per-category ``books_curated.parquet`` (incl. the
    legacy ``fantasy`` / ``history`` dirs) if the master table is missing.
    """
    if MASTER_PATH.exists():
        ids = pd.read_parquet(MASTER_PATH, columns=["book_id"])["book_id"].astype(str)
        return set(ids.tolist())

    from src.merge_master import resolve_curated_books_path

    ids: set[str] = set()
    for key in CATEGORIES:
        path = resolve_curated_books_path(key)
        if path is not None:
            ids |= set(pd.read_parquet(path, columns=["book_id"])["book_id"].astype(str).tolist())
    if not ids:
        raise FileNotFoundError(
            f"{MASTER_PATH} not found and no per-category books_curated.parquet available. "
            "Run `python -m src.merge_master` first."
        )
    return ids


def clean_interaction_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Type/parse a raw interaction chunk and derive engagement features.

    Recovers the implicit layer: ``rating == 0`` becomes ``rating_clean`` NA with
    ``rating_missing`` True, and rows with no read/rating/review are tagged
    ``want_to_read`` instead of being dropped.
    """
    out = empty_strings_to_na(df.copy())
    out = parse_goodreads_dates(out)

    for column in ("user_id", "book_id", "review_id"):
        if column in out.columns:
            out[column] = out[column].astype("string")

    rating = pd.to_numeric(out.get("rating"), errors="coerce")
    out["rating_missing"] = rating.isna() | rating.eq(0)
    out["rating_clean"] = rating.where(rating.between(RATING_MIN, RATING_MAX)).astype("float32")

    is_read = out["is_read"]
    if is_read.dtype == bool or is_read.dtype.name == "boolean":
        out["is_read"] = is_read.astype("boolean")
    else:
        out["is_read"] = parse_bool_series(is_read)

    # Normalize only the rows that actually carry text. ``empty_strings_to_na``
    # already nulled blank/whitespace reviews, so the expensive per-row Python
    # normalization runs on a tiny subset instead of all ~62M rows (load-bearing
    # for the full-catalog streaming build).
    review_clean = pd.Series(pd.NA, index=out.index, dtype="string")
    if "review_text_incomplete" in out.columns:
        raw_text = out["review_text_incomplete"]
        has_raw = raw_text.notna()
        if has_raw.any():
            review_clean.loc[has_raw] = raw_text[has_raw].map(normalize_review_text).astype("string")
    out["review_text_clean"] = review_clean
    out["has_review_text"] = out["review_text_clean"].notna()
    out["review_text_length"] = out["review_text_clean"].fillna("").str.len().astype("int32")

    if {"read_at", "started_at"}.issubset(out.columns):
        duration = (out["read_at"] - out["started_at"]).dt.total_seconds() / 86400.0
        out["reading_duration_days"] = duration.astype("float32")
    else:
        out["reading_duration_days"] = pd.Series(np.float32(np.nan), index=out.index, dtype="float32")

    has_review = out["has_review_text"].fillna(False).to_numpy(dtype=bool)
    has_rating = out["rating_clean"].notna().to_numpy(dtype=bool)
    read_flag = out["is_read"].fillna(False).to_numpy(dtype=bool)
    out["engagement_mode"] = pd.Series(
        np.select(
            [has_review, has_rating, read_flag],
            ["review", "rating_only", "read_no_rating"],
            default="want_to_read",
        ),
        index=out.index,
        dtype="string",
    )
    out["is_want_to_read"] = out["engagement_mode"].eq("want_to_read")
    out["interaction_weight"] = out["engagement_mode"].map(INTERACTION_WEIGHTS).astype("float32")
    return out


def add_interaction_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add a deterministic, portable ``uint64`` ``interaction_key``.

    Uses ``review_id`` when present, else ``user_id|book_id``. Hashed with
    ``pandas.util.hash_pandas_object`` (stable across processes), so it is a safe
    join key with ``review_texts.parquet``. ``review_id`` is kept as its own
    column.
    """
    out = df.copy()
    review_id = out["review_id"].astype("string")
    fallback = (
        out["user_id"].astype("string").fillna("")
        + "|"
        + out["book_id"].astype("string").fillna("")
    )
    key_source = review_id.where(review_id.notna(), fallback).astype("string")
    out["interaction_key"] = hash_pandas_object(key_source, index=False).to_numpy().astype("uint64")
    return out


def interaction_priority(df: pd.DataFrame) -> pd.Series:
    """Pack the dedup priority into an orderable ``uint64`` (keep-best basis).

    Bit layout (MSB->LSB): engagement rank (2 bits) | rating present (1) |
    is_read (1) | recency in epoch seconds (low 60 bits). Higher wins, so a later
    ``rating_only`` beats an earlier ``want_to_read`` for the same key, and exact
    ties fall back to the more recent row.
    """
    engagement = df["engagement_mode"].map(ENGAGEMENT_RANK).fillna(0).to_numpy(dtype="uint64")
    rating_present = df["rating_clean"].notna().to_numpy(dtype="uint64")
    read_flag = df["is_read"].fillna(False).to_numpy(dtype=bool).astype("uint64")

    recency = df["date_updated"]
    if "date_added" in df.columns:
        recency = recency.fillna(df["date_added"])
    seconds = (recency - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    seconds = pd.to_numeric(seconds, errors="coerce").fillna(0)
    seconds = np.clip(seconds.to_numpy(), 0, (1 << 60) - 1).astype("uint64")

    s62, s61, s60 = np.uint64(62), np.uint64(61), np.uint64(60)
    packed = (engagement << s62) | (rating_present << s61) | (read_flag << s60) | seconds
    return pd.Series(packed, index=df.index, dtype="uint64")


def _max_priority_index(keys: np.ndarray, priorities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``key -> max(priority)`` as sorted ``(unique_keys, max_priority)`` arrays.

    Memory-bounded: only one row per unique key survives (never a flat 62M-row
    join). Lookups are ``np.searchsorted`` against ``unique_keys``.
    """
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_priorities = priorities[order]
    unique_keys, starts = np.unique(sorted_keys, return_index=True)
    max_priority = np.maximum.reduceat(sorted_priorities, starts)
    return unique_keys, max_priority


def dedup_keep_best(
    df: pd.DataFrame,
    best_priority: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Dedup by ``interaction_key`` keeping the **highest-priority** row per key.

    Not keep-first: when duplicates differ (want_to_read -> later rating_only)
    the strong signal survives. With ``best_priority`` provided (the global
    index), it deduplicates a streamed chunk against the cross-category maximum;
    without it, it deduplicates ``df`` against its own per-key maximum.
    """
    if df.empty:
        return df.copy()
    keys = df["interaction_key"].to_numpy(dtype="uint64")
    priorities = interaction_priority(df).to_numpy(dtype="uint64")
    if best_priority is None:
        unique_keys, max_priority = _max_priority_index(keys, priorities)
    else:
        unique_keys, max_priority = best_priority
    positions = np.searchsorted(unique_keys, keys)
    is_max = priorities == max_priority[positions]
    winners = df[is_max]
    winners = winners[~winners["interaction_key"].duplicated(keep="first")]
    return winners.reset_index(drop=True)


def accumulate_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user partial sums over **deduplicated** rows (summable across chunks)."""
    rating = df["rating_clean"].astype("float64")
    frame = pd.DataFrame(
        {
            "user_id": df["user_id"].astype(str).to_numpy(),
            "read_or_rated_count": (~df["is_want_to_read"].fillna(False)).astype("int64").to_numpy(),
            "rating_sum": rating.to_numpy(),
            "rating_sq_sum": (rating**2).to_numpy(),
            "rating_count": rating.notna().astype("int64").to_numpy(),
        }
    )
    grouped = frame.groupby("user_id", sort=False).sum()
    return grouped


def finalize_user_features(partials: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    """Finalize global per-user features from accumulated partial sums.

    ``user_rating_bias`` is ``mean - global_mean`` (**global**), neutral (0.0)
    for users with no explicit rating. ``valid`` is the global K-core flag.
    """
    stats = partials.reset_index()
    count = stats["rating_count"].astype("float64")
    safe_count = count.where(count > 0)
    mean = stats["rating_sum"] / safe_count
    variance = (stats["rating_sq_sum"] - stats["rating_sum"] ** 2 / safe_count) / (count - 1).where(count > 1)
    std = np.sqrt(variance).where(count > 1, 0.0)
    bias = (mean - global_mean).fillna(0.0)
    return pd.DataFrame(
        {
            "user_id": stats["user_id"].astype("string"),
            "user_mean_rating": mean.astype("float32"),
            "user_rating_std": std.astype("float32"),
            "user_rating_count": stats["rating_count"].astype("int64"),
            "user_rating_bias": bias.astype("float32"),
            "read_or_rated_count": stats["read_or_rated_count"].astype("int64"),
            "valid": (stats["read_or_rated_count"] >= K_USER_MIN),
        }
    )


# --------------------------------------------------------------------------- #
# Streaming orchestration
# --------------------------------------------------------------------------- #
# Staging carries everything the temp does, plus the precomputed priority, so the
# expensive JSON parse + cleaning happens exactly once (scan 1). Scan 2 reads the
# staging parquet, which is roughly an order of magnitude cheaper than re-parsing
# the gzip JSON dumps.
_STAGING_DTYPES: dict[str, str] = {**_TEMP_DTYPES, "priority": "uint64"}


def _arrow_table(df: pd.DataFrame, dtypes: dict[str, str]) -> pa.Table:
    out = df.copy()
    for column, dtype in dtypes.items():
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = out[column].astype(dtype)
    for column in DATE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NaT
    return pa.Table.from_pandas(out[list(dtypes) + DATE_COLUMNS], preserve_index=False)


def _read_raw_chunks(path: Path, chunksize: int, progress: bool, desc: str) -> Iterator[pd.DataFrame]:
    """Yield raw JSONL chunks, optionally driving a per-file byte-progress bar.

    The bar is measured in **compressed bytes consumed** (``tqdm.wrapattr`` over the
    gzip file), which gives a real ETA for the dominant cost of the build — the
    one-time JSON parse — without knowing the row count up front.
    """
    if not (progress and tqdm is not None):
        yield from read_jsonl_chunks(path, chunksize)
        return
    with open(path, "rb") as raw:
        wrapped = tqdm.wrapattr(
            raw, "read", total=path.stat().st_size, desc=desc,
            unit="B", unit_scale=True, unit_divisor=1024, leave=False,
        )
        with wrapped as fobj:
            yield from pd.read_json(fobj, lines=True, compression="gzip", chunksize=chunksize)


def _read_parquet_chunks_progress(
    path: Path, chunksize: int, progress: bool, desc: str
) -> Iterator[pd.DataFrame]:
    """Yield parquet chunks with a row-based progress bar (total = ``num_rows``)."""
    bar = None
    if progress and tqdm is not None:
        total = pq.ParquetFile(path).metadata.num_rows
        bar = tqdm(total=total, unit="row", unit_scale=True, desc=desc, leave=False)
    for chunk in read_parquet_chunks(path, chunksize):
        yield chunk
        if bar is not None:
            bar.update(len(chunk))
    if bar is not None:
        bar.close()


def _iter_clean_chunks(
    category_files: dict[str, Path],
    valid_books: set[str],
    max_rows_per_file: int | None,
    progress: bool = False,
) -> Iterator[tuple[str, pd.DataFrame, int]]:
    """Stream raw dumps -> clean -> key -> filter to the valid book universe.

    Yields ``(category_key, filtered_chunk, raw_chunk_rows)``. Re-iterating runs
    the same deterministic transform, so the two scans see identical data.
    """
    n_files = len(category_files)
    for file_idx, (category, path) in enumerate(category_files.items(), start=1):
        seen = 0
        desc = f"scan1 parse [{file_idx}/{n_files}] {path.name}"
        for chunk in _read_raw_chunks(path, INTERACTION_CHUNKSIZE, progress, desc):
            if max_rows_per_file is not None:
                if seen >= max_rows_per_file:
                    break
                if seen + len(chunk) > max_rows_per_file:
                    chunk = chunk.iloc[: max_rows_per_file - seen]
            raw_rows = len(chunk)
            seen += raw_rows
            cleaned = add_interaction_key(clean_interaction_chunk(chunk))
            filtered = cleaned[cleaned["book_id"].astype(str).isin(valid_books)]
            yield category, filtered, raw_rows


def build_global_interactions(
    *,
    category_files: dict[str, Path] | None = None,
    valid_books: set[str] | None = None,
    out_interactions: Path = INTERACTIONS_CURATED_GLOBAL_PATH,
    out_review_texts: Path = REVIEW_TEXTS_PATH,
    out_user_features: Path = USER_FEATURES_GLOBAL_PATH,
    max_rows_per_file: int | None = None,
    with_source_category_count: bool = False,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Build the canonical global interactions artifact (streaming, two raw scans).

    Scan 1 builds the ``key -> max_priority`` index. Scan 2 emits keep-best
    winners to a temp parquet and accumulates global user stats. A cheap
    parquet post-pass applies the global K-core, merges global ``user_rating_bias``,
    splits review text out, and writes the three artifacts.
    """
    if category_files is None:
        if not MASTER_PATH.exists() and valid_books is None:
            raise FileNotFoundError(
                f"{MASTER_PATH} does not exist. Run `python -m src.merge_master` and "
                "`python -m src.reduction.build_master_feature_matrix` first."
            )
        category_files = {key: cfg.interactions_file for key, cfg in CATEGORIES.items()}
    if valid_books is None:
        valid_books = valid_book_ids()

    if not force and out_interactions.exists() and out_user_features.exists():
        print(f"Cached: {out_interactions} already exists (use force=True to rebuild).")
        return {"status": "cached", "interactions_path": str(out_interactions)}

    temp_path = out_interactions.parent / f"{out_interactions.stem}_tmp.parquet"
    staging_path = out_interactions.parent / f"{out_interactions.stem}_staging.parquet"
    remove_path(temp_path)
    remove_path(staging_path)

    category_index = {category: idx for idx, category in enumerate(category_files)}

    # ---- Scan 1 (only JSON parse): clean -> stage parquet + priority index ----
    key_parts: list[np.ndarray] = []
    priority_parts: list[np.ndarray] = []
    scc_key_parts: list[np.ndarray] = []
    scc_cat_parts: list[np.ndarray] = []
    raw_rows = 0
    rows_after_book = 0
    staging_writer: pq.ParquetWriter | None = None
    for category, df, raw_chunk in _iter_clean_chunks(
        category_files, valid_books, max_rows_per_file, progress=progress
    ):
        raw_rows += raw_chunk
        if df.empty:
            continue
        rows_after_book += len(df)
        keys = df["interaction_key"].to_numpy(dtype="uint64")
        priorities = interaction_priority(df).to_numpy(dtype="uint64")
        key_parts.append(keys)
        priority_parts.append(priorities)
        if with_source_category_count:
            unique_chunk_keys = np.unique(keys)
            scc_key_parts.append(unique_chunk_keys)
            scc_cat_parts.append(np.full(unique_chunk_keys.shape, category_index[category], dtype="uint16"))

        df = df.copy()
        df["priority"] = priorities
        table = _arrow_table(df, _STAGING_DTYPES)
        if staging_writer is None:
            staging_writer = pq.ParquetWriter(staging_path, table.schema)
        staging_writer.write_table(table)
    if staging_writer is not None:
        staging_writer.close()

    if not key_parts:
        raise ValueError("No interactions survived the book-universe filter; nothing to build.")

    unique_keys, max_priority = _max_priority_index(
        np.concatenate(key_parts), np.concatenate(priority_parts)
    )
    del key_parts, priority_parts
    n_unique = int(len(unique_keys))

    scc_index: tuple[np.ndarray, np.ndarray] | None = None
    if with_source_category_count and scc_key_parts:
        pairs = np.unique(
            np.stack([np.concatenate(scc_key_parts), np.concatenate(scc_cat_parts)]), axis=1
        )
        scc_keys, scc_counts = np.unique(pairs[0], return_counts=True)
        scc_index = (scc_keys, scc_counts.astype("uint16"))
    del scc_key_parts, scc_cat_parts

    # ---- Scan 2 (cheap parquet read): keep-best winners + global user stats ----
    emitted = np.zeros(n_unique, dtype=bool)
    user_stats: pd.DataFrame | None = None
    engagement_counts: Counter[str] = Counter()
    writer: pq.ParquetWriter | None = None
    winners_total = 0
    for df in _read_parquet_chunks_progress(
        staging_path, INTERACTION_CHUNKSIZE, progress, "scan2 keep-best winners"
    ):
        keys = df["interaction_key"].to_numpy(dtype="uint64")
        priorities = df["priority"].to_numpy(dtype="uint64")
        positions = np.searchsorted(unique_keys, keys)
        mask = (priorities == max_priority[positions]) & (~emitted[positions])
        candidates = df[mask]
        if candidates.empty:
            continue
        candidates = candidates[~candidates["interaction_key"].duplicated(keep="first")]
        emitted[np.searchsorted(unique_keys, candidates["interaction_key"].to_numpy(dtype="uint64"))] = True
        winners_total += len(candidates)

        chunk_stats = accumulate_user_stats(candidates)
        user_stats = chunk_stats if user_stats is None else user_stats.add(chunk_stats, fill_value=0.0)
        engagement_counts.update(candidates["engagement_mode"].astype(str).value_counts().to_dict())

        table = _arrow_table(candidates, _TEMP_DTYPES)
        if writer is None:
            writer = pq.ParquetWriter(temp_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()
    remove_path(staging_path)

    assert user_stats is not None  # guaranteed: key_parts non-empty -> at least one winner
    global_rating_sum = float(user_stats["rating_sum"].sum())
    global_rating_count = float(user_stats["rating_count"].sum())
    global_mean = global_rating_sum / global_rating_count if global_rating_count > 0 else 0.0
    user_features = finalize_user_features(user_stats, global_mean)
    safe_write_parquet(user_features, out_user_features)

    valid_users = set(user_features.loc[user_features["valid"], "user_id"].astype(str).tolist())
    bias_map = dict(zip(user_features["user_id"].astype(str), user_features["user_rating_bias"].astype("float32")))

    # ---- Post-pass: K-core filter + bias merge + text split (parquet only) ----
    canonical_columns = list(CANONICAL_COLUMNS)
    if scc_index is not None:
        canonical_columns.insert(canonical_columns.index("user_rating_bias") + 1, "source_category_count")

    canonical_writer: pq.ParquetWriter | None = None
    text_writer: pq.ParquetWriter | None = None
    canonical_rows = 0
    review_text_rows = 0
    if temp_path.exists():
        for chunk in _read_parquet_chunks_progress(
            temp_path, INTERACTION_CHUNKSIZE, progress, "post-pass K-core + text split"
        ):
            chunk = chunk[chunk["user_id"].astype(str).isin(valid_users)]
            if chunk.empty:
                continue
            chunk["user_rating_bias"] = chunk["user_id"].astype(str).map(bias_map).astype("float32")
            if scc_index is not None:
                positions = np.searchsorted(scc_index[0], chunk["interaction_key"].to_numpy(dtype="uint64"))
                chunk["source_category_count"] = scc_index[1][positions].astype("int32")

            text_rows = chunk[chunk["has_review_text"].fillna(False)][REVIEW_TEXT_COLUMNS]
            canonical_chunk = chunk[canonical_columns]

            canonical_table = pa.Table.from_pandas(canonical_chunk, preserve_index=False)
            if canonical_writer is None:
                canonical_writer = pq.ParquetWriter(out_interactions, canonical_table.schema)
            canonical_writer.write_table(canonical_table)
            canonical_rows += len(canonical_chunk)

            if not text_rows.empty:
                text_table = pa.Table.from_pandas(text_rows, preserve_index=False)
                if text_writer is None:
                    text_writer = pq.ParquetWriter(out_review_texts, text_table.schema)
                text_writer.write_table(text_table)
                review_text_rows += len(text_rows)
    if canonical_writer is not None:
        canonical_writer.close()
    else:
        safe_write_parquet(pd.DataFrame(columns=canonical_columns), out_interactions)
    if text_writer is not None:
        text_writer.close()
    else:
        safe_write_parquet(pd.DataFrame(columns=REVIEW_TEXT_COLUMNS), out_review_texts)
    remove_path(temp_path)

    summary = {
        "status": "built",
        "raw_rows": raw_rows,
        "rows_after_book_filter": rows_after_book,
        "dropped_by_book_universe": raw_rows - rows_after_book,
        "deduplicated_rows": n_unique,
        "duplicate_pairs_removed": rows_after_book - n_unique,
        "winners_emitted": winners_total,
        "canonical_rows": canonical_rows,
        "review_text_rows": review_text_rows,
        "users_total": int(len(user_features)),
        "users_valid": int(user_features["valid"].sum()),
        "dropped_by_user_kcore": n_unique - canonical_rows,
        "global_mean_rating": global_mean,
        "engagement_mode_counts": dict(engagement_counts),
        "want_to_read_pct": (
            100.0 * engagement_counts.get("want_to_read", 0) / winners_total if winners_total else 0.0
        ),
        "interactions_path": str(out_interactions),
        "review_texts_path": str(out_review_texts),
        "user_features_path": str(out_user_features),
    }
    return summary


def build_category_views(
    *,
    out_interactions: Path = INTERACTIONS_CURATED_GLOBAL_PATH,
    master_path: Path = MASTER_PATH,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Derive per-category ``interactions_view.parquet`` (EDA/debug only).

    Filters the canonical artifact by ``book_id`` carrying ``genre_<cat> == 1``.
    A multi-genre interaction appears in several views (by design); this is not a
    destructive re-partition and never touches the legacy curated files.
    """
    if not out_interactions.exists():
        raise FileNotFoundError(f"{out_interactions} not found. Run build_global_interactions first.")
    if not master_path.exists():
        raise FileNotFoundError(f"{master_path} not found; cannot derive genre-filtered views.")

    master = pd.read_parquet(master_path, columns=["book_id", *CATEGORY_TO_GENRE_COLUMN.values()])
    master["book_id"] = master["book_id"].astype(str)
    category_books = {
        category: set(master.loc[master[genre_col] == 1, "book_id"].tolist())
        for category, genre_col in CATEGORY_TO_GENRE_COLUMN.items()
    }

    writers: dict[str, pq.ParquetWriter] = {}
    paths = {category: CATEGORIES[category].processed_dir / "interactions_view.parquet" for category in category_books}
    counts = {category: 0 for category in category_books}
    for category, path in paths.items():
        if path.exists() and force:
            remove_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for chunk in _read_parquet_chunks_progress(
            out_interactions, INTERACTION_CHUNKSIZE, progress, "derive category views"
        ):
            book_ids = chunk["book_id"].astype(str)
            for category, books in category_books.items():
                view = chunk[book_ids.isin(books)]
                if view.empty:
                    continue
                table = pa.Table.from_pandas(view, preserve_index=False)
                if category not in writers:
                    writers[category] = pq.ParquetWriter(paths[category], table.schema)
                writers[category].write_table(table)
                counts[category] += len(view)
    finally:
        for writer in writers.values():
            writer.close()

    for category, path in paths.items():
        if category not in writers:
            safe_write_parquet(pd.DataFrame(columns=list(CANONICAL_COLUMNS)), path)

    return {"status": "built", "view_rows": counts, "view_paths": {c: str(p) for c, p in paths.items()}}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_summary(summary: dict[str, Any]) -> None:
    if summary.get("status") == "cached":
        return
    print("\nGlobal interactions build complete")
    print(f"Raw rows read:            {summary['raw_rows']:,}")
    print(f"After book-universe:      {summary['rows_after_book_filter']:,} "
          f"(dropped {summary['dropped_by_book_universe']:,})")
    print(f"Deduplicated rows:        {summary['deduplicated_rows']:,} "
          f"(removed {summary['duplicate_pairs_removed']:,} duplicate pairs)")
    print(f"Canonical rows (K-core):  {summary['canonical_rows']:,} "
          f"(dropped {summary['dropped_by_user_kcore']:,} below K={K_USER_MIN})")
    print(f"Review-text rows:         {summary['review_text_rows']:,}")
    print(f"Users total / valid:      {summary['users_total']:,} / {summary['users_valid']:,}")
    print(f"Global mean rating:       {summary['global_mean_rating']:.4f}")
    print(f"want_to_read share:       {summary['want_to_read_pct']:.2f}%")
    print(f"Engagement modes:         {summary['engagement_mode_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-rows-per-file", type=int, default=None, help="cap rows per raw dump (dry-run)")
    parser.add_argument("--skip-views", action="store_true", help="do not derive per-category EDA views")
    parser.add_argument("--views-only", action="store_true", help="only rebuild views from an existing canonical")
    parser.add_argument(
        "--with-source-category-count",
        action="store_true",
        help="compute the optional source_category_count diagnostic column",
    )
    parser.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    args = parser.parse_args()
    progress = not args.no_progress

    if args.views_only:
        view_summary = build_category_views(force=args.force, progress=progress)
        print(f"Category views rebuilt: {view_summary['view_rows']}")
        return

    summary = build_global_interactions(
        max_rows_per_file=args.max_rows_per_file,
        with_source_category_count=args.with_source_category_count,
        force=args.force,
        progress=progress,
    )
    _print_summary(summary)
    print(f"Canonical:      {summary['interactions_path']}")
    print(f"Review texts:   {summary['review_texts_path']}")
    print(f"User features:  {summary['user_features_path']}")

    if not args.skip_views and summary.get("status") != "cached":
        view_summary = build_category_views(force=args.force, progress=progress)
        print(f"Category views: {view_summary['view_rows']}")


if __name__ == "__main__":
    main()
