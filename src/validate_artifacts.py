"""Validate schemas and identifier alignment across the V1 artifact chain."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    BOOKS_MASTER_PATH,
    BOOK_COOCCURRENCE_PATH,
    INTERACTIONS_CURATED_PATH,
    MASTER_FEATURE_MATRIX_PATH,
    PROJECT_ROOT,
    USER_CENTROIDS_PATH,
    USER_FEATURES_GLOBAL_PATH,
    USER_MATRIX_PATH,
    USER_META_PATH,
)

BOOK_CLUSTERS_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "clustering" / "book_clusters_k100.parquet"
)

BOOKS_REQUIRED = {
    "book_id",
    "title",
    "description",
    "series",
    "language_code",
    "average_rating",
    "ratings_count",
    "text_reviews_count",
    "num_pages",
    "publication_year",
    "author_count",
    "genre_fantasy",
    "genre_mystery",
    "genre_history",
    "genre_ya",
    "genre_romance",
    "genre_count",
}
INTERACTIONS_REQUIRED = {
    "interaction_key",
    "user_id",
    "book_id",
    "is_read",
    "rating_clean",
    "rating_missing",
    "has_review_text",
    "engagement_mode",
    "interaction_weight",
    "user_rating_bias",
    "date_added",
}


def _schema_names(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return set(pq.ParquetFile(path).schema_arrow.names)


def _require_columns(path: Path, required: Iterable[str]) -> set[str]:
    names = _schema_names(path)
    missing = set(required) - names
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return names


def _unique_ids(path: Path, column: str = "book_id") -> pd.Index:
    values = pd.read_parquet(path, columns=[column])[column].astype(str)
    if values.isna().any() or values.eq("").any():
        raise ValueError(f"{path} contains empty {column} values.")
    if values.duplicated().any():
        raise ValueError(f"{path} contains duplicate {column} values.")
    return pd.Index(values)


def _assert_same_ids(label: str, expected: pd.Index, actual: pd.Index) -> None:
    expected_set = set(expected)
    actual_set = set(actual)
    if expected_set != actual_set:
        raise ValueError(
            f"{label} id mismatch: missing={len(expected_set - actual_set)}, "
            f"extra={len(actual_set - expected_set)}"
        )


def validate_artifacts(
    books_path: Path = BOOKS_MASTER_PATH,
    features_path: Path = MASTER_FEATURE_MATRIX_PATH,
    clusters_path: Path = BOOK_CLUSTERS_PATH,
    interactions_path: Path = INTERACTIONS_CURATED_PATH,
    user_features_path: Path = USER_FEATURES_GLOBAL_PATH,
    user_matrix_path: Path = USER_MATRIX_PATH,
    user_meta_path: Path = USER_META_PATH,
    user_centroids_path: Path = USER_CENTROIDS_PATH,
    cooccurrence_path: Path | None = BOOK_COOCCURRENCE_PATH,
) -> dict[str, int]:
    """Raise on contract violations and return useful artifact row counts."""
    _require_columns(books_path, BOOKS_REQUIRED)
    feature_names = _require_columns(features_path, {"book_id", "pc_0"})
    _require_columns(clusters_path, {"book_id", "cluster"})
    _require_columns(interactions_path, INTERACTIONS_REQUIRED)
    _require_columns(user_features_path, {"user_id", "valid"})
    matrix_names = _require_columns(user_matrix_path, {"user_id", "pc_0"})
    _require_columns(user_meta_path, {"user_id", "positive_count", "is_cold_start"})
    centroid_names = _require_columns(
        user_centroids_path,
        {"user_id", "centroid_id", "n_books", "centroid_weight", "pc_0"},
    )

    feature_pcs = sorted(name for name in feature_names if name.startswith("pc_"))
    matrix_pcs = sorted(name for name in matrix_names if name.startswith("pc_"))
    centroid_pcs = sorted(name for name in centroid_names if name.startswith("pc_"))
    if feature_pcs != matrix_pcs or feature_pcs != centroid_pcs:
        raise ValueError("Book, user-matrix and user-centroid PCA schemas do not match.")

    book_ids = _unique_ids(books_path)
    feature_ids = _unique_ids(features_path)
    cluster_ids = _unique_ids(clusters_path)
    _assert_same_ids("master feature matrix", book_ids, feature_ids)
    _assert_same_ids("book clusters", book_ids, cluster_ids)

    clusters = pd.read_parquet(clusters_path, columns=["cluster"])["cluster"]
    if clusters.isna().any() or clusters.nunique() != 100:
        raise ValueError("Production clustering must contain exactly 100 non-null clusters.")

    global_users = pd.read_parquet(user_features_path, columns=["user_id", "valid"])
    global_users["user_id"] = global_users["user_id"].astype(str)
    valid_users = set(global_users.loc[global_users["valid"].fillna(False), "user_id"])
    meta = pd.read_parquet(user_meta_path, columns=["user_id", "positive_count"])
    meta["user_id"] = meta["user_id"].astype(str)
    if meta["user_id"].duplicated().any():
        raise ValueError("user_meta contains duplicate user_id values.")
    if set(meta["user_id"]) != valid_users:
        raise ValueError("user_meta users do not match globally valid users.")

    matrix_users = _unique_ids(user_matrix_path, "user_id")
    positive_users = set(meta.loc[meta["positive_count"] > 0, "user_id"])
    if set(matrix_users) != positive_users:
        raise ValueError("user_matrix users do not match positive-history users in user_meta.")

    centroid_users = set(
        pd.read_parquet(user_centroids_path, columns=["user_id"])["user_id"].astype(str)
    )
    if not centroid_users.issubset(set(matrix_users)):
        raise ValueError("user_centroids contains users absent from user_matrix.")

    if cooccurrence_path is not None and cooccurrence_path.exists():
        _require_columns(
            cooccurrence_path,
            {"book_id_a", "book_id_b", "pmi", "co_count"},
        )
        pairs = pd.read_parquet(cooccurrence_path)
        pairs["book_id_a"] = pairs["book_id_a"].astype(str)
        pairs["book_id_b"] = pairs["book_id_b"].astype(str)
        if not (pairs["book_id_a"] < pairs["book_id_b"]).all():
            raise ValueError("book_cooccurrence pairs must satisfy book_id_a < book_id_b.")
        if pairs[["book_id_a", "book_id_b"]].duplicated().any():
            raise ValueError("book_cooccurrence contains duplicate pairs.")
        if not set(pairs["book_id_a"]).union(pairs["book_id_b"]).issubset(set(book_ids)):
            raise ValueError("book_cooccurrence references book IDs outside books_master.")
        pmi = pd.to_numeric(pairs["pmi"], errors="coerce").to_numpy(dtype=np.float64)
        co_count = pd.to_numeric(pairs["co_count"], errors="coerce")
        if not np.isfinite(pmi).all() or (pmi < 0).any():
            raise ValueError("book_cooccurrence PMI must be finite and non-negative.")
        if co_count.isna().any() or (co_count < 3).any():
            raise ValueError("book_cooccurrence co_count must be at least 3.")

    paths = {
        "books": books_path,
        "features": features_path,
        "clusters": clusters_path,
        "interactions": interactions_path,
        "global_users": user_features_path,
        "user_matrix": user_matrix_path,
        "user_meta": user_meta_path,
        "user_centroids": user_centroids_path,
    }
    if cooccurrence_path is not None and cooccurrence_path.exists():
        paths["book_cooccurrence"] = cooccurrence_path
    return {
        label: pq.ParquetFile(path).metadata.num_rows
        for label, path in paths.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    counts = validate_artifacts()
    print("V1 artifact validation passed.")
    for label, count in counts.items():
        print(f"- {label}: {count:,} rows")


if __name__ == "__main__":
    main()
