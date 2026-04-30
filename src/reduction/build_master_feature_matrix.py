from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.reduction.embeddings import (
    DEFAULT_EMBEDDINGS_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    embedding_columns,
    load_or_create_description_embeddings,
)
from src.reduction.pca import (
    component_block_norm_shares,
    count_embedding_dominated_components,
    fit_variance_pca,
    standardize_and_weight_blocks,
)
from src.utils.io import safe_write_parquet


FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MASTER_PATH = PROCESSED_DIR / "books_master.parquet"
FEATURE_MATRIX_PATH = FEATURES_DIR / "master_feature_matrix.parquet"
PCA_MODEL_PATH = FEATURES_DIR / "master_pca_model.joblib"
PCA_META_PATH = FEATURES_DIR / "master_pca_meta.json"
VARIANCE_THRESHOLD = 0.95

REQUIRED_MASTER_COLUMNS = [
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
]
GENRE_COLUMNS = [
    "genre_fantasy",
    "genre_mystery",
    "genre_history",
    "genre_ya",
    "genre_romance",
]
BINARY_BASE_COLUMNS = ["series", *GENRE_COLUMNS]


def _safe_median(values: pd.Series, fallback: float = 0.0) -> float:
    median = pd.to_numeric(values, errors="coerce").median()
    if pd.isna(median):
        return fallback
    return float(median)


def _numeric(values: pd.Series, fill_value: float | None = None) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    if fill_value is not None:
        out = out.fillna(fill_value)
    return out.astype("float32")


def build_numeric_block(master: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    medians = {
        "average_rating": _safe_median(master["average_rating"]),
        "num_pages": _safe_median(master["num_pages"]),
        "publication_year": _safe_median(master["publication_year"]),
        "author_count": _safe_median(master["author_count"], fallback=1.0),
        "genre_count": _safe_median(master["genre_count"], fallback=1.0),
    }

    block = pd.DataFrame(index=master.index)
    block["average_rating"] = _numeric(master["average_rating"], medians["average_rating"])
    block["log_ratings_count"] = np.log1p(_numeric(master["ratings_count"], 0.0).clip(lower=0))
    block["log_text_reviews_count"] = np.log1p(_numeric(master["text_reviews_count"], 0.0).clip(lower=0))

    num_pages = pd.to_numeric(master["num_pages"], errors="coerce")
    block["num_pages"] = num_pages.fillna(medians["num_pages"]).astype("float32")
    block["num_pages_missing"] = num_pages.isna().astype("float32")

    publication_year = pd.to_numeric(master["publication_year"], errors="coerce")
    block["publication_year"] = publication_year.fillna(medians["publication_year"]).astype("float32")
    block["publication_year_missing"] = publication_year.isna().astype("float32")

    block["author_count"] = _numeric(master["author_count"], medians["author_count"])
    block["genre_count"] = _numeric(master["genre_count"], medians["genre_count"])
    return block, medians


def _sanitize_language_code(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower()).strip("_")
    return sanitized or "unknown"


def _dedupe_column_names(values: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    names = []
    for value in values:
        base = _sanitize_language_code(value)
        count = seen.get(base, 0)
        seen[base] = count + 1
        names.append(base if count == 0 else f"{base}_{count}")
    return names


def build_language_categories(language_code: pd.Series, *, top_n: int = 10) -> list[str]:
    normalized = language_code.fillna("unknown").astype(str).str.strip().replace("", "unknown")
    counts = normalized[normalized.ne("other")].value_counts()
    return counts.head(top_n).index.tolist()


def build_binary_block(
    master: pd.DataFrame,
    *,
    lang_categories: list[str] | None = None,
    top_n: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    categories = lang_categories or build_language_categories(master["language_code"], top_n=top_n)
    block = pd.DataFrame(index=master.index)

    for column in BINARY_BASE_COLUMNS:
        block[column] = _numeric(master[column], 0.0).clip(lower=0, upper=1)

    language = master["language_code"].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    language = language.where(language.isin(categories), "other")
    category_column_names = _dedupe_column_names(categories)
    for category, column_name in zip(categories, category_column_names):
        block[f"language_code_{column_name}"] = language.eq(category).astype("float32")
    block["language_code_other"] = language.eq("other").astype("float32")
    return block, categories


def validate_master_columns(master: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_MASTER_COLUMNS if column not in master.columns]
    if missing:
        raise ValueError("books_master.parquet is missing required columns: " + ", ".join(missing))
    if master["book_id"].duplicated().any():
        raise ValueError("books_master.parquet must contain one row per book_id.")


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denominator)


def semantic_spot_checks(master: pd.DataFrame, reduced: np.ndarray, *, limit: int = 3) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for genre_column in GENRE_COLUMNS:
        same_idx = np.flatnonzero(master[genre_column].astype(bool).to_numpy())
        diff_idx = np.flatnonzero(~master[genre_column].astype(bool).to_numpy())
        if len(same_idx) < 2 or len(diff_idx) < 1:
            continue

        first, second = int(same_idx[0]), int(same_idx[1])
        other = int(diff_idx[0])
        checks.append(
            {
                "genre": genre_column,
                "same_genre_books": [
                    str(master.iloc[first]["book_id"]),
                    str(master.iloc[second]["book_id"]),
                ],
                "different_genre_book": str(master.iloc[other]["book_id"]),
                "same_genre_cosine_distance": _cosine_distance(reduced[first], reduced[second]),
                "different_genre_cosine_distance": _cosine_distance(reduced[first], reduced[other]),
            }
        )
        if len(checks) >= limit:
            break
    return checks


def build_feature_matrix(master: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    validate_master_columns(master)
    master = master.copy()
    master["book_id"] = master["book_id"].astype(str)

    numeric_block, numeric_medians = build_numeric_block(master)
    binary_block, lang_categories = build_binary_block(master)
    embeddings = load_or_create_description_embeddings(master)
    embedding_block = embeddings[embedding_columns(EMBEDDING_DIM)]

    scaled = standardize_and_weight_blocks(
        {
            "numeric": numeric_block,
            "binary": binary_block,
            "embeddings": embedding_block,
        }
    )
    reduced, pca = fit_variance_pca(scaled.matrix, variance_threshold=VARIANCE_THRESHOLD)

    feature_matrix = pd.DataFrame(
        reduced,
        columns=[f"pc_{idx}" for idx in range(reduced.shape[1])],
    )
    feature_matrix.insert(0, "book_id", master["book_id"].to_numpy())

    component_summaries = component_block_norm_shares(pca, scaled.block_slices, first_n=5)
    embedding_dominated_count = count_embedding_dominated_components(
        pca,
        scaled.block_slices,
        first_n=5,
    )
    spot_checks = semantic_spot_checks(master, reduced)

    model_artifact = {
        "scaler_a": scaled.scalers["numeric"],
        "scaler_b": scaled.scalers["binary"],
        "scaler_c": scaled.scalers["embeddings"],
        "scalers": scaled.scalers,
        "pca": pca,
        "block_dims": scaled.block_dims,
        "block_weights": scaled.block_weights,
        "block_slices": scaled.block_slices,
        "block_columns": scaled.block_columns,
        "numeric_medians": numeric_medians,
        "lang_categories": lang_categories,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "variance_threshold": VARIANCE_THRESHOLD,
        "source_path": str(MASTER_PATH),
    }

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(MASTER_PATH),
        "feature_matrix_path": str(FEATURE_MATRIX_PATH),
        "pca_model_path": str(PCA_MODEL_PATH),
        "embedding_cache_path": str(DEFAULT_EMBEDDINGS_PATH),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "variance_threshold": VARIANCE_THRESHOLD,
        "n_rows": int(len(master)),
        "n_components": int(reduced.shape[1]),
        "block_dims": scaled.block_dims,
        "block_weights": scaled.block_weights,
        "block_columns": scaled.block_columns,
        "lang_categories": lang_categories,
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
        "explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
        "component_block_norm_shares_first_5": component_summaries,
        "embedding_dominated_first_5_count": int(embedding_dominated_count),
        "semantic_spot_checks": spot_checks,
    }
    return feature_matrix, model_artifact, meta


def write_outputs(
    feature_matrix: pd.DataFrame,
    model_artifact: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    safe_write_parquet(feature_matrix, FEATURE_MATRIX_PATH)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, PCA_MODEL_PATH)
    PCA_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def print_validations(master_rows: int, feature_matrix: pd.DataFrame, meta: dict[str, Any]) -> None:
    explained_sum = meta["explained_variance_sum"]
    feature_rows = len(feature_matrix)
    print("\nFeature matrix + PCA build complete")
    print(f"Rows in master: {master_rows:,}")
    print(f"Rows in feature matrix: {feature_rows:,}")
    print(f"Explained variance sum: {explained_sum:.6f}")
    print(f"PCA components: {meta['n_components']}")
    print(
        "Embedding-dominated PCs among first 5 (>50% norm share): "
        f"{meta['embedding_dominated_first_5_count']}/5"
    )
    if feature_rows != master_rows:
        raise ValueError(
            f"Feature matrix row count {feature_rows:,} does not match master row count {master_rows:,}."
        )
    if explained_sum < VARIANCE_THRESHOLD:
        raise ValueError(
            f"PCA explained variance {explained_sum:.6f} is below {VARIANCE_THRESHOLD}."
        )
    if meta["embedding_dominated_first_5_count"] == 5:
        print("WARNING: all first 5 PCs are embedding-dominated; review block scaling diagnostics.")

    for check in meta["semantic_spot_checks"]:
        print(
            f"Spot-check {check['genre']}: same={check['same_genre_cosine_distance']:.4f}, "
            f"different={check['different_genre_cosine_distance']:.4f}"
        )


def main() -> None:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"{MASTER_PATH} does not exist. Run `python -m src.merge_master` first."
        )

    master = pd.read_parquet(MASTER_PATH)
    feature_matrix, model_artifact, meta = build_feature_matrix(master)
    write_outputs(feature_matrix, model_artifact, meta)
    print_validations(len(master), feature_matrix, meta)
    print(f"Feature matrix written to: {FEATURE_MATRIX_PATH}")
    print(f"PCA model written to: {PCA_MODEL_PATH}")
    print(f"Manifest written to: {PCA_META_PATH}")


if __name__ == "__main__":
    main()
