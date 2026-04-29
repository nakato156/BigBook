from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CATEGORIES, PROCESSED_DIR, PROJECT_ROOT
from src.utils.io import safe_write_parquet


FEATURES_DIR = PROJECT_ROOT / "data" / "features"

LEGACY_PROCESSED_DIRS = {
    "fantasy_paranormal": [PROCESSED_DIR / "fantasy"],
    "history_biography": [PROCESSED_DIR / "history"],
}

BOOK_FEATURE_COLUMNS = [
    "category",
    "book_id",
    "title",
    "description",
    "average_rating",
    "ratings_count_p99",
    "text_reviews_count_p99",
    "num_pages_p99",
    "publication_year_clean",
    "is_ebook",
    "language_code",
    "format",
    "publisher",
    "primary_author_id",
    "author_count",
    "is_in_series",
    "series_count",
    "top_shelf",
    "to_read_count",
    "interaction_count_p99",
    "mean_rating",
    "is_cold_start",
]

INTERACTION_FEATURE_COLUMNS = [
    "category",
    "user_id",
    "book_id",
    "review_id",
    "is_read",
    "rating",
    "rating_clean",
    "rating_missing",
    "has_review_text",
    "reading_duration_days",
    "has_reading_duration",
    "engagement_mode",
    "user_rating_bias",
    "date_added",
    "date_updated",
]

USER_FEATURE_COLUMNS = [
    "user_id",
    "interaction_count",
    "rated_count",
    "review_count",
    "read_count",
    "shelf_only_count",
    "mean_rating",
    "rating_std",
    "user_rating_bias",
    "avg_reading_duration_days",
    "has_reading_duration_rate",
    "category_count",
    "categories",
]


@dataclass(frozen=True)
class CategoryArtifacts:
    category: str
    processed_dir: Path
    books_path: Path
    interactions_path: Path


def _has_curated_files(path: Path) -> bool:
    return (path / "books_curated.parquet").exists() and (path / "interactions_curated.parquet").exists()


def resolve_category_artifacts(category_key: str) -> CategoryArtifacts | None:
    cfg = CATEGORIES[category_key]
    candidate_dirs = [cfg.processed_dir, *LEGACY_PROCESSED_DIRS.get(category_key, [])]
    for processed_dir in candidate_dirs:
        if _has_curated_files(processed_dir):
            return CategoryArtifacts(
                category=category_key,
                processed_dir=processed_dir,
                books_path=processed_dir / "books_curated.parquet",
                interactions_path=processed_dir / "interactions_curated.parquet",
            )
    return None


def discover_available_artifacts() -> tuple[list[CategoryArtifacts], list[dict[str, Any]]]:
    available = []
    missing = []
    for category_key, cfg in CATEGORIES.items():
        artifacts = resolve_category_artifacts(category_key)
        if artifacts is not None:
            available.append(artifacts)
            continue
        checked_dirs = [cfg.processed_dir, *LEGACY_PROCESSED_DIRS.get(category_key, [])]
        missing.append(
            {
                "category": category_key,
                "checked_dirs": [str(path) for path in checked_dirs],
                "reason": "missing books_curated.parquet or interactions_curated.parquet",
            }
        )
    return available, missing


def _empty_series(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NA, index=index, dtype="object")


def _series_or_default(df: pd.DataFrame, column: str, default: Any = pd.NA) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def _with_canonical_p99_columns(books: pd.DataFrame) -> pd.DataFrame:
    out = books.copy()
    for column in ["ratings_count", "text_reviews_count", "num_pages", "interaction_count"]:
        canonical = f"{column}_p99"
        capped = f"{column}_p99_capped"
        if canonical not in out.columns and capped in out.columns:
            out[canonical] = out[capped]
    return out


def _theme_columns(books: pd.DataFrame) -> list[str]:
    prefixes = ("theme_",)
    suffixes = ("_theme_count", "_theme_score")
    return [
        column
        for column in books.columns
        if column.startswith(prefixes) or column.endswith(suffixes)
    ]


def build_book_features(books: pd.DataFrame, category: str) -> pd.DataFrame:
    source = _with_canonical_p99_columns(books)
    features = pd.DataFrame(index=source.index)
    features["category"] = category

    for column in BOOK_FEATURE_COLUMNS:
        if column == "category":
            continue
        features[column] = source[column] if column in source.columns else _empty_series(source.index)

    features["mean_rating"] = pd.to_numeric(features["mean_rating"], errors="coerce")
    average_rating = pd.to_numeric(features["average_rating"], errors="coerce")
    features["mean_rating"] = features["mean_rating"].fillna(average_rating)

    for column in _theme_columns(source):
        if column not in features.columns:
            features[column] = source[column]

    return features.reset_index(drop=True)


def build_interaction_features(interactions: pd.DataFrame, category: str) -> pd.DataFrame:
    features = pd.DataFrame(index=interactions.index)
    features["category"] = category
    for column in INTERACTION_FEATURE_COLUMNS:
        if column == "category":
            continue
        features[column] = interactions[column] if column in interactions.columns else _empty_series(interactions.index)
    return features.reset_index(drop=True)


def build_global_user_features(interactions: pd.DataFrame) -> pd.DataFrame:
    if interactions.empty:
        return pd.DataFrame(columns=USER_FEATURE_COLUMNS)

    data = interactions.copy()
    data["rating_clean"] = pd.to_numeric(_series_or_default(data, "rating_clean"), errors="coerce")
    data["reading_duration_days"] = pd.to_numeric(_series_or_default(data, "reading_duration_days"), errors="coerce")
    data["has_review_text"] = _series_or_default(data, "has_review_text", False).fillna(False).astype(bool)
    data["has_reading_duration"] = _series_or_default(data, "has_reading_duration", False).fillna(False).astype(bool)
    data["is_read"] = _series_or_default(data, "is_read", False).fillna(False).astype(bool)
    data["engagement_mode"] = _series_or_default(data, "engagement_mode").astype("string")

    grouped = data.groupby("user_id", dropna=False)
    features = grouped.agg(
        interaction_count=("book_id", "size"),
        rated_count=("rating_clean", "count"),
        review_count=("has_review_text", "sum"),
        read_count=("is_read", "sum"),
        shelf_only_count=("engagement_mode", lambda values: int(values.eq("shelf_only").sum())),
        mean_rating=("rating_clean", "mean"),
        rating_std=("rating_clean", "std"),
        avg_reading_duration_days=("reading_duration_days", "mean"),
        has_reading_duration_rate=("has_reading_duration", "mean"),
        category_count=("category", "nunique"),
        categories=("category", lambda values: "|".join(sorted(values.dropna().astype(str).unique()))),
    ).reset_index()

    global_mean_rating = data["rating_clean"].mean()
    features["user_rating_bias"] = features["mean_rating"] - global_mean_rating
    features = features[USER_FEATURE_COLUMNS]

    int_columns = ["interaction_count", "rated_count", "review_count", "read_count", "shelf_only_count", "category_count"]
    for column in int_columns:
        features[column] = features[column].astype("int64")

    return features


def validate_book_features(source: pd.DataFrame, features: pd.DataFrame) -> None:
    if features["book_id"].duplicated().any():
        raise ValueError("book_features must have one row per book_id")

    average_rating = pd.to_numeric(features["average_rating"], errors="coerce")
    mean_rating = pd.to_numeric(features["mean_rating"], errors="coerce")
    if mean_rating[average_rating.notna()].isna().any():
        raise ValueError("mean_rating must be imputed wherever average_rating is available")

    comparable_columns = [
        column
        for column in BOOK_FEATURE_COLUMNS
        if column not in {"category", "mean_rating"} and column in source.columns and column in features.columns
    ]
    for column in comparable_columns:
        if features[column].isna().sum() < source[column].isna().sum():
            raise ValueError(f"{column} was imputed unexpectedly")


def validate_interaction_features(source: pd.DataFrame, features: pd.DataFrame) -> None:
    for column in ["rating_clean", "reading_duration_days"]:
        if column in source.columns and features[column].isna().sum() != source[column].isna().sum():
            raise ValueError(f"{column} missingness must be preserved")


def validate_user_features(features: pd.DataFrame) -> None:
    if features["user_id"].duplicated().any():
        raise ValueError("user_features_global must have unique user_id values")


def generate_feature_matrices(output_dir: Path = FEATURES_DIR) -> dict[str, Any]:
    available, missing = discover_available_artifacts()
    generated_categories = []
    interaction_frames = []

    for artifacts in available:
        category_output_dir = output_dir / artifacts.category
        books = pd.read_parquet(artifacts.books_path)
        interactions = pd.read_parquet(artifacts.interactions_path)

        book_features = build_book_features(books, artifacts.category)
        interaction_features = build_interaction_features(interactions, artifacts.category)

        validate_book_features(_with_canonical_p99_columns(books), book_features)
        validate_interaction_features(interactions, interaction_features)

        safe_write_parquet(book_features, category_output_dir / "book_features.parquet")
        safe_write_parquet(interaction_features, category_output_dir / "interaction_features.parquet")
        interaction_frames.append(interaction_features)

        generated_categories.append(
            {
                "category": artifacts.category,
                "processed_dir": str(artifacts.processed_dir),
                "book_features": str(category_output_dir / "book_features.parquet"),
                "interaction_features": str(category_output_dir / "interaction_features.parquet"),
                "book_rows": int(len(book_features)),
                "interaction_rows": int(len(interaction_features)),
            }
        )

    all_interactions = pd.concat(interaction_frames, ignore_index=True) if interaction_frames else pd.DataFrame()
    user_features = build_global_user_features(all_interactions)
    validate_user_features(user_features)
    safe_write_parquet(user_features, output_dir / "user_features_global.parquet")

    manifest = {
        "generated_categories": generated_categories,
        "missing_categories": missing,
        "user_features_global": str(output_dir / "user_features_global.parquet"),
        "user_rows": int(len(user_features)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def manifest_summary(manifest: dict[str, Any]) -> pd.DataFrame:
    generated = [
        {
            "category": item["category"],
            "status": "generated",
            "book_rows": item["book_rows"],
            "interaction_rows": item["interaction_rows"],
            "processed_dir": item["processed_dir"],
        }
        for item in manifest["generated_categories"]
    ]
    missing = [
        {
            "category": item["category"],
            "status": "missing",
            "book_rows": 0,
            "interaction_rows": 0,
            "processed_dir": "|".join(item["checked_dirs"]),
        }
        for item in manifest["missing_categories"]
    ]
    return pd.DataFrame(generated + missing)
