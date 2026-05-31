from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


@dataclass(frozen=True)
class CategoryConfig:
    key: str
    display_name: str
    books_file: Path
    interactions_file: Path
    interim_dir: Path
    processed_dir: Path


CATEGORIES: dict[str, CategoryConfig] = {
    "fantasy_paranormal": CategoryConfig(
        key="fantasy_paranormal",
        display_name="Fantasy & Paranormal",
        books_file=RAW_DIR / "goodreads_books_fantasy_paranormal.json.gz",
        interactions_file=RAW_DIR / "goodreads_interactions_fantasy_paranormal.json.gz",
        interim_dir=INTERIM_DIR / "fantasy_paranormal",
        processed_dir=PROCESSED_DIR / "fantasy_paranormal",
    ),
    "history_biography": CategoryConfig(
        key="history_biography",
        display_name="History & Biography",
        books_file=RAW_DIR / "goodreads_books_history_biography.json.gz",
        interactions_file=RAW_DIR / "goodreads_interactions_history_biography.json.gz",
        interim_dir=INTERIM_DIR / "history_biography",
        processed_dir=PROCESSED_DIR / "history_biography",
    ),
    "mystery_thriller_crime": CategoryConfig(
        key="mystery_thriller_crime",
        display_name="Mystery, Thriller & Crime",
        books_file=RAW_DIR / "goodreads_books_mystery_thriller_crime.json.gz",
        interactions_file=RAW_DIR / "goodreads_interactions_mystery_thriller_crime.json.gz",
        interim_dir=INTERIM_DIR / "mystery_thriller_crime",
        processed_dir=PROCESSED_DIR / "mystery_thriller_crime",
    ),
    "romance": CategoryConfig(
        key="romance",
        display_name="Romance",
        books_file=RAW_DIR / "goodreads_books_romance.json.gz",
        interactions_file=RAW_DIR / "goodreads_interactions_romance.json.gz",
        interim_dir=INTERIM_DIR / "romance",
        processed_dir=PROCESSED_DIR / "romance",
    ),
    "young_adult": CategoryConfig(
        key="young_adult",
        display_name="Young Adult",
        books_file=RAW_DIR / "goodreads_books_young_adult.json.gz",
        interactions_file=RAW_DIR / "goodreads_interactions_young_adult.json.gz",
        interim_dir=INTERIM_DIR / "young_adult",
        processed_dir=PROCESSED_DIR / "young_adult",
    ),
}


# Global deduplicated interaction artifacts (built by src.curation.interactions).
# Canonical, cross-category, single source of truth for user-side features.
INTERACTIONS_CURATED_GLOBAL_PATH = PROCESSED_DIR / "interactions_curated.parquet"
REVIEW_TEXTS_PATH = PROCESSED_DIR / "review_texts.parquet"
USER_FEATURES_GLOBAL_PATH = PROCESSED_DIR / "user_features_global.parquet"


BOOK_NUMERIC_COLUMNS = [
    "text_reviews_count",
    "average_rating",
    "num_pages",
    "publication_day",
    "publication_month",
    "publication_year",
    "ratings_count",
]

GOODREADS_DATE_COLUMNS = ["date_added", "date_updated", "read_at", "started_at"]
