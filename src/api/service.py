from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.api.schemas import (
    GENRE_LABELS,
    BookSearchResult,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    SampleUser,
)
from src.config import BOOKS_MASTER_PATH, USER_META_PATH
from src.reduction.recommend import GENRE_COLUMNS, Recommender

DESCRIPTION_LIMIT = 360


def _is_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return bool(str(value).strip())


def _clean_text(value: object, limit: int | None = None) -> str | None:
    if not _is_present(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if limit is not None and len(text) > limit:
        return text[: limit - 1].rstrip() + "..."
    return text


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value: object) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def genres_from_row(row: pd.Series) -> list[str]:
    genres: list[str] = []
    for column in GENRE_COLUMNS:
        if int(row.get(column, 0) or 0) == 1:
            key = column.replace("genre_", "")
            genres.append(GENRE_LABELS.get(key, key.title()))
    return genres


@dataclass
class Catalog:
    books: pd.DataFrame
    metadata_by_book_id: dict[str, dict]

    @classmethod
    def from_artifacts(cls) -> "Catalog":
        columns = [
            "book_id",
            "title",
            "description",
            "average_rating",
            "ratings_count",
            "num_pages",
            "publication_year",
            *GENRE_COLUMNS,
        ]
        books = pd.read_parquet(BOOKS_MASTER_PATH, columns=columns)
        books["book_id"] = books["book_id"].astype(str)
        books["title"] = books["title"].fillna("").astype(str)
        books["_search_title"] = books["title"].str.lower().str.normalize("NFKD")
        metadata_by_book_id = {
            str(row.book_id): {
                "description": _clean_text(row.description, DESCRIPTION_LIMIT),
                "average_rating": _safe_float(row.average_rating),
                "ratings_count": _safe_int(row.ratings_count),
                "num_pages": _safe_float(row.num_pages),
                "publication_year": _safe_float(row.publication_year),
                "genres": genres_from_row(pd.Series(row._asdict())),
            }
            for row in books.itertuples(index=False)
        }
        return cls(books=books, metadata_by_book_id=metadata_by_book_id)

    def search(self, query: str, limit: int = 10) -> list[BookSearchResult]:
        cleaned = query.strip().lower()
        if not cleaned:
            return []
        limit = max(1, min(int(limit), 25))
        contains = self.books["_search_title"].str.contains(re.escape(cleaned), na=False, regex=True)
        matches = self.books.loc[contains].copy()
        if matches.empty:
            return []
        matches["_starts"] = matches["_search_title"].str.startswith(cleaned).astype(int)
        matches["_ratings"] = pd.to_numeric(matches["ratings_count"], errors="coerce").fillna(0)
        matches = matches.sort_values(["_starts", "_ratings", "title"], ascending=[False, False, True])
        return [self._book_result(row) for _, row in matches.head(limit).iterrows()]

    def _book_result(self, row: pd.Series) -> BookSearchResult:
        meta = self.metadata_by_book_id.get(str(row["book_id"]), {})
        return BookSearchResult(
            book_id=str(row["book_id"]),
            title=str(row["title"]),
            description=meta.get("description"),
            genres=meta.get("genres", []),
            average_rating=meta.get("average_rating"),
            ratings_count=meta.get("ratings_count"),
            num_pages=meta.get("num_pages"),
            publication_year=meta.get("publication_year"),
        )


@dataclass
class ApiState:
    recommender: Recommender | None = None
    catalog: Catalog | None = None
    load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.recommender is not None and self.catalog is not None and self.load_error is None


def build_state() -> ApiState:
    from dataclasses import replace

    from src.reduction.ranking import RankingConfig

    config = replace(RankingConfig(), k=20)
    recommender = Recommender.from_artifacts(config=config)
    catalog = Catalog.from_artifacts()
    return ApiState(recommender=recommender, catalog=catalog)


def recommendation_response(
    state: ApiState,
    request: RecommendationRequest,
) -> RecommendationResponse:
    if state.recommender is None or state.catalog is None:
        raise RuntimeError(state.load_error or "Recommendation artifacts are not loaded.")

    if request.user_id:
        mode = "user"
        user_id = request.user_id
        frame = state.recommender.recommend(
            user_id,
            set(request.exclude_book_ids),
        )
    elif request.seed_book_ids:
        mode = "seed"
        user_id = "__seed_profile__"
        frame = state.recommender.recommend(
            user_id,
            set(request.exclude_book_ids),
            seed_book_ids=request.seed_book_ids,
        )
    else:
        mode = "cold_start"
        user_id = "__cold_start__"
        frame = state.recommender.recommend_cold_start(user_id, set(request.exclude_book_ids))

    records = frame.head(request.k).to_dict(orient="records")
    return RecommendationResponse(
        mode=mode,
        user_id=user_id,
        k=request.k,
        recommendations=[
            recommendation_item(record, state.catalog.metadata_by_book_id)
            for record in records
        ],
    )


def recommendation_item(record: dict, metadata_by_book_id: dict[str, dict]) -> RecommendationItem:
    book_id = str(record.get("book_id", ""))
    meta = metadata_by_book_id.get(book_id, {})
    genres = meta.get("genres")
    if not genres and _is_present(record.get("genres")):
        genres = [GENRE_LABELS.get(g, g.title()) for g in str(record["genres"]).split("|") if g]
    return RecommendationItem(
        rank=int(record.get("rank", 0)),
        book_id=book_id,
        title=str(record.get("title", "")),
        description=meta.get("description"),
        genres=genres or [],
        slot=str(record.get("slot", "interest")),
        fine_cluster=int(record.get("fine_cluster", -1)),
        macro_cluster=int(record.get("macro_cluster", -1)),
        popularity_segment=str(record.get("popularity_segment", "unknown")),
        ratings_count=int(record.get("ratings_count", meta.get("ratings_count") or 0)),
        average_rating=meta.get("average_rating"),
        num_pages=_safe_float(record.get("num_pages")) if record.get("num_pages") is not None else meta.get("num_pages"),
        publication_year=meta.get("publication_year"),
    )


def sample_users(limit: int = 8) -> list[SampleUser]:
    limit = max(1, min(int(limit), 25))
    if not USER_META_PATH.exists():
        return []
    meta = pd.read_parquet(USER_META_PATH, columns=["user_id", "positive_count"])
    meta = meta.sort_values("positive_count", ascending=False).head(limit)
    return [
        SampleUser(user_id=str(row.user_id), positive_count=int(row.positive_count))
        for row in meta.itertuples(index=False)
    ]


def known_book_ids(catalog: Catalog, book_ids: Iterable[str]) -> set[str]:
    available = set(catalog.metadata_by_book_id)
    return {str(book_id) for book_id in book_ids if str(book_id) in available}
