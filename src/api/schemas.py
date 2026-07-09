from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


GENRE_LABELS: dict[str, str] = {
    "fantasy": "Fantasy",
    "mystery": "Mystery",
    "history": "History",
    "ya": "Young Adult",
    "romance": "Romance",
}


class BookSearchResult(BaseModel):
    book_id: str
    title: str
    description: str | None = None
    genres: list[str] = Field(default_factory=list)
    average_rating: float | None = None
    ratings_count: int | None = None
    num_pages: float | None = None
    publication_year: float | None = None


class RecommendationRequest(BaseModel):
    user_id: str | None = None
    seed_book_ids: list[str] = Field(default_factory=list)
    exclude_book_ids: list[str] = Field(default_factory=list)
    k: int = Field(default=10, ge=1, le=20)

    @field_validator("user_id")
    @classmethod
    def clean_user_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("seed_book_ids", "exclude_book_ids")
    @classmethod
    def clean_ids(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = str(value).strip()
            if item and item not in seen:
                seen.add(item)
                cleaned.append(item)
        return cleaned

    @model_validator(mode="after")
    def choose_one_profile_source(self) -> "RecommendationRequest":
        if self.user_id and self.seed_book_ids:
            raise ValueError("Send either user_id or seed_book_ids, not both.")
        return self


class RecommendationItem(BaseModel):
    rank: int
    book_id: str
    title: str
    description: str | None = None
    genres: list[str] = Field(default_factory=list)
    slot: Literal["interest", "exploration", "cold_start", "hybrid_v12"] | str
    fine_cluster: int
    macro_cluster: int
    popularity_segment: str
    ratings_count: int
    average_rating: float | None = None
    num_pages: float | None = None
    publication_year: float | None = None


class RecommendationResponse(BaseModel):
    mode: Literal["seed", "user", "cold_start"]
    user_id: str
    k: int
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ready: bool
    books: int = 0
    users: int = 0
    error: str | None = None


class SampleUser(BaseModel):
    user_id: str
    positive_count: int | None = None
