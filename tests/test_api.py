from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.service import ApiState, Catalog


class FakeRecommender:
    user_ids = ["user_a", "user_b"]

    def recommend(self, user_id: str, exclude_book_ids: set[str], seed_book_ids=()):
        assert user_id
        assert isinstance(exclude_book_ids, set)
        slot = "interest" if seed_book_ids else "cold_start"
        return pd.DataFrame(
            [
                {
                    "user_id": user_id,
                    "rank": 1,
                    "book_id": "1",
                    "title": "A Test Book",
                    "slot": slot,
                    "fine_cluster": 2,
                    "macro_cluster": 1,
                    "genres": "fantasy",
                    "ratings_count": 42,
                    "popularity_segment": "tail",
                    "num_pages": 180.0,
                }
            ]
        )

    def recommend_cold_start(self, user_id: str, exclude_book_ids: set[str]):
        return self.recommend(user_id, exclude_book_ids)


def _catalog() -> Catalog:
    books = pd.DataFrame(
        [
            {
                "book_id": "1",
                "title": "A Test Book",
                "description": "A concise description.",
                "average_rating": 4.2,
                "ratings_count": 42,
                "num_pages": 180.0,
                "publication_year": 2020.0,
                "genre_fantasy": 1,
                "genre_mystery": 0,
                "genre_history": 0,
                "genre_ya": 0,
                "genre_romance": 0,
                "_search_title": "a test book",
            }
        ]
    )
    return Catalog(
        books=books,
        metadata_by_book_id={
            "1": {
                "description": "A concise description.",
                "average_rating": 4.2,
                "ratings_count": 42,
                "num_pages": 180.0,
                "publication_year": 2020.0,
                "genres": ["Fantasy"],
            }
        },
    )


def _client() -> TestClient:
    app = create_app(load_artifacts=False)
    app.state.bigbook = ApiState(recommender=FakeRecommender(), catalog=_catalog())
    return TestClient(app)


def test_health_reports_ready_state() -> None:
    client = _client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_search_books_returns_catalog_metadata() -> None:
    client = _client()
    response = client.get("/books/search", params={"q": "test"})
    assert response.status_code == 200
    assert response.json()[0]["book_id"] == "1"
    assert response.json()[0]["genres"] == ["Fantasy"]


def test_recommendations_support_seed_mode() -> None:
    client = _client()
    response = client.post("/recommendations", json={"seed_book_ids": ["1"], "k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "seed"
    assert body["recommendations"][0]["description"] == "A concise description."


def test_recommendations_reject_mixed_modes_with_400() -> None:
    client = _client()
    response = client.post(
        "/recommendations",
        json={"user_id": "user_a", "seed_book_ids": ["1"]},
    )
    assert response.status_code == 400
