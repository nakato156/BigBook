from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.validate_artifacts import BOOKS_REQUIRED, INTERACTIONS_REQUIRED, validate_artifacts


def _write(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_parquet(path, index=False)
    return path


def test_validate_artifacts_accepts_aligned_contracts(tmp_path) -> None:
    book_ids = ["b1", "b2"]
    books = pd.DataFrame({column: [0, 0] for column in BOOKS_REQUIRED})
    books["book_id"] = book_ids
    books["title"] = ["One", "Two"]
    books["description"] = ["", ""]
    books["language_code"] = ["eng", "eng"]

    features = pd.DataFrame({"book_id": book_ids, "pc_0": [0.1, 0.2]})
    clusters = pd.DataFrame(
        {
            "book_id": [f"b{i}" for i in range(100)],
            "cluster": list(range(100)),
        }
    )
    books = pd.concat(
        [
            books,
            pd.DataFrame(
                {
                    **{column: [0] * 98 for column in BOOKS_REQUIRED},
                    "book_id": [f"b{i}" for i in range(3, 101)],
                    "title": [f"Book {i}" for i in range(3, 101)],
                    "description": [""] * 98,
                    "language_code": ["eng"] * 98,
                }
            ),
        ],
        ignore_index=True,
    )
    features = pd.DataFrame(
        {"book_id": books["book_id"], "pc_0": range(len(books))}
    )
    clusters["book_id"] = books["book_id"]

    interactions = pd.DataFrame(
        {column: [0] for column in INTERACTIONS_REQUIRED}
    )
    interactions["interaction_key"] = [1]
    interactions["user_id"] = ["u1"]
    interactions["book_id"] = ["b1"]
    user_features = pd.DataFrame(
        {"user_id": ["u1", "u2"], "valid": [True, False]}
    )
    user_meta = pd.DataFrame(
        {"user_id": ["u1"], "positive_count": [1], "is_cold_start": [True]}
    )
    user_matrix = pd.DataFrame({"user_id": ["u1"], "pc_0": [0.1]})
    user_centroids = pd.DataFrame(
        {
            "user_id": ["u1"],
            "centroid_id": [0],
            "n_books": [1],
            "centroid_weight": [1.0],
            "pc_0": [0.1],
        }
    )

    counts = validate_artifacts(
        books_path=_write(tmp_path / "books.parquet", books),
        features_path=_write(tmp_path / "features.parquet", features),
        clusters_path=_write(tmp_path / "clusters.parquet", clusters),
        interactions_path=_write(tmp_path / "interactions.parquet", interactions),
        user_features_path=_write(tmp_path / "users.parquet", user_features),
        user_matrix_path=_write(tmp_path / "matrix.parquet", user_matrix),
        user_meta_path=_write(tmp_path / "meta.parquet", user_meta),
        user_centroids_path=_write(tmp_path / "centroids.parquet", user_centroids),
    )

    assert counts["books"] == 100
    assert counts["interactions"] == 1


def test_validate_artifacts_rejects_misaligned_book_ids(tmp_path) -> None:
    books = pd.DataFrame({column: [0] * 100 for column in BOOKS_REQUIRED})
    books["book_id"] = [f"b{i}" for i in range(100)]
    books["title"] = books["book_id"]
    books["description"] = ""
    books["language_code"] = "eng"
    features = pd.DataFrame(
        {"book_id": [*books["book_id"][:-1], "other"], "pc_0": range(100)}
    )
    clusters = pd.DataFrame({"book_id": books["book_id"], "cluster": range(100)})
    interactions = pd.DataFrame({column: [0] for column in INTERACTIONS_REQUIRED})
    interactions["interaction_key"] = [1]
    interactions["user_id"] = ["u"]
    interactions["book_id"] = ["b0"]
    users = pd.DataFrame({"user_id": ["u"], "valid": [True]})
    meta = pd.DataFrame(
        {"user_id": ["u"], "positive_count": [1], "is_cold_start": [True]}
    )
    matrix = pd.DataFrame({"user_id": ["u"], "pc_0": [0.1]})
    centroids = pd.DataFrame(
        {
            "user_id": ["u"],
            "centroid_id": [0],
            "n_books": [1],
            "centroid_weight": [1.0],
            "pc_0": [0.1],
        }
    )

    try:
        validate_artifacts(
            books_path=_write(tmp_path / "books.parquet", books),
            features_path=_write(tmp_path / "features.parquet", features),
            clusters_path=_write(tmp_path / "clusters.parquet", clusters),
            interactions_path=_write(tmp_path / "interactions.parquet", interactions),
            user_features_path=_write(tmp_path / "users.parquet", users),
            user_matrix_path=_write(tmp_path / "matrix.parquet", matrix),
            user_meta_path=_write(tmp_path / "meta.parquet", meta),
            user_centroids_path=_write(tmp_path / "centroids.parquet", centroids),
        )
    except ValueError as exc:
        assert "master feature matrix id mismatch" in str(exc)
    else:
        raise AssertionError("Expected an id-alignment failure.")
