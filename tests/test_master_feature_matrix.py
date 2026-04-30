from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.reduction.build_master_feature_matrix import (
    build_binary_block,
    build_numeric_block,
)
from src.reduction.pca import fit_variance_pca, standardize_and_weight_blocks


def _sample_master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "book_id": ["1", "2", "3", "4", "5", "6"],
            "title": ["A", "B", "C", "D", "E", "F"],
            "description": ["a", "", None, "d", "e", "f"],
            "series": [1, 0, 1, 0, 0, 1],
            "language_code": ["en", "en", "es", "fr", "de", "zz"],
            "average_rating": [4.1, 3.5, 4.8, 2.9, 3.9, 4.0],
            "ratings_count": [100, 0, 10, 5, 1, 20],
            "text_reviews_count": [10, 0, 1, 2, 0, 4],
            "num_pages": [300, None, 250, 180, None, 420],
            "publication_year": [2001, 1999, None, 2010, 2020, None],
            "author_count": [1, 2, 1, 1, 3, 1],
            "genre_fantasy": [1, 1, 0, 0, 0, 1],
            "genre_mystery": [0, 0, 1, 1, 0, 0],
            "genre_history": [0, 0, 0, 0, 1, 0],
            "genre_ya": [0, 1, 0, 0, 0, 1],
            "genre_romance": [0, 0, 0, 1, 0, 0],
            "genre_count": [1, 2, 1, 2, 1, 2],
        }
    )


def test_build_blocks_and_block_weights() -> None:
    master = _sample_master()
    numeric, medians = build_numeric_block(master)
    binary, lang_categories = build_binary_block(master, top_n=3)

    assert numeric["num_pages_missing"].tolist() == [0, 1, 0, 0, 1, 0]
    assert numeric["publication_year_missing"].tolist() == [0, 0, 1, 0, 0, 1]
    assert medians["num_pages"] == 275.0
    assert "en" in lang_categories
    assert "language_code_other" in binary.columns
    assert binary.loc[5, "language_code_other"] == 1.0

    embeddings = pd.DataFrame(
        np.arange(len(master) * 4, dtype=np.float32).reshape(len(master), 4),
        columns=[f"emb_{idx}" for idx in range(4)],
    )
    scaled = standardize_and_weight_blocks(
        {
            "numeric": numeric,
            "binary": binary,
            "embeddings": embeddings,
        }
    )

    assert scaled.block_dims["numeric"] == numeric.shape[1]
    assert scaled.block_weights["embeddings"] == 0.5
    assert math.isclose(
        scaled.block_weights["numeric"],
        1 / math.sqrt(numeric.shape[1]),
    )
    assert scaled.matrix.shape[0] == len(master)


def test_pca_smoke_preserves_row_count() -> None:
    master = _sample_master()
    numeric, _medians = build_numeric_block(master)
    binary, _langs = build_binary_block(master, top_n=2)
    embeddings = pd.DataFrame(
        np.eye(len(master), 4, dtype=np.float32),
        columns=[f"emb_{idx}" for idx in range(4)],
    )

    scaled = standardize_and_weight_blocks(
        {
            "numeric": numeric,
            "binary": binary,
            "embeddings": embeddings,
        }
    )
    reduced, pca = fit_variance_pca(scaled.matrix, variance_threshold=0.95)

    feature_matrix = pd.DataFrame(reduced)
    feature_matrix.insert(0, "book_id", master["book_id"].to_numpy())

    assert feature_matrix["book_id"].tolist() == master["book_id"].tolist()
    assert reduced.shape[0] == len(master)
    assert pca.explained_variance_ratio_.sum() >= 0.95
