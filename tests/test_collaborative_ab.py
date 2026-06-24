from __future__ import annotations

import pandas as pd

from scripts.run_collaborative_ab import select_collaborative_winner


def _row(label: str, system: str, recall: float, ndcg: float, coverage: float, novelty: float):
    return {
        "config_label": label,
        "system": system,
        "k": 10,
        "recall": recall,
        "ndcg": ndcg,
        "catalog_coverage": coverage,
        "novelty": novelty,
        "candidate_recall": 0.5,
    }


def test_select_collaborative_winner_requires_relevance_gain_and_b1_discovery_floor() -> None:
    results = pd.DataFrame(
        [
            _row("content_only", "model", 0.10, 0.10, 0.20, 8.0),
            _row("content_only", "B1_popularity", 0.20, 0.20, 0.01, 5.0),
            _row("cooccurrence_alpha=0.7", "model", 0.22, 0.23, 0.18, 8.1),
            _row("user_knn_alpha=0.7", "model", 0.24, 0.25, 0.005, 8.5),
        ]
    )
    assert select_collaborative_winner(results) == "cooccurrence_alpha=0.7"


def test_select_collaborative_winner_falls_back_to_content_only() -> None:
    results = pd.DataFrame(
        [
            _row("content_only", "model", 0.10, 0.10, 0.20, 8.0),
            _row("content_only", "B1_popularity", 0.20, 0.20, 0.01, 5.0),
            _row("cooccurrence_alpha=0.7", "model", 0.21, 0.19, 0.20, 8.1),
        ]
    )
    assert select_collaborative_winner(results) == "content_only"
