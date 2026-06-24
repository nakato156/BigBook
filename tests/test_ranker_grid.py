from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_ranker_grid import strict_b1_winner


def _row(label: str, system: str, k: int, recall: float, ndcg: float):
    return {
        "config_label": label,
        "system": system,
        "k": k,
        "recall": recall,
        "ndcg": ndcg,
        "catalog_coverage": 0.20 if system != "B1_popularity" else 0.01,
        "novelty": 8.0 if system != "B1_popularity" else 5.0,
        "long_tail_coverage": 0.10 if system != "B1_popularity" else 0.0,
    }


def test_strict_b1_winner_requires_all_k_recall_and_ndcg_wins() -> None:
    rows = []
    for k in [5, 10, 20]:
        rows.append(_row("content_only", "content_only", k, 0.05, 0.05))
        rows.append(_row("content_only", "B1_popularity", k, 0.10, 0.10))
        rows.append(_row("hybrid_bad", "hybrid_v12", k, 0.20, 0.20 if k != 20 else 0.09))
        rows.append(_row("hybrid_good", "hybrid_v12", k, 0.15, 0.15))

    assert strict_b1_winner(pd.DataFrame(rows)) == "hybrid_good"


def test_strict_b1_winner_falls_back_to_content_only() -> None:
    rows = []
    for k in [5, 10, 20]:
        rows.append(_row("content_only", "content_only", k, 0.05, 0.05))
        rows.append(_row("content_only", "B1_popularity", k, 0.10, 0.10))
        rows.append(_row("hybrid_bad", "hybrid_v12", k, 0.09, 0.20))

    assert strict_b1_winner(pd.DataFrame(rows)) == "content_only"


def test_strict_b1_winner_rejects_missing_model_rows() -> None:
    with np.testing.assert_raises(ValueError):
        strict_b1_winner(pd.DataFrame([_row("x", "B1_popularity", 10, 0.1, 0.1)]))
