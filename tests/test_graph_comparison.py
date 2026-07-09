from __future__ import annotations

import pandas as pd

from src.reduction.graph_comparison import compare_to_hybrid_ranker_grid


def test_compare_to_hybrid_ranker_grid_keeps_cooccurrence_rows_and_baselines(tmp_path) -> None:
    path = tmp_path / "ranker_grid_results.csv"
    pd.DataFrame(
        [
            {
                "config_label": "content_only",
                "weights_json": "{}",
                "system": "content_only",
                "k": 10,
                "recall": 0.01,
                "ndcg": 0.02,
                "candidate_recall": 0.30,
            },
            {
                "config_label": "content_only",
                "weights_json": "{}",
                "system": "B1_popularity",
                "k": 10,
                "recall": 0.03,
                "ndcg": 0.04,
                "candidate_recall": None,
            },
            {
                "config_label": "hybrid_v12_grid_1",
                "weights_json": '{"content": 0.35, "cooccurrence": 0.10}',
                "system": "hybrid_v12",
                "k": 10,
                "recall": 0.02,
                "ndcg": 0.03,
                "candidate_recall": 0.45,
            },
        ]
    ).to_csv(path, index=False)

    summary = compare_to_hybrid_ranker_grid(path)

    assert summary is not None
    assert "content_only" in summary
    assert "B1_popularity" in summary
    assert "hybrid_v12_grid_1" in summary
    assert "0.1" in summary
