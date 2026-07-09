"""Compare the book co-read graph against existing baselines.

Two comparison legs, both reusing existing artifacts/utilities rather than re-running
evaluation:

- ``compare_to_popularity``: graph centrality (PageRank, weighted degree) vs. B1 historical
  popularity (``src.reduction.baselines.historical_popularity_snapshot``), via rank
  correlation and top-k set overlap.
- ``compare_to_collaborative_ab``: summarizes the existing
  ``collaborative_ab_results.csv`` (produced by ``scripts/run_collaborative_ab.py``), which
  already evaluates the same PMI cooccurrence signal as a recommender score against B1 and
  content-only ranking.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import INTERACTIONS_CURATED_PATH
from src.reduction.baselines import historical_popularity_snapshot


def compare_to_popularity(
    nodes: pd.DataFrame,
    k: int = 100,
    interactions_path: Path = INTERACTIONS_CURATED_PATH,
) -> dict[str, float]:
    """Rank correlation + top-k overlap between graph centrality and B1 popularity."""
    book_ids = nodes["book_id"].astype(str).to_numpy()
    snapshot = historical_popularity_snapshot(
        interactions_path,
        book_ids,
        cutoff=pd.Timestamp.now(tz="UTC"),
    )
    popularity_count = snapshot.rating_count

    # Restrict to non-isolated nodes: an isolated node's pagerank/degree is a structural
    # artifact of having zero qualifying edges, not a meaningful rank for correlation.
    connected = nodes["degree"].to_numpy() > 0
    pagerank = nodes["pagerank"].to_numpy()
    weighted_degree = nodes["weighted_degree"].to_numpy()

    def _spearman(graph_metric: np.ndarray) -> float:
        if connected.sum() < 2:
            return float("nan")
        corr, _ = spearmanr(graph_metric[connected], popularity_count[connected])
        return float(corr)

    def _overlap(graph_metric: np.ndarray) -> float:
        if k <= 0 or len(book_ids) == 0:
            return float("nan")
        top_graph = set(book_ids[np.argsort(-graph_metric)[:k]])
        top_pop = set(book_ids[np.argsort(-popularity_count)[:k]])
        return len(top_graph & top_pop) / k

    return {
        "pagerank_vs_popularity_spearman": _spearman(pagerank),
        "weighted_degree_vs_popularity_spearman": _spearman(weighted_degree),
        "pagerank_top_k_overlap": _overlap(pagerank),
        "weighted_degree_top_k_overlap": _overlap(weighted_degree),
        "k": float(k),
    }


def compare_to_collaborative_ab(results_path: Path) -> str | None:
    """Summarize the existing cooccurrence-vs-baseline ablation, if it has been run."""
    if not results_path.exists():
        return None
    results = pd.read_csv(results_path)
    if "collaborative_signal" not in results.columns or "system" not in results.columns:
        return None
    cooccurrence_rows = results[results["collaborative_signal"] == "cooccurrence"]
    if cooccurrence_rows.empty:
        return None
    columns = [c for c in ("config_label", "system", "k", "recall", "ndcg") if c in cooccurrence_rows.columns]
    return cooccurrence_rows[columns].to_string(index=False)


def compare_to_hybrid_ranker_grid(results_path: Path) -> str | None:
    """Summarize grid rows where the production ranker uses cooccurrence evidence."""
    if not results_path.exists():
        return None
    results = pd.read_csv(results_path)
    required = {"config_label", "weights_json", "system", "k", "recall", "ndcg"}
    if not required.issubset(results.columns):
        return None

    def _cooccurrence_weight(raw: object) -> float:
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            return 0.0
        return float(parsed.get("cooccurrence", 0.0) or 0.0)

    co_weight = results["weights_json"].map(_cooccurrence_weight)
    model_rows = results.loc[(co_weight > 0.0) & (results["system"] == "hybrid_v12")].copy()
    if model_rows.empty:
        return None
    k_values = sorted(model_rows["k"].unique())
    baseline_mask = results["config_label"].eq("content_only") & results["system"].isin(
        ["content_only", "B1_popularity"]
    )
    model_mask = results.index.isin(model_rows.index)
    comparison = results.loc[(baseline_mask | model_mask) & results["k"].isin(k_values)].copy()
    comparison["cooccurrence_weight"] = co_weight.loc[comparison.index].to_numpy(dtype=float)
    columns = [
        c
        for c in (
            "config_label",
            "system",
            "cooccurrence_weight",
            "k",
            "recall",
            "ndcg",
            "candidate_recall",
        )
        if c in comparison.columns
    ]
    return comparison[columns].to_string(index=False)
