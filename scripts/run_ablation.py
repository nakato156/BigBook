"""Ablation harness: sweep RankingConfig variants and record evaluate_temporal's response.

Standalone script (not under src/), invoked by path like
scripts/build_deliverable3_clustering_outputs.py — same sys.path bootstrap, same
`from src.config import ...` style. Pure consumer of the Phase 1/2 evaluation pipeline:
does not modify src/reduction/recommend.py or src/reduction/evaluate_recommender.py.

`run_ablation` is side-effect-free (no disk I/O, no Recommender.from_artifacts() call) so it
is directly testable with a toy Recommender and a small interactions DataFrame. All I/O
(artifact loading, CSV writing) lives in main().
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    BOOKS_MASTER_PATH,
    INTERACTIONS_CURATED_PATH,
    PROJECT_ROOT,
    USER_FEATURES_GLOBAL_PATH,
)
from src.reduction.evaluate_recommender import (
    choose_global_cutoff,
    collect_users,
    collect_valid_user_ids,
    evaluate_temporal,
    historical_catalog_mask,
    historical_popularity_snapshot,
    MIN_VALID_DATE,
)
from src.reduction.recommend import Recommender, RankingConfig

RANDOM_STATE = 42
OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "recommendations" / "ablation_results.csv"


@dataclass(frozen=True)
class AblationConfig:
    """Names one variant of the sweep: a label plus RankingConfig field overrides."""

    label: str
    overrides: dict


# Not a full cartesian product — one parameter at a time from the default, plus 1-2 crossed
# combinations, to keep the sweep readable and the runtime bounded (each entry re-runs
# evaluate_temporal end-to-end).
ABLATION_CONFIGS = [
    AblationConfig("baseline", {}),
    AblationConfig("n_clusters_retrieve=10", {"n_clusters_retrieve": 10}),
    AblationConfig("n_clusters_retrieve=15", {"n_clusters_retrieve": 15}),
    AblationConfig("n_clusters_retrieve=20", {"n_clusters_retrieve": 20}),
    AblationConfig("mmr_lambda=0.5", {"mmr_lambda": 0.5}),
    AblationConfig("mmr_lambda=0.9", {"mmr_lambda": 0.9}),
    AblationConfig("explore_slots=0", {"explore_slots": 0}),
    AblationConfig("explore_slots=4", {"explore_slots": 4}),
    AblationConfig(
        "n_clusters_retrieve=15,explore_slots=0",
        {"n_clusters_retrieve": 15, "explore_slots": 0},
    ),
]


def run_ablation(
    recommender: Recommender,
    interactions: pd.DataFrame,
    configs: list[AblationConfig],
    **evaluate_temporal_kwargs,
) -> pd.DataFrame:
    """Run evaluate_temporal once per config; tag each summary row with config_label + overrides.

    Reuses evaluate_temporal() unmodified — only swaps recommender.config via
    dataclasses.replace() before each run, the same pattern evaluate_temporal already uses
    internally per (user, k). Restores the original config after each run, even on failure,
    so one config's override never leaks into the next sweep entry or back to the caller.
    """
    base_config = recommender.config
    frames = []
    for cfg in configs:
        recommender.config = replace(base_config, **cfg.overrides)
        try:
            summary, _ = evaluate_temporal(interactions, recommender, **evaluate_temporal_kwargs)
        finally:
            recommender.config = base_config
        summary = summary.copy()
        summary["config_label"] = cfg.label
        for param, value in cfg.overrides.items():
            summary[param] = value
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    # Lower than evaluate_recommender.py's default (5,000): this script runs
    # evaluate_temporal once per sweep config (len(ABLATION_CONFIGS) full passes), not once,
    # so the same --max-users would multiply runtime by the number of configs.
    parser.add_argument("--max-users", type=int, default=1_000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--k", type=int, nargs="+", default=[10])
    parser.add_argument("--cutoff", type=str)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    recommender = Recommender.from_artifacts()
    valid_user_ids = collect_valid_user_ids(
        USER_FEATURES_GLOBAL_PATH,
        args.max_users,
        args.random_state,
    )
    interactions = collect_users(INTERACTIONS_CURATED_PATH, valid_user_ids)
    cutoff = (
        pd.Timestamp(args.cutoff, tz="UTC")
        if args.cutoff
        else choose_global_cutoff(interactions, args.train_fraction)
    )
    if cutoff < MIN_VALID_DATE:
        raise ValueError(f"--cutoff must be on or after {MIN_VALID_DATE.date()}.")
    snapshot = historical_popularity_snapshot(
        INTERACTIONS_CURATED_PATH,
        recommender.book_ids,
        cutoff,
    )
    catalog_available = historical_catalog_mask(
        BOOKS_MASTER_PATH,
        recommender.book_ids,
        cutoff,
        snapshot.first_observed,
    )

    results = run_ablation(
        recommender,
        interactions,
        ABLATION_CONFIGS,
        popularity_count=snapshot.rating_count,
        average_rating=snapshot.average_rating,
        train_fraction=args.train_fraction,
        ks=args.k,
        temporal_cutoff=cutoff,
        catalog_available=catalog_available,
        invalid_date_count=snapshot.invalid_date_count,
        users_selected=len(valid_user_ids),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    print(f"Temporal cutoff: {cutoff}")
    print(f"Users selected: {len(valid_user_ids):,}")
    model_rows = results.loc[results["system"] == "model"]
    summary_cols = ["config_label", "k", "recall", "ndcg", "candidate_recall"]
    print(model_rows[summary_cols].to_string(index=False))
    print(f"Wrote ablation results to {args.output}")


if __name__ == "__main__":
    main()
