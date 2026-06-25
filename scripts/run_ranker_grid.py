"""Reproducible V1.2 ranker grid over historical temporal evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
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
from src.reduction.baselines import historical_catalog_mask, historical_popularity_snapshot
from src.reduction.collaborative import CooccurrenceIndex, score_map_callback
from src.reduction.evaluate_recommender import collect_users, collect_valid_user_ids, evaluate_temporal
from src.reduction.metrics import bootstrap_confidence_intervals
from src.reduction.ranking import HybridV12Weights
from src.reduction.recommend import Recommender
from src.reduction.temporal_split import MIN_VALID_DATE, choose_global_cutoff, global_temporal_split
from src.utils.io import safe_write_parquet

RANDOM_STATE = 42
OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "recommendations" / "ranker_grid_results.csv"
USERS_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "recommendations" / "temporal_evaluation_v12_users.parquet"
)
SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "recommendations" / "temporal_evaluation_v12.csv"
)
BOOTSTRAP_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "recommendations" / "temporal_evaluation_v12_bootstrap_ci.csv"
)
WINNER_PATH = PROJECT_ROOT / "data" / "outputs" / "recommendations" / "ranker_grid_winner.json"

WEIGHT_GRID = (
    HybridV12Weights(content=0.35, global_popularity=0.35, genre_popularity=0.20, cooccurrence=0.10, user_knn=0.00),
    HybridV12Weights(content=0.25, global_popularity=0.45, genre_popularity=0.20, cooccurrence=0.10, user_knn=0.00),
    HybridV12Weights(content=0.20, global_popularity=0.50, genre_popularity=0.20, cooccurrence=0.10, user_knn=0.00),
    HybridV12Weights(content=0.15, global_popularity=0.55, genre_popularity=0.20, cooccurrence=0.10, user_knn=0.00),
)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [ranker_grid] {message}", flush=True)


def _positive_books_by_user(train: pd.DataFrame) -> dict[str, list[str]]:
    rating = pd.to_numeric(train["rating_clean"], errors="coerce")
    keep = train["is_read"].fillna(False) & rating.ge(4.0)
    return (
        train.loc[keep]
        .groupby("user_id", sort=False)["book_id"]
        .agg(lambda values: list(dict.fromkeys(values.astype(str))))
        .to_dict()
    )


def _cooccurrence_callback(
    recommender: Recommender,
    pairs: pd.DataFrame,
    positives_by_user: dict[str, list[str]],
):
    index = CooccurrenceIndex(pairs)

    def callback(rows: np.ndarray, user_id: str) -> np.ndarray:
        return index.score(
            recommender.book_ids[np.asarray(rows, dtype=np.int64)],
            positives_by_user.get(str(user_id), []),
        )

    return callback


def _top_rows_by_score_callback(
    recommender: Recommender,
    score_fn,
    user_ids: list[str],
    candidate_rows: np.ndarray,
    limit: int,
):
    top_by_user: dict[str, np.ndarray] = {}
    for user_id in user_ids:
        scores = np.asarray(score_fn(candidate_rows, user_id), dtype=np.float64)
        if not len(scores) or not np.isfinite(scores).all() or np.all(scores <= 0):
            top_by_user[str(user_id)] = np.array([], dtype=np.int64)
            continue
        ids = recommender.book_ids[candidate_rows].astype(str)
        order = np.lexsort((ids, -scores))
        top_by_user[str(user_id)] = candidate_rows[order[:limit]].astype(np.int64)

    def callback(user_id: str) -> np.ndarray:
        return top_by_user.get(str(user_id), np.array([], dtype=np.int64))

    return callback


def strict_b1_winner(results: pd.DataFrame) -> str:
    """Return the winning config label, or content_only when no config beats B1 at every k."""
    model = results.loc[results["system"].isin(["content_only", "hybrid_v12"])].copy()
    b1 = results.loc[results["system"] == "B1_popularity"].copy()
    if model.empty or b1.empty:
        raise ValueError("Grid results must include model rows and B1_popularity rows.")
    b1_by_k = b1.groupby("k")[["recall", "ndcg", "catalog_coverage", "novelty", "long_tail_coverage"]].max()
    eligible_labels: list[str] = []
    for label, group in model.groupby("config_label", sort=False):
        if label == "content_only":
            continue
        aligned = group.set_index("k").reindex(b1_by_k.index)
        if aligned[["recall", "ndcg"]].isna().any().any():
            continue
        wins = (
            (aligned["recall"] > b1_by_k["recall"])
            & (aligned["ndcg"] > b1_by_k["ndcg"])
            & (aligned["catalog_coverage"] >= b1_by_k["catalog_coverage"])
            & (aligned["novelty"] >= b1_by_k["novelty"])
            & (aligned["long_tail_coverage"] >= b1_by_k["long_tail_coverage"])
        )
        if bool(wins.all()):
            eligible_labels.append(str(label))
    if not eligible_labels:
        return "content_only"
    candidates = model.loc[(model["config_label"].isin(eligible_labels)) & (model["k"] == 10)]
    return str(
        candidates.sort_values(
            ["ndcg", "recall", "catalog_coverage", "novelty"],
            ascending=False,
            kind="stable",
        ).iloc[0]["config_label"]
    )


def run_grid(
    recommender: Recommender,
    interactions: pd.DataFrame,
    cutoff: pd.Timestamp,
    available: np.ndarray,
    rating_count: np.ndarray,
    average_rating: np.ndarray,
    weights_grid: tuple[HybridV12Weights, ...] = WEIGHT_GRID,
    ks: tuple[int, ...] = (5, 10, 20),
    source_limit: int = 250,
    cooccurrence_pairs: pd.DataFrame | None = None,
    users_selected: int | None = None,
    progress_every_users: int = 250,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    log("Building train split and positive histories for hybrid signals.")
    train, _ = global_temporal_split(interactions, cutoff)
    positives = _positive_books_by_user(train)
    log(
        f"Train rows={len(train):,}; users with positive histories={len(positives):,}."
    )
    cooccurrence_score_fn = (
        _cooccurrence_callback(recommender, cooccurrence_pairs, positives)
        if cooccurrence_pairs is not None and not cooccurrence_pairs.empty
        else None
    )
    eligible_rows = np.nonzero(np.asarray(available, dtype=bool) & recommender.eligible_mask)[0]
    log(f"Evaluation-eligible rows={len(eligible_rows):,}; source_limit={source_limit:,}.")
    extra_rows_fn = (
        _top_rows_by_score_callback(
            recommender,
            cooccurrence_score_fn,
            sorted(train["user_id"].astype(str).unique().tolist()),
            eligible_rows,
            source_limit,
        )
        if cooccurrence_score_fn is not None
        else None
    )

    frames: list[pd.DataFrame] = []
    per_user_by_label: dict[str, pd.DataFrame] = {}
    log("Evaluating content_only baseline inside the grid.")
    content_summary, content_users = evaluate_temporal(
        interactions,
        recommender,
        popularity_count=rating_count,
        average_rating=average_rating,
        ks=ks,
        temporal_cutoff=cutoff,
        catalog_available=available,
        users_selected=users_selected,
        model_label="content_only",
        progress_every_users=progress_every_users,
        progress_label="content_only",
    )
    content_summary.insert(0, "config_label", "content_only")
    content_summary.insert(1, "weights_json", "{}")
    frames.append(content_summary)
    per_user_by_label["content_only"] = content_users

    for idx, weights in enumerate(weights_grid, start=1):
        label = f"hybrid_v12_grid_{idx}"
        log(f"Evaluating {label} with weights={json.dumps(asdict(weights), sort_keys=True)}.")
        summary, per_user = evaluate_temporal(
            interactions,
            recommender,
            popularity_count=rating_count,
            average_rating=average_rating,
            ks=ks,
            temporal_cutoff=cutoff,
            catalog_available=available,
            users_selected=users_selected,
            model_label="hybrid_v12",
            hybrid_v12_weights=weights,
            cooccurrence_score_fn=cooccurrence_score_fn,
            hybrid_extra_candidate_rows_fn=extra_rows_fn,
            hybrid_source_limit=source_limit,
            progress_every_users=progress_every_users,
            progress_label=label,
        )
        summary.insert(0, "config_label", label)
        summary.insert(1, "weights_json", json.dumps(asdict(weights), sort_keys=True))
        frames.append(summary)
        per_user_by_label[label] = per_user
        k10 = summary.loc[
            (summary["system"] == "hybrid_v12") & (summary["k"] == 10),
            ["recall", "ndcg", "candidate_recall"],
        ]
        if not k10.empty:
            row = k10.iloc[0]
            log(
                f"{label} k=10 recall={row['recall']:.6f}, "
                f"ndcg={row['ndcg']:.6f}, candidate_recall={row['candidate_recall']:.6f}."
            )
    return pd.concat(frames, ignore_index=True), per_user_by_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=5_000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--cutoff", type=str)
    parser.add_argument("--source-limit", type=int, default=250)
    parser.add_argument("--cooccurrence-path", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--winner-output", type=Path, default=WINNER_PATH)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT_PATH)
    parser.add_argument("--users-output", type=Path, default=USERS_OUTPUT_PATH)
    parser.add_argument("--bootstrap-output", type=Path, default=BOOTSTRAP_OUTPUT_PATH)
    parser.add_argument("--bootstrap-resamples", type=int, default=1_000)
    parser.add_argument("--collect-progress-batches", type=int, default=20)
    parser.add_argument("--snapshot-progress-batches", type=int, default=10)
    parser.add_argument("--progress-every-users", type=int, default=250)
    args = parser.parse_args()

    log(
        f"Starting V1.2 grid: max_users={args.max_users:,}, k={args.k}, "
        f"bootstrap_resamples={args.bootstrap_resamples:,}."
    )
    log("Loading recommender artifacts.")
    recommender = Recommender.from_artifacts()
    log(f"Loaded recommender with {len(recommender.book_ids):,} books.")
    log("Selecting valid user cohort.")
    selected_ids = collect_valid_user_ids(
        USER_FEATURES_GLOBAL_PATH,
        args.max_users,
        args.random_state,
    )
    log(f"Selected {len(selected_ids):,} valid users; collecting complete histories.")
    interactions = collect_users(
        INTERACTIONS_CURATED_PATH,
        selected_ids,
        progress_every_batches=args.collect_progress_batches,
    )
    log(f"Collected {len(interactions):,} interaction rows for selected users.")
    cutoff = (
        pd.Timestamp(args.cutoff)
        if args.cutoff
        else choose_global_cutoff(interactions, args.train_fraction)
    )
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    if cutoff < MIN_VALID_DATE:
        raise ValueError(f"--cutoff must be on or after {MIN_VALID_DATE.date()}.")
    log(f"Temporal cutoff resolved to {cutoff.isoformat()}.")
    log("Building historical popularity snapshot from canonical interactions.")
    snapshot = historical_popularity_snapshot(
        INTERACTIONS_CURATED_PATH,
        recommender.book_ids,
        cutoff,
        progress_every_batches=args.snapshot_progress_batches,
    )
    log(
        "Historical snapshot ready: "
        f"rated_count_total={int(snapshot.rating_count.sum()):,}, "
        f"invalid_dates={snapshot.invalid_date_count:,}."
    )
    log("Computing historical catalog availability mask.")
    available = historical_catalog_mask(
        BOOKS_MASTER_PATH,
        recommender.book_ids,
        cutoff,
        snapshot.first_observed,
    )
    cooccurrence_pairs = (
        pd.read_parquet(args.cooccurrence_path)
        if args.cooccurrence_path is not None and args.cooccurrence_path.exists()
        else None
    )
    if cooccurrence_pairs is not None:
        log(f"Loaded cooccurrence pairs: {len(cooccurrence_pairs):,}.")
    else:
        log("No cooccurrence parquet provided; cooccurrence signal is neutral.")
    results, per_user_by_label = run_grid(
        recommender,
        interactions,
        cutoff,
        available,
        snapshot.rating_count,
        snapshot.average_rating,
        ks=tuple(args.k),
        source_limit=args.source_limit,
        cooccurrence_pairs=cooccurrence_pairs,
        users_selected=len(selected_ids),
        progress_every_users=args.progress_every_users,
    )
    log("Selecting strict B1 winner.")
    winner = strict_b1_winner(results)
    results["selected_winner"] = results["config_label"].eq(winner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing full grid results to {args.output}.")
    results.to_csv(args.output, index=False)

    winner_summary = results.loc[results["config_label"] == winner].copy()
    winner_users = per_user_by_label[winner].copy()
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing winner summary to {args.summary_output}.")
    winner_summary.to_csv(args.summary_output, index=False)
    log(f"Writing winner per-user rows to {args.users_output}.")
    safe_write_parquet(winner_users, args.users_output)
    log("Computing bootstrap confidence intervals for winner.")
    bootstrap = bootstrap_confidence_intervals(
        winner_users,
        n_resamples=args.bootstrap_resamples,
        random_state=args.random_state,
    )
    bootstrap.to_csv(args.bootstrap_output, index=False)
    payload = {
        "winner": winner,
        "validated_against_b1": bool(winner != "content_only"),
        "cutoff": cutoff.isoformat(),
        "max_users": int(args.max_users),
        "source_limit": int(args.source_limit),
    }
    args.winner_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"Winner payload written to {args.winner_output}.")
    print(results.loc[results["system"].isin(["content_only", "hybrid_v12"])].to_string(index=False))
    print(f"Selected winner: {winner}")
    print(f"Wrote grid results to {args.output}")


if __name__ == "__main__":
    main()
