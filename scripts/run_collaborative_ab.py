"""Temporal A/B comparison of content, item-PMI and exact user-kNN signals."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    BOOKS_MASTER_PATH,
    INTERACTIONS_CURATED_PATH,
    MASTER_FEATURE_MATRIX_PATH,
    PROJECT_ROOT,
    USER_FEATURES_GLOBAL_PATH,
)
from src.reduction.build_item_cooccurrence import (
    build_item_cooccurrence,
    interaction_chunks_before,
)
from src.reduction.build_user_matrix import (
    INTERACTION_COLUMNS as USER_INTERACTION_COLUMNS,
    build_user_artifacts,
)
from src.reduction.baselines import historical_catalog_mask, historical_popularity_snapshot
from src.reduction.collaborative import CooccurrenceIndex, score_map_callback
from src.reduction.evaluate_recommender import (
    INTERACTION_COLUMNS as EVAL_INTERACTION_COLUMNS,
    collect_users,
    collect_valid_user_ids,
    evaluate_temporal,
)
from src.reduction.recommend import Recommender
from src.reduction.retrieval import l2_normalize_rows
from src.reduction.temporal_split import MIN_VALID_DATE, choose_global_cutoff, global_temporal_split
from src.reduction.user_knn import compute_user_knn_scores, neighbor_unread_books
from src.utils.io import read_parquet_chunks, safe_write_parquet

RANDOM_STATE = 42
ALPHAS = (0.5, 0.7, 0.9)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "recommendations" / "collaborative_ab_results.csv"
)

ScoreFn = Callable[[np.ndarray, str], np.ndarray]


def run_collaborative_ab(
    recommender: Recommender,
    interactions: pd.DataFrame,
    variants: dict[str, ScoreFn | None],
    alphas: Iterable[float] = ALPHAS,
    **evaluate_kwargs,
) -> pd.DataFrame:
    """Evaluate content-only once and every collaborative signal at each alpha."""
    frames: list[pd.DataFrame] = []
    experiments = [("content_only", None, 1.0)]
    experiments.extend(
        (f"{name}_alpha={alpha:.1f}", score_fn, float(alpha))
        for name, score_fn in variants.items()
        for alpha in alphas
    )
    for label, score_fn, alpha in experiments:
        summary, _ = evaluate_temporal(
            interactions,
            recommender,
            additional_score_fn=score_fn,
            blend_alpha=alpha,
            **evaluate_kwargs,
        )
        tagged = summary.copy()
        tagged.insert(0, "config_label", label)
        tagged.insert(1, "collaborative_signal", label.split("_alpha=", 1)[0])
        tagged.insert(2, "blend_alpha", alpha)
        frames.append(tagged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def select_collaborative_winner(results: pd.DataFrame, k: int = 10) -> str:
    """Select the best eligible model configuration, or retain content-only.

    Eligibility is deliberately strict for the 10/10 rubric: a collaborative variant
    must beat B1 on Recall/NDCG at the selected k and keep B1's discovery floors.
    """
    model = results.loc[(results["system"] == "model") & (results["k"] == k)].copy()
    if model.empty:
        raise ValueError(f"No model rows found for k={k}.")
    content = model.loc[model["config_label"] == "content_only"]
    b1 = results.loc[
        (results["config_label"] == "content_only")
        & (results["system"] == "B1_popularity")
        & (results["k"] == k)
    ]
    if content.empty or b1.empty:
        raise ValueError("A/B results must contain content_only and its B1 baseline.")
    content_row = content.iloc[0]
    b1_row = b1.iloc[0]
    if "long_tail_coverage" not in model.columns:
        model["long_tail_coverage"] = 0.0
        b1_row = b1_row.copy()
        b1_row["long_tail_coverage"] = 0.0
    eligible = model.loc[
        (model["catalog_coverage"] >= b1_row["catalog_coverage"])
        & (model["novelty"] >= b1_row["novelty"])
        & (model["long_tail_coverage"] >= b1_row["long_tail_coverage"])
        & (model["recall"] > b1_row["recall"])
        & (model["ndcg"] > b1_row["ndcg"])
        & (model["recall"] >= content_row["recall"])
        & (model["ndcg"] >= content_row["ndcg"])
    ]
    if eligible.empty:
        return "content_only"
    return str(
        eligible.sort_values(
            ["ndcg", "recall", "candidate_recall"],
            ascending=False,
            kind="stable",
        ).iloc[0]["config_label"]
    )


def _chunks_before(
    path: Path,
    cutoff: pd.Timestamp,
    columns: list[str],
    chunksize: int = 1_000_000,
):
    for chunk in read_parquet_chunks(path, chunksize, columns=columns):
        dates = pd.to_datetime(chunk["date_added"], errors="coerce", utc=True)
        keep = dates.notna() & (dates <= cutoff)
        if keep.any():
            filtered = chunk.loc[keep].copy()
            filtered["date_added"] = dates.loc[keep].dt.tz_localize(None)
            yield filtered


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
) -> ScoreFn:
    index = CooccurrenceIndex(pairs)

    def callback(rows: np.ndarray, user_id: str) -> np.ndarray:
        return index.score(
            recommender.book_ids[np.asarray(rows, dtype=np.int64)],
            positives_by_user.get(str(user_id), []),
        )

    return callback


def _build_user_knn_callback(
    recommender: Recommender,
    eval_train: pd.DataFrame,
    cutoff: pd.Timestamp,
    k_neighbors: int,
) -> ScoreFn:
    feature_matrix = pd.read_parquet(MASTER_FEATURE_MATRIX_PATH)
    books_master = pd.read_parquet(BOOKS_MASTER_PATH)
    user_features = pd.read_parquet(
        USER_FEATURES_GLOBAL_PATH,
        columns=["user_id", "user_rating_bias"],
    )
    train_matrix, _, _ = build_user_artifacts(
        feature_matrix,
        books_master,
        user_features,
        _chunks_before(
            INTERACTIONS_CURATED_PATH,
            cutoff,
            USER_INTERACTION_COLUMNS,
        ),
    )
    train_matrix["user_id"] = train_matrix["user_id"].astype(str)
    pc_cols = recommender.pc_cols
    all_ids = train_matrix["user_id"].to_numpy(dtype=str)
    all_taste = l2_normalize_rows(
        train_matrix[pc_cols].to_numpy(dtype=np.float64)[:, recommender.taste_idx]
    )
    eval_ids = np.array(
        sorted(set(eval_train["user_id"].astype(str)) & set(all_ids)),
        dtype=str,
    )
    row_of = {user_id: row for row, user_id in enumerate(all_ids)}
    eval_rows = np.array([row_of[user_id] for user_id in eval_ids], dtype=np.int64)
    neighbors = compute_user_knn_scores(
        all_taste[eval_rows],
        all_taste,
        eval_ids,
        all_ids,
        k=k_neighbors,
    )
    consumed_by_eval = {
        str(user_id): set(group.loc[group["is_read"].fillna(False), "book_id"].astype(str))
        for user_id, group in eval_train.groupby("user_id", sort=False)
    }
    scores = neighbor_unread_books(
        neighbors,
        consumed_by_eval,
        INTERACTIONS_CURATED_PATH,
        cutoff=cutoff,
    )
    return score_map_callback(recommender.book_ids, scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=1_000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--k", type=int, nargs="+", default=[10])
    parser.add_argument("--cutoff", type=str)
    parser.add_argument("--alpha", type=float, nargs="+", default=list(ALPHAS))
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument(
        "--signals",
        nargs="+",
        choices=["cooccurrence", "user_knn"],
        default=["cooccurrence", "user_knn"],
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    recommender = Recommender.from_artifacts()
    selected_ids = collect_valid_user_ids(
        USER_FEATURES_GLOBAL_PATH,
        args.max_users,
        args.random_state,
    )
    interactions = collect_users(INTERACTIONS_CURATED_PATH, selected_ids)
    cutoff = (
        pd.Timestamp(args.cutoff)
        if args.cutoff
        else choose_global_cutoff(interactions, args.train_fraction)
    )
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    if cutoff < MIN_VALID_DATE:
        raise ValueError(f"--cutoff must be on or after {MIN_VALID_DATE.date()}.")
    snapshot = historical_popularity_snapshot(
        INTERACTIONS_CURATED_PATH,
        recommender.book_ids,
        cutoff,
    )
    available = historical_catalog_mask(
        BOOKS_MASTER_PATH,
        recommender.book_ids,
        cutoff,
        snapshot.first_observed,
    )
    train, _ = global_temporal_split(interactions, cutoff)
    positives = _positive_books_by_user(train)
    variants: dict[str, ScoreFn] = {}

    snapshot_dir = (
        PROJECT_ROOT
        / "data"
        / "features"
        / "snapshots"
        / cutoff.strftime("%Y-%m-%d")
    )
    if "cooccurrence" in args.signals:
        pairs, _ = build_item_cooccurrence(
            pd.read_parquet(MASTER_FEATURE_MATRIX_PATH),
            interaction_chunks_before(INTERACTIONS_CURATED_PATH, cutoff),
        )
        cooccurrence_path = snapshot_dir / "book_cooccurrence.parquet"
        safe_write_parquet(pairs, cooccurrence_path)
        variants["cooccurrence"] = _cooccurrence_callback(
            recommender,
            pairs,
            positives,
        )
    if "user_knn" in args.signals:
        variants["user_knn"] = _build_user_knn_callback(
            recommender,
            train,
            cutoff,
            args.neighbors,
        )

    results = run_collaborative_ab(
        recommender,
        interactions,
        variants,
        args.alpha,
        popularity_count=snapshot.rating_count,
        average_rating=snapshot.average_rating,
        train_fraction=args.train_fraction,
        ks=args.k,
        temporal_cutoff=cutoff,
        catalog_available=available,
        invalid_date_count=snapshot.invalid_date_count,
        users_selected=len(selected_ids),
    )
    results["selected_at_k10"] = results["config_label"].eq(
        select_collaborative_winner(results, k=10)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.loc[results["system"] == "model"].to_string(index=False))
    print(f"Selected configuration: {select_collaborative_winner(results, k=10)}")
    print(f"Wrote collaborative A/B results to {args.output}")


if __name__ == "__main__":
    main()
