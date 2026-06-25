"""Temporal offline evaluation driver for the recommender and B0/B1/B2 baselines.

This is the orchestrator: it wires together chronological splitting
(:mod:`src.reduction.temporal_split`), historical popularity/baselines
(:mod:`src.reduction.baselines`), N0 ranking metrics (:mod:`src.reduction.metrics`) and
N1 habit-proxy metrics (:mod:`src.reduction.habit_proxies`) into one evaluation run, plus
the user-sampling helpers and CLI used to invoke it.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    BOOKS_MASTER_PATH,
    INTERACTIONS_CURATED_PATH,
    PROJECT_ROOT,
    USER_FEATURES_GLOBAL_PATH,
)
from src.reduction.baselines import (
    baseline_recommendations,
    historical_catalog_mask,
    historical_popularity_snapshot,
    popularity_from_training,
    prepare_baseline_rankings,
)
from src.reduction.build_user_centroids import compute_engagement_weight
from src.reduction.habit_proxies import _consumed_books, _positive_mask, build_habit_proxy_table, summarize_by_activity
from src.reduction.metrics import (
    _binary_metrics,
    _candidate_recall,
    _intra_list_diversity,
    _slot_metrics,
    bootstrap_confidence_intervals,
)
from src.reduction.ranking import HybridV12Weights
from src.reduction.recommend import GENRE_COLUMNS, Recommender
from src.reduction.retrieval import popularity_segments
from src.reduction.temporal_split import (
    MIN_VALID_DATE,
    RANDOM_STATE,
    _utc_timestamp,
    choose_global_cutoff,
    global_temporal_split,
    temporal_split,
)
from src.utils.io import safe_write_parquet

EVALUATION_MODE = "global_historical_snapshot_frozen_representation"
EVALUATION_DIR = PROJECT_ROOT / "data" / "outputs" / "recommendations"
EVALUATION_OUTPUT = EVALUATION_DIR / "temporal_evaluation.csv"
EVALUATION_USERS_OUTPUT = EVALUATION_DIR / "temporal_evaluation_users.parquet"
EVALUATION_ACTIVITY_OUTPUT = EVALUATION_DIR / "temporal_evaluation_by_activity.csv"
EVALUATION_BOOTSTRAP_OUTPUT = EVALUATION_DIR / "temporal_evaluation_bootstrap_ci.csv"
INTERACTION_COLUMNS = [
    "user_id",
    "book_id",
    "is_read",
    "rating_clean",
    "has_review_text",
    "reading_duration_days",
    "date_added",
]


def evaluate_temporal(
    interactions: pd.DataFrame,
    recommender: Recommender,
    popularity_count: np.ndarray | None = None,
    average_rating: np.ndarray | None = None,
    train_fraction: float = 0.8,
    ks: Sequence[int] | None = None,
    temporal_cutoff: pd.Timestamp | None = None,
    catalog_available: np.ndarray | None = None,
    invalid_date_count: int = 0,
    evaluation_mode: str | None = None,
    users_selected: int | None = None,
    additional_score_fn: Callable[[np.ndarray, str], np.ndarray] | None = None,
    blend_alpha: float = 1.0,
    model_label: str = "model",
    hybrid_v12_weights: HybridV12Weights | dict[str, float] | None = None,
    cooccurrence_score_fn: Callable[[np.ndarray, str], np.ndarray] | None = None,
    user_knn_score_fn: Callable[[np.ndarray, str], np.ndarray] | None = None,
    hybrid_extra_candidate_rows_fn: Callable[[str], np.ndarray] | None = None,
    hybrid_source_limit: int = 250,
    progress_every_users: int = 0,
    progress_label: str = "evaluate_temporal",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate with fold-local or historical popularity, never full-catalog future aggregates."""
    cutoffs = sorted({int(k) for k in (ks or (recommender.config.k,)) if int(k) > 0})
    if not cutoffs:
        raise ValueError("At least one positive k is required.")

    if temporal_cutoff is None:
        train, future = temporal_split(interactions, train_fraction)
    else:
        train, future = global_temporal_split(interactions, temporal_cutoff)
    if popularity_count is None or average_rating is None:
        popularity_count, average_rating = popularity_from_training(train, recommender.book_ids)
    popularity_count = np.asarray(popularity_count, dtype=np.float64)
    average_rating = np.asarray(average_rating, dtype=np.float64)
    if len(popularity_count) != len(recommender.book_ids) or len(average_rating) != len(
        recommender.book_ids
    ):
        raise ValueError("Historical popularity arrays must align 1:1 with recommender.book_ids.")
    evaluation_eligible_mask = recommender.eligible_mask.copy()
    if catalog_available is not None:
        available = np.asarray(catalog_available, dtype=bool)
        if len(available) != len(recommender.book_ids):
            raise ValueError("catalog_available must align 1:1 with recommender.book_ids.")
        evaluation_eligible_mask &= available
    available_book_ids = set(recommender.book_ids[evaluation_eligible_mask])

    train_groups = {uid: group for uid, group in train.groupby("user_id", sort=False)}
    future_groups = {uid: group for uid, group in future.groupby("user_id", sort=False)}
    genre_flags = recommender.genres.reindex(recommender.book_ids)[GENRE_COLUMNS].fillna(0).to_numpy()
    habit_proxies = build_habit_proxy_table(
        train,
        future,
        recommender.genres,
        temporal_cutoff,
    ).set_index("user_id")

    rows: list[dict] = []
    recommended_by_system: dict[tuple[str, int], set[str]] = {}
    popularity_by_book = dict(zip(recommender.book_ids, popularity_count, strict=False))
    historical_segments = np.full(len(popularity_count), "unknown", dtype=object)
    eligible_segments, historical_tail_cut, historical_head_cut = popularity_segments(
        popularity_count[evaluation_eligible_mask],
        recommender.config.popularity_tail_quantile,
        recommender.config.popularity_head_quantile,
    )
    historical_segments[evaluation_eligible_mask] = eligible_segments
    segment_by_book = dict(
        zip(recommender.book_ids, historical_segments, strict=False)
    )
    total_popularity = float(popularity_count.sum())
    popularity_denominator = total_popularity + len(popularity_count)
    original_segments = recommender.popularity_segment
    original_tail_cut = recommender.popularity_tail_cut
    original_head_cut = recommender.popularity_head_cut
    original_eligible_mask = recommender.eligible_mask
    original_ratings_count = recommender.ratings_count
    recommender.popularity_segment = historical_segments
    recommender.popularity_tail_cut = historical_tail_cut
    recommender.popularity_head_cut = historical_head_cut
    recommender.eligible_mask = evaluation_eligible_mask
    recommender.ratings_count = popularity_count
    baseline_rankings = prepare_baseline_rankings(
        recommender,
        popularity_count,
        average_rating,
    )

    try:
        for user_idx, (user_id, train_user) in enumerate(train_groups.items(), start=1):
            if progress_every_users and user_idx % progress_every_users == 0:
                print(
                    f"[ranker_grid] {progress_label}: processed_train_users="
                    f"{user_idx:,}/{len(train_groups):,}; metric_rows={len(rows):,}",
                    flush=True,
                )
            future_user = future_groups.get(user_id)
            if future_user is None:
                continue
            train_positive = train_user.loc[_positive_mask(train_user)]
            future_positive = future_user.loc[_positive_mask(future_user)]
            relevant = set(future_positive["book_id"].astype(str)) & available_book_ids
            if train_positive.empty or not relevant:
                continue
            user_habit = (
                habit_proxies.loc[str(user_id)].to_dict()
                if str(user_id) in habit_proxies.index
                else {}
            )

            consumed = _consumed_books(train_user)
            positive_ids = train_positive["book_id"].astype(str).tolist()
            engagement = compute_engagement_weight(
                pd.to_numeric(train_positive["rating_clean"], errors="coerce").to_numpy(dtype=float),
                train_positive["has_review_text"].fillna(False).to_numpy(dtype=bool),
                pd.to_numeric(
                    train_positive["reading_duration_days"], errors="coerce"
                ).to_numpy(dtype=float),
            )
            modes = recommender.modes_from_history(positive_ids, engagement)
            if modes is None:
                continue

            positive_rows = [
                recommender._book_row[book_id]
                for book_id in positive_ids
                if book_id in recommender._book_row
            ]
            train_genres = (
                genre_flags[np.asarray(positive_rows)].max(axis=0)
                if positive_rows
                else np.zeros(len(GENRE_COLUMNS), dtype=int)
            )
            genre_mask = sum(
                (1 << bit) for bit, enabled in enumerate(train_genres) if bool(enabled)
            )
            for k in cutoffs:
                original_config = recommender.config
                recommender.config = replace(
                    original_config,
                    k=k,
                    explore_slots=min(original_config.explore_slots, k),
                )
                try:
                    near_clusters, candidate_rows = recommender.retrieved_candidate_rows(
                        modes[0], modes[1], consumed
                    )
                    content_candidate_ids = set(recommender.book_ids[candidate_rows])
                    if hybrid_v12_weights is not None:
                        global_rows = baseline_rankings.global_popularity_rows
                        genre_rows = baseline_rankings.genre_popularity_rows[genre_mask]
                        model = recommender.recommend_hybrid_v12(
                            str(user_id),
                            modes[0],
                            modes[1],
                            consumed,
                            popularity_count,
                            average_rating,
                            train_genres,
                            global_rows,
                            genre_rows,
                            weights=hybrid_v12_weights,
                            cooccurrence_score_fn=cooccurrence_score_fn,
                            user_knn_score_fn=user_knn_score_fn,
                            extra_candidate_rows_fn=hybrid_extra_candidate_rows_fn,
                            source_limit=hybrid_source_limit,
                        )
                        extra_rows = (
                            hybrid_extra_candidate_rows_fn(str(user_id))
                            if hybrid_extra_candidate_rows_fn is not None
                            else np.array([], dtype=np.int64)
                        )
                        candidate_rows = np.unique(
                            np.concatenate(
                                [
                                    candidate_rows,
                                    recommender._ordered_unseen_rows(
                                        global_rows, consumed, hybrid_source_limit
                                    ),
                                    recommender._ordered_unseen_rows(
                                        genre_rows, consumed, hybrid_source_limit
                                    ),
                                    recommender._ordered_unseen_rows(
                                        extra_rows, consumed, hybrid_source_limit
                                    ),
                                ]
                            ).astype(np.int64)
                        )
                    else:
                        model = recommender.recommend_from_modes(
                            str(user_id), modes[0], modes[1], consumed,
                            additional_score_fn=additional_score_fn,
                            blend_alpha=blend_alpha,
                        )
                    candidate_ids = set(recommender.book_ids[candidate_rows])
                finally:
                    recommender.config = original_config

                systems = {model_label: model["book_id"].astype(str).tolist()}
                systems.update(
                    baseline_recommendations(
                        recommender,
                        popularity_count,
                        average_rating,
                        consumed,
                        train_genres,
                        str(user_id),
                        k,
                        rankings=baseline_rankings,
                    )
                )

                model_slot_metrics = {
                    **_slot_metrics(model, relevant, "interest"),
                    **_slot_metrics(model, relevant, "exploration"),
                }
                for system, recommended in systems.items():
                    metrics = _binary_metrics(recommended, relevant, k)
                    segments = [segment_by_book[book_id] for book_id in recommended]
                    popularity = [math.log1p(popularity_by_book[book_id]) for book_id in recommended]
                    novelty = [
                        -math.log2((popularity_by_book[book_id] + 1.0) / popularity_denominator)
                        for book_id in recommended
                    ]
                    recommended_by_system.setdefault((system, k), set()).update(recommended)
                    rows.append(
                        {
                            "user_id": str(user_id),
                            "system": system,
                            "k": k,
                            "relevant_count": len(relevant),
                            **metrics,
                            "diversity": _intra_list_diversity(recommender, recommended),
                            "avg_recommendation_popularity": (
                                float(np.mean(popularity)) if popularity else 0.0
                            ),
                            "novelty": float(np.mean(novelty)) if novelty else 0.0,
                            "tail_share": segments.count("tail") / len(segments) if segments else 0.0,
                            "mid_share": segments.count("mid") / len(segments) if segments else 0.0,
                            "head_share": segments.count("head") / len(segments) if segments else 0.0,
                            "candidate_recall": (
                                _candidate_recall(candidate_ids, relevant)
                                if system == model_label
                                else np.nan
                            ),
                            "content_candidate_recall": (
                                _candidate_recall(content_candidate_ids, relevant)
                                if system == model_label
                                else np.nan
                            ),
                            **user_habit,
                            **(
                                model_slot_metrics
                                if system == model_label
                                else {
                                    "interest_precision": np.nan,
                                    "interest_hit_rate": np.nan,
                                    "exploration_precision": np.nan,
                                    "exploration_hit_rate": np.nan,
                                }
                            ),
                        }
                    )
    finally:
        recommender.popularity_segment = original_segments
        recommender.popularity_tail_cut = original_tail_cut
        recommender.popularity_head_cut = original_head_cut
        recommender.eligible_mask = original_eligible_mask
        recommender.ratings_count = original_ratings_count

    per_user = pd.DataFrame(rows)
    if per_user.empty:
        return pd.DataFrame(), per_user

    eligible_books = available_book_ids
    eligible_tail = {
        book_id
        for book_id in eligible_books
        if segment_by_book.get(book_id) == "tail"
    }
    cutoff_text = _utc_timestamp(temporal_cutoff).isoformat() if temporal_cutoff is not None else ""
    resolved_evaluation_mode = evaluation_mode or (
        EVALUATION_MODE
        if temporal_cutoff is not None
        else "per_user_temporal_split_training_snapshot"
    )
    users_evaluable = int(per_user["user_id"].nunique())
    resolved_users_selected = (
        int(users_selected)
        if users_selected is not None
        else int(interactions["user_id"].astype(str).nunique())
    )
    metadata = {
        "temporal_cutoff": cutoff_text,
        "evaluation_mode": resolved_evaluation_mode,
        "books_available": int(evaluation_eligible_mask.sum()),
        "users_evaluable": users_evaluable,
        "users_selected": resolved_users_selected,
        "users_discarded": resolved_users_selected - users_evaluable,
        "invalid_dates_discarded": int(invalid_date_count),
    }
    for key, value in metadata.items():
        per_user[key] = value

    summary_rows = []
    for (system, k), group in per_user.groupby(["system", "k"], sort=False):
        exposed = recommended_by_system[(system, int(k))]
        summary_rows.append(
            {
                "system": system,
                "k": int(k),
                "users": int(group["user_id"].nunique()),
                "recall": float(group["recall"].mean()),
                "precision": float(group["precision"].mean()),
                "ndcg": float(group["ndcg"].mean()),
                "map": float(group["average_precision"].mean()),
                "diversity": float(group["diversity"].mean()),
                "catalog_coverage": len(exposed) / len(eligible_books) if eligible_books else 0.0,
                "long_tail_coverage": (
                    len(exposed & eligible_tail) / len(eligible_tail) if eligible_tail else 0.0
                ),
                "avg_recommendation_popularity": float(
                    group["avg_recommendation_popularity"].mean()
                ),
                "novelty": float(group["novelty"].mean()),
                "tail_share": float(group["tail_share"].mean()),
                "mid_share": float(group["mid_share"].mean()),
                "head_share": float(group["head_share"].mean()),
                "candidate_recall": float(group["candidate_recall"].mean()),
                "content_candidate_recall": float(group["content_candidate_recall"].mean()),
                "interest_precision": float(group["interest_precision"].mean()),
                "interest_hit_rate": float(group["interest_hit_rate"].mean()),
                "exploration_precision": float(group["exploration_precision"].mean()),
                "exploration_hit_rate": float(group["exploration_hit_rate"].mean()),
                **metadata,
            }
        )
    return pd.DataFrame(summary_rows), per_user


def collect_valid_user_ids(
    path: Path,
    max_users: int,
    random_state: int = RANDOM_STATE,
) -> list[str]:
    """Return a reproducible uniform sample from global K-core valid users."""
    valid = pd.read_parquet(path, columns=["user_id", "valid"])
    user_ids = (
        valid.loc[valid["valid"].fillna(False), "user_id"]
        .astype(str)
        .drop_duplicates()
        .sort_values(kind="stable")
        .to_numpy()
    )
    if max_users <= 0 or max_users >= len(user_ids):
        return user_ids.tolist()
    rng = np.random.default_rng(random_state)
    selected = rng.choice(user_ids, size=max_users, replace=False)
    return sorted(selected.tolist())


def collect_users(
    path: Path,
    user_ids: Sequence[str],
    progress_every_batches: int = 0,
) -> pd.DataFrame:
    """Scan the canonical parquet while retaining complete histories for selected users."""
    selected = {str(user_id) for user_id in user_ids}
    if not selected:
        return pd.DataFrame(columns=INTERACTION_COLUMNS)
    chunks: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(path)
    kept_rows = 0
    for batch_idx, batch in enumerate(
        parquet.iter_batches(columns=INTERACTION_COLUMNS, batch_size=250_000),
        start=1,
    ):
        frame = batch.to_pandas()
        frame["user_id"] = frame["user_id"].astype(str)
        kept = frame[frame["user_id"].isin(selected)]
        if len(kept):
            chunks.append(kept)
            kept_rows += len(kept)
        if progress_every_batches and batch_idx % progress_every_batches == 0:
            print(
                "[ranker_grid] collect_users "
                f"batch={batch_idx:,}, kept_rows={kept_rows:,}",
                flush=True,
            )
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=INTERACTION_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=5_000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--cutoff", type=str)
    parser.add_argument("--output", type=Path, default=EVALUATION_OUTPUT)
    parser.add_argument("--users-output", type=Path, default=EVALUATION_USERS_OUTPUT)
    parser.add_argument("--activity-output", type=Path, default=EVALUATION_ACTIVITY_OUTPUT)
    parser.add_argument("--bootstrap-ci", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=1_000)
    parser.add_argument("--bootstrap-output", type=Path, default=EVALUATION_BOOTSTRAP_OUTPUT)
    parser.add_argument(
        "--bootstrap-from-users",
        type=Path,
        help="Read an existing per-user parquet and write CIs without rerunning recommendation.",
    )
    args = parser.parse_args()

    if args.bootstrap_from_users is not None:
        per_user = pd.read_parquet(args.bootstrap_from_users)
        bootstrap = bootstrap_confidence_intervals(
            per_user,
            n_resamples=args.bootstrap_resamples,
            random_state=args.random_state,
        )
        args.bootstrap_output.parent.mkdir(parents=True, exist_ok=True)
        bootstrap.to_csv(args.bootstrap_output, index=False)
        print(f"Wrote bootstrap confidence intervals to {args.bootstrap_output}")
        return

    recommender = Recommender.from_artifacts()
    valid_user_ids = collect_valid_user_ids(
        USER_FEATURES_GLOBAL_PATH,
        args.max_users,
        args.random_state,
    )
    interactions = collect_users(INTERACTIONS_CURATED_PATH, valid_user_ids)
    cutoff = (
        _utc_timestamp(args.cutoff)
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
    summary, per_user = evaluate_temporal(
        interactions,
        recommender,
        snapshot.rating_count,
        snapshot.average_rating,
        args.train_fraction,
        ks=args.k,
        temporal_cutoff=cutoff,
        catalog_available=catalog_available,
        invalid_date_count=snapshot.invalid_date_count,
        users_selected=len(valid_user_ids),
    )
    by_activity = summarize_by_activity(per_user)
    bootstrap = (
        bootstrap_confidence_intervals(
            per_user,
            n_resamples=args.bootstrap_resamples,
            random_state=args.random_state,
        )
        if args.bootstrap_ci
        else pd.DataFrame()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    safe_write_parquet(per_user, args.users_output)
    args.activity_output.parent.mkdir(parents=True, exist_ok=True)
    by_activity.to_csv(args.activity_output, index=False)
    if args.bootstrap_ci:
        args.bootstrap_output.parent.mkdir(parents=True, exist_ok=True)
        bootstrap.to_csv(args.bootstrap_output, index=False)
    print(f"Temporal cutoff: {cutoff}")
    print(
        f"Users selected/evaluable/discarded: {len(valid_user_ids):,}/"
        f"{per_user['user_id'].nunique():,}/{len(valid_user_ids) - per_user['user_id'].nunique():,}"
    )
    print(summary.to_string(index=False))
    print(f"Wrote temporal evaluation to {args.output}")
    print(f"Wrote per-user evaluation to {args.users_output}")
    print(f"Wrote activity summary to {args.activity_output}")
    if args.bootstrap_ci:
        print(f"Wrote bootstrap confidence intervals to {args.bootstrap_output}")


if __name__ == "__main__":
    main()
