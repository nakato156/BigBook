"""Manual multi-snapshot backtest with independently fitted PCA and clustering."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    BOOKS_MASTER_PATH,
    INTERACTIONS_CURATED_PATH,
    PROJECT_ROOT,
    USER_FEATURES_GLOBAL_PATH,
)
from src.reduction.build_master_feature_matrix import build_feature_matrix
from src.reduction.build_user_centroids import build_user_centroids
from src.reduction.build_user_matrix import build_user_artifacts
from src.reduction.evaluate_recommender import (
    MIN_VALID_DATE,
    collect_users,
    collect_valid_user_ids,
    evaluate_temporal,
    historical_catalog_mask,
    historical_popularity_snapshot,
)
from src.reduction.recommend import GENRE_COLUMNS, RankingConfig, Recommender, pc_columns
from src.utils.io import safe_write_parquet

RANDOM_STATE = 42
DEFAULT_SNAPSHOTS = (
    "2014-01-01",
    "2015-01-01",
    "2016-06-09T19:30:05.200000+00:00",
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "recommendations" / "multi_snapshot_backtest.csv"
)
SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "outputs" / "snapshots"


def historical_master_for_snapshot(
    master: pd.DataFrame,
    book_ids: np.ndarray,
    available: np.ndarray,
    rating_count: np.ndarray,
    average_rating: np.ndarray,
    review_count: np.ndarray,
) -> pd.DataFrame:
    """Filter the catalog and replace future-sensitive numeric fields with historical values."""
    aligned = master.copy()
    aligned["book_id"] = aligned["book_id"].astype(str)
    aligned = aligned.set_index("book_id").reindex(np.asarray(book_ids, dtype=str)).reset_index()
    aligned["ratings_count"] = np.asarray(rating_count, dtype=np.float64)
    historical_average = np.asarray(average_rating, dtype=np.float64).copy()
    historical_average[np.asarray(rating_count, dtype=np.float64) == 0] = np.nan
    aligned["average_rating"] = historical_average
    aligned["text_reviews_count"] = np.asarray(review_count, dtype=np.float64)
    return aligned.loc[np.asarray(available, dtype=bool)].reset_index(drop=True)


def _snapshot_user_inputs(
    interactions: pd.DataFrame,
    cutoff: pd.Timestamp,
    available_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(interactions["date_added"], errors="coerce", utc=True)
    train = interactions.loc[
        dates.notna()
        & (dates <= cutoff)
        & interactions["book_id"].astype(str).isin(available_ids)
    ].copy()
    train["date_added"] = dates.loc[train.index].dt.tz_localize(None)
    train["is_want_to_read"] = (
        ~train["is_read"].fillna(False)
        & pd.to_numeric(train["rating_clean"], errors="coerce").isna()
    )
    ratings = pd.to_numeric(train["rating_clean"], errors="coerce")
    global_mean = float(ratings.mean()) if ratings.notna().any() else 0.0
    user_mean = ratings.groupby(train["user_id"].astype(str)).mean()
    user_ids = pd.Index(train["user_id"].astype(str).unique(), name="user_id")
    user_features = pd.DataFrame({"user_id": user_ids.astype(str)})
    user_features["user_rating_bias"] = (
        user_features["user_id"].map(user_mean).fillna(global_mean) - global_mean
    )
    return train, user_features


def build_snapshot_recommender(
    master: pd.DataFrame,
    interactions: pd.DataFrame,
    cutoff: pd.Timestamp,
    snapshot_dir: Path,
    n_clusters: int = 100,
    n_macro_clusters: int = 10,
    config: RankingConfig | None = None,
) -> tuple[Recommender, pd.DataFrame]:
    """Fit and persist one isolated item/user representation for a single cutoff."""
    feature_matrix, pca_model, pca_meta = build_feature_matrix(master)
    pc_cols = pc_columns(feature_matrix)
    x = feature_matrix[pc_cols].to_numpy(dtype=np.float32)
    if len(x) < n_clusters:
        raise ValueError(
            f"Snapshot has {len(x)} books, fewer than requested n_clusters={n_clusters}."
        )
    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init="auto",
    ).fit(x)
    labels = model.labels_.astype(np.int32)
    centroids = model.cluster_centers_.astype(np.float32)
    if n_clusters == 1:
        macro_of_cluster = np.zeros(1, dtype=np.int32)
    else:
        raw = fcluster(
            linkage(centroids, method="ward"),
            t=min(n_macro_clusters, n_clusters),
            criterion="maxclust",
        )
        ordered = sorted(np.unique(raw), key=lambda value: np.flatnonzero(raw == value)[0])
        relabel = {value: idx for idx, value in enumerate(ordered)}
        macro_of_cluster = np.array([relabel[value] for value in raw], dtype=np.int32)

    features_dir = snapshot_dir / "features"
    clustering_dir = snapshot_dir / "clustering"
    safe_write_parquet(feature_matrix, features_dir / "master_feature_matrix.parquet")
    features_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca_model, features_dir / "master_pca_model.joblib")
    (features_dir / "master_pca_meta.json").write_text(
        json.dumps(pca_meta, indent=2),
        encoding="utf-8",
    )
    safe_write_parquet(
        pd.DataFrame(
            {"book_id": feature_matrix["book_id"].astype(str), "cluster": labels}
        ),
        clustering_dir / f"book_clusters_k{n_clusters}.parquet",
    )
    clustering_dir.mkdir(parents=True, exist_ok=True)
    np.save(clustering_dir / f"kmeans_centroids_k{n_clusters}.npy", centroids)
    pd.DataFrame(
        {"cluster": np.arange(n_clusters), "macro_cluster": macro_of_cluster}
    ).to_csv(
        clustering_dir / f"macro_cluster_assignments_k{n_clusters}.csv",
        index=False,
    )

    available_ids = set(feature_matrix["book_id"].astype(str))
    train, user_features = _snapshot_user_inputs(
        interactions,
        cutoff,
        available_ids,
    )
    user_matrix, user_meta, _ = build_user_artifacts(
        feature_matrix,
        master,
        user_features,
        [train],
    )
    user_centroids, _ = build_user_centroids(feature_matrix, [train])
    safe_write_parquet(user_matrix, features_dir / "user_matrix.parquet")
    safe_write_parquet(user_meta, features_dir / "user_meta.parquet")
    safe_write_parquet(user_centroids, features_dir / "user_centroids.parquet")

    master_indexed = master.set_index(master["book_id"].astype(str)).reindex(
        feature_matrix["book_id"].astype(str)
    )
    genres = master_indexed[["title", *GENRE_COLUMNS]].copy()
    genres.index = feature_matrix["book_id"].astype(str)
    user_pc_cols = pc_columns(user_matrix)
    centroid_pc_cols = pc_columns(user_centroids)
    recommender = Recommender(
        book_ids=feature_matrix["book_id"].astype(str).to_numpy(),
        book_pc=x,
        ratings_count=pd.to_numeric(
            master_indexed["ratings_count"], errors="coerce"
        ).fillna(0).to_numpy(dtype=np.float64),
        num_pages=pd.to_numeric(
            master_indexed["num_pages"], errors="coerce"
        ).to_numpy(dtype=np.float64),
        genres=genres,
        book_cluster=labels,
        centroids=centroids,
        macro_of_cluster=macro_of_cluster,
        user_ids=user_matrix["user_id"].astype(str).to_numpy(),
        user_pc=(
            user_matrix[user_pc_cols].to_numpy(dtype=np.float32)
            if user_pc_cols
            else np.empty((0, len(pc_cols)), dtype=np.float32)
        ),
        positive_count_by_user=dict(
            zip(
                user_meta["user_id"].astype(str),
                user_meta["positive_count"].astype(int),
                strict=False,
            )
        ),
        centroid_user_ids=user_centroids["user_id"].astype(str).to_numpy(),
        user_centroid_pc=(
            user_centroids[centroid_pc_cols].to_numpy(dtype=np.float32)
            if centroid_pc_cols
            else np.empty((0, len(pc_cols)), dtype=np.float32)
        ),
        user_centroid_weight=user_centroids["centroid_weight"].to_numpy(dtype=np.float32),
        pc_cols=pc_cols,
        config=config or RankingConfig(),
    )
    return recommender, train


def run_multi_snapshot(
    snapshot_dates: Sequence[pd.Timestamp | str],
    runner: Callable[[pd.Timestamp, Path], pd.DataFrame],
    snapshot_root: Path,
) -> pd.DataFrame:
    """Run isolated snapshots and concatenate only their final aggregate metrics."""
    frames: list[pd.DataFrame] = []
    for value in snapshot_dates:
        cutoff = pd.Timestamp(value)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        snapshot_dir = snapshot_root / cutoff.strftime("%Y-%m-%d")
        summary = runner(cutoff, snapshot_dir).copy()
        summary.insert(0, "snapshot_date", cutoff.isoformat())
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", nargs="+", default=list(DEFAULT_SNAPSHOTS))
    parser.add_argument("--max-users", type=int, default=1_000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--n-macro-clusters", type=int, default=10)
    parser.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    full_master = pd.read_parquet(BOOKS_MASTER_PATH)
    full_master["book_id"] = full_master["book_id"].astype(str)
    full_book_ids = full_master["book_id"].to_numpy(dtype=str)
    selected_ids = collect_valid_user_ids(
        USER_FEATURES_GLOBAL_PATH,
        args.max_users,
        args.random_state,
    )
    cohort = collect_users(INTERACTIONS_CURATED_PATH, selected_ids)

    def runner(cutoff: pd.Timestamp, snapshot_dir: Path) -> pd.DataFrame:
        if cutoff < MIN_VALID_DATE:
            raise ValueError(f"Snapshot must be on or after {MIN_VALID_DATE.date()}.")
        snapshot = historical_popularity_snapshot(
            INTERACTIONS_CURATED_PATH,
            full_book_ids,
            cutoff,
        )
        available = historical_catalog_mask(
            BOOKS_MASTER_PATH,
            full_book_ids,
            cutoff,
            snapshot.first_observed,
        )
        master = historical_master_for_snapshot(
            full_master,
            full_book_ids,
            available,
            snapshot.rating_count,
            snapshot.average_rating,
            snapshot.review_count,
        )
        recommender, _ = build_snapshot_recommender(
            master,
            cohort,
            cutoff,
            snapshot_dir,
            n_clusters=args.n_clusters,
            n_macro_clusters=args.n_macro_clusters,
        )
        snapshot_rows = pd.Index(full_book_ids).get_indexer(recommender.book_ids)
        summary, _ = evaluate_temporal(
            cohort,
            recommender,
            popularity_count=snapshot.rating_count[snapshot_rows],
            average_rating=snapshot.average_rating[snapshot_rows],
            ks=args.k,
            temporal_cutoff=cutoff,
            catalog_available=np.ones(len(recommender.book_ids), dtype=bool),
            invalid_date_count=snapshot.invalid_date_count,
            evaluation_mode="strict_refit_snapshot",
            users_selected=len(selected_ids),
        )
        return summary

    results = run_multi_snapshot(args.snapshots, runner, args.snapshot_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))
    print(f"Wrote multi-snapshot backtest to {args.output}")


if __name__ == "__main__":
    main()
