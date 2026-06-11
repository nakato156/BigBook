from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import BOOKS_MASTER_PATH, MASTER_FEATURE_MATRIX_PATH, PROJECT_ROOT
from src.utils.io import safe_write_parquet

RANDOM_STATE = 42
SELECTED_K = 100
COMPARISON_KS = [50, 100]
N_MACRO_CLUSTERS = 10
SILHOUETTE_SAMPLE_SIZE = 10000

FEATURE_MATRIX_PATH = MASTER_FEATURE_MATRIX_PATH
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "clustering"

GENRE_COLUMNS = {
    "genre_fantasy": "fantasy",
    "genre_mystery": "mystery",
    "genre_history": "history",
    "genre_ya": "young_adult",
    "genre_romance": "romance",
}


def pc_sort_key(column: str) -> int:
    return int(column.split("_", maxsplit=1)[1])


def load_feature_matrix() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    pc_cols = sorted([col for col in features.columns if col.startswith("pc_")], key=pc_sort_key)
    if not pc_cols:
        raise ValueError(f"No PCA columns found in {FEATURE_MATRIX_PATH}")
    x = features[pc_cols].to_numpy(dtype=np.float32, copy=True)
    return features, pc_cols, x


def load_metadata() -> pd.DataFrame:
    metadata = pd.read_parquet(BOOKS_MASTER_PATH)
    available = [
        col
        for col in ["book_id", "title", "average_rating", "ratings_count", *GENRE_COLUMNS]
        if col in metadata.columns
    ]
    metadata = metadata[available].copy()

    genre_cols = [col for col in GENRE_COLUMNS if col in metadata.columns]
    if genre_cols:
        metadata["genres"] = metadata[genre_cols].apply(
            lambda row: "|".join(
                label for col, label in GENRE_COLUMNS.items() if col in row.index and row[col] == 1
            ),
            axis=1,
        )
        metadata["genres"] = metadata["genres"].replace("", "unknown")

    keep_cols = ["book_id", "title", "genres", "average_rating", "ratings_count"]
    return metadata[[col for col in keep_cols if col in metadata.columns]]


def get_kmeans_model(k: int, x: np.ndarray) -> KMeans:
    model_path = OUTPUT_DIR / f"kmeans_model_k{k}.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        if getattr(model, "n_clusters", None) == k and hasattr(model, "cluster_centers_"):
            return model

    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto").fit(x)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def cluster_size_stats(labels: np.ndarray) -> dict[str, int | float]:
    sizes = pd.Series(labels).value_counts().sort_index()
    return {
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "mean_cluster_size": float(sizes.mean()),
        "median_cluster_size": float(sizes.median()),
        "num_clusters_under_20_books": int((sizes < 20).sum()),
        "num_clusters_under_50_books": int((sizes < 50).sum()),
        "num_clusters_over_3000_books": int((sizes > 3000).sum()),
    }


def sampled_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    return float(
        silhouette_score(
            x,
            labels,
            sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(labels)),
            random_state=RANDOM_STATE,
        )
    )


def write_k50_k100_comparison(x: np.ndarray) -> pd.DataFrame:
    rows = []
    notes = {
        50: "Cleaner geometry, broader zones.",
        100: "More granular zones, lower silhouette.",
    }
    for k in COMPARISON_KS:
        model = get_kmeans_model(k, x)
        labels = model.predict(x).astype(np.int32)
        stats = cluster_size_stats(labels)
        silhouette = sampled_silhouette(x, labels)
        row = {
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": silhouette,
            **stats,
            "business_tradeoff_note": notes[k],
        }
        rows.append(row)

    comparison = pd.DataFrame(rows)
    path = OUTPUT_DIR / "k50_vs_k100_comparison.csv"
    comparison.to_csv(path, index=False)
    return comparison


def quality_flag(n_books: int) -> str:
    if n_books < 20:
        return "very_small_cluster"
    if n_books < 50:
        return "small_cluster"
    if n_books > 3000:
        return "very_large_cluster"
    return "normal"


def write_assignments_and_quality(
    features: pd.DataFrame,
    labels: np.ndarray,
    distances: np.ndarray,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = pd.DataFrame(
        {
            "book_id": features["book_id"].to_numpy(),
            "cluster": labels.astype(np.int32),
            "distance_to_centroid": distances.astype(np.float32),
        }
    )
    safe_write_parquet(
        assignments[["book_id", "cluster"]],
        OUTPUT_DIR / f"book_clusters_k{k}.parquet",
    )

    counts = assignments["cluster"].value_counts().sort_index()
    quality = counts.rename_axis("cluster").reset_index(name="n_books")
    quality["percentage"] = quality["n_books"] / len(assignments) * 100
    quality["flag"] = quality["n_books"].map(quality_flag)
    quality_path = OUTPUT_DIR / f"cluster_quality_flags_k{k}.csv"
    quality.to_csv(quality_path, index=False)

    return assignments, quality


def write_cluster_cohesion(assignments: pd.DataFrame, k: int) -> pd.DataFrame:
    cohesion = (
        assignments.groupby("cluster", sort=True)["distance_to_centroid"]
        .agg(
            n_books="size",
            mean_distance_to_centroid="mean",
            median_distance_to_centroid="median",
            max_distance_to_centroid="max",
            std_distance_to_centroid="std",
        )
        .reset_index()
    )
    cohesion["std_distance_to_centroid"] = cohesion["std_distance_to_centroid"].fillna(0.0)
    csv_path = OUTPUT_DIR / f"cluster_cohesion_k{k}.csv"
    cohesion.to_csv(csv_path, index=False)

    plot_data = cohesion.sort_values("cluster")
    plt.figure(figsize=(12, 5))
    plt.bar(plot_data["cluster"].astype(str), plot_data["mean_distance_to_centroid"], color="#2F6F7E")
    plt.xlabel("K-Means cluster")
    plt.ylabel("Mean distance to centroid")
    plt.title(f"K={k} cluster cohesion")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    png_path = OUTPUT_DIR / f"cluster_cohesion_k{k}.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    return cohesion


def write_cluster_examples(assignments: pd.DataFrame, metadata: pd.DataFrame, k: int) -> pd.DataFrame:
    examples = (
        assignments.sort_values(["cluster", "distance_to_centroid"], kind="mergesort")
        .groupby("cluster", as_index=False)
        .head(10)
        .merge(metadata, on="book_id", how="left")
    )
    ordered_cols = [
        "cluster",
        "book_id",
        "title",
        "genres",
        "average_rating",
        "ratings_count",
        "distance_to_centroid",
    ]
    examples = examples[[col for col in ordered_cols if col in examples.columns]]
    path = OUTPUT_DIR / f"cluster_examples_k{k}.csv"
    examples.to_csv(path, index=False)
    return examples


def write_hierarchical_centroid_outputs(
    centroids: np.ndarray,
    quality: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_ids = np.arange(k, dtype=np.int32)
    linkage_matrix = linkage(centroids, method="ward")

    plt.figure(figsize=(16, 7))
    dendrogram(
        linkage_matrix,
        labels=[str(cluster) for cluster in cluster_ids],
        leaf_rotation=90,
        leaf_font_size=6,
        color_threshold=None,
    )
    plt.title(f"Ward hierarchical clustering over K={k} centroids")
    plt.xlabel("K-Means cluster")
    plt.ylabel("Ward distance")
    plt.tight_layout()
    dendrogram_path = OUTPUT_DIR / f"hclust_dendrogram_k{k}_ward.png"
    plt.savefig(dendrogram_path, dpi=200)
    plt.close()

    raw_macro = fcluster(linkage_matrix, t=N_MACRO_CLUSTERS, criterion="maxclust")
    ordered_raw_labels = sorted(np.unique(raw_macro), key=lambda label: cluster_ids[raw_macro == label].min())
    relabel = {raw_label: idx for idx, raw_label in enumerate(ordered_raw_labels)}
    macro_labels = np.array([relabel[label] for label in raw_macro], dtype=np.int32)

    assignments = pd.DataFrame({"cluster": cluster_ids, "macro_cluster": macro_labels})
    assignments_path = OUTPUT_DIR / f"macro_cluster_assignments_k{k}.csv"
    assignments.to_csv(assignments_path, index=False)

    summary_base = assignments.merge(quality[["cluster", "n_books"]], on="cluster", how="left")
    summary = (
        summary_base.groupby("macro_cluster", sort=True)
        .agg(n_kmeans_clusters=("cluster", "size"), n_books=("n_books", "sum"))
        .reset_index()
    )
    total_books = int(summary["n_books"].sum())
    summary["percentage"] = summary["n_books"] / total_books * 100
    summary_path = OUTPUT_DIR / f"macro_cluster_summary_k{k}.csv"
    summary.to_csv(summary_path, index=False)
    return assignments, summary


def print_logs(
    n_books: int,
    n_features: int,
    selected_k: int,
    selected_silhouette: float,
    quality: pd.DataFrame,
    output_paths: list[Path],
) -> None:
    sizes = quality["n_books"]
    flagged_count = int((quality["flag"] != "normal").sum())
    print(f"Books: {n_books:,}")
    print(f"PCA features: {n_features:,}")
    print(f"Selected K: {selected_k}")
    print(f"Silhouette: {selected_silhouette:.4f}")
    print(
        "Cluster size min/max/median: "
        f"{int(sizes.min()):,}/{int(sizes.max()):,}/{float(sizes.median()):,.1f}"
    )
    print(f"Flagged small/large clusters: {flagged_count}")
    print("Saved outputs:")
    for path in output_paths:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, pc_cols, x = load_feature_matrix()
    metadata = load_metadata()

    comparison = write_k50_k100_comparison(x)

    model = get_kmeans_model(SELECTED_K, x)
    labels = model.predict(x).astype(np.int32)
    centroids = model.cluster_centers_.astype(np.float32)
    distances = np.linalg.norm(x - centroids[labels], axis=1)

    np.save(OUTPUT_DIR / f"kmeans_centroids_k{SELECTED_K}.npy", centroids)
    joblib.dump(model, OUTPUT_DIR / f"kmeans_model_k{SELECTED_K}.joblib")

    assignments, quality = write_assignments_and_quality(features, labels, distances, SELECTED_K)
    cohesion = write_cluster_cohesion(assignments, SELECTED_K)
    examples = write_cluster_examples(assignments, metadata, SELECTED_K)
    macro_assignments, macro_summary = write_hierarchical_centroid_outputs(
        centroids,
        quality,
        SELECTED_K,
    )

    silhouette = float(
        comparison.loc[comparison["k"] == SELECTED_K, "silhouette"].iloc[0]
    )
    output_paths = [
        OUTPUT_DIR / "k50_vs_k100_comparison.csv",
        OUTPUT_DIR / f"cluster_quality_flags_k{SELECTED_K}.csv",
        OUTPUT_DIR / f"cluster_cohesion_k{SELECTED_K}.csv",
        OUTPUT_DIR / f"cluster_cohesion_k{SELECTED_K}.png",
        OUTPUT_DIR / f"cluster_examples_k{SELECTED_K}.csv",
        OUTPUT_DIR / f"hclust_dendrogram_k{SELECTED_K}_ward.png",
        OUTPUT_DIR / f"macro_cluster_assignments_k{SELECTED_K}.csv",
        OUTPUT_DIR / f"macro_cluster_summary_k{SELECTED_K}.csv",
    ]
    print_logs(len(features), len(pc_cols), SELECTED_K, silhouette, quality, output_paths)

    # Keep locals visibly used for notebook-style debugging without changing output files.
    _ = cohesion, examples, macro_assignments, macro_summary


if __name__ == "__main__":
    main()
