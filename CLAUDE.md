# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project's virtualenv lives at `env/`. Always invoke Python through it and use module form (`-m`) from the repo root, not file paths:

```bash
env/bin/python -m src.merge_master                          # build data/processed/books_master.parquet
env/bin/python -m src.reduction.build_master_feature_matrix # build PCA feature matrix + model + meta
env/bin/python -m pytest                                    # run all tests
env/bin/python -m pytest tests/test_master_feature_matrix.py::test_pca_smoke_preserves_row_count  # single test
env/bin/python scripts/build_deliverable3_clustering_outputs.py  # KMeans + hierarchical clustering outputs
```

The two pipeline scripts are ordered: `merge_master` must run first because `build_master_feature_matrix` reads `data/processed/books_master.parquet` and refuses to start if the file is missing. The clustering script is downstream of both: it consumes `data/features/master_feature_matrix.parquet` and `data/processed/books_master.parquet`. Unlike the `src/` modules, it is a standalone file (not a package module), so it is invoked by file path, not `-m`.

## Product and Recommendation Logic

The business goal is to build a book recommendation platform that helps users discover books aligned with their interests and supports the habit of reading: ayudar a mas personas a tener el habito de lectura. Recommendation work in this repo should optimize for approachable, relevant and motivating discovery, not only for surfacing the most popular books.

Model user interests as multidimensional reading tastes, not as a single genre label. The platform should avoid basic logic like "if the user likes fantasy, recommend more fantasy." It should be able to discover cross-genre patterns such as: a reader likes youthful tone, romance, adventure, light fantasy and accessible reading, even when the books do not all belong to the exact same genre.

For k-means and similar clustering work, use book-level granularity across the full catalog:

```
one row = one book_id = one book vector
```

The main clustering input should be `data/features/master_feature_matrix.parquet`, using `pc_0..pc_N` as the vector columns. Cluster `book_id` rows, not genres. Genres should be used as signals, filters, explanations or diversity controls, but not as the unit of the main clustering model.

Avoid popularity bias. Popularity features such as `ratings_count`, `text_reviews_count` and `average_rating` are useful quality/confidence signals, but they should not become the whole recommendation objective. The current pipeline reduces popularity outlier influence with `log1p` transforms and block weighting. Downstream ranking should prioritize interest similarity first, then use popularity as a secondary signal.

## Pipeline Architecture

The end-to-end flow is `clean -> reduce -> curation -> merge -> feature matrix -> PCA`. The first three phases live in genre-specific notebooks under `notebooks/{cleaning,processing,reduction}/`; the last three are Python modules under `src/`.

### Stage boundary: curated parquets per genre

The notebook stages must produce these files for every genre key in `src/config.py::CATEGORIES`:

```
data/processed/<category>/books_curated.parquet
data/processed/<category>/interactions_curated.parquet
```

Category keys are `fantasy_paranormal`, `mystery_thriller_crime`, `history_biography`, `young_adult`, `romance`. `src/merge_master.py` and `src/reduction/feature_matrix.py` both consult `LEGACY_PROCESSED_DIRS` to also accept the older `data/processed/fantasy` and `data/processed/history` paths, so do not delete that fallback when refactoring.

### `src/merge_master.py` — master table

Loads each genre's `books_curated.parquet`, tags it with five `genre_*` flags (one per genre), concatenates, and deduplicates by `book_id` using `max` on genre flags and `first` on everything else (so a book in multiple genres has multiple flags set to 1). Then computes `genre_count`, recomputes `author_count` from the `authors` list, drops fields that aren't part of the final 17-column schema (including all `theme_*` columns), and enforces dtypes. The output schema is fixed in `final_columns` at the bottom of `main()` — adding a feature here also requires updating `REQUIRED_MASTER_COLUMNS` in `build_master_feature_matrix.py`.

Note the curated→master column rename: curated `is_in_series` (bool) becomes master `series` (int 0/1).

### `src/reduction/build_master_feature_matrix.py` — PCA representation

Builds three blocks from `books_master.parquet`, each documented in the README:

- **numeric** (9 cols): includes `log1p(ratings_count)`, `log1p(text_reviews_count)`, median imputation for `num_pages` and `publication_year`, plus explicit `*_missing` flag columns. The missingness flags are intentional signal — do not remove them.
- **binary** (11 cols): five `genre_*` flags, `series`, and one-hot `language_code_*` from the top-N most frequent codes (with a `language_code_other` bucket that is currently empty in production data — keep it anyway, the model artifact assumes it).
- **embeddings** (256 cols): from `data/embeddings/description_embeddings.parquet`, generated with `google/embeddinggemma-300m` (gated — requires `HF_TOKEN` env var or `huggingface-cli login`). The cache is incremental: only missing `book_id` rows are encoded, then merged back. Text fallback chain is `description → title → "[no description]"`.

The three blocks go through `src/reduction/pca.py::standardize_and_weight_blocks`, which `StandardScaler`s each block independently and multiplies by `1/sqrt(block_dim)`. This block weighting is load-bearing: without it, the 256-dim embedding block dominates the first PCs purely by column count. The diagnostic `embedding_dominated_first_5_count` in the meta JSON exists to detect regressions of this property — if it climbs above 0, suspect block-weighting changes first.

PCA is fit with `PCA(n_components=0.95, svd_solver="full")` so the component count is determined by variance, not hard-coded.

### Outputs

```
data/features/master_feature_matrix.parquet  # book_id + pc_0..pc_N
data/features/master_pca_model.joblib        # scalers, medians, lang categories, block weights, fitted PCA
data/features/master_pca_meta.json           # diagnostics, explained variance, spot checks
```

The `.joblib` bundle is the only supported way to transform new books consistently — it stores everything needed (per-block scaler, numeric medians, language category list, block weights, fitted PCA) so callers don't have to reproduce the preprocessing.

### `scripts/build_deliverable3_clustering_outputs.py` — clustering outputs

The concrete clustering implementation of the recommendation logic above. Clusters `book_id` rows on the `pc_*` columns of `master_feature_matrix.parquet` (one row = one book vector), joining `books_master.parquet` only for human-readable labels (`title`, `genres`, `average_rating`, `ratings_count`) — genres are explanation/diversity signals, not the clustering unit.

It runs KMeans at two granularities (`COMPARISON_KS = [50, 100]`, `RANDOM_STATE = 42`) and writes a `k50_vs_k100_comparison.csv` with inertia, sampled silhouette (sample size 10000), and size stats. `SELECTED_K = 100` is the production cut; its centroids feed a Ward hierarchical `linkage` that is cut into `N_MACRO_CLUSTERS = 10` macro-clusters, giving a two-level cluster hierarchy. Fitted KMeans models are cached as `kmeans_model_k{K}.joblib` and reused if the cached `n_clusters` matches.

Outputs go to `data/outputs/clustering/`: `book_clusters_k{K}.parquet` (book_id → cluster), per-cluster cohesion/quality/examples CSVs (+ PNG plots), macro-cluster assignments/summary, and the K=100 centroids (`.npy`). Clusters are flagged `very_small`/`small`/`very_large` by size for quality review.

## Conventions

- All paths derive from `src/config.py::PROJECT_ROOT`; do not hardcode absolute paths.
- Use `src/utils/io.py::safe_write_parquet` for writes (it handles `mkdir -p` of the parent and uses pyarrow); use `read_jsonl_chunks` / `read_parquet_chunks` for streaming reads of the raw Goodreads JSON dumps.
- Float features are `float32` everywhere in the reduction pipeline — the PCA matrix assembly asserts `np.isfinite(values).all()` and will reject NaN/Inf, so impute before passing into `standardize_and_weight_blocks`.
- The `src/reduction/feature_matrix.py` module (per-category book/interaction/user features) is a separate, older pipeline from `build_master_feature_matrix.py` (PCA on books_master). They share the curated-parquet inputs but produce different artifacts; do not conflate them.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- ALWAYS read graphify-out/GRAPH_REPORT.md before reading any source files, running grep/glob searches, or answering codebase questions. The graph is your primary map of the codebase.
- IF graphify-out/wiki/index.md EXISTS, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
