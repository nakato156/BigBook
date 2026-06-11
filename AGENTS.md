# AGENTS.md

This is the canonical agent-instructions file for this repository. It is read by both
**Claude Code** and **Codex** (and any agent following the `AGENTS.md` convention). `CLAUDE.md`
is a symlink to this file, so there is a single source of truth — edit this file, not the symlink.

## Commands

The project's virtualenv lives at `env/`. Always invoke Python through it and use module form (`-m`) from the repo root, not file paths:

```bash
env/bin/python -m src.merge_master                          # build data/processed/books_master.parquet
env/bin/python -m src.reduction.build_master_feature_matrix # build PCA feature matrix + model + meta
env/bin/python scripts/build_deliverable3_clustering_outputs.py  # KMeans + hierarchy
env/bin/python -m src.curation.interactions                 # build the global deduplicated interactions artifact (+ views)
env/bin/python -m src.curation.interactions --max-rows-per-file 200000 --skip-views  # fast dry-run
env/bin/python -m src.reduction.build_user_matrix           # build user_matrix + user_meta
env/bin/python -m src.reduction.build_user_centroids        # build multi-centroid taste modes
env/bin/python -m src.reduction.recommend                   # write a small recommendation sample
env/bin/python -m src.reduction.evaluate_recommender --max-users 5000 --k 5 10 20
env/bin/python -m src.validate_artifacts                    # validate schemas and aligned ids
env/bin/python -m src.report_project_status                 # build docs/estado_v1.md
env/bin/python -m pytest                                    # run all tests
env/bin/python -m pytest tests/test_master_feature_matrix.py::test_pca_smoke_preserves_row_count  # single test
```

The production order is: curated books → `merge_master` → feature matrix/PCA → clustering →
global interactions → user matrix/meta → user centroids → ranking → temporal evaluation →
artifact validation/status report. Global interaction curation requires `books_master.parquet`
because that table defines the valid book universe. The clustering script is a standalone file,
so it is invoked by file path rather than `-m`.

## Product and Recommendation Logic

The business goal is to build a book recommendation platform that helps users discover books aligned with their interests and supports the habit of reading: ayudar a mas personas a tener el habito de lectura. Recommendation work in this repo should optimize for approachable, relevant and motivating discovery, not only for surfacing the most popular books.

Model user interests as multidimensional reading tastes, not as a single genre label. The platform should avoid basic logic like "if the user likes fantasy, recommend more fantasy." It should be able to discover cross-genre patterns such as: a reader likes youthful tone, romance, adventure, light fantasy and accessible reading, even when the books do not all belong to the exact same genre.

For k-means and similar clustering work, use book-level granularity across the full catalog:

```
one row = one book_id = one book vector
```

The main clustering input should be `data/features/master_feature_matrix.parquet`, using `pc_0..pc_N` as the vector columns. Cluster `book_id` rows, not genres. Genres should be used as signals, filters, explanations or diversity controls, but not as the unit of the main clustering model.

Avoid popularity bias. `ratings_count` and `text_reviews_count` use `log1p`; `average_rating`
does not. All numeric signals are controlled by block standardization/weighting. Popularity does
not gate eligibility or order normal `interest` slots. It does control the dedicated exploration
policy: exploration excludes `head`, prefers `tail` over `mid`, and must pass an interest-similarity
floor. Popularity also defines exposure diagnostics and the B1/B2 evaluation baselines.

## Pipeline Architecture

The end-to-end V1 flow is:

```text
curated books -> master -> feature matrix/PCA -> clustering
              -> global interactions -> user profiles -> ranking -> evaluation
```

Genre-specific notebooks produce the curated book boundary. The canonical production stages after
that boundary are Python modules/scripts.

### Stage boundary: curated parquets per genre (book side)

The notebook stages must produce a `books_curated.parquet` for every genre key in `src/config.py::CATEGORIES`:

```
data/processed/<category>/books_curated.parquet
```

Category keys are `fantasy_paranormal`, `mystery_thriller_crime`, `history_biography`, `young_adult`, `romance`. `src/merge_master.py` consults `LEGACY_PROCESSED_DIRS` to also accept the older `data/processed/fantasy` and `data/processed/history` paths, so do not delete that fallback when refactoring.

The legacy per-genre `data/processed/<category>/interactions_curated.parquet` files are **deprecated** (kept only as a historical backup, never overwritten). They were biased (0% `rating_missing`, 100% `is_read` — the whole implicit layer was dropped upstream), wrongly partitioned (~71% of users span >=2 genres, so per-category K-core and rating bias are wrong), and duplicated (~19% of `(user, book)` pairs repeated across dumps). The interaction side is now a single **global** artifact (see below), not per-genre.

### `src/curation/interactions.py` — global deduplicated interactions

The single source of truth for the user side. Streams the five raw `goodreads_interactions_*.json.gz` dumps and writes three artifacts to `data/processed/`:

- `interactions_curated.parquet` — **canonical, cross-category, deduplicated, no review text**. One row per interaction (`interaction_key`), keyed by `review_id` when present else `user_id|book_id` (a deterministic `pandas.util.hash_pandas_object` uint64 — a stable join key). Dedup is **keep-best by priority** (`review > rating_only > read_no_rating > want_to_read`, then rating present, then `is_read`, then recency), NOT keep-first, so the strongest signal survives when duplicates differ. The implicit layer is **recovered**: `rating == 0` → `rating_clean` NA + `rating_missing`, and rows with no read/rating/review are tagged `engagement_mode == "want_to_read"` (kept, with `interaction_weight` 0.3, but excluded from the positive taste vector).
- `review_texts.parquet` — only rows with `has_review_text`, joinable to the canonical by `interaction_key` (keeps text out of the hot path).
- `user_features_global.parquet` — **global** per-user stats with `user_rating_bias` (= user mean − global mean, neutral 0.0 when no ratings) and the global K-core flag `valid` (`read_or_rated_count >= K_USER_MIN`, currently 3).

Implementation notes that are load-bearing:
- **K-core and `user_rating_bias` are GLOBAL** (computed across all five categories), never per-category. The build is always global — there is no `--category` flag.
- The valid **book universe** comes from `books_master.parquet` (aligned to PCA); the item side (`books_master`/PCA/clusters) is treated as stable and never recomputed here. Interactions whose `book_id` is not in that universe are dropped.
- Streaming is **one JSON parse + two parquet passes**: scan 1 parses the gzip dumps once, writes a `*_staging.parquet`, and builds the in-memory `key -> max_priority` index (sorted uint64 arrays + `searchsorted`, never a flat 62M-row join); scan 2 reads the cheap staging parquet to emit keep-best winners + accumulate global user stats; a final parquet post-pass applies the global K-core, merges bias, and splits review text out. Re-parsing the JSON twice would roughly double runtime — keep the staging hop.
- `source_category_count` (how many genre dumps an interaction appeared in) is an **optional diagnostic** behind `--with-source-category-count`; it must never block the main build.

Per-category `data/processed/<category>/interactions_view.parquet` files are derived **only for EDA/notebooks** (`build_category_views`, genre-filtered by `genre_<cat>` from `books_master`); a multi-genre interaction appears in several views by design, and these views are not a valid modeling input.

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
- The legacy per-category feature-matrix module was removed. `src/reduction/build_master_feature_matrix.py`, operating on `books_master.parquet`, is the canonical item representation pipeline.

## V1 Scope Boundary

The academic V1 includes reproducible artifacts, ranking, N0 temporal evaluation, descriptive N1
habit proxies, validation and a generated status report. API/UI, live telemetry, N2 causal claims,
A/B testing, per-cutoff PCA/clustering rebuilds, item cold-start and adaptive bandits are explicitly
out of scope.

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.
On some systems `cp`, `mv`, and `rm` are aliased to `-i` (interactive), which makes an agent hang
indefinitely waiting for `y/n` input.

```bash
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

Other commands that may prompt: `scp`/`ssh` (use `-o BatchMode=yes`), `apt-get` (use `-y`),
`brew` (use `HOMEBREW_NO_AUTO_UPDATE=1`).

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- ALWAYS read graphify-out/GRAPH_REPORT.md before reading any source files, running grep/glob searches, or answering codebase questions. The graph is your primary map of the codebase.
- IF graphify-out/wiki/index.md EXISTS, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
