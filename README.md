# Book Recommendation System

Course project for Big Data. The project builds a hybrid book representation for recommendation and analysis by combining Goodreads metadata, genre labels, interaction-derived popularity signals and text embeddings from book descriptions. The final artifact is a PCA-reduced feature matrix with one vector per `book_id`.

The business goal is to build a book recommendation platform that helps users discover books aligned with their interests and, more importantly, supports the habit of reading: ayudar a mas personas a tener el habito de lectura. The system should recommend books that feel approachable, relevant and motivating for each reader, not only the books that are already the most popular.

## Problem Definition

This is the core statement of what the system does. Everything else in this README (representation, clustering, ranking) exists to serve it.

> **We recommend books (`book_id`, each one a PCA vector `pc_0..pc_172`) to readers (`user_id`), based on the multidimensional reading taste inferred from their interaction history (reads, ratings, reviews) and on book-to-book similarity in the reduced feature space, in order to optimize relevant and motivating reading starts and completions — prioritizing interest similarity over popularity — so that readers sustain and grow the habit of reading.**

The same idea in `X / Y / Z` form:

```text
We recommend  X = books from the catalog, as PCA vectors, grouped into taste clusters
based on      Y = the user's interaction history + similarity in PCA space + book clusters
                  (genre as a filter/explanation/diversity signal; popularity as exposure diagnostic)
to optimize   Z = relevant readings started and finished (proxy: is_read + positive rating)
                  that sustain the reading habit (retention), avoiding popularity bias
                  and single-genre filter bubbles.
```

### What is a user

A reader identified by `user_id`, represented by their **interaction history** over books — what they read (`is_read`), rated (`rating_clean`) and reviewed (`has_review_text`) — as captured in the canonical `interactions_curated.parquet`. The implemented user-side artifacts are `user_matrix.parquet` (one PCA-space taste vector per user with positives), `user_meta.parquet` (behavior/confidence metadata) and `user_centroids.parquet` (multi-centroid taste modes for users with enough positive history).

A user is **not** modeled as a single genre label. The user profile is a multidimensional taste vector built by aggregating the PCA vectors of the books they engaged with positively. Users with 1–2 positives use shrinkage toward their nearest catalog cluster; users with no history use optional seed books or a diverse accessible sample across macro-clusters.

### What is an item

A book, identified by `book_id`, where **one row = one book vector** in PCA space (`pc_0..pc_172`). That vector is the unit of recommendation and of clustering. Genres are a signal, filter, explanation or diversity control — never the unit of the recommendation model.

### What a useful recommendation means

A short list of books the reader is likely to read and enjoy, that is also:

- **Relevant**: close to the user's taste by interest similarity first (distance in PCA space / shared cluster), not by popularity.
- **Approachable and motivating**: aligned with the product goal of building a reading habit, favoring accessible books rather than only the most popular ones.
- **Diverse yet coherent**: it exploits cross-genre patterns (youthful tone + romance + adventure + light fantasy) instead of locking the reader into one genre.
- **Explainable**: justifiable through its cluster / macro-cluster and genres ("because you liked X, from the same reading neighborhood").

### Target action

The primary action the system tries to provoke is **starting and completing a reading**. In Goodreads data the observable proxy is `is_read = True`, reinforced by a high `rating` and/or a written review. The ultimate objective is **retention of the reading habit**, not a single click or purchase.

| Level | Signal in the data | Role |
|---|---|---|
| Target action | `is_read` (reading) | What we optimize |
| Quality confirmation | high `rating`, `has_review_text` | Reinforcement |
| Engagement / interest | click / open book page | Early proxy (to instrument in the product) |
| Business objective | retention / habit | North-star metric |

## Evaluation & Reading-Habit Metrics

The problem statement optimizes for `Z = sustaining the reading habit`. That target is only meaningful if we can measure it, so this section defines how we measure reading habit and how we evaluate the recommender against it.

### The measurement challenge

"Reading habit" is a **longitudinal, causal** outcome: does a reader keep reading over time because of what we recommended? The Goodreads Dataset Collection (UCSD) is **observational and historical** — it is not a live A/B test, so we cannot measure causal impact directly. What we can do is:

1. Derive **habit proxies** from the temporal and behavioral signals already present in the data.
2. Evaluate the recommender **offline** with a temporal split, so the metric reflects "what the reader actually read next", not just "what they read in the past".
3. Document the **production telemetry** that would be required to measure true causal impact, which is out of scope for the static dataset.

### Features available for habit measurement

These columns already exist in the canonical interaction pipeline (`src/curation/interactions.py`) and are exposed through `interactions_curated.parquet`, `user_features_global.parquet`, `user_meta.parquet` and `user_centroids.parquet`. They are the raw material for any habit metric.

Per-interaction signals:

| Feature | What it captures | Reading-habit signal |
|---|---|---|
| `date_added`, `date_updated` | When each interaction happened | Cadence / frequency and temporal span — the basis of everything |
| `started_at`, `read_at` | Reading start and end (raw `GOODREADS_DATE_COLUMNS`) | Completion of a reading, not just saving it |
| `reading_duration_days`, `has_reading_duration` | Actual reading duration | Effective reading vs. mere intention |
| `engagement_mode` | `want_to_read`, `read_no_rating`, `rating_only`, `review` | Separates intention from completed/stronger engagement |
| `is_read` | Reading completed | Target action (direct proxy) |
| `rating`, `rating_clean`, `has_review_text` | Rated / reviewed | Depth of engagement (writing a review = high commitment) |

Per-user aggregates and user artifacts:

```text
user_features_global.parquet
  user_mean_rating, user_rating_std, user_rating_count, user_rating_bias
  read_or_rated_count, valid

user_matrix.parquet
  user_id + pc_0..pc_172
  baseline taste vector = mean PCA vector of positives:
  is_read == True AND rating_clean >= 4

user_meta.parquet
  positive_count, interaction_count, review_count, want_to_read_count
  user_rating_bias, category_count, last_date_added, is_cold_start

user_centroids.parquet
  user_id, centroid_id, n_books, weight, centroid_weight, pc_0..pc_172
  multi-centroid taste modes; centroid_weight uses rating + review + reading duration
```

### Derived habit metrics

These are **not yet stored as columns**; they are computed from the `date_*` fields and the aggregates above. They are the operational definition of "reading habit" per user:

```text
active_span_days   = last_interaction_date − first_interaction_date
reading_frequency  = completed reads / active_span (e.g. readings per month)
activity_recency   = days since last interaction        (churn proxy: higher = more at risk)
completion_rate    = completed reads / interaction_count
reading_breadth    = category_count                     (diversity across genres)
```

A reader "with a habit" shows a wide `active_span`, regular `reading_frequency`, low `activity_recency`, and high `completion_rate`. These per-user values are what we use as the **outcome label** when evaluating the recommender.

### The habit evidence ladder (N0 → N1 → N2)

Reading habit is **not a vague hypothesis** — it *is* the proxy set above. What matures over time is not the *metric* but the **strength of evidence** with which we can claim the recommender influences it. The promise is therefore staged in three levels, and **every claim is tagged with its level** so we never overclaim what we cannot yet prove:

| Level | What we claim | Metric | Evidence type | Available |
|---|---|---|---|---|
| **N0 — Action** | "We predict the next relevant read" | `Recall@k`/`NDCG@k` over future `is_read` (temporal split) | **Predictive** (relevance) | **Today** |
| **N1 — Habit, correlational** | "The model *is associated with* readers who read more, finish more, and read more broadly" | the 5 proxies as a per-user outcome label | **Correlational** | **Today** |
| **N2 — Habit, causal** | "Recommending this way *increases* reading frequency / retention" | the **same** 5 proxies as treatment-vs-control *lift* + live signals (return visits, books finished after a recommendation) | **Causal (A/B)** | **With telemetry** |

> **Separation that avoids the overclaim:** N0 (`Recall@k`) is a **relevance gate** — a necessary condition, *not* the habit itself. A temporal-split `Recall@k` is still a **relevance** metric ("did I hit the next book?"), not a **habit** metric ("does the reader read more over time?"). The habit lives in N1/N2 (the proxies). The north star (sustaining the habit) is **kept**: today we measure it correlationally (N1); with telemetry we measure the same proxies causally (N2). Only the evidence matures, not the goal.

### Evaluation layers (= the three evidence levels)

The three layers below implement the ladder: **Layer 1 → N0**, **Layer 2 → N1**, **Layer 3 → N2**.

**Layer 1 (N0) — Offline evaluation with a temporal split (doable today).**
Use one reproducible global cutoff over valid `date_added` values (on or after `2006-01-01`),
train on interactions at or before that cutoff, and hold out the future. Measure whether the
recommender would have surfaced the available books the reader **actually read later**
(`is_read = True`, ideally with a high rating):

- `Recall@k`, `Precision@k`, `NDCG@k`, `MAP` — relevance of the ranked list.
- `Coverage`, `Novelty`, intra-list `Diversity` — to confirm the model is not just amplifying popular books (enforces the popularity-bias rule from the Business Logic section).

The evaluation mode is `global_historical_snapshot_frozen_representation`. B1/B2, popularity
segments, novelty and average recommendation popularity use only rating evidence available by the
cutoff. Books with a known publication year must already be published; books without a year must
have been observed by the cutoff. Unavailable holdout books are excluded from relevance
denominators.

The temporal split improves the *predictive honesty* of the relevance metric: the question shifts from "did it match past reads?" to "does what it recommends match what the reader keeps reading?". But this is **still relevance (N0), not habit** — hitting the next book is a necessary condition; whether the reader *reads more over time* is Layer 2 (N1). Do not mistake a temporal `Recall@k` for a habit metric.

**Layer 2 (N1) — Habit-proxy evaluation (correlational, today).**
Compare the interest-similarity recommender against a popularity baseline and check whether readers exposed to neighborhood-based recommendations show higher `completion_rate`, `reading_frequency` and `reading_breadth` (`category_count`). In an offline setting this is correlational, not causal — the proxies describe the *user's* habit, not the recommender's *effect* on it (see Honest limitations).

**Layer 3 (N2) — Product telemetry (causal, future / out of scope for the dataset).**
Measuring true causal impact on retention requires instrumenting the live platform: sessions, return visits, books finished *after* a recommendation, click → reading conversion. This is future product work, not available in the static Goodreads dump, and is listed here as an explicit limitation.

### Honest limitations

- The dataset is observational; offline metrics approximate, but do not prove, causal impact on the reading habit.
- **Attribution gap (the big one).** Offline, the habit proxies describe the *user's* habit, not the recommender's *effect* on it — those books were read without the system existing. That is why N1 is correlational **by construction**, and only N2 (telemetry / A/B) closes the gap. This is the reason the evidence ladder exists.
- **`reading_frequency` divides by `active_span`**, so single-interaction users yield `active_span = 0` (division by zero). It needs a floor (e.g. clamp to 1 day, or drop `n = 1`) when computed.
- **`completion_rate` offline is biased by what each user *logged* on Goodreads**, not by what they actually read; in a live product (N2) the signal is clean. The offline proxy is noisier than its telemetry version — same name, different quality.
- `started_at` / `read_at` and `reading_duration_days` can be sparse depending on what each user filled in, so duration-based metrics rely on `has_reading_duration` / `has_reading_duration_rate` to stay honest about coverage.
- **Residual transductive leakage.** PCA, description embeddings and clusters remain frozen from
  the full catalog artifacts. The historical snapshot removes the main operational leakage from
  popularity and availability, but it is not a strict backtest. A strict protocol would rebuild
  the representation and clustering for every cutoff and is outside the current scope.
- The ranking layer implements retrieval, multi-centroid interest scoring, MMR, controlled exploration, mandatory consumed-book exclusions and staged cold start. The temporal evaluation runner and B0/B1/B2 baselines are implemented; measured results remain pending.

## Business Logic

The recommendation logic should model user interests as multidimensional reading tastes rather than as a single genre choice. A platform for reading habits should not be limited to:

```text
If you like fantasy, recommend more fantasy.
```

That behavior is too basic for the product goal. The system should be able to discover patterns such as:

```text
This reader likes books with a youthful tone, romance, adventure, light fantasy
and accessible reading, even when the books do not all belong to the exact same genre.
```

For that reason, genre is treated as one useful signal, not as the full definition of a reader's interests. The hybrid representation also includes semantic description embeddings, ratings metadata, page count, publication information, language, series status and multi-genre structure.

### K-Means Granularity

K-means should be applied at the book level across the full catalog:

```text
one row = one book_id = one book vector
```

The recommended input for k-means is:

```text
data/features/master_feature_matrix.parquet
```

Use `pc_0` through `pc_172` as the clustering features. The clustering unit is `book_id`, not genre. Genres can still be used for interpretation, filtering and personalization, but they should not be the granularity of the main clustering model.

This means clusters represent interest neighborhoods in the book catalog. A cluster may contain books that share tone, audience, themes, accessibility, popularity level or semantic content, even if they cross genre boundaries.

Recommended recommendation flow:

1. Cluster all books using the PCA vectors.
2. Build a user interest vector from books the user actually read and rated positively (`is_read == True AND rating_clean >= 4`).
3. Optionally split broad user histories into `user_centroids` so different taste modes are preserved instead of averaged away.
4. Find the closest clusters, nearest books or nearest user centroids to that user representation.
5. Recommend books from those neighborhoods.
6. Use genre as a filter, explanation or diversity control, not as the only recommendation rule.

### Avoiding Popularity Bias

The system should avoid popularity bias. A recommendation platform should not only surface books with the highest `ratings_count`, `text_reviews_count` or `average_rating`, because that would make popular books even more visible and reduce discovery for users with specific or emerging interests.

Popularity signals are still useful, but they should be controlled:

- `ratings_count` and `text_reviews_count` are transformed with `log1p` to reduce extreme outlier influence.
- Numeric, binary and embedding blocks are standardized and weighted before PCA so one feature family does not dominate only because it has more columns.
- Ranking eligibility is technical (`book_id`, title, PCA vector and cluster must be valid);
  `ratings_count` does not filter or order books.
- Popularity measures exposure through dynamic catalog segments: `tail` (≤ p25), `mid` and
  `head` (≥ p90).
- The normal interest ranking combines semantic MMR with an explicit genre-overlap penalty and a
  small accessibility tie-break based on `num_pages`.
- Exploration slots accept only relevant `tail`/`mid` books outside the retrieved neighborhood;
  if none passes the relevance floor, normal interest ranking fills the slots.

With the current catalog, `ratings_count` has minimum 49, p25 436 and p90 6,017. The previous
`ratings_count >= 5` rule retained all 108,227 books and therefore was not a meaningful gate.
Discovery in v1 means improving measurable exposure within this curated catalog, not solving
coverage of books outside the dataset or item cold start.

## Dataset

Source: [Goodreads Dataset Collection from UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html)

Selected subset:

- 5 literary genres: fantasy/paranormal, mystery/thriller/crime, history/biography, young adult and romance.
- Book metadata JSON files.
- User interaction and review data.
- Curated book-level artifacts by genre.

External project artifacts:

- [Cleaned and reduced data by genre](https://drive.google.com/drive/folders/1un4RNi8W0dvh7cRwx_0Ovgmb9Q4nNh7X?usp=drive_link)
- [books_master](https://drive.google.com/drive/folders/17vpKc3Q4OvQtRkkvOc4GL8sTpjJB-7bv)

## Repository Structure

```text
project/
  data/
    raw/
    interim/
    processed/
    features/
    embeddings/
  notebooks/
    cleaning/
    eda/
    processing/
    reduction/
  src/
    config.py
    merge_master.py
    reduction/
    utils/
  tests/
  README.md
  requirements.txt
```

## Pipeline

The main flow is:

```text
clean -> reduce -> curation -> merge -> feature matrix -> PCA
```

The `clean` and `reduce` phases are handled by genre-specific notebooks. The book curation expected input for the master pipeline is, per category:

```text
data/processed/<category>/books_curated.parquet
```

The categories used by the master flow are:

```text
fantasy_paranormal
mystery_thriller_crime
history_biography
young_adult
romance
```

Some legacy processed paths are also supported by `src/merge_master.py`, such as `data/processed/fantasy` and `data/processed/history`.

The **interaction** side is no longer per-genre. `src/curation/interactions.py` rebuilds a single global, deduplicated artifact from the five raw dumps and writes:

```text
data/processed/interactions_curated.parquet   # canonical, cross-category, deduplicated, no review text
data/processed/review_texts.parquet           # review text, joinable by interaction_key
data/processed/user_features_global.parquet   # global per-user stats + bias + K-core valid flag
```

Schema of the canonical `interactions_curated.parquet`: `interaction_key` (uint64, stable hash of `review_id` or `user_id|book_id`), `user_id`, `book_id`, `review_id`, `is_read`, `rating_clean` (1–5 or NA), `rating_missing`, `has_review_text`, `review_text_length`, `reading_duration_days`, `engagement_mode` (`want_to_read` / `read_no_rating` / `rating_only` / `review`), `is_want_to_read`, `interaction_weight`, `user_rating_bias` (global), the date columns, and `source_category_count` only when built with `--with-source-category-count`. K-core and `user_rating_bias` are **global** (cross-category); the implicit `want_to_read` / `read_no_rating` layer is recovered and kept. The legacy per-genre `data/processed/<category>/interactions_curated.parquet` files are deprecated (kept only as a historical backup); per-category `interactions_view.parquet` files are derived for EDA only.

## Running the Pipeline

Run commands from the repository root. With `python -m`, use the module name, not the filename with `.py`.

Build the master book table:

```bash
env/bin/python -m src.merge_master
```

Build the feature matrix, PCA model and metadata:

```bash
env/bin/python -m src.reduction.build_master_feature_matrix
```

Build the global deduplicated interactions artifact (requires `books_master.parquet`; add `--max-rows-per-file 200000 --skip-views` for a fast dry-run):

```bash
env/bin/python -m src.curation.interactions
```

Build the baseline user taste matrix and user metadata:

```bash
env/bin/python -m src.reduction.build_user_matrix
```

Build multi-centroid user taste modes:

```bash
env/bin/python -m src.reduction.build_user_centroids
```

Run temporal evaluation over a bounded user cohort:

```bash
env/bin/python -m src.reduction.evaluate_recommender --max-users 1000 --k 5 10 20
```

The runner selects only globally `valid` users, keeps their complete histories, and reports
relevance, MAP, diversity, exposure and model slot metrics for every requested `k`. Pass
`--cutoff YYYY-MM-DD` for an explicit snapshot; otherwise `--train-fraction` selects the global
date percentile from the bounded cohort.

Run tests:

```bash
env/bin/python -m pytest
```

## Master Merge

`src/merge_master.py` builds:

```text
data/processed/books_master.parquet
```

For each genre, the script loads `books_curated.parquet`, adds five genre flags and concatenates all genre datasets:

```text
genre_fantasy
genre_mystery
genre_history
genre_ya
genre_romance
```

Because a book can appear in more than one genre, rows are deduplicated by `book_id`.

Aggregation rules:

- `genre_*`: use `max`, so a flag is active if the book appeared in that genre at least once.
- Other columns: use `first`, preserving the first available value found for the book.

Then:

```text
genre_count = genre_fantasy + genre_mystery + genre_history + genre_ya + genre_romance
```

The final master table keeps:

```text
book_id
title
description
series
language_code
average_rating
ratings_count
text_reviews_count
num_pages
publication_year
author_count
genre_fantasy
genre_mystery
genre_history
genre_ya
genre_romance
genre_count
```

## Current Artifacts

| Artifact | Rows | Columns | Description |
|---|---:|---:|---|
| `data/processed/books_master.parquet` | 108,227 | 17 | Master book table, one row per `book_id`. |
| `data/features/master_feature_matrix.parquet` | 108,227 | 174 | PCA-reduced matrix: `book_id` + 173 principal components. |
| `data/features/master_pca_meta.json` | 1 JSON | 19 keys | PCA metadata, block definitions, weights and diagnostics. |
| `data/features/master_pca_model.joblib` | 1 model bundle | n/a | PCA, scalers, medians, language categories and block metadata. |
| `data/embeddings/description_embeddings.parquet` | 108,227 | 257 | Cached description embeddings: `book_id` + `emb_0` to `emb_255`. |

Last validated `books_master.parquet` result:

```text
Shape: (108227, 17)
File size: 63.76 MB
Duplicate book_id values: 0
Rows without an active genre: 0
Rows where genre_count disagrees with genre_* flags: 0
Negative values in count columns: 0
```

Genre counts:

| Genre flag | Books |
|---|---:|
| `genre_fantasy` | 31,281 |
| `genre_mystery` | 23,644 |
| `genre_history` | 23,360 |
| `genre_ya` | 12,314 |
| `genre_romance` | 40,530 |

The sum of genre flags is 131,129, greater than the number of books because 22,723 books appear in more than one genre.

`genre_count` distribution:

| `genre_count` | Books |
|---:|---:|
| 1 | 85,504 |
| 2 | 22,544 |
| 3 | 179 |

## Data Dictionary

`books_master.parquet` is the source of truth for feature engineering.

| Column | Type | Nulls | Description | Feature use |
|---|---|---:|---|---|
| `book_id` | `string` | 0 | Goodreads book identifier. | Traceability key; not used as a numeric PCA feature. |
| `title` | `str` | 0 | Book title. | Fallback text for embeddings if `description` is empty. |
| `description` | `str` | 0 | Book description; missing descriptions are empty strings. | Main text source for semantic embeddings. |
| `series` | `int64` | 0 | `0/1` indicator for whether the book belongs to a series. | Binary block. |
| `language_code` | `str` | 0 | Normalized language code. | One-hot encoded in the binary/categorical block. |
| `average_rating` | `float64` | 0 | Global average book rating. | Numeric block. |
| `ratings_count` | `int64` | 0 | Number of ratings. | `log1p` transformed as `log_ratings_count`. |
| `text_reviews_count` | `int64` | 0 | Number of text reviews. | `log1p` transformed as `log_text_reviews_count`. |
| `num_pages` | `float64` | 18,903 | Number of pages. | Median-imputed with `num_pages_missing` flag. |
| `publication_year` | `float64` | 21,756 | Publication year. | Median-imputed with `publication_year_missing` flag. |
| `author_count` | `int64` | 0 | Number of authors associated with the book. | Numeric block. |
| `genre_fantasy` | `int64` | 0 | Fantasy/paranormal flag. | Binary block. |
| `genre_mystery` | `int64` | 0 | Mystery/thriller/crime flag. | Binary block. |
| `genre_history` | `int64` | 0 | History/biography flag. | Binary block. |
| `genre_ya` | `int64` | 0 | Young adult flag. | Binary block. |
| `genre_romance` | `int64` | 0 | Romance flag. | Binary block. |
| `genre_count` | `int64` | 0 | Number of active genre flags. | Numeric block. |

Observed numeric ranges:

| Column | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|
| `series` | 0 | 1 | 0.7044 | 1 |
| `average_rating` | 1.28 | 3.94 | 3.9313 | 4.90 |
| `ratings_count` | 49 | 840 | 4,656.9474 | 4,899,965 |
| `text_reviews_count` | 1 | 83 | 287.6734 | 142,645 |
| `num_pages` | 0 | 320 | 324.2279 | 14,777 |
| `publication_year` | 1812 | 2011 | 2008.8897 | 2021 |
| `author_count` | 1 | 1 | 1.2409 | 51 |
| `genre_count` | 1 | 1 | 1.2116 | 3 |

Language distribution:

| `language_code` | Books |
|---|---:|
| `eng` | 81,187 |
| `en-US` | 18,942 |
| `en-GB` | 7,022 |
| `en-CA` | 1,076 |

Only these four language codes are present in the current export, so `language_code_other` has no active rows.

Additional useful checks:

```text
series = 1 in 76,234 books and 0 in 31,993 books
author_count minimum = 1
author_count median = 1
author_count mean = 1.2409
author_count maximum = 51
num_pages null rate = 17.47%
publication_year null rate = 20.10%
```

## Feature Matrix Construction

`src/reduction/build_master_feature_matrix.py` builds the final representation in three blocks:

```text
numeric
binary
embeddings
```

Before feature construction, the script validates that `books_master.parquet` exists, contains all required columns and has no duplicate `book_id` values.

### Numeric Block

Dimension:

```text
9 columns
```

Features:

| Feature | Source | Transformation |
|---|---|---|
| `average_rating` | `books_master.average_rating` | Numeric conversion; median fallback if needed. |
| `log_ratings_count` | `books_master.ratings_count` | `log1p(ratings_count)`, after clipping negatives to `0`. |
| `log_text_reviews_count` | `books_master.text_reviews_count` | `log1p(text_reviews_count)`, after clipping negatives to `0`. |
| `num_pages` | `books_master.num_pages` | Median imputation. |
| `num_pages_missing` | `books_master.num_pages` | `1` if missing, else `0`. |
| `publication_year` | `books_master.publication_year` | Median imputation. |
| `publication_year_missing` | `books_master.publication_year` | `1` if missing, else `0`. |
| `author_count` | `books_master.author_count` | Numeric conversion and median fallback. |
| `genre_count` | `books_master.genre_count` | Numeric conversion and median fallback. |

The missingness flags are intentional: missing page count or publication year can reflect metadata quality, edition type or registration patterns.

`ratings_count` and `text_reviews_count` use:

```text
log1p(x) = log(1 + x)
```

This preserves ordering while reducing the influence of very large popularity outliers.

### Binary and Categorical Block

Dimension:

```text
11 columns
```

Features:

```text
series
genre_fantasy
genre_mystery
genre_history
genre_ya
genre_romance
language_code_eng
language_code_en_us
language_code_en_gb
language_code_en_ca
language_code_other
```

Genre and series flags are clipped to `[0, 1]`. `language_code` is one-hot encoded using the most frequent categories, with a fallback `language_code_other` bucket.

### Embedding Block

Dimension:

```text
256 columns
```

Source:

```text
data/embeddings/description_embeddings.parquet
```

Embedding model:

```text
google/embeddinggemma-300m
```

Columns:

```text
emb_0, emb_1, ..., emb_255
```

Text used for each embedding:

1. `description`
2. `title`, if description is empty
3. `[no description]`, if both are missing

The embedding cache is incremental. Existing embeddings are reused, and only missing `book_id` values are computed.

## Standardization and Block Weighting

Each block is standardized separately with `StandardScaler`. Then each standardized block is multiplied by:

```text
1 / sqrt(block_dimension)
```

Weights in the current export:

| Block | Columns | Weight |
|---|---:|---:|
| `numeric` | 9 | 0.3333333333333333 |
| `binary` | 11 | 0.30151134457776363 |
| `embeddings` | 256 | 0.0625 |

This prevents the 256-dimensional embedding block from dominating the PCA simply because it has many more columns.

The pre-PCA matrix has:

```text
9 + 11 + 256 = 276 columns
```

## PCA

The PCA is fit with:

```text
PCA(n_components=0.95, svd_solver="full")
```

The number of components is therefore selected automatically to retain at least 95% of the variance.

Current PCA result:

```text
Input features before PCA: 276
Retained components: 173
Explained variance sum: 0.9501428604125977
Final feature matrix columns: book_id + pc_0 ... pc_172
```

The reduced matrix has 174 columns:

```text
174 columns = 1 book_id column + 173 pc_* columns
```

This removes 103 dimensions, a reduction of about 37.3% from the pre-PCA feature matrix while retaining about 95.01% of variance.

### Explained Variance Thresholds

| Target cumulative variance | Components needed | Actual cumulative variance |
|---:|---:|---:|
| 50% | 10 | 52.7610% |
| 60% | 13 | 61.9781% |
| 70% | 21 | 70.0497% |
| 75% | 36 | 75.0639% |
| 80% | 58 | 80.1415% |
| 85% | 85 | 85.0330% |
| 90% | 122 | 90.0970% |
| 95% | 173 | 95.0143% |

The first 10 components explain just over half of the variance. Moving from 90% to 95% requires 51 additional components, which shows a long tail of smaller signals.

### First 20 Components

| Component | Individual variance | Cumulative variance |
|---:|---:|---:|
| `pc_0` | 7.9367% | 7.9367% |
| `pc_1` | 7.1026% | 15.0393% |
| `pc_2` | 6.8006% | 21.8399% |
| `pc_3` | 5.9972% | 27.8370% |
| `pc_4` | 5.2199% | 33.0569% |
| `pc_5` | 4.6836% | 37.7405% |
| `pc_6` | 4.0353% | 41.7758% |
| `pc_7` | 3.9279% | 45.7037% |
| `pc_8` | 3.6351% | 49.3388% |
| `pc_9` | 3.4222% | 52.7610% |
| `pc_10` | 3.2342% | 55.9952% |
| `pc_11` | 3.1903% | 59.1856% |
| `pc_12` | 2.7925% | 61.9781% |
| `pc_13` | 2.7093% | 64.6873% |
| `pc_14` | 1.6840% | 66.3713% |
| `pc_15` | 0.8283% | 67.1996% |
| `pc_16` | 0.6819% | 67.8816% |
| `pc_17` | 0.6123% | 68.4939% |
| `pc_18` | 0.5816% | 69.0755% |
| `pc_19` | 0.5045% | 69.5800% |

There is a clear drop after `pc_14`. From `pc_15` onward, individual variance falls below 1% and embedding-dominated components become more common.

## PCA Block Diagnostics

Dominant block counts across retained components:

| Dominant block | Components | Cumulative explained variance from those components | Mean variance per component |
|---|---:|---:|---:|
| `numeric` | 10 | 46.3688% | 4.6369% |
| `binary` | 5 | 20.0025% | 4.0005% |
| `embeddings` | 158 | 28.6430% | 0.1813% |

Interpretation:

- Numeric and categorical metadata dominate a small number of high-variance components.
- Embeddings dominate many more components, but each explains a small amount of variance.
- The block weighting worked: embeddings do not dominate the first five components.

```text
embedding_dominated_first_5_count = 0
```

The first embedding-dominated component is:

```text
pc_15
```

## Interpreting Early Components

PCA component signs are arbitrary: an axis can be multiplied by `-1` without changing the model. Interpret which variables move together or against each other, not the sign alone.

Observed early-component readings:

- `pc_0`: popularity and interaction. Strongly associated with `log_text_reviews_count`, `log_ratings_count`, YA/fantasy and multigenre books.
- `pc_1`: romance, recent publication year and metadata missingness versus history/mystery and longer books.
- `pc_2`: language and missingness, especially `eng` versus `en-US`, plus publication/page missingness.
- `pc_3`: `eng` versus regional language variants, with fantasy/YA and multigenre structure.
- `pc_4`: average rating, page count, fantasy and `author_count`.
- `pc_5`: genre separation, especially history/romance versus mystery/fantasy/YA.
- `pc_7`: `author_count`, confirming that author count is now an informative feature rather than a constant.
- `pc_10`: multigenre, publication year and mystery/romance structure.

## Semantic Spot Checks

`master_pca_meta.json` includes small cosine-distance spot checks over the reduced matrix.

| Genre | Same-genre distance | Different-genre distance |
|---|---:|---:|
| `genre_fantasy` | 0.5533 | 0.9384 |
| `genre_mystery` | 0.9578 | 0.9384 |
| `genre_history` | 0.9870 | 0.4532 |

These are quick diagnostics, not a robust model evaluation. Fantasy behaves as expected in this sample, but mystery and history do not. Better evaluation options include recall@k by genre, intra/inter-genre distance averages or clustering purity.

## Main Insights

- The master table is structurally healthy: no duplicated `book_id`, no genre-count mismatches and no rows without a genre.
- `num_pages` and `publication_year` intentionally preserve missing values so the feature matrix can encode missingness as signal.
- `series` and `author_count` contain useful variation in the current export.
- The final representation is hybrid: it includes popularity, metadata, categorical genre/language signals and semantic text embeddings.
- PCA reduces 276 pre-PCA features to 173 components while retaining 95.0143% of variance.
- The strongest early PCA axes are tabular, not semantic: popularity, language, genre, author count, publication metadata and missingness.
- Embeddings provide a broad semantic tail: they dominate 158 retained components, but those components have low individual variance.
- `language_code_other` is inactive in the current export because only `eng`, `en-US`, `en-GB` and `en-CA` appear.

## Recommended Use

- Use `book_id` only as a key.
- Use `pc_0` through `pc_172` as the vector representation for similarity, clustering or downstream models.
- Join back to `books_master.parquet` by `book_id` for interpretation.
- Use `data/features/master_pca_model.joblib` to transform future books consistently. It stores scalers, medians, language categories, block weights and the fitted PCA model.
- If the goal is more semantic recommendation and less metadata-driven similarity, compare this PCA matrix against raw embeddings or experiment with different block weights.

## Caveats

- PCA is linear and does not capture complex nonlinear relationships between metadata, genres and embeddings.
- Component signs are arbitrary.
- Components are latent dimensions, not directly interpretable original variables.
- The semantic spot checks are too small to validate recommendation quality by themselves.
- Any changes to curation, embeddings, block weighting or master merge logic can change the PCA geometry.
