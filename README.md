# Book Recommendation System

Course project for Big Data. The project builds a hybrid book representation for recommendation and analysis by combining Goodreads metadata, genre labels, interaction-derived popularity signals and text embeddings from book descriptions. The final artifact is a PCA-reduced feature matrix with one vector per `book_id`.

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

The `clean`, `reduce` and `curation` phases are handled by genre-specific notebooks. The expected curated inputs for the master pipeline are:

```text
data/processed/<category>/books_curated.parquet
data/processed/<category>/interactions_curated.parquet
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
