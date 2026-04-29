# Book Recommendation System

Course project for Big Data


## Dataset

Source: [Goodreads Dataset Collection from the University of California San Diego (UCSD)](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html)

Selected subset:
- 5 literary genres
- Metadata JSON files
- User interaction and review data

[Cleaned and reduced data by genre](https://drive.google.com/drive/folders/1un4RNi8W0dvh7cRwx_0Ovgmb9Q4nNh7X?usp=drive_link)

[books_master](https://drive.google.com/drive/folders/17vpKc3Q4OvQtRkkvOc4GL8sTpjJB-7bv) -> processed book dataset for feature matrix + PCA. null handling is needed

## Pipeline Flow

`clean -> reduce -> curation -> merge`

- `clean` and `reduce` write category artifacts to `data/interim/<category>/`
- `curation` writes curated artifacts to `data/processed/<category>/`
- `merge` builds `data/processed/books_master.parquet` from curated book outputs

## Repo Structure

```
project/
  data/
    raw/
    interim/
    processed/
  notebooks/
  src/
  reports/
  artifacts/
  README.md
  requirements.txt or pyproject.toml
```
