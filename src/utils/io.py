from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


def compressed_size_gb(path: Path) -> float:
    return path.stat().st_size / 1024**3


def iter_jsonl_records(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                yield json.loads(line)


def read_jsonl_sample(path: Path, nrows: int = 50_000) -> pd.DataFrame:
    return pd.DataFrame.from_records(iter_jsonl_records(path, limit=nrows))


def read_jsonl_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    yield from pd.read_json(
        path,
        lines=True,
        compression="gzip",
        chunksize=chunksize,
    )


def read_parquet_chunks(
    path: Path,
    chunksize: int,
    columns: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=chunksize, columns=columns):
        yield batch.to_pandas()


def safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        for child in path.iterdir():
            remove_path(child)
        path.rmdir()
    else:
        path.unlink()
