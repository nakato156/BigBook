from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.utils.io import safe_write_parquet


EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"
EMBEDDING_DIM = 256
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "description_embeddings.parquet"
NO_DESCRIPTION_TEXT = "[no description]"


def embedding_columns(dim: int = EMBEDDING_DIM) -> list[str]:
    return [f"emb_{idx}" for idx in range(dim)]


def clean_description_texts(master: pd.DataFrame) -> pd.Series:
    descriptions = master["description"].fillna("").astype(str).str.strip()
    titles = master["title"].fillna("").astype(str).str.strip()
    cleaned = descriptions.mask(descriptions.eq(""), titles)
    cleaned = cleaned.mask(cleaned.eq(""), NO_DESCRIPTION_TEXT)
    return cleaned


def _get_huggingface_token() -> str | None:
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(env_name)
        if token:
            return token

    try:
        from huggingface_hub import get_token
    except ImportError:
        return None

    return get_token()


def _require_huggingface_auth(model_name: str) -> None:
    if _get_huggingface_token():
        return

    raise RuntimeError(
        f"{model_name} is a gated HuggingFace model. Set HF_TOKEN or run "
        "`huggingface-cli login` with an account that has accepted the model license."
    )


def _load_embedding_model(model_name: str):
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Embedding generation requires torch and sentence-transformers. "
            "Install them with `pip install -r requirements.txt`."
        ) from exc

    _require_huggingface_auth(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model {model_name} on {device}...")
    return SentenceTransformer(model_name, device=device), device


def _validate_cache_schema(cache: pd.DataFrame, dim: int) -> pd.DataFrame:
    expected_columns = ["book_id", *embedding_columns(dim)]
    missing_columns = [column for column in expected_columns if column not in cache.columns]
    if missing_columns:
        raise ValueError(
            "Embedding cache has an invalid schema. Missing columns: "
            + ", ".join(missing_columns[:10])
        )
    if cache["book_id"].duplicated().any():
        raise ValueError("Embedding cache contains duplicated book_id values.")

    out = cache[expected_columns].copy()
    out["book_id"] = out["book_id"].astype(str)
    for column in embedding_columns(dim):
        out[column] = pd.to_numeric(out[column], errors="raise").astype("float32")
    return out


def _read_cache(cache_path: Path, dim: int) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=["book_id", *embedding_columns(dim)])
    return _validate_cache_schema(pd.read_parquet(cache_path), dim)


def _encode_texts(
    texts: list[str],
    *,
    model_name: str,
    dim: int,
    batch_size: int,
) -> np.ndarray:
    model, _device = _load_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if embeddings.ndim != 2 or embeddings.shape[1] < dim:
        raise ValueError(
            f"Expected embeddings with at least {dim} dimensions, got shape {embeddings.shape}."
        )
    return np.asarray(embeddings[:, :dim], dtype=np.float32)


def load_or_create_description_embeddings(
    master: pd.DataFrame,
    *,
    cache_path: Path = DEFAULT_EMBEDDINGS_PATH,
    model_name: str = EMBEDDING_MODEL_NAME,
    dim: int = EMBEDDING_DIM,
    batch_size: int = 64,
) -> pd.DataFrame:
    if master["book_id"].duplicated().any():
        raise ValueError("books_master must contain one row per book_id.")

    required = master[["book_id", "title", "description"]].copy()
    required["book_id"] = required["book_id"].astype(str)
    cache = _read_cache(cache_path, dim)

    cached_ids = set(cache["book_id"])
    missing_mask = ~required["book_id"].isin(cached_ids)
    missing = required.loc[missing_mask].copy()

    if not missing.empty:
        print(f"Computing embeddings for {len(missing):,} missing descriptions...")
        missing["description"] = clean_description_texts(missing)
        encoded = _encode_texts(
            missing["description"].tolist(),
            model_name=model_name,
            dim=dim,
            batch_size=batch_size,
        )
        new_cache = pd.DataFrame(encoded, columns=embedding_columns(dim))
        new_cache.insert(0, "book_id", missing["book_id"].to_numpy())
        cache = pd.concat([cache, new_cache], ignore_index=True)
        cache = _validate_cache_schema(cache, dim)
        safe_write_parquet(cache, cache_path)
        print(f"Embedding cache updated: {cache_path}")
    else:
        print(f"Embedding cache covers all {len(required):,} books: {cache_path}")

    ordered = required[["book_id"]].merge(cache, on="book_id", how="left", validate="one_to_one")
    if ordered[embedding_columns(dim)].isna().any().any():
        raise ValueError("Embedding cache lookup failed for at least one required book_id.")
    return ordered
