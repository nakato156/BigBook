from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ScaledBlocks:
    matrix: np.ndarray
    scalers: dict[str, StandardScaler]
    block_dims: dict[str, int]
    block_weights: dict[str, float]
    block_slices: dict[str, tuple[int, int]]
    block_columns: dict[str, list[str]]


def _as_float_matrix(block: pd.DataFrame | np.ndarray) -> np.ndarray:
    values = block.to_numpy() if isinstance(block, pd.DataFrame) else block
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D block matrix, got shape {values.shape}.")
    if values.shape[1] == 0:
        raise ValueError("Feature blocks must contain at least one column.")
    if not np.isfinite(values).all():
        raise ValueError("Feature blocks cannot contain NaN or infinite values.")
    return values


def standardize_and_weight_blocks(
    blocks: Mapping[str, pd.DataFrame | np.ndarray],
) -> ScaledBlocks:
    weighted_blocks = []
    scalers: dict[str, StandardScaler] = {}
    block_dims: dict[str, int] = {}
    block_weights: dict[str, float] = {}
    block_slices: dict[str, tuple[int, int]] = {}
    block_columns: dict[str, list[str]] = {}
    start = 0
    row_count: int | None = None

    for name, block in blocks.items():
        values = _as_float_matrix(block)
        if row_count is None:
            row_count = values.shape[0]
        elif values.shape[0] != row_count:
            raise ValueError("All feature blocks must have the same number of rows.")

        columns = list(block.columns) if isinstance(block, pd.DataFrame) else [f"{name}_{idx}" for idx in range(values.shape[1])]
        scaler = StandardScaler()
        standardized = scaler.fit_transform(values).astype(np.float32)
        dim = standardized.shape[1]
        weight = 1.0 / sqrt(dim)
        weighted = (standardized * weight).astype(np.float32)

        end = start + dim
        weighted_blocks.append(weighted)
        scalers[name] = scaler
        block_dims[name] = dim
        block_weights[name] = weight
        block_slices[name] = (start, end)
        block_columns[name] = columns
        start = end

    if not weighted_blocks:
        raise ValueError("At least one feature block is required.")

    return ScaledBlocks(
        matrix=np.hstack(weighted_blocks).astype(np.float32),
        scalers=scalers,
        block_dims=block_dims,
        block_weights=block_weights,
        block_slices=block_slices,
        block_columns=block_columns,
    )


def fit_variance_pca(
    matrix: np.ndarray,
    *,
    variance_threshold: float = 0.95,
) -> tuple[np.ndarray, PCA]:
    if not 0.0 < variance_threshold < 1.0:
        raise ValueError("variance_threshold must be between 0 and 1.")
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    transformed = pca.fit_transform(np.asarray(matrix, dtype=np.float32)).astype(np.float32)
    return transformed, pca


def component_block_norm_shares(
    pca: PCA,
    block_slices: Mapping[str, tuple[int, int]],
    *,
    first_n: int = 5,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    components = pca.components_[:first_n]
    for component_idx, component in enumerate(components):
        norms = {
            name: float(np.linalg.norm(component[start:end]))
            for name, (start, end) in block_slices.items()
        }
        total_norm = sum(norms.values())
        shares = {
            name: (norm / total_norm if total_norm else 0.0)
            for name, norm in norms.items()
        }
        dominant_block = max(shares, key=shares.get)
        summaries.append(
            {
                "component": component_idx,
                "block_norm_shares": shares,
                "dominant_block": dominant_block,
                "dominant_share": shares[dominant_block],
            }
        )
    return summaries


def count_embedding_dominated_components(
    pca: PCA,
    block_slices: Mapping[str, tuple[int, int]],
    *,
    embedding_block_name: str = "embeddings",
    first_n: int = 5,
    threshold: float = 0.5,
) -> int:
    summaries = component_block_norm_shares(pca, block_slices, first_n=first_n)
    return sum(
        1
        for summary in summaries
        if summary["block_norm_shares"].get(embedding_block_name, 0.0) > threshold
    )
