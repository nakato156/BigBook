"""User-kNN collaborative signal — computed on-the-fly, no FAISS/ANN.

Second, independent collaborative signal (alternative to the item-item co-occurrence in
``src/reduction/build_item_cooccurrence.py``, Fase 4a): for each evaluated user, find their
most similar neighbours (cosine, in the **taste subspace** — same ``n_taste`` dimensions
that drop the tabular early axes ``pc_0..pc_5``, see A1 in ``src/reduction/recommend.py``)
among **all** users in the catalog, and use the books those neighbours read (that the
evaluated user has not) as a candidate signal.

Only ~2000 users are evaluated against ~699k total users (~2.3e11 flops total), so an
exact, chunked matmul with BLAS is sufficient — minutes, not hours. FAISS/ANN is
deliberately not used at this scale.

**Operational cost note (on-the-fly vs. precomputed):** this module recomputes a user's
neighbours from scratch every time they are evaluated, unlike the item co-occurrence
signal (Fase 4a), which is persisted once to parquet and reused for free thereafter. This
on-the-fly approach does not scale the same way in production: if this variant wins the
A/B (Fase 4c), Fase 4d will need to materialize a batch artifact instead of recomputing
user-kNN online.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.reduction.recommend import consumed_books_for_users

__all__ = ["compute_user_knn_scores", "neighbor_unread_books"]


def compute_user_knn_scores(
    eval_taste_norm: np.ndarray,
    all_taste_norm: np.ndarray,
    eval_user_ids: np.ndarray,
    all_user_ids: np.ndarray,
    k: int = 50,
    chunk_size: int = 50_000,
) -> dict[str, list[tuple[str, float]]]:
    """Top-k neighbours (cosine, taste subspace) per evaluated user, excluding themselves.

    Returns ``{eval_user_id: [(neighbor_user_id, cosine_similarity), ...]}`` sorted by
    descending similarity, length <= k per user (can be smaller if ``n_all - 1 < k``).

    Exact, not approximate (no FAISS/ANN — see module docstring for why that's fine at
    this scale): computed by chunking ``all_taste_norm`` and incrementally merging a
    running top-k per evaluated user across chunks.
    """
    eval_taste_norm = np.asarray(eval_taste_norm, dtype=np.float64)
    all_taste_norm = np.asarray(all_taste_norm, dtype=np.float64)
    eval_user_ids = np.asarray(eval_user_ids, dtype=str)
    all_user_ids = np.asarray(all_user_ids, dtype=str)

    n_eval = eval_taste_norm.shape[0]
    n_all = all_taste_norm.shape[0]

    # Running top-k accumulators per evaluated user (start empty, grow up to k per chunk).
    best_sims = np.full((n_eval, 0), -np.inf, dtype=np.float64)
    best_ids = np.empty((n_eval, 0), dtype=all_user_ids.dtype)

    for start in range(0, n_all, chunk_size):
        stop = min(start + chunk_size, n_all)
        chunk_taste = all_taste_norm[start:stop]
        chunk_ids = all_user_ids[start:stop]

        # (n_eval, chunk_len) cosine similarities (rows already L2-normalized upstream).
        sims = eval_taste_norm @ chunk_taste.T

        # Vectorized self-exclusion: no assumption about where eval users sit in
        # all_user_ids (not necessarily row 0, not necessarily aligned 1:1).
        self_mask = eval_user_ids[:, None] == chunk_ids[None, :]
        sims = np.where(self_mask, -np.inf, sims)

        # Merge this chunk with the running top-k, then keep only the best k per row.
        merged_sims = np.concatenate([best_sims, sims], axis=1)
        merged_ids = np.concatenate(
            [best_ids, np.broadcast_to(chunk_ids, (n_eval, chunk_ids.shape[0]))],
            axis=1,
        )

        width = merged_sims.shape[1]
        keep = min(k, width)
        if keep < width:
            # argpartition: keep the k largest per row (unordered among themselves).
            part = np.argpartition(-merged_sims, keep - 1, axis=1)[:, :keep]
        else:
            part = np.broadcast_to(np.arange(width), (n_eval, width))

        row_idx = np.arange(n_eval)[:, None]
        best_sims = merged_sims[row_idx, part]
        best_ids = merged_ids[row_idx, part]

    # Final descending sort per row.
    order = np.argsort(-best_sims, axis=1)
    row_idx = np.arange(n_eval)[:, None]
    best_sims = best_sims[row_idx, order]
    best_ids = best_ids[row_idx, order]

    result: dict[str, list[tuple[str, float]]] = {}
    for i, eval_id in enumerate(eval_user_ids):
        valid = np.isfinite(best_sims[i])
        neighbors = list(zip(best_ids[i][valid].tolist(), best_sims[i][valid].astype(float).tolist()))
        result[str(eval_id)] = neighbors
    return result


def neighbor_unread_books(
    neighbor_scores: dict[str, list[tuple[str, float]]],
    consumed_by_eval_user: dict[str, set[str]],
    interactions_path: Path,
) -> dict[str, dict[str, float]]:
    """Books read by neighbours (and not by the evaluated user) -> aggregated score.

    A book's score is the sum of similarities of neighbours (with similarity > 0) who
    read it. Reuses ``consumed_books_for_users`` (parquet predicate pushdown
    ``user_id IN (...)``) with a **single batched call** over the union of every
    ``neighbor_user_id`` across all evaluated users — never one call per evaluated user,
    which would defeat the predicate pushdown.
    """
    unique_neighbor_ids: set[str] = set()
    for neighbors in neighbor_scores.values():
        for neighbor_id, _sim in neighbors:
            unique_neighbor_ids.add(str(neighbor_id))

    consumed_by_neighbor = consumed_books_for_users(interactions_path, sorted(unique_neighbor_ids))

    result: dict[str, dict[str, float]] = {}
    for eval_user_id, neighbors in neighbor_scores.items():
        already_consumed = consumed_by_eval_user.get(eval_user_id, set())
        scores: dict[str, float] = {}
        for neighbor_id, sim in neighbors:
            if sim <= 0:
                continue
            neighbor_books = consumed_by_neighbor.get(str(neighbor_id), set())
            for book_id in neighbor_books:
                if book_id in already_consumed:
                    continue
                scores[book_id] = scores.get(book_id, 0.0) + float(sim)
        result[eval_user_id] = scores
    return result
