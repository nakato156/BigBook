"""Tests for the on-the-fly user-kNN collaborative signal (Fase 4b).

Each test pins one property: top-k ordering/truncation, self-exclusion at an arbitrary
position in ``all_user_ids``, chunking-invariance of the incremental top-k merge, and the
aggregation/exclusion/batching contract of ``neighbor_unread_books``.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src.reduction.retrieval import l2_normalize_rows
from src.reduction.user_knn import compute_user_knn_scores, neighbor_unread_books


def _toy_taste_vectors() -> tuple[np.ndarray, np.ndarray]:
    """8 users in a 2D taste subspace, spread around the unit circle so cosine ranks
    are unambiguous and stable across different ``k``/chunk configurations."""
    angles = np.array([0, 10, 20, 35, 90, 140, 200, 260], dtype=np.float64)
    radians = np.deg2rad(angles)
    vecs = np.stack([np.cos(radians), np.sin(radians)], axis=1)
    ids = np.array([f"u{i}" for i in range(len(angles))])
    return l2_normalize_rows(vecs), ids


# --------------------------------------------------------------------------- #
# compute_user_knn_scores
# --------------------------------------------------------------------------- #
def test_neighbors_sorted_descending_and_respect_k() -> None:
    all_taste_norm, all_ids = _toy_taste_vectors()
    eval_taste_norm = all_taste_norm[[0]]  # u0, angle 0
    eval_ids = np.array(["u0"])

    result = compute_user_knn_scores(
        eval_taste_norm, all_taste_norm, eval_ids, all_ids, k=3, chunk_size=100
    )

    neighbors = result["u0"]
    assert len(neighbors) == 3
    sims = [sim for _uid, sim in neighbors]
    assert sims == sorted(sims, reverse=True)
    # u1 (10deg) and u2 (20deg) are the closest to u0 (0deg); u3 (35deg) is next.
    neighbor_ids = [uid for uid, _sim in neighbors]
    assert neighbor_ids == ["u1", "u2", "u3"]


def test_self_exclusion_at_arbitrary_non_aligned_position() -> None:
    # all_user_ids deliberately ordered so the evaluated users' positions are neither
    # row 0 nor aligned 1:1 with their position in eval_user_ids.
    all_ids = np.array(["zz", "u_b", "yy", "xx", "u_a", "ww"])
    rng = np.random.default_rng(0)
    base_vecs = rng.normal(size=(6, 3))
    # Force u_a and u_b to be identical to themselves trivially, but distinct from others,
    # and make sure each eval user's own vector is the best possible match for itself
    # (cosine 1.0) so any self-exclusion bug would surface it as a top neighbor.
    base_vecs[4] = base_vecs[1] = np.array([1.0, 0.0, 0.0])  # u_a (row 4), u_b (row 1)
    all_taste_norm = l2_normalize_rows(base_vecs)

    # eval order intentionally reversed relative to all_user_ids positions.
    eval_ids = np.array(["u_a", "u_b"])
    eval_rows = np.array(
        [np.where(all_ids == "u_a")[0][0], np.where(all_ids == "u_b")[0][0]]
    )
    eval_taste_norm = all_taste_norm[eval_rows]

    result = compute_user_knn_scores(eval_taste_norm, all_taste_norm, eval_ids, all_ids, k=5)

    for eval_id in eval_ids:
        neighbor_ids = [uid for uid, _sim in result[eval_id]]
        assert eval_id not in neighbor_ids

    # u_a and u_b share the exact same vector -> each other's best neighbor (sim ~= 1.0),
    # which would not be the case if self-exclusion accidentally excluded the wrong row.
    assert result["u_a"][0][0] == "u_b"
    assert result["u_b"][0][0] == "u_a"


def test_chunking_does_not_change_result() -> None:
    rng = np.random.default_rng(42)
    n_all = 23
    vecs = rng.normal(size=(n_all, 4))
    all_taste_norm = l2_normalize_rows(vecs)
    all_ids = np.array([f"u{i}" for i in range(n_all)])

    eval_ids = np.array(["u3", "u10", "u17"])
    eval_rows = np.array([3, 10, 17])
    eval_taste_norm = all_taste_norm[eval_rows]

    single_chunk = compute_user_knn_scores(
        eval_taste_norm, all_taste_norm, eval_ids, all_ids, k=5, chunk_size=1_000
    )
    # chunk_size=4 forces >=6 chunks over 23 rows, exercising multi-chunk incremental merge.
    multi_chunk = compute_user_knn_scores(
        eval_taste_norm, all_taste_norm, eval_ids, all_ids, k=5, chunk_size=4
    )

    assert single_chunk.keys() == multi_chunk.keys()
    for eval_id in eval_ids:
        single = single_chunk[eval_id]
        multi = multi_chunk[eval_id]
        assert len(single) == len(multi) == 5
        single_ids = [uid for uid, _sim in single]
        multi_ids = [uid for uid, _sim in multi]
        assert single_ids == multi_ids
        single_sims = np.array([sim for _uid, sim in single])
        multi_sims = np.array([sim for _uid, sim in multi])
        np.testing.assert_allclose(single_sims, multi_sims, atol=1e-12)


def test_k_larger_than_available_candidates_truncates_gracefully() -> None:
    all_taste_norm, all_ids = _toy_taste_vectors()  # 8 users total
    eval_ids = np.array(["u0"])
    eval_taste_norm = all_taste_norm[[0]]

    result = compute_user_knn_scores(
        eval_taste_norm, all_taste_norm, eval_ids, all_ids, k=50, chunk_size=3
    )

    # 8 total users minus the eval user itself -> at most 7 candidates.
    assert len(result["u0"]) == 7


# --------------------------------------------------------------------------- #
# neighbor_unread_books
# --------------------------------------------------------------------------- #
def test_aggregates_similarity_sum_across_multiple_neighbors_sharing_a_book() -> None:
    neighbor_scores = {
        "eval1": [("n1", 0.8), ("n2", 0.5)],
    }
    consumed_by_eval_user = {"eval1": set()}

    def fake_consumed(_path, user_ids):
        assert set(user_ids) == {"n1", "n2"}
        return {"n1": {"bookX"}, "n2": {"bookX", "bookY"}}

    with patch("src.reduction.user_knn.consumed_books_for_users", side_effect=fake_consumed):
        result = neighbor_unread_books(neighbor_scores, consumed_by_eval_user, interactions_path="unused")

    assert result["eval1"]["bookX"] == 0.8 + 0.5
    assert result["eval1"]["bookY"] == 0.5


def test_excludes_already_consumed_and_non_positive_similarity() -> None:
    neighbor_scores = {
        "eval1": [("n1", 0.9), ("n2", -0.3), ("n3", 0.0)],
    }
    consumed_by_eval_user = {"eval1": {"bookAlready"}}

    def fake_consumed(_path, user_ids):
        return {
            "n1": {"bookAlready", "bookNew"},
            "n2": {"bookFromNegative"},
            "n3": {"bookFromZero"},
        }

    with patch("src.reduction.user_knn.consumed_books_for_users", side_effect=fake_consumed):
        result = neighbor_unread_books(neighbor_scores, consumed_by_eval_user, interactions_path="unused")

    assert result["eval1"] == {"bookNew": 0.9}


def test_single_batched_call_to_consumed_books_for_users() -> None:
    neighbor_scores = {
        "eval1": [("n1", 0.8), ("n2", 0.5)],
        "eval2": [("n2", 0.6), ("n3", 0.4)],  # n2 overlaps with eval1's neighbor set
    }
    consumed_by_eval_user = {"eval1": set(), "eval2": set()}

    def fake_consumed(_path, user_ids):
        return {uid: set() for uid in user_ids}

    with patch(
        "src.reduction.user_knn.consumed_books_for_users", side_effect=fake_consumed
    ) as mocked:
        neighbor_unread_books(neighbor_scores, consumed_by_eval_user, interactions_path="unused")

    assert mocked.call_count == 1
    called_user_ids = set(mocked.call_args.args[1])
    assert called_user_ids == {"n1", "n2", "n3"}
