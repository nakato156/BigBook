from __future__ import annotations

import math

import numpy as np
import pandas as pd

import src.reduction.build_item_cooccurrence as bic
from src.reduction.build_item_cooccurrence import build_item_cooccurrence


def _item_matrix(book_ids: list[str]) -> pd.DataFrame:
    n = len(book_ids)
    return pd.DataFrame(
        {
            "book_id": book_ids,
            "pc_0": np.arange(n, dtype=np.float32),
            "pc_1": np.arange(n, dtype=np.float32) * 2.0,
        }
    )


def _interactions(rows: list[tuple[str, str, bool, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "book_id", "is_read", "rating_clean"])


def test_pmi_formula_exact_on_hand_verifiable_fixture() -> None:
    # u1,u2,u3: {b0,b1} positive; u4: {b0} only; u5: {b1} only; u6: {b2} only.
    # count(b0)=4, count(b1)=4, co_count(b0,b1)=3, N=6 distinct users with >=1 positive.
    # PMI = log(co_count * N / (count_b0 * count_b1)) = log(3*6/(4*4)) = log(1.125)
    item_matrix = _item_matrix(["b0", "b1", "b2"])
    rows = [
        ("u1", "b0", True, 5.0), ("u1", "b1", True, 5.0),
        ("u2", "b0", True, 4.0), ("u2", "b1", True, 4.0),
        ("u3", "b0", True, 5.0), ("u3", "b1", True, 4.0),
        ("u4", "b0", True, 4.0),
        ("u5", "b1", True, 4.0),
        ("u6", "b2", True, 4.0),
    ]
    result, diag = build_item_cooccurrence(item_matrix, [_interactions(rows)])

    assert diag["n_users_with_positive"] == 6
    pair = result[(result["book_id_a"] == "b0") & (result["book_id_b"] == "b1")]
    assert len(pair) == 1
    assert pair.iloc[0]["co_count"] == 3
    assert math.isclose(pair.iloc[0]["pmi"], math.log(3 * 6 / (4 * 4)), rel_tol=1e-4)
    # b2 never co-occurs with anything -> no row at all involving it.
    assert not ((result["book_id_a"] == "b2") | (result["book_id_b"] == "b2")).any()


def test_min_co_count_filter_drops_low_count_pairs() -> None:
    # b0/b1 co-occur 3 times (passes MIN_CO_COUNT=3); b0/b2 co-occur only 2 times (filtered).
    item_matrix = _item_matrix(["b0", "b1", "b2"])
    rows = [
        ("u1", "b0", True, 5.0), ("u1", "b1", True, 5.0),
        ("u2", "b0", True, 4.0), ("u2", "b1", True, 4.0),
        ("u3", "b0", True, 5.0), ("u3", "b1", True, 4.0),
        ("u4", "b0", True, 4.0), ("u4", "b2", True, 4.0),
        ("u5", "b0", True, 4.0), ("u5", "b2", True, 4.0),
    ]
    result, diag = build_item_cooccurrence(item_matrix, [_interactions(rows)])

    assert diag["n_pairs_total"] == 2  # (b0,b1) and (b0,b2) both seen pre-filter
    assert diag["n_pairs_after_min_count"] == 1
    pairs = set(zip(result["book_id_a"], result["book_id_b"]))
    assert ("b0", "b1") in pairs
    assert ("b0", "b2") not in pairs


def test_positive_floor_clips_negative_raw_pmi() -> None:
    # b0 and b1 are both very popular (count=10 each) but only co-occur 3 times
    # (>= MIN_CO_COUNT, so it survives the count filter). Raw PMI = log(3*N/(10*10))
    # is negative for small N -> must be floored to 0.0, not dropped.
    book_ids = ["b0", "b1"] + [f"x{i}" for i in range(18)]
    item_matrix = _item_matrix(book_ids)

    rows: list[tuple[str, str, bool, float]] = []
    # 3 users positive on both b0 and b1 (the co-occurring trio).
    for i in range(3):
        rows.append((f"co{i}", "b0", True, 5.0))
        rows.append((f"co{i}", "b1", True, 5.0))
    # 7 more users positive on b0 only (b0 total count = 10).
    for i in range(7):
        rows.append((f"only_b0_{i}", "b0", True, 4.0))
    # 7 more users positive on b1 only (b1 total count = 10).
    for i in range(7):
        rows.append((f"only_b1_{i}", "b1", True, 4.0))
    # N = 17 distinct users total; raw PMI = log(3*17/(10*10)) = log(0.51) < 0.
    raw_pmi = math.log(3 * 17 / (10 * 10))
    assert raw_pmi < 0  # sanity check on the fixture's premise

    result, diag = build_item_cooccurrence(item_matrix, [_interactions(rows)])
    pair = result[(result["book_id_a"] == "b0") & (result["book_id_b"] == "b1")]
    assert len(pair) == 1
    assert pair.iloc[0]["co_count"] == 3
    assert pair.iloc[0]["pmi"] == 0.0  # floored, not the negative raw value
    assert (result["pmi"] >= 0.0).all()


def test_canonicalization_orders_book_id_as_string_not_row_index() -> None:
    # Row 0 = "z9", row 1 = "a1": row-index order is the OPPOSITE of string order.
    # If the module only canonicalized by row index (row_i < row_j => keep a=row_i),
    # the output would wrongly have book_id_a="z9" > book_id_b="a1".
    item_matrix = _item_matrix(["z9", "a1"])
    rows = [
        ("u1", "z9", True, 5.0), ("u1", "a1", True, 5.0),
        ("u2", "z9", True, 4.0), ("u2", "a1", True, 4.0),
        ("u3", "z9", True, 5.0), ("u3", "a1", True, 4.0),
    ]
    result, _diag = build_item_cooccurrence(item_matrix, [_interactions(rows)])

    assert len(result) == 1
    row = result.iloc[0]
    assert row["book_id_a"] == "a1"
    assert row["book_id_b"] == "z9"
    assert (result["book_id_a"] < result["book_id_b"]).all()


def test_max_positives_per_user_cap_limits_pairs_not_marginal_count(monkeypatch) -> None:
    # Lower the cap to 2 so a 4-positive user clearly exceeds it.
    monkeypatch.setattr(bic, "MAX_POSITIVES_PER_USER", 2)

    item_matrix = _item_matrix(["b0", "b1", "b2", "b3"])
    rows = [
        ("u1", "b0", True, 5.0),
        ("u1", "b1", True, 5.0),
        ("u1", "b2", True, 5.0),
        ("u1", "b3", True, 5.0),
    ]
    result, diag = build_item_cooccurrence(item_matrix, [_interactions(rows)])

    # Marginal count(i) must reflect the FULL (uncapped) positive set: all 4 books
    # were positived by u1, so each must have count(i) == 1, which is only
    # observable indirectly here via co_count/PMI sums — verify directly through the
    # module's internal counts by rebuilding via the same path with a tiny universe
    # where every pair would show up if uncapped (C(4,2) = 6 pairs) vs capped
    # (C(2,2) = 1 pair, since the cap truncates to the first 2 sorted rows: b0, b1).
    assert diag["n_users_truncated"] == 1
    assert diag["n_pairs_total"] == 1  # capped: only the (b0, b1) pair was generated
    pairs = set(zip(result["book_id_a"], result["book_id_b"]))
    # With MIN_CO_COUNT=3 (production default) this single co_count=1 pair is filtered
    # out of the final result, but n_pairs_total (pre-filter) already proves the cap
    # limited pair generation to C(2,2)=1 instead of the uncapped C(4,2)=6.
    assert diag["n_pairs_after_min_count"] == 0
    assert pairs == set()


def test_max_positives_per_user_cap_marginal_count_uses_full_set(monkeypatch) -> None:
    # Cross-check the marginal count(i) claim with a second user who shares only the
    # books that get truncated away from u1's pair generation. If count(i) were
    # computed from the capped set, b2/b3 would have count 0; the brief requires the
    # uncapped full set, so b2/b3 must still accumulate count 1 from u1, then +1 more
    # from u2 below -> count(b2) == count(b3) == 2, observable via their PMI with a
    # shared partner.
    monkeypatch.setattr(bic, "MAX_POSITIVES_PER_USER", 2)

    item_matrix = _item_matrix(["b0", "b1", "b2", "b3"])
    rows = [
        ("u1", "b0", True, 5.0),
        ("u1", "b1", True, 5.0),
        ("u1", "b2", True, 5.0),
        ("u1", "b3", True, 5.0),
        # u2, u3 also positive on b2 and b3 together (not truncated, only 2 books each)
        # so (b2, b3) reaches co_count >= MIN_CO_COUNT via u2, u3, u4.
        ("u2", "b2", True, 5.0),
        ("u2", "b3", True, 5.0),
        ("u3", "b2", True, 5.0),
        ("u3", "b3", True, 5.0),
        ("u4", "b2", True, 5.0),
        ("u4", "b3", True, 5.0),
    ]
    result, diag = build_item_cooccurrence(item_matrix, [_interactions(rows)])

    pair = result[(result["book_id_a"] == "b2") & (result["book_id_b"] == "b3")]
    assert len(pair) == 1
    # count(b2) and count(b3) must include u1's contribution (uncapped marginal),
    # even though u1's b2/b3 positives were truncated away from pair generation by the
    # cap (u1's sorted-and-capped pair source is b0,b1, so u1 contributes no pair
    # involving b2 or b3). co_count(b2,b3) is exactly 3 (u2,u3,u4 only -> NOT 4),
    # which proves the cap did limit u1's pair contribution...
    assert pair.iloc[0]["co_count"] == 3
    # ...while N (n_users_with_positive) still counts u1, and count(b2)/count(b3) used
    # for the PMI denominator must be 4 each (u1,u2,u3,u4), not 3 -- verify via the
    # formula: PMI = log(co_count * N / (count_b2 * count_b3)).
    N = diag["n_users_with_positive"]
    assert N == 4
    expected_pmi = max(0.0, math.log(3 * N / (4 * 4)))
    assert math.isclose(pair.iloc[0]["pmi"], expected_pmi, rel_tol=1e-4)


def test_dropped_positive_rows_for_absent_book() -> None:
    item_matrix = _item_matrix(["b0", "b1"])
    rows = [
        ("u1", "b0", True, 5.0),
        ("u1", "b1", True, 5.0),
        ("u1", "zzz", True, 5.0),  # not in the item universe
    ]
    _result, diag = build_item_cooccurrence(item_matrix, [_interactions(rows)])
    assert diag["dropped_positive_rows"] == 1


def test_non_positive_rows_excluded() -> None:
    item_matrix = _item_matrix(["b0", "b1"])
    rows = [
        ("u1", "b0", False, 5.0),  # not read
        ("u1", "b1", True, 3.0),  # rating below threshold
    ]
    _result, diag = build_item_cooccurrence(item_matrix, [_interactions(rows)])
    assert diag["n_users_with_positive"] == 0
    assert len(_result) == 0
