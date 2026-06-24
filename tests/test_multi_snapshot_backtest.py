from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_multi_snapshot_backtest import (
    historical_master_for_snapshot,
    run_multi_snapshot,
)


def test_run_multi_snapshot_isolates_paths_and_concatenates_results(tmp_path: Path) -> None:
    seen: list[Path] = []

    def runner(cutoff: pd.Timestamp, snapshot_dir: Path) -> pd.DataFrame:
        seen.append(snapshot_dir)
        return pd.DataFrame({"system": ["model"], "k": [10], "recall": [cutoff.year / 10_000]})

    out = run_multi_snapshot(["2014-01-01", "2015-01-01"], runner, tmp_path)

    assert seen == [tmp_path / "2014-01-01", tmp_path / "2015-01-01"]
    assert out["snapshot_date"].str.startswith(("2014-", "2015-")).all()
    assert len(out) == 2


def test_historical_master_replaces_future_sensitive_counts() -> None:
    master = pd.DataFrame(
        {
            "book_id": ["b1", "b2"],
            "ratings_count": [999, 999],
            "average_rating": [5.0, 5.0],
            "text_reviews_count": [999, 999],
        }
    )
    snapshot = historical_master_for_snapshot(
        master,
        np.array(["b1", "b2"]),
        np.array([True, False]),
        np.array([3, 1]),
        np.array([4.0, 2.0]),
        np.array([1, 1]),
    )
    assert snapshot["book_id"].tolist() == ["b1"]
    assert snapshot.iloc[0]["ratings_count"] == 3
    assert snapshot.iloc[0]["average_rating"] == 4.0
    assert snapshot.iloc[0]["text_reviews_count"] == 1
