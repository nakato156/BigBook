"""Materialize Deliverable 5 graph comparisons under ``exams/``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import BOOK_GRAPH_NODES_PATH
from src.reduction.graph_comparison import (
    compare_to_collaborative_ab,
    compare_to_hybrid_ranker_grid,
    compare_to_popularity,
)
from src.report_book_graph import COLLABORATIVE_AB_RESULTS_PATH, RANKER_GRID_RESULTS_PATH


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not BOOK_GRAPH_NODES_PATH.exists():
        raise FileNotFoundError(
            f"{BOOK_GRAPH_NODES_PATH} does not exist. "
            "Run `env/bin/python exams/deliverable5/build_book_graph.py` first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(BOOK_GRAPH_NODES_PATH)
    popularity_rows = [
        {"metric": key, "value": value}
        for key, value in compare_to_popularity(nodes).items()
    ]
    popularity_path = args.output_dir / "graph_vs_popularity.csv"
    pd.DataFrame(popularity_rows).to_csv(popularity_path, index=False)

    collaborative_summary = compare_to_collaborative_ab(COLLABORATIVE_AB_RESULTS_PATH)
    if collaborative_summary is None:
        collaborative_summary = compare_to_hybrid_ranker_grid(RANKER_GRID_RESULTS_PATH)
    collaborative_path = args.output_dir / "graph_collaborative_ab_summary.txt"
    collaborative_path.write_text(
        collaborative_summary
        or (
            f"{COLLABORATIVE_AB_RESULTS_PATH} not found. Run "
            "`env/bin/python scripts/run_collaborative_ab.py --max-users 1000 --k 10` "
            f"to generate it, or run `env/bin/python scripts/run_ranker_grid.py "
            f"--cooccurrence-path data/features/book_cooccurrence.parquet` to refresh "
            f"{RANKER_GRID_RESULTS_PATH}."
        ),
        encoding="utf-8",
    )

    print(f"Wrote popularity comparison to: {popularity_path}")
    print(f"Wrote collaborative comparison summary to: {collaborative_path}")


if __name__ == "__main__":
    main()
