"""Regenerate the complete Deliverable 5 graph artifact set from processed data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import BOOK_COOCCURRENCE_PATH


OUTPUT_DIR = Path(__file__).resolve().parent / "generated"


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=REPO_ROOT, check=True)


def main() -> None:
    if not BOOK_COOCCURRENCE_PATH.exists():
        _run(["-m", "src.reduction.build_item_cooccurrence"])
    _run(["-m", "src.reduction.build_book_graph"])
    _run(
        [
            "-m",
            "src.report_book_graph",
            "--output",
            str(OUTPUT_DIR / "grafo_libros.md"),
        ]
    )
    _run(["-m", "exams.deliverable5.run_graph_comparison", "--output-dir", str(OUTPUT_DIR)])


if __name__ == "__main__":
    main()
