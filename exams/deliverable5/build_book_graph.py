"""Regenerate the Deliverable 5 book graph from processed artifacts.

This wrapper lives under ``exams/`` for grading, while the canonical implementation
stays in ``src.reduction.build_book_graph``.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reduction.build_book_graph import main


if __name__ == "__main__":
    main()
