"""Regenerate the Deliverable 5 graph report.

This wrapper is intentionally thin so ``exams/`` exposes the submitted command
without forking the report logic from ``src.report_book_graph``.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.report_book_graph import main


if __name__ == "__main__":
    main()
