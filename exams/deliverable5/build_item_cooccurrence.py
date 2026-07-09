"""Regenerate the Deliverable 5 edge-list input from processed interactions."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reduction.build_item_cooccurrence import main


if __name__ == "__main__":
    main()
