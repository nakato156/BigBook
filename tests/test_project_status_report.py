from __future__ import annotations

import pandas as pd

from src.report_project_status import validation_verdict


def test_validation_verdict_requires_model_to_beat_b1_at_every_k() -> None:
    summary = pd.DataFrame(
        {
            "system": ["model", "B1_popularity"] * 2,
            "k": [5, 5, 10, 10],
            "recall": [0.2, 0.1, 0.1, 0.2],
            "ndcg": [0.2, 0.1, 0.1, 0.2],
        }
    )

    validated, message = validation_verdict(summary)

    assert not validated
    assert "k=10" in message


def test_validation_verdict_accepts_consistent_n0_wins() -> None:
    summary = pd.DataFrame(
        {
            "system": ["model", "B1_popularity"] * 2,
            "k": [5, 5, 10, 10],
            "recall": [0.2, 0.1, 0.3, 0.2],
            "ndcg": [0.2, 0.1, 0.3, 0.2],
        }
    )

    validated, message = validation_verdict(summary)

    assert validated
    assert "V1 validada" in message
