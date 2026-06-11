from __future__ import annotations

import numpy as np

from scripts import build_deliverable3_clustering_outputs as clustering


def test_get_kmeans_model_persists_and_reuses_each_k(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(clustering, "OUTPUT_DIR", tmp_path)
    x = np.array(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]],
        dtype=np.float32,
    )

    first = clustering.get_kmeans_model(2, x)
    second = clustering.get_kmeans_model(2, x)

    assert (tmp_path / "kmeans_model_k2.joblib").exists()
    assert first.n_clusters == second.n_clusters == 2
    np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_)
