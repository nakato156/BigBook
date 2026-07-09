from __future__ import annotations

import numpy as np
import pandas as pd

from src.reduction.build_book_graph import build_book_graph


def _pairs(rows: list[tuple[str, str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["book_id_a", "book_id_b", "pmi", "co_count"])


def test_isolated_node_has_zero_degree_and_singleton_component() -> None:
    # b0-b1-b2 form a triangle; b3 has no qualifying edge at all.
    book_ids = np.array(["b0", "b1", "b2", "b3"])
    pairs = _pairs(
        [
            ("b0", "b1", 1.0, 3),
            ("b0", "b2", 1.0, 3),
            ("b1", "b2", 1.0, 3),
        ]
    )
    _graph, nodes, diagnostics = build_book_graph(pairs, book_ids)

    isolated = nodes[nodes["book_id"] == "b3"].iloc[0]
    assert isolated["degree"] == 0
    assert isolated["component_size"] == 1
    assert diagnostics["isolated_node_count"] == 1
    assert diagnostics["n_nodes"] == 4
    assert diagnostics["n_edges"] == 3


def test_star_graph_ranks_hub_highest_in_degree_and_pagerank() -> None:
    # b0 is connected to b1, b2, b3 (a star); b1/b2/b3 are not connected to each other.
    book_ids = np.array(["b0", "b1", "b2", "b3"])
    pairs = _pairs(
        [
            ("b0", "b1", 2.0, 5),
            ("b0", "b2", 2.0, 5),
            ("b0", "b3", 2.0, 5),
        ]
    )
    _graph, nodes, diagnostics = build_book_graph(pairs, book_ids)

    hub = nodes[nodes["book_id"] == "b0"].iloc[0]
    leaves = nodes[nodes["book_id"] != "b0"]
    assert hub["degree"] == 3
    assert (leaves["degree"] == 1).all()
    assert hub["pagerank"] > leaves["pagerank"].max()
    assert diagnostics["n_components"] == 1
    assert diagnostics["largest_component_fraction"] == 1.0


def test_two_disconnected_components_are_counted_separately() -> None:
    book_ids = np.array(["a0", "a1", "b0", "b1"])
    pairs = _pairs(
        [
            ("a0", "a1", 1.0, 3),
            ("b0", "b1", 1.0, 3),
        ]
    )
    _graph, nodes, diagnostics = build_book_graph(pairs, book_ids)

    assert diagnostics["n_components"] == 2
    assert diagnostics["isolated_node_count"] == 0
    component_sizes = set(nodes.groupby("component_id")["component_size"].first())
    assert component_sizes == {2}


def test_min_co_count_sensitivity_is_monotonic_in_edges() -> None:
    book_ids = np.array(["b0", "b1", "b2"])
    pairs = _pairs(
        [
            ("b0", "b1", 1.0, 2),  # only survives min_co_count <= 2
            ("b1", "b2", 1.0, 5),  # survives all thresholds up to 5
        ]
    )
    _graph, _nodes, diagnostics = build_book_graph(pairs, book_ids)
    sensitivity = {row["min_co_count"]: row["n_edges"] for row in diagnostics["min_co_count_sensitivity"]}

    assert sensitivity[2] == 2
    assert sensitivity[3] == 1
    assert sensitivity[5] == 1
    assert sensitivity[10] == 0


def test_sensitivity_omits_thresholds_below_persisted_edge_list_minimum() -> None:
    book_ids = np.array(["b0", "b1", "b2"])
    pairs = _pairs(
        [
            ("b0", "b1", 1.0, 3),
            ("b1", "b2", 1.0, 5),
        ]
    )
    _graph, _nodes, diagnostics = build_book_graph(
        pairs,
        book_ids,
        sensitivity_min_co_counts=[2, 3, 5],
    )
    sensitivity = diagnostics["min_co_count_sensitivity"]
    meta = diagnostics["min_co_count_sensitivity_meta"]

    assert [row["min_co_count"] for row in sensitivity] == [3, 5]
    assert meta["source_min_co_count"] == 3
    assert meta["omitted_min_co_counts"] == [2]


def test_betweenness_can_be_skipped_for_fast_regeneration() -> None:
    book_ids = np.array(["b0", "b1"])
    pairs = _pairs([("b0", "b1", 1.0, 3)])
    _graph, nodes, diagnostics = build_book_graph(
        pairs,
        book_ids,
        betweenness_sample_size=None,
    )

    assert diagnostics["betweenness_sample_size"] == 0
    assert (nodes["betweenness_centrality"] == 0.0).all()
