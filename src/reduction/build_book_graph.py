"""Build the book co-read graph from ``book_cooccurrence.parquet``.

Graph definition (see ``docs/grafo_libros.md`` for the full write-up):

- **Nodes**: one per ``book_id`` in ``books_master.parquet`` (same grain as the rest of the
  pipeline: book, not genre, not author). Books with zero qualifying edges are kept as
  isolated nodes, not dropped, so isolation is a visible/measurable property of the graph.
- **Edges**: one per row of ``book_cooccurrence.parquet`` (``book_id_a``, ``book_id_b``),
  i.e. "at least ``MIN_CO_COUNT`` distinct users positived both books." Undirected: the
  underlying relation has no inherent direction.
- **Weight**: PMI, already computed by ``build_item_cooccurrence.py``.

This module only adds graph-theoretic structure (degree, PageRank, centrality, components) on
top of that existing edge list; it does not recompute or change the edges themselves.

Invoke as a module::

    env/bin/python -m src.reduction.build_book_graph
"""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.config import (
    BOOK_COOCCURRENCE_PATH,
    BOOK_GRAPH_DIAGNOSTICS_PATH,
    BOOK_GRAPH_NODES_PATH,
    BOOKS_MASTER_PATH,
)
from src.utils.io import safe_write_parquet

SENSITIVITY_MIN_CO_COUNTS = [2, 3, 5, 10]
BETWEENNESS_SAMPLE_SEED = 42
BETWEENNESS_MAX_SOURCES = 500


def _build_graph(pairs: pd.DataFrame, book_ids: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(book_ids)
    graph.add_weighted_edges_from(
        zip(pairs["book_id_a"], pairs["book_id_b"], pairs["pmi"]),
        weight="pmi",
    )
    return graph


def _component_assignment(graph: nx.Graph) -> tuple[dict[str, int], dict[int, int]]:
    component_id: dict[str, int] = {}
    component_size: dict[int, int] = {}
    for component_idx, nodes in enumerate(nx.connected_components(graph)):
        component_size[component_idx] = len(nodes)
        for node in nodes:
            component_id[node] = component_idx
    return component_id, component_size


def _sensitivity_table(pairs: pd.DataFrame, book_ids: np.ndarray) -> list[dict[str, Any]]:
    """Recompute coarse graph stats at alternate MIN_CO_COUNT thresholds.

    Re-filters the already-built pairs frame (co_count is already a persisted column);
    does not rerun build_item_cooccurrence.
    """
    rows = []
    n_nodes = len(book_ids)
    for min_co_count in SENSITIVITY_MIN_CO_COUNTS:
        filtered = pairs[pairs["co_count"] >= min_co_count]
        graph = _build_graph(filtered, book_ids)
        sizes = [len(c) for c in nx.connected_components(graph)]
        largest_fraction = (max(sizes) / n_nodes) if sizes else 0.0
        rows.append(
            {
                "min_co_count": min_co_count,
                "n_edges": int(len(filtered)),
                "n_components": len(sizes),
                "largest_component_fraction": float(largest_fraction),
            }
        )
    return rows


def build_book_graph(
    pairs: pd.DataFrame,
    book_ids: np.ndarray,
) -> tuple[nx.Graph, pd.DataFrame, dict[str, Any]]:
    """Build the graph and compute per-node metrics + graph-level diagnostics."""
    graph = _build_graph(pairs, book_ids)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="pmi"))
    pagerank = nx.pagerank(graph, weight="pmi") if n_edges else {node: 1.0 / n_nodes for node in graph}
    component_id, component_size = _component_assignment(graph)

    # ponytail: exact betweenness is O(V*E), intractable at catalog scale; sample up to
    # BETWEENNESS_MAX_SOURCES source nodes instead. Raise the cap only if the report needs
    # tighter estimates than the sampled approximation gives.
    k = min(BETWEENNESS_MAX_SOURCES, n_nodes) if n_nodes else 0
    betweenness = (
        nx.betweenness_centrality(graph, k=k, weight="pmi", seed=BETWEENNESS_SAMPLE_SEED)
        if k
        else {}
    )

    nodes = pd.DataFrame(
        {
            "book_id": book_ids,
            "degree": [degree.get(b, 0) for b in book_ids],
            "weighted_degree": [weighted_degree.get(b, 0.0) for b in book_ids],
            "pagerank": [pagerank.get(b, 0.0) for b in book_ids],
            "betweenness_centrality": [betweenness.get(b, 0.0) for b in book_ids],
            "component_id": [component_id.get(b, -1) for b in book_ids],
        }
    )
    nodes["component_size"] = nodes["component_id"].map(component_size).fillna(0).astype(int)
    nodes["degree"] = nodes["degree"].astype(np.int64)
    nodes["weighted_degree"] = nodes["weighted_degree"].astype(np.float32)
    nodes["pagerank"] = nodes["pagerank"].astype(np.float64)
    nodes["betweenness_centrality"] = nodes["betweenness_centrality"].astype(np.float64)
    nodes["component_id"] = nodes["component_id"].astype(np.int64)

    isolated_node_count = int((nodes["degree"] == 0).sum())
    component_sizes = list(component_size.values())
    largest_component_fraction = (max(component_sizes) / n_nodes) if component_sizes else 0.0
    density = nx.density(graph) if n_nodes > 1 else 0.0

    diagnostics: dict[str, Any] = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": float(density),
        "n_components": len(component_sizes),
        "largest_component_fraction": float(largest_component_fraction),
        "isolated_node_count": isolated_node_count,
        "betweenness_sample_size": k,
        "min_co_count_sensitivity": _sensitivity_table(pairs, book_ids),
    }
    return graph, nodes, diagnostics


def main() -> None:
    if not BOOK_COOCCURRENCE_PATH.exists():
        raise FileNotFoundError(
            f"{BOOK_COOCCURRENCE_PATH} does not exist. "
            "Run `python -m src.reduction.build_item_cooccurrence` first."
        )
    pairs = pd.read_parquet(BOOK_COOCCURRENCE_PATH)
    book_ids = pd.read_parquet(BOOKS_MASTER_PATH, columns=["book_id"])["book_id"].astype(str).to_numpy()

    _graph, nodes, diagnostics = build_book_graph(pairs, book_ids)

    safe_write_parquet(nodes, BOOK_GRAPH_NODES_PATH)
    BOOK_GRAPH_DIAGNOSTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOOK_GRAPH_DIAGNOSTICS_PATH.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"Book graph: {diagnostics['n_nodes']:,} nodes, {diagnostics['n_edges']:,} edges")
    print(f"Components: {diagnostics['n_components']:,}, largest fraction: {diagnostics['largest_component_fraction']:.4f}")
    print(f"Isolated nodes: {diagnostics['isolated_node_count']:,}")
    print(f"Nodes written to: {BOOK_GRAPH_NODES_PATH}")
    print(f"Diagnostics written to: {BOOK_GRAPH_DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
