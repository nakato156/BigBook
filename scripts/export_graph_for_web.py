"""Export a browser-renderable slice of the book co-read graph for webviz/.

The full graph (108k nodes / ~1M edges, see data/outputs/graph/) is too dense to render
interactively in a browser. This script selects the top-N non-isolated nodes by PageRank,
induces edges among only that set, and writes a single JSON consumed by webviz/public/app.js.

Invoke as a module:

    env/bin/python -m scripts.export_graph_for_web
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import (
    BOOK_COOCCURRENCE_PATH,
    BOOK_GRAPH_DIAGNOSTICS_PATH,
    BOOK_GRAPH_NODES_PATH,
    BOOKS_MASTER_PATH,
    PROJECT_ROOT,
)
from src.reduction.graph_comparison import compare_to_popularity

OUTPUT_PATH = PROJECT_ROOT / "webviz" / "public" / "graph_viz.json"
DEFAULT_TOP_N = 600
# ponytail: full induced subgraph on 600 nodes is a ~70k-edge hairball (avg degree 233);
# keep each node's strongest edges only (mutual kNN-style pruning) so the force layout in the
# browser stays readable. Raise if the report needs denser local structure.
DEFAULT_EDGES_PER_NODE = 8
GENRE_COLUMNS = ["genre_fantasy", "genre_mystery", "genre_history", "genre_ya", "genre_romance"]


def _dominant_genre(row: pd.Series) -> str:
    active = [g.removeprefix("genre_") for g in GENRE_COLUMNS if row[g] == 1]
    if not active:
        return "other"
    if len(active) > 1:
        return "multi"
    return active[0]


def _prune_to_strongest_edges(edges: pd.DataFrame, edges_per_node: int) -> pd.DataFrame:
    """Keep an edge if it's among either endpoint's strongest `edges_per_node` edges by PMI."""
    ranked_a = edges.sort_values("pmi", ascending=False).groupby("book_id_a").head(edges_per_node)
    ranked_b = edges.sort_values("pmi", ascending=False).groupby("book_id_b").head(edges_per_node)
    keep_index = ranked_a.index.union(ranked_b.index)
    return edges.loc[keep_index]


def build_export(top_n: int = DEFAULT_TOP_N, edges_per_node: int = DEFAULT_EDGES_PER_NODE) -> dict:
    nodes = pd.read_parquet(BOOK_GRAPH_NODES_PATH).astype({"book_id": str})
    master = pd.read_parquet(
        BOOKS_MASTER_PATH,
        columns=["book_id", "title", "average_rating", "ratings_count", *GENRE_COLUMNS],
    ).astype({"book_id": str})
    diagnostics = json.loads(BOOK_GRAPH_DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    comparison = compare_to_popularity(nodes)

    master["genre_label"] = master.apply(_dominant_genre, axis=1)
    nodes = nodes.merge(
        master[["book_id", "title", "average_rating", "ratings_count", "genre_label"]],
        on="book_id",
        how="left",
    )

    selected = nodes[nodes["degree"] > 0].sort_values("pagerank", ascending=False).head(top_n)
    selected_ids = set(selected["book_id"])

    pairs = pd.read_parquet(BOOK_COOCCURRENCE_PATH)
    edges = pairs[pairs["book_id_a"].isin(selected_ids) & pairs["book_id_b"].isin(selected_ids)]
    edges = _prune_to_strongest_edges(edges, edges_per_node)

    node_records = [
        {
            "id": r.book_id,
            "title": r.title,
            "degree": int(r.degree),
            "weighted_degree": float(r.weighted_degree),
            "pagerank": float(r.pagerank),
            "betweenness_centrality": float(r.betweenness_centrality) if pd.notna(r.betweenness_centrality) else 0.0,
            "component_id": int(r.component_id),
            "component_size": int(r.component_size),
            "genre": r.genre_label,
            "average_rating": float(r.average_rating) if pd.notna(r.average_rating) else None,
            "ratings_count": int(r.ratings_count) if pd.notna(r.ratings_count) else 0,
        }
        for r in selected.itertuples(index=False)
    ]
    edge_records = [
        {"source": r.book_id_a, "target": r.book_id_b, "pmi": float(r.pmi), "co_count": int(r.co_count)}
        for r in edges.itertuples(index=False)
    ]

    return {
        "nodes": node_records,
        "edges": edge_records,
        "meta": {
            "selected_node_count": len(node_records),
            "selected_edge_count": len(edge_records),
            "selection": (
                f"top-{top_n} non-isolated nodes by PageRank, induced edges pruned to each "
                f"node's strongest {edges_per_node} edges by PMI"
            ),
            "full_graph_diagnostics": diagnostics,
            "popularity_comparison": comparison,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--edges-per-node", type=int, default=DEFAULT_EDGES_PER_NODE)
    args = parser.parse_args()

    export = build_export(args.top_n, args.edges_per_node)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(export), encoding="utf-8")
    print(f"Wrote {export['meta']['selected_node_count']} nodes, "
          f"{export['meta']['selected_edge_count']} edges to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
