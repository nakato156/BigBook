"""Generate the book co-read graph report (docs/grafo_libros.md)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import (
    BOOK_GRAPH_DIAGNOSTICS_PATH,
    BOOK_GRAPH_NODES_PATH,
    BOOKS_MASTER_PATH,
    PROJECT_ROOT,
)
from src.reduction.graph_comparison import compare_to_collaborative_ab, compare_to_popularity

COLLABORATIVE_AB_RESULTS_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "recommendations" / "collaborative_ab_results.csv"
)
REPORT_PATH = PROJECT_ROOT / "docs" / "grafo_libros.md"


def _top_titles(nodes: pd.DataFrame, titles: pd.Series, metric: str, n: int = 10) -> str:
    top = nodes.sort_values(metric, ascending=False).head(n)
    lines = [
        f"{i+1}. {titles.get(row.book_id, row.book_id)} (`{metric}`={getattr(row, metric):.4f})"
        for i, row in enumerate(top.itertuples(index=False))
    ]
    return "\n".join(lines)


def _sensitivity_table(rows: list[dict]) -> str:
    header = "| min_co_count | n_edges | n_components | largest_component_fraction |"
    separator = "| --- | --- | --- | --- |"
    body = "\n".join(
        f"| {r['min_co_count']} | {r['n_edges']:,} | {r['n_components']:,} | {r['largest_component_fraction']:.4f} |"
        for r in rows
    )
    return "\n".join([header, separator, body])


def build_report(
    nodes_path: Path = BOOK_GRAPH_NODES_PATH,
    diagnostics_path: Path = BOOK_GRAPH_DIAGNOSTICS_PATH,
) -> str:
    if not nodes_path.exists() or not diagnostics_path.exists():
        raise FileNotFoundError("Run `python -m src.reduction.build_book_graph` before building the report.")

    nodes = pd.read_parquet(nodes_path)
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    titles = (
        pd.read_parquet(BOOKS_MASTER_PATH, columns=["book_id", "title"])
        .astype({"book_id": str})
        .set_index("book_id")["title"]
    )

    popularity_comparison = compare_to_popularity(nodes)
    collaborative_summary = compare_to_collaborative_ab(COLLABORATIVE_AB_RESULTS_PATH)

    isolated_fraction = diagnostics["isolated_node_count"] / diagnostics["n_nodes"] if diagnostics["n_nodes"] else 0.0

    return f"""# Grafo de co-lectura de libros

## 1. Definición del grafo

- **Grano / nodos**: un nodo por `book_id` de `books_master.parquet` (mismo grano que el resto
  del pipeline: libro, no género ni autor). Los libros sin aristas que califiquen quedan como
  nodos aislados, no se eliminan — la aislación es una propiedad medible del grafo, no un
  artefacto de limpieza.
- **Aristas**: una por fila de `book_cooccurrence.parquet`: al menos `MIN_CO_COUNT=3` usuarios
  distintos marcaron ambos libros como positivos (`is_read AND rating_clean >= 4`).
- **Direccionalidad**: **no dirigido**. La relación "un usuario positivó ambos libros" es
  simétrica por construcción; no implica orden temporal ni causal entre los dos libros. (Nota
  de defensa: el grafo bipartito usuario→libro sí tendría dirección natural; este es la
  proyección libro-libro de ese bipartito, que colapsa la dirección.)
- **Peso**: PMI (`log(co_count * N / (count_i * count_j))`, piso en 0), ya calculado por
  `build_item_cooccurrence.py`. PMI normaliza la popularidad individual de cada libro — un peso
  alto significa afinidad de co-lectura más fuerte que el azar, no que ambos libros sean
  populares.

## 2. Construcción

Regenerar con:

```bash
env/bin/python -m src.reduction.build_item_cooccurrence   # si book_cooccurrence.parquet no existe
env/bin/python -m src.reduction.build_book_graph
env/bin/python -m src.report_book_graph
```

Fuente de aristas: `data/features/book_cooccurrence.parquet`. Salidas de este módulo:
`data/outputs/graph/book_graph_nodes.parquet` (métricas por nodo) y
`data/outputs/graph/book_graph_diagnostics.json` (métricas a nivel de grafo).

## 3. Reporte del grafo

| métrica | valor |
| --- | --- |
| nodos | {diagnostics['n_nodes']:,} |
| aristas | {diagnostics['n_edges']:,} |
| densidad | {diagnostics['density']:.8f} |
| componentes conexas | {diagnostics['n_components']:,} |
| fracción del componente más grande | {diagnostics['largest_component_fraction']:.4f} |
| nodos aislados | {diagnostics['isolated_node_count']:,} ({isolated_fraction:.2%}) |
| fuentes muestreadas para betweenness | {diagnostics['betweenness_sample_size']:,} |

### Top 10 por PageRank

{_top_titles(nodes, titles, "pagerank")}

### Top 10 por grado ponderado

{_top_titles(nodes, titles, "weighted_degree")}

## 4. Comparación

Correlación de Spearman entre las métricas del grafo y la popularidad histórica (B1,
`rating_count` acumulado), y solapamiento del top-{int(popularity_comparison['k'])}, restringido a
nodos no aislados:

- PageRank vs. popularidad: ρ={popularity_comparison['pagerank_vs_popularity_spearman']:.4f},
  solapamiento top-k={popularity_comparison['pagerank_top_k_overlap']:.2%}.
- Grado ponderado vs. popularidad: ρ={popularity_comparison['weighted_degree_vs_popularity_spearman']:.4f},
  solapamiento top-k={popularity_comparison['weighted_degree_top_k_overlap']:.2%}.

Una correlación moderada (no perfecta) es la lectura esperada: el grafo y la popularidad
capturan señal relacionada pero no idéntica — PMI descuenta exactamente la popularidad
individual que B1 mide.

{"### Señal de cooccurrencia en la comparación colaborativa offline\n\n" + collaborative_summary if collaborative_summary else "(`collaborative_ab_results.csv` no encontrado; correr `scripts/run_collaborative_ab.py` para esta pierna de comparación.)"}

## 5. Nota de interpretación

**Qué significa** un PMI/PageRank alto entre dos libros: afinidad de co-lectura positiva más
fuerte que el azar, dado el patrón de lectura observado en Goodreads. Es señal colaborativa, no
editorial ni de contenido.

**Qué NO significa**: no implica similitud temática, no implica causalidad ("leer A causa leer
B"), y un PageRank alto no significa "mejor libro" — es centralidad estructural en la red de
co-lectura observada, sesgada por quién está en el dataset.

**Qué significa un nodo aislado**: no hay señal colaborativa que califique (menos de
`MIN_CO_COUNT` lectores compartidos con rating positivo en ambos). Es el caso esperado para
libros nicho o de cola larga — no implica baja calidad, implica baja exposición compartida
observada.

## 6. Validez

- **Sparsity**: densidad de {diagnostics['density']:.8f} — esperado a esta escala de catálogo;
  confirma que el grafo es disperso por construcción (el filtro `MIN_CO_COUNT` ya descarta ruido
  de baja confianza).
- **Nodos aislados**: {diagnostics['isolated_node_count']:,} ({isolated_fraction:.2%} del
  catálogo) sin ninguna arista que califique.
- **Estructura de componentes**: {diagnostics['n_components']:,} componentes conexas; la más
  grande concentra {diagnostics['largest_component_fraction']:.2%} de los nodos no aislados.
- **Sensibilidad a `MIN_CO_COUNT`**: recomputado en {{2, 3, 5, 10}} sobre el mismo
  `book_cooccurrence.parquet`, sin volver a escanear interacciones:

{_sensitivity_table(diagnostics['min_co_count_sensitivity'])}

## 7. Preguntas de defensa

- **¿Por qué no dirigido?** La co-lectura positiva no tiene orden implícito; un grafo dirigido
  exigiría una relación asimétrica (p. ej. "leyó A antes de B"), que no es la señal disponible.
- **¿Por qué PMI y no co-ocurrencia cruda?** La co-ocurrencia cruda premia a los libros populares
  por aparecer juntos solo por volumen; PMI normaliza por la popularidad individual de cada
  libro, igual que el resto del pipeline evita el sesgo de popularidad.
- **¿Por qué betweenness aproximado?** Betweenness exacto es O(V·E); a escala de catálogo es
  intratable. Se usa el muestreo estándar de networkx
  (`k={diagnostics['betweenness_sample_size']}` fuentes, semilla fija) — ver el comentario
  `ponytail:` en `build_book_graph.py`.
- **¿Por qué comparar contra B1 y no contra el ranker de producción?** El grafo es un análisis
  estructural independiente; compararlo contra B1 (popularidad histórica) aísla si el grafo
  aporta señal más allá de la popularidad cruda, sin acoplar esta entrega a cambios del ranker
  de producción. La señal de cooccurrencia ya se evalúa contra el ranker en
  `scripts/run_collaborative_ab.py` (sección 4).
- **¿Qué significa un nodo aislado para un usuario o libro nuevo?** Ausencia de señal
  colaborativa, no una propiedad negativa del libro — ver sección 5.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote book graph report to {args.output}")


if __name__ == "__main__":
    main()
