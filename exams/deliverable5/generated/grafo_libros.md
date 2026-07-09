# Grafo de co-lectura de libros

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
| nodos | 108,227 |
| aristas | 1,049,382 |
| densidad | 0.00017918 |
| componentes conexas | 71,326 |
| fracción del componente más grande | 0.3335 |
| nodos aislados | 70,765 (65.39%) |
| fuentes muestreadas para betweenness | 50 |

### Top 10 por PageRank

1. The Hitchhiker's Guide to the Galaxy (Hitchhiker's Guide to the Galaxy, #1) (`pagerank`=0.0007)
2. Jane Eyre (`pagerank`=0.0007)
3. A Study in Scarlet (`pagerank`=0.0006)
4. 11/22/63 (`pagerank`=0.0006)
5. Norwegian Wood (`pagerank`=0.0006)
6. Steve Jobs (`pagerank`=0.0006)
7. Fifty Shades of Grey (Fifty Shades, #1) (`pagerank`=0.0005)
8. The Name of the Rose (`pagerank`=0.0005)
9. 1776 (`pagerank`=0.0005)
10. Misery (`pagerank`=0.0005)

### Top 10 por grado ponderado

1. A Study in Scarlet (`weighted_degree`=1632.3118)
2. Gabriel's Inferno (Gabriel's Inferno, #1) (`weighted_degree`=1602.6042)
3. Pushing the Limits (Pushing the Limits, #1) (`weighted_degree`=1510.4921)
4. Seduction of a Highland Lass (McCabe Trilogy, #2) (`weighted_degree`=1491.9269)
5. The Affair (Jack Reacher, #16) (`weighted_degree`=1490.0159)
6. 1776 (`weighted_degree`=1476.7281)
7. 11/22/63 (`weighted_degree`=1449.8328)
8. Deadlocked (Sookie Stackhouse, #12) (`weighted_degree`=1408.7621)
9. Up from the Grave (Night Huntress, #7) (`weighted_degree`=1405.2190)
10. Rules of Civility (`weighted_degree`=1318.0482)

## 4. Comparación

Correlación de Spearman entre las métricas del grafo y la popularidad histórica (B1,
`rating_count` acumulado), y solapamiento del top-100, restringido a
nodos no aislados:

- PageRank vs. popularidad: ρ=0.1968,
  solapamiento top-k=11.00%.
- Grado ponderado vs. popularidad: ρ=-0.0377,
  solapamiento top-k=5.00%.

Una correlación moderada (no perfecta) es la lectura esperada: el grafo y la popularidad
capturan señal relacionada pero no idéntica — PMI descuenta exactamente la popularidad
individual que B1 mide.

### Señal de cooccurrencia en la comparación colaborativa offline

     config_label        system  cooccurrence_weight  k   recall     ndcg  candidate_recall
     content_only  content_only                  0.0  5 0.002733 0.006181          0.379758
     content_only B1_popularity                  0.0  5 0.015553 0.037230               NaN
     content_only  content_only                  0.0 10 0.004745 0.006773          0.379758
     content_only B1_popularity                  0.0 10 0.025292 0.034830               NaN
     content_only  content_only                  0.0 20 0.011873 0.009256          0.379758
     content_only B1_popularity                  0.0 20 0.036133 0.034181               NaN
hybrid_v12_grid_1    hybrid_v12                  0.1  5 0.009068 0.020949          0.450584
hybrid_v12_grid_1    hybrid_v12                  0.1 10 0.015355 0.020371          0.450584
hybrid_v12_grid_1    hybrid_v12                  0.1 20 0.024528 0.021888          0.450584
hybrid_v12_grid_2    hybrid_v12                  0.1  5 0.010260 0.021251          0.450584
hybrid_v12_grid_2    hybrid_v12                  0.1 10 0.016419 0.021429          0.450584
hybrid_v12_grid_2    hybrid_v12                  0.1 20 0.025930 0.022706          0.450584
hybrid_v12_grid_3    hybrid_v12                  0.1  5 0.010747 0.022143          0.450584
hybrid_v12_grid_3    hybrid_v12                  0.1 10 0.016441 0.021814          0.450584
hybrid_v12_grid_3    hybrid_v12                  0.1 20 0.026619 0.023405          0.450584
hybrid_v12_grid_4    hybrid_v12                  0.1  5 0.011405 0.023244          0.450584
hybrid_v12_grid_4    hybrid_v12                  0.1 10 0.017446 0.023057          0.450584
hybrid_v12_grid_4    hybrid_v12                  0.1 20 0.028738 0.024678          0.450584

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

- **Sparsity**: densidad de 0.00017918 — esperado a esta escala de catálogo;
  confirma que el grafo es disperso por construcción (el filtro `MIN_CO_COUNT` ya descarta ruido
  de baja confianza).
- **Nodos aislados**: 70,765 (65.39% del
  catálogo) sin ninguna arista que califique.
- **Estructura de componentes**: 71,326 componentes conexas; la más
  grande concentra 33.35% de los nodos no aislados.
- **Sensibilidad a `MIN_CO_COUNT`**: recomputado para [3, 5, 10] sobre el mismo `book_cooccurrence.parquet`. Los umbrales [2] no se reportan porque el edge list persistido ya empieza en `co_count >= 3`; para medirlos exactamente hay que regenerar `book_cooccurrence.parquet` con un `--min-co-count` menor.

| min_co_count | n_edges | n_components | largest_component_fraction |
| --- | --- | --- | --- |
| 3 | 1,049,382 | 71,326 | 0.3335 |
| 5 | 568,384 | 83,164 | 0.2251 |
| 10 | 266,835 | 93,138 | 0.1353 |

## 7. Preguntas de defensa

- **¿Por qué no dirigido?** La co-lectura positiva no tiene orden implícito; un grafo dirigido
  exigiría una relación asimétrica (p. ej. "leyó A antes de B"), que no es la señal disponible.
- **¿Por qué PMI y no co-ocurrencia cruda?** La co-ocurrencia cruda premia a los libros populares
  por aparecer juntos solo por volumen; PMI normaliza por la popularidad individual de cada
  libro, igual que el resto del pipeline evita el sesgo de popularidad.
- **¿Por qué betweenness aproximado?** Betweenness exacto es O(V·E); a escala de catálogo es
  intratable. Se usa el muestreo estándar de networkx
  (`k=50` fuentes, semilla fija) — ver el comentario
  en `build_book_graph.py`.
- **¿Por qué comparar contra B1 y no contra el ranker de producción?** El grafo es un análisis
  estructural independiente; compararlo contra B1 (popularidad histórica) aísla si el grafo
  aporta señal más allá de la popularidad cruda, sin acoplar esta entrega a cambios del ranker
  de producción. La señal de cooccurrencia ya se evalúa contra el ranker en
  `scripts/run_collaborative_ab.py` (sección 4).
- **¿Qué significa un nodo aislado para un usuario o libro nuevo?** Ausencia de señal
  colaborativa, no una propiedad negativa del libro — ver sección 5.
