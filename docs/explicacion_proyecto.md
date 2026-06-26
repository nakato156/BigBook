# Documento de Explicacion — BigBook

Objetivo: que alguien nuevo en el proyecto entienda qué se construyó, dónde vive cada pieza y por
qué, **centrado exclusivamente en los últimos 2 commits**:

- `8ba33cb` — refactor: split de `recommend.py` y `evaluate_recommender.py` por responsabilidad
- `ff770f2` — feat: grafo de co-lectura de libros (nodos/aristas/PageRank/reporte)

Para el contexto completo del pipeline (curación, PCA, clustering, etc.) ver `AGENTS.md` /
`CLAUDE.md` en la raíz — es la fuente canónica de instrucciones del proyecto y no se repite aquí.
Para el detalle exhaustivo del entregable del grafo (definición formal, rúbrica, preguntas de
defensa), ver `Deliverable5.md` — este documento es un resumen de transferencia, no lo reemplaza.

---

## 1. Contexto mínimo para ubicarse

BigBook es un sistema de recomendación de libros. La lógica de negocio (ver `AGENTS.md`) es:
modelar gustos de lectura como vectores multidimensionales (no como "género favorito"), evitar
sesgo de popularidad, y optimizar por descubrimiento que sostenga el hábito de lectura, no solo
por relevancia cruda. El flujo productivo es:

```
libros curados -> books_master -> PCA (master_feature_matrix) -> clustering
                -> interacciones globales -> perfiles de usuario -> ranking -> evaluación
```

Los dos commits de esta entrega tocan, respectivamente, la capa de **ranking/evaluación**
(reorganización interna, sin cambiar comportamiento) y agregan una **pieza analítica nueva**
(grafo de co-lectura) que vive al lado del pipeline productivo, sin participar en él.

---

## 2. Commit `8ba33cb` — split de `recommend.py` / `evaluate_recommender.py`

### El problema

`src/reduction/recommend.py` y `src/reduction/evaluate_recommender.py` habían crecido como
"god-files", mezclando retrieval, scoring, diversificación, orquestación, baselines, métricas y
proxies de hábito en dos archivos enormes. Localizar "dónde está el ranker" vs. "dónde está el
evaluador" era lento.

### Qué se hizo

Relocalización pura de código siguiendo las costuras ya documentadas
`retrieve -> score -> diversify -> explain`. **Sin cambio de comportamiento** — los comandos de
CLI siguen igual:

```bash
env/bin/python -m src.reduction.recommend
env/bin/python -m src.reduction.evaluate_recommender
```

### Mapa de archivos (dónde está cada cosa ahora)

| Archivo | Responsabilidad | Qué contiene |
|---|---|---|
| [src/reduction/retrieval.py](../src/reduction/retrieval.py) | **Retrieve**: candidatos y elegibilidad | `eligibility_mask` (elegibilidad técnica, nunca por popularidad), `popularity_segments` (tail/mid/head), `retrieve_top_clusters` / `retrieve_clusters_per_mode`, `taste_pc_indices` (subespacio de gusto sin las PCs tabulares), `consumed_books_for_users`. Sin estado, sin dependencia de la clase `Recommender`. |
| [src/reduction/ranking.py](../src/reduction/ranking.py) | **Score + Diversify**: config y lógica de ranking | `RankingConfig` (todos los tunables: `k`, `explore_slots`, `mmr_lambda`, etc.), `HybridV12Weights`, `mmr_select` (diversidad vía MMR), `accessibility_scores`, `select_exploration_rows` (exploración solo en tail/mid, con piso de relevancia). |
| [src/reduction/recommend.py](../src/reduction/recommend.py) | **Orquestador** | Solo la clase `Recommender` (carga artefactos, ensambla retrieve+score+diversify+explain) y el CLI. Métodos clave: `recommend` (ruta normal), `recommend_from_modes` (núcleo del ranking), `recommend_hybrid_v12` (ranker de unión con señales históricas/colaborativas), `recommend_cold_start` (diversidad por macro-cluster, sin popularidad). |
| [src/reduction/temporal_split.py](../src/reduction/temporal_split.py) | Splitting cronológico | `temporal_split` (por usuario), `global_temporal_split` / `choose_global_cutoff` (corte global compartido). Deliberadamente sin import de `Recommender`, para no crear ciclos. |
| [src/reduction/baselines.py](../src/reduction/baselines.py) | Baselines B0/B1/B2 + snapshots históricos | `historical_popularity_snapshot`, `popularity_from_training`, `prepare_baseline_rankings`, `baseline_recommendations`. |
| [src/reduction/metrics.py](../src/reduction/metrics.py) | Métricas N0 (ranking) | recall/precision/NDCG/AP (`_binary_metrics`), `_candidate_recall`, `_intra_list_diversity`, `bootstrap_confidence_intervals`. |
| [src/reduction/habit_proxies.py](../src/reduction/habit_proxies.py) | Proxies N1 (hábito, descriptivo) | `habit_proxy_features`, `build_habit_proxy_table`, `summarize_by_activity`. Correlacional, no causal. |
| [src/reduction/evaluate_recommender.py](../src/reduction/evaluate_recommender.py) | **Driver de evaluación** | Solo orquesta: split temporal → baselines → métricas N0 → proxies N1, más el CLI. |

### Cómo navegar el código nuevo (regla rápida)

- ¿Qué libros se consideran candidatos? → `retrieval.py`
- ¿Cómo se puntúan/diversifican/eligen? → `ranking.py`
- ¿Dónde está el objeto que junta todo y expone `.recommend(user_id, ...)`? → `recommend.py`,
  clase `Recommender`
- ¿Cómo se mide la calidad offline (recall@k, NDCG, bootstrap CI, hábito)? →
  `evaluate_recommender.py` orquesta; la métrica concreta vive en `metrics.py` o
  `habit_proxies.py`
- ¿Dónde están los baselines de control (B0/B1/B2)? → `baselines.py`

Archivos que solo actualizaron imports (sin lógica nueva): `scripts/run_ablation.py`,
`scripts/run_collaborative_ab.py`, `scripts/run_multi_snapshot_backtest.py`,
`scripts/run_ranker_grid.py`, `tests/test_ablation.py`, `tests/test_recommend.py`,
`tests/test_user_knn.py`.

---

## 3. Commit `ff770f2` — grafo de co-lectura de libros

### Qué resuelve

Construye un grafo libro-libro no dirigido a partir de la señal de co-ocurrencia PMI ya existente
(`data/features/book_cooccurrence.parquet`, producida por `build_item_cooccurrence.py` — **no se
toca en este commit**). Calcula métricas estructurales (degree, PageRank, centralidad aproximada,
componentes conexas), compara contra el baseline B1 de popularidad, y genera un reporte en
`docs/grafo_libros.md`. Es un **deliverable analítico aditivo**: no participa en el flujo de
recomendación en producción.

### Definición del grafo (lo mínimo para no malinterpretarlo)

- **Nodos**: uno por `book_id` de `books_master.parquet` (mismo grano que el resto del pipeline).
  Los libros sin aristas se mantienen como nodos aislados, no se eliminan.
- **Aristas**: una por fila de `book_cooccurrence.parquet` — "al menos `MIN_CO_COUNT` usuarios
  distintos positivaron ambos libros". No dirigidas (la relación es simétrica por construcción).
- **Peso**: PMI, ya calculado upstream; este módulo solo añade estructura de grafos encima, no
  recalcula las aristas.

### Mapa de archivos

| Archivo | Rol |
|---|---|
| [src/reduction/build_book_graph.py](../src/reduction/build_book_graph.py) | Construye el grafo con `networkx`: `degree`, `weighted_degree`, `pagerank`, `betweenness_centrality` (aproximada por muestreo, ver abajo), `component_id`/`component_size`, y una tabla de sensibilidad a distintos `MIN_CO_COUNT`. Escribe `data/outputs/graph/book_graph_nodes.parquet` + `book_graph_diagnostics.json`. |
| [src/reduction/graph_comparison.py](../src/reduction/graph_comparison.py) | `compare_to_popularity` (Spearman + overlap top-k entre PageRank/weighted-degree vs. popularidad histórica B1, solo nodos no aislados) y `compare_to_collaborative_ab` (resume el `collaborative_ab_results.csv` ya existente). Reutiliza artefactos, no re-evalúa nada. |
| [src/report_book_graph.py](../src/report_book_graph.py) | Genera `docs/grafo_libros.md`: top-10 por métrica, tabla de sensibilidad, comparación con popularidad, notas de interpretación. |
| `src/config.py` (+3 líneas) | Paths nuevos: `GRAPH_OUTPUTS_DIR`, `BOOK_GRAPH_NODES_PATH`, `BOOK_GRAPH_DIAGNOSTICS_PATH`. |
| `src/validate_artifacts.py` (+21 líneas) | Chequeo opcional: si `book_graph_nodes.parquet` existe, valida columnas, sin `book_id` duplicado, `book_id` ⊆ universo de `books_master`, `pagerank` finito/no-negativo y que `pagerank` sume ≈1.0. |
| `tests/test_book_graph.py`, `tests/test_artifact_validation.py` (+13) | Cobertura de construcción del grafo y de la validación nueva. |

### Decisiones a conocer antes de tocar este código

- **Betweenness es aproximada por muestreo** (marcado con comentario `ponytail:` en
  `build_book_graph.py:101-103`): exacta es `O(V·E)`, intratable a escala de catálogo (108k nodos
  / ~1M aristas). Se muestrean hasta `BETWEENNESS_MAX_SOURCES = 500` nodos fuente con seed fijo
  (42). Subir el cap es la única palanca si se necesita mayor precisión.
- **La comparación con popularidad excluye nodos aislados**: su PageRank/degree es 0 por
  construcción, no una señal real — incluirlos ensuciaría la correlación.
- Comando completo:
  ```bash
  env/bin/python -m src.reduction.build_book_graph   # requiere book_cooccurrence.parquet
  env/bin/python -m src.report_book_graph
  ```

Detalle exhaustivo (formato rúbrica, con cifras de la corrida real y preguntas de defensa) en
[`Deliverable5.md`](../Deliverable5.md).

---

## 4. Qué NO cambió en estos 2 commits

- La señal de co-ocurrencia PMI (`build_item_cooccurrence.py`, `book_cooccurrence.parquet`) — se
  **reutiliza**, no se modifica.
- El comportamiento observable de `recommend` / `evaluate_recommender` por CLI — el refactor es
  "pure relocation + import fixups, no behavior change" (mensaje del propio commit).
- El pipeline de curación, PCA, clustering y perfiles de usuario — fuera de alcance de ambos
  commits.
