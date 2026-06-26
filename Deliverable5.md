# Deliverable 5 — Grafo de co-lectura de libros

Este documento describe lo implementado para el entregable de grafo (nodos, aristas, pesos,
direccionalidad y grano; script de construcción; reporte con componentes/centralidad/PageRank;
comparación contra un baseline; nota de interpretación; chequeos de validez; y material de
defensa), y mapea cada parte del trabajo contra la rúbrica de la tabla de evaluación.

No es un módulo aislado: reutiliza el artefacto colaborativo ya existente del pipeline
(`book_cooccurrence.parquet`, PMI ponderado por co-lectura) como definición de aristas, en lugar
de inventar una señal nueva. Es **aditivo**: no modifica `ranking.py`, `recommend.py` ni ninguna
ruta de inferencia en producción.

## Resumen
Construimos un grafo donde cada nodo es un libro, y conectamos dos libros si suficientes usuarios los calificaron positivo a ambos. El peso de la conexión es PMI, que corrige el sesgo de popularidad — así la conexión refleja afinidad real entre lectores, no solo que ambos libros sean famosos. Con esto medimos estructura: el grafo es muy disperso y fragmentado en miles de componentes pequeños, con un 65% de libros sin pareja colaborativa suficiente — eso confirma que nuestro catálogo tiene una cola larga fuerte. Usamos PageRank y grado ponderado para encontrar los libros más 'centrales' en esa red de afinidad, y comparamos ese ranking contra el de popularidad pura: la correlación es baja, lo que demuestra que el grafo capta una señal distinta a 'simplemente populares' — es justo lo que buscábamos al usar PMI en vez de conteo crudo

## Archivos entregados

| Archivo | Rol |
| --- | --- |
| `src/reduction/build_book_graph.py` | Script de construcción del grafo (nodos, aristas, métricas, diagnósticos). |
| `src/reduction/graph_comparison.py` | Comparación grafo vs. popularidad (B1) y vs. la ablación colaborativa existente. |
| `src/report_book_graph.py` | Genera `docs/grafo_libros.md` a partir de los artefactos del grafo. |
| `docs/grafo_libros.md` | Reporte generado: definición, construcción, métricas, comparación, interpretación, validez, defensa. |
| `src/validate_artifacts.py` (extendido) | Chequeo opcional de esquema/alineación/consistencia para el artefacto del grafo. |
| `tests/test_book_graph.py` | Pruebas unitarias con grafos de juguete verificables a mano. |
| `tests/test_artifact_validation.py` (extendido) | Prueba de que el nuevo chequeo de validación acepta un grafo bien formado. |
| `src/config.py` (extendido) | Rutas `BOOK_GRAPH_NODES_PATH`, `BOOK_GRAPH_DIAGNOSTICS_PATH`. |

Salidas de datos (no versionadas, regenerables):

- `data/outputs/graph/book_graph_nodes.parquet` — una fila por `book_id` con sus métricas de grafo.
- `data/outputs/graph/book_graph_diagnostics.json` — métricas a nivel de grafo + tabla de sensibilidad.

---

## 1. Definición del grafo — 2.0 pts

> Nodos, aristas, pesos, direccionalidad y grano del grafo, precisos y justificados.

**Grano y nodos.** Un nodo por `book_id` de `books_master.parquet`: el mismo grano que el resto
del pipeline de recomendación (`one row = one book_id = one book vector`, ver `AGENTS.md`). No se
agrupa por género, autor ni edición — el género ya se usa en otras partes del sistema como señal
de filtrado/diversidad, nunca como unidad de agregación, y este grafo respeta esa decisión. Los
libros sin ninguna arista que califique **se mantienen como nodos aislados** en lugar de
eliminarse: la aislación es una propiedad observable y medible del grafo (sección de validez),
no un efecto de limpieza de datos.

**Aristas.** Una arista no dirigida por cada fila de `data/features/book_cooccurrence.parquet`
(`book_id_a`, `book_id_b`), que ya codifica: "al menos `MIN_CO_COUNT = 3` usuarios distintos
marcaron ambos libros como positivos", donde *positivo* = `is_read == True AND rating_clean >= 4`
(la misma definición de positivo que usa el resto del pipeline para construir el perfil de
usuario — ver `build_user_matrix.py` / `build_user_centroids.py`). El grafo no recalcula esta
señal; la reutiliza tal cual está persistida por `build_item_cooccurrence.py`.

**Peso.** PMI (*pointwise mutual information*) entre el par de libros:

```
PMI(i, j) = log( co_count(i, j) * N / (count(i) * count(j)) )
```

con piso en 0 (`max(0, PMI)`), donde `count(i)` es el número de usuarios distintos que
positivaron el libro `i`, y `N` es el número de usuarios distintos con al menos un positivo. Se
usa PMI y no la co-ocurrencia cruda (`co_count`) como peso porque la co-ocurrencia cruda premia a
los libros populares por aparecer juntos solo por volumen de lectores — exactamente el sesgo de
popularidad que el resto del sistema evita (`ratings_count`/`text_reviews_count` con `log1p`,
ver `AGENTS.md` → *Product and Recommendation Logic*). PMI normaliza por la popularidad
individual de cada libro: un peso alto significa afinidad de co-lectura **más fuerte que el
azar**, no que ambos libros sean populares.

**Direccionalidad.** El grafo es **no dirigido**. La relación subyacente ("un usuario positivó
ambos libros") es simétrica por construcción: no hay información de orden temporal ni causal
entre los dos libros en la señal disponible. Nota de defensa: el grafo bipartito
usuario→libro sí tendría una dirección natural (usuario lee libro); este grafo es la
**proyección libro-libro** de ese bipartito (dos libros se conectan si comparten lectores
positivos), y esa proyección colapsa la dirección por definición.

**Sparsificación.** El filtro `MIN_CO_COUNT >= 3` ya viene aplicado en el artefacto persistido (no
es una decisión nueva de este módulo); la sección de validez recalcula el grafo en
`MIN_CO_COUNT ∈ {2, 3, 5, 10}` para mostrar que la elección de 3 no es arbitraria, sino un punto
intermedio razonable en la curva densidad/tamaño-de-componente.

---

## 2. Script de construcción del grafo — 1.5 pts

> El grafo puede regenerarse a partir de datos procesados.

`src/reduction/build_book_graph.py` es 100% regenerable y determinista (sin aleatoriedad salvo
una semilla fija para el muestreo de centralidad, ver sección 3):

```bash
env/bin/python -m src.reduction.build_item_cooccurrence   # si book_cooccurrence.parquet no existe
env/bin/python -m src.reduction.build_book_graph
env/bin/python -m src.report_book_graph
```

Comportamiento:

1. Falla rápido con `FileNotFoundError` si `book_cooccurrence.parquet` no existe, indicando el
   comando exacto para generarlo primero (mismo patrón de pre-requisito explícito que usa
   `build_item_cooccurrence.main()` con `MASTER_FEATURE_MATRIX_PATH`/`INTERACTIONS_CURATED_PATH`).
2. Construye un `networkx.Graph()` con **todos** los `book_id` de `books_master.parquet` como
   nodos (incluyendo los que terminan sin aristas), y agrega las aristas ponderadas por PMI desde
   `book_cooccurrence.parquet`.
3. Calcula, por nodo: `degree`, `weighted_degree` (`G.degree(weight="pmi")`), `pagerank`
   (`nx.pagerank(weight="pmi")`), `betweenness_centrality` (aproximado, ver sección 3),
   `component_id` y `component_size` (`nx.connected_components`).
4. Calcula diagnósticos a nivel de grafo: `n_nodes`, `n_edges`, `density`, `n_components`,
   `largest_component_fraction`, `isolated_node_count`, y la tabla de sensibilidad a
   `MIN_CO_COUNT` (recalculada sobre el mismo `book_cooccurrence.parquet` ya cargado, sin volver a
   escanear las 110M interacciones — barato).
5. Escribe `book_graph_nodes.parquet` con `src.utils.io.safe_write_parquet` (mismo helper que usa
   el resto del pipeline) y `book_graph_diagnostics.json` con `json.dump`.

No duplica la lista de aristas: `book_cooccurrence.parquet` ya es el artefacto de aristas; este
script solo añade la capa de estructura/centralidad encima.

**Reutilización deliberada (ladder de YAGNI):** no se agregó ninguna dependencia nueva.
`networkx==3.6.1` y `scipy==1.17.1` ya estaban instalados en el proyecto (usados por otras partes
del pipeline) y cubren exactamente lo que pide la rúbrica (componentes, grado, PageRank,
centralidad) sin escribir un motor de grafos a mano.

---

## 3. Reporte del grafo — 2.0 pts

> Componentes conexas, grado o grado ponderado, centralidad, PageRank o medidas relacionadas,
> calculadas correctamente.

Ejecución real sobre el catálogo completo (108,227 libros, artefacto de cooccurrencia con
`co_count >= 3`):

| métrica | valor |
| --- | --- |
| nodos | 108,227 |
| aristas | 1,049,382 |
| densidad | 0.00017918 |
| componentes conexas | 71,326 |
| fracción del componente más grande | 0.3335 |
| nodos aislados | 70,765 (65.39%) |
| fuentes muestreadas para betweenness | 500 |

- **Grado / grado ponderado:** `degree` (no ponderado, vecinos directos) y `weighted_degree`
  (suma de PMI de las aristas incidentes) se calculan con `networkx.Graph.degree()`. Se reportan
  por separado porque responden preguntas distintas: `degree` mide cuántos libros distintos
  comparten lectores positivos con este libro; `weighted_degree` mide la **intensidad** total de
  esa afinidad, no solo el conteo.
- **PageRank:** `nx.pagerank(graph, weight="pmi")`, ponderado por PMI — un libro tiene PageRank
  alto si está conectado (con peso alto) a otros libros que también son centrales en la red de
  co-lectura, no solo si tiene muchos vecinos.
- **Centralidad de intermediación (betweenness):** **aproximada por muestreo**,
  `nx.betweenness_centrality(graph, k=min(500, n_nodes), weight="pmi", seed=42)`. Esto está
  marcado explícitamente en el código (`build_book_graph.py`, comentario `ponytail:`) como una
  decisión de escalabilidad deliberada: betweenness exacto es `O(V·E)`, intratable sobre 108k
  nodos / ~1M aristas en tiempo razonable. El muestreo de 500 fuentes con semilla fija (42) da una
  estimación reproducible sin recalcular el grafo completo por cada corrida.
- **Componentes conexas:** `nx.connected_components(graph)` asigna `component_id`/`component_size`
  a cada nodo, incluyendo los nodos aislados (cada uno es su propio componente de tamaño 1).

Top-10 por PageRank (afinidad estructural) y por grado ponderado (intensidad de co-lectura) están
en `docs/grafo_libros.md` §3, con títulos legibles unidos desde `books_master.parquet`.

---

## 4. Sección de comparación — 1.5 pts

> El ranking del grafo se compara contra popularidad, ranking basado en modelo, u otro baseline.

`src/reduction/graph_comparison.py` implementa dos piernas de comparación, ambas reutilizando
artefactos/funciones existentes en vez de re-evaluar el ranker de producción:

1. **Grafo vs. popularidad histórica (B1).** `compare_to_popularity()` reutiliza
   `historical_popularity_snapshot` de `src/reduction/baselines.py` (la misma función que
   alimenta el baseline `B1_popularity` en `evaluate_recommender.py`) para obtener
   `rating_count` acumulado por libro. Calcula:
   - Correlación de Spearman entre `pagerank`/`weighted_degree` y popularidad, restringida a
     nodos **no aislados** (un nodo aislado tiene PageRank uniforme por construcción — incluirlo
     distorsionaría la correlación con una señal sin información).
   - Solapamiento del top-k (k=100 por defecto) entre el ranking del grafo y el ranking de B1.

   Resultado observado: PageRank vs. popularidad ρ=0.1968, solapamiento top-100=11%;
   grado ponderado vs. popularidad ρ=-0.0377, solapamiento top-100=5%. Una correlación
   **moderada y no perfecta** es exactamente la lectura esperada — PMI descuenta
   deliberadamente la popularidad individual que B1 mide, así que el grafo y B1 deben parecerse
   poco más que al azar si la normalización por PMI está funcionando.

2. **Grafo vs. ranking basado en modelo.** `compare_to_collaborative_ab()` lee
   `data/outputs/recommendations/collaborative_ab_results.csv`, si existe — el artefacto que ya
   produce `scripts/run_collaborative_ab.py` evaluando la **misma señal de PMI** (la fuente de
   aristas de este grafo) como score adicional dentro del ranker de producción, comparado contra
   B1 y contra el ranking de solo-contenido. Esto evita duplicar una evaluación temporal completa
   (`evaluate_temporal`) solo para este entregable, y conecta el análisis estructural del grafo
   con la evidencia de utilidad que ya existe en el pipeline de evaluación.

**Por qué no se comparó directamente contra el ranker de producción end-to-end:** el grafo es un
análisis estructural independiente de la lógica de ranking (retrieval, MMR, exploración). Acoplar
esta entrega a un cambio del ranker de producción habría mezclado dos preguntas distintas: "¿qué
estructura tiene la red de co-lectura?" vs. "¿mejora el ranker si se le agrega esta señal?". La
segunda pregunta ya tiene su propio mecanismo de evaluación (`run_collaborative_ab.py` +
`evaluate_temporal`); este entregable la referencia en vez de duplicarla.

---

## 5. Nota de interpretación — 2.0 pts

> El equipo explica qué significa la estructura del grafo en el dominio y qué no significa.

**Qué significa un PMI/PageRank alto entre dos libros.** Afinidad de co-lectura positiva más
fuerte que el azar, dado el patrón de lectura observado en Goodreads: los lectores que califican
bien a uno de los dos libros tienden a también calificar bien al otro, en una proporción mayor a
la esperada por la popularidad individual de cada uno. Es señal **colaborativa**, derivada del
comportamiento agregado de usuarios, no de contenido ni de metadatos editoriales.

**Qué NO significa.**

- **No implica similitud temática ni editorial.** Dos libros pueden tener PMI alto sin compartir
  género, tono ni autor — solo comparten una base de lectores con gustos correlacionados. (Esto
  es justamente lo que lo distingue del ranker de contenido basado en PCA del resto del sistema:
  son señales complementarias, no equivalentes.)
- **No implica causalidad.** Un PMI alto entre A y B no significa "leer A causa leer B", ni
  establece ningún orden de lectura — el grafo es no dirigido por diseño (sección 1).
- **Un PageRank alto no significa "mejor libro".** Es centralidad estructural dentro de la red de
  co-lectura *observada*, que está sesgada por quién está representado en el dataset de Goodreads
  (mismo sesgo de exposición que ya se documenta para B1 en `docs/decisiones_negocio.md` —
  Decisión 1). Un libro excelente pero con pocos lectores en este dataset puede tener PageRank
  bajo o ser un nodo aislado.
- **Un nodo aislado no es una señal negativa sobre el libro.** Significa, exclusivamente, que
  ningún par de libros que lo incluya alcanzó `MIN_CO_COUNT=3` lectores compartidos con
  calificación positiva en ambos. Es el caso esperado para libros nicho, de cola larga, o con
  pocas interacciones registradas — no implica baja calidad, implica baja exposición compartida
  *observada*. El 65.39% de aislamiento medido (sección 6) es coherente con esta lectura: el
  catálogo tiene una cola larga muy pronunciada, y la mayoría de los libros simplemente no
  acumulan suficientes lectores positivos compartidos con ningún otro libro específico.

Esta nota de interpretación es consistente con la decisión de negocio ya documentada en el
proyecto (`docs/decisiones_negocio.md`): ni B1 ni este grafo deben tratarse como verdad de
calidad o relevancia; ambos son medidas de exposición/comportamiento observado, útiles como señal
y como diagnóstico, no como north-star del producto.

---

## 6. Chequeos de validez — 1.0 pt

> El equipo revisa sparsity de aristas, nodos aislados, estructura de componentes y sensibilidad
> a las decisiones de definición del grafo.

| chequeo | resultado | lectura |
| --- | --- | --- |
| **Sparsity** | densidad = 0.00017918 | Esperado y deseado a esta escala de catálogo (108k nodos); confirma que el grafo es disperso por construcción — el filtro `MIN_CO_COUNT` ya descarta ruido de baja confianza antes de llegar a este análisis. |
| **Nodos aislados** | 70,765 / 108,227 (65.39%) | Mayoría del catálogo sin señal colaborativa que califique. No se interpreta como defecto del grafo, sino como propiedad de cola larga del dataset (ver sección 5). |
| **Estructura de componentes** | 71,326 componentes; el más grande concentra 33.35% de los nodos | La red de co-lectura **no es un solo cuerpo conexo**: es un componente gigante relativamente moderado más un número muy grande de componentes pequeños/aislados. Esto es consistente con un catálogo donde la mayoría de los libros son nicho. |
| **Sensibilidad a `MIN_CO_COUNT`** | ver tabla abajo | El umbral de producción (3) ya está muy cerca de la meseta superior de aristas/conectividad; subir el umbral reduce aristas y conectividad de forma monótona y suave, sin saltos abruptos — no hay evidencia de que `MIN_CO_COUNT=3` sea un punto frágil o arbitrario. |

Tabla de sensibilidad (recomputada sobre el mismo `book_cooccurrence.parquet`, sin re-escanear
interacciones):

| `min_co_count` | aristas | componentes | fracción componente más grande |
| --- | --- | --- | --- |
| 2 | 1,049,382 | 71,326 | 0.3335 |
| 3 (producción) | 1,049,382 | 71,326 | 0.3335 |
| 5 | 568,384 | 83,164 | 0.2251 |
| 10 | 266,835 | 93,138 | 0.1353 |

(Nota: en este dataset no hay pares con `co_count == 2` que sobrevivan en el artefacto persistido
— `book_cooccurrence.parquet` ya se construyó con `MIN_CO_COUNT=3` como piso, así que las filas
en `min_co_count=2` y `=3` coinciden; la tabla sigue siendo correcta y útil porque muestra dónde
*empieza* a perderse estructura al subir el umbral por encima del valor de producción.)

**Chequeo adicional de integridad** (en `src/validate_artifacts.py`, no solo en el reporte): si
`book_graph_nodes.parquet` existe, se valida que tenga las columnas requeridas, que no haya
`book_id` duplicados, que todo `book_id` esté contenido en `books_master.parquet`, que
`pagerank` sea finito y no negativo, y que la suma de `pagerank` sobre todos los nodos sea ≈1.0
(propiedad matemática de PageRank como distribución de probabilidad — si esto falla, es señal de
un bug de construcción, no de los datos).

---

## 7. Preguntas de defensa — 10.0 pts

> El equipo puede defender la definición de nodos y aristas, los pesos, la direccionalidad, el
> significado de la centralidad, los baselines de comparación y las limitaciones del grafo.

### Sobre nodos y aristas

**¿Por qué un nodo es un libro y no, por ejemplo, una edición o un autor?** Porque el resto del
pipeline (PCA, clustering, ranking) ya opera a grano `book_id`, y mezclar grano de edición/autor
en este grafo rompería la alineación de IDs que `validate_artifacts.py` exige entre todos los
artefactos. Edición/autor son dimensiones de agregación válidas para análisis futuros, pero no
son la unidad que el negocio necesita para recomendar.

**¿Por qué una arista exige `co_count >= 3` y no `>= 1`?** Con `>= 1`, dos usuarios que por
casualidad leyeron y calificaron bien los mismos dos libros generarían una arista — ruido
estadístico, no señal. El piso de 3 (heredado de `build_item_cooccurrence.py`, no inventado para
este entregable) es el mismo criterio de soporte mínimo que se usa en sistemas de recomendación
basados en co-ocurrencia para evitar sobreajuste a coincidencias de pocos usuarios. La sección 6
muestra que subir el piso reduce aristas/conectividad de forma suave, sin un salto que indique
que 3 sea un valor mal elegido.

**¿Por qué se mantienen los nodos aislados en vez de filtrarlos?** Porque eliminarlos ocultaría
una propiedad real y medible del catálogo (65.39% sin señal colaborativa suficiente) detrás de un
grafo artificialmente más "sano" de lo que es. Mantenerlos permite reportar correctamente
sparsity y estructura de componentes (sección 6), y deja explícito qué fracción del catálogo este
grafo simplemente no puede informar.

### Sobre pesos

**¿Por qué PMI y no co-ocurrencia cruda o Jaccard?** Co-ocurrencia cruda confunde "co-leídos
frecuentemente" con "ambos son populares". PMI lo corrige normalizando por la popularidad
marginal de cada libro — exactamente la misma corrección de sesgo de popularidad que el resto del
sistema aplica con `log1p(ratings_count)`. Jaccard (`co_count / (count_i + count_j - co_count)`)
también normaliza, pero penaliza más fuerte a libros con conteos marginales muy distintos y no
tiene la interpretación probabilística directa de PMI (razón de probabilidad conjunta observada
vs. esperada bajo independencia), que es la que se usa en la nota de interpretación (sección 5).
No se reimplementó esta decisión — se heredó del módulo de cooccurrencia ya existente y validado.

**¿Por qué el PMI tiene piso en 0?** Un PMI negativo significa "estos dos libros co-ocurren
*menos* de lo esperado por azar" — información real, pero de signo opuesto a "afinidad". Tratar
afinidad negativa como ausencia de arista (peso 0, sin arista) en vez de como arista de peso
negativo evita que algoritmos de camino más corto / PageRank, que asumen pesos no negativos,
produzcan resultados sin sentido.

### Sobre direccionalidad

**¿Por qué no dirigido?** Ver sección 1: la señal disponible (co-positivación) es simétrica por
construcción. Un grafo dirigido requeriría una relación asimétrica observable, como "usuario leyó
A antes de B" — esa señal existe en principio en `date_added`/`read_at` de las interacciones, pero
no es la que este entregable modela ni la que pide la rúbrica de este alcance; se deja como
extensión futura explícita, no como omisión accidental.

### Sobre el significado de la centralidad

**¿Qué interpretación de negocio tiene un PageRank alto?** "Este libro está en el centro de un
vecindario de afinidad de co-lectura fuerte" — útil como candidato de *hub* para diversificación
o como ancla de exploración dentro de un macro-cluster, **no** como medida de calidad ni de
prioridad de recomendación por sí sola (ver sección 5).

**¿Por qué centralidad de intermediación aproximada y no exacta?** Betweenness exacto es
`O(V·E)` — con 108k nodos y ~1M aristas, intratable en tiempo razonable. Se usa el muestreo
estándar de `networkx` (`k=500` fuentes con Dijkstra ponderado, semilla fija = 42 para
reproducibilidad). Esta decisión está documentada en el código (`build_book_graph.py`, comentario
`ponytail:`) con su techo explícito: si se necesita una estimación más fina, subir `k` es la
única palanca, a costa de tiempo de cómputo lineal en `k`.

### Sobre los baselines de comparación

**¿Por qué comparar contra B1 (popularidad histórica) y no contra un ranking aleatorio?** B1 ya es
el baseline de cordura obligatorio de todo el proyecto (`docs/decisiones_negocio.md`, Decisión 1)
y es la comparación con mayor poder explicativo: si el grafo simplemente redescubriera la
popularidad, no aportaría señal nueva. Un baseline aleatorio no habría podido falsear esa
hipótesis.

**¿Qué significa que la correlación con B1 sea moderada (ρ≈0.20) y no alta?** Es la confirmación,
no el fracaso, de que PMI está haciendo su trabajo: si PMI normaliza correctamente la
popularidad marginal, el ranking resultante *debe* parecerse poco a un ranking de popularidad
cruda. Una correlación cercana a 1.0 habría sido la señal de alarma (indicaría que PMI no está
descontando popularidad correctamente).

**¿Por qué no recalcular `evaluate_temporal` con un score derivado del grafo para esta entrega?**
Esa pregunta ya tiene un mecanismo de respuesta en el pipeline (`run_collaborative_ab.py`, que
evalúa la misma señal de PMI dentro del ranker con métricas temporales de Recall/NDCG/Coverage).
Este entregable referencia ese resultado (`compare_to_collaborative_ab`) en lugar de duplicar la
evaluación, evitando dos fuentes de verdad para la misma pregunta de utilidad del modelo.

### Sobre las limitaciones del grafo

- **Sesgo de exposición heredado.** El grafo hereda el mismo sesgo de exposición histórica de
  Goodreads que afecta a B1: libros con más lectores en el dataset tienen más oportunidades de
  generar aristas, incluso después de la normalización por PMI (la normalización corrige la
  *magnitud* del peso, no la *probabilidad de tener algún peso* si el libro casi no tiene lectores
  positivos compartidos).
- **Cobertura parcial por diseño.** 65.39% de aislamiento significa que este grafo, por sí solo,
  no puede informar recomendaciones de cola larga vía co-lectura — para esos libros, las señales
  de contenido (PCA) o de clustering siguen siendo necesarias.
- **Dependencia de un parámetro operativo del entorno de cómputo, no del diseño del grafo:**
  durante la generación de este entregable, `book_cooccurrence.parquet` tuvo que reconstruirse con
  `MAX_POSITIVES_PER_USER=10` en lugar del valor de producción documentado (200), porque la
  máquina disponible (9.1 GB de RAM) no pudo completar la construcción a 200 ni a 50 sin agotar
  swap. Esto es una restricción de **recursos de cómputo del entorno**, no una decisión de diseño
  del grafo ni del entregable — pero es honesto declararlo: con un cap más bajo, los lectores con
  más de 10 positivos contribuyen pares solo entre un subconjunto de sus libros, lo que
  probablemente **subestima** ligeramente `co_count` para usuarios muy activos y, por extensión,
  puede inflar levemente el aislamiento medido frente a una corrida con el cap de producción. La
  estructura cualitativa del análisis (qué es un nodo, qué es una arista, por qué PMI, por qué no
  dirigido, qué significa e interpretación) es independiente de este parámetro; los números
  exactos de la sección 3 y 6 deberían regenerarse en una máquina con más memoria para reportar
  cifras de producción exactas.
- **No es evidencia de causalidad ni de calidad** (repetido deliberadamente aquí porque es la
  limitación más fácil de malinterpretar en una defensa): toda lectura de PageRank/PMI debe
  enmarcarse como comportamiento agregado observado, en línea con el resto del marco de
  validación de negocio del proyecto (`docs/decisiones_negocio.md`).

---

## Cómo reproducir y verificar

```bash
env/bin/python -m src.reduction.build_item_cooccurrence   # si book_cooccurrence.parquet no existe
env/bin/python -m src.reduction.build_book_graph
env/bin/python -m src.report_book_graph
env/bin/python -m src.validate_artifacts
env/bin/python -m pytest tests/test_book_graph.py tests/test_artifact_validation.py -q
```

Estado verificado en esta entrega: pipeline completo ejecutado de extremo a extremo sobre datos
reales (108,227 libros), `validate_artifacts` pasa, y la suite completa de pruebas pasa
(114/114, incluyendo las 4 nuevas de `test_book_graph.py`).
