# QA — Defensa del proyecto (últimos 2 commits)

20 preguntas y respuestas sobre lo introducido por `8ba33cb` (refactor del ranker/evaluador) y
`ff770f2` (grafo de co-lectura de libros). Pensado como material de preparación para defensa oral.

---

**1. ¿Qué es PMI y por qué se usó como peso de las aristas del grafo?**

PMI (Pointwise Mutual Information) mide cuánto más co-ocurren dos eventos de lo esperado por
azar: `PMI(i,j) = log(co_count(i,j) * N / (count(i) * count(j)))`, con piso en 0. Se usó porque
normaliza por la popularidad individual de cada libro — sin eso, dos libros muy populares
tendrían peso alto solo por volumen de lectores, no por afinidad real. Es la misma corrección de
sesgo de popularidad que el resto del sistema aplica con `log1p(ratings_count)`.

**2. ¿El grafo se construye por usuario o sobre todo el catálogo?**

Sobre todo el catálogo: un solo grafo global libro-libro (108,227 nodos). No hay un grafo por
usuario. El usuario participa solo como fuente de la señal agregada (define qué pares de libros
co-ocurren), pero no aparece como nodo — es la proyección libro-libro del bipartito
usuario→libro.

**3. ¿Por qué un nodo es un libro y no una edición o un autor?**

Porque todo el pipeline (PCA, clustering, ranking) ya opera a grano `book_id`. Mezclar grano de
edición/autor rompería la alineación de IDs que `validate_artifacts.py` exige entre artefactos.

**4. ¿Por qué la arista exige `co_count >= 3` y no `>= 1`?**

Con `>= 1`, dos usuarios que por coincidencia leyeron los mismos dos libros generarían una
arista — ruido estadístico, no señal. El piso de 3 (heredado de `build_item_cooccurrence.py`) es
un criterio mínimo de soporte estándar en recomendación por co-ocurrencia.

**5. ¿Por qué el grafo es no dirigido?**

Porque la señal disponible ("un usuario positivó ambos libros") es simétrica por construcción —
no hay orden temporal ni causal en esa señal. Un grafo dirigido necesitaría algo como "leyó A
antes de B", que no es lo que este entregable modela.

**6. ¿Por qué se mantienen los nodos aislados en vez de eliminarlos?**

Porque eliminarlos ocultaría una propiedad real del catálogo: 65.39% de los libros no alcanza el
soporte mínimo de co-lectura. Mantenerlos permite reportar correctamente sparsity y estructura de
componentes en vez de mostrar un grafo artificialmente "más sano".

**7. ¿Qué significa un PageRank alto en este grafo, en términos de negocio?**

Que el libro está en el centro de un vecindario de afinidad de co-lectura fuerte — útil como
candidato de *hub* para diversificación o ancla de exploración. **No** significa "mejor libro" ni
debe usarse como prioridad de recomendación por sí solo.

**8. ¿Qué NO significa un PMI o PageRank alto entre dos libros?**

No implica similitud temática/editorial (pueden no compartir género ni autor), no implica
causalidad ni orden de lectura, y no es una medida de calidad — es comportamiento agregado
observado, sesgado por quién está representado en el dataset de Goodreads.

**9. ¿Por qué la centralidad de intermediación (betweenness) es aproximada y no exacta?**

Betweenness exacto es `O(V·E)`, intratable con ~108k nodos y ~1M aristas. Se usa el muestreo
estándar de `networkx` con `k=500` fuentes y semilla fija (42) para reproducibilidad. Está
marcado en el código con un comentario `ponytail:` que documenta el techo y la palanca (subir
`k` si se necesita más precisión).

**10. ¿Contra qué se comparó el grafo y por qué esos baselines?**

Contra B1 (popularidad histórica, vía `historical_popularity_snapshot`) por correlación de
Spearman y overlap de top-k, y contra el resultado ya existente de `run_collaborative_ab.py`
(que evalúa la misma señal PMI dentro del ranker de producción). B1 es el baseline de cordura
obligatorio del proyecto: si el grafo solo redescubriera popularidad, no aportaría señal nueva.

**11. ¿Por qué la correlación con popularidad es baja (ρ≈0.20) y eso es bueno?**

Porque confirma que PMI está haciendo su trabajo: si normaliza correctamente la popularidad
marginal, el ranking del grafo *debe* parecerse poco a un ranking de popularidad cruda. Una
correlación cercana a 1.0 habría sido la señal de alarma.

**12. ¿Por qué no se evaluó el grafo end-to-end dentro del ranker de producción en este
entregable?**

Porque es una pregunta distinta ("¿qué estructura tiene la red de co-lectura?" vs. "¿mejora el
ranker si se agrega esta señal?"). La segunda ya tiene su propio mecanismo
(`run_collaborative_ab.py` + `evaluate_temporal`); este entregable la referencia en vez de
duplicarla.

**13. ¿Qué limitación de cómputo afectó los números reportados?**

`book_cooccurrence.parquet` se reconstruyó con `MAX_POSITIVES_PER_USER=10` en vez del valor de
producción (200) porque la máquina disponible no soportó la construcción completa sin agotar
memoria. Es una restricción de entorno, no de diseño — probablemente subestima `co_count` para
usuarios muy activos e infla levemente el aislamiento medido.

**14. ¿Qué columnas/propiedades valida `validate_artifacts.py` sobre el grafo?**

Que `book_graph_nodes.parquet` tenga las columnas requeridas, sin `book_id` duplicados, que todo
`book_id` esté contenido en `books_master`, que `pagerank` sea finito y no negativo, y que la
suma total de `pagerank` sea ≈1.0 (propiedad matemática de PageRank como distribución de
probabilidad).

**15. ¿Por qué se hizo el refactor de `recommend.py` / `evaluate_recommender.py`?**

Ambos archivos habían crecido como "god-files" mezclando retrieval, scoring, diversificación,
orquestación, baselines, métricas y proxies de hábito. Encontrar la lógica concreta era lento. El
refactor separa por las costuras ya documentadas `retrieve -> score -> diversify -> explain`.

**16. ¿El refactor cambió el comportamiento del sistema?**

No. Es relocalización pura de código con ajustes de imports; los comandos CLI
(`python -m src.reduction.recommend`, `python -m src.reduction.evaluate_recommender`) producen el
mismo resultado que antes.

**17. ¿Dónde quedó la lógica de candidatos/elegibilidad después del split?**

En `src/reduction/retrieval.py`: `eligibility_mask` (elegibilidad técnica, nunca por
popularidad), `popularity_segments`, `retrieve_top_clusters`/`retrieve_clusters_per_mode`. Es un
módulo sin estado y sin dependencia de la clase `Recommender`.

**18. ¿Dónde quedaron los baselines B0/B1/B2 y las métricas de evaluación?**

Los baselines (snapshots históricos, rankers de control) en `src/reduction/baselines.py`; las
métricas N0 de ranking (recall/precision/NDCG/AP, bootstrap CI) en `src/reduction/metrics.py`; los
proxies N1 de hábito (descriptivos, no causales) en `src/reduction/habit_proxies.py`.

**19. ¿Por qué `temporal_split.py` no importa la clase `Recommender`?**

Es deliberado: lo usan `baselines.py`, `habit_proxies.py` y `evaluate_recommender.py`, y mantenerlo
sin esa dependencia evita un ciclo de imports entre módulos de evaluación y el módulo de
recomendación.

**20. Si tuviera que extender el grafo a una versión dirigida en el futuro, ¿qué señal usaría?**

`date_added` / `read_at` de las interacciones ya existe en el dataset y permitiría inferir "leyó A
antes de B" — la señal asimétrica necesaria para direccionalidad. No se usó en este entregable
porque no era el alcance pedido; queda como extensión futura explícita, no como omisión.

---

Fuente de detalle ampliado: [`Deliverable5.md`](Deliverable5.md) (rúbrica del grafo) y
[`docs/transferencia_proyecto.md`](docs/transferencia_proyecto.md) (mapa de código de ambos
commits).
