# Alcance y limitaciones de v1

Este documento fija qué resuelve BigBook hoy, qué debe medirse y qué queda fuera. La regla central
es separar tres conceptos que antes estaban mezclados: **elegibilidad técnica**, **popularidad** y
**descubrimiento**.

## 1. Estado de los problemas de negocio

| # | Problema | Estado en v1 |
|---|---|---|
| **P1** | Sostener el hábito de lectura | Objetivo de negocio. N0/N1 son evaluación predictiva y correlacional; el efecto causal requiere telemetría N2. |
| **P2** | Evitar sesgo de popularidad | Objetivo principal. La popularidad no filtra ni ordena el ranking; se usa para medir exposición y segmentar resultados. |
| **P3** | Exponer libros menos explorados | **Objetivo secundario medible.** v1 reserva exploración controlada para libros afines de cola/media, sin prometer cobertura total. |

P3 vuelve al alcance porque el catálogo real no presenta la tensión que se había supuesto. En
`books_master.parquet`, `ratings_count` tiene mínimo **49**, percentil 25 **436**, mediana **840** y
percentil 90 **6,017**. El antiguo gate `ratings_count >= 5` conservaba **108,227/108,227 libros**:
era un no-op y no eliminaba ninguna cola larga.

## 2. Decisiones implementadas

### A1 — Mitigación geométrica

La similitud de interés excluye `pc_0..pc_5`, los ejes tempranos más tabulares. Esto reduce la
influencia directa de popularidad, idioma y missingness, pero **no demuestra que desaparezcan de
toda la geometría**: PCA mezcla señales y los clusters se ajustaron con el espacio completo. A1 es
una mitigación pendiente de validación empírica, no una eliminación perfecta del sesgo.

### A2 — Elegibilidad técnica, sin gate de popularidad

Un libro participa si:

- tiene `book_id` y título no vacíos;
- tiene vector PCA finito;
- tiene una asignación de cluster válida.

`ratings_count` no excluye libros y no entra al score. Su función es diagnóstica: medir qué parte
de las recomendaciones cae en cabeza, zona media o cola.

### A3 — Exploración controlada por afinidad y exposición

En un top-10 se reservan por defecto **2 slots exploratorios**. Los candidatos:

1. están fuera de los macro-clusters de la vecindad recuperada;
2. conservan al menos el **75% de la mejor similitud** de interés del usuario;
3. solo admiten segmentos `tail` o `mid`, priorizando `tail` y luego similitud;
4. respetan exclusiones de libros ya consumidos;
5. si ninguno supera el piso de relevancia, los slots vuelven al ranking normal de interés.

Los segmentos se recalculan desde el catálogo, no se hardcodean:

```text
tail = ratings_count <= percentil 25
mid  = percentil 25 < ratings_count < percentil 90
head = ratings_count >= percentil 90
```

Con los artefactos actuales: `tail <= 436` y `head >= 6,017`.

### A4 — Cold start sin bestseller list

El fallback mantiene un libro técnicamente elegible y accesible por macro-cluster. No ordena por
popularidad. `num_pages` es un proxy débil de accesibilidad y debe presentarse como limitación.

En el ranking normal, `num_pages` también actúa como desempate suave: favorece libros más cortos
entre candidatos de afinidad comparable, sin reemplazar la similitud como criterio principal.

La diversidad de la lista base combina MMR semántico con una penalización explícita cuando un
candidato repite los géneros ya presentes en la lista.

## 3. Cómo se mide P2/P3

Además de `Recall@k`, `NDCG@k` y diversidad, la evaluación debe reportar:

- **Catalog Coverage**: fracción del catálogo que aparece al menos una vez.
- **Long-tail Coverage**: fracción del segmento `tail` que recibe exposición.
- **Mix de exposición**: porcentaje recomendado de `tail`, `mid` y `head`.
- **Average Recommendation Popularity**: media de `log1p(ratings_count)` en las listas.
- **Novelty**: menor popularidad histórica implica mayor novedad.
- **Relevancia por tipo de slot**: métricas separadas para `interest` y `exploration`.

No se declara éxito por tener dos slots etiquetados como exploración. P3 solo mejora si aumenta la
exposición de cola/media sin una caída inaceptable de relevancia.

## 4. Límites de v1

- El catálogo está previamente curado: P3 mejora exposición **dentro de esos 108,227 libros**, no
  descubre obras fuera del dataset.
- No se resuelve item cold start para libros nuevos sin vector, metadata o cluster.
- No hay bandit, aprendizaje por feedback en línea ni presupuesto adaptativo de exploración.
- Los segmentos de popularidad miden evidencia histórica, no calidad literaria.
- La afinidad sigue dependiendo de una representación PCA híbrida con señal residual de
  popularidad.
- El efecto causal sobre hábito requiere producto vivo y experimento A/B.

## 5. Estado de implementación

| Punto | Estado |
|---|---|
| A1 subespacio sin `pc_0..pc_5` | Implementado; pendiente medir sesgo residual |
| A2 elegibilidad técnica | Implementado en `eligibility_mask` |
| Segmentos dinámicos `tail/mid/head` | Implementados en `popularity_segments` |
| A3 exploración con piso de relevancia | Implementado en `select_exploration_rows` |
| Fallback a interés si explorar degrada demasiado | Implementado |
| A4 cold start diverso y sin orden de popularidad | Implementado |
| Evaluación temporal N0 y baselines B0/B1/B2 | Implementada; resultados pendientes de ejecutar/reportar |
| Item cold start y exploración adaptativa | Fuera de v1 |

La siguiente deuda no es volver a diseñar el gate, sino ejecutar la evaluación temporal y comprobar
si el nuevo mix de exposición mejora `Long-tail Coverage` y `Novelty` manteniendo relevancia.
