# Métricas de evaluación — ¿cuándo el recomendador es válido?

Pieza esencial del diseño: un recomendador no es "bueno" porque ordene bien, sino porque
**realmente mejora o mantiene el hábito de lectura**. Este documento define **con qué métricas**
declaramos válido el sistema y, sobre todo, **cómo medimos el hábito** —no solo la relevancia—.
Es la base honesta de todo lo anterior: el scoring (ver [criterio_scoring](criterio_scoring.md))
solo tiene sentido si podemos comprobar que su orden conduce a más lectura sostenida.

> **El reto de fondo.** "Hábito de lectura" es un resultado **longitudinal y causal**: ¿el lector
> sigue leyendo *porque* recomendamos bien? El dataset Goodreads (UCSD) es **observacional e
> histórico**, no un A/B test en vivo. Por tanto **no podemos medir causalidad directamente**.
> Lo que sí podemos: (1) derivar **proxies de hábito** de las señales temporales y de
> comportamiento; (2) evaluar **offline con split temporal** ("qué leyó el lector después"); y
> (3) documentar la **telemetría de producto** que haría falta para medir impacto causal real
> (fuera del alcance del dataset estático).

---

## 1. Cómo medimos el "hábito de lectura"

El objetivo `Z = sostener el hábito` solo es accionable si se mide. Lo operacionalizamos con
**proxies por usuario**, derivados de los campos `date_*` y de los agregados de interacción:

| Métrica de hábito (por usuario) | Definición | Qué indica |
|---|---|---|
| `active_span_days` | última − primera interacción | Cuánto tiempo el lector sigue activo |
| `reading_frequency` | lecturas completadas / `active_span` | Regularidad (p. ej. lecturas/mes) |
| `activity_recency` | días desde la última interacción | **Proxy de churn**: más alto = más en riesgo |
| `completion_rate` | lecturas completadas / `interaction_count` | Cuánto de lo que guarda termina leyendo |
| `reading_breadth` | `category_count` | Diversidad de lectura (señal anti-burbuja) |

Un lector "con hábito" muestra `active_span` amplio, `reading_frequency` regular,
`activity_recency` baja y `completion_rate` alta. **Estos valores por usuario son la etiqueta de
resultado** contra la que se evalúa el recomendador. (Aún no son columnas almacenadas; se computan
desde `interactions_curated.parquet`.)

---

## 1bis. La escalera de evidencia del hábito (N0 → N1 → N2)

El hábito **no es una hipótesis vaga**: es este conjunto de proxies. Lo que cambia con el tiempo no
es la *métrica*, sino la **fuerza de la evidencia** con que podemos afirmar que el recomendador
influye en ella. Por eso la promesa se estructura en tres niveles, y **cada afirmación se etiqueta
con su nivel** para no inflar lo que aún no se puede probar:

| Nivel | Qué afirmamos | Métrica | Tipo de evidencia | Disponible |
|---|---|---|---|---|
| **N0 — Acción** | "Predecimos la próxima lectura relevante" | `Recall@k`/`NDCG@k` sobre `is_read` futuro (split temporal) | **Predictiva** (relevancia) | **Hoy** |
| **N1 — Hábito correlacional** | "El modelo *se asocia* a lectores que leen más, terminan más y leen más amplio" | los 5 proxies como *outcome label* por usuario | **Correlacional** | **Hoy** |
| **N2 — Hábito causal** | "Recomendar así *aumenta* la frecuencia/retención de lectura" | los **mismos** 5 proxies como *lift* tratamiento-vs-control + señales en vivo (retornos, libros terminados tras una recomendación) | **Causal (A/B)** | **Con telemetría** |

> **Separación que evita el overclaim:** N0 (`Recall@k`) es una **puerta de relevancia** —condición
> necesaria, *no* es el hábito—. El hábito vive en N1/N2 (los proxies). Un `Recall@k` con split
> temporal sigue siendo una métrica de **relevancia** ("¿acerté el próximo libro?"), no de **hábito**
> ("¿lee más a lo largo del tiempo?"). No se deben conflacionar.

El norte del proyecto (mantener el hábito) **se conserva**: hoy lo medimos de forma correlacional
(N1) con proxies derivables del dataset; con telemetría mediremos los mismos proxies de forma causal
(N2). Lo único que maduramos es la evidencia, no el objetivo.

---

## 2. La acción objetivo y su jerarquía de señales

El sistema intenta provocar **empezar y completar una lectura**. En los datos el proxy observable
es `is_read = True`, reforzado por rating alto y/o review escrita:

| Nivel | Señal en los datos | Rol |
|---|---|---|
| **Acción objetivo** | `is_read` (lectura) | Lo que optimizamos |
| Confirmación de calidad | `rating` alto, `has_review_text` | Refuerzo de la señal positiva |
| Engagement / interés | click / abrir ficha del libro | Proxy temprano (a instrumentar en el producto) |
| **Objetivo de negocio** | retención / hábito | Métrica norte (north-star) |

---

## 3. Las tres capas de evaluación (= los tres niveles de evidencia)

Las tres capas implementan la escalera de §1bis: **Capa 1 → N0**, **Capa 2 → N1**, **Capa 3 → N2**.

### Capa 1 (N0) — Evaluación offline con split temporal *(hacible hoy)*

Se usa un único corte global reproducible sobre `date_added` válidos (desde `2006-01-01`):
**entrenar con interacciones hasta el corte y retener el futuro**. El corte puede fijarse con
`--cutoff` o derivarse como el percentil `--train-fraction` de la cohorte. Se mide si el
recomendador habría mostrado libros disponibles que el lector **realmente leyó después**
(`is_read = True`, idealmente con rating alto):

- **Relevancia del ranking:** `Recall@k`, `Precision@k`, `NDCG@k`, `MAP`.
- **Anti-popularidad / descubrimiento:** `Coverage`, `Long-tail Coverage`, `Novelty`,
  **Diversity**, mix `tail/mid/head` y `Average Recommendation Popularity`.
- **Slots exploratorios:** relevancia y tasa de acierto de `slot=exploration` separadas de
  `slot=interest`, para verificar que la exposición adicional no sea irrelevante.

El modo implementado es `global_historical_snapshot_frozen_representation`. B1/B2, los segmentos
`tail/mid/head`, novedad y popularidad media se calculan con conteos y promedios observados solo
hasta el corte. Un libro con año conocido requiere `publication_year <= año del corte`; sin año,
requiere una primera interacción válida hasta el corte. Los objetivos del holdout que no estaban
disponibles se excluyen del denominador.

> El **split temporal** mejora la *honestidad predictiva* de la métrica de relevancia: la pregunta
> deja de ser *"¿acertó lo que ya leyó?"* y pasa a ser *"¿lo que recomienda coincide con lo que el
> lector **sigue** leyendo?"*. Pero ojo: esto **sigue siendo relevancia (N0)**, no hábito. Acertar
> el próximo libro es condición necesaria; medir si el lector *lee más a lo largo del tiempo* es la
> Capa 2 (N1). No confundir un `Recall@k` temporal con una métrica de hábito.

### Capa 2 (N1) — Evaluación por proxy de hábito *(correlacional, hoy)*

Comparar, sobre los mismos usuarios históricos, la calidad predictiva del recomendador y sus
proxies de hábito con una **línea base de popularidad**. Offline no hay usuarios realmente
expuestos al sistema: solo puede estudiarse asociación entre perfiles, predicciones y
`completion_rate`, `reading_frequency` o `reading_breadth`. Esto es correlacional, no causal.

### Capa 3 (N2) — Telemetría de producto *(causal, futuro)*

Medir el impacto causal real en retención exige instrumentar la plataforma viva: sesiones,
retornos, libros terminados **después** de una recomendación, conversión click → lectura. Es
trabajo de producto futuro, **no disponible** en el dump estático, y se declara como limitación
explícita.

---

## 4. Criterio de validez (cómo decidimos "sí sirve")

El recomendador se considera **válido** si, en evaluación offline con split temporal:

1. **Supera a la línea base de popularidad** en `Recall@k` / `NDCG@k` (predice mejor lo que el
   lector leyó después que recomendar solo bestsellers).
2. **No lo logra a costa del descubrimiento**: mejora o conserva `Coverage`, `Long-tail Coverage`,
   `Novelty`, `Diversity` y el mix de exposición frente a la base de popularidad.
3. **Correlaciona con mejores proxies de hábito** (Capa 2): los grupos servidos por vecindad
   tienden a mayor `completion_rate` / `reading_frequency` / `reading_breadth`.

Si gana relevancia pero **colapsa la diversidad**, o si mejora métricas de ranking pero **no se
asocia a más lectura completada**, el recomendador **no** se considera válido para el objetivo de
hábito, aunque sus números recsys "se vean bien".

---

## 5. Limitaciones honestas de la medición

- El dataset es **observacional**: las métricas offline **aproximan**, no **prueban**, impacto
  causal en el hábito.
- **Brecha de atribución (la grande).** Offline, los proxies describen el hábito *del usuario*, no
  el *efecto del recomendador* sobre él: esos libros se leyeron sin que el sistema existiera. Por eso
  N1 es correlacional **por construcción**, y solo N2 (telemetría, A/B) cierra la brecha. Esta es la
  *razón de ser* de la escalera de §1bis.
- **`reading_frequency` divide por `active_span`**, así que usuarios de una sola interacción dan
  `active_span = 0` (división por cero). Requiere un piso (p. ej. *clamp* a 1 día, o excluir `n = 1`)
  al computar la métrica.
- **`completion_rate` offline está sesgado por lo que cada usuario *registró* en Goodreads**, no por
  lo que leyó de verdad; en producto vivo (N2) la señal es limpia. El proxy offline es más ruidoso
  que su versión telemétrica — mismo nombre, distinta calidad.
- `started_at` / `read_at` y `reading_duration_days` son **dispersos** según lo que cada usuario
  rellenó → las métricas de duración se reportan junto a `has_reading_duration_rate` para ser
  honestos con la cobertura.
- **Fuga transductiva residual:** PCA, embeddings y clusters permanecen congelados con artefactos
  del catálogo completo. El snapshot corrige la fuga operativa principal de popularidad y
  disponibilidad, pero no constituye un backtest estricto. Reconstruir representación y clustering
  por snapshot queda fuera de este alcance.
- La capa de perfil, ranking, split temporal, cohorte global `valid`, cortes `k = 5, 10, 20`,
  baselines B0/B1/B2, `MAP`, diversidad, exposición y métricas por slot están implementados en
  `src/reduction/evaluate_recommender.py`. Sigue pendiente ejecutar y reportar resultados sobre la
  cohorte acordada; no son resultados ya medidos.

---

### Conclusión (para el entregable)

El norte —**mantener el hábito de lectura**— se conserva, estructurado en una **escalera de
evidencia de tres niveles** (§1bis): **N0** relevancia (`Recall@k`/`NDCG@k` con split temporal sobre
`is_read` — una *puerta*, no el hábito), **N1** proxy de hábito correlacional (`completion_rate`,
`reading_frequency`, `reading_breadth`, `activity_recency`) y **N2** hábito causal con telemetría
(A/B + señales en vivo). El recomendador se declara **válido** no por ordenar bien en abstracto,
sino por **superar a la popularidad en relevancia (N0) sin colapsar la diversidad** y **asociarse a
mejores proxies de hábito (N1)**. La afirmación causal ("recomendar así *aumenta* la retención")
queda etiquetada como **N2** y requiere telemetría de producto en vivo, límite explícito del dataset
estático: lo que madura con los datos es la **fuerza de la evidencia**, no el objetivo.
