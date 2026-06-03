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

## 3. Las tres capas de evaluación

### Capa 1 — Evaluación offline con split temporal *(hacible hoy)*

Para cada usuario, ordenar sus interacciones por `date_added`, **entrenar con el pasado y
retener el futuro**. Medir si el recomendador habría mostrado los libros que el lector
**realmente leyó después** (`is_read = True`, idealmente con rating alto):

- **Relevancia del ranking:** `Recall@k`, `Precision@k`, `NDCG@k`, `MAP`.
- **Anti-popularidad / descubrimiento:** `Coverage`, `Novelty`, **Diversity** intra-lista — para
  confirmar que el modelo **no** se limita a amplificar bestsellers (hace cumplir la regla de
  sesgo de popularidad).

> El **split temporal** es lo que convierte una métrica recsys estándar en una métrica orientada
> al hábito: la pregunta no es *"¿acertó lo que ya leyó?"* sino *"¿lo que recomienda coincide con
> lo que el lector **sigue** leyendo?"*.

### Capa 2 — Evaluación por proxy de hábito

Comparar el recomendador por **similitud de interés** contra una **línea base de popularidad** y
verificar si los lectores expuestos a recomendaciones por vecindad muestran mayor
`completion_rate`, `reading_frequency` y `reading_breadth` (`category_count`). En entorno offline
esto es **correlacional, no causal**.

### Capa 3 — Telemetría de producto *(fuera del alcance del dataset)*

Medir el impacto causal real en retención exige instrumentar la plataforma viva: sesiones,
retornos, libros terminados **después** de una recomendación, conversión click → lectura. Es
trabajo de producto futuro, **no disponible** en el dump estático, y se declara como limitación
explícita.

---

## 4. Criterio de validez (cómo decidimos "sí sirve")

El recomendador se considera **válido** si, en evaluación offline con split temporal:

1. **Supera a la línea base de popularidad** en `Recall@k` / `NDCG@k` (predice mejor lo que el
   lector leyó después que recomendar solo bestsellers).
2. **No lo logra a costa de la diversidad**: mantiene `Coverage`/`Novelty`/`Diversity` por encima
   de la base de popularidad (no es una burbuja ni un amplificador de bestsellers).
3. **Correlaciona con mejores proxies de hábito** (Capa 2): los grupos servidos por vecindad
   tienden a mayor `completion_rate` / `reading_frequency` / `reading_breadth`.

Si gana relevancia pero **colapsa la diversidad**, o si mejora métricas de ranking pero **no se
asocia a más lectura completada**, el recomendador **no** se considera válido para el objetivo de
hábito, aunque sus números recsys "se vean bien".

---

## 5. Limitaciones honestas de la medición

- El dataset es **observacional**: las métricas offline **aproximan**, no **prueban**, impacto
  causal en el hábito.
- `started_at` / `read_at` y `reading_duration_days` son **dispersos** según lo que cada usuario
  rellenó → las métricas de duración se reportan junto a `has_reading_duration_rate` para ser
  honestos con la cobertura.
- La capa de perfil usuario↔libro ya tiene artefactos en el mismo espacio PCA (`user_matrix`,
  `user_meta`, `user_centroids`). Lo que sigue pendiente es implementar y ejecutar la capa final
  de retrieval → scoring → diversificación → evaluación temporal; por eso esta evaluación sigue
  siendo el **plan de validación**, no un resultado ya medido.

---

### Conclusión (para el entregable)

El recomendador se declara **válido** no por ordenar bien en abstracto, sino por **predecir lo
que el lector sigue leyendo** y **asociarse a más lectura completada y diversa**. Se mide en tres
capas: **relevancia** (`Recall@k`, `NDCG@k`, `MAP` con **split temporal** sobre `is_read`),
**anti-popularidad** (`Coverage`, `Novelty`, `Diversity`) y **proxy de hábito** (`completion_rate`,
`reading_frequency`, `reading_breadth`, `activity_recency`) frente a una línea base de popularidad.
El impacto causal real sobre la retención requiere telemetría de producto en vivo, que se reconoce
como límite explícito del dataset estático.
