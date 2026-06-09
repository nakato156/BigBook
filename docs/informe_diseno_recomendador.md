# Informe de diseño del recomendador — BigBook

Documento consolidado de la fase de diseño conceptual del sistema de recomendación de libros.
Reúne los siete entregables de la fase en un solo informe. El detalle completo de cada punto vive
en su documento propio dentro de `docs/`.

**Objetivo de negocio (norte del proyecto):** ayudar a más personas a sostener el **hábito de
lectura**, recomendando libros relevantes, accesibles y motivadores — no solo los más populares.

**Dataset:** Goodreads Dataset Collection (UCSD), 5 géneros (fantasy/paranormal,
mystery/thriller/crime, history/biography, young adult, romance). 108,227 libros; millones de
interacciones. Es **observacional e histórico** (sin telemetría, sin transacciones, sin grafo
social).

**Artefacto central:** `data/features/master_feature_matrix.parquet` — un vector PCA de 173
dimensiones (`pc_0..pc_172`) por `book_id`.

---

## Índice

1. [Definición del problema](#1-definición-del-problema) · *(detalle: [definicion_problema.md](definicion_problema.md))*
2. [Justificación del recomendador](#2-justificación-del-recomendador) · *(detalle: [justificacion_recommender.md](justificacion_recommender.md))*
3. [Información disponible](#3-información-disponible) · *(detalle: [informacion_disponible.md](informacion_disponible.md))*
4. [Perfil del usuario](#4-perfil-del-usuario) · *(detalle: [perfil_usuario.md](perfil_usuario.md))*
5. [Representación de los ítems](#5-representación-de-los-ítems) · *(detalle: [representacion_items.md](representacion_items.md))*
6. [Criterio de similitud y scoring](#6-criterio-de-similitud-y-scoring) · *(detalle: [criterio_scoring.md](criterio_scoring.md))*
7. [Métricas de evaluación: ¿cuándo es válido?](#7-métricas-de-evaluación-cuándo-es-válido) · *(detalle: [metricas_evaluacion.md](metricas_evaluacion.md))*

> **Alcance y límites de v1:** qué resolvemos, qué admitimos y qué aplazamos (estado real de los 3
> problemas de negocio + 4 contradicciones, en tres cubos) → [alcance_y_limitaciones.md](alcance_y_limitaciones.md).

---

## 1. Definición del problema

**¿Qué recomienda el sistema y a quién?** Enunciado en forma X / Y / Z:

> **Recomendamos libros (`book_id`, cada uno un vector PCA `pc_0..pc_172`) a lectores (`user_id`),
> con base en el gusto multidimensional inferido de su historial de lectura y en la similitud
> libro-a-libro en el espacio reducido, buscando optimizar lecturas relevantes y motivadoras que
> se empiezan y se completan, para que los lectores sostengan el hábito de lectura.**

```text
Recomendamos   X = libros del catálogo (vectores PCA) agrupados en clusters de gusto
con base en    Y = historial de interacción + similitud en espacio PCA + clusters de libros
                   (género = filtro/explicación/diversidad; popularidad = diagnóstico de exposición)
para optimizar Z = lecturas empezadas y terminadas (proxy: is_read + rating positivo) que
                   sostienen el hábito (retención), evitando sesgo de popularidad y burbujas
```

| Concepto | Definición en este proyecto |
|---|---|
| **Usuario** | Lector (`user_id`) = **vector de gusto** agregado de su historial de lectura. No es una etiqueta de género. |
| **Ítem** | Libro (`book_id`) = **un vector PCA** (`pc_0..pc_172`). El género es señal, no la unidad. |
| **Recomendación útil** | Lista corta, **relevante** (afinidad antes que popularidad), **accesible**, **diversa pero coherente** (cross-género) y **explicable** (por cluster/género). |
| **Acción objetivo** | **Empezar y completar una lectura** (`is_read`), reforzada por rating alto/review. North-star = **retención del hábito**. No compra ni clic. |

---

## 2. Justificación del recomendador

**¿Tiene sentido recomendar por similitud en este dominio?** Sí, **pero con condiciones**.

- El consumo de libros es un problema de **similitud/descubrimiento** ("dame otro libro que
  disfrute"), no de complementariedad de cesta. La complementariedad existe solo como
  **progresión** (sagas vía `series`, rampa de accesibilidad).
- La similitud aporta valor **solo si se mide sobre el vector de gusto multidimensional** (PCA:
  contenido + metadata + género), **no sobre el género** ni **sobre la popularidad**. Riesgo
  real: el primer eje del PCA (`pc_0`) es popularidad, así que una similitud cruda recomendaría
  "igual de popular", no "del mismo gusto".
- Riesgo de **"más de lo mismo" (filtro burbuja)**: especialmente grave aquí, porque encerrar al
  lector mata el hábito. Se neutraliza con diversidad y progresión.

**Conclusión:** la similitud es **necesaria pero no suficiente** — motor correcto siempre que se
mida sobre el gusto, saque la popularidad del orden explícito y se complemente con diversidad.

---

## 3. Información disponible

| Bloque | Tenemos | No tenemos / débil |
|---|---|---|
| **Ítems (libros)** | Metadata (título, descripción, género multi-etiqueta, idioma, páginas, año, autores, serie), popularidad (`ratings_count`, `average_rating`), **embeddings** de la descripción | `format`, `publisher`, `theme_*`, tags libres (fuera del master) |
| **Usuarios** | `user_matrix` (vector PCA), `user_meta` (comportamiento) y `user_centroids` (modos de lectura); todo derivado de interacciones | Sin demografía, sin perfil social, sin datos declarados |
| **Interacciones** | `is_read`, `rating`/`rating_clean`, `has_review_text`, `engagement_mode`, `reading_duration_days`, timestamps | **Sin clicks/views/sesiones, sin compras/precio, sin likes/follows** |

Señales más fuertes: **contenido del ítem** (metadata + semántica) y **feedback de lectura**
(ratings, lecturas, reviews con timestamp). Más débiles/ausentes: **telemetría, demografía,
social, transacciones**. Metadata con nulos relevantes: `num_pages` (17.5%), `publication_year`
(20.1%) — imputados con flag de missingness.

> **Nota de estado:** la representación de usuario ya está implementada en el mismo espacio PCA que
> los libros. Los artefactos actuales son `user_matrix.parquet`, `user_meta.parquet` y
> `user_centroids.parquet`, construidos desde el canonical global
> `data/processed/interactions_curated.parquet`. El ranking v1 implementa multi-centroides,
> exclusión de consumidos, MMR, exploración y cold-start escalonado. El runner temporal B0/B1/B2
> y los proxies N1 están implementados; los resultados se publican en `docs/estado_v1.md`.

---

## 4. Perfil del usuario

La preferencia **no se declara, se infiere del comportamiento** (relación **asociativa, no
causal**: leer y calificar alto *correlaciona* con agrado, no lo prueba).

- **Fuente:** historial de consumo (`is_read`, `engagement_mode`) como base, con ratings
  (`rating_clean`, `user_rating_bias`) y reviews (`has_review_text`) como señal de intensidad.
- **Definición conceptual:** el perfil es un **vector de gusto en el mismo espacio PCA de los
  libros**, construido como **agregación de los vectores de los libros que el usuario leyó y
  valoró positivamente**. Es una **estimación** inferida del comportamiento, no la medición de una
  preferencia "verdadera".
- **Decisiones fijadas (build):** positivo = `is_read AND rating_clean ≥ 4`; agregación = **media
  simple** en `user_matrix` (baseline reproducible; la ponderación por rating/recencia no mueve la
  geometría baseline):

  ```text
  positivos(u) = { b : is_read ∧ rating_clean ≥ 4 }
  vector_gusto(u) = (1/|positivos(u)|) · Σ_b pca(b)      # media simple
  ```
- **Dos capas:** gusto (`user_matrix`/`user_centroids`) + comportamiento (`user_meta`: cuánta
  confianza y diversidad inyectar).
- **Multi-centroides:** `user_centroids` conserva varios modos de lectura por usuario cuando hay
  suficiente historial. `weight` indica proporción de libros del modo y `centroid_weight` resume
  compromiso/hábito con rating, review y duración de lectura.
- **No usa** compras/clicks (no existen); **explícito solo en cold-start**.
- **Reales vs. simulados:** validación con **usuarios reales** del dataset (split temporal: mide
  *predicción*, no causalidad); perfiles **semilla/simulados** para cold-start y demo.
- **Cold-start (escalonado):** semillas explícitas → *shrinkage* hacia el centroide del cluster →
  perfil individual completo, evitando popularidad.
- **Estado:** implementado en `src/reduction/build_user_matrix.py` y
  `src/reduction/build_user_centroids.py`. Salidas:
  `user_matrix.parquet` (`user_id` + `pc_0..pc_172`), `user_meta.parquet` (comportamiento) y
  `user_centroids.parquet` (`user_id`, `centroid_id`, `weight`, `centroid_weight`, `pc_0..pc_172`).

---

## 5. Representación de los ítems

Cada libro = **un vector PCA de 173 dimensiones**; la comparabilidad es la **distancia en ese
espacio común**. Representación **híbrida** en tres bloques (276 cols → PCA 95% varianza → 173):

| Bloque | Cols | Qué aporta |
|---|---:|---|
| **Numérico** | 9 | Calidad/popularidad atenuada (`log1p`), accesibilidad (páginas), época, autores, missingness |
| **Binario/categórico** | 11 | Género multi-etiqueta, `series` (progresión), idioma |
| **Embeddings** | 256 | **Semántica** de la descripción: tono, temática, audiencia (lo cross-género) |

**Block-weighting** (`1/sqrt(dim)`) es crítico: evita que los 256 embeddings dominen el PCA por
conteo de columnas (`embedding_dominated_first_5_count = 0`).

**Captura lo importante** (contenido, popularidad sin dominar, género, accesibilidad, época), pero
con **límites**: PCA lineal, ejes no interpretables 1:1, popularidad como primer eje, dependencia
de la descripción, idioma casi constante, atributos fuera del master.

---

## 6. Criterio de similitud y scoring

**Criterio primario:** **similitud coseno** entre el vector de gusto del usuario y el vector PCA
del libro. "Más cercano" = afinidad de tono/temática/accesibilidad, no de género. Se usa coseno
(dirección del gusto) y no euclidiana cruda (que reintroduce sesgo de popularidad por magnitud).

**Ranking operativo v1:**

```text
elegible(b) = id/título/vector PCA/cluster válidos
score(u,b)  = similitud_interés(u,b)         ← coseno en subespacio de gusto
lista_base  = MMR semántico + penalización por género
              + desempate suave por accesibilidad
exploración = fuera de vecindad + ≥75% de la mejor similitud
              + solo tail/mid
```

Regla de oro: **interés primero**. Popularidad no filtra elegibilidad ni ordena los slots normales:
segmenta el catálogo para
medir y controlar exposición. Con los datos actuales, `tail <= 436` y `head >= 6,017` ratings.

**Arquitectura del ranking:** `retrieve` (candidatos por cluster/macro-cluster cercano) → `score`
→ `diversify` (MMR semántico, género, accesibilidad suave y slots solo tail/mid) → `explain`
(por vecindad/género). Escalable,
explicable y **evaluable** con `Recall@k`/`NDCG@k`/`MAP` (relevancia), `Coverage`/`Novelty`/
`Diversity` (anti-popularidad) y proxies de hábito, con **split temporal** sobre `is_read`.

---

## 7. Métricas de evaluación: ¿cuándo es válido?

Un recomendador no es válido por ordenar bien en abstracto, sino porque **realmente sostiene el
hábito de lectura**. Como el dataset es **observacional** (no A/B test), no medimos causalidad
directa: usamos **proxies de hábito** y **split temporal**.

**Cómo medimos el hábito** (proxies por usuario, derivados de `interactions_curated.parquet`):
`completion_rate`, `reading_frequency`, `active_span_days`, `activity_recency` (churn),
`reading_breadth` (`category_count`). La **acción objetivo** es `is_read` (empezar/completar una
lectura), reforzada por rating alto y review.

**El hábito se conserva como norte, en una escalera de evidencia de 3 niveles** (la métrica no
cambia; cambia la *fuerza de la evidencia*):

| Nivel | Qué afirma | Métrica | Evidencia | Disponible |
|---|---|---|---|---|
| **N0** Relevancia (split temporal) | ¿Predice lo que el lector leyó **después**? (una *puerta*, no el hábito) | `Recall@k`, `NDCG@k`, `MAP` + `Coverage`/`Novelty`/`Diversity` | Predictiva | Hoy |
| **N1** Proxy de hábito | ¿Cómo cambian los proxies futuros según la actividad previa? | `completion_rate`, `reading_frequency`, `reading_breadth`, `activity_recency` | Descriptiva/correlacional | Hoy |
| **N2** Hábito causal | ¿Recomendar así *aumenta* la retención? | mismos proxies como *lift* A/B + señales en vivo | Causal | Con telemetría |

> N0 (`Recall@k`) es **relevancia, no hábito**: acertar el próximo libro es condición necesaria, no
> el objetivo. El hábito vive en N1/N2 (proxies). No conflacionar un `Recall@k` temporal con una
> métrica de hábito.

**Criterio de validez:** el sistema es válido si (1) **supera a la base de popularidad** en
`Recall@k`/`NDCG@k` y (2) mejora o conserva `Coverage`, `Long-tail Coverage`, `Novelty` y
`Diversity`. N1 describe diferencias de hábito por actividad previa, pero no es un gate atribuible
al recomendador porque el dataset no contiene exposición real al sistema.

**Límite explícito:** el impacto causal real sobre la retención requiere **telemetría de producto
en vivo** (sesiones, retornos, libros terminados tras una recomendación), fuera del alcance del
dataset estático. El runner offline produce la evidencia N0/N1 y el veredicto reproducible en
`docs/estado_v1.md`.

---

## Hilo conductor del diseño

Las cinco decisiones cuelgan de un solo principio: **modelar el gusto como un vector
multidimensional, priorizar el interés sobre la popularidad, y proteger la diversidad para
sostener el hábito de lectura.**

```text
Problema       →  recomendar libros (X) a lectores según su gusto (Y) para sostener el hábito (Z)
Justificación  →  la similitud sirve, si se mide sobre el gusto y se diversifica
Información     →  tenemos contenido + feedback de lectura; no telemetría ni social
Perfil usuario  →  user_matrix + user_centroids: gusto PCA inferido de consumo positivo
Representación   →  vector PCA híbrido por libro (texto + metadata + popularidad atenuada)
Scoring          →  coseno + MMR/género + accesibilidad suave; exploración solo tail/mid
Evaluación       →  válido si predice lo que se lee DESPUÉS (split temporal) y se asocia a
                    mejores proxies de hábito (correlacional, no causal — A/B queda fuera de alcance)
```

Todo en un **espacio común PCA** donde usuario y libros son comparables, el género es señal y no
unidad, y la popularidad es una medida de exposición, no un factor explícito de orden.
