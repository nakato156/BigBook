# Información disponible para construir el recomendador

Fase de definición de datos. El objetivo es mostrar **qué datos tenemos realmente** para
construir recomendaciones, distinguir lo fuerte de lo débil, y declarar **qué usaremos** en
el recomendador y qué dejaremos fuera (y por qué). Fuente: Goodreads Dataset Collection
(UCSD), 5 géneros, ~108k libros. Las variables se toman tal cual existen en el código
(`src/config.py`, `src/curation/interactions.py`, `src/reduction/build_user_matrix.py`,
`src/reduction/build_user_centroids.py`, `src/reduction/build_master_feature_matrix.py`,
`src/merge_master.py`) y el README.

## Mapa rápido del catálogo

| Entidad | Granularidad | Artefacto | Filas |
|---|---|---|---:|
| Ítems (libros) | 1 fila = 1 `book_id` | `data/processed/books_master.parquet` | 108,227 |
| Ítems (vector) | 1 fila = 1 `book_id` | `data/features/master_feature_matrix.parquet` (`pc_0..pc_172`) | 108,227 |
| Interacciones | 1 fila = 1 (`user_id`,`book_id`) | `data/processed/interactions_curated.parquet` global | — |
| Usuarios (stats) | 1 fila = 1 `user_id` | `data/processed/user_features_global.parquet` | — |
| Usuarios (vector) | 1 fila = 1 `user_id` con positivos | `data/features/user_matrix.parquet` | — |
| Usuarios (metadatos) | 1 fila = 1 `user_id` presente | `data/features/user_meta.parquet` | — |
| Usuarios (modos) | 1..m filas por `user_id` | `data/features/user_centroids.parquet` | — |

---

## 1. Datos de ítems (libros)

Es nuestra señal **más fuerte y completa**: 108,227 libros sin `book_id` duplicado, sin
nulos en las columnas clave del master. Dos fuentes complementarias:

**a) Metadata del master (`books_master.parquet`, 17 columnas):**

| Variable | Tipo | Qué es | Nulos |
|---|---|---|---|
| `book_id` | str | Identificador Goodreads (clave) | 0 |
| `title` | str | Título | 0 |
| `description` | str | Descripción (texto libre) | 0 (vacío si falta) |
| `series` | int 0/1 | Pertenece a una saga | 0 |
| `language_code` | str | Idioma (eng, en-US, en-GB, en-CA) | 0 |
| `average_rating` | float | Rating medio global | 0 |
| `ratings_count` | int | Nº de ratings (popularidad) | 0 |
| `text_reviews_count` | int | Nº de reviews de texto | 0 |
| `num_pages` | float | Nº de páginas | **17.5%** |
| `publication_year` | float | Año de publicación | **20.1%** |
| `author_count` | int | Nº de autores | 0 |
| `genre_fantasy/mystery/history/ya/romance` | int 0/1 | 5 flags de género (multi-etiqueta) | 0 |
| `genre_count` | int | Nº de géneros activos (1–3) | 0 |

**b) Embeddings semánticos** (`description_embeddings.parquet`, 256 dims `emb_0..emb_255`)
generados con `google/embeddinggemma-300m` sobre la descripción (fallback: título →
`[no description]`). Capturan **tono, temática y contenido** del libro más allá del género.

**c) Pipeline por categoría (más rico, pero más antiguo)** — `book_features.parquet` añade
señales que el master *descarta*: `is_ebook`, `format`, `publisher`, `primary_author_id`,
`series_count`, `top_shelf`, `to_read_count`, `publication_day/month`, y los `theme_*`
(temas). Existen pero hoy **no entran** en la representación PCA.

---

## 2. Datos de usuarios

Existen en tres capas, derivadas **enteramente del historial de interacciones** (no hay perfil
demográfico):

| Variable | Qué captura |
|---|---|
| `user_features_global`: `user_mean_rating`, `user_rating_std`, `user_rating_count`, `user_rating_bias`, `read_or_rated_count`, `valid` | Sesgo y confiabilidad mínima del usuario en el canonical global |
| `user_matrix`: `user_id + pc_0..pc_172` | Vector baseline de gusto en el mismo espacio PCA que los libros |
| `user_meta`: `positive_count`, `interaction_count`, `review_count`, `want_to_read_count`, `user_rating_bias`, `category_count`, `last_date_added`, `is_cold_start` | Evidencia, compromiso, diversidad y confianza del perfil |
| `user_centroids`: `centroid_id`, `n_books`, `weight`, `centroid_weight`, `pc_0..pc_172` | Modos de lectura intra-usuario; `centroid_weight` resume compromiso/hábito del modo |

**Importante:** un usuario **no** se modela como una etiqueta de género, sino como un
**vector de gusto multidimensional** ya implementado en `user_matrix`: media simple de los
vectores PCA de libros con interacción positiva (`is_read=True` y `rating_clean >= 4`). Para
usuarios con historial suficiente, `user_centroids` conserva varios modos de lectura en vez de
forzar un único promedio.

---

## 3. Historial de interacciones

La señal que conecta usuarios e ítems (`interactions_curated.parquet`, 1 fila por
`user_id`×`book_id`):

| Variable | Qué es | Rol recsys |
|---|---|---|
| `user_id`, `book_id` | Quién interactúa con qué | Claves del par |
| `is_read` | Lectura completada | **Acción objetivo** (proxy principal) |
| `rating`, `rating_clean`, `rating_missing` | Calificación (1–5) | Feedback explícito |
| `has_review_text` | Escribió review | Compromiso alto |
| `engagement_mode` | `want_to_read`, `read_no_rating`, `rating_only`, `review` | Separa intención de señales leídas/calificadas/revisadas |
| `reading_duration_days`, `has_reading_duration` | Duración real de lectura | Lectura efectiva vs. intención |
| `user_rating_bias` | Sesgo del usuario al calificar | Normalización |
| `date_added`, `date_updated` | Cuándo ocurrió | Base para split temporal y cadencia |
| `review_id` | Identificador de review | Trazabilidad |

Métricas de hábito **derivables** de estos campos (aún no almacenadas como columnas):
`active_span_days`, `reading_frequency`, `activity_recency` (proxy de churn),
`completion_rate`, `reading_breadth`.

### Tipos de feedback: qué tenemos y qué no

| Señal típica de recsys | ¿Disponible? | Detalle |
|---|---|---|
| Ratings (explícito) | ✅ Sí | `rating` 1–5 + `average_rating` agregado |
| Reviews de texto | ✅ Sí | `has_review_text` (+ descripciones para embeddings) |
| Lecturas / "consumo" | ✅ Sí (proxy) | `is_read`, `reading_duration_days` |
| Saves / wishlists | ✅ Sí | `engagement_mode = want_to_read`, `is_want_to_read`, `want_to_read_count` |
| Categorías / tags | ✅ Parcial | 5 flags de género; `theme_*`, `top_shelf` solo en pipeline viejo |
| **Clicks / views / sesiones** | ❌ **No** | Dataset es histórico, no de telemetría |
| **Compras / precio** | ❌ **No** | No es un dataset transaccional |
| **Likes / follows / social** | ❌ **No** | No hay grafo social |
| **Datos demográficos** | ❌ **No** | No hay edad, ubicación, idioma del usuario |

---

## 4. Qué datos faltan o son débiles

- **Sin señales de navegación (clicks, views, sesiones, dwell time).** El dataset es
  observacional/histórico, no telemetría de producto. → No podemos medir impacto causal en
  retención; solo proxies offline con split temporal (ver README, *Evaluation layers*).
- **Sin demografía ni contexto del usuario** (edad, país, dispositivo, momento del día). El
  usuario es 100% "comportamiento", no "perfil".
- **Sin datos sociales** (amigos, follows, likes) → no es posible recsys social/grafo.
- **Sin transacciones** (compras, precio, conversión a venta).
- **Metadata con nulos relevantes:** `num_pages` (17.5%) y `publication_year` (20.1%). Se
  imputan por mediana **conservando flags de missingness** (`*_missing`) porque la ausencia
  es señal (calidad de metadata, tipo de edición).
- **Sesgo de popularidad estructural:** `ratings_count`/`text_reviews_count` con colas
  enormes (máx ~4.9M ratings). Señal fuerte pero peligrosa → se controla con `log1p` y
  block-weighting.
- **`language_code` casi constante:** todo inglés (4 variantes); `language_code_other`
  vacío. Poco poder discriminante real.
- **Fechas de lectura dispersas:** `started_at`/`read_at` y `reading_duration_days` dependen
  de lo que cada usuario rellenó → métricas de duración se reportan junto a
  `has_reading_duration_rate` para ser honestos con la cobertura.
- **Spot checks semánticos débiles:** las pruebas de coseno del README son diagnósticos
  pequeños, no validación de calidad.
- **Ranking y evaluación implementados:** retrieval, scoring por interés, MMR, exploración
  controlada y cold-start viven en `src/reduction/recommend.py`; N0 y N1 descriptivo se generan con
  `src/reduction/evaluate_recommender.py` y se resumen en `docs/estado_v1.md`.

---

## 5. Qué variables usaremos en el recomendador

**Representación del ítem (núcleo del modelo) — vector PCA `pc_0..pc_172`**, que ya
combina tres bloques:

- **Numérico (9):** `average_rating`, `log_ratings_count`, `log_text_reviews_count`,
  `num_pages`(+`_missing`), `publication_year`(+`_missing`), `author_count`, `genre_count`.
- **Binario/categórico (11):** 5 flags `genre_*`, `series`, one-hot de `language_code`.
- **Embeddings (256):** semántica de la descripción.

Sobre ese vector se calcula la **similitud entre libros** y se hace el **clustering**
(KMeans k=100 + 10 macro-clusters), que es la base de la recomendación.

**Representación del usuario:** vector de gusto = **agregación de los vectores PCA** de los
libros con interacción positiva (`is_read=True` y `rating_clean >= 4`). El baseline
`user_matrix` usa media simple. `user_centroids` divide historiales amplios en varios modos de
lectura y usa `centroid_weight` como señal de compromiso: rating, review y duración coherente.

**Para evaluación y ranking (no para definir el gusto):**

- `date_added`/`date_updated` → **split temporal** (entrenar en pasado, evaluar en futuro).
- `is_read` (+ `rating` alto, `has_review_text`) → **etiqueta objetivo** (Recall@k, NDCG…).
- `ratings_count` → segmentación dinámica `tail/mid/head` y métricas de exposición; no filtra ni
  ordena. `average_rating` permanece en la representación PCA, no como factor explícito del ranker.
- `genre_*`, `category_count`, `weight`, `centroid_weight` → **filtro, explicación, confianza y
  control de diversidad/hábito** (anti-burbuja).

**Qué dejamos fuera (por ahora):** `clicks/views/compras` (no existen), demografía y social
(no existen), y las señales del pipeline viejo (`format`, `publisher`, `to_read_count`,
`theme_*`, `is_ebook`) que no están integradas en el PCA — quedan como mejoras futuras.

---

### Conclusión

Tenemos un **dominio rico en contenido de ítem** (metadata + semántica) y **medio rico en
interacción** (ratings, lecturas, reviews, saves con timestamp), pero **pobre en
telemetría, demografía y señal social**. Por eso el recomendador se ancla en la
**similitud de contenido/gusto sobre el vector PCA del libro**, usa el historial de
interacciones para construir el vector de usuario y para evaluar con split temporal, y
trata la popularidad como diagnóstico de exposición y el género como señal de diversidad — no
como objetivos ni factores explícitos de orden.
