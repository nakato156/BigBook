# Definición del perfil de usuario (user profile)

Fase de definición del usuario. El objetivo es explicar **de dónde sale la preferencia del
usuario**, decidir qué señales la componen, y definir conceptualmente el *user profile* que
alimentará al recomendador.

> **Nota de estado del proyecto.** El perfil de usuario ya tiene artefactos implementados en
> el mismo espacio PCA que los libros: `data/features/user_matrix.parquet`,
> `data/features/user_meta.parquet` y `data/features/user_centroids.parquet`. Se construyen
> desde la fuente global canónica `data/processed/interactions_curated.parquet`; los archivos
> per-categoría quedan como respaldo histórico/EDA, no como fuente principal del perfil.

---

## 1. ¿De dónde sale la preferencia del usuario?

En nuestro dominio (Goodreads, dataset observacional/histórico) la preferencia **no** se
declara: se **infiere del comportamiento de lectura**. La fuente única y verificada es
`interactions_curated.parquet` (1 fila = un par `user_id`×`book_id`; ~7.16M filas solo en
fantasy_paranormal). Sus columnas reales son la materia prima del perfil:

| Señal | Columnas | Qué aporta al perfil |
|---|---|---|
| **Consumo / lectura** | `is_read`, `engagement_mode`, `read_at`, `started_at`, `reading_duration_days` | Qué leyó de verdad vs. qué solo guardó. Señal **principal** de preferencia, *inferida del comportamiento* (asociativa, no causal). |
| **Ratings (explícito implícito)** | `rating_clean`, `rating_missing`, `user_mean_rating`, `user_rating_std`, `user_rating_count`, `user_rating_bias` | Dirección del gusto y sesgo del usuario al calificar. En el baseline implementado se usa como filtro: `rating_clean >= 4`. |
| **Reviews (compromiso alto)** | `has_review_text`, `review_text_length` | Escribir una review = implicación alta; en `user_centroids` refuerza `centroid_weight`. |
| **Temporalidad** | `date_added`, `date_updated` | Cadencia, recencia y split temporal para evaluación. |

### Decisiones explícitas (lo que el enunciado pide resolver)

| Pregunta | Decisión | Por qué |
|---|---|---|
| ¿Viene del **historial de consumo**? | **Sí, es la base.** `is_read` + `engagement_mode` definen el núcleo de la preferencia. | Es preferencia *inferida del comportamiento* (asociativa): leer y calificar alto **correlaciona** con agrado; es la señal más confiable disponible en datos observacionales, no una prueba de causalidad. |
| ¿Viene de **ratings**? | **Sí, como filtro en el baseline.** Positivo = `is_read=True` y `rating_clean >= 4`. | El rating da dirección (gustó/no gustó). La ponderación por rating/review/recencia queda fuera de la geometría baseline y se usa como señal de compromiso en `user_centroids`. |
| ¿Viene de **compras/clicks**? | **No.** No existen en el dataset (no es transaccional ni hay telemetría). | Limitación honesta; se documenta como trabajo de producto futuro. |
| ¿Viene de **preferencias explícitas**? | **Solo en cold-start.** Géneros/libros semilla elegidos al registrarse. | No hay perfil declarado en los datos; el explícito solo cubre el arranque. |

---

## 2. Definición conceptual del user profile

El perfil **no es una etiqueta de género** ni una lista de libros. Es un **vector de gusto
multidimensional en el mismo espacio PCA que los libros** (`pc_0..pc_172`). Esto permite
comparar usuario y libro con la misma métrica de similitud y descubrir patrones cross-género
(p. ej. *tono juvenil + romance + aventura + fantasía ligera*).

```text
perfil(u) = agregación de los vectores PCA de los libros
            con los que u interactuó positivamente

positivos(u) = { b : is_read(u,b)=True ∧ rating_clean(u,b) ≥ 4 }
vector_gusto(u) = (1/|positivos(u)|) · Σ_{b ∈ positivos(u)} pca(b)
```

Esta es la decisión implementada para `user_matrix`: media simple (`w = 1`) de positivos
limpios. Las señales de compromiso no mueven la geometría del centroide baseline; se reservan
para metadatos, evaluación y `centroid_weight` en el artefacto multi-centroide:

```text
engagement_weight = (rating_clean - 3)
                    · (1.3 si has_review_text)
                    · (1.2 si 1 <= reading_duration_days <= 180)
```

El perfil se acompaña de **metadatos de comportamiento** (agregados por `user_id`, derivables
de la misma tabla) que **no definen el gusto pero contextualizan la confianza y la
diversidad** a aplicar:

| Agregado (derivable) | Rol |
|---|---|
| `positive_count`, `interaction_count` | Cuánta evidencia hay → confianza del perfil |
| `review_count`, `want_to_read_count` | Compromiso fuerte vs. intención pendiente |
| `user_rating_bias` | Normalización del rating del usuario |
| `category_count` / amplitud de géneros leídos | Señal anti-burbuja: ¿es lector mono-género o ecléctico? |
| `last_date_added` | Frescura del perfil / riesgo de churn |
| `is_cold_start` | Perfil con menos de 3 positivos; baja confianza |

> **Dos capas del perfil:**
> 1. **Capa de gusto** = `user_matrix` o `user_centroids` → *qué* recomendar (similitud).
> 2. **Capa de comportamiento** = `user_meta` → *cuánta confianza* tener y *cuánta diversidad*
>    inyectar.

---

## 3. ¿Usuarios reales o perfiles simulados?

**Ambos, en dos contextos distintos:**

- **Evaluación / desarrollo offline → usuarios reales.** El dataset trae `user_id` reales con
  historiales extensos (millones de interacciones). Sobre ellos se construyen perfiles reales
  y se evalúa con **split temporal** (entrenar con el pasado del usuario, predecir lo que leyó
  después). No necesitamos simular para validar el modelo.
- **Producto en vivo / demo → perfiles semilla (parcialmente simulados).** Un usuario nuevo de
  la plataforma no tiene historial Goodreads; su perfil arranca de preferencias explícitas
  (géneros/libros semilla) y se va volviendo "real" a medida que lee. Para *demostrar* el
  recomendador antes de tener tráfico, se pueden construir perfiles sintéticos sembrando 3–5
  libros y agregando sus vectores PCA.

**Decisión:** la validación se hace con **usuarios reales del dataset**; los perfiles
simulados se reservan para demo y para probar el flujo de cold-start.

---

## 4. Usuarios sin historial suficiente (cold-start)

Es el caso crítico para el objetivo de *crear hábito de lectura* (justo los usuarios nuevos).
Estrategia escalonada según cuánta evidencia hay:

| Nivel de historial | Estrategia de perfil |
|---|---|
| **Sin historial (cold-start puro)** | Libros semilla opcionales: el perfil es la media de sus vectores PCA. Sin semillas → un libro accesible por macro-cluster, respetando exclusiones. |
| **Historial mínimo (1–2 positivos)** | Perfil híbrido: 50% vector individual + 50% centroide del cluster más cercano. |
| **Historial suficiente** | Perfil = vector de gusto agregado completo (sección 2). |

Principios para cold-start (alineados con el negocio):

- **Umbral de confianza:** definir un mínimo de `positive_count`/`interaction_count` para "confiar"
  en el perfil individual; por debajo, apoyarse en clusters.
- **Evitar sesgo de popularidad** incluso en el arranque: no llenar de bestsellers; usar la
  jerarquía de macro-clusters para ofrecer variedad accesible.
- **Aprendizaje incremental:** cada `is_read`/`rating` nuevo actualiza el vector de gusto, así
  el cold-start se disuelve rápido con el uso.
- **Exclusión obligatoria:** los libros ya consumidos se excluyen en perfiles normales, semillas,
  shrinkage y fallback sin historial.

---

## 5. Qué señales NO usamos (y por qué)

- **Clicks, views, sesiones, dwell time** → no existen (dataset histórico, sin telemetría).
- **Compras, precio, conversión** → no es un dataset transaccional.
- **Likes, follows, grafo social, demografía** → no disponibles; el usuario es 100%
  comportamiento de lectura, sin perfil social ni demográfico.

---

## 6. Construcción implementada del perfil (`user_matrix` y `user_centroids`)

Receta concreta implementada sobre la fuente global canónica.
**Estado: implementado y cubierto por tests** (`tests/test_user_matrix.py`,
`tests/test_user_centroids.py`).

**Decisiones fijadas:**
- **Interacción positiva** = `is_read == True` **AND** `rating_clean >= 4` (señal limpia).
- **Ponderación** = **media simple** (`w_i = 1`) como baseline reproducible. La ponderación
  por rating/recencia de la sección 2 queda como refinamiento posterior, no para el primer corte.
- **Implementación baseline** = módulo `src/reduction/build_user_matrix.py` (invocable con `-m`).
- **Implementación multi-centroide** = módulo `src/reduction/build_user_centroids.py` (invocable con `-m`).
- **Modo de hábito/engagement** = `centroid_weight` usa rating, review y duración de lectura para
  indicar qué sub-centroide representa un modo de lectura más comprometido.

> **Nota sobre causalidad (importante).** La construcción es **asociativa, no causal**. Que un
> usuario tenga `is_read=True` y `rating_clean ≥ 4` es un **proxy de comportamiento** que
> *correlaciona* con que el libro le gustó; **no probamos** que el libro *causara* su agrado, ni
> que leerlo *cause* hábito de lectura. El `user_vec` es por tanto una **estimación** del gusto
> inferida del historial, no una medición de una preferencia "verdadera". Toda afirmación de
> impacto causal (p. ej. "recomendar así aumenta la retención") queda **fuera de alcance** con
> este dataset observacional y solo sería verificable con un experimento controlado (A/B) en
> producción.

### Inputs
```text
data/processed/interactions_curated.parquet     — fuente global canónica de interacciones
data/processed/user_features_global.parquet     — censo de usuarios + user_rating_bias
data/processed/books_master.parquet             — genre_* para category_count
data/features/master_feature_matrix.parquet     — book_id + pc_0..pc_172 (espacio de ítems)
```

### Pasos

| # | Paso | Detalle |
|---|---|---|
| 1 | **Filtrar positivos** | `is_read == True AND rating_clean >= 4` por fila de interacción. `want_to_read` y ratings bajos quedan fuera del vector. |
| 2 | **Mapear libro → vector** | Cargar los 108k vectores PCA en memoria (~75 MB) como `book_id → pc_vec`; mapear, **no** hacer join pesado de millones de filas. |
| 3 | **Acumular por usuario** | Leer el canonical global en chunks; acumular `Σ pc_vec` y `count` por `user_id`. |
| 4 | **Centroide (media simple)** | `user_vec(u) = Σ pc_vec / count` — **estimación** del vector de gusto en el espacio PCA (inferida del comportamiento, no medición directa). |
| 5 | **Tabla lateral de comportamiento** | Aparte: `positive_count`, `interaction_count`, `review_count`, `want_to_read_count`, `user_rating_bias`, `category_count`, `last_date_added`, `is_cold_start`. No entran al vector de gusto. |
| 6 | **Cold-start** | `count < 3` positivos → `is_cold_start=True`; no confiar en el centroide (ver sección 4). |
| 7 | **(Eval) split temporal** | Para evaluar: agregar solo el **pasado** del usuario (orden por `date_added`), dejar el futuro como holdout. Mide **capacidad predictiva** (¿el modelo ordena lo que el usuario efectivamente leyó después?), **no** impacto causal del recomendador sobre la lectura. |
| 8 | **Persistir** | `safe_write_parquet`, `float32`, paths desde `src/config.py`. |

### Multi-centroides (`user_centroids`)

`user_matrix` promedia todo el gusto del usuario en un solo vector. Para usuarios con varios modos
de lectura, `user_centroids` conserva sub-centroides:

```text
positive_count < 6  →  m = 1
positive_count ≥ 6  →  m = min(4, positive_count // 3)
```

Cada sub-centroide es la media de un cluster KMeans sobre los libros positivos del usuario. El
campo `weight` indica proporción de libros del modo; `centroid_weight` indica compromiso relativo
del modo usando `rating_clean`, `has_review_text` y `reading_duration_days`.

### Fórmula (baseline)
```text
positivos(u) = { b : is_read(u,b)=True ∧ rating_clean(u,b) ≥ 4 }
user_vec(u)  = (1/|positivos(u)|) · Σ_{b ∈ positivos(u)} pca(b)      # media simple
```

### Outputs
```text
data/features/user_matrix.parquet   # user_id + pc_0..pc_172   (vector de gusto, mismo espacio que libros)
data/features/user_meta.parquet     # user_id + agregados de comportamiento (confianza/diversidad)
data/features/user_centroids.parquet # user_id + centroid_id + weight/centroid_weight + pc_0..pc_172
```

### Validación
- **Sanity**: un usuario que solo leyó fantasía debe caer cerca del cluster de fantasía.
- **Cuantitativa**: `Recall@k` / `NDCG@k` sobre el holdout temporal vs. baseline de popularidad.
  Es una comparación **correlacional** de calidad de ranking; si el modelo supera al baseline,
  eso indica mejor *predicción*, no que *cause* más lectura ni más hábito.

> Con `user_matrix` en el mismo espacio que `master_feature_matrix`, recomendar = buscar los
> libros vecinos del `user_vec`, diversificar con MMR y reservar exploración relevante hacia
> segmentos menos expuestos. La popularidad se mide, pero no ordena el ranking.

---

### Conclusión (para el entregable)

El perfil de usuario es un **vector de gusto multidimensional en el espacio PCA de los libros**,
**inferido del historial de consumo**. El baseline implementado (`user_matrix`) se construye como
media simple de los libros que el usuario **leyó y valoró positivamente** (`is_read=True` y
`rating_clean >= 4`). `user_meta` conserva agregados de comportamiento para confianza,
cold-start y diversidad. `user_centroids` añade una representación multi-modo: varios
sub-centroides por usuario cuando hay historial suficiente, con `centroid_weight` como señal de
compromiso/hábito basada en rating, review y duración de lectura. No usa compras ni clicks (no
existen) y solo usa preferencias explícitas en el arranque. La validación se hace con usuarios
reales del dataset vía split temporal; mide capacidad predictiva, no impacto causal sobre hábito.
