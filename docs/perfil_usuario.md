# Definición del perfil de usuario (user profile)

Fase de definición del usuario. El objetivo es explicar **de dónde sale la preferencia del
usuario**, decidir qué señales la componen, y definir conceptualmente el *user profile* que
alimentará al recomendador.

> **Nota de estado del proyecto.** El commit `5191bcd` *eliminó* el pipeline legacy de
> features de usuario (`src/reduction/feature_matrix.py` y su salida
> `user_features_global.parquet`). La fuente cruda **sigue intacta** en
> `data/processed/<category>/interactions_curated.parquet` y la matriz usuario/interacción
> **se reconstruirá desde cero** sobre ella. Por eso este documento es una **definición
> conceptual** (qué *será* el perfil), no la descripción de un artefacto ya construido. La
> representación de ítems (`master_feature_matrix.parquet`, `pc_0..pc_172`) sí está intacta.

---

## 1. ¿De dónde sale la preferencia del usuario?

En nuestro dominio (Goodreads, dataset observacional/histórico) la preferencia **no** se
declara: se **infiere del comportamiento de lectura**. La fuente única y verificada es
`interactions_curated.parquet` (1 fila = un par `user_id`×`book_id`; ~7.16M filas solo en
fantasy_paranormal). Sus columnas reales son la materia prima del perfil:

| Señal | Columnas | Qué aporta al perfil |
|---|---|---|
| **Consumo / lectura** | `is_read`, `engagement_mode` (`shelf_only` vs. leído), `read_at`, `started_at`, `reading_duration_days`, `has_reading_duration` | Qué leyó de verdad vs. qué solo guardó. Señal **principal** de preferencia, *inferida del comportamiento* (asociativa, no causal). |
| **Ratings (explícito implícito)** | `rating`, `rating_clean`, `rating_missing`, `user_mean_rating`, `user_rating_std`, `user_rating_count`, `user_rating_bias` | Intensidad y dirección del gusto; el `user_rating_bias` permite normalizar usuarios duros/generosos. |
| **Reviews (compromiso alto)** | `has_review_text`, `review_text_clean` | Escribir una review = máxima implicación; refuerza la señal positiva. |
| **Temporalidad** | `date_added`, `date_updated` | Cadencia, recencia y split temporal para evaluación. |

### Decisiones explícitas (lo que el enunciado pide resolver)

| Pregunta | Decisión | Por qué |
|---|---|---|
| ¿Viene del **historial de consumo**? | **Sí, es la base.** `is_read` + `engagement_mode` definen el núcleo de la preferencia. | Es preferencia *inferida del comportamiento* (asociativa): leer y calificar alto **correlaciona** con agrado; es la señal más confiable disponible en datos observacionales, no una prueba de causalidad. |
| ¿Viene de **ratings**? | **Sí, como peso y filtro.** Un libro con `is_read=True` y `rating_clean` alto pesa más en el perfil. | El rating da dirección (gustó/no gustó), no solo presencia. |
| ¿Viene de **compras/clicks**? | **No.** No existen en el dataset (no es transaccional ni hay telemetría). | Limitación honesta; se documenta como trabajo de producto futuro. |
| ¿Viene de **preferencias explícitas**? | **Solo en cold-start.** Géneros/libros semilla elegidos al registrarse. | No hay perfil declarado en los datos; el explícito solo cubre el arranque. |

---

## 2. Definición conceptual del user profile

El perfil **no es una etiqueta de género** ni una lista de libros. Es un **vector de gusto
multidimensional en el mismo espacio PCA que los libros** (`pc_0..pc_172`). Esto permite
comparar usuario y libro con la misma métrica de similitud y descubrir patrones cross-género
(p. ej. *tono juvenil + romance + aventura + fantasía ligera*).

```text
perfil(u) = agregación ponderada de los vectores PCA de los libros
            con los que u interactuó positivamente

vector_gusto(u) = Σ_b  w(u,b) · pca(b)   /   Σ_b w(u,b)
                  para b en libros con interacción positiva de u
```

Donde el **peso de preferencia** `w(u,b)` se construye con las señales de interacción
(no todas las interacciones valen igual):

```text
w(u,b)  crece con:  is_read = True            (leyó, no solo guardó)
                    rating_clean alto         (normalizado por user_rating_bias)
                    has_review_text = True    (compromiso alto)
                    reading_duration corta/coherente (lectura efectiva)
        cae con:    engagement_mode = shelf_only (solo guardado)
                    rating bajo / negativo
```

El perfil se acompaña de **metadatos de comportamiento** (agregados por `user_id`, derivables
de la misma tabla) que **no definen el gusto pero contextualizan la confianza y la
diversidad** a aplicar:

| Agregado (derivable) | Rol |
|---|---|
| `interaction_count`, `read_count` | Cuánta evidencia hay → confianza del perfil |
| `user_mean_rating`, `user_rating_std`, `user_rating_bias` | Normalización del rating del usuario |
| `category_count` / amplitud de géneros leídos | Señal anti-burbuja: ¿es lector mono-género o ecléctico? |
| recencia (`days desde última interacción`) | Frescura del perfil / riesgo de churn |

> **Dos capas del perfil:**
> 1. **Capa de gusto** = vector PCA agregado → *qué* recomendar (similitud).
> 2. **Capa de comportamiento** = agregados → *cuánta confianza* tener y *cuánta diversidad*
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
| **Sin historial (cold-start puro)** | Preferencias **explícitas** al registrarse: elegir géneros/libros semilla. El perfil = agregación de los vectores PCA de esos libros semilla. Si no elige nada → mezcla **popularidad moderada + diversidad** (un libro accesible por macro-cluster). |
| **Historial mínimo (1–N interacciones, perfil ruidoso)** | Perfil híbrido: mezclar el vector individual con el **centroide del cluster/macro-cluster** más cercano (shrinkage hacia el grupo). Da estabilidad cuando hay pocos datos. |
| **Historial suficiente** | Perfil = vector de gusto agregado completo (sección 2). |

Principios para cold-start (alineados con el negocio):

- **Umbral de confianza:** definir un mínimo de `read_count`/`interaction_count` para "confiar"
  en el perfil individual; por debajo, apoyarse en clusters.
- **Evitar sesgo de popularidad** incluso en el arranque: no llenar de bestsellers; usar la
  jerarquía de macro-clusters para ofrecer variedad accesible.
- **Aprendizaje incremental:** cada `is_read`/`rating` nuevo actualiza el vector de gusto, así
  el cold-start se disuelve rápido con el uso.

---

## 5. Qué señales NO usamos (y por qué)

- **Clicks, views, sesiones, dwell time** → no existen (dataset histórico, sin telemetría).
- **Compras, precio, conversión** → no es un dataset transaccional.
- **Likes, follows, grafo social, demografía** → no disponibles; el usuario es 100%
  comportamiento de lectura, sin perfil social ni demográfico.

---

## 6. Plan de construcción del `user_matrix` (build recipe)

Receta concreta para reconstruir la matriz de usuario desde cero sobre la fuente intacta.
**Estado: diseño aprobado, sin implementar todavía.**

**Decisiones fijadas:**
- **Interacción positiva** = `is_read == True` **AND** `rating_clean >= 4` (señal limpia).
- **Ponderación** = **media simple** (`w_i = 1`) como baseline reproducible. La ponderación
  por rating/recencia de la sección 2 queda como refinamiento posterior, no para el primer corte.
- **Implementación** = módulo `src/reduction/build_user_matrix.py` (invocable con `-m`), pendiente.

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
data/processed/<genero>/interactions_curated.parquet   (×5)  — fuente de interacciones
data/features/master_feature_matrix.parquet                  — book_id + pc_0..pc_172 (espacio de ítems)
```

### Pasos

| # | Paso | Detalle |
|---|---|---|
| 1 | **Filtrar positivos** | `is_read == True AND rating_clean >= 4` por fila de interacción. `shelf_only` y ratings bajos quedan fuera. |
| 2 | **Mapear libro → vector** | Cargar los 108k vectores PCA en memoria (~75 MB) como `book_id → pc_vec`; mapear, **no** hacer join pesado de millones de filas. |
| 3 | **Acumular por usuario** | Leer los 5 archivos en chunks; acumular `Σ pc_vec` y `count` por `user_id` **a través de los 5 géneros** (un usuario lee en varios). |
| 4 | **Centroide (media simple)** | `user_vec(u) = Σ pc_vec / count` — **estimación** del vector de gusto en el espacio PCA (inferida del comportamiento, no medición directa). |
| 5 | **Tabla lateral de comportamiento** | Aparte: `read_count`, `completion_rate`, `user_rating_bias`, `category_count`, recencia, `is_cold_start`. No entran al vector de gusto. |
| 6 | **Cold-start** | `count < 3` positivos → `is_cold_start=True`; no confiar en el centroide (ver sección 4). |
| 7 | **(Eval) split temporal** | Para evaluar: agregar solo el **pasado** del usuario (orden por `date_added`), dejar el futuro como holdout. Mide **capacidad predictiva** (¿el modelo ordena lo que el usuario efectivamente leyó después?), **no** impacto causal del recomendador sobre la lectura. |
| 8 | **Persistir** | `safe_write_parquet`, `float32`, paths desde `src/config.py`. |

### Fórmula (baseline)
```text
positivos(u) = { b : is_read(u,b)=True ∧ rating_clean(u,b) ≥ 4 }
user_vec(u)  = (1/|positivos(u)|) · Σ_{b ∈ positivos(u)} pca(b)      # media simple
```

### Outputs
```text
data/features/user_matrix.parquet   # user_id + pc_0..pc_172   (vector de gusto, mismo espacio que libros)
data/features/user_meta.parquet     # user_id + agregados de comportamiento (confianza/diversidad)
```

### Validación
- **Sanity**: un usuario que solo leyó fantasía debe caer cerca del cluster de fantasía.
- **Cuantitativa**: `Recall@k` / `NDCG@k` sobre el holdout temporal vs. baseline de popularidad.
  Es una comparación **correlacional** de calidad de ranking; si el modelo supera al baseline,
  eso indica mejor *predicción*, no que *cause* más lectura ni más hábito.

> Con `user_matrix` en el mismo espacio que `master_feature_matrix`, recomendar = buscar los
> libros vecinos del `user_vec`, re-rankeados con popularidad (secundaria) y diversidad
> (cross-cluster/género).

---

### Conclusión (para el entregable)

El perfil de usuario es un **vector de gusto multidimensional en el espacio PCA de los libros**,
**inferido del historial de consumo** (`is_read`, `engagement_mode`) y **ponderado por ratings
y reviews** (`rating_clean` normalizado por `user_rating_bias`, `has_review_text`), construido
como **agregación de los vectores PCA de los libros que el usuario leyó y valoró
positivamente**. No usa compras ni clicks (no existen) y solo usa preferencias explícitas en el
**arranque**. La validación se hace con **usuarios reales** del dataset vía split temporal; los
**perfiles semilla/simulados** cubren cold-start y demo. Para usuarios sin historial suficiente
se aplica una estrategia escalonada: semillas explícitas → shrinkage hacia el centroide de
cluster → perfil individual completo, evitando siempre el sesgo de popularidad. La fuente es
`interactions_curated.parquet` (intacta); la matriz usuario/interacción se reconstruirá desde
cero sobre ella.
