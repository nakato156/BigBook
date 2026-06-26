# Reporte del sistema de recomendación BigBook

## Resumen ejecutivo

BigBook es un sistema de recomendación de libros cuyo objetivo de negocio es ayudar a más
personas a sostener el hábito de lectura. Recomienda libros del catálogo a lectores reales de
Goodreads utilizando el gusto multidimensional inferido de su historial: libros leídos,
calificaciones positivas y señales de compromiso. El sistema no representa a una persona con una
sola etiqueta de género, sino mediante uno o varios vectores de gusto en el mismo espacio PCA que
los libros.

La versión V1 combina:

- una representación híbrida de 108,227 libros;
- 110,450,288 interacciones globales deduplicadas;
- perfiles de 699,381 usuarios con al menos una señal positiva;
- similitud coseno en un subespacio de gusto;
- recuperación por clusters, diversificación MMR y exploración controlada;
- exclusión obligatoria de libros ya consumidos;
- evaluación temporal frente a azar, popularidad global y popularidad por género.

El resultado principal es mixto. El modelo supera ampliamente al azar y produce mucha más
cobertura, novedad y exposición de cola que los baselines de popularidad. Sin embargo, no supera
a la popularidad global en `Precision@k`, `Recall@k` ni `NDCG@k` para `k = 5, 10, 20`. Por eso, el
veredicto académico actual es:

> **BigBook V1 no está validada como modelo final de ranking.**

Esto no significa que todo el sistema falle. La representación, el perfilado, la diversidad y el
control de exposición funcionan como infraestructura reproducible. El problema pendiente está en
el equilibrio entre personalización, recuperación de candidatos y capacidad de predecir la
próxima lectura.

---

## 1. Problema de recomendación

### 1.1 Qué recomienda el sistema

El sistema recomienda **libros**, identificados por `book_id`. Cada libro se representa como un
vector PCA de 173 dimensiones (`pc_0..pc_172`) y pertenece a:

- uno de 100 clusters finos de libros;
- uno de 10 macro-clusters construidos sobre los centroides de esos clusters.

El género no es la unidad de recomendación. Es una señal dentro de la representación y también se
usa para diversificar y explicar la lista.

### 1.2 A quién recomienda

Recomienda a **lectores**, identificados por `user_id`, con tres situaciones posibles:

1. **Usuario con historial suficiente:** se utilizan uno o varios centroides de gusto.
2. **Usuario con 1–2 positivos:** su vector se contrae un 50% hacia el centroide de catálogo más
   cercano para reducir el ruido.
3. **Usuario sin historial:** se usan libros semilla o un fallback diverso y accesible por
   macro-cluster.

La validación offline utiliza usuarios reales del dataset. Los perfiles con semillas se reservan
para cold-start y demostraciones.

### 1.3 Qué significa una buena recomendación

Una buena recomendación debe cumplir simultáneamente cuatro propiedades:

- **Relevancia:** coincidir con el gusto del lector y, en evaluación, recuperar libros que leerá
  después.
- **Accesibilidad:** evitar que toda la lista esté formada por lecturas innecesariamente difíciles
  o extensas. En V1, el número de páginas es un proxy débil y solo actúa como desempate.
- **Diversidad coherente:** introducir variedad semántica y cross-género sin convertir la lista en
  azar.
- **Novedad y descubrimiento:** no limitarse a los libros más populares.

La acción observable objetivo es una lectura completada (`is_read`), reforzada por un rating alto
o una review. El objetivo superior es el hábito de lectura, pero el dataset histórico solo permite
evaluarlo de forma predictiva o correlacional, no causal.

### 1.4 Formulación del problema

```text
Recomendamos  X = libros del catálogo representados como vectores PCA
a              Y = lectores representados por uno o varios vectores de gusto
para optimizar Z = próximas lecturas relevantes y motivadoras,
                   preservando diversidad y descubrimiento.
```

### 1.5 Task framing formal

BigBook resuelve una tarea de **recomendación personalizada mediante ranking top-k**:

```text
entrada  = historial anterior a t + catálogo disponible en t
salida   = lista ordenada de k libros para un usuario
objetivo = recuperar futuras lecturas positivas dentro de las primeras posiciones
```

La descripción técnica del stronger system es:

> **ranking híbrido de contenido y comportamiento, con perfiles multi-interés, candidate
> generation por clusters, similitud coseno, MMR y exploración controlada.**

No es predicción de ratings: el sistema no estima una nota continua, sino que ordena ítems. No es
segmentación de usuarios: los segmentos de actividad solo se usan para análisis. Tampoco es una
tarea de clustering como salida final: KMeans agrupa libros para recuperar candidatos, pero la
salida entregada al usuario es un ranking individual de libros.

La unidad de score es el par `(user_id, book_id)`. La unidad de salida es una lista top-k por
usuario. El detalle formal está en `docs/task_framing.md`.

---

## 2. Datos disponibles

La fuente es Goodreads Dataset Collection de UCSD, restringida a cinco familias:

- fantasy/paranormal;
- mystery/thriller/crime;
- history/biography;
- young adult;
- romance.

### 2.1 Ítems

El catálogo maestro contiene **108,227 libros** y 17 columnas:

| Grupo | Variables principales | Uso |
|---|---|---|
| Identidad | `book_id`, `title` | Clave, salida y explicación |
| Texto | `description` | Embedding semántico |
| Estructura | `series`, `author_count` | Saga, coautoría y progresión |
| Accesibilidad | `num_pages` | Esfuerzo aproximado de lectura |
| Contexto | `publication_year`, `language_code` | Época e idioma |
| Calidad/exposición | `average_rating`, `ratings_count`, `text_reviews_count` | Calidad agregada y popularidad |
| Género | cinco flags `genre_*`, `genre_count` | Señal multi-etiqueta y diversidad |

La unidad del modelo es siempre:

```text
una fila = un book_id = un vector de libro
```

### 2.2 Usuarios

No se dispone de edad, país, dispositivo, perfil social ni preferencias declaradas. Toda la
información del usuario se deriva de su comportamiento.

Artefactos actuales:

| Artefacto | Filas | Contenido |
|---|---:|---|
| `user_features_global.parquet` | 821,387 | Estadísticas globales y flag K-core `valid` |
| `user_matrix.parquet` | 699,381 | Un vector de gusto por usuario con positivos |
| `user_meta.parquet` | 706,367 | Confianza, actividad, amplitud y cold-start |
| `user_centroids.parquet` | 2,356,255 | Uno o varios modos de gusto por usuario |

### 2.3 Interacciones

El artefacto global canónico contiene **110,450,288 registros deduplicados**. Integra las cinco
fuentes de género y conserva la interacción de mayor prioridad cuando hay duplicados:

```text
review > rating_only > read_no_rating > want_to_read
```

Sus señales principales son:

| Señal | Columnas | Interpretación |
|---|---|---|
| Lectura | `is_read` | Acción objetivo observable |
| Rating | `rating_clean`, `rating_missing` | Feedback explícito; cero se convierte en ausente |
| Review | `has_review_text`, `review_text_length` | Compromiso alto |
| Intención | `is_want_to_read`, `engagement_mode` | Interés todavía no convertido en lectura |
| Intensidad | `interaction_weight`, `user_rating_bias` | Fuerza y calibración del feedback |
| Tiempo | `date_added`, `date_updated`, `started_at`, `read_at` | Split temporal y proxies de hábito |
| Duración | `reading_duration_days` | Señal de lectura efectiva cuando está disponible |

### 2.4 Ratings y otras señales relevantes

La señal positiva usada para construir el perfil baseline es:

```text
is_read == True AND rating_clean >= 4
```

Las demás señales tienen roles distintos:

- `has_review_text` y una duración coherente aumentan el peso de compromiso de los modos de gusto;
- `want_to_read` se conserva como intención, pero no entra al vector positivo;
- `user_rating_bias` ayuda a describir si una persona suele calificar por encima o por debajo de
  la media;
- `ratings_count` mide exposición histórica, pero no ordena los slots normales del modelo;
- fechas válidas permiten separar pasado y futuro.

No existen clicks, impresiones, sesiones, compras, precio, dwell time, likes, follows ni
telemetría de exposición al recomendador.

### 2.5 Data alignment

El pipeline declara contratos explícitos entre artefactos:

| Origen | Destino | Contrato |
|---|---|---|
| `books_master` | `master_feature_matrix` | Mismo conjunto exacto de `book_id` |
| `master_feature_matrix` | `book_clusters_k100` | Un cluster por libro |
| `interactions_curated` | `books_master` | Toda interacción modelable pertenece al master |
| `user_features_global` | `user_meta` | `user_meta` contiene exactamente usuarios `valid` |
| `user_meta` | `user_matrix` | Vector solo para usuarios con positivos |
| `user_matrix` | `user_centroids` | Los centroides son un subconjunto de usuarios con vector |
| PCA de libros | PCA de usuarios | Mismos `pc_*`, orden y dimensionalidad |

La igualdad clave del lado de ítems es:

```text
ids(books_master)
  = ids(master_feature_matrix)
  = ids(book_clusters_k100)
```

Al cargar el ranker, el orden de `master_feature_matrix` gobierna el catálogo y metadata y
clusters se reindexan por `book_id`. El módulo `src.validate_artifacts` falla ante IDs vacíos,
duplicados, faltantes o extras, esquemas PCA distintos o usuarios incompatibles.

El candidate pool común en una fecha `t` es:

```text
C(u,t) =
    catálogo master
    ∩ disponibilidad histórica en t
    ∩ elegibilidad técnica
    - consumidos por u hasta t
```

Los baselines ordenan sobre `C(u,t)`. El modelo aplica una segunda reducción a clusters cercanos y
exploración. Por ello, un libro puede ser elegible para la evaluación y quedar fuera del ranker por
candidate generation.

La clave operativa `book_id` identifica una edición o registro Goodreads, no necesariamente una
obra. Dos ediciones pueden tener el mismo título. La V1 alinea y excluye correctamente por
`book_id`, pero necesita una identidad canónica de obra para evitar recomendar otra edición de un
libro ya leído.

Temporalmente:

- perfil, consumidos y géneros se construyen solo con train;
- popularidad y segmentos se calculan solo hasta el corte;
- objetivos futuros no disponibles en el corte se excluyen;
- PCA, embeddings y clusters permanecen congelados, por lo que subsiste fuga transductiva
  residual.

Supuestos y riesgos que condicionan la interpretación:

| Supuesto | Riesgo |
|---|---|
| `book_id` equivale al ítem de producto | Dos ediciones de una obra pueden tratarse como libros distintos |
| Ausencia de interacción equivale a desconocido | No existen negativos de exposición |
| Leído con rating ≥4 equivale a positivo | Sesgo de selección y diferencias personales al calificar |
| Fechas Goodreads representan el orden real | Fechas faltantes o carga retrospectiva |
| Metadata imputada conserva comparabilidad | Páginas/año faltantes pueden introducir ruido |
| El catálogo completo representa un snapshot | La evaluación congelada conserva fuga transductiva |
| Un cluster cercano contiene los próximos positivos | El retrieval puede eliminar objetivos antes del scoring |

V1.1 añade un backtest manual que reconstruye PCA, clustering y perfiles por corte para medir el
último riesgo. Los espacios de snapshots distintos se mantienen aislados: solo se comparan
métricas agregadas, nunca componentes PCA ni identificadores de cluster.

---

## 3. Perfil de usuario

### 3.1 Definición conceptual

El perfil es una **estimación del gusto multidimensional** inferida del comportamiento observado.
No es una preferencia verdadera medida directamente y tampoco implica causalidad.

El perfil baseline se obtiene como la media de los vectores PCA de los libros positivos:

```text
positivos(u) = {b : is_read(u,b) = True y rating_clean(u,b) >= 4}

user_vector(u) =
    (1 / |positivos(u)|) * suma[pca(b) para b en positivos(u)]
```

Este diseño coloca usuarios y libros en el mismo espacio, por lo que pueden compararse mediante
similitud coseno.

### 3.2 Información utilizada

El perfil tiene dos capas:

**Capa de gusto**

- libros leídos y calificados con 4 o 5;
- vector PCA de cada libro;
- varios sub-centroides cuando el historial contiene modos distintos.

**Capa de comportamiento**

- `positive_count` e `interaction_count`;
- cantidad de reviews y libros `want_to_read`;
- `user_rating_bias`;
- amplitud de categorías;
- fecha de última actividad;
- flag de cold-start.

Para los centroides, la intensidad de una interacción se aproxima mediante:

```text
engagement_weight =
    (rating_clean - 3)
    * 1.3 si escribió review
    * 1.2 si la duración está entre 1 y 180 días
```

Cuando hay seis o más positivos, el sistema puede conservar hasta cuatro modos:

```text
positive_count < 6  -> 1 centroide
positive_count >= 6 -> min(4, positive_count // 3) centroides
```

### 3.3 Limitaciones del perfil

- Solo aprende de feedback observado; un libro no registrado es indistinguible de uno no leído.
- El filtro `rating >= 4` descarta señales negativas que podrían ayudar a separar gustos.
- La media simple puede borrar preferencias minoritarias; los multi-centroides reducen, pero no
  eliminan, este problema.
- Las preferencias cambian con el tiempo, mientras que el perfil V1 no aplica decaimiento por
  recencia.
- No hay contexto situacional: una persona puede querer lecturas diferentes según momento,
  disponibilidad o intención.
- Los ratings tienen sesgo de selección y estilo personal.
- En cold-start, `num_pages` es una aproximación insuficiente de accesibilidad.

---

## 4. Representación de ítems

### 4.1 Variables que describen cada libro

Antes de PCA se construyen **276 variables** divididas en tres bloques:

| Bloque | Dimensiones | Variables |
|---|---:|---|
| Numérico | 9 | rating, popularidad logarítmica, páginas, año, autores, amplitud de género y flags de missingness |
| Binario/categórico | 11 | serie, cinco géneros y cinco categorías de idioma |
| Embeddings | 256 | semántica de descripción o fallback de título |

El bloque numérico incluye:

```text
average_rating
log1p(ratings_count)
log1p(text_reviews_count)
num_pages
num_pages_missing
publication_year
publication_year_missing
author_count
genre_count
```

Los nulos de páginas y año se imputan por mediana y se conservan indicadores explícitos de
ausencia. Así se evita introducir `NaN` y no se pierde completamente la información de que el dato
faltaba.

### 4.2 Por qué capturan preferencias

- Los embeddings modelan tema, tono, audiencia y estilo más allá del género.
- Los géneros aportan una estructura temática gruesa y multi-etiqueta.
- Las páginas aproximan esfuerzo y accesibilidad.
- El año diferencia preferencias por época.
- `series` aporta continuidad y progresión.
- Ratings y reviews agregados aportan calidad y exposición, aunque deben controlarse para evitar
  que la popularidad domine.

La mezcla permite descubrir afinidades como “aventura juvenil, romance y fantasía ligera” aunque
los libros no compartan exactamente las mismas etiquetas.

### 4.3 Estandarización, ponderación y PCA

Cada bloque se estandariza de forma independiente y se pondera con:

```text
peso_bloque = 1 / sqrt(numero_de_dimensiones_del_bloque)
```

Pesos actuales:

| Bloque | Peso |
|---|---:|
| Numérico | 0.333333 |
| Binario | 0.301511 |
| Embeddings | 0.062500 |

Esta ponderación evita que las 256 columnas de embeddings dominen solo por ser más numerosas.

Después se ajusta:

```text
PCA(n_components=0.95, svd_solver="full")
```

El resultado conserva **95.014% de la varianza** en **173 componentes**, reduciendo 276 variables
a `pc_0..pc_172`. El diagnóstico
`embedding_dominated_first_5_count = 0` indica que el bloque semántico no domina artificialmente
los primeros componentes.

### 4.4 Relación con clustering

KMeans se ajusta sobre los 173 componentes y a nivel de libro. V1 selecciona:

- `k = 100` clusters finos;
- 10 macro-clusters mediante Ward sobre los centroides.

La comparación disponible muestra:

| k | Silhouette muestreado | Tamaño medio | Tamaño máximo |
|---:|---:|---:|---:|
| 50 | 0.085851 | 2,164.54 | 6,115 |
| 100 | 0.070210 | 1,082.27 | 3,147 |

El corte `k=100` sacrifica algo de separación geométrica para obtener vecindades más granulares,
útiles para recuperación y explicación.

### 4.5 Limitaciones de la representación

- PCA es lineal y puede perder relaciones complejas.
- Los componentes no son explicaciones legibles por sí mismos.
- Los primeros ejes todavía contienen señal tabular y de popularidad.
- Los embeddings dependen de la calidad de la descripción.
- Idioma aporta poca discriminación porque el catálogo es casi completamente inglés.
- Faltan formato, editorial, tags libres y temas más ricos.
- La representación es estática y contiene fuga transductiva residual en la evaluación temporal,
  porque fue ajustada con el catálogo completo.

---

## 5. Estrategia de recomendación

### 5.1 Resumen del ranking

La arquitectura es:

```text
perfil -> retrieve por clusters -> score de interés -> MMR -> exploración -> explicación
```

La configuración V1 por defecto es:

| Parámetro | Valor |
|---|---:|
| Tamaño de lista | 10 |
| Clusters finos recuperados | 5 |
| Slots de exploración | 2 |
| `mmr_lambda` | 0.70 |
| Penalización de solapamiento de género | 0.15 |
| Bonus máximo de accesibilidad | 0.05 |
| Piso relativo de exploración | 75% de la mejor similitud |
| Umbral mínimo de páginas para accesibilidad | 50 |

### 5.2 Similitud de interés

Para reducir la influencia directa de popularidad, idioma y missingness, el ranking excluye
`pc_0..pc_5` del coseno de interés.

Para un usuario con varios modos:

```text
sim_interes(u,b) =
    max_c [peso_normalizado(u,c) * cos(centroide(u,c), libro(b))]
```

La popularidad no entra a esta fórmula.

### 5.3 Generación de candidatos

1. Se calcula la cercanía de los modos del usuario a los centroides de los 100 clusters.
2. Se eligen los cinco clusters finos con mayor relevancia.
3. Se recuperan los libros técnicamente válidos de esos clusters.
4. Se eliminan los libros ya consumidos.
5. Se puntúan los candidatos por similitud de interés.

Este enfoque reduce el espacio de búsqueda y mantiene una explicación por vecindad.

### 5.4 Orden y diversidad

Al score de interés se suma un bonus pequeño para libros más cortos:

```text
ranking_relevance = interest_similarity + 0.05 * accessibility_score
```

Después se aplica Maximal Marginal Relevance:

```text
MMR =
    0.70 * relevancia
    - 0.30 * redundancia_semantica
    - 0.15 * solapamiento_de_genero
```

La selección es greedy. El primer libro es el de mayor relevancia y los siguientes equilibran
relevancia con novedad respecto de los ya seleccionados.

### 5.5 Exploración

En un top-10 se intentan reservar dos posiciones de exploración. Un libro exploratorio debe:

- estar fuera de los macro-clusters ocupados por la vecindad recuperada;
- alcanzar al menos el 75% de la mejor similitud del usuario;
- pertenecer a `tail` o `mid`, nunca a `head`;
- no haber sido consumido.

La prioridad es `tail` y luego `mid`, ordenando por similitud dentro del segmento. Si no hay
candidatos que superen el piso, las posiciones vuelven al ranking normal. Por eso una lista puede
tener menos de dos slots exploratorios.

Los segmentos se calculan dinámicamente:

```text
tail = ratings_count <= percentil 25
mid  = percentil 25 < ratings_count < percentil 90
head = ratings_count >= percentil 90
```

En el catálogo actual:

```text
tail <= 436 ratings
head >= 6,017 ratings
```

### 5.6 Ítems excluidos

Se excluyen:

- libros ya leídos/consumidos por el usuario;
- semillas utilizadas para construir un perfil cold-start;
- libros sin `book_id` o título válido;
- libros con coordenadas PCA no finitas;
- libros sin cluster válido;
- libros no disponibles en la fecha de corte durante la evaluación histórica.

No se excluye un libro por tener poca popularidad.

---

## 6. Baselines

### 6.1 B0: azar

Selecciona libros elegibles no consumidos usando una semilla determinista por usuario.

```text
B0(u,k) = muestra_determinista(C(u,t) - consumidos(u,t), k)
```

Es razonable como piso de cordura: el modelo debe superarlo en relevancia. Su diversidad y
cobertura pueden ser altas, pero no tienen valor si la lista no es relevante.

### 6.2 B1: popularidad global

Ordena los libros mediante:

```text
pop_score(b) = log1p(ratings_count_historico(b)) * average_rating_historico(b)
```

Entrega esencialmente los mismos libros populares a todos, después de excluir los ya consumidos.
Es el baseline principal porque:

- es fácil de implementar;
- obtiene aciertos no triviales por la concentración de consumo;
- representa el comportamiento que BigBook quiere mejorar: recomendar bestsellers sin modelar el
  gusto individual.

### 6.3 B2: popularidad por género

Aplica el mismo score de popularidad, pero restringe los candidatos a los géneros presentes en el
historial positivo del usuario.

Es un rival más fuerte conceptualmente porque incorpora personalización básica:

```text
si le gusta fantasía -> recomendar fantasía popular
```

Si el usuario tiene varios géneros positivos, B2 usa la unión de esos géneros y aplica el mismo
`pop_score` histórico de B1. No usa el género del holdout futuro.

Superar B2 sería evidencia de que el espacio multidimensional aporta más que una regla simple de
género.

### 6.4 Comparación justa

Modelo y baselines comparten:

- los mismos usuarios;
- el mismo corte temporal;
- el mismo catálogo históricamente disponible;
- los mismos `k`;
- la exclusión de libros consumidos;
- popularidad calculada solo con evidencia anterior al corte.

El pool común se define como:

```text
C(u,t) =
    libros disponibles en t
    ∩ ids/títulos/vectores/clusters técnicamente válidos
    - libros consumidos por u hasta t
```

B1 y B2 ordenan todo `C(u,t)`; el modelo añade retrieval por clusters. Por eso
`candidate_recall` se reporta solo para el modelo.

---

## 7. Resultados top-k

Los siguientes ejemplos proceden de `recommendations_v1_sample.csv`. Son listas operativas
generadas con los artefactos actuales; sirven para inspección cualitativa, no prueban por sí solas
que el ranking sea correcto.

### 7.1 Usuario A

`user_id = 00000377eea48021d3002730d56aca9a`

Perfil:

- 36 positivos de 87 interacciones;
- 2 reviews y 43 libros `want_to_read`;
- actividad en las cinco categorías;
- historial positivo concentrado en young adult y fantasía;
- ejemplos de 5 estrellas: *An Ember in the Ashes*, *The Scorpio Races*, *The Raven Boys*,
  *Six of Crows*, varios libros de *Harry Potter* y *Neverwhere*.

Top-10:

| Rank | Libro | Slot | Géneros | Segmento |
|---:|---|---|---|---|
| 1 | Project Princess | interés | YA | head |
| 2 | Holding Smoke | interés | fantasy, mystery, YA | mid |
| 3 | Everything, Everything | interés | YA, romance | head |
| 4 | Scarlet | interés | history, YA | head |
| 5 | The Hunger Games | interés | YA | head |
| 6 | City of Ashes | interés | fantasy, YA | mid |
| 7 | The Fiery Trial | interés | fantasy, YA | head |
| 8 | The Raven Boys | interés | fantasy, YA | head |
| 9 | Lost at Sea | exploración | YA | tail |
| 10 | Out of Reach | exploración | YA | tail |

Lectura crítica: la lista refleja bien el eje YA/fantasía del usuario y utiliza los dos slots de
exploración para títulos de cola. Sin embargo, *The Raven Boys* aparece entre los positivos
históricos con `book_id=17675462` y vuelve a recomendarse con `book_id=13449693`. La exclusión de
consumidos sí se cumple por identificador, pero el catálogo contiene ediciones o registros
distintos de una misma obra. Para producto hace falta una clave de nivel obra que impida recomendar
otra edición de un libro ya leído.

### 7.2 Usuario B

`user_id = 00004584d524ec468619e81b176cc991`

Perfil:

- 60 positivos de 92 interacciones;
- amplitud en cuatro categorías;
- historial especialmente fuerte en YA, fantasía y romance;
- ejemplos de 5 estrellas: *Between the Lines*, *The Notebook*, *Gossamer*, *Esperanza Rising*,
  *The Help*, *Catching Fire*, *Mockingjay* y *Scarlet*.

Top-10:

| Rank | Libro | Slot | Géneros | Segmento |
|---:|---|---|---|---|
| 1 | Secrets | interés | YA | mid |
| 2 | A Moveable Feast | interés | history | head |
| 3 | The Girl Who Kicked the Hornet's Nest | interés | mystery | head |
| 4 | The Dragon Reborn | interés | fantasy | head |
| 5 | Persuasion | interés | romance | head |
| 6 | Daimon | interés | fantasy, YA | head |
| 7 | The Battle of the Labyrinth | interés | fantasy, YA | head |
| 8 | Meet Josefina: An American Girl | interés | history | mid |
| 9 | Out of Reach | exploración | YA | tail |
| 10 | Lost at Sea | exploración | YA | tail |

Lectura crítica: este ejemplo muestra diversidad cross-género real, pero parte de la lista puede
ser demasiado dispersa: memorias, thriller, fantasía épica, romance clásico y YA aparecen juntos.
MMR está evitando redundancia, aunque posiblemente a costa de coherencia fina.

### 7.3 Usuario C

`user_id = 000079c580bbe45e1500acabe551b276`

Perfil:

- 8 positivos de 49 interacciones;
- historial positivo repartido entre historia, mystery y fantasía;
- rating bias de `-0.633`, por lo que es un usuario relativamente estricto;
- ejemplos positivos: *Our Moon Has Blood Clots*, *A Game of Thrones*, *Steve Jobs*,
  *Red Rising*, *Jerusalem: The Biography*, *The Bourne Identity* y *Angels & Demons*.

Top-10:

| Rank | Libro | Slot | Géneros | Segmento |
|---:|---|---|---|---|
| 1 | My Boyfriend's Back | interés | fantasy, romance | mid |
| 2 | Inferno | interés | mystery | head |
| 3 | The Art of War | interés | history | head |
| 4 | The Son of Neptune | interés | YA | head |
| 5 | Dragon Ours | interés | fantasy, romance | mid |
| 6 | The Bourne Legacy | interés | mystery | head |
| 7 | The Darkest Night | interés | fantasy, romance | head |
| 8 | Golden Son | interés | fantasy | head |
| 9 | Rising Sun | interés | mystery | head |
| 10 | A Clash of Kings | interés | fantasy | head |

Lectura crítica: varias continuaciones son coherentes con su historial (*Inferno*, *The Bourne
Legacy*, *Golden Son*, *A Clash of Kings*). En cambio, algunos romances paranormales parecen
menos alineados. No aparecen slots exploratorios porque ningún candidato externo cumplió las
reglas; el fallback completó la lista con interés.

### 7.4 Conclusión cualitativa

Los ejemplos muestran tres fortalezas:

- reconocimiento de sagas y vecindades temáticas;
- listas cross-género;
- exposición explícita de libros tail cuando pasan el piso de relevancia.

También muestran tres alertas:

- posible exceso de diversidad;
- fuerte presencia de `head` en los slots normales;
- deduplicación insuficiente entre distintas ediciones de una misma obra.

---

## 8. Evaluación

### 8.1 Protocolo

La evaluación N0 utiliza un corte global:

```text
2016-06-09T19:30:05.200000+00:00
```

Datos del protocolo:

| Concepto | Valor |
|---|---:|
| Usuarios seleccionados | 5,000 |
| Usuarios evaluables | 2,038 |
| Usuarios descartados | 2,962 |
| Libros disponibles en el corte | 103,664 |
| Fechas inválidas descartadas | 1,949 |

El pasado construye el perfil y el futuro define los libros relevantes. Un positivo futuro debe
estar leído, tener rating positivo y estar disponible en el snapshot histórico.

La evaluación se denomina
`global_historical_snapshot_frozen_representation`: popularidad, disponibilidad y holdout son
históricos, pero PCA, embeddings y clusters permanecen congelados con información del catálogo
completo.

### 8.2 Definición de métricas

Para una lista de tamaño `k`:

```text
Precision@k = aciertos_en_top_k / k

Recall@k = aciertos_en_top_k / numero_de_libros_relevantes_futuros
```

`NDCG@k` descuenta los aciertos por posición:

```text
DCG@k = suma(hit_i / log2(i + 1))
NDCG@k = DCG@k / IDCG@k
```

Un acierto al principio de la lista vale más que uno al final.

También se reportan:

- `MAP`: precisión acumulada en las posiciones con acierto;
- diversidad: distancia coseno media entre pares de la lista;
- cobertura: fracción del catálogo expuesta al menos una vez;
- cobertura long-tail: fracción del segmento tail expuesta;
- novedad: menor probabilidad histórica implica mayor valor;
- mezcla de exposición `tail/mid/head`.

V1.1 añade:

```text
candidate_recall =
    objetivos futuros elegibles presentes en el pool recuperado
    / objetivos futuros elegibles
```

Esta métrica separa E1 (el objetivo nunca llegó a scoring) de E2 (sí llegó, pero perdió en el
ranking). El CLI opcional `--bootstrap-ci` remuestrea usuarios y genera intervalos percentiles
reproducibles para Recall, Precision, NDCG y MAP.

### 8.3 Precision, Recall y NDCG

| Sistema | k | Precision | Recall | NDCG | MAP |
|---|---:|---:|---:|---:|---:|
| Modelo | 5 | 0.004318 | 0.002733 | 0.006181 | 0.003633 |
| B0 azar | 5 | 0.000098 | 0.000061 | 0.000083 | 0.000033 |
| B1 popularidad | 5 | **0.034053** | **0.015553** | **0.037230** | **0.028593** |
| B2 género-popularidad | 5 | 0.031992 | 0.014603 | 0.035294 | 0.026818 |
| Modelo | 10 | 0.004563 | 0.004745 | 0.006773 | 0.003048 |
| B0 azar | 10 | 0.000147 | 0.000110 | 0.000132 | 0.000034 |
| B1 popularidad | 10 | **0.026398** | **0.025292** | **0.034830** | **0.022126** |
| B2 género-popularidad | 10 | 0.024485 | 0.024151 | 0.032898 | 0.020028 |
| Modelo | 20 | 0.005177 | 0.011873 | 0.009256 | 0.003191 |
| B0 azar | 20 | 0.000123 | 0.000131 | 0.000149 | 0.000032 |
| B1 popularidad | 20 | **0.018057** | **0.036133** | **0.034181** | **0.018162** |
| B2 género-popularidad | 20 | 0.016364 | 0.034619 | 0.032305 | 0.016578 |

Interpretación:

- El modelo supera con holgura al azar.
- B1 supera al modelo en las tres métricas principales para todos los valores de `k`.
- En `k=10`, el modelo alcanza solo 18.8% del Recall y 19.4% del NDCG de B1.
- En `k=20` la brecha relativa se reduce, pero el modelo todavía alcanza solo 32.9% del Recall de
  B1.
- B2 queda muy cerca de B1, lo que muestra que la popularidad sigue siendo una señal predictiva
  dominante en este dataset.

Bootstrap por usuario (1,000 remuestreos, IC 95%) para `k=10`:

| Sistema | Recall medio | IC 95% Recall | NDCG medio | IC 95% NDCG |
|---|---:|---:|---:|---:|
| Modelo | 0.004745 | [0.003238, 0.006635] | 0.006773 | [0.004970, 0.008899] |
| B1 popularidad | 0.025292 | [0.021059, 0.030366] | 0.034830 | [0.029536, 0.041232] |
| B2 género-popularidad | 0.024151 | [0.020206, 0.028930] | 0.032898 | [0.027732, 0.038813] |

Los intervalos del modelo y B1 no se solapan en estas métricas: la brecha observada no parece una
fluctuación pequeña de la cohorte evaluada.

### 8.4 Descubrimiento, diversidad y exposición

| Sistema | k | Diversidad | Cobertura catálogo | Cobertura tail | Novedad | Tail | Mid | Head |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Modelo | 5 | 0.752907 | 0.039358 | 0.039614 | 16.230789 | 23.9% | 34.9% | 41.2% |
| B1 | 5 | 0.553450 | 0.000328 | 0.000000 | 8.243777 | 0.0% | 0.0% | 100.0% |
| Modelo | 10 | 0.767539 | 0.081320 | 0.059631 | 15.738482 | 15.5% | 35.4% | 49.1% |
| B1 | 10 | 0.675652 | 0.000492 | 0.000000 | 8.459652 | 0.0% | 0.0% | 100.0% |
| Modelo | 20 | 0.731984 | 0.144795 | 0.093679 | 15.626177 | 11.1% | 36.5% | 52.4% |
| B1 | 20 | 0.776237 | 0.000723 | 0.000000 | 8.926029 | 0.0% | 0.0% | 100.0% |

Interpretación:

- A `k=10`, el modelo cubre **165.3 veces** más catálogo que B1.
- La cobertura long-tail es positiva en el modelo y cero en B1/B2.
- La novedad del modelo es aproximadamente 7.28 puntos mayor que B1 en `k=10`.
- El modelo es más diverso que B1 en `k=5` y `k=10`.
- En `k=20`, B1 obtiene una diversidad intra-lista ligeramente mayor. Esto muestra que una métrica
  aislada no basta: B1 sigue recomendando 100% head y tiene cobertura casi nula.

### 8.5 Slots de interés y exploración

En la evaluación agregada:

| k | Precision interés | Hit rate interés | Precision exploración | Hit rate exploración |
|---:|---:|---:|---:|---:|
| 5 | 0.006035 | 0.020608 | 0.000280 | 0.000561 |
| 10 | 0.005313 | 0.040236 | 0.000280 | 0.000561 |
| 20 | 0.005584 | 0.082924 | 0.000280 | 0.000561 |

La exploración aporta exposición, pero casi ningún acierto temporal. El piso relativo de 75% no
es suficiente para garantizar relevancia predictiva o el criterio de “fuera del macro-cluster”
está alejando demasiado los candidatos.

### 8.6 Evidencia N1 sobre hábito

Los proxies futuros se describen por actividad previa:

| Segmento | Usuarios | Lecturas previas | Lecturas futuras | Completion futura | Breadth futura | Recencia futura |
|---|---:|---:|---:|---:|---:|---:|
| Low | 305 | 7.50 | 13.31 | 0.571 | 2.531 | 133.22 días |
| Mid | 639 | 34.29 | 13.05 | 0.527 | 2.779 | 113.51 días |
| High | 1,094 | 195.17 | 29.63 | 0.567 | 3.394 | 81.59 días |

Los usuarios previamente más activos conservan más lecturas futuras, mayor amplitud y menor
recencia. Esto describe persistencia del comportamiento, pero **no demuestra que BigBook la haya
causado**, porque nadie en el dataset fue expuesto al sistema.

### 8.7 Veredicto

El criterio predictivo N0 exigía superar B1 en Recall y NDCG sin perder descubrimiento. V1 cumple
la parte de descubrimiento, pero no la de relevancia. La decisión de negocio posterior separa esta
lectura: B1 es baseline de exposición observada, no métrica norte; BigBook debe reportar
superioridad predictiva N0 solo si gana a B1, y alineamiento de hábito si además mejora
descubrimiento, exposición no-head y proxies N1 sin afirmar causalidad.

> **Veredicto N0: V1 no validada.**

### 8.8 Estado experimental de V1.1

El código incorpora tres evaluaciones adicionales:

- retrieval por modo y ablaciones de clusters/MMR/exploración;
- comparación temporal train-only entre contenido, coocurrencia PMI y user-kNN;
- backtest con refit independiente en varios snapshots.

Estos jobs son manuales y costosos sobre `interactions_curated.parquet` (110,450,288 filas). Este
informe no atribuye una mejora ni declara una señal colaborativa ganadora hasta que existan
`ablation_results.csv`, `collaborative_ab_results.csv` y `multi_snapshot_backtest.csv` completos.
La ausencia de esos resultados no se sustituye por métricas estimadas.

---

## 9. Análisis de errores y riesgos

### 9.1 Objetivo del error analysis

El análisis de errores no pregunta solamente cuántos aciertos obtuvo el modelo. Busca localizar en
qué etapa se perdió un objetivo futuro:

```text
catálogo elegible
    -> candidate generation
    -> scoring
    -> MMR/diversificación
    -> exploración
    -> top-k
```

En `k=10`, 83 de los 2,038 usuarios evaluables tienen al menos un acierto y 1,955 no tienen
aciertos. Esta cifra no atribuye la causa. Para hacerlo se reconstruyeron casos con el mismo corte,
catálogo histórico, perfil de train y configuración del ranker.

### 9.2 Taxonomía de fallos

| Código | Tipo | Condición |
|---|---|---|
| E1 | Candidate generation | El objetivo es elegible, pero su cluster no se recupera |
| E2 | Scoring/diversificación | El objetivo está en un cluster recuperado, pero queda fuera del top-k |
| E3 | Exploración irrelevante | El slot aumenta exposición, pero no coincide con el futuro positivo |
| E4 | Identidad de obra | Se recomienda otra edición de una obra ya consumida |
| E5 | Etiqueta incompleta | Un no-hit puede ser relevante pero nunca observado/expuesto |
| E6 | Perfil amplio | Pocos centroides/clusters no cubren todos los modos de lectura |

Esta separación es accionable: aumentar `mmr_lambda` no arregla E1, y recuperar más clusters no
resuelve por sí solo E4.

### 9.3 Caso fuerte: acierto en rank 1

Usuario `f54b46386ef443e4fe44c33bc4cd35b4`:

- 35 positivos en train;
- un único positivo futuro evaluable;
- clusters recuperados `[54, 82, 0, 75, 24]`;
- objetivo: *Harry Potter and the Chamber of Secrets*, `book_id=15881`, cluster 54;
- resultado: `rank=1`, slot `interest`.

El objetivo pertenecía al cluster más cercano, candidate generation lo incluyó y el score lo
colocó en la primera posición. Es un caso limpio donde perfil, retrieval y ranking están
alineados.

### 9.4 Caso mixto: retrieval correcto y ordenamiento parcial

Usuario `9f1c9f43f46d6504712a4429dcf229d7`:

- 3 positivos en train;
- 8 positivos futuros;
- clusters recuperados `[67, 13, 95, 11, 55]`;
- 3 aciertos en top-10.

| Rank | Acierto | Cluster |
|---:|---|---:|
| 3 | Angels Flight | 13 |
| 5 | A Darkness More Than Night | 13 |
| 9 | The Last Coyote | 13 |

Otros objetivos del cluster 13, como *City of Bones*, *Mr. Mercedes*, *The Concrete Blonde* y
*Trunk Music*, llegaron a la vecindad recuperada pero quedaron fuera del top-10. El sistema
identificó correctamente el modo policial, pero la competencia entre libros del mismo cluster y
la diversificación limitaron el recall final. Es una combinación de éxito de retrieval y E2.

### 9.5 Caso de fallo: objetivo fuera del retrieval

Usuario `b11dee4ef20822ad0281a474baf9023f`:

- 331 positivos en train;
- un único objetivo futuro;
- clusters recuperados `[72, 42, 91, 30, 63]`;
- objetivo: *Reflected in You (Crossfire, #2)*, `book_id=13596809`, cluster 54.

El cluster 54 no fue recuperado. El libro estaba en el catálogo histórico, pero nunca llegó a
scoring.

> Clasificación: **E1, fallo de candidate generation**.

La mejora correcta es aumentar recall del retrieval, recuperar por cada modo del usuario o añadir
búsqueda ANN global. Modificar solo MMR no puede recuperar un libro que nunca fue candidato.

### 9.6 Caso de fallo mixto: gusto amplio

Usuario `9ebb0290a3a302189bb5712eb8898cf9`:

- 223 positivos en train;
- 220 positivos futuros;
- clusters recuperados `[71, 68, 57, 2, 8]`;
- cero aciertos en top-10.

Parte del futuro estaba en clusters recuperados, por ejemplo *Panic*, *Sweet Little Thing* y
*Three, Two, One* en el cluster 2, y *Dangerous Secrets* y *Dirty* en el cluster 8. Esos libros no
entraron al top-10: E2.

Otra parte importante estaba distribuida por clusters no recuperados, incluidos numerosos
mystery/thriller del cluster 13: E1. Con cuatro centroides máximos y cinco clusters recuperados, el
presupuesto es rígido para un historial tan amplio: E6.

### 9.7 Error de exploración

En `k=10`:

```text
interest_hit_rate    = 0.040236
exploration_hit_rate = 0.000561
```

La exploración mejora coverage y novelty, pero casi nunca coincide con futuros positivos. Las
hipótesis principales son:

- salir completamente de los macro-clusters ocupados es demasiado agresivo;
- el piso relativo del 75% puede aceptar similitudes absolutas bajas;
- priorizar tail antes que similitud puede perjudicar el orden;
- dos slots fijos no se adaptan a la confianza o amplitud del perfil.

Las pruebas adecuadas son explorar primero macro-clusters hermanos, combinar piso relativo y
absoluto, y comparar 0/1/2 slots.

### 9.8 Error de identidad de obra

En el ejemplo cualitativo del Usuario A:

```text
consumido:   The Raven Boys, book_id=17675462
recomendado: The Raven Boys, book_id=13449693
```

La exclusión por `book_id` funciona, pero ambas filas representan la misma obra en ediciones
distintas. Es E4: un problema de alineamiento semántico de identidad, no del score. La solución es
crear `canonical_work_id` y aplicar exclusiones y evaluación a nivel obra.

### 9.9 Riesgo de over-specialization

El riesgo existe porque la recuperación empieza en solo cinco clusters cercanos y el perfil se
construye con positivos. Esto puede reforzar preferencias históricas y dejar fuera intereses
nuevos.

Las mitigaciones actuales son MMR, penalización de género y exploración externa. No obstante, los
resultados muestran una tensión:

- el ranking normal todavía concentra entre 41% y 52% de exposición en `head`;
- la exploración mejora cobertura, pero casi no recupera futuros positivos;
- una diversificación agresiva puede explicar parte de la baja relevancia.

### 9.10 Diversidad de recomendaciones

La diversidad es una fortaleza real en `k=5` y `k=10`, y los ejemplos muestran listas
cross-género. Sin embargo, “más diversidad” no siempre equivale a “mejor lista”. El Usuario B
recibe una mezcla muy amplia que puede perder coherencia.

La mejora futura debe optimizar diversidad **condicionada a relevancia**, no diversidad como
objetivo independiente.

### 9.11 Novedad

El modelo supera ampliamente a B1 en novedad y cobertura long-tail. Esta es la evidencia más
fuerte a favor del diseño anti-popularidad.

La limitación es que novedad se deriva de popularidad histórica. Un libro poco popular no es
necesariamente novedoso para una persona ni tiene calidad suficiente. Hacen falta señales de
exposición individual y satisfacción posterior.

### 9.12 Sesgos y limitaciones

**Sesgo de popularidad del dataset.** Los libros populares aparecen en más historiales y son más
fáciles de acertar. Por eso B1 obtiene resultados fuertes.

**Sesgo de selección.** Solo se observan acciones registradas en Goodreads. No leer o no calificar
no implica rechazo.

**Sesgo de usuarios activos.** La evaluación requiere historial en ambos lados del corte. Los
usuarios evaluables no representan necesariamente a lectores nuevos o casuales.

**Datos faltantes.** Páginas y año tienen ausencias relevantes. La imputación evita fallos, pero
no recupera la información original.

**Fuga transductiva residual.** La popularidad histórica se congela correctamente, pero PCA,
embeddings y clusters fueron construidos con el catálogo completo.

**Métrica objetivo incompleta.** Los futuros positivos son una señal razonable de relevancia, pero
no incluyen exposición. Un libro relevante que el usuario nunca conoció aparece como negativo.

**Sin causalidad.** Precision, Recall y NDCG no demuestran que recomendar un libro aumente el
hábito.

**Exploración rígida.** Dos slots y un piso fijo del 75% no se adaptan al nivel de confianza del
perfil ni a la actividad del usuario.

**Explicabilidad limitada.** Cluster y género son explicaciones aproximadas; los componentes PCA
no producen razones semánticas legibles.

### 9.13 Métricas diagnósticas implementadas

La evaluación ya calcula:

```text
candidate_recall =
    objetivos futuros presentes en candidatos recuperados
    / objetivos futuros elegibles
```

Los CSV regenerados permiten reportar:

- porcentaje de objetivos fuera del retrieval;
- porcentaje recuperado pero no rankeado;
- errores por amplitud de usuario;
- duplicados de obra;
- casos reproducibles con objetivos, clusters y posiciones.

Además, el bootstrap opcional cuantifica incertidumbre por usuario y el backtest multi-snapshot
separa sensibilidad temporal de la representación. Los resultados numéricos deben regenerarse
antes de reemplazar las tablas históricas de esta sección.

---

## 10. Conclusiones

### 10.1 Qué funcionó

- Se construyó una fuente global de interacciones deduplicada y alineada con el catálogo.
- Usuarios e ítems viven en el mismo espacio vectorial.
- Los multi-centroides conservan gustos distintos mejor que una sola media.
- El ranking excluye consumidos y evita usar popularidad como score explícito.
- MMR y la exploración aumentan cobertura, novedad y exposición long-tail.
- La evaluación temporal y los baselines comparten un protocolo reproducible.
- El sistema supera claramente al azar.

### 10.2 Qué no funcionó

- Los slots exploratorios casi nunca aciertan futuros positivos.
- La diversidad puede estar desplazando demasiado la relevancia.
- El retrieval por clusters puede excluir candidatos relevantes antes del scoring.
- El score de interés no aprovecha señales colaborativas entre usuarios.
- La exclusión por `book_id` no evita recomendar otra edición de una obra ya consumida.

La comparación frente a B1/B2 se conserva como resultado experimental y criterio académico de
validación, pero no se clasifica aquí como defecto de implementación. El error analysis se centra
en etapas concretas y corregibles del pipeline.

### 10.3 Mejoras para una versión futura

Implementado en V1.1, pendiente de ejecutar a escala completa:

1. Ablation study del ranking con retrieval por modo, distintos presupuestos, MMR y exploración.
2. Blend contenido-colaborativo calibrado por percentiles, comparando PMI y user-kNN train-only.
3. `candidate_recall` para separar candidate generation de scoring.
4. Bootstrap por usuario e intervalos de confianza.
5. Backtest con PCA/clustering/perfiles reconstruidos por snapshot.

Trabajo futuro:

1. **Crear una identidad canónica de obra**, agrupando ediciones y validando que ninguna
   recomendación represente un libro ya consumido bajo otro `book_id`.
2. **Aumentar el recall de candidatos:** comparar retrieval por modo contra búsqueda ANN global.
3. **Calibrar la exploración con datos:** probar pisos más altos, explorar dentro de
   macro-clusters hermanos y reducir slots cuando la confianza sea baja.
4. **Usar negativos y preferencias relativas:** incorporar ratings bajos y pares
   positivo/negativo en un modelo de ranking.
5. Evaluar por amplitud de gusto, cold-start y popularidad de los objetivos.

Prioridad de producto:

6. Instrumentar impresiones, clicks, guardados, inicios y finalizaciones posteriores a una
   recomendación.
7. Medir novedad individual, satisfacción y repetición de sesiones.
8. Ejecutar un A/B test online para estimar impacto causal en frecuencia, completion y retención.

### 10.4 Cierre

BigBook V1 demuestra una arquitectura sólida para recomendación personalizada y descubrimiento,
pero sus resultados obligan a separar dos conclusiones:

1. **Sí funciona como mecanismo de diversificación y exposición:** muestra más catálogo, más cola
   y más novedad que popularidad.
2. **Todavía no funciona como ranker validado de próxima lectura:** la popularidad global predice
   mejor los futuros positivos.

La siguiente versión no debería abandonar el control de popularidad. Debe conservar esa fortaleza
y mejorar el retrieval y el aprendizaje de relevancia hasta cerrar la brecha con B1.

---

## 11. Trazabilidad

Este reporte se elaboró con los artefactos locales actuales:

- `data/processed/books_master.parquet`
- `data/processed/interactions_curated.parquet`
- `data/processed/user_features_global.parquet`
- `data/features/master_feature_matrix.parquet`
- `data/features/master_pca_meta.json`
- `data/features/user_matrix.parquet`
- `data/features/user_meta.parquet`
- `data/features/user_centroids.parquet`
- `data/outputs/recommendations/recommendations_v1_sample.csv`
- `data/outputs/recommendations/temporal_evaluation.csv`
- `data/outputs/recommendations/temporal_evaluation_by_activity.csv`
- `data/outputs/recommendations/temporal_evaluation_users.parquet`
- `docs/estado_v1.md`
- `docs/task_framing.md`
- `docs/data_alignment.md`
- `docs/error_analysis.md`
- `src/reduction/recommend.py`
- `src/reduction/evaluate_recommender.py`

Fecha de elaboración: **10 de junio de 2026**.
