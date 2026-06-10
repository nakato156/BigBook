# Data alignment y contratos entre artefactos

Este documento hace explícitas las claves, cardinalidades, filtros y supuestos que conectan
interacciones, catálogo, features, perfiles y evaluación. El pipeline no depende de joins
implícitos: cada frontera tiene un contrato verificable.

## 1. Mapa de alineamiento

```text
raw interactions
    -> canonical global deduplicado
    -> filtro al universo books_master
    -> perfiles de usuario en espacio PCA

books_master
    <-> master_feature_matrix
    <-> book_clusters_k100
    -> ranking

user_features_global
    <-> user_meta
    -> user_matrix
    -> user_centroids
    -> ranking
```

## 2. Contratos de claves y cardinalidad

| Origen | Clave | Cardinalidad | Destino | Contrato |
|---|---|---:|---|---|
| `books_master` | `book_id` | 1:1 | `master_feature_matrix` | Mismo conjunto exacto de libros |
| `master_feature_matrix` | `book_id` | 1:1 | `book_clusters_k100` | Un cluster por cada libro |
| `interactions_curated` | `book_id` | N:1 | `books_master` | Toda interacción modelable pertenece al universo master |
| `interactions_curated` | `interaction_key` | 1:1 | canonical | Una fila ganadora por interacción deduplicada |
| `user_features_global` | `user_id` | 1:1 | `user_meta` | `user_meta` contiene exactamente los usuarios `valid` |
| `user_meta` | `user_id` | 1:0..1 | `user_matrix` | Hay vector solo si `positive_count > 0` |
| `user_matrix` | `user_id` | 1:N | `user_centroids` | Los centroides solo pertenecen a usuarios con vector |
| PCA de libros | `pc_*` | esquema | PCA de usuarios | Mismos nombres, orden y dimensionalidad |

## 3. Identidad de ítem

La clave operativa es `book_id` de Goodreads:

```text
una fila = un book_id = un vector PCA = un cluster
```

Supuesto explícito:

- `book_id` identifica un registro o edición;
- no garantiza identidad de obra;
- dos ediciones pueden compartir título y contenido.

Consecuencia: excluir un `book_id` consumido no impide recomendar otra edición de la misma obra.
La V1 cumple el contrato técnico por ID, pero una versión de producto necesita una clave canónica
de obra, por ejemplo basada en `work_id`, ISBN normalizado o reglas título-autor.

## 4. Universo de libros

`books_master.parquet` define el universo modelable. La curación global de interacciones descarta
filas cuyo `book_id` no pertenece a ese universo.

El catálogo y sus representaciones deben contener exactamente los mismos 108,227 IDs:

```text
ids(books_master)
  = ids(master_feature_matrix)
  = ids(book_clusters_k100)
```

El módulo `src.validate_artifacts` falla si existen IDs faltantes, extras, vacíos o duplicados.

## 5. Alineamiento del espacio vectorial

Libros, `user_matrix` y `user_centroids` comparten `pc_0..pc_172`.

No basta con tener 173 columnas: deben coincidir nombres y orden. Un cambio en el PCA exige
reconstruir perfiles y centroides. El `.joblib` del PCA es la única forma soportada de transformar
nuevos libros con la misma imputación, escalado y pesos.

Al cargar el ranker:

1. `master_feature_matrix` fija el orden de filas del catálogo;
2. metadata y clusters se reindexan a ese orden por `book_id`;
3. centroides KMeans se interpretan en el mismo orden de componentes;
4. perfiles de usuario usan el mismo conjunto ordenado de `pc_*`.

## 6. Alineamiento de usuarios

La fuente de usuarios válidos es `user_features_global.parquet`:

```text
valid = read_or_rated_count >= K_USER_MIN
```

Contratos:

- `user_meta` contiene exactamente los usuarios válidos;
- `user_matrix` contiene los usuarios válidos con al menos un positivo;
- `user_centroids` es un subconjunto de `user_matrix`;
- un usuario sin vector usa semillas o cold-start.

El K-core y `user_rating_bias` se calculan globalmente sobre las cinco categorías. Nunca se
calculan por género.

## 7. Alineamiento temporal

La evaluación fija un corte global `t`:

```text
train  = date_added <= t
future = date_added > t
```

Fechas inválidas o anteriores a `2006-01-01` no son evaluables.

Para evitar fuga operativa:

- el perfil se reconstruye solo con positivos de `train`;
- consumidos se obtienen solo de `train`;
- B1/B2 usan ratings observados hasta `t`;
- los segmentos `tail/mid/head` usan popularidad hasta `t`;
- un objetivo futuro solo entra si el libro estaba disponible en `t`;
- libros con año conocido requieren `publication_year <= year(t)`;
- sin año, se exige primera observación válida anterior a `t`.

Limitación declarada: PCA, embeddings y clusters permanecen congelados con el catálogo completo.
Existe fuga transductiva residual, aunque no se usan interacciones futuras para construir el
perfil ni la popularidad histórica.

## 8. Alineamiento del candidate pool

El pool común por usuario es:

```text
C(u,t) =
    catalogo_master
    ∩ disponibilidad_historica(t)
    ∩ elegibilidad_tecnica
    - consumidos_train(u)
```

Los baselines y el modelo comparten ese universo inicial. Luego:

- B0 muestrea de `C(u,t)`;
- B1 ordena `C(u,t)` por popularidad global histórica;
- B2 filtra `C(u,t)` por géneros de train y ordena por popularidad;
- el modelo recupera clusters cercanos dentro de `C(u,t)` y añade exploración elegible.

Esto evita comparar sistemas con libros futuros o consumidos, pero el modelo puede perder recall
en su etapa de recuperación aunque el libro relevante exista en `C(u,t)`.

## 9. Deduplicación de interacciones

`interactions_curated.parquet` contiene una fila por `interaction_key`:

- usa `review_id` cuando existe;
- si no, usa una clave determinista `user_id|book_id`;
- selecciona el registro de mayor prioridad;
- conserva `want_to_read`, pero lo excluye del vector positivo.

La prioridad es:

```text
review > rating_only > read_no_rating > want_to_read
```

La deduplicación es global, no por categoría. Las vistas por género son solo para EDA.

## 10. Validaciones automatizadas

`env/bin/python -m src.validate_artifacts` verifica:

- schemas mínimos;
- IDs únicos y no vacíos;
- igualdad exacta de IDs catálogo/PCA/clusters;
- exactamente 100 clusters no nulos;
- igualdad de esquemas PCA;
- alineamiento de usuarios válidos, positivos y centroides.

Los tests relevantes incluyen:

- `tests/test_artifact_validation.py`;
- `tests/test_user_matrix.py`;
- `tests/test_user_centroids.py`;
- `tests/test_recommend.py`;
- `tests/test_interactions_curation.py`.

## 11. Supuestos y límites

| Supuesto | Riesgo |
|---|---|
| `book_id` equivale al ítem de producto | Ediciones duplicadas de una misma obra |
| Ausencia de interacción equivale a desconocido | No hay negativos de exposición |
| Rating alto + leído equivale a positivo | Sesgo de selección y estilo de rating |
| Fechas Goodreads representan orden real | Fechas faltantes o carga retrospectiva |
| Metadata imputada conserva comparabilidad | Páginas/año pueden introducir ruido |
| PCA completo es válido para snapshot histórico | Fuga transductiva residual |

Estos supuestos no están ocultos: forman parte del contrato de interpretación de V1.
