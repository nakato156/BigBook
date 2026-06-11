# Task framing del sistema de recomendación

Este documento clasifica formalmente la tarea que resuelve BigBook y separa el objetivo final de
los componentes auxiliares. La distinción importa porque clustering, predicción de ratings,
segmentación y ranking pueden aparecer juntos en un recommender, pero no son la misma tarea.

## 1. Clasificación formal

BigBook resuelve una tarea de:

> **recomendación personalizada mediante ranking top-k de ítems**.

Para cada usuario `u`, fecha de decisión `t` y conjunto de candidatos elegibles `C(u,t)`, el
sistema produce una lista ordenada:

```text
R_k(u,t) = [b_1, b_2, ..., b_k]
```

donde cada `b_i` es un libro no consumido y disponible en el catálogo histórico. El orden intenta
maximizar afinidad con el gusto del usuario, conservando diversidad, novedad y exposición
controlada.

En términos de producto:

```text
entrada  = historial anterior a t + catálogo disponible en t
salida   = top-k libros ordenados para el usuario
objetivo = recuperar futuras lecturas positivas sin reducir el problema a popularidad
```

## 2. Qué tarea NO es

### No es predicción de ratings

El sistema no intenta estimar `rating(u,b)` como una variable continua. Los ratings se usan para:

- definir positivos históricos (`is_read=True AND rating_clean>=4`);
- estimar compromiso de los modos de gusto;
- construir los baselines históricos de popularidad.

La salida no es una calificación predicha, sino una lista ordenada.

### No es segmentación de usuarios

BigBook no asigna usuarios a segmentos de mercado para recomendar una lista fija por segmento.
Los segmentos `low/mid/high` se usan solamente para describir actividad en la evaluación N1.

### No es clustering como resultado final

KMeans y la jerarquía Ward agrupan **libros**, no usuarios. El clustering es una etapa de
**candidate generation**:

```text
100 clusters finos -> recuperar vecindades cercanas -> puntuar libros -> producir top-k
```

El usuario nunca recibe "un cluster"; recibe libros individualmente ordenados.

### No es ranking de feed

No hay un flujo continuo de posts, noticias o anuncios ni señales de impresión/click. Es un
ranking top-k bajo demanda sobre un catálogo relativamente estable.

## 3. Stronger system: cómo debe describirse

La descripción técnica correcta es:

> **Ranking híbrido personalizado de contenido y comportamiento, con perfiles multi-interés,
> candidate generation por clusters, similitud coseno, MMR y exploración controlada.**

Es híbrido porque combina:

- comportamiento: lecturas, ratings, reviews y duración;
- contenido: embeddings de descripciones;
- metadata: género, páginas, año, idioma, serie y popularidad atenuada;
- geometría de catálogo: PCA, clusters finos y macro-clusters.

No debe presentarse como collaborative filtering puro. V1 no aprende factores latentes de una
matriz usuario-ítem ni usa vecinos de usuarios como score principal.

## 4. Unidad de predicción y unidad de evaluación

| Concepto | Unidad |
|---|---|
| Usuario | `user_id` |
| Ítem | `book_id` de Goodreads |
| Score | par `(user_id, book_id)` |
| Salida | lista ordenada de `k` libros por usuario |
| Etiqueta offline | libro futuro con `is_read=True AND rating_clean>=4` |
| Evaluación | usuario, sistema y cutoff `k` |

La unidad de catálogo actual es una **edición/registro Goodreads**, no una obra canónica. Dos
`book_id` pueden tener el mismo título. Esta diferencia es relevante para exclusiones y análisis
de errores.

## 5. Candidate pool

El pool común de evaluación se define como:

```text
C(u,t) =
    libros del master
    ∩ libros técnicamente válidos
    ∩ libros disponibles en t
    - libros consumidos por u hasta t
```

La validez técnica requiere:

- `book_id` y título no vacíos;
- vector PCA finito;
- cluster válido.

Los baselines ordenan directamente sobre `C(u,t)`. El modelo fuerte hace una recuperación
adicional:

```text
C_retrieve(u,t) =
    libros de los 5 clusters finos más cercanos
    + candidatos exploratorios de macro-clusters no ocupados
```

Esta diferencia debe declararse porque un libro relevante puede estar en el pool común y aun así
no llegar al ranker del modelo.

## 6. Definición de relevancia

Un objetivo futuro es relevante si:

```text
date_added > cutoff
AND is_read == True
AND rating_clean >= 4
AND el libro estaba disponible en el snapshot histórico
```

La etiqueta mide coincidencia con comportamiento futuro registrado, no satisfacción causal ni
exposición al sistema.

## 7. Resumen para el entregable

```text
Tipo de tarea:
    personalized top-k recommendation/ranking

Strong model:
    content-behavior hybrid ranker

Candidate generation:
    book clusters + controlled exploration

No es:
    rating prediction, user segmentation, feed ranking

Objetivo offline:
    ordenar futuros positivos dentro de los primeros k
```
