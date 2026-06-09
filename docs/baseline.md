# Línea base (baseline) — contra qué se compara el recomendador

Pieza de cierre de la fase de evaluación. Un recomendador **no se presenta sin punto de
comparación**: para afirmar que el motor por similitud (ver
[criterio_scoring](criterio_scoring.md)) "funciona", primero hay que mostrar **qué resultado da
una recomendación ingenua** y exigir que el modelo lo supere. Este documento define ese punto de
comparación: qué es una recomendación ingenua en nuestro dominio, qué performance cabe esperar de
un sistema simple, y **contra qué baseline concreto** se mide el recomendador principal.

> Engancha directamente con el **criterio de validez** de
> [metricas_evaluacion](metricas_evaluacion.md): el sistema es válido si **supera a la base de
> popularidad** en `Recall@k`/`NDCG@k` **sin colapsar** `Coverage`/`Novelty`/`Diversity`. Aquí se
> formaliza qué es esa "base de popularidad" y cómo se construye, de modo que la comparación sea
> reproducible y honesta.

---

## 1. ¿Por qué necesitamos un baseline?

Sin línea base, cualquier número del recomendador (`Recall@10 = 0.12`, p. ej.) es ininterpretable:
no se sabe si es bueno, malo o trivial. El baseline cumple tres funciones:

- **Piso de cordura (sanity floor):** si el modelo no supera ni al azar, está roto.
- **Techo de la trivialidad:** la popularidad es lo que consigue cualquiera sin modelar gusto;
  superarla es la barra mínima para justificar todo el pipeline (PCA + clusters + perfil de
  usuario).
- **Control del objetivo de negocio:** el baseline de popularidad es precisamente lo que el
  producto **quiere evitar** (sesgo de popularidad, burbuja de bestsellers). Compararse contra él
  no solo mide relevancia: mide si de verdad estamos descubriendo, no amplificando.

---

## 2. ¿Qué es una recomendación ingenua en nuestro dominio?

Una recomendación ingenua es la que **no modela el gusto multidimensional del usuario**. En este
proyecto hay tres formas, de menos a más informada, y todas son baselines legítimos:

| # | Baseline | Qué recomienda | Qué NO usa | Rol |
|---|---|---|---|---|
| **B0** | **Random** | `k` libros al azar del catálogo válido (`books_master`) | Nada | Piso absoluto de cordura |
| **B1** | **Popularidad global (MostPopular)** | Los `k` libros con mayor popularidad del catálogo, iguales para todos | El historial del usuario | **Baseline principal de comparación** |
| **B2** | **Popularidad por género (regla simple)** | Los `k` libros más populares **dentro de los géneros que el usuario ya leyó** | Similitud fina; trata el género como etiqueta única | Baseline intermedio "if te gusta fantasía, más fantasía popular" |

- **B0 (Random)** es el `if-the-user-likes-X` llevado a cero: ninguna señal. Existe solo para
  acotar el piso; un recomendador que no le gane está mal implementado.
- **B1 (Popularidad global)** es la recomendación ingenua **canónica** y la que el README y
  [metricas_evaluacion](metricas_evaluacion.md) ya nombran como "la base de popularidad". Ordena el
  catálogo por una señal de popularidad y devuelve el top-`k`, **idéntico para todos los usuarios**.
- **B2 (Popularidad por género)** es la "regla simple" del enunciado del curso: usa una pizca de
  personalización (el conjunto de géneros leídos por el usuario) pero sigue siendo la lógica básica
  *"si te gusta fantasía, recomienda fantasía popular"* que el producto declara insuficiente
  (ver [Business Logic](../README.md#business-logic)). Es el rival más interesante: si nuestro
  recomendador no le gana, entonces el gusto cross-género no estaba aportando valor.

### Definición operativa de "popularidad" (B1 / B2)

Para no reintroducir información futura, la señal de popularidad se reconstruye desde las cinco
categorías del canonical usando únicamente ratings con `date_added` válido hasta el corte:

```text
pop_score(b) = log1p(ratings_count(b)) · average_rating(b)
```

`log1p` atenúa el outlier de popularidad (igual que en el pipeline PCA) y `average_rating` evita
premiar libros muy contados pero mal valorados. Los empates se resuelven por `book_id` para que el
resultado sea reproducible. **Ambos** baselines excluyen los libros consumidos en entrenamiento y
los que todavía no estaban disponibles en el snapshot.

---

## 3. ¿Qué performance tendría un sistema simple?

Como el dataset es **observacional**, esta sección conserva las expectativas conceptuales de cada
baseline. Las cifras medidas y su comparación con el modelo se publican en
[`estado_v1.md`](estado_v1.md). La expectativa nace de propiedades conocidas de los datos
(ver *PCA Block Diagnostics* del README: `pc_0` = popularidad).

| Baseline | `Recall@k` / `NDCG@k` (relevancia) | `Coverage` / `Novelty` / `Diversity` (anti-popularidad) |
|---|---|---|
| **B0 Random** | ~0 (cercano al azar; reparte sobre 108k libros) | **Coverage altísima, Novelty alta**, pero relevancia nula → inútil |
| **B1 Popularidad global** | **Bajo-medio pero no trivial**: los bestsellers son leídos por mucha gente, así que aciertan por frecuencia, no por gusto | **Coverage pésima, Novelty pésima**: el mismo top-`k` para todos amplifica la burbuja |
| **B2 Popularidad por género** | Medio (un poco mejor que B1: filtra por el género del usuario) | Coverage/Novelty algo mejores que B1, pero sigue siendo burbuja de género |

La intuición clave: **B1 acierta sin entender al usuario**. Si mucha gente leyó *Harry Potter*,
recomendar *Harry Potter* a todos consigue un `Recall@k` no nulo "gratis". Por eso ganar en
relevancia es necesario, pero **no suficiente**: el recomendador debe ganar en relevancia
**manteniendo** la diversidad/cobertura que B1 destruye. Ese es el doble criterio del entregable.

---

## 4. Contra qué se compara el recomendador principal

El recomendador híbrido (retrieve por cluster → score coseno → diversify MMR; ver
[criterio_scoring](criterio_scoring.md)) se compara contra los tres baselines **con el mismo
protocolo, el mismo `k` y los mismos usuarios**:

```text
B0 Random            → piso de cordura      (debe superarse con holgura)
B1 Popularidad       → BASELINE PRINCIPAL   (debe superarse en relevancia SIN perder diversidad)
B2 Popularidad/género→ rival realista       (ganar aquí justifica el gusto cross-género)
```

**Criterio de éxito (hereda el de [metricas_evaluacion](metricas_evaluacion.md) §4):** el
recomendador es válido si

1. `Recall@k` / `NDCG@k` (modelo) **>** los de **B1** (y de B2), y
2. `Coverage` / `Long-tail Coverage` / `Novelty` / `Diversity` (modelo) **≥** los de **B1**
   (B1 fija el piso de diversidad que **no** se debe empeorar — y dado lo malo que es B1 ahí, el
   modelo debería superarlo con claridad), y
3. el modelo se asocia a mejores **proxies de hábito** (Capa 2: `completion_rate`,
   `reading_frequency`, `reading_breadth`).

Si el modelo solo gana relevancia copiando a B1 (recomendando populares), las métricas
anti-popularidad lo delatan y **no se considera válido** para el objetivo de hábito.

---

## 5. Protocolo de evaluación común (idéntico para baseline y modelo)

Para que la comparación sea justa, baselines y recomendador corren bajo el **mismo** montaje:

- **Datos:** `interactions_curated.parquet` (canonical global), universo de ítems = `books_master`.
- **Usuarios:** solo los `valid` (K-core global `read_or_rated_count ≥ 3`, de
  `user_features_global`), para no evaluar sobre historial ruidoso.
- **Snapshot temporal global:** un único corte compartido, explícito con `--cutoff` o derivado por
  `--train-fraction`, separa pasado y futuro. Fechas anteriores a `2006-01-01` se descartan. El
  *holdout* relevante contiene libros con `is_read = True` y `rating_clean ≥ 4` posteriores al
  corte que además estaban disponibles en el catálogo histórico.
- **Popularidad histórica:** B1/B2 y las métricas de exposición usan únicamente conteos y ratings
  observados hasta el corte, nunca agregados posteriores.
- **Tarea:** cada sistema produce un top-`k` por usuario (`k ∈ {5, 10, 20}`), **excluyendo** lo ya
  leído en entrenamiento.
- **Métricas:** relevancia (`Recall@k`, `Precision@k`, `NDCG@k`, `MAP`) + exposición
  (`Coverage`, `Long-tail Coverage`, `Novelty`, `Diversity`, mix `tail/mid/head` y
  `Average Recommendation Popularity`), reportadas lado a lado.

> Nota de implementación: los tres baselines viven en
> `src/reduction/evaluate_recommender.py` y comparten split, `k`, exclusiones y usuarios con el
> modelo. El runner selecciona la cohorte desde `user_features_global.valid`, ejecuta por defecto
> `k = 5, 10, 20`, calcula `MAP` y diversidad, y separa precisión/hit-rate de los slots `interest`
> y `exploration` del modelo. B0 usa semilla determinista. El modo se reporta como
> `global_historical_snapshot_frozen_representation`: PCA, embeddings y clusters siguen congelados,
> por lo que persiste fuga transductiva residual. Un backtest estricto requeriría reconstruir esos
> artefactos en cada snapshot.

---

### Conclusión (para el entregable)

La **línea base** del proyecto es la **recomendación por popularidad** (**B1**, top-`k` global por
`log1p(ratings_count) · average_rating`, igual para todos), acompañada de un **piso de azar**
(**B0 Random**) y un **rival de regla simple por género** (**B2**, popularidad dentro de los
géneros leídos). El recomendador principal **no se presenta solo**: se reporta siempre junto a
estos baselines, bajo el mismo split temporal, mismo `k` y mismos usuarios. Se declara una mejora
real **solo si supera a la popularidad en relevancia (`Recall@k`/`NDCG@k`) sin sacrificar la
diversidad/cobertura** que la popularidad destruye — porque en este producto ganar copiando a los
bestsellers es, precisamente, perder.
