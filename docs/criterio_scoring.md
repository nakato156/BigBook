# Definición del criterio de similitud y scoring

Fase de scoring. El objetivo es explicar **cómo el sistema decide qué recomendar primero**:
qué significa que un libro esté "más cerca" del usuario, qué significa un score mayor, qué
diferencia hay entre las formas de scoring y por qué el ranking resultante tiene sentido para
el objetivo de *sostener el hábito de lectura*.

> Recordatorio del espacio: usuario y libros viven en el **mismo espacio PCA** (`pc_0..pc_172`).
> El perfil de usuario ya existe en dos formas compatibles: `user_matrix` (un vector baseline por
> usuario) y `user_centroids` (varios modos de gusto por usuario cuando hay historial suficiente;
> ver [perfil_usuario](perfil_usuario.md)). Cada libro es un vector (ver
> [representacion_items](representacion_items.md)). Por eso se pueden comparar directamente.

---

## 1. ¿Qué significa que un ítem esté "más cercano" al usuario?

"Cercano" = **poca distancia entre el vector de gusto del usuario y el vector del libro** en el
espacio PCA. Como el espacio mezcla contenido, género, accesibilidad y popularidad atenuada,
cercanía significa: *este libro se parece, en su combinación de tono/temática/accesibilidad, a
los libros que el usuario leyó y valoró positivamente* — no "es del mismo género".

**Métrica base: similitud coseno.** Medimos el ángulo entre vectores, no su magnitud:

```text
sim(u, b) = cos(θ) = (vec_gusto(u) · pca(b)) / (||vec_gusto(u)|| · ||pca(b)||)
```

Con `user_centroids`, la similitud de interés se evalúa por modo de lectura y se agrega con el
peso de compromiso del modo:

```text
sim_interés(u,b) = max_c [ centroid_weight(u,c) · cos(centroide(u,c), pca(b)) ]
```

El ranker usa esos multi-centroides cuando existen; `user_matrix` es el fallback de un solo vector.

Por qué coseno y no euclidiana cruda:

- El **gusto es una dirección**, no una intensidad. Dos libros del mismo "sabor" deben ser
  cercanos aunque uno sea más popular o más largo (mayor magnitud en algunos ejes).
- La euclidiana penaliza diferencias de magnitud que en parte vienen de la popularidad
  (`pc_0`), reintroduciendo sesgo de popularidad por la puerta de atrás. El coseno lo atenúa.
- Es consistente con los *spot checks* de coseno que ya usa `master_pca_meta.json`.

`sim ∈ [-1, 1]`; más alto = más cercano = más afín al gusto.

---

## 2. ¿Qué significa que un ítem tenga mayor score?

La similitud produce el orden de interés, pero la lista final también incorpora diversidad y
exploración. v1 separa elegibilidad, ranking y exposición:

```text
elegible(b)  = id/título/vector PCA/cluster válidos
score(u, b)  = similitud_de_interés(u, b)          ← coseno en subespacio de gusto
lista_base   = MMR(score, redundancia, género)      ← diversidad semántica + cross-género
               + desempate suave por accesibilidad
exploración  = fuera de vecindad + piso de afinidad
               + solo tail/mid
```

Las reglas de prioridad (qué manda sobre qué) son las que importan:

1. **Interés primero.** `similitud_de_interés` domina el orden.
2. **Elegibilidad técnica.** Solo se excluyen artefactos inválidos. `ratings_count` no funciona
   como gate ni como multiplicador.
3. **Popularidad como diagnóstico.** El catálogo se divide dinámicamente en `tail` (≤ p25),
   `mid` y `head` (≥ p90). El segmento prioriza exploración, pero no modifica el score.
4. **Diversidad explícita.** MMR penaliza redundancia semántica y solapamiento con los géneros ya
   presentes en la lista.
5. **Descubrimiento controlado.** Por defecto, 2 de 10 slots buscan libros fuera de los
   macro-clusters recuperados, exigen al menos 75% de la mejor similitud y solo aceptan
   `tail`/`mid`. Si no hay candidatos suficientemente afines, se completa con interés.
6. **Accesibilidad subordinada.** `num_pages` aporta un bonus pequeño para desempatar candidatos
   de afinidad similar; no sustituye el score de interés.

---

## 3. Diferencia conceptual entre formas de scoring

| Enfoque de scoring | Qué ordena | Problema en nuestro dominio |
|---|---|---|
| **Solo popularidad** (`ratings_count`, `average_rating`) | Lo más conocido primero | Ignora el gusto; amplifica bestsellers; mata el descubrimiento. **Rechazado** como criterio principal. |
| **Solo similitud** (coseno puro) | Lo más parecido primero | "Más de lo mismo"; burbuja; no distingue calidad ni accesibilidad. Necesario pero **insuficiente**. |
| **Similitud + diversidad (MMR)** | Parecido pero variado | Mejor; evita redundancia. Base de nuestra lista. |
| **Interés + MMR + exploración controlada** (el nuestro) | Afinidad primero; tail/mid dentro de slots con piso de relevancia | Equilibra relevancia, diversidad y exposición. **Adoptado.** |
| **Cluster-first** (vecindad → ranking interno) | Primero el barrio, luego dentro | Eficiente y explicable; lo usamos como **arquitectura** del retrieval (ver §4). |

La diferencia conceptual central: un score por **popularidad** responde "¿qué leen todos?"; uno
por **similitud** responde "¿a qué se parece lo que te gustó?"; el nuestro responde "¿qué te va a
gustar y cómo damos una exposición controlada a libros menos vistos sin romper la afinidad?".

---

## 4. Arquitectura del ranking (retrieve → score → diversify)

Coherente con el clustering ya construido (KMeans k=100 + 10 macro-clusters):

```text
1. RETRIEVE   Encontrar los clusters/macro-clusters más cercanos al vec_gusto(u)
              y traer los libros candidatos de esas vecindades (+ algo de exploración).
2. SCORE      Ordenar candidatos por coseno de interés; popularidad no entra al score.
3. DIVERSIFY  Reordenar con MMR semántico + penalización de género + accesibilidad suave,
              y reservar slots relevantes exclusivamente para tail/mid.
4. EXPLAIN    Justificar cada ítem por su cluster/género ("porque te gustó X, del mismo
              vecindario de lectura").
```

El paso de retrieve por cluster hace el sistema **escalable** (no se puntúan 108k libros por
usuario) y **explicable** (la vecindad da la razón de la recomendación).

---

## 5. ¿Por qué el ranking producido tiene sentido?

- **Relevante:** ordena por afinidad de gusto real (coseno en un espacio que captura contenido),
  no por popularidad. Responde a *"esto te va a gustar"*.
- **Técnicamente válida:** la elegibilidad exige artefactos completos, no popularidad mínima.
- **Pro-hábito como objetivo, no como factor demostrado:** `is_read` futuro alinea la validación
  predictiva con lecturas posteriores; el efecto sobre hábito se evalúa aparte.
- **Anti-burbuja:** la diversificación combina MMR, macro-clusters y penalización explícita por
  repetición de género.
- **Evaluable:** el orden se valida con `Recall@k`, `Precision@k`, `NDCG@k`, `MAP` (relevancia) y
  `Coverage`, `Novelty`, `Diversity` (anti-popularidad), tal como define el README.

---

### Conclusión (para el entregable)

El criterio de ranking v1 es **interés + diversidad + exploración controlada**. El criterio
**primario** es la
**similitud de interés** —coseno entre el perfil de usuario (`user_matrix` o `user_centroids`) y
el vector PCA del libro—, que define qué significa "más cercano": afinidad de
tono/temática/accesibilidad, no de género. Sobre esa base se aplica MMR semántico y de género,
con un desempate suave por accesibilidad, y se reservan slots exploratorios con piso de afinidad
exclusivamente para `tail`/`mid`. La popularidad no
filtra ni ordena; sirve para segmentar y medir exposición. El ranking se produce con una arquitectura
**retrieve (por cluster) → score → diversify → explain**, que lo hace escalable, explicable y
evaluable con métricas de relevancia, anti-popularidad y proxy de hábito.
