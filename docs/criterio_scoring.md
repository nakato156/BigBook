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

Con `user_centroids`, la similitud de interés puede evaluarse por modo de lectura y agregarse con
el peso de compromiso del modo:

```text
sim_interés(u,b) = max_c [ centroid_weight(u,c) · cos(centroide(u,c), pca(b)) ]
```

El baseline `user_matrix` es el caso simple con un solo vector y peso 1.

Por qué coseno y no euclidiana cruda:

- El **gusto es una dirección**, no una intensidad. Dos libros del mismo "sabor" deben ser
  cercanos aunque uno sea más popular o más largo (mayor magnitud en algunos ejes).
- La euclidiana penaliza diferencias de magnitud que en parte vienen de la popularidad
  (`pc_0`), reintroduciendo sesgo de popularidad por la puerta de atrás. El coseno lo atenúa.
- Es consistente con los *spot checks* de coseno que ya usa `master_pca_meta.json`.

`sim ∈ [-1, 1]`; más alto = más cercano = más afín al gusto.

---

## 2. ¿Qué significa que un ítem tenga mayor score?

La **similitud no es el score final**. Un ranking basado solo en similitud produce "más de lo
mismo" (ver [justificacion_recommender](justificacion_recommender.md)). El **score** es una
combinación que ordena las recomendaciones según el objetivo del producto, no solo según el
parecido. Un score mayor significa: *este libro es afín al gusto **y** tiene buena calidad/
confianza **y** aporta a una experiencia diversa, accesible y motivadora*.

Forma conceptual del score (jerárquica, no una suma plana):

```text
score(u, b) =  similitud_de_interés(u, b)        ← criterio PRIMARIO (coseno en PCA)
             · f_calidad(b)                        ← señal SECUNDARIA (rating, popularidad log)
             · f_habito(u,b)                       ← accesibilidad + compromiso del modo lector
             − penalización_redundancia(b | lista) ← diversidad (MMR sobre la lista)
             + bonus_descubrimiento(b)             ← novelty / anti-burbuja (controlado)
```

Las reglas de prioridad (qué manda sobre qué) son las que importan:

1. **Interés primero.** `similitud_de_interés` domina el orden. La popularidad **nunca** puede
   superar a un libro más afín; solo desempata o ajusta dentro de niveles de afinidad similar.
2. **Popularidad/calidad como factor secundario y atenuado.** `f_calidad` usa `average_rating` y
   `log1p(ratings_count)` como **confianza** ("este libro afín además es de calidad y no es un
   error de datos"), no como objetivo. Es un multiplicador suave, acotado, para no recrear el
   ranking por bestseller.
3. **Hábito lector como factor explícito.** Ante afinidad parecida, favorecer lo más accesible
   (extensión razonable, etc.) y lo conectado con modos de lectura comprometidos del usuario. En
   los artefactos actuales, esa segunda parte existe como `centroid_weight` en
   `user_centroids`: rating alto, review y duración de lectura coherente no mueven la geometría
   del gusto, pero sí indican qué modo lector tiene más compromiso.
4. **Diversidad explícita.** Se penaliza la redundancia con lo ya incluido en la lista (estilo
   **MMR**: maximal marginal relevance) y se usan los **macro-clusters** y el género para que la
   lista cubra más de una vecindad. Esto evita el filtro burbuja.

---

## 3. Diferencia conceptual entre formas de scoring

| Enfoque de scoring | Qué ordena | Problema en nuestro dominio |
|---|---|---|
| **Solo popularidad** (`ratings_count`, `average_rating`) | Lo más conocido primero | Ignora el gusto; amplifica bestsellers; mata el descubrimiento. **Rechazado** como criterio principal. |
| **Solo similitud** (coseno puro) | Lo más parecido primero | "Más de lo mismo"; burbuja; no distingue calidad ni accesibilidad. Necesario pero **insuficiente**. |
| **Similitud + diversidad (MMR)** | Parecido pero variado | Mejor; evita redundancia. Base de nuestra lista. |
| **Score híbrido jerárquico** (el nuestro) | Interés → calidad → hábito/accesibilidad → diversidad/novelty | Equilibra relevancia, confianza y hábito. **Adoptado.** |
| **Cluster-first** (vecindad → ranking interno) | Primero el barrio, luego dentro | Eficiente y explicable; lo usamos como **arquitectura** del retrieval (ver §4). |

La diferencia conceptual central: un score por **popularidad** responde "¿qué leen todos?"; uno
por **similitud** responde "¿a qué se parece lo que te gustó?"; el nuestro responde "¿qué te va a
gustar, que además valga la pena, sea accesible y no sea siempre lo mismo?".

---

## 4. Arquitectura del ranking (retrieve → score → diversify)

Coherente con el clustering ya construido (KMeans k=100 + 10 macro-clusters):

```text
1. RETRIEVE   Encontrar los clusters/macro-clusters más cercanos al vec_gusto(u)
              y traer los libros candidatos de esas vecindades (+ algo de exploración).
2. SCORE      Ordenar candidatos por score(u,b): interés (coseno) como criterio primario,
              calidad y hábito/accesibilidad como factores secundarios.
3. DIVERSIFY  Reordenar con MMR + control de género/macro-cluster para que la lista final
              sea diversa y cubra más de una vecindad.
4. EXPLAIN    Justificar cada ítem por su cluster/género ("porque te gustó X, del mismo
              vecindario de lectura").
```

El paso de retrieve por cluster hace el sistema **escalable** (no se puntúan 108k libros por
usuario) y **explicable** (la vecindad da la razón de la recomendación).

---

## 5. ¿Por qué el ranking producido tiene sentido?

- **Relevante:** ordena por afinidad de gusto real (coseno en un espacio que captura contenido),
  no por popularidad. Responde a *"esto te va a gustar"*.
- **Confiable:** la calidad/popularidad atenuada filtra ruido de datos sin secuestrar el orden.
- **Pro-hábito:** el score no se queda en similitud. Usa accesibilidad y, cuando se consume
  `user_centroids`, `centroid_weight` como señal de modos de lectura más comprometidos. Además,
  el objetivo offline `is_read` (lo que el usuario realmente leyó después, medible con split
  temporal) alinea la validación con *empezar y terminar lecturas*.
- **Anti-burbuja:** la diversificación (MMR + macro-clusters + género) garantiza que el top-k no
  sea "cinco veces el mismo libro".
- **Evaluable:** el orden se valida con `Recall@k`, `Precision@k`, `NDCG@k`, `MAP` (relevancia) y
  `Coverage`, `Novelty`, `Diversity` (anti-popularidad), tal como define el README.

---

### Conclusión (para el entregable)

El criterio de ranking es un **score híbrido y jerárquico**. El criterio **primario** es la
**similitud de interés** —coseno entre el perfil de usuario (`user_matrix` o `user_centroids`) y
el vector PCA del libro—, que define qué significa "más cercano": afinidad de
tono/temática/accesibilidad, no de género. Sobre esa base se aplican, **subordinados**, factores
de **calidad/popularidad atenuada** (confianza, vía `average_rating` y `log1p` de conteos), de
**hábito/accesibilidad** (`centroid_weight`, extensión razonable, compromiso lector) y de
**diversidad/novelty** (MMR + macro-clusters + género para evitar "más de lo mismo"). Un score
mayor significa "afín **y** valioso **y** que aporta a una lista diversa y motivadora", no
simplemente "más parecido" ni "más popular". El ranking se produce con una arquitectura
**retrieve (por cluster) → score → diversify → explain**, que lo hace escalable, explicable y
evaluable con métricas de relevancia, anti-popularidad y proxy de hábito.
