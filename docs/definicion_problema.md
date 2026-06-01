# Definición del problema de recomendación

Punto de partida del diseño: antes de justificar técnicas, datos o métricas, hay que enunciar con
precisión **qué recomienda el sistema y a quién**, y **qué acción se quiere provocar**. Todo lo
demás (representación, perfil, scoring, evaluación) existe para servir a este enunciado.

---

## Enunciado del problema (X / Y / Z)

> **Nuestro sistema recomienda libros (`book_id`, cada uno un vector PCA `pc_0..pc_172`) a
> lectores (`user_id`), con base en el gusto multidimensional inferido de su historial de lectura
> (lecturas, ratings, reviews) y en la similitud libro-a-libro en el espacio reducido, buscando
> optimizar lecturas relevantes y motivadoras que se empiezan y se completan —priorizando la
> afinidad de interés sobre la popularidad— para que los lectores sostengan y hagan crecer el
> hábito de lectura.**

La misma idea en forma `X / Y / Z`:

```text
Recomendamos  X = libros del catálogo, como vectores PCA, agrupados en clusters de gusto
con base en   Y = el historial de interacción del usuario + similitud en el espacio PCA +
                  clusters de libros (género como filtro/explicación/diversidad; popularidad
                  como señal secundaria)
para optimizar Z = lecturas relevantes empezadas y terminadas (proxy: is_read + rating positivo)
                  que sostienen el hábito de lectura (retención), evitando el sesgo de
                  popularidad y las burbujas de un solo género.
```

---

## ¿Qué es un "usuario"?

Un **lector** identificado por `user_id`, representado por su **historial de interacción** sobre
libros: qué leyó (`is_read`), calificó (`rating`) y reseñó (`has_review_text`), capturado en
`interactions_curated.parquet`.

- **No** se modela como una etiqueta de género. El usuario es un **vector de gusto
  multidimensional** construido agregando los vectores PCA de los libros con los que interactuó
  positivamente.
- Un usuario nuevo sin historial (cold-start) se representa con libros/géneros semilla elegidos al
  registrarse, o con una mezcla inicial de popularidad moderada + diversidad.

*(Detalle en [perfil_usuario.md](perfil_usuario.md).)*

---

## ¿Qué es un "ítem"?

Un **libro**, identificado por `book_id`, donde **una fila = un vector de libro** en el espacio
PCA (`pc_0..pc_172`). Ese vector es la unidad de recomendación y de clustering.

- El género es una **señal, filtro, explicación o control de diversidad** — nunca la unidad del
  modelo de recomendación.

*(Detalle en [representacion_items.md](representacion_items.md).)*

---

## ¿Qué significa una recomendación útil?

Una lista corta de libros que el lector probablemente **lea y disfrute**, y que además sea:

- **Relevante:** cercana al gusto por afinidad de interés primero (distancia en el espacio PCA /
  cluster compartido), no por popularidad.
- **Accesible y motivadora:** alineada con el objetivo de construir hábito, favoreciendo libros
  abordables, no solo los más populares.
- **Diversa pero coherente:** explota patrones cross-género (tono juvenil + romance + aventura +
  fantasía ligera) en vez de encerrar al lector en un solo género.
- **Explicable:** justificable por su cluster/macro-cluster y género ("porque te gustó X, del mismo
  vecindario de lectura").

---

## ¿Qué acción se quiere provocar?

La acción primaria es **empezar y completar una lectura**. En los datos de Goodreads el proxy
observable es `is_read = True`, reforzado por un `rating` alto y/o una review escrita. El objetivo
último es la **retención del hábito de lectura**, no un clic ni una compra aislada.

| Nivel | Señal en los datos | Rol |
|---|---|---|
| **Acción objetivo** | `is_read` (lectura) | Lo que optimizamos |
| Confirmación de calidad | `rating` alto, `has_review_text` | Refuerzo |
| Engagement / interés | click / abrir ficha del libro | Proxy temprano (a instrumentar en producto) |
| **Objetivo de negocio** | retención / hábito | Métrica norte (north-star) |

> No optimizamos **compra** (no es un dataset transaccional) ni **click/reproducción** (no hay
> telemetría); optimizamos **lectura completada** como proxy del hábito. *(Cómo se mide eso, en
> [metricas_evaluacion.md](metricas_evaluacion.md).)*

---

### Conclusión (para el entregable)

**Nuestro sistema recomienda libros (vectores PCA) a lectores, con base en el gusto
multidimensional inferido de su historial de lectura y en la similitud entre libros en el espacio
reducido, buscando optimizar lecturas relevantes y accesibles que se empiezan y se completan, para
sostener el hábito de lectura.** El usuario es un vector de gusto (no una etiqueta de género); el
ítem es un vector de libro (no su género); una recomendación útil es relevante, accesible, diversa
y explicable; y la acción objetivo es la **lectura completada** (`is_read`), con la **retención**
como norte — no la compra ni el clic.
