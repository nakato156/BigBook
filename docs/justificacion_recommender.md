# ¿Tiene sentido un recommender por similitud en nuestro dominio?

Fase de justificación. El objetivo no es asumir que "recomendar ítems parecidos" siempre
es correcto, sino argumentar de forma explícita por qué (o por qué no) la similitud entre
ítems aporta valor real en **nuestro dominio**: una plataforma de descubrimiento de libros
cuyo norte es *ayudar a más personas a sostener el hábito de lectura*, no maximizar clics
ni amplificar los libros ya populares.

## Resumen ejecutivo

> **En nuestro caso, recomendar por similitud SÍ tiene sentido — pero con tres condiciones
> que no son opcionales:** (1) la similitud debe medirse sobre el **vector de gusto
> multidimensional** (`pc_0..pc_172`), no sobre el género ni sobre la popularidad;
> (2) la recomendación por similitud debe **complementarse con diversidad y progresión**
> (macro-clusters, novelty, control cross-género) para no producir burbujas; y (3) la
> popularidad entra solo como **señal secundaria** de calidad/confianza. Sin estas tres
> condiciones, la similitud "pura" degenera en *más de lo mismo* y trabaja en contra del
> objetivo de hábito.

## 1. ¿Conviene recomendar cosas parecidas o complementarias?

En el dominio del consumo de libros el modo dominante **sí es la similitud, no la
complementariedad**, y por una razón estructural del dominio:

- Un libro **no es un producto complementario** como en e-commerce (no compras una funda
  *después* de un teléfono). El usuario que termina un libro quiere, sobre todo, *otro
  libro que también vaya a disfrutar*. La pregunta natural es "dame algo parecido a lo que
  me gustó", lo cual es un problema de similitud.
- La complementariedad sí existe, pero en una forma distinta a la de retail: aparece como
  **secuencia/progresión** (continuar una saga — tenemos la señal `series`, que está activa
  en ~70% del catálogo), o como **rampa de accesibilidad** (de lecturas más accesibles a
  más exigentes para construir hábito). Esto es complementariedad *temporal*, no de cesta.

**Conclusión del punto 1:** la columna vertebral del recommender es la similitud entre
vectores de libro; la complementariedad se modela como diversidad y progresión *encima* de
esa base, no como el mecanismo principal.

## 2. ¿La similitud entre ítems representa realmente valor para el usuario?

Esto depende **enteramente de cómo definimos la similitud**, y aquí está la parte fina del
argumento:

- **Sí representa valor** porque en este proyecto el "ítem" no es una etiqueta de género,
  sino un **vector PCA de gusto** que combina tono, audiencia, accesibilidad, estructura
  multi-género y **semántica de la descripción** (embeddings `embeddinggemma-300m`). Dos
  libros cercanos en este espacio comparten una *vecindad de lectura* que puede cruzar
  géneros — exactamente el patrón que el producto quiere explotar: *"tono juvenil + romance
  + aventura + fantasía ligera + lectura accesible"*, aunque no sean el mismo género.
  Esto es justo lo contrario de la lógica básica "si te gusta fantasía, más fantasía".
- **Riesgo concreto y medible de que la similitud NO represente valor:** según el propio
  análisis de PCA (README, *Interpreting Early Components* / *PCA Block Diagnostics*), los
  ejes de mayor varianza son **tabulares, no semánticos**:
  - `pc_0` = popularidad/engagement (`log_ratings_count`, `log_text_reviews_count`),
  - `pc_1..pc_5` = idioma, missingness de metadata, año de publicación, separación de
    género;
  - lo **semántico** (embeddings) recién domina desde `pc_15` en adelante y con varianza
    individual <1%.

  Es decir: una similitud euclidiana ingenua sobre *todos* los PCs ponderaría la
  **popularidad y la metadata por encima del contenido**. En ese caso "parecido"
  significaría "igual de popular / mismo idioma", lo cual **no es valor de gusto para el
  usuario**. El pipeline ya mitiga esto (`log1p` sobre conteos, *block weighting*
  `1/sqrt(dim)` para que los 256 embeddings no se diluyan, diagnóstico
  `embedding_dominated_first_5_count = 0`), pero la geometría sigue teniendo a la
  popularidad como primer eje.

**Conclusión del punto 2:** la similitud aporta valor **solo si el ranking prioriza la
similitud de interés/semántica y trata la popularidad como secundaria** (regla de negocio
*Avoiding Popularity Bias*). La similitud "tal cual sale de PCA" debe ponderarse o
filtrarse, no usarse cruda.

## 3. ¿Hay riesgo de recomendar "más de lo mismo"?

**Sí, y en nuestro dominio es un riesgo especialmente grave**, por dos motivos:

1. **El objetivo es el hábito, no el clic.** Una burbuja de filtro que encierra al lector
   en una vecindad estrecha puede dar buenas métricas de relevancia a corto plazo, pero
   *no hace crecer el hábito*: el lector se satura, no descubre, y abandona. El daño de
   "más de lo mismo" es mayor aquí que en un recommender de consumo puntual.
2. **Sesgo de popularidad.** Como `pc_0` es popularidad, la similitud cruda tiende a
   reforzar los libros ya populares, reduciendo el descubrimiento para gustos específicos
   o emergentes — exactamente lo que la regla *Avoiding Popularity Bias* pide evitar.

Mitigaciones ya previstas en el diseño (y por qué aplican):

- **Jerarquía de dos niveles de clusters**: 100 clusters finos (vecindades coherentes) +
  10 macro-clusters Ward sobre los centroides. Permite recomendar *dentro* de la vecindad
  (coherencia) pero también *saltar* a vecindades hermanas del mismo macro-cluster
  (diversidad controlada).
- **Género como control de diversidad, no como unidad**: usar los flags `genre_*` para
  forzar variedad intra-lista y romper burbujas cross-género.
- **Métricas anti-burbuja**: `Coverage`, `Novelty` e *intra-list Diversity* (README,
  *Evaluation layers*) existen precisamente para detectar si el modelo está amplificando lo
  popular en vez de descubrir.

**Conclusión del punto 3:** la similitud por sí sola **sí** corre el riesgo de "más de lo
mismo"; el diseño lo neutraliza inyectando diversidad/exploración explícita (macro-clusters,
control de género, novelty) sobre la base de similitud.

## 4. Justificación final (para el entregable)

**En nuestro caso, recomendar por similitud SÍ tiene sentido, porque:**

- El consumo de libros es un problema naturalmente de *substitución/descubrimiento* ("otro
  libro que disfrute"), no de cesta complementaria, así que la similitud entre ítems es el
  mecanismo correcto de base.
- Nuestra unidad de similitud **no es el género** sino un **vector de gusto
  multidimensional** (PCA híbrido: popularidad + metadata + género + semántica de la
  descripción), lo que permite capturar vecindades de lectura cross-género y reales para el
  usuario — el valor que el producto busca.

**…pero NO tiene sentido si la aplicamos de forma ingenua, porque:**

- La geometría PCA tiene la **popularidad como primer eje** y lo semántico en la cola; una
  similitud cruda recomendaría "igual de popular / mismo idioma", no "del mismo gusto".
  → Por eso el ranking prioriza interés/semántica y usa popularidad solo como señal
  secundaria.
- La similitud pura produce **burbujas de filtro**, que en un producto de *hábito de
  lectura* son más dañinas que en uno de consumo puntual.
  → Por eso se complementa con diversidad (macro-clusters, control de género, novelty) y con
  progresión (sagas vía `series`, rampa de accesibilidad).

En una frase: **la similitud es necesaria pero no suficiente**. Es el motor correcto para
este dominio, siempre que se mida sobre el gusto multidimensional, se subordine la
popularidad y se complemente con diversidad y progresión orientadas al hábito de lectura.
