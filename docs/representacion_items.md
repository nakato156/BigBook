# Definición de la representación de los ítems

Fase de representación de ítems. El objetivo es explicar **cómo los libros se vuelven
comparables entre sí**: qué información define a cada ítem, por qué es relevante para el
dominio (descubrimiento que sostiene el hábito de lectura) y qué limitaciones tiene.

> **Principio rector:** `one row = one book_id = one book vector`. Cada libro es **un vector**
> en un espacio común; la comparabilidad entre ítems es, literalmente, la distancia entre sus
> vectores en ese espacio.

---

## 1. ¿Cómo se vuelven comparables los ítems?

Cada libro se representa como un **vector PCA de 173 dimensiones** (`pc_0..pc_172`) en
`data/features/master_feature_matrix.parquet` (108,227 libros). Ese vector es el resultado de
fusionar tres familias de atributos heterogéneos (texto, categorías, metadata, popularidad) en
**un único espacio numérico homogéneo**, donde dos libros son "parecidos" si su distancia
(coseno/euclidiana) es pequeña.

La clave de la comparabilidad es que **todo se proyecta al mismo espacio**: un libro con mucha
descripción y otro con poca, uno de fantasía y otro de romance, uno popular y otro de nicho,
quedan todos descritos por el mismo conjunto de 173 ejes latentes. Sin esa unificación no
serían comparables (texto vs. flags vs. conteos no se pueden restar entre sí).

---

## 2. ¿Qué atributos representan cada ítem?

La representación es **híbrida**, en tres bloques (276 columnas antes de PCA):

### Bloque numérico (9 columnas) — calidad, tamaño y popularidad

| Atributo | Qué captura | Por qué es relevante |
|---|---|---|
| `average_rating` | Calidad percibida | Señal de satisfacción, no de gusto individual |
| `log_ratings_count` | Volumen de ratings (`log1p`) | Popularidad/confianza, atenuada para no dominar |
| `log_text_reviews_count` | Volumen de reviews (`log1p`) | Engagement de la comunidad |
| `num_pages` (+ `num_pages_missing`) | Extensión / esfuerzo de lectura | **Accesibilidad**: clave para crear hábito |
| `publication_year` (+ `publication_year_missing`) | Época | Clásico vs. contemporáneo; gusto generacional |
| `author_count` | Coautoría | Antologías/colaboraciones vs. autor único |
| `genre_count` | Amplitud multi-género | Estructura cross-género del libro |

Los flags `*_missing` son **señal intencional**: la ausencia de metadata correlaciona con tipo
de edición y calidad del registro.

### Bloque binario/categórico (11 columnas) — género e idioma

| Atributo | Qué captura |
|---|---|
| `series` | Pertenece a saga (0/1) — relevante para **progresión** de lectura |
| `genre_fantasy/mystery/history/ya/romance` | 5 flags multi-etiqueta de género |
| `language_code_*` (one-hot, 5) | Idioma (eng, en-US, en-GB, en-CA, other) |

El género es **multi-etiqueta** (un libro puede tener varios flags), lo que ya rompe la lógica
"un libro = un género". Aquí el género entra como **atributo del vector**, pero downstream se
usa como filtro/diversidad, no como unidad de clustering.

### Bloque de embeddings (256 columnas) — **el contenido semántico**

`emb_0..emb_255`, generados con `google/embeddinggemma-300m` sobre la **descripción** del libro
(fallback: título → `[no description]`). Capturan **tono, temática, audiencia y estilo** —
exactamente lo que permite descubrir patrones cross-género (*"tono juvenil + romance + aventura
+ fantasía ligera"*) que ni el género ni los ratings expresan.

---

## 3. ¿Usamos texto, categorías, metadata, ratings…? Decisión y por qué

| Tipo de señal | ¿Se usa? | Rol en la representación |
|---|---|---|
| **Texto** (descripción) | ✅ Sí, vía 256 embeddings | Núcleo semántico: *de qué trata y cómo se siente* el libro |
| **Categorías** (género) | ✅ Sí, 5 flags multi-etiqueta | Estructura temática gruesa; señal, no unidad |
| **Metadata** (páginas, año, idioma, serie, autores) | ✅ Sí | Accesibilidad, época, progresión, formato |
| **Ratings/popularidad** | ✅ Sí, pero **atenuados** | `log1p` + block-weighting → señal secundaria de calidad |
| **Tags libres / shelves** | ❌ No | No están en el master de 17 columnas (eran del pipeline removido) |
| **Reviews individuales** | ❌ No (como atributo del ítem) | Se usan en el perfil de usuario, no en el vector del libro |

**Por qué la mezcla y no solo embeddings:** los embeddings solos darían similitud puramente
temática, ignorando accesibilidad, popularidad y época, que importan para el objetivo de hábito.
Por eso se combinan los tres bloques.

**Por qué el block-weighting es crítico:** cada bloque se estandariza por separado y se multiplica
por `1/sqrt(dim)` (pesos 0.333 / 0.302 / 0.0625). Sin esto, los 256 embeddings dominarían el PCA
solo por contar con más columnas, y la metadata desaparecería. El diagnóstico
`embedding_dominated_first_5_count = 0` confirma que el balance funciona.

---

## 4. ¿La representación captura lo importante del dominio?

**Sí, en lo esencial**, y se puede verificar en los ejes del PCA (README, *Interpreting Early
Components*):

- Captura **contenido** (embeddings → 158 componentes de cola semántica).
- Captura **popularidad/engagement** (`pc_0`) sin dejar que lo domine todo.
- Captura **género y estructura cross-género** (`pc_5`, `pc_10`, `pc_13`).
- Captura **accesibilidad/extensión** (páginas en `pc_4`, `pc_12`) y **época** (`pc_1`, `pc_10`).
- Captura **calidad de metadata** como señal (ejes de missingness `pc_2`, `pc_14`).

Es decir, los ejes que el modelo descubrió solo coinciden con las dimensiones que el dominio
considera relevantes: *de qué trata, qué tan accesible es, qué tan conocido es, de qué género y
época*.

PCA retiene **95.01% de la varianza** con 173 de 276 dimensiones (−37%), así que la compresión
no sacrifica información sustancial.

---

## 5. Limitaciones de la representación

Honestas y explícitas:

- **PCA es lineal:** no captura relaciones no lineales entre metadata, género y semántica. Dos
  libros con interacción temática compleja pueden quedar más lejos de lo real.
- **Ejes latentes, no interpretables 1:1:** los `pc_*` son combinaciones; el signo es arbitrario.
  Sirven para similitud, no como explicación directa al usuario (para eso se usa género/cluster).
- **La popularidad es el primer eje (`pc_0`):** aunque atenuada, la semántica solo domina desde
  `pc_15` con varianza <1%. Una similitud cruda aún pesa metadata/popularidad sobre contenido →
  el ranking debe priorizar interés/semántica.
- **Dependencia de la descripción:** libros con descripción vacía caen al título o a
  `[no description]`; su vector semántico es pobre y pueden agruparse artificialmente.
- **Idioma casi constante:** todo inglés (4 variantes), `language_code_other` vacío → poco poder
  discriminante real de ese atributo.
- **Metadata faltante relevante:** `num_pages` (17.5%) y `publication_year` (20.1%) imputados por
  mediana; se conserva el flag de missingness, pero la imputación añade ruido.
- **Atributos que quedaron fuera:** `format`, `publisher`, `is_ebook`, `to_read_count`, `theme_*`
  y tags libres no entran (vivían en el pipeline per-category que fue removido) → posible mejora
  futura.
- **Representación estática:** el vector del libro no cambia con el tiempo ni con el contexto del
  usuario; cualquier cambio en curación, embeddings o pesos altera la geometría (hay que re-fitear).

---

### Conclusión (para el entregable)

Cada **ítem se define como un vector PCA de 173 dimensiones** que fusiona tres familias de
atributos: **texto** (256 embeddings de la descripción → tono y temática), **categorías y
metadata** (género multi-etiqueta, serie, idioma, páginas, año, autores → accesibilidad, época y
estructura) y **popularidad atenuada** (`average_rating`, `log1p` de conteos → calidad como señal
secundaria). Esa información es relevante porque cubre exactamente las dimensiones que importan
para *descubrir lecturas relevantes y accesibles que sostengan el hábito*: **de qué trata, qué tan
accesible es, qué tan conocido es, y de qué género/época**. La comparabilidad entre ítems es la
distancia en ese espacio común, balanceado por block-weighting para que ninguna familia domine.
Sus límites principales —linealidad del PCA, ejes no interpretables, peso residual de la
popularidad y dependencia de la descripción— se conocen y se compensan en el ranking
(interés/semántica primero, popularidad y género como señales secundarias de calidad y diversidad).
