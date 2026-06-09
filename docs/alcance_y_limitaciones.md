# Alcance y limitaciones — qué resolvemos en v1, qué admitimos y qué aplazamos

Pieza de cierre honesto del diseño. El resto de documentos de `docs/` explican **cómo** queremos
recomendar; este declara **hasta dónde llega v1**, qué problemas siguen vivos, y por qué eso es una
decisión consciente y no un descuido. El principio rector es explícito:

> **Atacamos el sesgo de popularidad (P2) y planteamos honestamente la medición del hábito (P1).
> En cambio *no* afirmamos explorar todo el catálogo: el universo de libros sin explorar (P3) es hoy
> una carencia declarada, no un objetivo de v1.** Preferimos un baseline medido, con sesgo residual
> acotado y *declarado*, antes que prometer una cobertura del catálogo que el diseño actual no puede
> sostener. Ser consciente y autocrítico vale más que aparentar que todo está resuelto desde el inicio.

Se conecta con la escalera de evidencia del hábito (ver
[metricas_evaluacion §1bis](metricas_evaluacion.md)): igual que el hábito se *etiqueta* por su nivel
de evidencia (N0/N1/N2) en vez de prometerse entero, aquí cada problema se *etiqueta* por su estado
real en vez de declararse resuelto.

---

## 1. Los tres problemas de negocio y su estado real

El proyecto nació para atacar tres problemas. En v1 mantenemos **dos como objetivos** (P1, P2) y
**reclasificamos el tercero (P3) como carencia declarada**. Este es su estado **honesto** a día de hoy.

| # | Problema | Estado | Detalle |
|---|---|---|---|
| **P1** | **Mantener el hábito de lectura** | **Objetivo — promesa honesta, capacidad pendiente** | La *medición* está bien planteada (escalera N0/N1/N2). El *mecanismo* que construye hábito (rampa de accesibilidad) se apoya en una sola feature ruidosa (`num_pages`, 17.5% imputado). Medimos mejor el hábito; aún no lo generamos mejor. |
| **P2** | **Sesgo de popularidad** | **Objetivo — subordinado en el ranking; residual acotado** | Atenuado en representación (`log1p` + block-weighting) y **sacado del orden**: la similitud usa un subespacio de gusto sin `pc_0..pc_5` (A1) y la popularidad es gate, no factor (A2). Residual *correcto*: los bestsellers afines aún aparecen, pero no porque la popularidad ordene. Falta **verificarlo empíricamente** (evaluación N0/N1). |
| **P3** | **Universo de libros sin explorar** | **⚠️ Carencia declarada (fuera del objetivo de v1)** | El top-k reserva un slot de exploración (A3), pero es un *gesto*, no una solución. Razón de fondo: el **gate de A2 (`ratings_count >= 5`) elimina la cola larga** —los libros de nicho que *son* el universo sin explorar—, y A3 solo salta al macro-cluster más cercano *dentro* de lo que pasa el gate. Atacar P3 de verdad (item cold-start, cobertura como objetivo, presupuesto de exploración) **entra en conflicto con el gate** y se aplaza a v2. No prometemos cubrir el catálogo; lo admitimos como límite. |

**Estado de la capa de ranking:** la capa `retrieve → score → diversify → explain` —donde viven las
mitigaciones— **ya existe** ([`src/reduction/recommend.py`](../src/reduction/recommend.py)) con
A1–A4 incrustados y testeados. Lo que queda pendiente downstream es la **evaluación** (N0/N1) sobre
ella (ver [metricas_evaluacion §5](metricas_evaluacion.md), [informacion_disponible §4](informacion_disponible.md)).

---

## 2. Las cuatro contradicciones de diseño

Detectadas en la revisión crítica. Se resolvieron incrustando el fix **desde el inicio** de la capa
de ranking (barato así; caro si se parchea después). Estado actual en la columna derecha.

| # | Contradicción | Dónde | Estado |
|---|---|---|---|
| **C1** | La similitud coseno **no** elimina `pc_0` (popularidad), solo la magnitud; y el vector de usuario hereda `pc_0` al promediar libros | [criterio_scoring §1](criterio_scoring.md), [perfil_usuario §2](perfil_usuario.md) | ✅ Resuelto (A1) |
| **C2** | El factor `f_calidad` del modelo **es la misma fórmula** que la baseline B1 (`log1p(ratings_count)·average_rating`) → "superar a B1" es parcialmente circular | [criterio_scoring §2](criterio_scoring.md), [baseline §2](baseline.md) | ✅ Resuelto (A2) |
| **C3** | La exploración del catálogo **solo se mide** (`Coverage`/`Novelty`), no se genera; `retrieve` por vecindad nunca propone clusters lejanos | [criterio_scoring §4](criterio_scoring.md) | ◑ Mitigada: A3 ya *genera* algo de exploración. Pero la **ambición P3 se reclasifica como carencia** (ver §1): el gate entierra la cola larga, así que la cobertura real del catálogo queda fuera de v1. |
| **C4** | Positivo = `is_read AND rating_clean >= 4` **vacía** a los lectores casuales/nuevos, que caen a un fallback de cold-start que **es popularidad** | [perfil_usuario §4](perfil_usuario.md) | ✅ Resuelto (A4, fallback); C4-positivos = v1.5 |

---

## 3. Plan de mitigación en tres cubos

La estrategia no es resolver todo, sino clasificar cada punto en **arréglalo barato**, **admítelo** o
**aplázalo**.

### Cubo A — Arréglalo barato (✅ implementado en la capa de ranking)

Cuatro cambios pequeños, ninguno refitea PCA ni rompe artefactos existentes
(`master_feature_matrix`, `user_matrix`, clusters intactos). **Ya viven** en
[`src/reduction/recommend.py`](../src/reduction/recommend.py), testeados en
[`tests/test_recommend.py`](../tests/test_recommend.py).

| Fix | Contradicción | Cambio | Costo |
|---|---|---|---|
| **A1** | C1 | Calcular el coseno de interés sobre un **subespacio de gusto** que excluya `pc_0..pc_5` (ejes tabulares = popularidad/idioma/missingness, ya identificados en `master_pca_meta.json`) | Selección de columnas |
| **A2** | C2 | Popularidad como **gate**, no multiplicador: descartar libros con `ratings_count < N` (piso de calidad de datos) y **no** usar popularidad en el orden | Un filtro |
| **A3** | C3 | **Slot fijo de exploración** en el top-k (p. ej. 8 de vecindad + 2 de un macro-cluster que el usuario no toca) | Ensamblado de lista |
| **A4** | C4 (cold-start) | Fallback de cold-start = **un libro accesible por macro-cluster**, sin popularidad | Cambio de regla |

> A4 es además una corrección de texto en [perfil_usuario §4](perfil_usuario.md): hoy dice
> "popularidad moderada + diversidad"; basta quedarse con la parte de diversidad.

### Cubo B — Admítelo como límite consciente (efecto sobre el resultado)

No se resuelve en v1; se declara y se explica **cómo afecta al resultado**.

- **`Recall@k` premia lo predecible.** El número offline mide ajuste predictivo (N0), no crecimiento
  de hábito. *Efecto:* un modelo puede ganar Recall recomendando secuelas obvias; por eso
  `Coverage`/`Novelty` se tratan como **co-primarias**, no como guardrail secundario.
- **Cross-género sobre geometría débil.** El descubrimiento cross-género lo aportan embeddings en
  `pc_15+` (<1% varianza c/u), y solo hay 5 géneros correlacionados. *Efecto:* v1 captura
  cross-género **grueso** (multi-etiqueta), no sorpresas semánticas profundas.
- **Accesibilidad = solo `num_pages`** (17.5% imputado). *Efecto:* la "rampa de accesibilidad"
  **no es una feature de v1**; `num_pages` queda como desempate suave. Quitar la promesa de rampa de
  los docs que la insinúan ([justificacion_recommender §1](justificacion_recommender.md),
  [criterio_scoring §2](criterio_scoring.md)).
- **Causalidad del hábito.** No medible offline; es N2 (telemetría). Ya etiquetado en la escalera.
- **Universo sin explorar (P3) = carencia declarada.** Reclasificado: ya no es objetivo de v1.
  Razón de fondo —**tensión gate↔cola larga**: el gate de A2 (`ratings_count >= 5`) descarta justo
  los libros de nicho/poco valorados que *forman* el universo sin explorar, y A3 solo explora
  *dentro* de lo que pasa el gate y salta al macro más cercano. *Efecto:* la cobertura del catálogo
  no puede crecer mucho por construcción; el item cold-start y la cola larga real son **deuda
  explícita de v2**, no una promesa de v1.

### Cubo C — Aplázalo explícitamente a v2

- Mecanismo de exploración serio (bandit / serendipia presupuestada) más allá del slot fijo A3.
- Recuperar la **capa implícita** en el perfil (hoy `is_read AND rating>=4` descarta
  `want_to_read`/`read_no_rating`). *v1.5 barato:* añadir `read_no_rating` como positivo débil para
  no vaciar a los casuales.
- **Telemetría de producto** (sesiones, retornos, libros terminados tras recomendación) → habilita
  N2 (medición causal del hábito).

---

## 4. Fases del "poco a poco"

```text
v1   (honesto, con guardrails)  → A1–A4 al construir el ranking + Cubo B documentado como límites
v1.5 (barato)                   → positivos débiles (read_no_rating) para no vaciar a los casuales
v2   (cuando haya tráfico)      → exploración seria + telemetría de hábito (N2, causal)
```

| Fase | P1 hábito *(objetivo)* | P2 popularidad *(objetivo)* | P3 exploración *(carencia)* |
|---|---|---|---|
| **Estado actual (v1)** | Promesa honesta (N0/N1) + guardrails de relevancia/diversidad | ✅ A1 (subespacio) + A2 (gate) implementados | ⚠️ Solo gesto A3; gate↔cola larga sin resolver — **carencia declarada** |
| **Pendiente v1** | evaluación N0/N1 sobre el ranking | verificación empírica (correlación popularidad↔output) | — (no es objetivo de v1) |
| **v1.5** | + casuales no se vacían (C4-positivos) | — | — |
| **v2** | N2 causal con telemetría | revisión por evidencia A/B | exploración seria + item cold-start *(si se retoma)* |

---

## 5. Estado de implementación (para no confundir doc con código)

| Punto | Documentado | Implementado |
|---|---|---|
| Escalera de evidencia del hábito (N0/N1/N2) | ✅ | n/a (es marco de evaluación) |
| A1 subespacio de gusto sin `pc_0..pc_5` | ✅ (aquí) | ✅ `src/reduction/recommend.py::taste_pc_indices` (test `test_a1_*`) |
| A2 popularidad como gate | ✅ (aquí) | ✅ `quality_gate_mask` + scoring sin popularidad (test `test_a2_*`) |
| A3 slot de exploración | ✅ (aquí) | ✅ `pick_exploration_macro` (test `test_a3_*`) |
| A4 cold-start sin popularidad | ✅ (aquí) | ✅ `Recommender.recommend_cold_start` (test `test_a4_*`) |
| Recuperar capa implícita en perfil | ✅ (Cubo C) | ❌ (v1.5) |

La capa de ranking (`retrieve → score → diversify → explain`) ya existe en
[`src/reduction/recommend.py`](../src/reduction/recommend.py), con los cuatro fixes incrustados
**de raíz** y cubiertos por [`tests/test_recommend.py`](../tests/test_recommend.py) (8 tests, uno
por contradicción). Ejecutable con `env/bin/python -m src.reduction.recommend` (escribe una muestra
honesta en `data/outputs/recommendations/`, no batch de los 699k usuarios).

> **Lectura honesta (actualizada):** **C1–C4 ya están resueltos en código**, no solo documentados —
> la capa de ranking donde vivían las mitigaciones existe y está testeada. Lo que **sigue presente**
> es: (a) el residual de P2 —los bestsellers genuinamente afines aún aparecen en el top, lo cual es
> *correcto* (la popularidad ya no **ordena**, pero tampoco se esconde lo popular relevante); (b)
> **P3 reclasificado como carencia** —A3 es un gesto mínimo, y el gate de A2 entierra la cola larga,
> así que la cobertura real del catálogo queda fuera del scope de v1 (deuda de v2); (c) los límites
> del **Cubo B** (accesibilidad sobre `num_pages` ruidoso, cross-
> género sobre geometría débil); y (d) lo aplazado al **Cubo C** (capa implícita, exploración seria,
> telemetría N2). La deuda restante queda **consciente y declarada**, no escondida.

### Fuera del scope de `recommend.py` (a propósito)

Resolver C1–C4 no es construir el sistema entero. El módulo deja fuera, deliberadamente:

- el set de **"ya leídos"** por usuario (es la capa de split temporal / evaluación — `recommend()`
  lo acepta como parámetro `exclude_book_ids`, pero no lo construye);
- el **batch** completo de los 699k usuarios (solo genera una muestra);
- la **evaluación** N0/N1 (Recall@k, Coverage, proxies de hábito) — sigue siendo el siguiente paso.

---

### Conclusión (para el entregable)

v1 mantiene **dos objetivos** y admite el tercero. **Ataca el sesgo de popularidad (P2)** —subordinado
en código (A1+A2), con residual acotado y declarado, pendiente de verificación empírica— y **plantea
el hábito (P1)** con una promesa etiquetada por evidencia (N0/N1/N2). En cambio **declara el universo
sin explorar (P3) como carencia, no como objetivo de v1**: el gate de calidad y la cobertura del
catálogo se contradicen (tensión gate↔cola larga), y resolverlo de verdad —item cold-start,
exploración seria— se aplaza a v2. Las contradicciones de diseño C1–C4 quedan subordinadas en código
(Cubo A: A1–A4 en [`src/reduction/recommend.py`](../src/reduction/recommend.py), con tests); lo demás
se admite (Cubo B) o se aplaza (Cubo C) de forma explícita. La madurez se mide por **cuánta de esta
deuda se salda con evidencia**, no por cuánto se promete al inicio — y el siguiente saldo es
**evaluar** (N0/N1) esta capa, no construirla.
