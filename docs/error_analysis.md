# Análisis de errores del recomendador

Este documento estudia **dónde** falla o acierta el pipeline, no solo cuánto marca una métrica
agregada. Que el modelo supere o no un baseline es una comparación de sistemas; el análisis de
errores busca atribuir cada resultado a candidate generation, scoring, diversidad, exploración,
identidad de ítem o límites del dato observado.

## 1. Protocolo

Los casos se reconstruyeron con el mismo protocolo de evaluación:

- corte global: `2016-06-09T19:30:05.200000+00:00`;
- perfil construido solo con positivos anteriores al corte;
- positivo: `is_read=True AND rating_clean>=4`;
- catálogo y popularidad históricos;
- exclusión de consumidos de train;
- top-10 con cinco clusters recuperados y hasta dos slots exploratorios.

En `k=10`, 83 de 2,038 usuarios evaluables tienen al menos un acierto del modelo y 1,955 no tienen
aciertos. Esto no identifica por sí solo la causa: hay que inspeccionar el recorrido del objetivo
futuro.

## 2. Taxonomía de errores

### E1. Fallo de candidate generation

El libro relevante está en el catálogo elegible, pero su cluster no pertenece a los cinco
recuperados y tampoco entra como exploración.

Diagnóstico:

```text
relevant_book ∈ C(u,t)
relevant_cluster ∉ top_5_clusters(u)
relevant_book ∉ exploration
```

El ranker nunca tuvo oportunidad de puntuar el libro. Cambiar MMR no resolvería este caso.

### E2. Fallo de scoring o diversificación

El libro relevante pertenece a un cluster recuperado, pero queda fuera del top-k:

```text
relevant_cluster ∈ top_5_clusters(u)
relevant_book ∉ top_k
```

Posibles causas:

- similitud menor que otros candidatos;
- bonus de accesibilidad;
- penalización MMR por redundancia;
- penalización de género;
- competencia entre distintas ediciones o títulos muy cercanos.

### E3. Exploración irrelevante

El slot de exploración cumple las reglas de exposición, pero no coincide con futuros positivos.
En la evaluación:

```text
exploration_precision@k = 0.000280
exploration_hit_rate@k  = 0.000561
```

La exploración aumenta coverage y novelty, pero el criterio actual de “fuera del macro-cluster”
puede alejar demasiado el candidato del gusto observable.

### E4. Identidad de obra incompleta

El usuario consumió una edición y recibe otra edición con diferente `book_id`. Técnicamente no se
viola la exclusión, pero desde producto es una recomendación duplicada.

Ejemplo observado:

```text
consumido:    The Raven Boys, book_id=17675462
recomendado:  The Raven Boys, book_id=13449693
```

La corrección requiere identidad canónica de obra, no un cambio en el score.

### E5. Etiqueta offline incompleta

Un libro recomendado se cuenta como fallo si no aparece como positivo futuro, aunque:

- el usuario nunca haya sido expuesto a él;
- lo haya leído sin registrarlo;
- lo haya agregado después del periodo observado;
- sea relevante pero no forme parte del único futuro registrado.

Es un límite de observación, no necesariamente un error semántico del modelo.

### E6. Perfil demasiado amplio o desactualizado

Usuarios con cientos de positivos y cinco géneros pueden tener muchos modos de lectura. V1
conserva como máximo cuatro centroides y recupera cinco clusters globales. Una preferencia futura
minoritaria puede quedar fuera.

## 3. Caso fuerte A: acierto en el primer puesto

Usuario:

```text
f54b46386ef443e4fe44c33bc4cd35b4
```

Contexto:

- 35 positivos en train;
- un único positivo futuro evaluable;
- clusters recuperados: `[54, 82, 0, 75, 24]`.

Objetivo futuro:

```text
Harry Potter and the Chamber of Secrets
book_id=15881
cluster=54
```

Resultado:

```text
rank=1
slot=interest
```

Interpretación:

- el objetivo pertenecía al cluster más cercano;
- candidate generation lo incluyó;
- el score de interés lo colocó antes de candidatos diversos;
- no dependió de un slot exploratorio.

Es un caso limpio donde perfil, retrieval y ranking están alineados.

## 4. Caso fuerte B: continuidad de saga y varios aciertos

Usuario:

```text
9f1c9f43f46d6504712a4429dcf229d7
```

Contexto:

- solo 3 positivos en train;
- 8 positivos futuros;
- clusters recuperados: `[67, 13, 95, 11, 55]`;
- 3 aciertos en el top-10.

Aciertos:

| Rank | Libro | Cluster |
|---:|---|---:|
| 3 | Angels Flight | 13 |
| 5 | A Darkness More Than Night | 13 |
| 9 | The Last Coyote | 13 |

Otros objetivos futuros del cluster 13 quedaron fuera:

- *City of Bones*;
- *Mr. Mercedes*;
- *The Concrete Blonde*;
- *Trunk Music*.

Interpretación:

- el sistema identificó correctamente el modo mystery/policial;
- la recuperación funcionó;
- el top-10 privilegió continuidad de Harry Bosch;
- algunos objetivos quedaron fuera por competencia dentro del mismo cluster.

Este caso combina éxito de candidate generation con error parcial de ordenamiento.

## 5. Caso de fallo A: objetivo fuera del retrieval

Usuario:

```text
b11dee4ef20822ad0281a474baf9023f
```

Contexto:

- 331 positivos en train;
- un único objetivo futuro;
- clusters recuperados: `[72, 42, 91, 30, 63]`.

Objetivo futuro:

```text
Reflected in You (Crossfire, #2)
book_id=13596809
cluster=54
```

El cluster 54 no fue recuperado. El top-10 se concentró en romance histórico y fantasy romance de
los clusters 72, 42, 91, 30 y 63.

Clasificación:

> **E1: fallo de candidate generation.**

El objetivo estaba en el catálogo histórico, pero nunca llegó a scoring. Las acciones relevantes
son aumentar el recall de retrieval, recuperar por cada modo de usuario o usar ANN global. Ajustar
solo MMR no puede corregir este caso.

## 6. Caso de fallo B: gusto amplio y presupuesto de clusters insuficiente

Usuario:

```text
9ebb0290a3a302189bb5712eb8898cf9
```

Contexto:

- 223 positivos en train;
- 220 positivos futuros;
- clusters recuperados: `[71, 68, 57, 2, 8]`;
- cero aciertos en top-10.

Parte de los objetivos sí estaba en clusters recuperados:

- *Panic*, *Sweet Little Thing* y *Three, Two, One* en cluster 2;
- *Dangerous Secrets* y *Dirty* en cluster 8.

Muchos otros estaban fuera:

- mystery/thriller en cluster 13;
- romance/YA en cluster 12;
- otros modos distribuidos por el catálogo.

Clasificación:

- **E1:** gran parte del futuro quedó fuera de los cinco clusters;
- **E2:** algunos objetivos entraron al pool recuperado, pero no al top-10;
- **E6:** cuatro centroides y cinco clusters no representan suficientemente un historial tan
  amplio.

Este caso demuestra que el fallo no tiene una causa única. El presupuesto fijo de retrieval y la
competencia dentro de clusters interactúan.

## 7. Error específico de exploración

Los slots de interés tienen un hit rate de 4.02% en `k=10`; exploración tiene 0.056%. La regla
actual exige:

- salir de los macro-clusters ocupados;
- mantener 75% de la mejor similitud;
- usar solo tail/mid.

Hipótesis:

1. salir completamente de la macro-vecindad es demasiado agresivo;
2. un ratio relativo puede aceptar similitudes absolutas bajas;
3. priorizar tail antes que similitud puede perjudicar el orden;
4. dos slots fijos no se adaptan a confianza ni amplitud del usuario.

Pruebas propuestas:

- explorar en macro-clusters hermanos antes de saltar fuera;
- combinar piso relativo y absoluto;
- ordenar por una mezcla continua de relevancia y novelty;
- reducir exploración en perfiles dispersos o poco confiables;
- medir ablations con 0, 1 y 2 slots.

## 8. Error de identidad de edición

La exclusión se verifica por `book_id` y está cubierta por tests. El fallo aparece en otro nivel:
Goodreads puede asignar IDs distintos a ediciones de la misma obra.

Impacto:

- repetición percibida por el usuario;
- métricas offline potencialmente inconsistentes;
- diversidad artificialmente alta;
- consumo previo no reconocido a nivel obra.

Corrección:

```text
canonical_work_id =
    work_id oficial
    o ISBN normalizado
    o fingerprint(title_normalizado, autor_principal)
```

La exclusión y evaluación deberían operar por `canonical_work_id`, conservando `book_id` para
mostrar la edición concreta.

## 9. Matriz de causa y mejora

| Error | Evidencia observable | Mejora |
|---|---|---|
| E1 retrieval | objetivo fuera de clusters recuperados | ANN global, más clusters, retrieval por modo |
| E2 scoring/MMR | objetivo en cluster, fuera de top-k | ablations, learning-to-rank, calibrar MMR |
| E3 exploración | slot tail/mid sin hit | piso absoluto, macro-cluster hermano, slots adaptativos |
| E4 identidad | misma obra con otro `book_id` | canonicalización de obra |
| E5 etiqueta | recomendado no observado | telemetría de exposición y feedback |
| E6 perfil amplio | muchos modos futuros dispersos | más modos o retrieval proporcional a amplitud |

## 10. Qué debe reportarse en futuras evaluaciones

Además de métricas agregadas:

- porcentaje de objetivos fuera del retrieval;
- recall del candidate pool antes del ranking;
- porcentaje de objetivos recuperados pero no rankeados;
- hits por slot `interest/exploration`;
- errores por actividad y amplitud de usuario;
- duplicados de obra en recomendaciones;
- ejemplos reproducibles con historial, objetivos futuros, clusters y ranks.

La métrica prioritaria para separar retrieval de ranking es:

```text
candidate_recall =
    objetivos futuros presentes en candidatos recuperados
    / objetivos futuros elegibles
```

Sin esta métrica, un `Recall@k` bajo no permite saber si el problema ocurrió antes o después del
scoring.
