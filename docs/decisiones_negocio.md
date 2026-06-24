# Decisiones de negocio y criterio de validación

Este documento fija la decisión metodológica posterior a la evaluación V1.2. La conclusión no es
que las métricas estén "mal", sino que miden una parte del problema: `Recall@k` y `NDCG@k` sobre
Goodreads predicen lecturas observadas bajo exposición histórica. Eso es valioso como compuerta de
relevancia, pero no agota el objetivo de BigBook: ayudar a sostener el hábito lector mediante
recomendaciones relevantes, accesibles y con descubrimiento.

## Decisión 1: B1 es baseline de exposición, no north-star

`B1_popularity` sigue siendo obligatorio porque representa la estrategia trivial: recomendar a
todos los libros más populares disponibles en el snapshot histórico. Si un modelo no se acerca a
B1 en relevancia, debe explicar por qué pierde aciertos observados.

Pero B1 no debe convertirse en la métrica norte del producto. En un dataset observacional como
Goodreads, los libros populares aparecen más en los historiales porque tuvieron más exposición
fuera del sistema. Por eso B1 mide muy bien "qué se leyó mucho", pero no mide bien "qué ayuda a un
lector específico a descubrir una próxima lectura motivadora".

Implicación:

- B1 se conserva como control de cordura y sesgo de exposición.
- Ganar a B1 en `Recall@k`/`NDCG@k` es evidencia fuerte, pero no suficiente si el ranking colapsa a
  bestsellers.
- Perder contra B1 en relevancia pura no invalida automáticamente el objetivo de descubrimiento,
  pero sí impide reclamar superioridad predictiva N0 sin matices.

## Decisión 2: el éxito de negocio exige relevancia y descubrimiento

BigBook no optimiza "popularidad promedio recomendada". Optimiza lecturas que el usuario pueda
empezar y completar, con suficiente afinidad para ser relevantes y suficiente diversidad para no
encerrar al lector en una lista de bestsellers.

El criterio operativo queda así:

1. **Compuerta de relevancia N0:** el modelo debe superar claramente a B0 y mantener una brecha
   defendible frente a B1/B2 en `Recall@k`, `NDCG@k` y `MAP`. Si no supera B1, el informe debe
   decirlo explícitamente.
2. **Utilidad de descubrimiento:** el modelo debe mejorar frente a B1 en `Coverage`,
   `Long-tail Coverage`, `Novelty`, menor `head_share`, diversidad y baja repetición de obra o
   edición.
3. **Alineamiento con hábito N1:** los resultados deben interpretarse junto a proxies futuros de
   hábito (`completion_rate`, `reading_frequency`, `activity_recency`, `reading_breadth`,
   `active_span_days`). En offline esto es correlacional, no causal.
4. **Causalidad N2:** la afirmación "el recomendador aumenta el hábito lector" requiere producto
   vivo, telemetría y A/B test. Queda fuera del dataset estático.

## Decisión 3: V1.2 no debe copiar B1 para ganar la tabla

La ruta `hybrid_v12` puede usar popularidad histórica como señal calibrada, porque en la práctica
un recomendador útil no ignora por completo la validación social ni la disponibilidad cultural de
un libro. La restricción de negocio es que esa señal no domine el ranking ni destruya el
descubrimiento.

Por eso la popularidad se acepta solo bajo estas reglas:

- se calcula con train/snapshot histórico, nunca con interacciones futuras;
- entra como percentil comparable a otras señales, no como magnitud cruda;
- se audita con `head_share`, `tail_share`, `Novelty` y `Long-tail Coverage`;
- si mejora `Recall@k` pero reduce la lista a head items, el sistema no se declara alineado con el
  negocio.

## Decisión 4: métricas nuevas recomendadas

Para evaluar mejor el caso BigBook, el reporte debe separar métricas de predicción observada y
métricas de descubrimiento:

| Familia | Métricas | Qué responde |
|---|---|---|
| Relevancia N0 | `Recall@k`, `NDCG@k`, `MAP`, `Precision@k` | ¿Acierta lecturas futuras observadas? |
| Descubrimiento | `DiscoveryRecall@k` sobre futuros no-head, `TailNDCG@k`, novelty, coverage, long-tail coverage | ¿Acierta sin depender solo de bestsellers? |
| Exposición | `head_share`, `mid_share`, `tail_share`, average recommendation popularity | ¿Qué parte de la lista amplifica popularidad? |
| Calidad de lista | diversidad intra-lista, duplicate-title/work rate, accesibilidad | ¿La lista es usable y no redundante? |
| Hábito N1 | `completion_rate`, `reading_frequency`, `activity_recency`, `reading_breadth`, `active_span_days` | ¿Cómo se comportan los lectores en el futuro observado? |

La métrica compuesta, si se usa, debe presentarse como `HabitAlignedUtility@k` y no como causalidad:

```text
HabitAlignedUtility@k =
  relevancia observada
  + descubrimiento no-head
  + diversidad/accesibilidad
  - penalización por head_share excesivo
  - penalización por duplicados de obra/edición
```

Los pesos exactos son una decisión de producto y deben declararse antes de mirar el resultado final.

## Decisión 5: cómo se comunica el veredicto actual

El veredicto actual de V1.2 es honesto:

- no está validado como ranker superior a B1 en relevancia N0;
- sí muestra valor como sistema de descubrimiento frente a B1 por cobertura, novedad y exposición
  de cola;
- el siguiente paso no es maquillar el informe, sino reportar ambos ejes: predicción observada y
  utilidad de descubrimiento/hábito.

La frase recomendada para el entregable es:

> BigBook V1.2 no supera todavía a la popularidad global como predictor puro de próximas lecturas
> observadas, pero el criterio de negocio no es copiar la exposición histórica. El sistema debe
> evaluarse como un recomendador orientado a hábito lector: relevancia suficiente, descubrimiento
> medible, menor dependencia de head items y proxies de hábito reportados sin afirmar causalidad.
