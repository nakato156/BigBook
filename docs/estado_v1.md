# Estado de BigBook V1

Generado desde los artefactos y métricas locales. Estado académico: **NO VALIDADA**.

## Veredicto

V1 no validada: el modelo no supera B1 en Recall y NDCG para k=5, 10, 20.

Este veredicto cubre evidencia N0. N1 es descriptivo/correlacional y N2 requiere telemetría
de producto; ninguno de los dos se interpreta como efecto causal del recomendador.

## Artefactos validados

- Libros master/PCA/clusters alineados: 108,227.
- Interacciones globales deduplicadas: 110,450,288.
- Usuarios globales: 821,387; usuarios con vector: 699,381.
- Centroides de gusto: 2,356,255.
- Cohorte seleccionada/evaluable/descartada: 5,000/2,038/2,962.
- Corte temporal global: `2016-06-09T19:30:05.200000+00:00`.
- Segmentos de actividad previa: high=1,094, low=305, mid=639.

## Resultados N0

| system | k | users | recall | ndcg | map | diversity | catalog_coverage | long_tail_coverage | novelty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model | 5 | 2038 | 0.002733 | 0.006181 | 0.003633 | 0.752907 | 0.039358 | 0.039614 | 16.230789 |
| B0_random | 5 | 2038 | 0.000061 | 0.000083 | 0.000033 | 0.999955 | 0.093572 | 0.094899 | 18.536719 |
| B1_popularity | 5 | 2038 | 0.015553 | 0.037230 | 0.028593 | 0.553450 | 0.000328 | 0.000000 | 8.243777 |
| B2_genre_popularity | 5 | 2038 | 0.014603 | 0.035294 | 0.026818 | 0.549305 | 0.000482 | 0.000000 | 8.375239 |
| model | 10 | 2038 | 0.004745 | 0.006773 | 0.003048 | 0.767539 | 0.081320 | 0.059631 | 15.738482 |
| B0_random | 10 | 2038 | 0.000110 | 0.000132 | 0.000034 | 0.999336 | 0.178519 | 0.179884 | 18.548488 |
| B1_popularity | 10 | 2038 | 0.025292 | 0.034830 | 0.022126 | 0.675652 | 0.000492 | 0.000000 | 8.459652 |
| B2_genre_popularity | 10 | 2038 | 0.024151 | 0.032898 | 0.020028 | 0.680366 | 0.000791 | 0.000000 | 8.654455 |
| model | 20 | 2038 | 0.011873 | 0.009256 | 0.003191 | 0.731984 | 0.144795 | 0.093679 | 15.626177 |
| B0_random | 20 | 2038 | 0.000131 | 0.000149 | 0.000032 | 0.999471 | 0.325185 | 0.330677 | 18.552095 |
| B1_popularity | 20 | 2038 | 0.036133 | 0.034181 | 0.018162 | 0.776237 | 0.000723 | 0.000000 | 8.926029 |
| B2_genre_popularity | 20 | 2038 | 0.034619 | 0.032305 | 0.016578 | 0.780251 | 0.001215 | 0.000000 | 9.153246 |

## Evidencia N1

`temporal_evaluation_by_activity.csv` compara proxies futuros de hábito por nivel de actividad
previa (`low`, `mid`, `high`). Es una descripción de usuarios observados, no evidencia de que las
recomendaciones hayan causado cambios en frecuencia, finalización, recencia o amplitud de lectura.

## Límites de V1

Quedan fuera de la V1 académica: API/UI, telemetría N2, experimentos A/B, backtest que reconstruya
PCA y clustering por corte, item cold-start y bandits o exploración adaptativa.
