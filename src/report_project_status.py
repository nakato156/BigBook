"""Generate the academic V1 closure report from validated artifacts and metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.validate_artifacts import validate_artifacts

EVALUATION_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "recommendations" / "temporal_evaluation.csv"
)
EVALUATION_V12_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "recommendations" / "temporal_evaluation_v12.csv"
)
RANKER_WINNER_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "recommendations" / "ranker_grid_winner.json"
)
ACTIVITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "recommendations"
    / "temporal_evaluation_by_activity.csv"
)
REPORT_PATH = PROJECT_ROOT / "docs" / "estado_v1.md"


def _model_system(summary: pd.DataFrame) -> str:
    systems = set(summary["system"].astype(str))
    for candidate in ("hybrid_v12", "model", "content_only"):
        if candidate in systems:
            return candidate
    raise ValueError("No model-like system found in evaluation summary.")


def validation_verdict(summary: pd.DataFrame) -> tuple[bool, str]:
    """Require the model to beat B1 recall and NDCG at every reported k."""
    model_system = _model_system(summary)
    model_name = "el modelo" if model_system == "model" else model_system
    pivot = summary.pivot(index="k", columns="system", values=["recall", "ndcg"])
    required = {
        ("recall", model_system),
        ("recall", "B1_popularity"),
        ("ndcg", model_system),
        ("ndcg", "B1_popularity"),
    }
    if not required.issubset(set(pivot.columns)):
        return False, f"V1 no validada: faltan filas de {model_name} o de B1."
    wins = (
        (pivot[("recall", model_system)] > pivot[("recall", "B1_popularity")])
        & (pivot[("ndcg", model_system)] > pivot[("ndcg", "B1_popularity")])
    )
    if bool(wins.all()):
        return True, f"V1 validada: {model_name} supera B1 en Recall y NDCG para todos los k."
    failed = ", ".join(str(k) for k in wins.index[~wins])
    return False, f"V1 no validada: {model_name} no supera B1 en Recall y NDCG para k={failed}."


def _metric_table(summary: pd.DataFrame) -> str:
    columns = [
        "system",
        "k",
        "users",
        "recall",
        "ndcg",
        "map",
        "candidate_recall",
        "diversity",
        "catalog_coverage",
        "long_tail_coverage",
        "novelty",
    ]
    columns = [column for column in columns if column in summary.columns]
    table = summary[columns].copy()
    for column in columns[3:]:
        table[column] = table[column].map(lambda value: f"{value:.6f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def build_report(
    evaluation_path: Path = EVALUATION_PATH,
    activity_path: Path = ACTIVITY_PATH,
) -> str:
    counts = validate_artifacts()
    winner_payload: dict = {}
    if RANKER_WINNER_PATH.exists():
        winner_payload = json.loads(RANKER_WINNER_PATH.read_text(encoding="utf-8"))
    if (
        evaluation_path == EVALUATION_PATH
        and EVALUATION_V12_PATH.exists()
        and int(winner_payload.get("max_users", 0)) >= 1000
    ):
        evaluation_path = EVALUATION_V12_PATH
    if not evaluation_path.exists() or not activity_path.exists():
        raise FileNotFoundError("Run src.reduction.evaluate_recommender before building the report.")
    summary = pd.read_csv(evaluation_path)
    activity = pd.read_csv(activity_path)
    validated, verdict = validation_verdict(summary)
    if winner_payload and int(winner_payload.get("max_users", 0)) >= 1000:
        winner_text = (
            f"- Configuracion seleccionada V1.2: `{winner_payload.get('winner')}`; "
            f"validada contra B1: `{winner_payload.get('validated_against_b1')}`.\n"
        )
    elif winner_payload:
        winner_text = (
            f"- Smoke V1.2 disponible (`max_users={winner_payload.get('max_users')}`), "
            "pero el veredicto usa la evaluacion temporal base hasta correr la cohorte final.\n"
        )
    else:
        winner_text = "- Configuracion seleccionada V1.2: no disponible; se usa la evaluacion temporal base.\n"
    selected = int(summary["users_selected"].max())
    evaluable = int(summary["users_evaluable"].max())
    discarded = int(summary["users_discarded"].max())
    cutoff = str(summary["temporal_cutoff"].iloc[0])

    segments = (
        activity.groupby("activity_segment", sort=True)["users"]
        .max()
        .astype(int)
        .to_dict()
    )
    segment_text = ", ".join(f"{key}={value:,}" for key, value in segments.items())
    status = "VALIDADA" if validated else "NO VALIDADA"
    return f"""# Estado de BigBook V1

Generado desde los artefactos y métricas locales. Estado académico: **{status}**.

## Veredicto

{verdict}

{winner_text}

Este veredicto cubre evidencia N0. N1 es descriptivo/correlacional y N2 requiere telemetría
de producto; ninguno de los dos se interpreta como efecto causal del recomendador.

## Artefactos validados

- Libros master/PCA/clusters alineados: {counts["books"]:,}.
- Interacciones globales deduplicadas: {counts["interactions"]:,}.
- Usuarios globales: {counts["global_users"]:,}; usuarios con vector: {counts["user_matrix"]:,}.
- Centroides de gusto: {counts["user_centroids"]:,}.
- Cohorte seleccionada/evaluable/descartada: {selected:,}/{evaluable:,}/{discarded:,}.
- Corte temporal global: `{cutoff}`.
- Segmentos de actividad previa: {segment_text}.

## Resultados N0

{_metric_table(summary)}

## Evidencia N1

`temporal_evaluation_by_activity.csv` compara proxies futuros de hábito por nivel de actividad
previa (`low`, `mid`, `high`). Es una descripción de usuarios observados, no evidencia de que las
recomendaciones hayan causado cambios en frecuencia, finalización, recencia o amplitud de lectura.

## Límites de V1

Quedan fuera de la V1 académica: API/UI, telemetría N2, experimentos A/B online, item cold-start
y bandits o exploración adaptativa. V1.1 incorpora como jobs manuales la comparación colaborativa
offline, bootstrap y backtest con refit por corte.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote V1 status report to {args.output}")


if __name__ == "__main__":
    main()
