"""Sesión 10: monitorización programática de drift en producción.

Script para ejecutar periódicamente (cron/GitHub Actions). Compara datos
actuales con referencia y decide si conviene reentrenar.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
import pandas as pd

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
except ImportError:  # Compatibilidad con versiones antiguas.
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

DRIFT_THRESHOLD = 0.20
ACCURACY_DROP_THRESHOLD = 0.05


def _safe_run_report(report: Any, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Any:
    """Ejecuta el informe Evidently soportando firmas viejas/nuevas."""
    try:
        evaluation = report.run(
            reference_data=reference_data, current_data=current_data)
    except TypeError:
        evaluation = report.run(reference_data, current_data)
    return evaluation if evaluation is not None else report


def _result_to_dict(evaluation: Any) -> dict[str, Any]:
    """Convierte el resultado de Evidently en diccionario Python."""
    if hasattr(evaluation, "dict"):
        return evaluation.dict()
    if hasattr(evaluation, "as_dict"):
        return evaluation.as_dict()
    if hasattr(evaluation, "json"):
        return json.loads(evaluation.json())
    raise RuntimeError("No se pudo convertir el resultado del informe a dict.")


def _find_key_recursively(obj: Any, key: str) -> Any | None:
    """Busca una clave recursivamente en estructuras dict/list."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_key_recursively(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_recursively(item, key)
            if found is not None:
                return found
    return None


def _extract_drift_share(results: dict[str, Any]) -> float:
    """Extrae `drift_share` desde la salida de Evidently.

    Compatible con salida moderna (`DriftedColumnsCount`) y formatos previos.

    Args:
        results: Resultado serializado del informe.

    Returns:
        Proporción de columnas con drift en rango [0, 1].
    """
    metrics = results.get("metrics", [])
    if isinstance(metrics, list):
        for metric in metrics:
            if not isinstance(metric, dict):
                continue

            metric_name = str(metric.get("metric_name", ""))
            if "DriftedColumnsCount" not in metric_name:
                continue

            value = metric.get("value", {})
            share = value.get("share")
            try:
                return float(share)
            except (TypeError, ValueError):
                break

    legacy_share = _find_key_recursively(results, "share_of_drifted_columns")
    try:
        return float(legacy_share)
    except (TypeError, ValueError):
        return 0.0


def estimate_drift_share(reference_df: pd.DataFrame, production_df: pd.DataFrame) -> float:
    """Calcula proporción de columnas con drift detectado por Evidently.

    Args:
        reference_df: DataFrame de referencia.
        production_df: DataFrame de producción actual.

    Returns:
        Proporción de columnas con drift en rango [0, 1].
    """
    report = Report(metrics=[DataDriftPreset()])
    evaluation = _safe_run_report(
        report=report,
        reference_data=reference_df,
        current_data=production_df,
    )

    results = _result_to_dict(evaluation)
    return _extract_drift_share(results)


def check_drift(reference_df: pd.DataFrame, production_df: pd.DataFrame) -> dict[str, Any]:
    """Evalúa alertas de drift y caída de accuracy.

    Args:
        reference_df: Datos de referencia con columna `correct`.
        production_df: Datos de producción con columna `correct`.

    Returns:
        Diccionario con alertas y recomendación de reentrenamiento.
    """
    drift_share = estimate_drift_share(reference_df, production_df)

    ref_accuracy = float(reference_df["correct"].mean())
    prod_accuracy = float(production_df["correct"].mean())
    accuracy_drop = ref_accuracy - prod_accuracy

    alerts: list[str] = []
    if drift_share > DRIFT_THRESHOLD:
        alerts.append(
            f"DATA DRIFT: {drift_share:.1%} de columnas con drift"
        )
    if accuracy_drop > ACCURACY_DROP_THRESHOLD:
        alerts.append(
            "ACCURACY DROP: "
            f"bajó {accuracy_drop:.1%} "
            f"(ref={ref_accuracy:.3f}, prod={prod_accuracy:.3f})"
        )

    return {
        "alerts": alerts,
        "drift_share": round(drift_share, 4),
        "accuracy_ref": round(ref_accuracy, 4),
        "accuracy_prod": round(prod_accuracy, 4),
        "accuracy_drop": round(accuracy_drop, 4),
        "needs_retraining": len(alerts) > 0,
    }


def build_simulated_data(size: int = 500, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera datos simulados para demostrar la lógica de alertas.

    Args:
        size: Número de filas por dataset.
        seed: Semilla para reproducibilidad.

    Returns:
        Tupla (referencia, producción) con drift y caída de rendimiento.
    """
    rng = np.random.default_rng(seed)

    reference_df = pd.DataFrame(
        {
            "text_length": rng.normal(80, 20, size),
            "word_count": rng.normal(15, 5, size),
            "correct": rng.binomial(1, 0.92, size),
        }
    )

    production_df = pd.DataFrame(
        {
            "text_length": rng.normal(120, 40, size),
            "word_count": rng.normal(22, 8, size),
            "correct": rng.binomial(1, 0.85, size),
        }
    )

    return reference_df, production_df


def main() -> int:
    """Ejecuta chequeo de drift y devuelve código de salida para CI/CD."""
    reference_df, production_df = build_simulated_data()
    result = check_drift(reference_df, production_df)

    print(json.dumps(result, indent=2))

    if result["needs_retraining"]:
        print("\nACCION REQUERIDA: Re-entrenamiento recomendado")
        for alert in result["alerts"]:
            print(f" - {alert}")
        return 1

    print("\nModelo en buen estado. No se requiere acción.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
