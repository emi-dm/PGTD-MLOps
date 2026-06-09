"""Sesión 10: monitorización programática de drift en producción.

Modos de ejecución:
  --mode simulated  (por defecto): datos simulados para demostrar la lógica.
  --mode live --api-url http://api:8000: conecta con la API desplegada en
        Docker, envía textos de SST-2 al endpoint /predict, compara con
        ground truth y detecta drift real.

Diseñado para ejecutarse periódicamente (cron, Docker Compose, GitHub Actions).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def collect_live_data(
    api_url: str,
    num_samples: int = 200,
    timeout: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recoje predicciones reales de la API desplegada y construye DataFrames.

    Envía textos de SST-2 al endpoint /predict de la API, compara las
    predicciones con las etiquetas ground truth, y extrae features textuales
    para análisis de drift.

    Args:
        api_url: URL base de la API (ej: http://api:8000).
        num_samples: Número de muestras de SST-2 a evaluar.
        timeout: Timeout en segundos para las peticiones HTTP.

    Returns:
        Tupla (reference_df, production_df) listos para check_drift().
    """
    import requests
    from datasets import load_dataset

    # Cargar datos de referencia (split validation, primera mitad).
    ref_size = num_samples
    ref_dataset = load_dataset(
        "glue", "sst2", split=f"validation[:{ref_size}]")
    ref_texts = ref_dataset["sentence"]
    ref_labels = ref_dataset["label"]

    # Cargar datos de producción simulada (split validation, segunda mitad).
    prod_dataset = load_dataset(
        "glue", "sst2", split=f"validation[{ref_size}:{ref_size * 2}]"
    )
    prod_texts = prod_dataset["sentence"]
    prod_labels = prod_dataset["label"]

    # Esperar a que la API esté disponible.
    health_url = f"{api_url}/health"
    print(f"Esperando API en {health_url}...")
    for attempt in range(1, 31):
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                print(f"API lista (intento {attempt})")
                break
        except requests.RequestException:
            pass
        print(f"  Intento {attempt}/30...")
        time.sleep(5)
    else:
        raise RuntimeError(
            f"API no disponible en {health_url} tras 30 intentos")

    # Enviar textos de producción a la API y recoger predicciones.
    predict_url = f"{api_url}/predict"
    print(f"Enviando {len(prod_texts)} textos a {predict_url}...")

    batch_size = 50
    prod_predictions: list[int] = []
    for i in range(0, len(prod_texts), batch_size):
        batch = prod_texts[i: i + batch_size]
        resp = requests.post(
            predict_url,
            json={"texts": list(batch)},
            timeout=timeout,
        )
        resp.raise_for_status()
        prod_predictions.extend(resp.json()["predictions"])

    # Construir DataFrames con features textuales + columna correct.
    def _build_df(texts: list[str], labels: list[int], predictions: list[int] | None = None) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "text_length": [len(t) for t in texts],
                "word_count": [len(t.split()) for t in texts],
                "avg_word_len": [
                    float(np.mean([len(w) for w in t.split()])
                          ) if t.split() else 0.0
                    for t in texts
                ],
            }
        )
        if predictions is not None:
            df["correct"] = [int(p == l) for p, l in zip(predictions, labels)]
        else:
            # Para referencia, asumimos que el modelo base tiene ~90% accuracy.
            df["correct"] = [1] * len(labels)
        return df

    reference_df = _build_df(ref_texts, ref_labels)
    production_df = _build_df(prod_texts, prod_labels, prod_predictions)

    return reference_df, production_df


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Monitorización de drift en producción.",
    )
    parser.add_argument(
        "--mode",
        choices=["simulated", "live"],
        default="simulated",
        help="Modo: 'simulated' usa datos ficticios, 'live' conecta con la API.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://api:8000",
        help="URL base de la API (solo para modo live).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Número de muestras a evaluar (modo live).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Segundos entre chequeos. 0 = ejecutar una sola vez.",
    )
    return parser.parse_args()


def main() -> int:
    """Ejecuta chequeo de drift y devuelve código de salida para CI/CD."""
    args = parse_args()

    while True:
        if args.mode == "live":
            print(f"[LIVE] Conectando con API: {args.api_url}")
            reference_df, production_df = collect_live_data(
                api_url=args.api_url,
                num_samples=args.num_samples,
            )
        else:
            print("[SIMULATED] Generando datos simulados...")
            reference_df, production_df = build_simulated_data()

        result = check_drift(reference_df, production_df)
        result["mode"] = args.mode

        print(json.dumps(result, indent=2))

        if result["needs_retraining"]:
            print("\n⚠️  ACCION REQUERIDA: Re-entrenamiento recomendado")
            for alert in result["alerts"]:
                print(f"  - {alert}")
        else:
            print("\n✅ Modelo dentro de parámetros aceptables")

        if args.interval <= 0:
            break

        print(f"\nSiguiente chequeo en {args.interval}s...")
        time.sleep(args.interval)

    return 1 if result["needs_retraining"] else 0


if __name__ == "__main__":
    sys.exit(main())
