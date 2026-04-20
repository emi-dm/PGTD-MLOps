"""Sesión 10: generación de informe de drift con Evidently.

Este script compara datos de referencia vs datos de producción simulada,
extrae features textuales, añade predicciones de un modelo de sentimiento y
crea un informe HTML de drift.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import pipeline

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
except ImportError:  # Compatibilidad con versiones antiguas.
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

try:
    from evidently.metrics import (
        ColumnDriftMetric,
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
    )
except ImportError:  # Algunas versiones no exponen estas métricas.
    ColumnDriftMetric = None
    DatasetDriftMetric = None
    DatasetMissingValuesMetric = None

try:
    from dotenv import load_dotenv
except ImportError:  # Dependencia opcional.
    load_dotenv = None

warnings.filterwarnings("ignore")

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SAMPLE_SIZE = 300
REPORT_PATH = "drift_report.html"


def _load_optional_env_file() -> None:
    """Carga variables de entorno desde `.env` si la librería está presente."""
    if load_dotenv is not None:
        load_dotenv()


def load_reference_and_production(sample_size: int) -> tuple[Any, Any]:
    """Carga datasets de referencia y producción simulada para SST-2.

    Nota:
        El split `test` de GLUE puede no incluir labels utilizables para evaluar
        calidad. Si ocurre, se usa un tramo distinto de `validation`.

    Args:
        sample_size: Número de filas por dataset.

    Returns:
        Tupla (dataset_referencia, dataset_produccion).
    """
    reference = load_dataset(
        "glue", "sst2", split=f"validation[:{sample_size}]")

    production = load_dataset("glue", "sst2", split=f"test[:{sample_size}]")

    if "label" not in production.column_names:
        print(
            "Split test sin label; usando validation[300:600] como producción.")
        production = load_dataset(
            "glue",
            "sst2",
            split=f"validation[{sample_size}:{sample_size * 2}]",
        )
    else:
        labels = production["label"]
        if len(labels) == 0 or all(int(lbl) < 0 for lbl in labels):
            print("Labels no válidos en test; usando validation[300:600].")
            production = load_dataset(
                "glue",
                "sst2",
                split=f"validation[{sample_size}:{sample_size * 2}]",
            )

    return reference, production


def extract_features(dataset: Any) -> pd.DataFrame:
    """Extrae features textuales para análisis de drift.

    Args:
        dataset: Dataset de Hugging Face con columnas `sentence` y `label`.

    Returns:
        DataFrame con texto, etiqueta y features derivadas.
    """
    texts = dataset["sentence"]
    labels = dataset["label"]

    def avg_word_len(text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        return float(np.mean([len(word) for word in words]))

    return pd.DataFrame(
        {
            "text": texts,
            "label": labels,
            "text_length": [len(text) for text in texts],
            "word_count": [len(text.split()) for text in texts],
            "avg_word_len": [avg_word_len(text) for text in texts],
            "excl_count": [text.count("!") for text in texts],
            "question_count": [text.count("?") for text in texts],
            "upper_ratio": [
                sum(1 for char in text if char.isupper()) / max(len(text), 1)
                for text in texts
            ],
        }
    )


def add_predictions(df: pd.DataFrame, clf: Any) -> pd.DataFrame:
    """Añade predicción, confianza y corrección del modelo al DataFrame.

    Args:
        df: DataFrame de entrada con columna `text` y `label`.
        clf: Pipeline de clasificación de sentimiento.

    Returns:
        DataFrame con columnas nuevas: prediction, confidence, correct.
    """
    label_map = {"POSITIVE": 1, "NEGATIVE": 0}
    results = clf(df["text"].tolist(), batch_size=32)

    mapped_predictions: list[int] = []
    confidences: list[float] = []
    for result in results:
        raw_label = str(result["label"]).upper()
        if raw_label in label_map:
            mapped_predictions.append(label_map[raw_label])
        elif "POS" in raw_label:
            mapped_predictions.append(1)
        else:
            mapped_predictions.append(0)
        confidences.append(float(result["score"]))

    enriched = df.copy()
    enriched["prediction"] = mapped_predictions
    enriched["confidence"] = confidences
    enriched["correct"] = (enriched["prediction"] ==
                           enriched["label"]).astype(int)
    return enriched


def _safe_run_report(report: Any, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Any:
    """Ejecuta el informe Evidently soportando distintas firmas de API."""
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


def _extract_drift_summary(results: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Extrae resumen de drift soportando múltiples formatos de Evidently.

    Args:
        results: Resultado serializado del informe.

    Returns:
        Tupla (drift_columns, drift_share, dataset_drifted).
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
            config = metric.get("config", {})
            drift_columns = value.get("count")
            drift_share = value.get("share")

            dataset_drifted: Any = None
            threshold = config.get("drift_share")
            try:
                if threshold is not None and drift_share is not None:
                    dataset_drifted = float(drift_share) >= float(threshold)
            except (TypeError, ValueError):
                dataset_drifted = None

            return drift_columns, drift_share, dataset_drifted

    # Fallback para formatos previos.
    drift_columns = _find_key_recursively(results, "number_of_drifted_columns")
    drift_share = _find_key_recursively(results, "share_of_drifted_columns")
    dataset_drifted = _find_key_recursively(results, "dataset_drift")
    return drift_columns, drift_share, dataset_drifted


def build_drift_report() -> Any:
    """Construye el reporte Evidently con métricas disponibles en la versión."""
    metrics: list[Any] = []

    if DatasetDriftMetric is not None:
        metrics.append(DatasetDriftMetric())
    if DatasetMissingValuesMetric is not None:
        metrics.append(DatasetMissingValuesMetric())
    if ColumnDriftMetric is not None:
        metrics.extend(
            [
                ColumnDriftMetric(column_name="text_length"),
                ColumnDriftMetric(column_name="word_count"),
                ColumnDriftMetric(column_name="confidence"),
                ColumnDriftMetric(column_name="correct"),
            ]
        )

    metrics.append(DataDriftPreset())
    return Report(metrics=metrics)


def main() -> None:
    """Ejecuta el flujo completo de análisis de drift y guarda el HTML."""
    _load_optional_env_file()

    print("Cargando datos...")
    reference_dataset, production_dataset = load_reference_and_production(
        sample_size=SAMPLE_SIZE
    )

    print("Extrayendo features...")
    reference_df = extract_features(reference_dataset)
    production_df = extract_features(production_dataset)

    print("Generando predicciones del modelo...")
    clf = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        truncation=True,
        max_length=128,
    )

    reference_df = add_predictions(reference_df, clf)
    production_df = add_predictions(production_df, clf)

    ref_acc = float(reference_df["correct"].mean())
    prod_acc = float(production_df["correct"].mean())
    print(f"Referencia → accuracy: {ref_acc:.3f}")
    print(f"Producción → accuracy: {prod_acc:.3f}")

    print("Generando informe de drift...")
    report = build_drift_report()
    evaluation = _safe_run_report(
        report=report,
        reference_data=reference_df.drop(columns=["text"]),
        current_data=production_df.drop(columns=["text"]),
    )

    if hasattr(evaluation, "save_html"):
        evaluation.save_html(REPORT_PATH)
    elif hasattr(report, "save_html"):
        report.save_html(REPORT_PATH)
    else:
        raise RuntimeError(
            "La versión de Evidently no soporta save_html esperado.")

    print(f"Informe guardado en {REPORT_PATH}")

    results = _result_to_dict(evaluation)
    drift_columns, drift_share, dataset_drifted = _extract_drift_summary(
        results)

    print("\nResumen de drift:")
    print(
        f"Columnas con drift: {drift_columns if drift_columns is not None else 'N/A'}")
    if drift_share is None:
        print("Proporción de drift: N/A")
    else:
        try:
            print(f"Proporción de drift: {float(drift_share):.1%}")
        except (TypeError, ValueError):
            print(f"Proporción de drift: {drift_share}")
    print(
        "Dataset drifted: "
        f"{dataset_drifted if dataset_drifted is not None else 'N/A'}"
    )


if __name__ == "__main__":
    main()
