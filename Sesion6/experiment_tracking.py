"""Sesión 6: seguimiento de experimentos con MLflow Tracking.

Este script evalúa configuraciones de inferencia para clasificación de
sentimientos con un modelo de Hugging Face y registra en MLflow:
- Parámetros
- Métricas
- Artefactos (reporte y muestras de predicción)

Uso típico:
1) Levantar UI de MLflow en otra terminal: `mlflow ui --port 5000`
2) Ejecutar este script: `python experiment_tracking.py`
3) Abrir http://localhost:5000 y comparar runs.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import pipeline

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependencia opcional.
    load_dotenv = None

EXPERIMENT_NAME = "sentiment-analysis-hf"

# Configuraciones a comparar (según guía de la sesión).
CONFIGS: list[dict[str, Any]] = [
    {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "batch_size": 16,
        "confidence_threshold": 0.5,
        "truncation": True,
        "max_length": 128,
    },
    {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "batch_size": 32,
        "confidence_threshold": 0.7,
        "truncation": True,
        "max_length": 128,
    },
    {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "batch_size": 16,
        "confidence_threshold": 0.9,
        "truncation": True,
        "max_length": 64,
    },
]


def _load_optional_env_file() -> None:
    """Carga variables desde `.env` si `python-dotenv` está disponible."""
    if load_dotenv is None:
        return

    load_dotenv()


def _tmp_file(name: str) -> Path:
    """Construye una ruta temporal portable para artefactos locales.

    Args:
        name: Nombre de archivo a crear bajo el directorio temporal.

    Returns:
        Path con ruta final.
    """
    return Path(tempfile.gettempdir()) / name


def _map_model_label(raw_label: str) -> int | None:
    """Mapea etiquetas de modelos HF a binario (0 negativo, 1 positivo).

    Nota:
        Se incluyen variantes comunes para soportar modelos alternativos
        usados en trabajo autónomo.

    Args:
        raw_label: Etiqueta textual devuelta por el pipeline.

    Returns:
        0 o 1 si se puede mapear, o None si no es compatible.
    """
    label = raw_label.strip().upper()

    direct_map = {
        "NEGATIVE": 0,
        "POSITIVE": 1,
        "LABEL_0": 0,
        "LABEL_2": 1,
    }
    if label in direct_map:
        return direct_map[label]

    # Algunos modelos devuelven etiquetas más verbosas.
    if "NEG" in label:
        return 0
    if "POS" in label:
        return 1

    return None


def run_experiment(
    config: dict[str, Any],
    texts: list[str],
    labels: list[int],
) -> float | None:
    """Ejecuta una configuración y registra resultados en MLflow.

    Args:
        config: Configuración de inferencia.
        texts: Frases a evaluar.
        labels: Etiquetas reales binarias (SST-2).

    Returns:
        Accuracy de la run si hubo predicciones válidas, o None.
    """
    run_name = (
        "thr="
        f"{config['confidence_threshold']}_bs={config['batch_size']}_"
        f"ml={config['max_length']}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_name": config["model_name"],
                "batch_size": config["batch_size"],
                "confidence_threshold": config["confidence_threshold"],
                "max_length": config["max_length"],
                "dataset": f"sst2-validation-{len(texts)}",
                "truncation": config["truncation"],
            }
        )

        print(f"Cargando modelo {config['model_name']}...")
        clf = pipeline(
            "sentiment-analysis",
            model=config["model_name"],
            truncation=config["truncation"],
            max_length=config["max_length"],
        )

        start = time.time()
        raw_preds = clf(list(texts), batch_size=config["batch_size"])
        latency = time.time() - start

        preds_filtered: list[int] = []
        labels_filtered: list[int] = []
        uncertain_count = 0
        dropped_by_label_map = 0

        prediction_rows: list[dict[str, Any]] = []

        for text, true_label, raw_pred in zip(texts, labels, raw_preds):
            score = float(raw_pred["score"])
            mapped = _map_model_label(str(raw_pred["label"]))
            accepted = (
                mapped is not None
                and score >= float(config["confidence_threshold"])
            )

            if accepted:
                preds_filtered.append(mapped)
                labels_filtered.append(int(true_label))
            elif mapped is None:
                dropped_by_label_map += 1
                uncertain_count += 1
            else:
                uncertain_count += 1

            prediction_rows.append(
                {
                    "text": text,
                    "true": int(true_label),
                    "raw_label": str(raw_pred["label"]),
                    "pred": mapped,
                    "score": round(score, 4),
                    "accepted": accepted,
                }
            )

        if len(preds_filtered) == 0:
            mlflow.log_metrics(
                {
                    "accuracy": 0.0,
                    "f1_weighted": 0.0,
                    "coverage": 0.0,
                    "uncertain_samples": uncertain_count,
                    "dropped_by_label_map": dropped_by_label_map,
                    "avg_latency_ms": round((latency / len(texts)) * 1000, 2),
                    "total_latency_s": round(latency, 2),
                }
            )
            print("WARN: umbral demasiado alto o etiquetas incompatibles.")
            return None

        acc = accuracy_score(labels_filtered, preds_filtered)
        f1 = f1_score(labels_filtered, preds_filtered, average="weighted")
        coverage = len(preds_filtered) / len(labels)
        avg_latency_ms = (latency / len(texts)) * 1000

        mlflow.log_metrics(
            {
                "accuracy": round(float(acc), 4),
                "f1_weighted": round(float(f1), 4),
                "coverage": round(float(coverage), 4),
                "uncertain_samples": uncertain_count,
                "dropped_by_label_map": dropped_by_label_map,
                "avg_latency_ms": round(float(avg_latency_ms), 2),
                "total_latency_s": round(float(latency), 2),
            }
        )

        report = classification_report(
            labels_filtered,
            preds_filtered,
            target_names=["Negativo", "Positivo"],
            zero_division=0,
        )

        report_path = _tmp_file("classification_report_sesion6.txt")
        report_path.write_text(
            f"Config: {json.dumps(config, indent=2)}\n\n{report}",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(report_path))

        preds_path = _tmp_file("predictions_sesion6.json")
        preds_path.write_text(
            json.dumps(prediction_rows[:20], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(preds_path))

        print(
            f"acc={acc:.4f} | f1={f1:.4f} | "
            f"coverage={coverage:.2%} | latencia={avg_latency_ms:.1f}ms"
        )
        return float(acc)


def print_best_run() -> None:
    """Imprime la mejor run por accuracy del experimento actual."""
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print("No se encontró el experimento para buscar la mejor run.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )
    if not runs:
        print("No hay runs registradas en el experimento aún.")
        return

    best_run = runs[0]
    print("\nMejor run detectada:")
    print(f"run_id={best_run.info.run_id}")
    print(f"accuracy={best_run.data.metrics.get('accuracy', 'N/A')}")
    print(f"params={best_run.data.params}")


def main() -> None:
    """Orquesta la evaluación de todas las configuraciones de la sesión."""
    _load_optional_env_file()
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Cargando dataset SST-2...")
    dataset = load_dataset("glue", "sst2", split="validation[:200]")
    texts = dataset["sentence"]
    labels = dataset["label"]

    print(f"Lanzando {len(CONFIGS)} experimentos...")
    for i, cfg in enumerate(CONFIGS, start=1):
        print(
            f"\n[{i}/{len(CONFIGS)}] "
            f"threshold={cfg['confidence_threshold']} "
            f"batch={cfg['batch_size']}"
        )
        run_experiment(cfg, texts=texts, labels=labels)

    print("\nExperimentos completados. Abre http://localhost:5000 para comparar.")
    print_best_run()


if __name__ == "__main__":
    main()
