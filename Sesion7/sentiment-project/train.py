"""Entrenamiento/evaluación de sentimiento y logueo en MLflow.

Este script está preparado para ejecutarse desde MLflow Projects y registrar
un modelo `pyfunc` en cada run.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import pipeline


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos para el experimento.

    Returns:
        argparse.Namespace: Parámetros de ejecución del experimento.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=200)
    return parser.parse_args()


def _build_temp_file(name: str) -> Path:
    """Construye una ruta temporal portable para artefactos locales.

    Args:
        name: Nombre del archivo temporal.

    Returns:
        Path: Ruta dentro del directorio temporal del sistema.
    """
    return Path(tempfile.gettempdir()) / name


def main() -> None:
    """Ejecuta el pipeline de evaluación y registra métricas/modelo en MLflow."""
    args = parse_args()
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment("sentiment-analysis-hf")

    with mlflow.start_run():
        # Log de todos los parámetros en bloque para trazabilidad.
        mlflow.log_params(vars(args))

        # Cargar subset de datos de validación de SST-2.
        split = f"validation[:{args.num_samples}]"
        dataset = load_dataset("glue", "sst2", split=split)
        texts, labels = dataset["sentence"], dataset["label"]

        # Cargar modelo y medir latencia de inferencia por lote.
        clf = pipeline(
            "sentiment-analysis",
            model=args.model_name,
            truncation=True,
            max_length=args.max_length,
        )
        t0 = time.time()
        raw = clf(list(texts), batch_size=args.batch_size)
        latency = time.time() - t0

        # Filtrar predicciones con umbral de confianza.
        label_map = {"POSITIVE": 1, "NEGATIVE": 0}
        preds: list[int] = []
        true: list[int] = []
        for pred, label in zip(raw, labels):
            if pred["score"] >= args.confidence_threshold:
                preds.append(label_map[pred["label"]])
                true.append(label)

        if not preds:
            print("WARN: umbral demasiado alto, sin predicciones")
            return

        # Métricas principales del experimento.
        acc = accuracy_score(true, preds)
        f1 = f1_score(true, preds, average="weighted")
        coverage = len(preds) / len(labels)

        mlflow.log_metrics(
            {
                "accuracy": round(acc, 4),
                "f1_weighted": round(f1, 4),
                "coverage": round(coverage, 4),
                "avg_latency_ms": round((latency / len(texts)) * 1000, 2),
            }
        )

        # Guardar reporte de clasificación como artefacto.
        report = classification_report(
            true,
            preds,
            target_names=["Negativo", "Positivo"],
            zero_division=np.nan,
        )
        report_path = _build_temp_file("report.txt")
        report_path.write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(report_path))

        # Definir wrapper pyfunc para máxima compatibilidad de inferencia.
        class SentimentWrapper(mlflow.pyfunc.PythonModel):
            """Wrapper MLflow pyfunc para servir pipeline de Transformers."""

            def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
                """Carga config y pipeline al inicializar el modelo."""
                with open(context.artifacts["config"], encoding="utf-8") as f:
                    self.config = json.load(f)

                self.clf = pipeline(
                    "sentiment-analysis",
                    model=self.config["model_name"],
                    truncation=True,
                    max_length=self.config["max_length"],
                )

            def predict(
                self,
                context: mlflow.pyfunc.PythonModelContext,
                model_input: Any,
            ) -> list[int]:
                """Realiza inferencia sobre un DataFrame con columna `text`."""
                del context
                texts = model_input["text"].tolist()
                results = self.clf(texts)
                label_map = {"POSITIVE": 1, "NEGATIVE": 0}
                return [label_map[r["label"]] for r in results]

        # Guardar configuración como artefacto auxiliar del modelo.
        config_path = _build_temp_file("model_config.json")
        config_path.write_text(
            json.dumps(
                {
                    "model_name": args.model_name,
                    "max_length": args.max_length,
                }
            ),
            encoding="utf-8",
        )

        mlflow.pyfunc.log_model(
            artifact_path="sentiment_model",
            python_model=SentimentWrapper(),
            artifacts={"config": str(config_path)},
        )

        print(f"acc={acc:.4f} | f1={f1:.4f} | coverage={coverage:.2%}")


if __name__ == "__main__":
    main()
