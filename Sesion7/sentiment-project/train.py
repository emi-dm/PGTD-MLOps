"""Fine-tuning de DistilBERT para análisis de sentimientos con MLflow.

Este script realiza un fine-tuning real del modelo base DistilBERT sobre el
dataset SST-2, evalúa en un split de validación y registra todo en MLflow:
parámetros, métricas, artefactos y modelo pyfunc.

Preparado para ejecutarse desde MLflow Projects.
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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

# Nombre del experimento MLflow (compartido con sesión 6 y 8).
EXPERIMENT_NAME = "sentiment-analysis-hf"


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos para el experimento.

    Returns:
        argparse.Namespace: Parámetros de ejecución del experimento.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT en SST-2 y registrar en MLflow.",
    )
    # Modelo base (sin fine-tuning previo) para entrenar desde cero.
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Modelo base de HuggingFace (sin fine-tuning).",
    )
    # Hiperparámetros de entrenamiento.
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # Datos.
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=500,
        help="Número de muestras de entrenamiento.",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=200,
        help="Número de muestras de validación.",
    )
    # Evaluación.
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    return parser.parse_args()


def _build_temp_file(name: str) -> Path:
    """Construye una ruta temporal portable para artefactos locales.

    Args:
        name: Nombre del archivo temporal.

    Returns:
        Path: Ruta dentro del directorio temporal del sistema.
    """
    return Path(tempfile.gettempdir()) / name


def _tokenize(
    examples: dict,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> dict:
    """Tokeniza ejemplos para el modelo.

    Args:
        examples: Batch de ejemplos del dataset.
        tokenizer: Tokenizador de HuggingFace.
        max_length: Longitud máxima de tokens.

    Returns:
        Diccionario con input_ids y attention_mask.
    """
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def main() -> None:
    """Ejecuta fine-tuning, evaluación y registro en MLflow."""
    args = parse_args()

    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # ── 1. Cargar datos ───────────────────────────────────────────
        print(f"Cargando SST-2: train={args.num_train_samples}, "
              f"eval={args.num_eval_samples}")
        train_ds = load_dataset(
            "glue", "sst2", split=f"train[:{args.num_train_samples}]"
        )
        eval_ds = load_dataset(
            "glue", "sst2", split=f"validation[:{args.num_eval_samples}]"
        )

        # ── 2. Tokenizar ──────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        train_ds = train_ds.map(
            lambda ex: _tokenize(ex, tokenizer, args.max_length),
            batched=True,
        )
        eval_ds = eval_ds.map(
            lambda ex: _tokenize(ex, tokenizer, args.max_length),
            batched=True,
        )
        train_ds.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"])
        eval_ds.set_format("torch", columns=[
                           "input_ids", "attention_mask", "label"])

        # ── 3. Cargar modelo base y fine-tune ─────────────────────────
        print(f"Fine-tuning {args.model_name} durante {args.epochs} epochs...")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
        )

        output_dir = str(_build_temp_file("hf_trainer_output"))
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_steps=10,
            report_to="none",  # MLflow se gestiona manualmente.
            disable_tqdm=False,
        )

        def compute_metrics(eval_pred: tuple) -> dict[str, float]:
            """Calcula accuracy para el Trainer."""
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"accuracy": accuracy_score(labels, preds)}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )

        t0 = time.time()
        trainer.train()
        train_time = time.time() - t0

        # ── 4. Evaluación final ───────────────────────────────────────
        print("Evaluando modelo fine-tuneado...")
        eval_results = trainer.evaluate()
        train_loss = (
            trainer.state.log_history[-2].get("eval_loss", 0.0)
            if len(trainer.state.log_history) >= 2
            else 0.0
        )

        # Inferencia con pipeline para métricas detalladas.
        fine_tuned_path = str(_build_temp_file("fine_tuned_model"))
        trainer.save_model(fine_tuned_path)
        tokenizer.save_pretrained(fine_tuned_path)

        clf = pipeline(
            "sentiment-analysis",
            model=fine_tuned_path,
            truncation=True,
            max_length=args.max_length,
        )

        eval_texts = load_dataset(
            "glue", "sst2", split=f"validation[:{args.num_eval_samples}]"
        )["sentence"]
        eval_labels = load_dataset(
            "glue", "sst2", split=f"validation[:{args.num_eval_samples}]"
        )["label"]

        t0 = time.time()
        raw = clf(list(eval_texts), batch_size=args.batch_size)
        latency = time.time() - t0

        # Filtrar por umbral de confianza.
        # El modelo fine-tuneado usa LABEL_0/LABEL_1; el pre-entrenado usa
        # NEGATIVE/POSITIVE. Soportamos ambos formatos.
        label_map = {
            "POSITIVE": 1, "NEGATIVE": 0,
            "LABEL_1": 1, "LABEL_0": 0,
        }
        preds: list[int] = []
        true: list[int] = []
        for pred, label in zip(raw, eval_labels):
            if pred["score"] >= args.confidence_threshold:
                preds.append(label_map[pred["label"]])
                true.append(label)

        if not preds:
            print("WARN: umbral demasiado alto, sin predicciones tras filtrar")
            # Usar métricas del Trainer sin filtrar.
            acc = eval_results.get("eval_accuracy", 0.0)
            f1 = 0.0
            coverage = 0.0
        else:
            acc = accuracy_score(true, preds)
            f1 = f1_score(true, preds, average="weighted")
            coverage = len(preds) / len(eval_labels)

        # Log de métricas en MLflow.
        mlflow.log_metrics(
            {
                "accuracy": round(acc, 4),
                "f1_weighted": round(f1, 4),
                "coverage": round(coverage, 4),
                "avg_latency_ms": round((latency / len(eval_texts)) * 1000, 2),
                "train_time_s": round(train_time, 2),
                "train_loss": round(train_loss, 4),
                "eval_loss": round(eval_results.get("eval_loss", 0.0), 4),
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

        # ── 6. Registrar modelo pyfunc en MLflow ──────────────────────
        class SentimentWrapper(mlflow.pyfunc.PythonModel):
            """Wrapper MLflow pyfunc para servir el modelo fine-tuneado."""

            def load_context(
                self, context: mlflow.pyfunc.PythonModelContext
            ) -> None:
                """Carga config y pipeline al inicializar el modelo."""
                with open(context.artifacts["config"], encoding="utf-8") as f:
                    self.config = json.load(f)

                self.clf = pipeline(
                    "sentiment-analysis",
                    model=self.config["model_path"],
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
                label_map = {
                    "POSITIVE": 1, "NEGATIVE": 0,
                    "LABEL_1": 1, "LABEL_0": 0,
                }
                return [label_map[r["label"]] for r in results]

        # Guardar configuración como artefacto auxiliar del modelo.
        config_path = _build_temp_file("model_config.json")
        config_path.write_text(
            json.dumps(
                {
                    "model_path": fine_tuned_path,
                    "base_model": args.model_name,
                    "max_length": args.max_length,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                }
            ),
            encoding="utf-8",
        )

        mlflow.pyfunc.log_model(
            artifact_path="sentiment_model",
            python_model=SentimentWrapper(),
            artifacts={"config": str(config_path)},
        )

        print(
            f"Fine-tuning completado: acc={acc:.4f} | f1={f1:.4f} | "
            f"coverage={coverage:.2%} | tiempo={train_time:.1f}s"
        )


if __name__ == "__main__":
    main()
