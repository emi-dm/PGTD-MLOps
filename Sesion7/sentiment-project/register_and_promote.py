"""Registro y promoción de modelo en MLflow Model Registry.

Flujo:
1) busca mejor run por accuracy,
2) registra en Registry,
3) agrega metadata,
4) promueve a Staging,
5) (opcional) promueve a Production.
"""

from __future__ import annotations

import argparse

import mlflow
from mlflow import MlflowClient


def parse_args() -> argparse.Namespace:
    """Parsea argumentos para controlar el flujo de promoción."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="SentimentAnalyzer")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="sentiment-analysis-hf",
    )
    parser.add_argument(
        "--promote-to-production",
        action="store_true",
        help="Si se activa, promueve a Production y archiva previas.",
    )
    return parser.parse_args()


def main() -> None:
    """Ejecuta registro y transición de etapas en Model Registry."""
    args = parse_args()
    client = MlflowClient()

    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None and args.experiment_name != "Default":
        experiment = client.get_experiment_by_name("Default")
        if experiment is not None:
            print(
                f"Experimento '{args.experiment_name}' no encontrado; "
                "usando 'Default'."
            )

    if experiment is None:
        raise ValueError(
            f"No existe el experimento: {args.experiment_name}. "
            "Ejecuta primero train.py al menos una vez."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No hay runs disponibles para registrar.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    accuracy = best_run.data.metrics.get("accuracy", "N/A")

    print(f"Mejor run: {best_run_id}")
    print(f"Accuracy:  {accuracy}")

    model_uri = f"runs:/{best_run_id}/sentiment_model"
    registered = mlflow.register_model(model_uri, args.model_name)
    print(f"Versión registrada: {registered.version}")

    client.update_registered_model(
        name=args.model_name,
        description=(
            "Clasificador de sentimientos basado en DistilBERT "
            "(SST-2)."
        ),
    )
    client.set_registered_model_tag(
        args.model_name,
        "task",
        "sentiment-classification",
    )
    client.set_registered_model_tag(
        args.model_name,
        "base_model",
        "distilbert",
    )

    client.transition_model_version_stage(
        name=args.model_name,
        version=registered.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"Modelo v{registered.version} en Staging")

    if args.promote_to_production:
        client.transition_model_version_stage(
            name=args.model_name,
            version=registered.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Modelo v{registered.version} en PRODUCCION")


if __name__ == "__main__":
    main()
