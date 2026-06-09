"""Ejemplo de carga del modelo en Production desde MLflow Registry."""

from __future__ import annotations

import argparse

import mlflow.pyfunc
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parsea nombre del modelo para carga desde Registry."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="SentimentAnalyzer")
    return parser.parse_args()


def main() -> None:
    """Carga el modelo en Production y realiza una predicción de prueba."""
    args = parse_args()
    prod_model = mlflow.pyfunc.load_model(
        f"models:/{args.model_name}/Production"
    )
    test = pd.DataFrame({"text": ["very bad!"]})
    pred = prod_model.predict(test)
    print("Predicción:", pred)


if __name__ == "__main__":
    main()
