"""API de inferencia para la Sesión 8.

Expone endpoints HTTP para cargar el modelo desde MLflow Model Registry y
realizar predicciones de sentimiento.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Esquema de entrada para predicción por lote.

    Attributes:
        texts: Lista de textos a clasificar.
    """

    texts: list[str] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    """Esquema de salida del endpoint de predicción.

    Attributes:
        predictions: Etiquetas predichas (1 positivo, 0 negativo).
        model_uri: URI del modelo cargado desde MLflow.
    """

    predictions: list[int]
    model_uri: str


app = FastAPI(title="Sentiment API - Sesión 8", version="1.0.0")
_model_cache: Any | None = None


def configure_mlflow_uris() -> tuple[str, str]:
    """Configura URIs de tracking/registry con fallback al proyecto sesión 7.

    Returns:
        tuple[str, str]: URIs efectivas de tracking y registry.
    """
    default_sqlite = (
        Path(__file__).resolve().parent.parent /
        "sentiment-project" / "mlflow.db"
    )
    fallback_uri = f"sqlite:///{default_sqlite}"

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", fallback_uri)
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_REGISTRY_URI"] = registry_uri
    return tracking_uri, registry_uri


def get_model_uri() -> str:
    """Construye el URI del modelo desde variables de entorno."""
    model_name = os.getenv("MODEL_NAME", "SentimentAnalyzer")
    model_stage = os.getenv("MODEL_STAGE", "Production")
    return f"models:/{model_name}/{model_stage}"


def load_model() -> Any:
    """Carga perezosamente el modelo MLflow para reutilizarlo entre requests."""
    global _model_cache
    if _model_cache is None:
        _model_cache = mlflow.pyfunc.load_model(get_model_uri())
    return _model_cache


@app.get("/health")
def health() -> dict[str, Any]:
    """Endpoint de salud para monitoreo y diagnóstico básico."""
    tracking_uri, registry_uri = configure_mlflow_uris()
    return {
        "status": "ok",
        "model_uri": get_model_uri(),
        "model_loaded": _model_cache is not None,
        "tracking_uri": tracking_uri,
        "registry_uri": registry_uri,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Realiza inferencia de sentimiento sobre una lista de textos."""
    try:
        configure_mlflow_uris()
        model = load_model()
        model_input = pd.DataFrame({"text": request.texts})
        raw_predictions = model.predict(model_input)
        predictions = [int(value) for value in raw_predictions]
        return PredictResponse(predictions=predictions, model_uri=get_model_uri())
    except Exception as exc:  # pragma: no cover
        model_uri = get_model_uri()
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "<no-configurado>")
        raise HTTPException(
            status_code=500,
            detail=(
                "Error en inferencia: "
                f"{exc}. "
                f"model_uri={model_uri}, tracking_uri={tracking_uri}. "
                "Verifica que el modelo exista en Registry y que la versión de "
                "Python sea compatible con la del modelo registrado."
            ),
        ) from exc
