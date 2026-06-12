"""Sesión 10: API de inferencia con interfaz interactiva HTML.

Expone endpoints HTTP para predicción de sentimiento y sirve una página
HTML interactiva para realizar peticiones desde el navegador.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

try:
    from transformers import pipeline
except ImportError:
    pipeline = None  # type: ignore[assignment]


class HuggingFaceSentimentWrapper:
    """Wrapper para que un pipeline de HuggingFace tenga la misma interfaz que mlflow.pyfunc."""

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> None:
        if pipeline is None:
            raise RuntimeError("La librería 'transformers' no está instalada.")
        self._clf = pipeline(
            "sentiment-analysis",
            model=model_name,
            truncation=True,
            max_length=128,
        )
        self._model_name = model_name

    def predict(self, model_input: pd.DataFrame) -> Any:
        """Espera un DataFrame con columna 'text' y devuelve un array de ints (1/0)."""
        texts = model_input["text"].tolist()
        results = self._clf(texts, batch_size=32)
        label_map: dict[str, int] = {"POSITIVE": 1, "NEGATIVE": 0}
        mapped: list[int] = []
        for result in results:
            raw_label = str(result["label"]).upper()
            if raw_label in label_map:
                mapped.append(label_map[raw_label])
            elif "POS" in raw_label:
                mapped.append(1)
            else:
                mapped.append(0)
        return mapped

    def __repr__(self) -> str:
        return f"HuggingFaceSentimentWrapper(model={self._model_name})"


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
        labels: Etiquetas legibles (POSITIVE / NEGATIVE).
        model_uri: URI del modelo cargado desde MLflow.
    """

    predictions: list[int]
    labels: list[str]
    model_uri: str


app = FastAPI(title="Sentiment API - Sesión 10", version="1.0.0")
_model_cache: Any | None = None


def configure_mlflow_uris() -> tuple[str, str]:
    """Configura URIs de tracking/registry con fallback al proyecto Sesión 7.

    Returns:
        tuple[str, str]: URIs efectivas de tracking y registry.
    """
    default_sqlite = (
        Path(__file__).resolve().parent.parent /
        "Sesion7" / "sentiment-project" / "mlflow.db"
    )
    fallback_uri = f"sqlite:///{default_sqlite}"

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", fallback_uri)
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_REGISTRY_URI"] = registry_uri
    return tracking_uri, registry_uri


_model_uri: str = ""


def get_model_uri() -> str:
    """Construye el URI del modelo desde variables de entorno."""
    model_name = os.getenv("MODEL_NAME", "SentimentAnalyzer")
    model_stage = os.getenv("MODEL_STAGE", "Production")
    return f"models:/{model_name}/{model_stage}"


def load_model() -> Any:
    """Carga perezosamente el modelo MLflow para reutilizarlo entre requests.

    Si no existe el modelo en MLflow Registry, hace fallback al modelo
    distilbert-base-uncased-finetuned-sst-2-english de HuggingFace.
    """
    global _model_cache, _model_uri
    if _model_cache is None:
        configure_mlflow_uris()
        mlflow_uri = get_model_uri()
        try:
            _model_cache = mlflow.pyfunc.load_model(mlflow_uri)
            _model_uri = mlflow_uri
        except Exception as mlflow_exc:
            # Fallback a HuggingFace para que la sesión funcione de forma independiente.
            fallback_name = "distilbert-base-uncased-finetuned-sst-2-english"
            try:
                _model_cache = HuggingFaceSentimentWrapper(model_name=fallback_name)
                _model_uri = f"hf://{fallback_name}"
            except Exception as hf_exc:
                raise RuntimeError(
                    f"No se pudo cargar el modelo MLflow ({mlflow_exc}) ni el fallback de HuggingFace ({hf_exc})."
                ) from hf_exc
    return _model_cache


@app.get("/")
def serve_ui() -> FileResponse:
    """Sirve la página HTML interactiva."""
    html_path = Path(__file__).parent / "index.html"
    return FileResponse(html_path, media_type="text/html")


@app.get("/health")
def health() -> dict[str, Any]:
    """Endpoint de salud para monitoreo y diagnóstico básico."""
    tracking_uri, registry_uri = configure_mlflow_uris()
    return {
        "status": "ok",
        "model_uri": _model_uri if _model_uri else get_model_uri(),
        "model_loaded": _model_cache is not None,
        "tracking_uri": tracking_uri,
        "registry_uri": registry_uri,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Realiza inferencia de sentimiento sobre una lista de textos."""
    try:
        model = load_model()
        model_input = pd.DataFrame({"text": request.texts})
        raw_predictions = model.predict(model_input)
        predictions = [int(value) for value in raw_predictions]
        labels = ["POSITIVE" if p == 1 else "NEGATIVE" for p in predictions]
        return PredictResponse(
            predictions=predictions,
            labels=labels,
            model_uri=_model_uri if _model_uri else get_model_uri(),
        )
    except Exception as exc:
        model_uri = _model_uri if _model_uri else get_model_uri()
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "<no-configurado>")
        raise HTTPException(
            status_code=500,
            detail=(
                f"Error en inferencia: {exc}. "
                f"model_uri={model_uri}, tracking_uri={tracking_uri}. "
                "Verifica que el modelo exista en Registry o que el fallback esté disponible."
            ),
        ) from exc
