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
    hola: str = "mundo"  # Campo adicional para demostrar cambios en el modelo
    model_uri: str


class TokenInfluence(BaseModel):
    """Representa la contribución estimada de un token.

    Attributes:
        token: Token evaluado.
        influence: Impacto sobre la clase predicha. Mayor valor absoluto,
            mayor influencia.
    """

    token: str
    influence: float


class ExplainItem(BaseModel):
    """Explicación para un texto individual.

    Attributes:
        text: Texto original de entrada.
        prediction: Etiqueta predicha (1 positivo, 0 negativo).
        top_tokens: Tokens más influyentes en la predicción.
    """

    text: str
    prediction: int
    top_tokens: list[TokenInfluence]


class ExplainResponse(BaseModel):
    """Respuesta del endpoint de predicción con explicación.

    Attributes:
        predictions: Etiquetas predichas para todos los textos.
        explanations: Explicación por texto con sus tokens más influyentes.
        model_uri: URI del modelo cargado desde MLflow.
    """

    predictions: list[int]
    explanations: list[ExplainItem]
    model_uri: str


app = FastAPI(title="Sentiment API - Sesión 8", version="1.0.0")
_model_cache: Any | None = None


def configure_mlflow_uris() -> tuple[str, str]:
    """Configura URIs de tracking/registry con fallback al proyecto Sesión 7.

    Returns:
        tuple[str, str]: URIs efectivas de tracking y registry.
    """
    default_sqlite = (
        Path(__file__).resolve().parent.parent.parent /
        "Sesion7" / "sentiment-project" / "mlflow.db"
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


def _get_pipeline_from_model(model: Any) -> Any | None:
    """Extrae pipeline de Transformers desde pyfunc si está disponible."""
    try:
        return model._model_impl.python_model.clf  # type: ignore[attr-defined]
    except Exception:
        return None


def _tokenize_text(text: str) -> list[str]:
    """Tokeniza de forma simple preservando el orden de palabras."""
    return [token for token in text.split() if token]


def _score_for_label(label: str, score: float, target_label: int) -> float:
    """Obtiene un score comparable para la etiqueta objetivo."""
    positive_score = score if label.upper() == "POSITIVE" else 1.0 - score
    return positive_score if target_label == 1 else 1.0 - positive_score


def _explain_single_text(
    model: Any,
    pipeline_model: Any | None,
    text: str,
    prediction: int,
    top_k: int,
) -> list[TokenInfluence]:
    """Genera explicación token-level para un texto por oclusión."""
    tokens = _tokenize_text(text)
    if not tokens:
        return []

    if pipeline_model is not None:
        base_result = pipeline_model([text])[0]
        base_conf = _score_for_label(
            str(base_result["label"]),
            float(base_result["score"]),
            prediction,
        )
        influences: list[TokenInfluence] = []
        for idx, token in enumerate(tokens):
            reduced = tokens[:idx] + tokens[idx + 1:]
            perturbed = " ".join(reduced) if reduced else "[MASK]"
            p_result = pipeline_model([perturbed])[0]
            p_conf = _score_for_label(
                str(p_result["label"]),
                float(p_result["score"]),
                prediction,
            )
            influences.append(
                TokenInfluence(token=token, influence=base_conf - p_conf)
            )
    else:
        influences = []
        for idx, token in enumerate(tokens):
            reduced = tokens[:idx] + tokens[idx + 1:]
            perturbed = " ".join(reduced) if reduced else "[MASK]"
            p_df = pd.DataFrame({"text": [perturbed]})
            p_pred = int(model.predict(p_df)[0])
            inf = 1.0 if p_pred != prediction else 0.0
            influences.append(TokenInfluence(token=token, influence=inf))

    influences.sort(key=lambda item: abs(item.influence), reverse=True)
    return influences[:top_k]


@app.get("/health") # ENDPOINT
async def health() -> dict[str, Any]:
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
async def predict(request: PredictRequest) -> PredictResponse:
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


@app.post("/predict/explain", response_model=ExplainResponse)
async def predict_explain(request: PredictRequest, top_k: int = 3) -> ExplainResponse:
    """Realiza inferencia y devuelve tokens influyentes por cada texto.

    Args:
        request: Lista de textos para predecir y explicar.
        top_k: Máximo de tokens influyentes a devolver por texto.
    """
    if top_k < 1:
        raise HTTPException(status_code=422, detail="top_k debe ser >= 1")

    try:
        configure_mlflow_uris()
        model = load_model()
        model_input = pd.DataFrame({"text": request.texts})
        raw_predictions = model.predict(model_input)
        predictions = [int(value) for value in raw_predictions]

        pipeline_model = _get_pipeline_from_model(model)
        explanations: list[ExplainItem] = []
        for text, prediction in zip(request.texts, predictions):
            top_tokens = _explain_single_text(
                model=model,
                pipeline_model=pipeline_model,
                text=text,
                prediction=prediction,
                top_k=top_k,
            )
            explanations.append(
                ExplainItem(
                    text=text,
                    prediction=prediction,
                    top_tokens=top_tokens,
                )
            )

        return ExplainResponse(
            predictions=predictions,
            explanations=explanations,
            model_uri=get_model_uri(),
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail=f"Error en explicación: {exc}",
        ) from exc
