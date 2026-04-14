# Sesión 8 - API de inferencia con MLflow Registry

Este proyecto expone una API REST con FastAPI que carga el modelo desde
MLflow Model Registry usando el alias por etapa:

`models:/SentimentAnalyzer/Production`

## Estructura

- `MLproject`: ejecución estandarizada para servir y probar.
- `conda.yaml`: entorno reproducible con conda.
- `requirements.txt`: dependencias de la API.
- `app.py`: endpoints `/health` y `/predict`.
- `test_api.py`: pruebas unitarias con `TestClient`.

## Ejecutar pruebas

`mlflow run . -e test`

## Levantar API

`mlflow run . -e serve`

## Levantar API con parámetros

`mlflow run . -e serve -P port=8080 -P model_name=SentimentAnalyzer -P model_stage=Production`

## Flags de MLflow en esta sesión (Sesión 8)

- `-e <entrypoint>`
  - Selecciona el entrypoint del `MLproject` (`test` o `serve`).
- `-P port=<puerto>`
  - Define el puerto donde se levanta la API.
- `-P model_name=<nombre>`
  - Nombre del modelo registrado a cargar desde Model Registry.
- `-P model_stage=<stage>`
  - Etapa del modelo a resolver (por ejemplo, `Production`).
- `-P tracking_uri=<uri>`
  - Sobrescribe la ubicación del tracking server.
- `-P registry_uri=<uri>`
  - Sobrescribe la ubicación del Model Registry.

## Endpoint de explicación

Además de `/predict`, la API expone `/predict/explain` para devolver los
tokens más influyentes en cada predicción.

Ejemplo:

`POST /predict/explain?top_k=3`

Body:

`{"texts": ["I love this product", "This is terrible"]}`

## Nota sobre Model Registry

Por defecto, la API apunta al tracking/registry local de la Sesión 8:

`sqlite:///../sentiment-project/mlflow.db`

Si necesitas otra ubicación, puedes sobreescribir:

`mlflow run . -e serve -P tracking_uri=sqlite:////ruta/a/mlflow.db -P registry_uri=sqlite:////ruta/a/mlflow.db`

## Solución rápida: "Registered Model not found"

Si `/predict` devuelve que `SentimentAnalyzer` no existe, significa que el
Registry (la base `mlflow.db` configurada) está vacío.

Registra una versión y promuévela a `Production` antes de servir la API.
Una opción es ejecutar el flujo de registro/promoción de `Sesion7/sentiment-project`
apuntando al tracking de Sesión 8 (mismas variables para tracking y registry).
