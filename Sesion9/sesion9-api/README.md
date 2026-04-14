# Sesión 9 - API de inferencia con Docker (basada en Sesión 8)

Esta carpeta contiene la implementación de despliegue de la API de
sentimientos en Docker y Docker Compose, ubicada en `Sesion9`.

> Requisito de compatibilidad: la imagen usa Python 3.13 para coincidir con
> la versión con la que se guardaron los modelos en MLflow.

## Archivos incluidos

- `app.py`: API FastAPI con endpoints `/health` y `/predict`.
- `requirements.txt`: dependencias del servicio.
- `Dockerfile`: imagen de la API.
- `.dockerignore`: exclusiones para build de Docker.
- `docker-compose.yml`: stack completo (`mlflow` + `api`).
- `.env`: variables de entorno de referencia.
- `test_stack.py`: prueba de integración end-to-end.

## Build de imagen

Desde `Sesion9/sesion9-api`:

- `docker build -t sentiment-api:1.0 .`

Ejecución directa:

- `docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MODEL_NAME=SentimentAnalyzer \
  -e MODEL_STAGE=Production \
  sentiment-api:1.0`

## Stack completo con Docker Compose

> Nota: en esta configuración MLflow se expone en `http://localhost:5001`
> para evitar conflictos comunes con servicios locales en `5000`.

- `docker compose up -d`
- `docker compose logs -f`
- `docker compose logs -f api`
- `docker compose ps`
- `docker compose up -d --build api`
- `docker compose down`
- `docker compose down -v`

## Validación automática

Con stack arriba:

- `python test_stack.py`

## Nota sobre Model Registry

La API busca el modelo en `models:/SentimentAnalyzer/Production`.
Si el MLflow del stack está vacío, primero registra y promueve el modelo.

## Flags de MLflow en esta sesión (Sesión 9)

Se usan estos flags en el servicio `mlflow` de `docker-compose.yml`:

- `--host 0.0.0.0`
  - Expone el servidor en todas las interfaces del contenedor.
- `--port 5000`
  - Puerto interno del servidor MLflow.
- `--allowed-hosts ...`
  - Lista de hosts permitidos para evitar rechazo por Host header
    (protección contra DNS rebinding).
- `--backend-store-uri sqlite:////mlflow/mlflow.db`
  - Base de datos backend para metadatos de tracking/registry.
- `--artifacts-destination /mlflow/artifacts`
  - Ruta donde se almacenan artefactos de runs/modelos.
- `--serve-artifacts`
  - Habilita proxy HTTP de artefactos para clientes remotos (evita errores
    de escritura local en rutas del contenedor).
