# Sesión 9 - API de inferencia con Docker + Frontend

Despliegue containerizado completo del stack MLOps:
- **MLflow Tracking/Registry** (puerto 5001)
- **API FastAPI** con CORS (puerto 8000)
- **Frontend web** (puerto 3000)

## Arquitectura

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Frontend   │──────│  API (8000) │──────│ MLflow (5001│
│  (nginx)    │      │  (FastAPI)  │      │  (gunicorn) │
│  :3000      │      │  :8000      │      │  :5000/5001 │
└─────────────┘      └─────────────┘      └─────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Model Registry   │
                    │  SentimentAnalyzer│
                    │  /Production      │
                    └─────────────────┘
```

## Archivos incluidos

### Servicios Docker
- `Dockerfile` — API FastAPI (Python 3.11 + dependencias MLflow/transformers)
- `Dockerfile.mlflow` — Imagen MLflow con preinstalación de `mlflow==2.21.0`
- `docker-compose.yml` — Orquesta los 3 servicios (mlflow, api, frontend)
- `app.py` — API con endpoints `/health`, `/predict`, `/predict/explain`
- `requirements.txt` — Dependencias exactas (versiones probadas en Docker)
- `test_stack.py` — Tests de integración end-to-end

### Frontend
- `../frontend/index.html` — Interfaz de usuario
- `../frontend/style.css` — Estilos responsive
- `../frontend/app.js` — Cliente JavaScript que consume la API
- `../frontend/Dockerfile` — Imagen nginx:alpine

## Prerrequisitos

- Docker + Docker Compose
- El stack requiere que el modelo `SentimentAnalyzer` esté en `Production` en el Model Registry
- Si el volumen `mlflow_data` está vacío (primer arranque o después de `docker compose down -v`), hay que entrenar y registrar un modelo (ver sección "Poblar el Model Registry")

## Levantar el stack completo

```bash
cd Sesion9/sesion9-api

# Construir y arrancar todo
docker compose up --build -d

# Verificar estado
docker compose ps
# Debe mostrar: mlflow (healthy), api (up), frontend (up)
```

Servicios disponibles:
- **MLflow UI**: http://localhost:5001
- **API FastAPI**: http://localhost:8000
  - Docs interactivos: http://localhost:8000/docs
  - Endpoints: `GET /health`, `POST /predict`, `POST /predict/explain`
- **Frontend**: http://localhost:3000

## Poblar el Model Registry (si está vacío)

Si la API devuelve `RESOURCE_DOES_NOT_EXIST: Registered Model with name=SentimentAnalyzer not found`, el registro está vacío. Ejecuta los scripts de entrenamiento desde un contenedor Docker con la misma versión de MLflow que el servidor:

```bash
# Ejecutar desde la raíz del repositorio (MLOPs)
docker run --rm \
  --network=sesion9-api_default \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -v $(pwd)/Sesion7/sentiment-project:/workspace \
  -w /workspace \
  python:3.11-slim \
  sh -c 'pip install --no-cache-dir mlflow==2.21.0 transformers==4.48.0 torch==2.5.1 datasets==3.4.1 scikit-learn==1.5.2 evaluate==0.4.3 numpy==2.0.2 pandas==2.2.2 pyarrow==17.0.0 && \
  python train.py && \
  python register_and_promote.py --promote-to-production'
```

Esto:
1. Instala las mismas versiones que usa el contenedor `api`
2. Ejecuta `train.py` contra el MLflow de Docker (`http://mlflow:5000`)
3. Registra y promueve el modelo a `Production`

## Validar el stack

```bash
# Desde Sesion9/sesion9-api

# Test de integración
python test_stack.py

# O manualmente:
# Health check
curl http://localhost:8000/health

# Predicción simple
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This is amazing!", "I hate this product."]}'
# → {"predictions": [1, 0], "model_uri": "models:/SentimentAnalyzer/Production"}

# Predicción con explicación
curl -X POST "http://localhost:8000/predict/explain?top_k=3" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This is amazing!"]}'
```

## Comandos útiles de Docker Compose

```bash
# Ver logs en tiempo real
docker compose logs -f

# Ver logs solo de un servicio
docker compose logs -f api

# Reconstruir un servicio específico
docker compose up -d --build api

# Detener todo
docker compose down

# Detener y borrar volumen (⚠️ borra el registro de MLflow)
docker compose down -v

# Reiniciar un servicio
docker compose restart api
```

## Cambios principales respecto a la sesión original

1. **Python 3.11** en lugar de 3.13 (compatibilidad con transformers/torch)
2. **MLflow 2.21.0** en lugar de 3.11.1 (versión que existe en PyPI)
3. **Dockerfile.mlflow** — evita instalar MLflow en cada inicio del contenedor
4. **CORS** habilitado en `app.py` para permitir requests desde el frontend
5. **Ruta fallback** corregida para apuntar a `Sesion7/sentiment-project/mlflow.db`
6. **Frontend HTML/CSS/JS** nuevo con Docker Compose

## Notas de implementación

### Dockerfile.mlflow
El contenedor MLflow original instalaba `mlflow` via `pip` en el `command` de inicio. Esto tomaba más de 40 segundos en ARM (Mac M1/M2/M3), superando el healthcheck. El Dockerfile.mlflow preinstala la dependencia durante el build, logrando arranque inmediato.

### Compatibilidad de versiones
Es crítico que la versión del cliente MLflow (en `train.py`) coincida con la del servidor. El contenedor `api` usa `mlflow==2.21.0` y la imagen base del `mlflow` también. Usar MLflow 3.x desde el host contra un servidor 2.x provoca errores de API (`logged-models` 404).

### Volumen mlflow_data
El volumen Docker persiste la base de datos SQLite y artefactos. Si se borra (`docker compose down -v`), se pierde el registro. En ese caso, se debe ejecutar el pipeline de entrenamiento/registro nuevamente.

## Variables de entorno de la API

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | URI del servidor MLflow |
| `MODEL_NAME` | `SentimentAnalyzer` | Nombre del modelo en Registry |
| `MODEL_STAGE` | `Production` | Etapa/alias del modelo a cargar |
