# Sesión 10 — Guía de ejecución paso a paso

## 1. Prerrequisitos

- **Python 3.10+** (recomendado 3.13)
- **`uv`** instalado (gestor de paquetes y entornos virtuales)
- **Acceso a internet** (para descargar modelos de HuggingFace y datasets)
- **Git** (si vas a clonar el repo)

## 2. Preparar el entorno

Desde la raíz del proyecto (`/home/emi/PGTD-MLOps` o tu carpeta local):

```bash
# Crear entorno virtual
uv venv

# Activar entorno
source .venv/bin/activate

# Instalar dependencias de la Sesión 10
uv pip install -r Sesion10/requirements.txt
```

> Nota: la primera instalación puede tardar varios minutos porque descarga PyTorch, Transformers, Evidently, etc.

---

## 3. Opción A — Ejecutar la API de inferencia (`app.py`)

La API expone una interfaz web interactiva y endpoints REST para predicción de sentimiento.

```bash
cd Sesion10
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Verificar que está funcionando

Abre en navegador: http://localhost:8000

Endpoints disponibles:

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Interfaz HTML interactiva |
| `/health` | GET | Estado de la API y modelo cargado |
| `/predict` | POST | Predicción de sentimiento (JSON) |
| `/docs` | GET | Documentación Swagger/OpenAPI |

### Ejemplo de petición con curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This movie is great!", "Terrible film."]}'
```

**Respuesta esperada:**

```json
{
  "predictions": [1, 0],
  "labels": ["POSITIVE", "NEGATIVE"],
  "model_uri": "hf://distilbert-base-uncased-finetuned-sst-2-english"
}
```

> **Nota sobre el modelo:** `app.py` intenta cargar el modelo desde MLflow Registry (`models:/SentimentAnalyzer/Production`). Si no lo encuentra (porque la Sesión 7 aún no fue ejecutada o no hay registro), hace **fallback automático** al modelo `distilbert-base-uncased-finetuned-sst-2-english` de HuggingFace. La API funciona igual en ambos casos.

---

## 4. Opción B — Generar informe HTML de drift (`generate_drift_report.py`)

Este script descarga datos de SST-2, extrae features textuales, genera predicciones y compara distribuciones con Evidently.

```bash
cd Sesion10
python generate_drift_report.py
```

### Salida esperada

```
Cargando datos...
Labels no válidos en test; usando validation[300:600].
Extrayendo features...
Generando predicciones del modelo...
Referencia → accuracy: 0.907
Producción → accuracy: 0.927
Generando informe de drift...
Informe guardado en drift_report.html

Resumen de drift:
Columnas con drift: 1.0
Proporción de drift: 10.0%
Dataset drifted: False
```

Abre `drift_report.html` en el navegador para ver el análisis visual.

---

## 5. Opción C — Monitorización programática (`monitor_production.py`)

### Modo simulado (por defecto)

Usa datos generados aleatoriamente para demostrar la lógica de alertas sin necesidad de la API.

```bash
cd Sesion10
python monitor_production.py --mode simulated
```

**Salida esperada:**

```json
{
  "alerts": [
    "DATA DRIFT: 100.0% de columnas con drift",
    "ACCURACY DROP: bajó 5.2% (ref=0.904, prod=0.852)"
  ],
  "drift_share": 1.0,
  "accuracy_ref": 0.904,
  "accuracy_prod": 0.852,
  "accuracy_drop": 0.052,
  "needs_retraining": true,
  "mode": "simulated"
}

⚠️  ACCION REQUERIDA: Re-entrenamiento recomendado
  - DATA DRIFT: 100.0% de columnas con drift
  - ACCURACY DROP: bajó 5.2% (ref=0.904, prod=0.852)
```

> El script devuelve código de salida `1` si se detecta drift, útil para pipelines CI/CD.

### Modo live (conectado a la API desplegada)

Requiere que la API de la Sesión 9/10 esté levantada previamente.

```bash
# Si la API está en local
python monitor_production.py --mode live --api-url http://localhost:8000

# Con chequeo periódico cada 120 segundos
python monitor_production.py --mode live --api-url http://localhost:8000 --interval 120
```

**Parámetros disponibles:**

| Parámetro | Default | Descripción |
|---|---|---|
| `--mode` | `simulated` | `simulated` o `live` |
| `--api-url` | `http://api:8000` | URL base de la API (solo live) |
| `--num-samples` | `200` | Muestras a evaluar (solo live) |
| `--interval` | `0` | Segundos entre chequeos. `0` = una sola vez |

---

## 6. Integración con Docker Compose (Sesión 9)

Si ya completaste la Sesión 9, puedes levantar el stack completo que incluye la monitorización automática:

```bash
cd Sesion9/sesion9-api
docker compose up -d
```

Servicios levantados:

| Servicio | Puerto | Función |
|---|---|---|
| `mlflow` | `5001` | Tracking server + Model Registry |
| `api` | `8000` | API de inferencia FastAPI |
| `monitor` | — | Monitorización periódica de drift |

Ver logs del monitor:

```bash
docker compose logs -f monitor
```

---

## 7. Posibles problemas y soluciones

| Problema | Causa | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'evidently'` | No se activó el entorno o no se instaló | `source .venv/bin/activate` y `uv pip install -r Sesion10/requirements.txt` |
| `Registered Model with name=SentimentAnalyzer not found` | MLflow no tiene el modelo registrado | Es normal si no se ejecutó la Sesión 7. `app.py` hace fallback automático a HuggingFace. |
| `API no disponible` en modo live | La API no está levantada o el puerto es diferente | Levanta la API primero (`uvicorn app:app --port 8000`) y usa `--api-url http://localhost:8000` |
| Descarga lenta de modelos / HuggingFace | Sin token de autenticación | Opcional: `export HF_TOKEN=tu_token` para límites más altos |
| Error de CUDA / GPU | PyTorch intenta usar GPU y no está disponible | El script funciona en CPU por defecto; no requiere GPU |

---

## 8. Resumen de archivos

| Archivo | Propósito |
|---|---|
| `app.py` | API FastAPI + interfaz web interactiva |
| `index.html` | Frontend HTML para enviar textos y ver predicciones |
| `generate_drift_report.py` | Genera reporte HTML visual de drift con Evidently |
| `monitor_production.py` | Monitor programático: drift + accuracy + alertas |
| `requirements.txt` | Dependencias de la sesión |
| `drift_report.html` | Ejemplo de salida del reporte (generado) |
| `steps.md` | Esta guía |
