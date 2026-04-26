# Sesión 8 — API de inferencia con FastAPI y MLflow Registry

## Objetivo de la sesión

Publicar el modelo registrado en la Sesión 7 como una **API REST** accesible
por HTTP. Hasta ahora el modelo vive dentro de MLflow; en esta sesión lo
"saquemos" al mundo real para que cualquier aplicación pueda consumirlo.

Esta sesión responde a la pregunta: *"Una vez que tengo un modelo en
Production, ¿cómo lo sirvo?"*

## Conceptos clave

### ¿Qué es una API REST?

Una **API REST** (Representational State Transfer) es un servicio web que
expone endpoints HTTP para que otros programas se comuniquen con él. En
nuestro caso:

- Un cliente envía textos en formato JSON.
- La API ejecuta el modelo de sentimiento.
- Devuelve las predicciones como JSON.

### ¿Qué es FastAPI?

[FastAPI](https://fastapi.tiangolo.com/) es un framework Python para crear
APIs REST de alto rendimiento. Ventajas clave:

- **Validación automática**: usa modelos Pydantic para validar peticiones y
  respuestas.
- **Documentación automática**: genera Swagger UI en `/docs`.
- **Async**: soporta peticiones asíncronas para alto rendimiento.

### ¿Qué es un endpoint?

Un **endpoint** es una ruta URL que responde a peticiones HTTP. Cada endpoint
tiene un método (GET, POST...) y una función asociada:

| Endpoint | Método | Función |
|---|---|---|
| `/health` | GET | Verificar que el servicio está activo |
| `/predict` | POST | Clasificar sentimiento de textos |
| `/predict/explain` | POST | Clasificar + explicar qué tokens influyeron |

### ¿Qué es Pydantic?

[Pydantic](https://docs.pydantic.dev/) es una librería de validación de
datos. Define **modelos** (esquemas) que validan automáticamente los datos
de entrada y salida:

```python
class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
```

Si alguien envía `{"texts": []}`, FastAPI devuelve automáticamente un error
422 (Unprocessable Entity) sin que escribamos código de validación.

### ¿Qué es un modelo perezoso (lazy loading)?

El **lazy loading** es un patrón donde el modelo se carga en memoria solo
la primera vez que se necesita, no al iniciar la aplicación. Esto es
importante porque:

- El modelo puede tardar varios segundos en cargarse.
- Si nadie llama a `/predict`, no se gasta memoria.
- En contenedores Docker, permite que el health check responda rápido.

### ¿Qué es la oclusión de tokens (token occlusion)?

La **oclusión de tokens** es una técnica de explicabilidad simple: se elimina
un token del texto y se observa cómo cambia la predicción. Si al quitar
"amazing" la confianza baja mucho, ese token era influyente.

```
Texto original:    "This product is amazing"    → POSITIVE (0.98)
Sin "amazing":     "This product is"            → POSITIVE (0.72)
Influencia de "amazing": 0.98 - 0.72 = 0.26
```

### ¿Qué es un modelo pyfunc?

Un **modelo pyfunc** es el formato universal de MLflow para servir modelos.
Envuelve cualquier modelo (PyTorch, scikit-learn, etc.) en una clase con
método `predict()`. La API carga el modelo con:

```python
model = mlflow.pyfunc.load_model("models:/SentimentAnalyzer/Production")
```

Esto desacopla el código de la API del modelo concreto: si cambias el modelo
en el Registry, la API lo recoge automáticamente sin modificar código.

## Relación con la Sesión 7

En la Sesión 7 registramos el modelo en el Model Registry y lo promovimos a
**Production**. En esta sesión:

1. La API se conecta al mismo `mlflow.db` de la Sesión 7.
2. Carga el modelo desde `models:/SentimentAnalyzer/Production`.
3. No referencia ningún `run_id` concreto: siempre carga la versión en
   Production.

Esto significa que si entrenas un modelo mejor y lo promueves a Production,
la API lo usa automáticamente sin cambios de código.

## Archivos incluidos

| Archivo | Descripción |
|---|---|
| `MLproject` | Ejecución estandarizada con entrypoints `serve` y `test` |
| `conda.yaml` | Entorno reproducible con conda |
| `requirements.txt` | Dependencias de la API |
| `app.py` | API FastAPI con endpoints `/health`, `/predict` y `/predict/explain` |
| `app_opcional.py` | Versión alternativa con explicación refactorizada en funciones separadas |
| `test_api.py` | Pruebas unitarias con `TestClient` de FastAPI |

## Endpoints

### `GET /health`

Verifica que el servicio está activo y muestra el modelo configurado.

**Respuesta:**
```json
{
  "status": "ok",
  "model_uri": "models:/SentimentAnalyzer/Production",
  "model_loaded": false,
  "tracking_uri": "sqlite:///../../Sesion7/sentiment-project/mlflow.db",
  "registry_uri": "sqlite:///../../Sesion7/sentiment-project/mlflow.db"
}
```

### `POST /predict`

Clasifica sentimiento de una lista de textos.

**Petición:**
```json
{"texts": ["I love this product", "This is terrible"]}
```

**Respuesta:**
```json
{
  "predictions": [1, 0],
  "model_uri": "models:/SentimentAnalyzer/Production"
}
```

Donde `1` = Positivo, `0` = Negativo.

### `POST /predict/explain?top_k=3`

Clasifica sentimiento y devuelve los tokens más influyentes por cada texto,
usando la técnica de oclusión.

**Petición:**
```json
{"texts": ["I love this product", "This is terrible"]}
```

**Respuesta:**
```json
{
  "predictions": [1, 0],
  "explanations": [
    {
      "text": "I love this product",
      "prediction": 1,
      "top_tokens": [
        {"token": "love", "influence": 0.35},
        {"token": "product", "influence": 0.02}
      ]
    }
  ],
  "model_uri": "models:/SentimentAnalyzer/Production"
}
```

## Ejecución

### Ejecutar pruebas unitarias

```bash
mlflow run . -e test
```

### Levantar la API

```bash
mlflow run . -e serve
```

### Levantar con parámetros personalizados

```bash
mlflow run . -e serve -P port=8080 -P model_name=SentimentAnalyzer -P model_stage=Production
```

### Ejecución directa (sin MLflow Projects)

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Flags de MLflow en esta sesión

| Flag | Descripción |
|---|---|
| `-e <entrypoint>` | Selecciona el entrypoint del `MLproject` (`test` o `serve`) |
| `-P port=<puerto>` | Puerto donde se levanta la API |
| `-P model_name=<nombre>` | Nombre del modelo registrado a cargar |
| `-P model_stage=<stage>` | Etapa del modelo (`Production`, `Staging`) |
| `-P tracking_uri=<uri>` | Ubicación del tracking server |
| `-P registry_uri=<uri>` | Ubicación del Model Registry |

## Nota sobre Model Registry

Por defecto, la API apunta al tracking/registry local de la Sesión 7:

```
sqlite:///../../Sesion7/sentiment-project/mlflow.db
```

Si necesitas otra ubicación, sobreescribe con variables de entorno o flags:

```bash
mlflow run . -e serve -P tracking_uri=sqlite:////ruta/a/mlflow.db
```

## Solución rápida: "Registered Model not found"

Si `/predict` devuelve que `SentimentAnalyzer` no existe, significa que el
Registry está vacío. Ejecuta primero el flujo de la Sesión 7:

```bash
cd ../Sesion7/sentiment-project
mlflow run . --experiment-name sentiment-analysis-hf -P epochs=2
python register_and_promote.py --promote-to-production
```

## Relación con sesiones anteriores y siguientes

```
Sesión 6 (Tracking)
  └─ Experimento "sentiment-analysis-hf" con runs

Sesión 7 (Projects + Registry)
  └─ Modelo fine-tuneado y registrado en Model Registry
  └─ Promovido a Production

Sesión 8 (API) ← ESTAMOS AQUÍ
  └─ Carga modelo desde models:/SentimentAnalyzer/Production
  └─ Expone /predict y /predict/explain como API REST

Sesión 9 (Docker)
  └─ Conteneriza esta misma API
  └─ Orquesta con Docker Compose (MLflow + API)

Sesión 10 (Monitorización)
  └─ Detecta drift en datos de producción
  └─ Señala cuándo reentrenar
```
