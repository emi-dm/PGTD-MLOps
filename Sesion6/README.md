# Sesión 6 — Seguimiento de experimentos con MLflow Tracking

## Objetivo de la sesión

Aprender a **registrar y comparar experimentos** de machine learning de forma
sistemática. En lugar de imprimir métricas por consola o guardarlas en
hojas de cálculo, usamos **MLflow Tracking** para que cada ejecución quede
documentada con sus parámetros, métricas y artefactos.

Esta sesión es el **punto de partida** del curso: sin tracking no hay
trazabilidad, y sin trazabilidad no es posible tomar decisiones informadas
sobre qué modelo desplegar.

## Conceptos clave

### ¿Qué es MLflow?

[MLflow](https://mlflow.org/) es una plataforma open-source para gestionar el
ciclo de vida del ML. Tiene cuatro módulos principales:

| Módulo | Función | Sesión del curso |
|---|---|---|
| **MLflow Tracking** | Registrar experimentos (parámetros, métricas, artefactos) | Sesión 6 |
| **MLflow Projects** | Estandarizar la ejecución del código | Sesión 7 |
| **MLflow Models** | Empaquetar modelos con formato estándar | Sesión 7 |
| **MLflow Model Registry** | Gestionar el ciclo de vida de modelos (versionado, etapas) | Sesión 7 |

### ¿Qué es un experimento?

Un **experimento** es un grupo lógico de ejecuciones relacionadas. En nuestro
caso, el experimento se llama `sentiment-analysis-hf` y agrupa todas las
pruebas de clasificación de sentimiento.

### ¿Qué es un run?

Un **run** es una ejecución individual del código. Cada run registra:

- **Parámetros** (`params`): valores de entrada que definen la configuración
  (modelo, batch size, umbral de confianza, etc.).
- **Métricas** (`metrics`): resultados numéricos del experimento (accuracy,
  f1, coverage, latencia).
- **Artefactos** (`artifacts`): ficheros generados (reporte de clasificación,
  muestras de predicción).
- **Tags**: metadatos opcionales para organizar runs.

### ¿Qué es un pipeline de inferencia?

Un **pipeline** de HuggingFace encapsula un modelo pre-entrenado y su
preprocesamiento en una sola interfaz. En nuestro caso:

```python
clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
resultado = clf("I love this product")
# → [{'label': 'POSITIVE', 'score': 0.9998}]
```

El pipeline se encarga de tokenizar el texto, pasar por el modelo y devolver
la etiqueta con su confianza.

### ¿Qué es el umbral de confianza?

El modelo devuelve una probabilidad para cada predicción (por ejemplo, 0.95
de ser positivo). El **umbral de confianza** (`confidence_threshold`) es el
valor mínimo que debe superar esa probabilidad para aceptar la predicción.

- Umbral **bajo** (0.5): acepta más predicciones, pero incluye las
  inciertas.
- Umbral **alto** (0.9): solo acepta predicciones muy seguras, pero pierde
  cobertura.

### ¿Qué es coverage?

**Coverage** es la proporción de muestras que superan el umbral de
confianza. Si evaluamos 200 textos y solo 150 superan el umbral, el
coverage es 75%. Es un trade-off con la calidad: a mayor umbral, menor
cobertura.

## Dataset: SST-2

El dataset **SST-2** (Stanford Sentiment Treebank) del benchmark
[GLUE](https://gluebenchmark.com/) contiene reseñas de películas de
Rotten Tomatoes clasificadas como positivas o negativas:

| Propiedad | Valor |
|---|---|
| Tarea | Clasificación binaria de sentimiento |
| Clases | `0` = Negativo, `1` = Positivo |
| Train | ~67 349 frases |
| Validation | ~872 frases |

En esta sesión usamos solo el split de **validación** para evaluar un modelo
ya entrenado. En la Sesión 7, usaremos el split de **entrenamiento** para
fine-tuning real.

## Modelo utilizado

**DistilBERT fine-tuneado en SST-2** (`distilbert-base-uncased-finetuned-sst-2-english`):
un modelo de HuggingFace que ya viene entrenado para clasificar sentimiento.
En esta sesión lo usamos tal cual, sin modificar sus pesos.

> En la Sesión 7 cambiaremos a `distilbert-base-uncased` (modelo base **sin**
> fine-tuning) para entrenar desde cero.

## Archivos incluidos

| Archivo | Descripción |
|---|---|
| `experiment_tracking.py` | Script principal: evalúa 3 configuraciones y registra en MLflow |
| `requirements.txt` | Dependencias Python |

## ¿Qué hace `experiment_tracking.py`?

El script ejecuta **3 configuraciones** diferentes sobre el mismo modelo y
dataset, registrando cada una como un run independiente en MLflow:

| Run | Batch size | Umbral confianza | Max length |
|---|---|---|---|
| Config 1 | 16 | 0.5 | 128 |
| Config 2 | 32 | 0.7 | 128 |
| Config 3 | 16 | 0.9 | 64 |

Para cada configuración:

1. Carga el modelo pre-entrenado de HuggingFace.
2. Ejecuta inferencia sobre las muestras de validación.
3. Filtra predicciones por umbral de confianza.
4. Calcula métricas: accuracy, f1, coverage, latencia.
5. Registra todo en MLflow (parámetros, métricas, artefactos).

## Ejecución

### 1. Levantar el servidor MLflow

En una terminal separada:

```bash
mlflow ui --port 5000
```

Esto abre la interfaz web en [http://localhost:5000](http://localhost:5000).

### 2. Ejecutar el experimento

```bash
cd Sesion6
pip install -r requirements.txt
python experiment_tracking.py
```

### 3. Comparar resultados

Abre [http://localhost:5000](http://localhost:5000) y navega al experimento
`sentiment-analysis-hf`. Podrás comparar las 3 runs lado a lado, ver
gráficas de métricas y descargar artefactos.

## Métricas registradas

| Métrica | Descripción |
|---|---|
| `accuracy` | Proporción de predicciones correctas |
| `f1_weighted` | Media armónica ponderada de precisión y recall |
| `coverage` | Fracción de muestras que superan el umbral de confianza |
| `avg_latency_ms` | Tiempo medio de inferencia por muestra (ms) |
| `uncertain_samples` | Muestras descartadas por no superar el umbral |

## Relación con la siguiente sesión

En la **Sesión 7** tomaremos este mismo experimento y daremos el siguiente paso:

- **MLflow Projects**: estandarizar cómo se ejecuta el código (entorno,
  parámetros, entrypoints).
- **Fine-tuning real**: entrenar DistilBERT desde cero en lugar de usar un
  modelo pre-entrenado.
- **Model Registry**: registrar el mejor modelo y gestionar su ciclo de vida
  (Staging → Production).

El experimento `sentiment-analysis-hf` creado aquí se reutiliza en todas las
sesiones siguientes.
