# Sesión 7 — Empaquetado y registro con MLflow Projects & Model Registry

## Objetivo de la sesión

Estandarizar la ejecución de proyectos ML con **MLflow Projects** y
establecer un registro oficial de modelos con **MLflow Model Registry**.

En la Sesión 6 aprendimos a *trackear* experimentos. Ahora damos dos pasos
adelante:

1. **Ejecución reproducible**: cualquiera que haga `git clone` puede
   ejecutar el mismo experimento con un solo comando, sin adivinar
   dependencias ni parámetros.
2. **Gestión del ciclo de vida del modelo**: el mejor modelo se registra
   oficialmente y se promueve por etapas (None → Staging → Production),
   respondiendo a la pregunta *"¿qué modelo está en producción ahora mismo?"*.

## Conceptos clave

### MLflow Projects

Un **MLflow Project** es un directorio con un fichero `MLproject` que
describe cómo ejecutar el código de forma estándar. Define:

- **Nombre** del proyecto.
- **Entorno** reproducible (`conda_env` o `python_env`).
- **Entry points** (comandos ejecutables con parámetros tipados).

Con `mlflow run .` se ejecuta todo automáticamente: se crea el entorno, se
resuelven los parámetros y se corre el comando. No hace falta leer un README
para saber cómo ejecutar el proyecto.

### MLflow Model Registry

El **Model Registry** es el catálogo oficial de modelos de la organización.
Cada modelo registrado tiene:

- **Versiones**: cada vez que se registra un modelo desde un run, se crea
  una nueva versión (v1, v2, v3...).
- **Etapas** (stages): `None` → `Staging` → `Production` → `Archived`.
  Cada etapa indica el nivel de confianza del modelo.
- **Tags y descripción**: metadatos para documentar el modelo.

El flujo típico es:

```
Experimentar (runs) → Registrar mejor run → Promover a Staging →
Validar → Promover a Production → Servir en API
```

### MLflow Models y sabores (flavors)

Cuando guardamos un modelo con `mlflow.pyfunc.log_model`, MLflow crea una
estructura estándar con:

- `MLmodel`: metadatos y sabores disponibles.
- `python_model.pkl`: el wrapper serializado.
- `conda.yaml` / `requirements.txt`: entorno reproducible.
- `artifacts/`: archivos auxiliaries (config, etc.).

Un **sabor** (flavor) es la interfaz para cargar el modelo. El sabor
`pyfunc` es el universal: funciona con cualquier modelo envuelto en una
clase que implemente `predict()`.

### Wrapper pyfunc

Un **wrapper pyfunc** (`mlflow.pyfunc.PythonModel`) es una clase Python que
envuelve un modelo para que MLflow pueda serializarlo, versionarlo y
servirlo. Implementa dos métodos:

- `load_context()`: se ejecuta al cargar el modelo (inicializa el pipeline).
- `predict()`: recibe un DataFrame y devuelve predicciones.

### Fine-tuning

**Fine-tuning** es el proceso de tomar un modelo pre-entrenado y
entrenarlo adicionalmente sobre un dataset específico. En nuestro caso:

- **Modelo base**: `distilbert-base-uncased` (entrenado en texto general,
  pero **no** en sentimiento).
- **Dataset**: SST-2 (reseñas de películas clasificadas como
  positivas/negativas).
- **Resultado**: un modelo especializado en análisis de sentimiento.

> Usamos el modelo base (sin fine-tuning previo) y no
> `distilbert-base-uncased-finetuned-sst-2-english` (ya fine-tuneado) para
> que cada run con distintos hiperparámetros produzca un modelo realmente
> diferente.

### Hiperparámetros

Los **hiperparámetros** son valores que configuran el proceso de
entrenamiento (no se aprenden de los datos):

| Hiperparámetro | Qué controla |
|---|---|
| `epochs` | Cuántas veces el modelo ve todo el dataset |
| `learning_rate` | Qué tan grandes son los ajustes de pesos |
| `batch_size` | Cuántas muestras se procesan antes de actualizar pesos |
| `weight_decay` | Regularización para evitar sobreajuste |
| `num_train_samples` | Cuántos ejemplos de entrenamiento usar |

## Dataset: SST-2 (Stanford Sentiment Treebank)

El dataset **SST-2** del benchmark [GLUE](https://gluebenchmark.com/),
accesible vía HuggingFace Hub con `load_dataset("glue", "sst2")`.

| Propiedad | Valor |
|---|---|
| Tarea | Clasificación binaria de sentimiento |
| Clases | `0` = Negativo, `1` = Positivo |
| Origen | Reseñas de películas de Rotten Tomatoes |
| Train | ~67 349 frases |
| Validation | ~872 frases |
| Formato | `sentence` (texto) + `label` (0/1) |

**Ejemplos del dataset:**

| sentence | label |
|---|---|
| `"a stirring , funny and finally transporting re-imagining of beauty and the beast"` | 1 (Positivo) |
| `"unflinchingly bleak and desperate"` | 0 (Negativo) |

SST-2 es uno de los benchmarks más utilizados para evaluar modelos de
procesamiento de lenguaje natural (NLP) en tareas de sentimiento.

## Finalidad del experimento

El objetivo **no** es conseguir la mejor accuracy posible, sino ilustrar el
ciclo de vida MLOps completo:

1. **Entrenar** (`train.py`): fine-tuning de DistilBERT sobre SST-2 con
   hiperparámetros configurables. Cada combinación de parámetros produce un
   modelo diferente, lo que justifica el versionado.
2. **Evaluar**: se registran métricas (accuracy, f1, loss, latencia) y un
   reporte de clasificación como artefacto en MLflow.
3. **Registrar** (`register_and_promote.py`): el mejor run se registra en el
   **Model Registry** y se promueve por etapas (None → Staging → Production).
4. **Servir** (Sesión 8): la API carga siempre el modelo en etapa
   `Production` sin referenciar un `run_id` concreto, permitiendo actualizar
   el modelo sin modificar el código de despliegue.

## Estructura del proyecto

| Archivo | Descripción |
|---|---|
| `MLproject` | Manifiesto de ejecución: nombre, entorno, entry points y parámetros |
| `conda.yaml` | Entorno reproducible con conda |
| `requirements.txt` | Dependencias Python (referenciado por conda.yaml) |
| `train.py` | Fine-tuning de DistilBERT + evaluación + registro de modelo pyfunc |
| `register_and_promote.py` | Busca el mejor run, lo registra en Model Registry y promueve etapa |
| `load_production_model.py` | Carga el modelo desde `models:/SentimentAnalyzer/Production` |
| `MLPROJECT_EXPLICACION.md` | Guía docente línea por línea del fichero MLproject |
| `Log.md` | Registro de errores encontrados y soluciones aplicadas |

## Ejecución con MLflow Projects

### Ejecutar con parámetros por defecto

```bash
mlflow run .
```

### Ejecutar con hiperparámetros personalizados

```bash
mlflow run . -P epochs=3 -P learning_rate=3e-5 -P num_train_samples=1000
```

### Ejecución rápida para pruebas

```bash
mlflow run . -P epochs=1 -P num_train_samples=100 -P num_eval_samples=50
```

### Forzar experimento de tracking

```bash
mlflow run . --experiment-name sentiment-analysis-hf -P epochs=2
```

### Ejecutar desde un repo remoto (sin clonar)

```bash
mlflow run https://github.com/usuario/sentiment-project -P epochs=2
```

## Hiperparámetros disponibles

| Parámetro | Default | Descripción |
|---|---|---|
| `model_name` | `distilbert-base-uncased` | Modelo base de HuggingFace |
| `epochs` | `2` | Número de epochs de fine-tuning |
| `learning_rate` | `5e-5` | Tasa de aprendizaje |
| `batch_size` | `16` | Tamaño de batch |
| `max_length` | `128` | Longitud máxima de tokens |
| `weight_decay` | `0.01` | Regularización L2 |
| `num_train_samples` | `500` | Muestras de entrenamiento |
| `num_eval_samples` | `200` | Muestras de validación |
| `confidence_threshold` | `0.5` | Umbral mínimo de confianza |

## Flags de MLflow en esta sesión

| Flag | Descripción |
|---|---|
| `-P <param>=<valor>` | Pasa parámetros al entrypoint definido en `MLproject` |
| `--experiment-name sentiment-analysis-hf` | Fuerza que los runs se registren en el experimento indicado |
| `--env-manager local` | Ejecuta con el entorno local actual en lugar de crear uno aislado |

## Registro y promoción

### Registrar mejor run y mover a Staging

```bash
python register_and_promote.py
```

### Registrar y mover también a Production

```bash
python register_and_promote.py --promote-to-production
```

### Cargar modelo en producción

```bash
python load_production_model.py
```

## Relación con sesiones anteriores y siguientes

```
Sesión 6 (Tracking)
  └─ Crea el experimento "sentiment-analysis-hf"
  └─ Evalúa modelo pre-entrenado (baseline)

Sesión 7 (Projects + Registry) ← ESTAMOS AQUÍ
  └─ Estandariza ejecución con MLproject
  └─ Fine-tuning real de DistilBERT
  └─ Registra y promueve modelo en Model Registry

Sesión 8 (API)
  └─ Carga modelo desde models:/SentimentAnalyzer/Production
  └─ Expone endpoints /predict y /predict/explain

Sesión 9 (Docker)
  └─ Conteneriza la API con Docker
  └─ Orquesta stack completo con Docker Compose

Sesión 10 (Monitorización)
  └─ Detecta data drift y caída de rendimiento
  └─ Decide si el modelo necesita reentrenamiento
```
