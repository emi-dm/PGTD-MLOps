# Guía docente: ¿Cómo funciona `MLproject`? (Sesión 7)

Este documento explica el archivo `MLproject` del proyecto de sentimientos y
cómo MLflow lo usa para ejecutar experimentos de forma estandarizada.

## 1) ¿Qué es `MLproject`?

`MLproject` es el **manifiesto de ejecución** del proyecto. Define:

- Nombre del proyecto.
- Entorno reproducible.
- Entrypoints (comandos ejecutables).
- Parámetros y tipos de datos.

En resumen: describe **cómo correr** el proyecto sin ambigüedades.

---

## 2) Archivo actual (referencia)

```yaml
name: sentiment-analysis

# Usar entorno reproducible con conda
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      model_name:
        type: str
        default: distilbert-base-uncased-finetuned-sst-2-english
      batch_size:
        type: int
        default: 16
      confidence_threshold:
        type: float
        default: 0.7
      max_length:
        type: int
        default: 128
      num_samples:
        type: int
        default: 200
    command: "python train.py --model_name {model_name} --batch_size {batch_size} --confidence_threshold {confidence_threshold} --max_length {max_length} --num_samples {num_samples}"
```

---

## 3) Explicación línea por línea

### `name: sentiment-analysis`

Identificador lógico del proyecto en MLflow Projects.

### `conda_env: conda.yaml`

Le dice a MLflow que el entorno de ejecución se crea desde `conda.yaml`.
Esto garantiza reproducibilidad de dependencias y versión de Python.

### `entry_points.main`

Define el entrypoint principal llamado `main` (el que corre por defecto con
`mlflow run .`).

### `parameters`

Cada parámetro tiene:

- `type`: validación de tipo (`str`, `int`, `float`).
- `default`: valor por defecto si no se pasa `-P`.

### `command`

Comando final que MLflow ejecuta, sustituyendo placeholders:

- `{model_name}`
- `{batch_size}`
- `{confidence_threshold}`
- `{max_length}`
- `{num_samples}`

Por ejemplo, `python train.py ...` se construye automáticamente con defaults o
con valores recibidos por CLI.

---

## 4) Flujo real al ejecutar `mlflow run .`

Cuando ejecutas `mlflow run .`, ocurre esto:

1. MLflow lee `MLproject`.
2. Detecta `conda_env: conda.yaml`.
3. Crea/activa el entorno de conda reproducible.
4. Resuelve parámetros (defaults + overrides `-P`).
5. Ejecuta el `command` del entrypoint `main`.
6. `train.py` registra parámetros, métricas, artefactos y modelo en MLflow.

---

## 5) Ejemplos prácticos para clase

### Ejecutar con valores por defecto

`mlflow run .`

### Cambiar parámetros sin tocar código

`mlflow run . -P confidence_threshold=0.8 -P batch_size=32`

### Forzar experimento de tracking

`mlflow run . --experiment-name sentiment-analysis-hf -P num_samples=100`

---

## 6) Ventaja pedagógica clave

`MLproject` separa **configuración de ejecución** del **código Python**. Esto
permite enseñar buenas prácticas MLOps:

- Reproducibilidad.
- Trazabilidad.
- Parametrización.
- Portabilidad entre equipos.

---

## 7) Errores comunes y cómo evitarlos

- Definir mal el entorno (`python_env` vs `conda_env`): usar uno coherente con
  la estrategia del proyecto.
- Comandos multilinea ambiguos: preferir `command` claro y verificable.
- Desalinear experimento de tracking: usar `--experiment-name` cuando aplique.

---

## 8) Mensaje para alumnos

Piensa en `MLproject` como el "README ejecutable" del experimento: no solo
explica qué hacer, sino que **lo ejecuta de forma estándar**.
