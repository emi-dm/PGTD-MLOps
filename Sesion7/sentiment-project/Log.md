# Log de ejecución y pruebas

## 2026-04-14 — Error 1 (MLflow Project no inicia)

- **Comando probado:** `mlflow run . -P num_samples=20`
- **Error observado:**
  - `TypeError: mlflow.utils.environment._PythonEnv() argument after ** must be a mapping, not str`
- **Motivo:**
  - En `MLproject`, `python_env` apuntaba a `requirements.txt`, pero en la versión actual de MLflow `python_env` espera un archivo YAML con estructura de entorno (mapping), no un TXT de dependencias.
- **Solución aplicada:**
  - Se actualizó `MLproject` para usar `python_env: python_env.yaml`.
  - Se creó `python_env.yaml` con `build_dependencies` y `dependencies` referenciando `requirements.txt`.
- **Estado:** corregido y listo para reintento.

## 2026-04-14 — Error 2 (dependencia implícita de pyenv)

- **Comando probado:** `mlflow run . -P num_samples=20`
- **Error observado:**
  - `FileNotFoundError: [Errno 2] No such file or directory: 'None'`
  - Stacktrace en `mlflow.utils.virtualenv` intentando resolver Python con `pyenv`.
- **Motivo:**
  - MLflow intentó crear/gestionar un entorno aislado y en este equipo no hay
    `pyenv` configurado para ese flujo.
- **Solución aplicada:**
  - Ejecutar el proyecto con entorno local ya preparado: `--env-manager local`.
- **Estado:** corregido con workaround de ejecución local.

## 2026-04-14 — Error 3 (mismatch de experimento + comando multilínea)

- **Comando probado:** `mlflow run . --env-manager local -P num_samples=20`
- **Errores observados:**
  - `MlflowException: active experiment ID does not match environment run ID`
  - `bash: --model_name: command not found` (y flags similares)
- **Motivo:**
  - `train.py` hacía `mlflow.set_experiment(...)` dentro de un run ya creado
    por MLflow Projects.
  - El `command` de `MLproject` estaba en formato multilínea y el shell terminó
    interpretando flags como comandos separados.
- **Solución aplicada:**
  - En `train.py`, solo se llama `mlflow.set_experiment(...)` cuando **no** existe
    la variable `MLFLOW_RUN_ID`.
  - En `MLproject`, se reescribió `command` en una sola línea.
- **Estado:** corregido y listo para reintento.

## 2026-04-14 — Error 4 (sin runs para registrar)

- **Comando probado:** `python register_and_promote.py --promote-to-production`
- **Error observado:**
  - `ValueError: No hay runs disponibles para registrar.`
- **Motivo:**
  - El run exitoso ejecutado por `mlflow run` quedó en el experimento por
    defecto, mientras que `register_and_promote.py` busca en
    `sentiment-analysis-hf`.
- **Solución aplicada:**
  - Re-ejecutar `mlflow run` especificando explícitamente
    `--experiment-name sentiment-analysis-hf`.
  - Mantener consistencia entre ejecución y búsqueda del mejor run.
- **Estado:** corregido con re-ejecución dirigida al experimento correcto.
