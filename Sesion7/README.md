# sentiment-project

Proyecto de la **Sesión 7** para estandarizar ejecución con **MLflow Projects**
y gestionar ciclo de vida con **MLflow Model Registry**.

## Estructura

- `MLproject`: entrada y parámetros del proyecto MLflow.
- `conda.yaml`: entorno principal reproducible con conda.
- `requirements.txt`: dependencias Python instaladas vía pip dentro de conda.
- `train.py`: entrenamiento/evaluación + registro de modelo `pyfunc`.
- `register_and_promote.py`: registro del mejor run y promoción de etapa.
- `load_production_model.py`: carga por alias de etapa `Production`.

## Ejecución con MLflow Projects

Ejecutar con parámetros por defecto:

`mlflow run .`

Ejecutar con parámetros personalizados:

`mlflow run . -P confidence_threshold=0.8 -P batch_size=32`

Para asegurar que los runs queden en el experimento esperado por el script de
registro:

`mlflow run . --experiment-name sentiment-analysis-hf -P num_samples=20`

Ejecutar desde un repo remoto (sin clonar):

`mlflow run https://github.com/usuario/sentiment-project -P num_samples=100`

## Flags de MLflow en esta sesión (Sesión 7)

- `-P <param>=<valor>`
  - Pasa parámetros al entrypoint definido en `MLproject`.
  - Ejemplos usados en esta sesión: `confidence_threshold`, `batch_size`,
    `num_samples`.
- `--experiment-name sentiment-analysis-hf`
  - Fuerza que los runs se registren en el experimento indicado.
- `--env-manager local`
  - Ejecuta con el entorno local actual en lugar de crear uno aislado.

## Registro y promoción

Registrar mejor run y mover a *Staging*:

`python register_and_promote.py`

Registrar y mover también a *Production*:

`python register_and_promote.py --promote-to-production`

## Cargar modelo en producción

`python load_production_model.py`
