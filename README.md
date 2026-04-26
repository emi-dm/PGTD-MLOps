<table bgcolor="#ffffff" width="100%" style="width: 100%; border-collapse: collapse; border: 1px solid #d0d0d0; background-color: #ffffff !important; font-family: Arial, sans-serif; opacity: 1 !important;">
  <tr bgcolor="#ffffff" style="background-color: #ffffff !important;">
    <td align="center" bgcolor="#ffffff" style="padding: 25px; vertical-align: middle; background-color: #ffffff !important;">
      <div style="background-color: #ffffff !important; padding: 10px;">
        <img src="https://talentodigitalextremadura.com/wp-content/uploads/2025/05/junta-ext-transforma2.png" height="140" style="height: 140px; width: auto; display: inline-block; background-color: #ffffff !important;">
      </div>
      <hr style="border: 0; border-top: 1px solid #dddddd; width: 85%; margin: 20px auto;">
      <table width="100%" bgcolor="#ffffff" style="width: 100%; border-collapse: collapse; background-color: #ffffff !important;">
        <tr bgcolor="#ffffff" style="background-color: #ffffff !important;">
          <td align="center" width="33.3%" bgcolor="#ffffff" style="padding: 10px; background-color: #ffffff !important; border: none;">
            <img src="https://talentodigitalextremadura.com/wp-content/uploads/2024/10/Grafismo-UEx-Color-3.png" height="110" style="height: 110px; width: auto; background-color: #ffffff !important;">
          </td>
          <td align="center" width="33.3%" bgcolor="#ffffff" style="padding: 10px; background-color: #ffffff !important; border: none;">
            <img src="https://talentodigitalextremadura.com/wp-content/uploads/2026/01/logointia.png" height="110" style="height: 110px; width: auto; background-color: #ffffff !important;">
          </td>
          <td align="center" width="33.3%" bgcolor="#ffffff" style="padding: 10px; background-color: #ffffff !important; border: none;">
            <img src="https://talentodigitalextremadura.com/wp-content/uploads/2024/10/Group-59651.png" height="110" style="height: 110px; width: auto; background-color: #ffffff !important;">
          </td>
        </tr>
      </table>
    </td>
  </tr>
</table>

# PGTD MLOps · Plan de Generación de Talento Digital

Repositorio del **curso de MLOps** del **Plan de Generación de Talento Digital (PGTD)**.

Este proyecto recoge, por sesiones, la evolución completa de un flujo MLOps aplicado a análisis de sentimiento:
- seguimiento de experimentos,
- registro y promoción de modelos,
- exposición como API,
- despliegue con contenedores,
- monitorización de drift en producción.

---

## Objetivo académico

Consolidar un pipeline MLOps de extremo a extremo, pasando de experimentación local a operación en producción con monitoreo y señales de reentrenamiento.

---

## Estructura del repositorio

- `Sesion6/` · **MLflow Tracking**
  - `README.md`
  - `experiment_tracking.py`
  - `requirements.txt`
- `Sesion7/` · **MLflow Projects + Model Registry**
  - `README.md`
  - `sentiment-project/` (MLproject, train.py, register_and_promote.py, ...)
- `Sesion8/` · **API de inferencia (FastAPI) usando Registry**
  - `sesion8-api/README.md`
  - `sesion8-api/app.py`
  - `sesion8-api/test_api.py`
- `Sesion9/` · **Despliegue en Docker / Docker Compose**
  - `sesion9-api/README.md`
  - `sesion9-api/app.py`
  - `sesion9-api/docker-compose.yml`
- `Sesion10/` · **Monitorización (Evidently)**
  - `README.md`
  - `generate_drift_report.py`
  - `monitor_production.py`
  - `drift_report.html`

---

## Ruta de aprendizaje por sesión

1. **Sesión 6 — MLflow Tracking**: registrar experimentos, parámetros,
   métricas y artefactos. Evaluar un modelo pre-entrenado con distintas
   configuraciones y comparar runs en la UI de MLflow.
2. **Sesión 7 — MLflow Projects + Model Registry**: estandarizar ejecución
   con `MLproject`, fine-tuning real de DistilBERT en SST-2, y gestionar
   el ciclo de vida del modelo en Model Registry (None → Staging →
   Production).
3. **Sesión 8 — API de inferencia**: publicar el modelo como API REST con
   FastAPI, cargándolo desde `models:/SentimentAnalyzer/Production` sin
   referenciar un run concreto. Incluye endpoint de explicabilidad por
   oclusión de tokens.
4. **Sesión 9 — Docker + Docker Compose**: contenerizar la API y orquestar
   el stack completo (MLflow + API) con Docker Compose, health checks y
   volúmenes persistentes.
5. **Sesión 10 — Monitorización**: detectar data drift y caída de
   rendimiento con Evidently. Scripts diseñados para integración con CI/CD
   que cierran el ciclo MLOps (si hay drift → reentrenar).

---

## Prerrequisitos generales

- Python 3.10+ (recomendado usar entorno virtual)
- `pip`
- Docker + Docker Compose (para Sesión 9)
- MLflow instalado en las sesiones que lo requieren

> Nota: cada sesión tiene su propio `requirements.txt` para mantener independencia y reproducibilidad.

---

## Inicio rápido

Desde la raíz del repositorio (`MLOPs`):

```bash
# Crear entorno virtual (opcional)
python -m venv .venv

# Activar entorno (macOS/Linux)
source .venv/bin/activate
```

Luego entra a la sesión que quieras trabajar e instala dependencias de esa sesión:

```bash
cd Sesion10
pip install -r requirements.txt
```

---

## Ejecución por sesión

### Sesión 6 · Tracking de experimentos

```bash
cd Sesion6
pip install -r requirements.txt
mlflow ui --port 5000
python experiment_tracking.py
```

### Sesión 7 · Projects + Registry

```bash
cd Sesion7/sentiment-project
pip install -r requirements.txt
mlflow run . --experiment-name sentiment-analysis-hf
python register_and_promote.py
python load_production_model.py
```

### Sesión 8 · API con FastAPI + Registry

```bash
cd Sesion8/sesion8-api
pip install -r requirements.txt
mlflow run . -e test
mlflow run . -e serve
```

### Sesión 9 · Despliegue con Docker Compose

```bash
cd Sesion9/sesion9-api
docker compose up -d
python test_stack.py
docker compose down
```

### Sesión 10 · Monitorización y drift

```bash
cd Sesion10
pip install -r requirements.txt
python generate_drift_report.py
python monitor_production.py
```

> `monitor_production.py` puede terminar con código de salida `1` cuando detecta alertas (comportamiento esperado para integrarlo en CI/CD y activar acciones de reentrenamiento).

---

## Artefactos y salidas clave

- **MLflow runs**: en carpetas `mlruns/` de sesiones de tracking/registro.
- **Reporte de drift**: `Sesion10/drift_report.html`.
- **Modelo en producción (registry)**: alias/etapa `Production` en sesiones 7–9.

---

## Recomendación de uso docente

Para seguir el curso en orden, trabaja secuencialmente de **Sesión 6 a Sesión 10**, ya que varias sesiones reutilizan artefactos, configuración de MLflow o estados del Model Registry creados en fases anteriores.

---

## Referencia

Material desarrollado para el **curso de MLOps del Plan de Generación de Talento Digital (PGTD)**.