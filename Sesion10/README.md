# Sesión 10 — Monitorización de modelos en producción

## Objetivo de la sesión

Aprender a **detectar cuándo un modelo en producción necesita ser
reentrenado**. Un modelo desplegado no es estático: el mundo cambia, los
datos cambian y el rendimiento del modelo se degrada con el tiempo. Esta
sesión cierra el **ciclo MLOps completo** añadiendo el último eslabón:
la monitorización continua.

```
Sesión 6: Tracking → Sesión 7: Train + Registry → Sesión 8: API → Sesión 9: Docker → Sesión 10: Monitorización
```

## Conceptos clave

### ¿Por qué se degrada un modelo?

Un modelo de ML aprende patrones de los datos de entrenamiento. Si los datos
del mundo real cambian después del despliegue, el modelo pierde precisión.
Esto ocurre por dos razones principales:

### Data Drift (deriva de datos)

**Data drift** ocurre cuando la **distribución de los datos de entrada**
cambia respecto a los datos con los que se entrenó el modelo.

**Ejemplo concreto**: si el modelo se entrenó con reseñas de películas
cortas y neutras, pero en producción recibe reseñas largas y muy
emotivas, las características estadísticas de los datos han cambiado
(longitud media, vocabulario, tono).

Se detecta comparando **features estadísticas** entre datos de referencia
(los que se usaron para entrenar/validar) y datos de producción actuales:

| Feature | Referencia | Producción | ¿Drift? |
|---|---|---|---|
| `text_length` (media) | 80 caracteres | 120 caracteres | ✅ Sí |
| `word_count` (media) | 15 palabras | 22 palabras | ✅ Sí |
| `avg_word_len` (media) | 4.2 | 4.3 | ❌ No |

Si muchas features derivan significativamente, hay **data drift**.

### Concept Drift (deriva de concepto)

**Concept drift** ocurre cuando la **relación entre inputs y outputs**
cambia. El modelo sigue viendo datos similares, pero lo que antes era
positivo ahora puede ser negativo (o viceversa).

**Ejemplo**: antes "increíble" siempre indicaba sentimiento positivo, pero
en un contexto irónico puede ser negativo. El concepto de "positivo" ha
cambiado.

En esta sesión nos centramos en **data drift** (más fácil de detectar
estadísticamente), pero la caída de accuracy también captura efectos de
concept drift.

### ¿Qué es Evidently?

[Evidently](https://www.evidentlyai.com/) es una librería open-source de
Python para **monitorización de modelos ML**. Proporciona:

- **Reportes de drift**: comparan distribuciones estadísticas entre
  datasets de referencia y producción.
- **Reportes de calidad**: evalúan métricas de rendimiento del modelo.
- **Reportes HTML**: visualizaciones interactivas para explorar los
  resultados.

Evidently usa tests estadísticos (como Kolmogorov-Smirnov o Chi-cuadrado)
para determinar si las diferencias entre distribuciones son significativas.

### ¿Qué es un dataset de referencia?

El **dataset de referencia** (reference) es el conjunto de datos que
representa el comportamiento "esperado" o "normal". Se usa como línea base
para comparar con los datos de producción.

En nuestro caso, usamos el split de validación de SST-2 como referencia,
porque es el mismo conjunto con el que evaluamos el modelo antes de
desplegarlo.

### ¿Qué es un dataset de producción?

El **dataset de producción** (current/production) son los datos reales que
el modelo está procesando en el mundo real. En un sistema real, estos datos
se recogen de los logs de la API.

En esta sesión **simulamos** datos de producción usando un tramo distinto
del mismo dataset SST-2, para demostrar la detección de drift sin necesidad
de un sistema real en producción.

### ¿Qué es un umbral de alerta?

Un **umbral de alerta** es el valor a partir del cual se considera que hay
un problema. En nuestro monitor:

| Umbral | Valor | Significado |
|---|---|---|
| `DRIFT_THRESHOLD` | 0.20 (20%) | Si más del 20% de las columnas tienen drift, alertar |
| `ACCURACY_DROP_THRESHOLD` | 0.05 (5%) | Si la accuracy baja más de 5 puntos, alertar |

Si se supera **cualquier** umbral, el script recomienda reentrenamiento.

### ¿Qué es coverage en monitorización?

En el contexto de monitorización, **coverage** se refiere a la proporción
de datos de producción que el modelo puede procesar con confianza
suficiente. Un coverage bajo en producción puede indicar que los datos
han cambiado tanto que el modelo ya no "entiende" las entradas.

### ¿Qué es un pipeline de CI/CD para ML?

Un **pipeline de CI/CD para ML** (MLOps pipeline) automatiza el ciclo
completo:

```
Código → Entrenar → Evaluar → Registrar → Desplegar → Monitorizar
                                                        ↓
                                                  ¿Drift detectado?
                                                        ↓
                                              Re-entrenar (vuelve al inicio)
```

`monitor_production.py` está diseñado para integrarse en este ciclo:
devuelve código de salida `1` si se necesita reentrenamiento, lo que
permite que un sistema de CI/CD (GitHub Actions, Jenkins, etc.) lance
automáticamente el reentrenamiento.

## Relación con sesiones anteriores

| Sesión | Entrega | Conexión con Sesión 10 |
|---|---|---|
| **6** | Experiment tracking | Métricas base de referencia (accuracy, f1) |
| **7** | Modelo fine-tuneado + Registry | El modelo cuyo drift se monitoriza |
| **8** | API de inferencia | Los endpoints que generan datos de producción |
| **9** | Docker Compose | El stack donde se desplegaría la monitorización |
| **10** | **Drift detection + alertas** | **Decisión de reentrenamiento** |

## Dataset utilizado

Se usa **SST-2** (mismo que Sesiones 6-7) para simular datos de referencia
y producción:

| Dataset | Split SST-2 | Uso |
|---|---|---|
| Referencia | `validation[:300]` | Línea base del modelo |
| Producción | `validation[300:600]` o `test[:300]` | Datos "nuevos" simulados |

### Features extraídas para análisis de drift

Se extraen **features textuales derivadas** de cada frase para comparar
distribuciones:

| Feature | Descripción | ¿Por qué importa? |
|---|---|---|
| `text_length` | Longitud del texto en caracteres | Reseñas más largas pueden indicar otro dominio |
| `word_count` | Número de palabras | Cambio en la verbosidad de los usuarios |
| `avg_word_len` | Longitud media de palabra | Vocabulario más técnico o más simple |
| `excl_count` | Número de signos de exclamación | Mayor emotividad en los textos |
| `question_count` | Número de signos de interrogación | Cambio en el tipo de texto |
| `upper_ratio` | Proporción de mayúsculas | Textos más enfáticos o de otro estilo |

## Archivos incluidos

| Archivo | Descripción |
|---|---|
| `generate_drift_report.py` | Genera un informe HTML visual de drift con Evidently |
| `monitor_production.py` | Script de monitorización programática con alertas (diseñado para cron/CI) |
| `requirements.txt` | Dependencias del módulo |
| `drift_report.html` | Informe generado (ejemplo de salida) |

## Ejecución

### 1. Generar informe HTML de drift

```bash
cd Sesion10
pip install -r requirements.txt
python generate_drift_report.py
```

Abre `drift_report.html` en el navegador para ver el análisis visual con
gráficas de distribuciones, tests estadísticos y resumen de drift.

### 2. Ejecutar monitorización programática

```bash
python monitor_production.py
```

Salida JSON con alertas y recomendación de reentrenamiento:

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
  "needs_retraining": true
}
```

### 3. Integración con CI/CD

`monitor_production.py` devuelve código de salida `1` si se detecta que
el modelo necesita reentrenamiento:

```bash
python monitor_production.py || echo "Reentrenamiento necesario"
```

En un pipeline real de GitHub Actions:

```yaml
- name: Check model drift
  run: python Sesion10/monitor_production.py

- name: Retrain model
  if: failure()
  run: mlflow run Sesion7/sentiment-project --experiment-name sentiment-analysis-hf
```

## Umbrales de alerta

| Métrica | Umbral | Significado |
|---|---|---|
| `DRIFT_THRESHOLD` | 0.20 (20%) | Proporción de columnas con drift significativo |
| `ACCURACY_DROP_THRESHOLD` | 0.05 (5%) | Caída de accuracy respecto a referencia |

## Dependencias clave

| Librería | Función |
|---|---|
| **Evidently** | Generación de reportes de drift y monitorización |
| **Transformers** | Pipeline de inferencia para generar predicciones |
| **Datasets** | Carga del dataset SST-2 desde HuggingFace |
| **Pandas / NumPy** | Manipulación de datos y cálculos estadísticos |

## Ciclo MLOps completo

Esta sesión cierra el ciclo. El flujo completo del curso es:

```
┌─────────────────────────────────────────────────────────────────┐
│                     CICLO MLOPS COMPLETO                        │
│                                                                 │
│  Sesión 6: TRACKING                                             │
│  └─ Registrar experimentos en MLflow                            │
│  └─ Comparar runs, métricas, parámetros                         │
│                                                                 │
│  Sesión 7: TRAINING + REGISTRY                                  │
│  └─ Fine-tuning de DistilBERT en SST-2                          │
│  └─ Estandarizar ejecución con MLproject                        │
│  └─ Registrar modelo en Model Registry                          │
│  └─ Promover a Production                                       │
│                                                                 │
│  Sesión 8: API                                                  │
│  └─ Servir modelo como API REST con FastAPI                     │
│  └─ Cargar desde models:/SentimentAnalyzer/Production           │
│  └─ Endpoints /predict y /predict/explain                       │
│                                                                 │
│  Sesión 9: DOCKER                                               │
│  └─ Contenerizar API con Docker                                 │
│  └─ Orquestar stack con Docker Compose                          │
│  └─ Entorno reproducible en cualquier máquina                   │
│                                                                 │
│  Sesión 10: MONITORIZACIÓN                                      │
│  └─ Detectar data drift con Evidently                           │
│  └─ Monitorizar caída de accuracy                               │
│  └─ Alertar y recomendar reentrenamiento                        │
│  └─ Integrar con CI/CD para cerrar el ciclo                     │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│  Si hay drift → Volver a Sesión 7 y reentrenar                  │
└─────────────────────────────────────────────────────────────────┘
```
