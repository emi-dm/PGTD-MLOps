# Sesión 9 — Despliegue con Docker y Docker Compose

## Objetivo de la sesión

**Contenerizar** la API de la Sesión 8 y orquestar el stack completo
(MLflow + API) con **Docker Compose**. Hasta ahora ejecutábamos todo en
local; ahora empaquetamos el sistema en contenedores reproducibles que
funcionan en cualquier máquina con Docker instalado.

Esta sesión responde a la pregunta: *"¿Cómo despliego mi API en un entorno
que sea idéntico en desarrollo, testing y producción?"*

## Conceptos clave

### ¿Qué es Docker?

[Docker](https://www.docker.com/) es una plataforma de **contenedores**. Un
contenedor es una unidad ligera y aislada que empaqueta:

- El código de la aplicación.
- Sus dependencias (librerías, runtime).
- La configuración del sistema operativo.

A diferencia de una máquina virtual, un contenedor comparte el kernel del
sistema anfitrión, por lo que es mucho más ligero y rápido.

**Analogía**: si una máquina virtual es un ordenador completo dentro de otro
ordenador, un contenedor es una carpeta aislada con todo lo necesario para
ejecutar una aplicación.

### ¿Qué es una imagen Docker?

Una **imagen** es la plantilla para crear contenedores. Se define en un
`Dockerfile` y contiene:

- El sistema operativo base (por ejemplo, `python:3.13-slim`).
- Las dependencias instaladas.
- El código de la aplicación.
- El comando de arranque.

Un **contenedor** es una instancia en ejecución de una imagen. Puedes crear
muchos contenedores a partir de la misma imagen.

### ¿Qué es un Dockerfile?

Un `Dockerfile` es un archivo de texto con instrucciones paso a paso para
construir una imagen:

```dockerfile
FROM python:3.13-slim          # Imagen base
WORKDIR /app                   # Directorio de trabajo
COPY requirements.txt .        # Copiar dependencias
RUN pip install -r requirements.txt  # Instalar dependencias
COPY . .                       # Copiar código
EXPOSE 8000                    # Puerto expuesto
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Cada instrucción crea una **capa** en la imagen. Docker cachea las capas,
por lo que si solo cambia el código (no las dependencias), el rebuild es
muy rápido.

### ¿Qué es Docker Compose?

[Docker Compose](https://docs.docker.com/compose/) es una herramienta para
definir y ejecutar **múltiples contenedores** como un stack unitario. Se
define en un fichero `docker-compose.yml`:

```yaml
services:
  mlflow:        # Servidor MLflow
    image: python:3.11-slim
    ports:
      - "5001:5000"
  api:           # API de sentimiento
    build: .
    ports:
      - "8000:8000"
    depends_on:
      mlflow:
        condition: service_healthy
```

Con `docker compose up -d` se levantan ambos servicios automáticamente,
en el orden correcto y con la red configurada.

### ¿Qué es un health check?

Un **health check** es una verificación periódica de que un servicio está
funcionando. Docker ejecuta el comando definido y marca el contenedor como
`healthy` o `unhealthy`:

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
  interval: 10s
  timeout: 5s
  retries: 8
```

En nuestro caso, el servicio `api` usa `depends_on` con
`condition: service_healthy` para esperar a que MLflow esté listo antes de
arrancar.

### ¿Qué es una red Docker?

Docker Compose crea automáticamente una **red** virtual donde los
contenedores se comunican por nombre de servicio. Por eso la API puede
referenciar a MLflow como `http://mlflow:5000` (el nombre del servicio)
en lugar de una IP.

### ¿Qué es un volumen?

Un **volumen** es un almacenamiento persistente que sobrevive a la
destrucción de contenedores. En nuestro caso:

```yaml
volumes:
  mlflow_data:    # Persiste la base de datos y artefactos de MLflow
```

Sin volumen, al hacer `docker compose down` se perderían todos los runs,
modelos y el registry.

### ¿Qué es `.dockerignore`?

Un `.dockerignore` es similar a `.gitignore`: lista archivos y carpetas
que **no** deben copiarse al construir la imagen. Esto reduce el tamaño
de la imagen y evita copiar datos sensibles o innecesarios.

## Relación con la Sesión 8

En la Sesión 8 creamos la API con FastAPI y la ejecutábamos en local. En
esta sesión:

1. **Misma API**: el `app.py` es prácticamente idéntico al de Sesión 8.
2. **Empaquetado**: la API se empaqueta en una imagen Docker con todas sus
   dependencias.
3. **Orquestación**: Docker Compose levanta MLflow y la API juntos, con
   red interna, health checks y volúmenes persistentes.
4. **Variables de entorno**: la configuración (URI del modelo, puerto, etc.)
   se pasa por variables de entorno, no por código.

## Archivos incluidos

| Archivo | Descripción |
|---|---|
| `app.py` | API FastAPI (misma que Sesión 8, adaptada para Docker) |
| `requirements.txt` | Dependencias del servicio API |
| `Dockerfile` | Instrucciones para construir la imagen de la API |
| `docker-compose.yml` | Stack completo: servicio `mlflow` + servicio `api` |
| `test_stack.py` | Prueba de integración end-to-end del stack |
| `README.md` | Este archivo |

## Stack Docker Compose

El `docker-compose.yml` define tres servicios:

### Servicio `mlflow`

- Imagen: `python:3.11-slim`
- Arranca un servidor MLflow con backend SQLite.
- Expone el puerto 5000 interno como 5001 en el host.
- Usa un volumen (`mlflow_data`) para persistir datos.
- Health check: verifica que `/health` responde.

### Servicio `api`

- Se construye desde el `Dockerfile` local.
- Expone el puerto 8000.
- Se conecta a MLflow vía `MLFLOW_TRACKING_URI=http://mlflow:5000`.
- Espera a que MLflow esté `healthy` antes de arrancar.
- Reinicio automático (`restart: unless-stopped`).

### Servicio `monitor` (Sesión 10)

- Imagen: `python:3.11-slim`
- Monta `Sesion10/monitor_production.py` como volumen de solo lectura.
- Ejecuta monitorización en modo `live` cada 120 segundos.
- Envía textos de SST-2 al endpoint `/predict` de la API.
- Compara predicciones con ground truth y detecta drift con Evidently.
- Espera a que la API esté arrancada antes de iniciar.
- Reinicio automático (`restart: unless-stopped`).

Este servicio **cierra el ciclo MLOps**: conecta la monitorización de la
Sesión 10 con el despliegue de la Sesión 9, detectando cuándo el modelo
necesita reentrenarse.

## Ejecución

### Build de imagen individual

```bash
cd Sesion9/sesion9-api
docker build -t sentiment-api:1.0 .
```

### Ejecución directa (sin Compose)

```bash
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MODEL_NAME=SentimentAnalyzer \
  -e MODEL_STAGE=Production \
  sentiment-api:1.0
```

### Stack completo con Docker Compose

```bash
docker compose up -d          # Levantar stack (mlflow + api + monitor)
docker compose logs -f        # Ver todos los logs
docker compose logs -f api    # Logs solo de la API
docker compose logs -f monitor  # Logs del monitor de drift
docker compose ps             # Estado de servicios
docker compose up -d --build api  # Rebuild solo la API
docker compose down           # Parar stack
docker compose down -v        # Parar y borrar volúmenes
```

### Validación automática

Con el stack arriba:

```bash
python test_stack.py
```

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio y modelo |
| POST | `/predict` | Predicción de sentimiento por lote |
| POST | `/predict/explain?top_k=3` | Predicción + tokens influyentes |

## Nota sobre Model Registry

La API busca el modelo en `models:/SentimentAnalyzer/Production`.
Si el MLflow del stack está vacío, primero registra y promueve el modelo
desde `Sesion7/sentiment-project` con `register_and_promote.py`.

## Flags de MLflow en Docker Compose

Se usan estos flags en el servicio `mlflow` de `docker-compose.yml`:

| Flag | Descripción |
|---|---|
| `--host 0.0.0.0` | Expone el servidor en todas las interfaces del contenedor |
| `--port 5000` | Puerto interno del servidor MLflow |
| `--allowed-hosts ...` | Hosts permitidos (protección contra DNS rebinding) |
| `--backend-store-uri sqlite:////mlflow/mlflow.db` | Base de datos para metadatos de tracking/registry |
| `--artifacts-destination /mlflow/artifacts` | Ruta de almacenamiento de artefactos |
| `--serve-artifacts` | Proxy HTTP de artefactos para clientes remotos |

## Relación con sesiones anteriores y siguientes

```
Sesión 6 (Tracking)
  └─ Experimentos registrados en MLflow

Sesión 7 (Projects + Registry)
  └─ Modelo fine-tuneado y en Production

Sesión 8 (API)
  └─ API FastAPI sirviendo el modelo en local

Sesión 9 (Docker) ← ESTAMOS AQUÍ
  └─ API + MLflow empaquetados en contenedores
  └─ Stack orquestado con Docker Compose
  └─ Entorno reproducible en cualquier máquina

Sesión 10 (Monitorización)
  └─ Detecta drift en datos de producción
  └─ Señala cuándo reentrenar el modelo
```
