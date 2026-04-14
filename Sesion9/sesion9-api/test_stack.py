"""Prueba de integración del stack Docker (MLflow + API).

Verifica disponibilidad de servicios y realiza predicciones sobre el
endpoint /predict usando el contrato actual de la API (lista de textos).
"""

from __future__ import annotations

import time
from typing import Final

import requests

API_URL: Final[str] = "http://localhost:8000"
MLFLOW_URL: Final[str] = "http://localhost:5001"


def wait_for_service(url: str, service_name: str, max_retries: int = 30) -> bool:
    """Espera hasta que un servicio responda con HTTP 200 en /health.

    Args:
        url: URL base del servicio.
        service_name: Nombre legible del servicio para logs.
        max_retries: Número máximo de reintentos.

    Returns:
        True si el servicio estuvo disponible durante los reintentos,
        False en caso contrario.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                print(f"{service_name} listo")
                return True
        except requests.RequestException:
            pass

        print(f"Esperando a {service_name}... ({attempt}/{max_retries})")
        time.sleep(2)

    return False


def run_prediction_tests() -> None:
    """Ejecuta predicciones de ejemplo y valida forma de respuesta."""
    test_inputs = [
        ["This is amazing!"],
        ["I hate this product."],
        ["Not sure about this."],
    ]

    print("\n--- Tests de predicción ---")
    for texts in test_inputs:
        response = requests.post(
            f"{API_URL}/predict",
            json={"texts": texts},
            timeout=10,
        )
        response.raise_for_status()

        payload = response.json()
        predictions = payload.get("predictions", [])
        assert isinstance(
            predictions, list), "'predictions' debe ser una lista"
        assert len(predictions) == len(
            texts), "La cantidad de predicciones no coincide"

        print(f"✓ {texts[0][:40]!r} -> {predictions[0]}")


if __name__ == "__main__":
    print("=== Test del stack completo (Sesión 9) ===")

    if not wait_for_service(MLFLOW_URL, "MLflow"):
        raise SystemExit("MLflow no estuvo disponible a tiempo")

    if not wait_for_service(API_URL, "API"):
        raise SystemExit("API no estuvo disponible a tiempo")

    run_prediction_tests()
    print("\nStack funcionando correctamente")
