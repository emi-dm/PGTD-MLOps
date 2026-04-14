"""Pruebas unitarias para la API de la Sesión 8."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as app_module


class FakeModel:
    """Modelo de prueba para evitar dependencia del registry en tests."""

    def predict(self, model_input):
        return [1 if "good" in text.lower() else 0 for text in model_input["text"]]


class SentimentApiTests(unittest.TestCase):
    """Suite de pruebas para endpoints de salud y predicción."""

    def setUp(self) -> None:
        app_module._model_cache = None
        self.client = TestClient(app_module.app)

    def test_health_endpoint(self) -> None:
        """El endpoint de salud debe responder estado y URI de modelo."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("model_uri", payload)

    def test_predict_endpoint_success(self) -> None:
        """La predicción debe devolver una etiqueta por cada texto."""
        with patch("app.load_model", return_value=FakeModel()):
            response = self.client.post(
                "/predict",
                json={"texts": ["Good product", "Bad service"]},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["predictions"], [1, 0])

    def test_predict_endpoint_validation(self) -> None:
        """Una lista vacía debe fallar por validación del esquema."""
        response = self.client.post("/predict", json={"texts": []})
        self.assertEqual(response.status_code, 422)

    def test_predict_explain_endpoint_success(self) -> None:
        """El endpoint explain debe devolver tokens influyentes por texto."""
        with patch("app.load_model", return_value=FakeModel()):
            response = self.client.post(
                "/predict/explain?top_k=2",
                json={"texts": ["Good product", "Bad service"]},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertIn("predictions", payload)
        self.assertEqual(payload["predictions"], [1, 0])
        self.assertIn("explanations", payload)
        self.assertEqual(len(payload["explanations"]), 2)

        first = payload["explanations"][0]
        self.assertEqual(first["text"], "Good product")
        self.assertEqual(first["prediction"], 1)
        self.assertIn("top_tokens", first)
        self.assertGreaterEqual(len(first["top_tokens"]), 1)
        self.assertLessEqual(len(first["top_tokens"]), 2)
        self.assertIn("token", first["top_tokens"][0])
        self.assertIn("influence", first["top_tokens"][0])

    def test_predict_explain_top_k_validation(self) -> None:
        """top_k inválido debe retornar error 422."""
        with patch("app.load_model", return_value=FakeModel()):
            response = self.client.post(
                "/predict/explain?top_k=0",
                json={"texts": ["Good product"]},
            )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
