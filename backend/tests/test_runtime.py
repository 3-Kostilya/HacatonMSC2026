"""Small end-to-end checks for the packaged runtime, without user data."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


class RuntimeSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = TemporaryDirectory(prefix="backend-runtime-test-")
        cls.previous_data_dir = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = cls.temporary.name
        source = BACKEND_ROOT / "data" / "model" / "current"
        destination = Path(cls.temporary.name) / "model" / "current"
        destination.mkdir(parents=True)
        for filename in ("model.cbm", "model_metadata.json"):
            shutil.copy2(source / filename, destination / filename)

        from app.main import app

        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)
        if cls.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = cls.previous_data_dir
        cls.temporary.cleanup()

    def test_health_and_no_implicit_sample_references(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        capabilities = self.client.get("/api/raw/capabilities")
        self.assertEqual(capabilities.status_code, 200)
        self.assertTrue(capabilities.json()["modelReady"])
        self.assertFalse(capabilities.json()["referencesReady"])
        self.assertEqual(self.client.get("/api/raw/import/not-a-batch-id").status_code, 404)

    def test_raw_csv_import_with_explicit_references(self) -> None:
        journal = (
            "ид_события,ид_канала_данных,дата,время,тревожное,значение_датчика\n"
            "e1,smoke-1,2025-01-01,00:00:00,false,Норма\n"
            "e2,smoke-1,2025-01-09,00:00:00,false,Норма\n"
        )
        channels = (
            "ид_канала_данных,тип_инж_системы,тип_датчика,"
            "тег_инженерной_системы,название_датчика\n"
            "smoke-1,test,Датчик дыма,,Test smoke\n"
        )
        objects = (
            "ид_объект,иерархия_уровень,родитель,вид_объекта,"
            "диспетчерское_название_объекта\n"
            "building-1,1,,test,Test building\n"
        )
        response = self.client.post(
            "/api/raw/import",
            files={
                "journal": ("journal.csv", journal.encode("utf-8"), "text/csv"),
                "channels": ("channels.csv", channels.encode("utf-8"), "text/csv"),
                "objects": ("objects.csv", objects.encode("utf-8"), "text/csv"),
            },
        )
        self.assertEqual(response.status_code, 202, response.text)
        batch_id = response.json()["batchId"]
        status = self.client.get(f"/api/raw/import/{batch_id}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["status"], "processed", status.json())
        self.assertEqual(status.json()["rowsCount"], 2)
        self.assertEqual(status.json()["forecastsCount"], 1)
        sensors = self.client.get("/api/sensors")
        self.assertEqual(sensors.status_code, 200)
        self.assertEqual(len(sensors.json()), 1)
        self.assertEqual(sensors.json()[0]["id"], "smoke-1")
        self.assertEqual(sensors.json()[0]["predictionStatus"], "scored")
        self.assertIsNotNone(sensors.json()[0]["researchScore"])
        forecast = self.client.get("/api/sensors/smoke-1/assessment").json()
        self.assertEqual(forecast["policyVersion"], "round7-raw-upload-v1")
        self.assertEqual(forecast["predictionTime"], "2025-01-08T23:00:00")


if __name__ == "__main__":
    unittest.main()
