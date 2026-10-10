import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import MODEL_PATH

client = TestClient(app)
SAMPLE = {"MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.984, "AveBedrms": 1.0238,
 "Population": 322, "AveOccup": 2.5556, "Latitude": 37.88, "Longitude": -122.23}


def test_health():
 assert client.get("/health").json() == {"status": "ok"}
 
 
def test_invalid_input_is_rejected():
 bad = {**SAMPLE, "MedInc": -5}
 assert client.post("/predict", json=bad).status_code == 422
 
 
@pytest.mark.skipif(not MODEL_PATH.exists(), reason="train the model first: python -m src.train")
def test_predict_returns_a_price():
 r = client.post("/predict", json=SAMPLE)
 assert r.status_code == 200
 assert r.json()["predicted_price_usd"] > 0