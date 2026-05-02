import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "service"))

from main import app
from schemas import PredictPurchaseRequest


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_purchase(client):
    response = client.get("/predict_purchase", params=PredictPurchaseRequest(item_id=98113, hour=12, weekday=3).dict())
    assert response.status_code == 200
