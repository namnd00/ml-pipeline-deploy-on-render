"""
Rest API module test
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
import sys
sys.path.append("../")
from main import app


@pytest.fixture(scope="module")
def test_client():
    with TestClient(app) as client:
        yield client


class MockUser(BaseModel):
    age: int
    hours_per_week: int
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str


def test_get(test_client):
    r = test_client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "MLOps - heroku fastapi!"}


def test_get_failed(test_client):
    r = test_client.get("/wrong_url")
    assert r.status_code != 200


def test_predict(test_client):
    # Simulate a valid user input (note: these values may need to be adjusted
    # based on the actual model)
    user_input = MockUser(
        age=40,
        hours_per_week=40,
        workclass='Private',
        education='Bachelors',
        marital_status='Married-civ-spouse',
        occupation='Exec-managerial',
        relationship='Husband',
        race='White',
        sex='Male',
        native_country='United-States'
    )

    # Ensure the response is successful
    response = test_client.post("/", json=user_input.dict())
    assert response.status_code == 200

    # Ensure the response has expected keys
    response_json = response.json()
    assert set(response_json.keys()) == {'prediction', 'class_name', 'success'}


def test_predict_failed(test_client):
    user_input = MockUser(
        age=40,
        hours_per_week=40,
        workclass='Private',
        education='Bachelors',
        marital_status='ERROR',
        occupation='ERROR',
        relationship='Husband',
        race='White',
        sex='Male',
        native_country='United-States'
    )
    r = test_client.post("/", json=user_input.dict())
    assert r.status_code == 422
