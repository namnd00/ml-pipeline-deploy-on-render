"""
Rest API module test
"""
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
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
    assert r.json() == {"message": "MLOps - Render API!"}


def test_get_failed(test_client):
    r = test_client.get("/wrong_url")
    assert r.status_code != 200


def test_predict_1(test_client):
    user_input = MockUser(
        age=32,
        hours_per_week=60,
        workclass='Private',
        education='Some-college',
        marital_status='Married-civ-spouse',
        occupation='Exec-managerial',
        relationship='Husband',
        race='White',
        sex='Male',
        native_country='United-States'
    )
    response = test_client.post("/", json=user_input.dict())
    assert response.status_code == 200
    assert response.json()['class_name'] == ">50K"


def test_predict_2(test_client):

    user_input = MockUser(
        age=19,
        hours_per_week=40,
        workclass='Private',
        education='HS-grad',
        marital_status='Never-married',
        occupation='Other-service',
        relationship='Own-child',
        race='Black',
        sex='Male',
        native_country='United-States'
    )
    response = test_client.post("/", json=user_input.dict())
    assert response.status_code == 200
    assert response.json()['class_name'] == "<=50K"


def test_predict_3(test_client):
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
