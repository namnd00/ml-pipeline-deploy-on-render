"""
This script for Rest API
"""
import logging
from fastapi import FastAPI, HTTPException, status
from cachetools import cached, TTLCache
from pydantic import BaseModel
from typing import Literal
from starter.ml.data import get_categorical_features, process_data
from starter.ml.model import inference
import pandas as pd
from typing import Any
import numpy as np
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Create a new instance of the FastAPI class
app = FastAPI(title="MLOps app", version='1.0.0')


class User(BaseModel):
    """Define User Schema

    Attributes:
        - age, hours_per_week: continous columns
        - workclass, education, marital_status, occupation, relationship, race, sex, native_country:
            categorical columns.
    """
    age: int
    hours_per_week: int
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
        'Local-gov', 'Self-emp-inc', 'Without-pay']
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    marital_status: Literal[
        'Never-married', 'Married-civ-spouse', 'Divorced',
        'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
        'Widowed']
    occupation: Literal[
        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
        'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
        'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
        'Craft-repair', 'Protective-serv', 'Armed-Forces',
        'Priv-house-serv']
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child',
        'Unmarried', 'Other-relative']
    race: Literal[
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
        'Other']
    sex: Literal['Male', 'Female']
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']


class PredictionResponse(BaseModel):
    """Defining the response body of an inference API"""
    prediction: Any
    class_name: Any
    success: bool


# define ttl cache, maxsize is the maximum number of models and ttl is the time-to-live in seconds
cache = TTLCache(maxsize=100, ttl=300)


@cached(cache)
def get_model():
    """Get the model and save to cache"""
    logger.info("Loading model...")
    model_path, encoder_path, lb_path = \
        ("./deploy/model.joblib", "./deploy/encoder.joblib", "./deploy/lb.joblib")
    # load model, encoder and label encoder
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)

    return model, encoder, lb


@app.get("/")
async def get_items():
    return {"message": "MLOps - heroku fastapi!"}


@app.post("/")
async def predict(user_data: User):
    """Predict method"""
    logger.info(f"Received user data: {user_data}")

    model, encoder, lb = get_model()
    array = np.array([[
                     user_data.age,
                     user_data.hours_per_week,
                     user_data.workclass,
                     user_data.education,
                     user_data.marital_status,
                     user_data.occupation,
                     user_data.relationship,
                     user_data.race,
                     user_data.sex,
                     user_data.native_country
                     ]])
    temp_df = pd.DataFrame(data=array, columns=[
        "age",
        "hours-per-week",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ])

    # process input user data
    X, _, _, _ = process_data(
        temp_df,
        categorical_features=get_categorical_features(),
        encoder=encoder, lb=lb, training=False)
    # get the prediction
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    if pred is None or y is None:
        # If the prediction failed, return a 500 error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to make prediction"
        )

    response = PredictionResponse(
        prediction=str(pred),
        class_name=y,
        success=True,
    )
    return response
