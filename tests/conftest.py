"""
Constant for pytest
"""
import pandas as pd
import pytest
from starter.ml.data import get_categorical_features
from sklearn.model_selection import train_test_split


@pytest.fixture
def data_path():
    return "./data/census.csv"

@pytest.fixture
def data(data_path):
    return pd.read_csv(data_path)

@pytest.fixture
def prepared_data_path():
    return "./data/prepared_census.csv"

@pytest.fixture
def prepared_data(prepared_data_path):
    return pd.read_csv(prepared_data_path)

@pytest.fixture
def cat_features():
    return get_categorical_features()

@pytest.fixture
def train_data(prepared_data):
    train, _ = train_test_split(prepared_data, test_size=0.20)
    return train

@pytest.fixture
def val_data(prepared_data):
    _, val = train_test_split(prepared_data, test_size=0.20)
    return val