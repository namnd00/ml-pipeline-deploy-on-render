"""
Basic cleaning module test
"""
import pytest
from starter import basic_cleaning


@pytest.fixture
def cleaned_data(data_path):
    """
    Get dataset
    """
    return basic_cleaning.run(input_artifact=data_path)


def test_null(cleaned_data):
    """
    Data is assumed to have no null values
    """
    assert cleaned_data.shape == cleaned_data.dropna().shape


def test_question_mark(cleaned_data):
    """
    Data is assumed to have no question marks value
    """
    assert '?' not in cleaned_data.values


def test_removed_columns(cleaned_data):
    """
    Data is assumed to have no question marks value
    """
    assert "fnlgt" not in cleaned_data.columns
    assert "education-num" not in cleaned_data.columns
    assert "capital-gain" not in cleaned_data.columns
    assert "capital-loss" not in cleaned_data.columns