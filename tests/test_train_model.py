"""
Train model module test
"""
import pytest
import os
from starter import train_model


@pytest.fixture
def args():
    class Args:
        data_file = './data/prepared_census.csv'
        output_model = './model/model.joblib'
        output_encoder = './model/encoder.joblib'
        output_lb = './model/lb.joblib'
    return Args()


def test_run(args):
    train_model.run(args)

    # Check that trained model, encoder and lb exist
    assert os.path.isfile(args.output_model)
    assert os.path.isfile(args.output_encoder)
    assert os.path.isfile(args.output_lb)
