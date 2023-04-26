"""
Evalute model module test
"""
import pytest
import joblib
import os

from starter import evalute_model
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics


@pytest.fixture
def args():
    class Args:
        data_path = './data/prepared_census.csv'
        model_path = './model/model.joblib'
        encoder_path = './model/encoder.joblib'
        lb_path = './model/lb.joblib'
        output_slice = './model/slice_output.txt'
    return Args()


def test_run(args, val_data, cat_features):
    evalute_model.run(args)

    # Check that slice output file was created with expected lines.
    assert os.path.isfile(args.output_slice)
    with open(args.output_slice, 'r') as file:
        slice_lines = file.readlines()
    assert len(slice_lines) >= 81 # number of unique combinations of categorical features and their classes in prepared_census.csv.

    # Load the trained model and ensure it has expected score on overall test data.
    model = joblib.load(args.model_path)
    encoder = joblib.load(args.encoder_path)
    lb = joblib.load(args.lb_path)

    X_test, y_test, _, _ = process_data(
                    val_data,
                    categorical_features=cat_features,
                    label="salary", encoder=encoder, lb=lb, training=False
                )
    y_preds = model.predict(X_test)
    precision_score, recall_score, f1_score = compute_model_metrics(y_test, y_preds)
    assert precision_score > 0.7 and recall_score > 0.7 and f1_score > 0.7