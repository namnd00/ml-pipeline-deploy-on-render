"""
Train machine learning model
"""
import pandas as pd
import joblib
import logging
import argparse

from starter.ml.data import process_data
from starter.ml.model import train_model
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def run(
    args=None,
    data_file="./data/prepared_census.csv",
    output_model="./model/model.joblib",
    output_encoder="./model/encoder.joblib",
    output_lb="./model/lb.joblib"
):
    """Execute the model training"""
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    if args is not None:
        _data_file, _output_model, _output_encoder, _output_lb = \
            (args.data_file, args.output_model, args.output_encoder, args.output_lb)
    else:
        _data_file, _output_model, _output_encoder, _output_lb = \
            (data_file, output_model, output_encoder, output_lb)

    data = pd.read_csv(_data_file)
    train, _ = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        X=train, categorical_features=cat_features, label="salary", training=True
    )

    logger.info("Starting training...")
    trained_model = train_model(X_train, y_train)

    joblib.dump(trained_model, _output_model)
    joblib.dump(encoder, _output_encoder)
    joblib.dump(lb, _output_lb)

    logger.info("Save trained model %s to %s" % (trained_model.__str__, _output_model))
    logger.info("Save encoder %s to %s" % (encoder.__str__, _output_encoder))
    logger.info("Save label encoder %s to %s" % (lb.__str__, _output_lb))


if __name__ == "__main__":
    # Create an argument parser with description
    parser = argparse.ArgumentParser(description="This steps train model")

    # Add input_artifact argument to the parser with required=True
    parser.add_argument(
        "--data_file",
        type=str,
        default="./data/prepared_census.csv",
        help="Name of the prepared data",
        required=False
    )

    # Add output_artifact argument to the parser with required=True
    parser.add_argument(
        "--output_model",
        type=str,
        default="./model/model.joblib",
        help="Name of the trained model",
        required=False
    )

    # Add output_artifact argument to the parser with required=True
    parser.add_argument(
        "--output_encoder",
        type=str,
        default="./model/encoder.joblib",
        help="Name of the encoder model",
        required=False
    )

    # Add output_artifact argument to the parser with required=True
    parser.add_argument(
        "--output_lb",
        type=str,
        default="./model/lb.joblib",
        help="Name of the label encoder model",
        required=False
    )
    # Parse the arguments using the given parser
    args = parser.parse_args()

    # Call the go function with the parsed arguments
    _ = run(args)
