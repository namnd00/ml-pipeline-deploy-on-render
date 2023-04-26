"""
Model Evaluation
"""
import argparse
import logging
import pandas as pd
import joblib
from starter.ml.data import get_categorical_features, process_data
from sklearn.model_selection import train_test_split
from starter.ml.model import compute_model_metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def run(
    args=None,
    data_path="./data/prepared_census.csv",
    model_path="./model/model.joblib",
    encoder_path="./model/encoder.joblib",
    lb_path="./model/lb.joblib",
    output_slice="./model/slice_output.txt"
):
    """The model evaluation function"""

    if args is not None:
        _data_path, _model_path, _encoder_path, _lb_path, _output_slice = \
            (args.data_path, args.model_path, args.encoder_path, args.lb_path, args.output_slice)
    else:
        _data_path, _model_path, _encoder_path, _lb_path, _output_slice = \
            (data_path, model_path, encoder_path, lb_path, output_slice)

    logger.info("Loading data from %s", _data_path)
    df = pd.read_csv(_data_path)
    _, test = train_test_split(df, test_size=0.20)

    logger.info("Loading model...")
    model = joblib.load(_model_path)

    logger.info("Loading encoder...")
    encoder = joblib.load(_encoder_path)

    logger.info("Loading label binarizer...")
    lb = joblib.load(_lb_path)

    # define slice values
    slice_values = []
    for cat in get_categorical_features():
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp,
                categorical_features=get_categorical_features(),
                label="salary", encoder=encoder, lb=lb, training=False
            )

            y_preds = model.predict(X_test)

            precision_score, recall_score, f1_score = compute_model_metrics(y_test, y_preds)

            line = "[%s->%s] Precision: %s " \
                   "Recall: %s F1: %s" % (cat, cls, precision_score, recall_score, f1_score)
            logging.info(line)
            slice_values.append(line)

    # compute overall metrics
    logger.info("Computing overall metrics...")
    _X_test, _y_test, _, _ = process_data(
        test,
        categorical_features=get_categorical_features(),
        label="salary", encoder=encoder, lb=lb, training=False
    )
    _y_preds = model.predict(_X_test)
    _precision_score, _recall_score, _f1_score = compute_model_metrics(_y_test, _y_preds)
    line = "[Overall Score] - Precision: %s " \
        "Recall: %s F1: %s" % (_precision_score, _recall_score, _f1_score)
    logging.info(line)
    slice_values.append(line)

    with open(_output_slice, 'w') as file:
        for slice_value in slice_values:
            file.write(slice_value + '\n')


if __name__ == "__main__":
    # Create an argument parser with description
    parser = argparse.ArgumentParser(description="This steps train model")

    # Add input_artifact argument to the parser with required=True
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/prepared_census.csv",
        help="Name of the prepared data",
        required=False
    )

    # Add output_artifact argument to the parser with required=True
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model/model.joblib",
        help="Name of the trained model",
        required=False
    )

    # Add output_artifact argument to the parser with required=True
    parser.add_argument(
        "--encoder_path",
        type=str,
        default="./model/encoder.joblib",
        help="Name of the encoder model",
        required=False
    )

    # Add output_artifact argument to the parser with required=True
    parser.add_argument(
        "--lb_path",
        type=str,
        default="./model/lb.joblib",
        help="Name of the label encoder model",
        required=False
    )
    # Add output_artifact argument to the parser with required=True
    parser.add_argument(
        "--output_slice",
        type=str,
        default="./model/slice_output.txt",
        help="Name of the slice output",
        required=False
    )
    # Parse the arguments using the given parser
    args = parser.parse_args()

    # Call the go function with the parsed arguments
    _ = run(args)
