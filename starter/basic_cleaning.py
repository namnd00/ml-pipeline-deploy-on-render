"""
The basic cleaning data
"""
import argparse
import logging
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def run(
    args=None,
    input_artifact="./data/census.csv",
    output_artifact="./data/prepared_census.csv"
):
    """Execute the function"""
    if args is not None:
        _input_artifact = args.input_artifact
        _output_artifact = args.output_artifact
    else:
        _input_artifact = input_artifact
        _output_artifact = output_artifact

    logger.info("Loading data...")
    df = pd.read_csv(_input_artifact, skipinitialspace=True)

    # Cleaning the data
    logger.info("Start to cleaning data...")
    try:
        df.replace({'?': None}, inplace=True)
        df.dropna(inplace=True)
        df.drop("fnlgt", axis="columns", inplace=True)
        df.drop("education-num", axis="columns", inplace=True)
        df.drop("capital-gain", axis="columns", inplace=True)
        df.drop("capital-loss", axis="columns", inplace=True)

    except Exception as e:
        logger.error(e)

    # Save the results to a CSV file
    logger.info("Saving preprocessing data to CSV")
    df.to_csv(_output_artifact, index=False)

    return df


if __name__ == "__main__":
    # Create an argument parser with description
    parser = argparse.ArgumentParser(description="This steps cleans the data")

    # Add input_artifact argument to the parser with required=True
    parser.add_argument(
        "--input_artifact",
        type=str,
        default="./data/census.csv",
        help="Name of the artifact to do preprocessing on",
        required=False
    )

    # Add output_artifact argument to the parser with required=True
    parser.add_argument(
        "--output_artifact",
        type=str,
        default="./data/prepared_census.csv",
        help="Name of the clean artifact",
        required=False
    )
    # Parse the arguments using the given parser
    args = parser.parse_args()

    # Call the go function with the parsed arguments
    _ = run(args)
