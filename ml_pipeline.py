"""
ML Pipeline
"""
import argparse
from starter import basic_cleaning, train_model, evalute_model
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def execute(args):
    """
    Execute the pipeline
    """
    if args.steps == "all":
        active_steps = ["basic_cleaning", "train_model", "evaluate_model"]
    else:
        active_steps = args.steps

    if "basic_cleaning" in active_steps:
        logging.info("Basic cleaning started")
        basic_cleaning.run()

    if "train_model" in active_steps:
        logging.info("Train model started")
        train_model.run()

    if "evaluate_model"in active_steps:
        logging.info("Evaluate model started")
        evalute_model.run()


if __name__ == "__main__":
    """
    Main entrypoint
    """
    parser = argparse.ArgumentParser(description="ML training pipeline")

    parser.add_argument(
        "--steps",
        type=str,
        choices=["basic_cleaning",
                 "train_model",
                 "evaluate_model",
                 "all"],
        default="all",
        help="Pipeline steps"
    )

    main_args = parser.parse_args()

    execute(main_args)