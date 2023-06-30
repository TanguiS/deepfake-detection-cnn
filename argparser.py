import argparse
from pathlib import Path
from typing import Dict, Union


def args_parser() -> Dict[str, any]:
    parser = argparse.ArgumentParser(
        prog="argparse, args_parser.py",
        description="Args parser for training and prediction on deepfake detection",
    )

    subparsers = parser.add_subparsers(title="Action", dest="action")

    train_action_parser(subparsers)

    plot_history_parser(subparsers)

    return vars(parser.parse_args())


def train_action_parser(subparsers):
    parser = subparsers.add_parser("train", help="Train a model or resume a training.")

    parser.add_argument("--arch", choices=["VGG19"], required=True, help="Choice an available architecture model.")
    parser.add_argument("-root", "--root_face_folder", type=Path, required=True,
                        help="Path to the folder that contains all the faces.")
    parser.add_argument("-df", "--df_faces_path", type=Path, required=True,
                        help="Path to the pickle dataframe that contains the faces informations.")
    parser.add_argument("-o", "--output_folder", type=Path, required=True,
                        help="Path to the saving folder for the models and log.")
    parser.add_argument("-odf", "--output_dataframe", type=Path, required=False,
                        help="Path to save the train, validation and test dataframes for reproducibility.")
    parser.add_argument("--shape", type=int, default=128, help="Shape/Dimension of the input face for the model.")
    parser.add_argument("--grayscale", action='store_true', help="use Grayscale for input faces, default: False.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("-epoch", "--target_epoch", type=int, required=True, help="Epoch to reach for the training.")
    parser.add_argument("-m", "--model_name", type=str, default=None, required=False, help="Name of the model to "
                                                                                           "create or to resume "
                                                                                           "training, "
                                                                                           "if not provided, "
                                                                                           "will be ask to choose.")


def plot_history_parser(subparsers):
    parser = subparsers.add_parser("plot", help="Plot the history saved from the csv files.")

    parser.add_argument("-t", "--train_history_csv_path", required=True,
                        help="Path to the train history csv file generated during training")
    parser.add_argument("-v", "--validation_history_csv_path", required=True,
                        help="Path to the validation history csv file generated during training")