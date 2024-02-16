import argparse
import importlib
import os
import sys
from typing import List

dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")


def available_datasets() -> List[str]:
    """
    Returns the available datasets in the dataset folder.
    """
    result = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".py"):
            result.append(os.path.basename(file)[:-3])
    return result


def load_dataset(name: str):
    """
    Loads a dataset by importing the requested module.
    """
    sys.path.append(dataset_dir)
    return importlib.import_module(name)


def add_dataset_arguments(args: argparse.Namespace, default: str):
    """
    Adds dataset-related arguments to an existing argparse object.
    """
    args.add_argument(
        "--dataset",
        type=str,
        default=default,
        help=f"Which dataset to use (default: {default})",
        choices=available_datasets(),
    )
    args.add_argument(
        "--dataset-fraction",
        action="store",
        default=1.0,
        type=float,
        help="Fraction of dataset to use (default: 1.0)",
        metavar="NUM",
    )
