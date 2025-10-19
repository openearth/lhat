import pandas as pd
import numpy as np
from scipy.stats import poisson, expon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rain_modeling.utils import *
from rain_modeling.plotting import *
import json
import joblib
from pathlib import Path
from argparse import ArgumentParser
from numpy.typing import NDArray
from typing import Dict, Optional, List, Tuple


def main(
        file_number: int = 1
) -> None:

    script_path = Path(__file__).parent

    rainfall_data_path = script_path.parent / f"data/rainfall"
    lsi_data_path = script_path.parent / "data/lsi"

    rainfall_files = [f for f in rainfall_data_path.iterdir() if f.is_file()]
    rainfall_file_path = rainfall_files[file_number-1]

    rainfall_result_path = script_path.parent /  f"results/rainfall/{rainfall_file_path.stem}"
    lsi_result_path = script_path.parent / "results/lsi"

    with open(rainfall_result_path/"rainfall_data.json", "r") as f:
        rainfall_data = json.load(f)

    with open(lsi_result_path/"lsi_data.json", "r") as f:
        lsi_data = json.load(f)

    plot_path = script_path.parent / "results/plots"
    plot_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--file_number", type=int, default=1)
    args = parser.parse_args()

    main(
        file_number=args.file_number
    )

