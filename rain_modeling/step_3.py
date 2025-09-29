import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rain_modeling.utils import *
from argparse import ArgumentParser
from numpy.typing import NDArray
from typing import Dict, Optional, List, Tuple


def make_histogram(X: NDArray, y: NDArray, n_grid: int = 50) -> Dict[str, List[float]]:

    x_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), n_grid)
    y_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), n_grid)

    x_edges = np.append(0, x_grid)
    y_edges = np.append(0, y_grid)

    n_rainfall, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[x_edges, y_edges])
    n_landslide, _, _ = np.histogram2d(X[y == 1, 0], X[y == 1, 1], bins=[x_edges, y_edges])

    p_rainfall, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[x_edges, y_edges], density=True)
    p_landslide, _, _ = np.histogram2d(X[y == 1, 0], X[y == 1, 1], bins=[x_edges, y_edges], density=True)

    return {
        "n_grid": n_grid,
        "x_grid": x_grid.tolist(),
        "y_grid": y_grid.tolist(),
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "n_rainfall": p_rainfall.tolist(),
        "n_landslide": p_landslide.tolist(),
        "p_rainfall": p_rainfall.tolist(),
        "p_landslide": p_landslide.tolist()
    }


def fit_logistic(X: NDArray, y: NDArray) -> LogisticRegression:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)
    model = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy={accuracy_score(y_test, y_pred)*100:.0f}%")
    return model


def main(
        file_number: int = 1,
        x_feat: str = "intensity",
        y_feat: str  = "cumulative",
        n_grid: int = 50
) -> None:

    script_path = Path(__file__).parent
    data_path = script_path.parent / "data/rainfall"

    files = [f for f in data_path.iterdir() if f.is_file()]
    file_path = files[file_number-1]

    result_path = script_path.parent / "results/rainfall/step3"
    result_path.mkdir(parents=True, exist_ok=True)

    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".xlsx":
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unknown file type {file_path.suffix}.")

    X, y = prepare_data(df, x_feat, y_feat)

    histogram_data = make_histogram(X, y)
    with open(result_path/"histogram_data.json", "w") as f:
        json.dump(histogram_data, f, indent=4)

    model = fit_logistic(X, y)
    joblib.dump(model, result_path/"logistic_model.pkl")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--file_number", type=int, default=1)
    parser.add_argument("--x_feat", default="intensity")
    parser.add_argument("--y_feat", default="cumulative")
    parser.add_argument("--n_grid", type=int, default=50)
    args = parser.parse_args()

    main(
        file_number=args.file_number,
        x_feat=args.x_feat,
        y_feat=args.y_feat,
        n_grid=args.n_grid
    )
