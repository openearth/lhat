import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rain_modeling.utils import *
import json
import joblib
from pathlib import Path
from argparse import ArgumentParser
from numpy.typing import NDArray
from typing import Dict, Optional, List, Tuple


def get_density(X: NDArray, y: NDArray, n_grid: int = 50) -> Dict[str, List[float]]:
    """
    Calculates densities of rainfall events for step 3.
    """

    x_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), n_grid)
    y_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), n_grid)

    x_edges = np.append(0, x_grid)
    y_edges = np.append(0, y_grid)

    x_bins = np.vstack((x_edges[:-1], x_edges[1:])).T
    y_bins = np.vstack((y_edges[:-1], y_edges[1:])).T

    x_bin_centers = x_bins.mean(axis=-1)
    y_bin_centers = y_bins.mean(axis=-1)

    mesh_centers = np.stack(np.meshgrid(x_bin_centers, y_bin_centers))
    mesh_centers = np.transpose(mesh_centers, axes=(1, 2, 0))

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
        "x_bins": x_bins.tolist(),
        "y_bins": y_bins.tolist(),
        "x_bin_centers": x_bin_centers.tolist(),
        "y_bin_centers": y_bin_centers.tolist(),
        "mesh_centers": mesh_centers.tolist(),
        "n_rainfall": p_rainfall.tolist(),
        "n_landslide": p_landslide.tolist(),
        "p_rainfall": p_rainfall.tolist(),
        "p_landslide": p_landslide.tolist()
    }


def fit_logistic(X: NDArray, y: NDArray) -> LogisticRegression:
    """
    Performs logistic regression for step 3.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)
    model = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy={accuracy_score(y_test, y_pred)*100:.0f}%")
    return model


def get_rainfall_rate(event_starts: NDArray, X: NDArray, rainfall_data: Dict[str, List[float]]) -> Dict[str, List[[float]]]:

    x_bins = np.asarray(rainfall_data["x_bins"])
    y_bins = np.asarray(rainfall_data["y_bins"])
    n_rainfall = np.asarray(rainfall_data["n_rainfall"])
    n_landslide = np.asarray(rainfall_data["n_landslide"])

    x_feat = X[:, 0].copy()
    y_feat = X[:, 1].copy()

    event_period_years = event_starts.max() - event_starts.min()

    poisson_lambdas = np.zeros_like(n_rainfall).astype(float)
    poisson_lambdas_period = np.zeros_like(n_rainfall).astype(float)

    for i_x, x_bin in enumerate(x_bins):

        for i_y, y_bin in enumerate(y_bins):

            x_cond = np.logical_and(x_feat >= x_bin.min(), x_feat <= x_bin.max())
            y_cond = np.logical_and(y_feat >= y_bin.min(), y_feat <= y_bin.max())
            conds = np.vstack((x_cond, y_cond)).T

            idx = np.where(np.all(conds, axis=-1))[0]

            if idx.size == 0:

                continue

            else:

                event_start = event_starts[idx]

                bin_period = event_start.max() - event_start.min()

                poisson_lambdas_period[i_x, i_y] = event_start.size() / event_period_years

                if period > 0:

                    poisson_lambdas[i_x, i_y] = event_start.size / bin_period

                    pass
                pass


        pass



    return rainfall_data


def main(
        file_number: int = 1,
        x_feat: str = "intensity",
        y_feat: str  = "cumulative",
        n_grid: int = 50
) -> None:

    script_path = Path(__file__).parent
    data_path = script_path.parent / f"data/rainfall"

    files = [f for f in data_path.iterdir() if f.is_file()]
    file_path = files[file_number-1]

    result_path = script_path.parent / f"results/rainfall/{file_path.name}"
    result_path.mkdir(parents=True, exist_ok=True)

    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".xlsx":
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unknown file type {file_path.suffix}.")

    X, y = prepare_data(df, x_feat, y_feat)

    rainfall_data = get_density(X, y, n_grid)

    if "event_start" in df.columns:
        event_starts = pd.to_datetime(df["event_start"]).dt.year.values
        rainfall_data = get_rainfall_rate(event_starts, X, rainfall_data)

    model = fit_logistic(X, y)
    joblib.dump(model, result_path/"logistic_model.pkl")

    p_mesh_centers = model.predict_proba(np.asarray(rainfall_data["mesh_centers"]).reshape(-1, 2))[:, 1]
    rainfall_data["p_mesh_centers"] = p_mesh_centers.reshape(n_grid, n_grid).tolist()

    with open(result_path/"rainfall_data.json", "w") as f:
        json.dump(rainfall_data, f, indent=4)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--file_number", type=int, default=2)
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

