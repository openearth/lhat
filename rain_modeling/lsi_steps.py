import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json
import joblib
from pathlib import Path
from rain_modeling.plotting import *
from argparse import ArgumentParser
from numpy.typing import NDArray
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt


def count_bins(df: pd.DataFrame, n_quantiles: int = 10) -> Dict[str, List[float]]:

    lsis = df["LSI"].values
    landslides = df["landslide_"].values

    idx_sort = np.argsort(lsis)
    lsis = lsis[idx_sort]
    landslides = landslides[idx_sort]

    quantile_lvls = np.linspace(1/n_quantiles, 1, n_quantiles)
    lsi_quantiles = np.quantile(lsis, q=quantile_lvls)
    lsi_bin_centers = (lsi_quantiles[:-1] + lsi_quantiles[1:]) / 2

    quantile_counts = np.array([np.sum(lsis<=q) for q in lsi_bin_centers]).astype(int)
    first_count = quantile_counts[0].item()
    quantile_counts = np.append(first_count, np.diff(quantile_counts))

    quantile_landslide_counts = np.array([np.count_nonzero(landslides[lsis<=q]) for q in lsi_bin_centers]).astype(int)
    first_count = quantile_landslide_counts[0].item()
    quantile_landslide_counts = np.append(first_count, np.diff(quantile_landslide_counts))

    landslide_frequency = np.where(
        quantile_counts > 0,
        quantile_landslide_counts / quantile_counts,
        0
    )

    return {
        "lsi": lsis.tolist(),
        "landslide": landslides.tolist(),
        "quantile_lvls": quantile_lvls.tolist(),
        "lsi_bins": lsi_quantiles.tolist(),
        "lsi_bin_centers": lsi_bin_centers.tolist(),
        "pixel_counts": quantile_counts.tolist(),
        "landslide_counts": quantile_landslide_counts.tolist(),
        "landslide_frequency": landslide_frequency.tolist(),
    }


def fit_linear(lsi_data: Dict[str, List[float]]) -> LinearRegression:
    """
    Performs linear regression for step 2.
    """
    X = np.asarray(lsi_data["lsi_bin_centers"])
    y = np.asarray(lsi_data["landslide_frequency"])
    y = np.log(y)

    X_train = X[~np.isinf(y)].reshape(-1, 1)
    y_train = y[~np.isinf(y)].reshape(-1, 1)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_train)
    print(f"R2={r2_score(y_train, y_pred)*100:.0f}%")

    return model


def predict(model: LinearRegression, X: NDArray) -> NDArray:
    return np.exp(model.predict(X)).squeeze()


def main(n_quantiles: int = 10) -> None:

    script_path = Path(__file__).parent

    data_path = script_path.parent / "data/lsi"

    result_path = script_path.parent / "results/lsi"
    result_path.mkdir(parents=True, exist_ok=True)

    plot_path = script_path.parent / "results/plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path/"LSI_pixels_ERA5_Pixels.csv")

    df["landslide_"] = df["landslide_"].fillna(0)
    df["landslide_"] = [0 if landslide == 0 else 1 for landslide in df["landslide_"]]

    df = df.dropna(how="any", subset=["LSI", "landslide_"])

    lsi_data = count_bins(df)

    model = fit_linear(lsi_data)
    joblib.dump(model, result_path/"linear_model.pkl")

    equation = f"landslide frequency = g(LSI) = exp({model.intercept_.item():.3f} + {model.coef_.item():.3f} * LSI)"
    with open(result_path/"equation.txt", "w") as f:
        f.writelines(equation)

    lsi_data["lsi_bins_frequency_hat"] = predict(model, np.asarray(lsi_data["lsi_bin_centers"]).reshape(-1, 1)).tolist()
    lsi_data["frequency_hat"] = predict(model, np.asarray(lsi_data["lsi"]).reshape(-1, 1)).tolist()

    with open(result_path/"lsi_data.json", "w") as f:
        json.dump(lsi_data, f, indent=4)

    df["regression"] = lsi_data["frequency_hat"]
    df.to_csv(result_path/"LSI_pixels_ERA5_Pixels.csv", index=False)

    plot_lsi(lsi_data, plot_path)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--n_quantiles", type=int, default=10)
    args = parser.parse_args()

    main(n_quantiles=args.n_quantiles)

