import pandas as pd
import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import joblib
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def threshold(mesh, threshold_param):
    x, y = mesh
    exceeds = (x + y) > threshold_param
    return exceeds


def calculate_p_landslide(x_grid, y_grid, landslide_probs, p_group, threshold_param):
    mesh = np.meshgrid(x_grid, y_grid)
    cell_exceeds_threshold = threshold(mesh, threshold_param)
    landslide_exceeds_threshold = cell_exceeds_threshold * landslide_probs
    landslide_exceeds_threshold_prob = landslide_exceeds_threshold * p_group
    volume_p_landslide = trapezoid(landslide_exceeds_threshold_prob, y_grid, axis=-1)
    volume_p_landslide = trapezoid(volume_p_landslide, x_grid, axis=-1)
    return volume_p_landslide


def calculate_cumulative_p_landslide(x_grid, y_grid, landslide_probs, p_group, threshold_param):
    mesh = np.meshgrid(x_grid, y_grid)
    cell_exceeds_threshold = threshold(mesh, threshold_param)
    landslide_exceeds_threshold = cell_exceeds_threshold * landslide_probs
    landslide_exceeds_threshold_prob = landslide_exceeds_threshold * p_group
    cumulative_volume_p_landslide = cumulative_trapezoid(landslide_exceeds_threshold_prob, y_grid, axis=-1)
    cumulative_volume_p_landslide = cumulative_trapezoid(cumulative_volume_p_landslide, x_grid, axis=0)
    return cumulative_volume_p_landslide


def make_plots(model, X, y, feats, path, n_grid, log_X):

    x_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), n_grid)
    y_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), n_grid)
    mesh = np.meshgrid(x_grid, y_grid)
    mesh = np.c_[[m.flatten() for m in mesh]].T

    probs = model.predict_proba(mesh)
    landslide_probs = probs[:, 1].reshape(n_grid, n_grid)

    x_edges = np.append(0, x_grid)
    y_edges = np.append(0, y_grid)
    p_group, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[x_edges, y_edges], density=True)
    hist_survival, _, _ = np.histogram2d(X[y == 0, 0], X[y == 0, 1], bins=[x_edges, y_edges])
    hist_landslide, _, _ = np.histogram2d(X[y == 1, 0], X[y == 1, 1], bins=[x_edges, y_edges])
    landslide_ratio = np.where(hist_survival + hist_landslide > 0, hist_landslide / (hist_survival + hist_landslide), 0.)

    if log_X:
        x_grid = np.exp(x_grid) - 1
        y_grid = np.exp(y_grid) - 1
        x_edges = np.exp(x_edges) - 1
        y_edges = np.exp(y_edges) - 1

    fig = plt.figure()
    plt.imshow(hist_survival + hist_landslide, origin="lower",
               extent=[x_edges[1], x_edges[-1], y_edges[1], y_edges[-1]], aspect="auto", cmap="Reds")
    plt.colorbar(label=r"#Occurences")
    plt.xlabel(feats[0], fontsize=12)
    plt.ylabel(feats[1], fontsize=12)
    if log_X:
        plt.xscale("log")
        plt.yscale("log")
    plt.savefig(path / "number_occurrences.png")
    plt.close()

    fig = plt.figure()
    plt.imshow(hist_landslide, origin="lower", extent=[x_edges[1], x_edges[-1], y_edges[1], y_edges[-1]], aspect="auto",
               cmap="Reds")
    plt.colorbar(label=r"#Landslides")
    plt.xlabel(feats[0], fontsize=12)
    plt.ylabel(feats[1], fontsize=12)
    if log_X:
        plt.xscale("log")
        plt.yscale("log")
    plt.savefig(path / "number_landslides.png")
    plt.close()

    fig = plt.figure()
    lvls = [p for p in np.arange(0.1, 1.01, 0.1)]
    contours = plt.contour(x_grid, y_grid, landslide_probs, levels=lvls, colors='k')
    plt.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
    plt.imshow(landslide_ratio, origin="lower", extent=[x_edges[1], x_edges[-1], y_edges[1], y_edges[-1]],
               aspect="auto", cmap="Reds")
    plt.colorbar(label=r"$\frac{Landslides}{Landslides+Survivals}$")
    plt.xlabel(feats[0], fontsize=12)
    plt.ylabel(feats[1], fontsize=12)
    if log_X:
        plt.xscale("log")
        plt.yscale("log")
    plt.savefig(path / "landslide_ratio_regression.png")
    plt.close()


def main(filename, x_feat="Intensity [mm/d]", y_feat="Cumulative rainfall [mm]", n_grid=50, log_X=False):

    script_path = Path(__file__).parent
    data_path = filename
    result_path = script_path.parent / f"results/{data_path.stem}_log_{log_X}"
    result_path.mkdir(parents=True, exist_ok=True)

    if data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    elif data_path.suffix == ".xlsx":
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"File type {data_path.suffix} cannot be read.")

    for col in df.columns:
        if "intensity" in col:
            df = df.rename(columns={col: "Intensity [mm/d]"})
        if "cumulative" in col:
            df = df.rename(columns={col: "Cumulative rainfall [mm]"})
        if "duration" in col:
            df = df.rename(columns={col: "Duration [d]"})
        if "occurrence" in col:
            df = df.rename(columns={col: "occurrences"})

    feats = [x_feat]+[y_feat]
    for i, feat in enumerate(feats):
        if "intensity" in feat:
            feats[i] = "Intensity [mm/d]"
        if "cumulative" in feat:
            feats[i] = "Cumulative rainfall [mm]"
        if "duration" in feat:
            feats[i] = "Duration [d]"

    X = df[feats].values
    y = df["occurrences"].values

    if log_X:
        X = np.log1p(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)

    model = LogisticRegression(random_state=0).fit(X_train, y_train)
    joblib.dump(model, result_path/"model.pkl")

    y_pred = model.predict(X_test)

    print(f"Accuracy={accuracy_score(y_test, y_pred) * 100:.0f}%")

    make_plots(model, X, y, feats, result_path, n_grid, log_X)


if __name__ == "__main__":

    script_path = Path(__file__).parent
    data_path = script_path.parent / f"data"
    files = [f for f in data_path.iterdir() if f.is_file()]

    parser = ArgumentParser()
    parser.add_argument("--filename", type=int, default=1, help="Select a file by its number (see list below)")
    parser.add_argument("--x_feat", default="intensity")
    parser.add_argument("--y_feat", default="cumulative")
    parser.add_argument("--n_grid", type=int, default=50)
    parser.add_argument("--log_X", action="store_true")
    args = parser.parse_args()

    filename = files[args.filename-1]

    main(
        filename=filename,
        x_feat=args.x_feat,
        y_feat=args.y_feat,
        n_grid=args.n_grid,
        log_X=args.log_X,
    )

