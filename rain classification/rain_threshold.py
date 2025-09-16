import pandas as pd
import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
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


if __name__ == "__main__":

    n_grid = 50
    prob_threshold = 0.5
    log_X = False
    # log_X = True

    script_path = Path(__file__).parent
    # data_path = script_path.parent / "data/FULL_INPUT_Historical_Rainfall_Data_with_Landslides 5.9.25.csv"
    data_path = script_path.parent / "data/rainfall_events_mergedERA5_Land_1950-2024.xlsx"
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
    
    feats = ["Intensity [mm/d]", "Duration [d]"]
    X = df[feats].values
    y = df["occurrences"].values

    if log_X:
        X = np.log1p(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)

    model = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Accuracy={accuracy_score(y_test, y_pred)*100:.0f}%")

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
    landslide_ratio = np.where(hist_survival+hist_landslide>0 , hist_landslide/(hist_survival+hist_landslide), np.nan)

    p_group_test, _, _ = np.histogram2d(X_test[:, 0], X_test[:, 1], bins=[x_edges, y_edges], density=True)
    hist_survival_test, _, _ = np.histogram2d(X_test[y_test == 0, 0], X_test[y_test == 0, 1], bins=[x_edges, y_edges])
    hist_landslide_test, _, _ = np.histogram2d(X_test[y_test == 1, 0], X_test[y_test == 1, 1], bins=[x_edges, y_edges])
    landslide_exceeds_threshold_prob_test = p_group_test * hist_landslide_test / np.sum(hist_survival_test+hist_landslide_test)
    volume_p_landslide_test = trapezoid(landslide_exceeds_threshold_prob_test, y_grid, axis=-1)
    volume_p_landslide_test = trapezoid(volume_p_landslide_test, x_grid, axis=-1)

    f = lambda x: calculate_p_landslide(x_grid, y_grid, landslide_probs, p_group, x)
    threshold_params = np.linspace(0, 5_000, 100_000)
    p_landslide = list(map(f, threshold_params))
    p_diff = [np.abs(volume_p_landslide_test-p).item() for p in p_landslide]
    opt_threshold_params = threshold_params[p_diff.index(min(p_diff))]

    cumulative_p = calculate_cumulative_p_landslide(x_grid, y_grid, landslide_probs, p_group, opt_threshold_params)

    if log_X:
        x_grid = np.exp(x_grid) - 1
        y_grid = np.exp(y_grid) - 1
        x_edges = np.exp(x_edges) - 1
        y_edges = np.exp(y_edges) - 1

    fig = plt.figure()
    plt.imshow(hist_survival.T+hist_landslide.T, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="auto", cmap="Reds")
    plt.colorbar(label=r"#Occurences")
    plt.xlabel(feats[0], fontsize=12)
    plt.ylabel(feats[1], fontsize=12)
    if log_X:
        plt.xscale("log")
        plt.yscale("log")
    plt.savefig(result_path/"number_occurences.png")
    plt.close()

    fig = plt.figure()
    plt.imshow(hist_landslide.T, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="auto", cmap="Reds")
    plt.colorbar(label=r"#Landslides")
    plt.xlabel(feats[0], fontsize=12)
    plt.ylabel(feats[1], fontsize=12)
    if log_X:
        plt.xscale("log")
        plt.yscale("log")
    plt.savefig(result_path/"number_landslides.png")
    plt.close()

    fig = plt.figure()
    lvls = [p for p in np.arange(0.1, 1.01, 0.1) if p != prob_threshold]
    contours = plt.contour(x_grid, y_grid, landslide_probs, levels=lvls, colors='k')
    plt.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
    contours_threshold = plt.contour(x_grid, y_grid, landslide_probs, levels=[0.5],
                                     colors='b', linewidths=2)
    plt.clabel(contours_threshold, inline=True, fontsize=8, fmt="%.1f")
    plt.imshow(landslide_ratio.T, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="auto", cmap="Reds")
    plt.colorbar(label=r"$\frac{Landslides}{Landslides+Survivals}$")
    plt.xlabel(feats[0], fontsize=12)
    plt.ylabel(feats[1], fontsize=12)
    if log_X:
        plt.xscale("log")
        plt.yscale("log")
    plt.savefig(result_path/"landslide_ratio_regression.png")
    plt.close()

    fig = plt.figure()
    threshold_line = opt_threshold_params - x_grid
    plt.plot(x_grid, threshold_line, c="g", label="Threshold")
    plt.imshow(landslide_ratio.T, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="auto", cmap="Reds")
    plt.colorbar(label=r"$\frac{Landslide}{Landslide+Survival}$")
    plt.xlabel(feats[0], fontsize=12)
    plt.ylabel(feats[1], fontsize=12)
    if log_X:
        plt.xscale("log")
        plt.yscale("log")
    plt.legend()
    plt.savefig(result_path/"rainfall_threshold.png")
    plt.close()

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(threshold_params, p_landslide, c="b")
    ax.plot(threshold_params, p_diff, c="r")
    ax.axhline(volume_p_landslide_test, c="k", label="Target probability")
    ax.set_xlabel("Threshold model parameter", fontsize=12)
    ax.set_ylabel("Volume (Σ[P(L|G)*P(G)])", fontsize=12)
    ax2.set_ylabel("|Volume - Volume test|", fontsize=12)
    ax.grid()
    plt.savefig(result_path/"volume_optimization.png")
    plt.close()

