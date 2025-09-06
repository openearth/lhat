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


def calculate_p_landslide(rainfall_intensity_grid, cumulative_rainfall_grid, landslide_probs, p_group, threshold_param):
    mesh = np.meshgrid(rainfall_intensity_grid, cumulative_rainfall_grid)
    cell_exceeds_threshold = threshold(mesh, threshold_param)
    landslide_exceeds_threshold = cell_exceeds_threshold * landslide_probs
    landslide_exceeds_threshold_prob = landslide_exceeds_threshold * p_group
    volume_p_landslide = trapezoid(landslide_exceeds_threshold_prob, cumulative_rainfall_grid, axis=-1)
    volume_p_landslide = trapezoid(volume_p_landslide, rainfall_intensity_grid, axis=-1)
    return volume_p_landslide


def calculate_cumulative_p_landslide(rainfall_intensity_grid, cumulative_rainfall_grid, landslide_probs, p_group, threshold_param):
    mesh = np.meshgrid(rainfall_intensity_grid, cumulative_rainfall_grid)
    cell_exceeds_threshold = threshold(mesh, threshold_param)
    landslide_exceeds_threshold = cell_exceeds_threshold * landslide_probs
    landslide_exceeds_threshold_prob = landslide_exceeds_threshold * p_group
    cumulative_volume_p_landslide = cumulative_trapezoid(landslide_exceeds_threshold_prob, cumulative_rainfall_grid, axis=-1)
    cumulative_volume_p_landslide = cumulative_trapezoid(cumulative_volume_p_landslide, rainfall_intensity_grid, axis=0)
    return cumulative_volume_p_landslide


if __name__ == "__main__":

    n_grid = 50
    prob_threshold = 0.5

    script_path = Path(__file__).parent
    data_path = script_path.parent / "data/FULL_INPUT_Historical_Rainfall_Data_with_Landslides 5.9.25.csv"
    result_path = script_path.parent / "results"
    result_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    X = df[["rainfall_intensity", "cumulative_rainfall"]].values
    y = df["occurrences"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)

    model = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rainfall_intensity_grid = np.linspace(df["rainfall_intensity"].min(), df["rainfall_intensity"].max(), n_grid)
    cumulative_rainfall_grid = np.linspace(df["cumulative_rainfall"].min(), df["cumulative_rainfall"].max(), n_grid)
    mesh = np.meshgrid(rainfall_intensity_grid, cumulative_rainfall_grid)
    mesh = np.c_[[m.flatten() for m in mesh]].T

    probs = model.predict_proba(mesh)
    landslide_probs = probs[:, 1].reshape(n_grid, n_grid)

    print(f"Accuracy={accuracy_score(y_test, y_pred):.2f}%")

    x_edges = np.append(0, rainfall_intensity_grid)
    y_edges = np.append(0, cumulative_rainfall_grid)
    hist_survival, _, _ = np.histogram2d(X[y == 0, 0], X[y == 0, 1], bins=[x_edges, y_edges])  # Histogram for y==0
    hist_landslide, _, _ = np.histogram2d(X[y == 1, 0], X[y == 1, 1], bins=[x_edges, y_edges])  # Histogram for y==1
    landslide_ratio = np.where(hist_survival+hist_landslide>0 , hist_landslide/(hist_survival+hist_landslide), np.nan)
    p_group = (hist_survival+hist_landslide) / (hist_survival.sum()+hist_landslide.sum())

    f = lambda x: calculate_p_landslide(rainfall_intensity_grid, cumulative_rainfall_grid, landslide_probs, p_group, x)
    threshold_params = np.linspace(0, 500, 10_000)
    p_landslide = list(map(f, threshold_params))
    p_landslide_test = y_test.mean()
    p_diff = [np.abs(p_landslide_test-p).item() for p in p_landslide]
    opt_threshold_params = threshold_params[p_diff.index(min(p_diff))]

    cumulative_p = calculate_cumulative_p_landslide(rainfall_intensity_grid, cumulative_rainfall_grid, landslide_probs, p_group, opt_threshold_params)

    fig = plt.figure()
    lvls = [p for p in np.arange(0.1, 1.01, 0.1) if p != prob_threshold]
    contours = plt.contour(rainfall_intensity_grid, cumulative_rainfall_grid, landslide_probs, levels=lvls, colors='k')
    plt.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
    contours_threshold = plt.contour(rainfall_intensity_grid, cumulative_rainfall_grid, landslide_probs, levels=[0.5],
                                     colors='b', linewidths=2)
    plt.clabel(contours_threshold, inline=True, fontsize=8, fmt="%.1f")
    plt.imshow(landslide_ratio.T, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="auto", cmap="Reds")
    plt.colorbar(label=r"$\frac{Landslide}{Landslide+Survival}$")
    plt.xlabel("Rainfall intensity [mm/d]", fontsize=12)
    plt.ylabel("Cumulative rainfall [mm]", fontsize=12)
    plt.savefig(result_path/"landslide_threshold.png")
    plt.close()

    fig = plt.figure()
    # lvls = [p for p in np.arange(0.1, 1.01, 0.1) if p != prob_threshold]
    # contours = plt.contour(rainfall_intensity_grid, cumulative_rainfall_grid, cumulative_p, levels=lvls, colors='k')
    contours = plt.contour(rainfall_intensity_grid[1:], cumulative_rainfall_grid[1:], cumulative_p, colors='k')
    # plt.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
    # contours_threshold = plt.contour(rainfall_intensity_grid, cumulative_rainfall_grid, landslide_probs, levels=[0.5],
    #                                  colors='b', linewidths=2)
    # plt.clabel(contours_threshold, inline=True, fontsize=8, fmt="%.1f")
    plt.imshow(landslide_ratio.T, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="auto", cmap="Reds")
    plt.colorbar(label=r"$\frac{Landslide}{Landslide+Survival}$")
    plt.xlabel("Rainfall intensity [mm/d]", fontsize=12)
    plt.ylabel("Cumulative rainfall [mm]", fontsize=12)
    plt.savefig(result_path/"landslide_p_volumes.png")
    plt.close()

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(threshold_params, p_landslide, c="b")
    ax.plot(threshold_params, p_diff, c="r")
    ax.set_xlabel("Threshold model parameter", fontsize=12)
    ax.set_ylabel("Volume (Σ[P(L|G)*P(G)])", fontsize=12)
    ax2.set_ylabel("|Volume - Volume test|", fontsize=12)
    ax.grid()
    plt.savefig(result_path/"volumes.png")
    plt.close()

