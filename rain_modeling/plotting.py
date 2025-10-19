import numpy as np
import pandas as pd
import json
from pathlib import Path
from numpy.typing import NDArray
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rcParams['agg.path.chunksize'] = 10000


def plot_lsi(lsi_data: Dict[str, List[float]], path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)
    path = path / "lsi_regression.png"

    lsis = lsi_data["lsi"]
    lsi_bins = lsi_data["lsi_bins"]
    frequency_hat = lsi_data["frequency_hat"]
    lsi_bin_centers = lsi_data["lsi_bin_centers"]
    landslide_frequency = lsi_data["landslide_frequency"]

    fig = plt.figure(figsize=(8, 6))
    plt.stem(lsi_bin_centers, landslide_frequency, basefmt=" ", label="LSI bin center data")
    plt.plot(lsis, frequency_hat, c="r", label="Regression model")
    plt.vlines(lsi_bins, color="k", ymin=-max(frequency_hat)*0.02, ymax=max(frequency_hat)*0.02)
    plt.xlabel("LSI [-]", fontsize=14)
    plt.ylabel("Landslide frequency [-]", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid()
    plt.close()

    fig.savefig(path)


def plot_rainfall_prob(rainfall_data: Dict[str, List[float]], path: Path) -> None:

    path = path / "rainfall_probability_plot.png"

    x_feat = rainfall_data["x_feat"]
    y_feat = rainfall_data["y_feat"]
    x_edges = rainfall_data["x_edges"]
    y_edges = rainfall_data["y_edges"]
    p_rainfall = rainfall_data["p_rainfall"]
    p_landslide = rainfall_data["p_landslide"]

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    
    for ax in axs:
        ax.set_aspect('equal', adjustable='box')

    ax = axs[0]
    im = ax.imshow(
        p_rainfall,
        origin="lower",
        interpolation="nearest",
        cmap="Reds"
    )
    x_tick_indices = [i * (len(x_edges) - 1) // 4 for i in range(5)]
    y_tick_indices = [i * (len(y_edges) - 1) // 4 for i in range(5)]
    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels([f'{round(x_edges[i])}' for i in x_tick_indices])
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels([f'{round(y_edges[i])}' for i in y_tick_indices])
    ax.set_xlabel(f"{x_feat.title()}_rainfall", fontsize=14)
    ax.set_ylabel(f"{y_feat.title()}_rainfall", fontsize=14)

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4.5%", pad=0.06)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Rainfall probability [-]", fontsize=14)
    cbar.ax.get_yaxis().labelpad = 15

    ax = axs[1]
    im = ax.imshow(p_landslide, origin="lower", interpolation="nearest", cmap="Reds")
    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels([f'{round(x_edges[i])}' for i in x_tick_indices])
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels([f'{round(y_edges[i])}' for i in y_tick_indices])
    ax.set_xlabel(f"{x_feat.title()}_rainfall", fontsize=14)
    ax.set_ylabel(f"{y_feat.title()}_rainfall", fontsize=14)

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4.5%", pad=0.06)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Landslide probability [-]", fontsize=14)
    cbar.ax.get_yaxis().labelpad = 15

    plt.close()

    fig.savefig(path)


def plot_logistic(rainfall_data: Dict[str, List[float]], path: Path) -> None:

    path = path / "rainfall_logistic.png"

    x_feat = rainfall_data["x_feat"]
    y_feat = rainfall_data["y_feat"]
    mesh_centers = np.array(rainfall_data["mesh_centers"])
    p_mesh_centers = rainfall_data["p_mesh_centers"]
    p_landslide = rainfall_data["p_landslide"]

    dx = mesh_centers[1, 0, 0] - mesh_centers[0, 0, 0]
    dy = mesh_centers[0, 1, 1] - mesh_centers[0, 0, 1]
    extent = [
        mesh_centers[..., 0].min() - dx / 2,
        mesh_centers[..., 0].max() + dx / 2,
        mesh_centers[..., 1].min() - dy / 2,
        mesh_centers[..., 1].max() + dy / 2,
    ]

    fig = plt.figure(figsize=(8, 6))
    im = plt.imshow(p_landslide, origin="lower", interpolation="nearest", cmap="Reds", extent=extent, aspect="auto")
    contour = plt.contour(mesh_centers[..., 0], mesh_centers[..., 1], p_mesh_centers, aspect="auto", levels=[i*0.1 for i in range(11)], colors='black')
    cbar = fig.colorbar(im)
    cbar.set_label("Landslide probability [-]", fontsize=14)
    cbar.ax.get_yaxis().labelpad = 15
    plt.xlabel(f"{x_feat.title()}_rainfall", fontsize=14)
    plt.ylabel(f"{y_feat.title()}_rainfall", fontsize=14)
    plt.clabel(contour, inline=True, fontsize=10, fmt="%.2f")

    plt.close()

    fig.savefig(path)


def plot_rainfall(rainfall_data: Dict[str, List[float]], path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)

    plot_rainfall_prob(rainfall_data, path)

    plot_logistic(rainfall_data, path)



if __name__ == "__main__":

    pass

