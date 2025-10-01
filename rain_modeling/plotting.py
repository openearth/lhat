import numpy as np
import pandas as pd
import json
from pathlib import Path
from numpy.typing import NDArray
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000


def plot_lsi(lsi_data: Dict[str, List[float]], path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)
    path = path / "lsi_plot.png"

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


if __name__ == "__main__":

    pass

