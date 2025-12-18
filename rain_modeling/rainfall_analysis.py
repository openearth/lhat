from rain_modeling.plotting import *
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
from pathlib import Path
from argparse import ArgumentParser


def plot_landslide_map(df: pd.DataFrame, result_path: Path) -> None:

    X = df["longitude"].values
    Y = df["latitude"].values
    Z = df["p_landslide"].values

    idx = np.where(~np.isnan(X))
    X = X[idx]
    Y = Y[idx]
    Z = Z[idx]

    grid_res = 300  # increase for smoother image
    xi = np.linspace(X.min(), X.max(), grid_res)
    yi = np.linspace(Y.min(), Y.max(), grid_res)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolate scattered data to grid
    ZI = griddata(
        (X, Y), Z,
        (XI, YI),
        method='linear'  # or 'nearest', 'cubic'
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        ZI,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin='lower',
        aspect='equal',
        cmap='viridis'
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("P(Landslide)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Landslide Probability Map")
    ax.grid()

    plt.savefig(result_path/"landslide_map.png")
    plt.close()


def main(
        file_number: int = 1
) -> None:

    script_path = Path(__file__).parent
    result_path = script_path.parent / "results/plots"

    rainfall_data_path = script_path.parent / f"data/rainfall"
    rainfall_files = [f for f in rainfall_data_path.iterdir() if f.is_file()]
    rainfall_file_path = rainfall_files[file_number-1]
    rainfall_result_path = script_path.parent /  f"results/rainfall/{rainfall_file_path.stem}"

    lsi_data_path = script_path.parent / "data/lsi"
    lsi_result_path = script_path.parent / "results/lsi"

    df_rainfall = pd.read_csv(rainfall_file_path)
    df_lsi = pd.read_csv(lsi_data_path/"LSI_pixels_ERA5_Pixels.csv")
    df_lsi["landslide_"] = df_lsi["landslide_"].fillna(0)
    df_lsi["landslide_"] = [0 if landslide == 0 else 1 for landslide in df_lsi["landslide_"]]
    df_lsi = df_lsi.dropna(how="any", subset=["LSI", "landslide_"])

    with open(rainfall_result_path/"rainfall_data.json", "r") as f:
        rainfall_data = json.load(f)
    rainfall_data = {key: np.array(val) for (key, val) in rainfall_data.items()}

    with open(lsi_result_path/"lsi_data.json", "r") as f:
        lsi_data = json.load(f)
    lsi_data = {key: np.array(val) for (key, val) in lsi_data.items()}

    plot_path = script_path.parent / "results/plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    # 1. Regional expected number of triggering events per year (for period t, with t=0 “now” and t=50 “+50 years”)
    T = round((pd.to_datetime(df_rainfall["end_time"]).max() - pd.to_datetime(df_rainfall["start_time"]).min()).days / 365, 0)
    rainfall_rate = rainfall_data["n_rainfall"] / T
    p_mesh_centers = rainfall_data["p_mesh_centers"]
    # p_mesh_centers = np.zeros_like(rainfall_data["p_mesh_centers"])
    # idx_intensity_rainfall = np.argmin(np.abs(rainfall_data["x_bin_centers"]-3.5))
    # idx_cumulative_rainfall = np.argmin(np.abs(rainfall_data["y_bin_centers"]-300))
    # p_mesh_centers[idx_intensity_rainfall, idx_cumulative_rainfall] = 1
    mu_trig = np.sum(rainfall_data["n_rainfall"]*p_mesh_centers)

    # 2. Calibrate to observed regional landslide rate:
    mu_obs = df_rainfall["occurrences_sum"].sum() / T
    c = mu_obs / mu_trig

    # 3. Annual landslide rate in cell x: λ(x) = c * μ_trig_t * w(x)
    w = lsi_data["frequency_weight_hat"]
    lams = c * mu_trig * w

    # 4. Annual probability of at least one landslide in cell x: P(L|x)=1−exp(−λ(x)). This is valid under the
    # assumption of a Poisson process (i.e., landslides occur independently and rarely).
    p_landslide_t = 1 - np.exp(-lams)

    df_lsi["lambda"] = lams
    df_lsi["c"] = [c] * len(lams)
    df_lsi["mu_trig"] = [mu_trig] * len(lams)
    df_lsi["w"] = w
    df_lsi["p_landslide"] = p_landslide_t

    df_lsi.to_csv(result_path.parent/"final.csv", index=False)

    plot_landslide_map(df_lsi, result_path)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--file_number", type=int, default=1)
    args = parser.parse_args()

    main(
        file_number=args.file_number
    )

