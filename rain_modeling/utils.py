import pandas as pd
from numpy.typing import NDArray
from typing import Tuple


def prepare_data(df: pd.DataFrame, x_feat: str, y_feat: str) -> Tuple[NDArray, NDArray]:

    for col in df.columns:
        if "intensity" in col:
            df = df.rename(columns={col: "Intensity [mm/d]"})
        if "cumulative" in col:
            df = df.rename(columns={col: "Cumulative rainfall [mm]"})
        if "duration" in col:
            df = df.rename(columns={col: "Duration [d]"})
        if "occurrence" in col:
            df = df.rename(columns={col: "occurrences"})

    feats = [x_feat] + [y_feat]
    for i, feat in enumerate(feats):
        if "intensity" in feat:
            feats[i] = "Intensity [mm/d]"
        if "cumulative" in feat:
            feats[i] = "Cumulative rainfall [mm]"
        if "duration" in feat:
            feats[i] = "Duration [d]"

    X = df[feats].values
    y = df["occurrences"].values

    return X, y


if __name__ == "__main__":

    pass

