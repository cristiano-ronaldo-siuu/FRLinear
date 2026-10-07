from pathlib import Path

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.config import DATASET_DIR, OUTPUT_DIR, DATASET
from src.utils.io import load_dataset


DATA_PATH = DATASET_DIR / DATASET

SAVE_DIR = OUTPUT_DIR / "figures" / Path(DATASET).stem / "acf_pacf"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

df = load_dataset(DATA_PATH)

features = [c for c in df.columns if c != "date"]

LAGS = 500

for col in features:

    print(f"Processing {col}")

    series = df[col]

    # ACF
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(
        series,
        lags=LAGS,
        ax=ax
    )

    plt.title(f"ACF - {col}")

    plt.tight_layout()

    plt.savefig(
        SAVE_DIR / f"{col}_acf.png"
    )

    plt.close()

    # PACF
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_pacf(
        series,
        lags=100,
        ax=ax,
        method="ywm"
    )

    plt.title(f"PACF - {col}")

    plt.tight_layout()

    plt.savefig(
        SAVE_DIR / f"{col}_pacf.png"
    )

    plt.close()

print("Done")