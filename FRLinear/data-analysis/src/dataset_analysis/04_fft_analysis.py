from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.config import DATASET_DIR, OUTPUT_DIR, DATASET
from src.utils.io import load_dataset


DATA_PATH = DATASET_DIR / DATASET

SAVE_DIR = OUTPUT_DIR / "figures" / Path(DATASET).stem / "fft"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

df = load_dataset(DATA_PATH)

features = [c for c in df.columns if c != "date"]

for col in features:

    print(f"Processing {col}")

    x = df[col].values

    x = x - x.mean()

    fft = np.fft.rfft(x)

    magnitude = np.abs(fft)

    freq = np.fft.rfftfreq(
        len(x),
        d=1
    )

    plt.figure(figsize=(12, 4))

    plt.plot(
        freq[1:],
        magnitude[1:]
    )

    plt.title(f"FFT Spectrum - {col}")

    plt.xlabel("Frequency")

    plt.ylabel("Magnitude")

    plt.tight_layout()

    plt.savefig(
        SAVE_DIR / f"{col}_fft.png"
    )

    plt.close()

    # Top frequencies

    idx = np.argsort(magnitude)[::-1]

    print("\nTop frequencies")

    for i in range(1, 11):

        f = freq[idx[i]]

        period = (
            np.inf
            if f == 0
            else 1 / f
        )

        print(
            f"rank={i} "
            f"freq={f:.6f} "
            f"period={period:.2f}"
        )

print("Done")