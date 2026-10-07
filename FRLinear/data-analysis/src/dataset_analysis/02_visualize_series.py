import matplotlib.pyplot as plt

from pathlib import Path

from src.config import DATASET_DIR, OUTPUT_DIR, DATASET
from src.utils.io import load_dataset

SAVE_DIR = OUTPUT_DIR / "figures" / Path(DATASET).stem / "raw_series"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATASET_DIR / DATASET

df = load_dataset(DATA_PATH)

features = [c for c in df.columns if c != "date"]

# =====================================================
# full series
# =====================================================

for col in features:

    plt.figure(figsize=(16, 4))

    plt.plot(df[col])

    plt.title(f"{col} full series")

    plt.tight_layout()

    plt.savefig(
        f"{SAVE_DIR}/{col}_full.png"
    )

    plt.close()

# =====================================================
# first 1000 points
# =====================================================

for col in features:

    plt.figure(figsize=(16, 4))

    plt.plot(df[col][:1000])

    plt.title(f"{col} first 1000")

    plt.tight_layout()

    plt.savefig(
        f"{SAVE_DIR}/{col}_1000.png"
    )

    plt.close()

# =====================================================
# first 500 points
# =====================================================

for col in features:

    plt.figure(figsize=(16, 4))

    plt.plot(df[col][:500])

    plt.title(f"{col} first 500")

    plt.tight_layout()

    plt.savefig(
        f"{SAVE_DIR}/{col}_500.png"
    )

    plt.close()


print("Done.")