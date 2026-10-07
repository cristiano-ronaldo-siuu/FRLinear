from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATASET = "ETTm2.csv"

DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "FRLinear" / "data-analysis" / "outputs"