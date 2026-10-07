"""
05_generate_regime.py -- offline regime discovery for FRLinear (v2).

Changes vs. v1
  * A1: descriptors are computed on the TRAINING split only (same borders as
        data_provider/data_loader.py), so no validation/test data is used.
  * Channel order reproduces the data loader exactly, so cached indices match
    the tensor fed to the model.
  * Ward agglomerative clustering on standardised (trend, season) descriptors
    produces NESTED partitions for k = 1..k_max (k=2 is a merge of k=3 groups).
  * A2: no minimum-group-size rule; singleton groups are allowed.
  * Random partitions with the same group sizes (ablation D2).
  * C4: separability diagnostics (score ratios, silhouette, Ward merge heights).
  * Seasonal periods / trend lags come from the dataset registry, so the script
    works for ETT, Weather, Electricity, Traffic, ILI and Exchange.
  * The series is de-meaned (train mean) before the FFT so the DC term does not
    dominate the spectral-energy denominator.

Run from the repository root:
    python FRLinear/data-analysis/src/dataset_analysis/05_generate_regime.py --dataset all
    python FRLinear/data-analysis/src/dataset_analysis/05_generate_regime.py --dataset ETTh1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.fft import rfft, rfftfreq
from sklearn.metrics import silhouette_score
from statsmodels.tsa.stattools import acf

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
from FRLinear.datasets import DATASETS, PRIORITY, cache_name, train_end  # noqa: E402


# ----------------------------------------------------------------------------
# Descriptors
# ----------------------------------------------------------------------------
def trend_score(x, n_lags):
    n_lags = min(n_lags, len(x) - 1)
    rho = acf(x, nlags=n_lags, fft=True)
    return float(np.mean(np.abs(rho[:n_lags])))


def season_score(x, periods):
    x = x - x.mean()
    power = np.abs(rfft(x)) ** 2
    freqs = rfftfreq(len(x), d=1.0)
    total = power.sum() + 1e-12
    idx = {int(np.argmin(np.abs(freqs - 1.0 / p))) for p in periods if p < len(x)}
    return float(sum(power[i] for i in idx) / total)


def channel_order(df, split):
    cols = list(df.columns)
    if split in ("ett_h", "ett_m"):          # Dataset_ETT_*: columns[1:]
        return cols[1:]
    cols.remove("date")                      # Dataset_Custom: target moved last
    if "OT" in cols:
        cols.remove("OT")
        cols.append("OT")
    return cols


# ----------------------------------------------------------------------------
# Partitions
# ----------------------------------------------------------------------------
def standardise(feats):
    mu, sd = feats.mean(0), feats.std(0)
    sd[sd < 1e-12] = 1.0
    return (feats - mu) / sd


def order_groups(groups, season):
    """Sort groups by mean seasonal score (descending); indices sorted inside."""
    groups = [sorted(int(i) for i in g) for g in groups if len(g) > 0]
    return sorted(groups, key=lambda g: -float(np.mean(season[g])))


def ward_partitions(Z, n_channels, season, k_max):
    parts = {1: [list(range(n_channels))]}
    for k in range(2, k_max + 1):
        if k > n_channels:
            break
        labels = fcluster(Z, t=k, criterion="maxclust")
        groups = [np.where(labels == lab)[0] for lab in np.unique(labels)]
        if len(groups) < k:          # degenerate (ties): fewer clusters available
            break
        parts[k] = order_groups(groups, season)
    return parts


def random_partitions(ward, n_channels, seeds):
    out = {}
    for k, groups in ward.items():
        if k == 1:
            continue
        sizes = [len(g) for g in groups]
        out[k] = {}
        for s in seeds:
            perm = np.random.default_rng(s).permutation(n_channels)
            cut = np.cumsum(sizes)[:-1]
            out[k][s] = [sorted(int(i) for i in g) for g in np.split(perm, cut)]
    return out


# ----------------------------------------------------------------------------
def process(name, root_path, out_dir, k_max, seeds):
    cfg = DATASETS[name]
    df = pd.read_csv(Path(root_path) / cfg["file"])
    cols = channel_order(df, cfg["split"])
    n_train = train_end(cfg["split"], len(df))
    train = df[cols].iloc[:n_train].to_numpy(dtype=float)
    C = len(cols)
    assert C == cfg["enc_in"], f"{name}: {C} channels but enc_in={cfg['enc_in']}"

    tr = np.array([trend_score(train[:, i], cfg["trend_lags"]) for i in range(C)])
    se = np.array([season_score(train[:, i], cfg["periods"]) for i in range(C)])
    X = standardise(np.stack([tr, se], axis=1))

    Z = linkage(X, method="ward")
    ward = ward_partitions(Z, C, se, k_max)
    rnd = random_partitions(ward, C, seeds)

    def label_of(groups):
        lab = np.zeros(C, dtype=int)
        for g_id, g in enumerate(groups):
            lab[g] = g_id
        return lab

    sil = {}
    for k, groups in ward.items():
        if 2 <= k <= C - 1:
            sil[k] = float(silhouette_score(X, label_of(groups)))

    sep = {
        "season_ratio": float(se.max() / max(se.min(), 1e-12)),
        "trend_ratio": float(tr.max() / max(tr.min(), 1e-12)),
        "season_range": [float(se.min()), float(se.max())],
        "trend_range": [float(tr.min()), float(tr.max())],
        "silhouette": {str(k): v for k, v in sil.items()},
        "ward_merge_heights_last3": [float(h) for h in Z[-3:, 2]],
    }

    result = {
        "dataset": name,
        "file": cfg["file"],
        "n_channels": C,
        "n_train_rows": int(n_train),
        "periods": cfg["periods"],
        "trend_lags": cfg["trend_lags"],
        "order": cols,
        "ward": {str(k): g for k, g in ward.items()},
        "random": {str(k): {str(s): g for s, g in d.items()} for k, d in rnd.items()},
        "features": {c: {"index": i, "trend_score": float(tr[i]),
                         "season_score": float(se[i])} for i, c in enumerate(cols)},
        "separability": sep,
        "linkage": Z.tolist(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cache_name(cfg['file'])}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[{name}] C={C} train_rows={n_train} -> {path}")
    print(f"   season ratio={sep['season_ratio']:.1f}x  trend ratio={sep['trend_ratio']:.2f}x"
          f"  silhouette={ {k: round(v, 3) for k, v in sil.items()} }")
    if C <= 30:
        for k, groups in ward.items():
            print(f"   ward k={k}: {[[cols[i] for i in g] for g in groups]}")
    else:
        for k, groups in ward.items():
            print(f"   ward k={k}: group sizes {[len(g) for g in groups]}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all", help="registry name or 'all'")
    p.add_argument("--root_path", default="./dataset/")
    p.add_argument("--out_dir", default=str(REPO_ROOT / "FRLinear" / "regime_cache"))
    p.add_argument("--k_max", type=int, default=3)
    p.add_argument("--rand_seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()

    names = PRIORITY if a.dataset == "all" else [a.dataset]
    for name in names:
        if not (Path(a.root_path) / DATASETS[name]["file"]).exists():
            print(f"[{name}] skipped: {DATASETS[name]['file']} not found in {a.root_path}")
            continue
        process(name, a.root_path, Path(a.out_dir), a.k_max, a.rand_seeds)


if __name__ == "__main__":
    main()
