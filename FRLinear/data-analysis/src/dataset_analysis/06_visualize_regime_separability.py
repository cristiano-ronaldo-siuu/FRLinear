"""
06_visualize_regime_separability.py (v2)

For every dataset with a regime cache (05_generate_regime.py), draws
  (a) standardised (trend, season) descriptors coloured by the Ward partition
      at the selected k (results_summary/selected.json) or k=3, and
  (b) the Ward dendrogram (datasets with <= 30 channels).
Also plots learned kernel-bank weights per group if results_summary/kernel_weights.csv exists.

  python FRLinear/data-analysis/src/dataset_analysis/06_visualize_regime_separability.py
Output: ./output_figures/*.pdf|png
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from FRLinear.datasets import DATASETS, PRIORITY, cache_name  # noqa: E402

CACHE = REPO / "FRLinear" / "regime_cache"
SUMMARY = REPO / "results_summary"
OUT = Path("./output_figures")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8})
COLORS = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd"]


def main():
    selected = {}
    if (SUMMARY / "selected.json").exists():
        selected = json.load(open(SUMMARY / "selected.json"))

    caches = []
    for ds in PRIORITY:
        p = CACHE / f"{cache_name(DATASETS[ds]['file'])}.json"
        if p.exists():
            c = json.load(open(p))
            if "ward" in c:                  # skip caches from the old (v1) format
                caches.append((ds, c))
    if not caches:
        sys.exit("no regime cache found")

    n = len(caches)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.2), squeeze=False)
    for col, (ds, c) in enumerate(caches):
        feats = c["features"]
        tr = np.array([v["trend_score"] for v in feats.values()])
        se = np.array([v["season_score"] for v in feats.values()])
        k = selected.get(ds, {}).get("n_regimes", min(3, max(int(x) for x in c["ward"])))
        groups = c["ward"][str(k)]
        ax = axes[0, col]
        for gi, g in enumerate(groups):
            ax.scatter(tr[g], se[g], s=14, color=COLORS[gi % len(COLORS)], label=f"G{gi}")
        ax.set_title(f"{ds} (k={k})")
        ax.set_xlabel("trend score")
        if col == 0:
            ax.set_ylabel("season score")
        ax.set_yscale("log")
        ax.legend(fontsize=6, frameon=False)

        ax = axes[1, col]
        if c["n_channels"] <= 30:
            dendrogram(np.array(c["linkage"]), labels=c["order"], ax=ax,
                       leaf_rotation=90, leaf_font_size=5, color_threshold=0)
            ax.set_title("Ward dendrogram", fontsize=7)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{c['n_channels']} channels\n(dendrogram omitted)",
                    ha="center", va="center", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"regime_separability.{ext}", dpi=300)
    print(f"saved {OUT / 'regime_separability.pdf'}")

    kw = SUMMARY / "kernel_weights.csv"
    if kw.exists():
        import pandas as pd
        df = pd.read_csv(kw)
        df = df[df.pred_len == "all"]
        wcols = [x for x in df.columns if x.startswith("w_")]
        dss = list(dict.fromkeys(df.dataset))
        fig, axes = plt.subplots(1, len(dss), figsize=(2.2 * len(dss), 1.9), squeeze=False)
        for i, ds in enumerate(dss):
            sub = df[df.dataset == ds]
            ax = axes[0, i]
            x = np.arange(len(wcols))
            width = 0.8 / max(len(sub), 1)
            for j, (_, r) in enumerate(sub.iterrows()):
                ax.bar(x + j * width, r[wcols].to_numpy(float), width,
                       color=COLORS[j % len(COLORS)], label=f"G{int(r.group)}")
            ax.set_xticks(x + 0.4 - width / 2)
            ax.set_xticklabels([w[2:] for w in wcols], fontsize=6)
            ax.set_title(ds)
            ax.legend(fontsize=6, frameon=False)
        axes[0, 0].set_ylabel("softmax weight")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"kernel_weights.{ext}", dpi=300)
        print(f"saved {OUT / 'kernel_weights.pdf'}")


if __name__ == "__main__":
    main()
