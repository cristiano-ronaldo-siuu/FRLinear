"""
select_config.py -- choose (n_regimes, norm) per dataset using VALIDATION loss only.

Rule (fixed before looking at test results):
  1. For every grid configuration (k, norm) and every run r, average the best
     validation loss over all horizons  ->  one value per run.
  2. mean and standard error over runs.
  3. One-standard-error rule: among configurations whose mean is within one SE
     of the best mean, pick the simplest one: smallest k, then norm='none'.

Output: <summary_dir>/selected.json and a printed table.

  python FRLinear_scripts/select_config.py
  python FRLinear_scripts/select_config.py --results_dir ./results_json_smoke/ --summary_dir ./results_summary_smoke/
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from FRLinear.datasets import DATASETS, PRIORITY  # noqa: E402


def load(results_dir):
    recs = []
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        with open(f) as fh:
            recs.append(json.load(fh))
    return recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./results_json/")
    p.add_argument("--summary_dir", default="./results_summary/")
    p.add_argument("--grid_runs", type=int, default=3, help="must match run_all.py --grid_runs")
    p.add_argument("--grid_runs_big", type=int, default=2, help="must match run_all.py --grid_runs_big")
    a = p.parse_args()

    recs = [r for r in load(a.results_dir) if r["model"] == "FRLinear" and r["variant"] == "grid"]
    selected, report = {}, {}
    for ds in PRIORITY:
        n_grid = a.grid_runs_big if DATASETS[ds]["size"] == "big" else a.grid_runs
        # only the grid runs (not the later top-up runs of the selected config)
        rows = [r for r in recs if r["dataset"] == ds and r["run_id"] < n_grid]
        if not rows:
            continue
        horizons = sorted({r["pred_len"] for r in rows})
        # (k, norm) -> run -> {pred_len: val}
        table = defaultdict(lambda: defaultdict(dict))
        for r in rows:
            table[(r["n_regimes"], r["norm"])][r["run_id"]][r["pred_len"]] = r["best_val_loss"]

        stats = []
        for (k, norm), runs in table.items():
            per_run = [np.mean([v[h] for h in horizons]) for v in runs.values()
                       if all(h in v for h in horizons)]
            if not per_run:
                continue
            m = float(np.mean(per_run))
            se = float(np.std(per_run, ddof=1) / np.sqrt(len(per_run))) if len(per_run) > 1 else 0.0
            stats.append(dict(n_regimes=k, norm=norm, val_mean=m, val_se=se, n_runs=len(per_run)))
        if not stats:
            continue

        best = min(stats, key=lambda s: s["val_mean"])
        threshold = best["val_mean"] + best["val_se"]
        candidates = [s for s in stats if s["val_mean"] <= threshold]
        choice = min(candidates, key=lambda s: (s["n_regimes"], s["norm"] != "none"))

        selected[ds] = dict(n_regimes=choice["n_regimes"], norm=choice["norm"],
                            val_mean=choice["val_mean"], best_val_config=[best["n_regimes"], best["norm"]],
                            threshold=threshold, horizons=horizons)
        report[ds] = sorted(stats, key=lambda s: (s["n_regimes"], s["norm"]))

        print(f"\n=== {ds}  (val loss averaged over horizons {horizons})")
        for s in report[ds]:
            mark = "  <== selected" if (s["n_regimes"], s["norm"]) == (choice["n_regimes"], choice["norm"]) else ""
            star = " (min)" if s is best else ""
            print(f"   k={s['n_regimes']} norm={s['norm']:4s}  val={s['val_mean']:.5f} "
                  f"± {s['val_se']:.5f} (n={s['n_runs']}){star}{mark}")

    os.makedirs(a.summary_dir, exist_ok=True)
    with open(os.path.join(a.summary_dir, "selected.json"), "w") as f:
        json.dump(selected, f, indent=2)
    with open(os.path.join(a.summary_dir, "selection_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwritten {os.path.join(a.summary_dir, 'selected.json')}")


if __name__ == "__main__":
    main()
