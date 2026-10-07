"""
extract_results.py -- aggregate per-run json results into CSV + LaTeX tables.

Inputs : <results_dir>/*.json (one per run), <summary_dir>/selected.json,
         FRLinear/regime_cache/*.json
Outputs (in <summary_dir>):
  all_runs.csv            every run
  summary.csv             mean / std (ddof=1) / n per configuration and horizon
  main_table.tex          NLinear vs DLinear vs FRLinear (selected config), mean±std
  ablation_mse.tex        horizon-averaged MSE of every ablation variant
  ablation_mae.tex        same for MAE
  ablation.csv            both metrics, all variants
  selection_table.tex     separability diagnostics vs. selected k and gains
  kernel_weights.csv      learned kernel-bank weights per group (selected config)
  sanity_check.csv        FRLinear(k=1, fixed MA25, no norm) vs DLinear

  python FRLinear_scripts/extract_results.py
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from FRLinear.datasets import DATASETS, PRIORITY, cache_name  # noqa: E402

DEC = 3


# ----------------------------------------------------------------------------
def load_runs(results_dir):
    rows = []
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        r.setdefault("n_regimes", None)
        r.setdefault("norm", None)
        r.setdefault("decomp", None)
        r.setdefault("regime_mode", None)
        r.setdefault("rand_seed", None)
        rows.append(r)
    if not rows:
        sys.exit(f"no result json found in {results_dir}")
    return pd.DataFrame(rows)


def sel_mask(df, ds, model, variant=None, **fr):
    m = (df.dataset == ds) & (df.model == model)
    if variant is not None:
        m &= df.variant == variant
    for k, v in fr.items():
        m &= df[k] == v
    return m


def stats_by_horizon(sub, metric):
    """{pred_len: (mean, std, n)} and 'Avg' computed from per-run horizon means."""
    out = {}
    for pl, g in sub.groupby("pred_len"):
        v = g[metric].to_numpy()
        out[int(pl)] = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, len(v))
    horizons = sorted(out)
    # per-run average over horizons (only runs present at every horizon; random
    # partitions are distinguished by rand_seed as well)
    key = ["run_id"] + (["rand_seed"] if sub["rand_seed"].notna().any() else [])
    piv = sub.pivot_table(index=key, columns="pred_len", values=metric, aggfunc="mean")
    piv = piv.dropna()
    if len(piv) and len(horizons):
        v = piv[horizons].mean(axis=1).to_numpy()
        out["Avg"] = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, len(v))
    return out


def cell(mean, std, style=""):
    s = f"{mean:.{DEC}f}"
    if style == "best":
        s = f"\\textbf{{{s}}}"
    elif style == "second":
        s = f"\\underline{{{s}}}"
    return f"{s}{{\\scriptsize$\\pm${std:.{DEC}f}}}"


def rank_styles(values):
    """values: list of float|None -> list of '', 'best', 'second' (ties share rank)."""
    rounded = [None if v is None else round(v, DEC) for v in values]
    uniq = sorted({v for v in rounded if v is not None})
    styles = []
    for v in rounded:
        if v is None:
            styles.append("")
        elif v == uniq[0]:
            styles.append("best")
        elif len(uniq) > 1 and v == uniq[1]:
            styles.append("second")
        else:
            styles.append("")
    return styles


# ----------------------------------------------------------------------------
def main_table(df, selected, out):
    models = [("NLinear", dict(model="NLinear")),
              ("DLinear", dict(model="DLinear")),
              ("FRLinear", None)]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Multivariate long-term forecasting results (mean$\pm$std over runs). "
             r"All models are trained in the same code base with identical look-back windows, "
             r"data splits and optimisation settings. FRLinear uses the configuration $(k, \text{norm})$ "
             r"selected on the validation set. Best in bold, second best underlined.}",
             r"\label{tab:main_results}",
             r"\resizebox{\textwidth}{!}{",
             r"\begin{tabular}{c|c|cc|cc|cc}", r"\toprule",
             r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{$H$} & \multicolumn{2}{c|}{NLinear} & "
             r"\multicolumn{2}{c|}{DLinear} & \multicolumn{2}{c}{FRLinear (ours)} \\",
             r" & & MSE & MAE & MSE & MAE & MSE & MAE \\", r"\midrule"]
    csv_rows = []
    for ds in PRIORITY:
        if ds not in selected:
            continue
        k, norm = selected[ds]["n_regimes"], selected[ds]["norm"]
        res = {}
        for name, spec in models:
            if spec is None:
                sub = df[sel_mask(df, ds, "FRLinear", "grid", n_regimes=k, norm=norm)]
            else:
                sub = df[sel_mask(df, ds, spec["model"])]
            res[name] = {m: stats_by_horizon(sub, m) for m in ("mse", "mae")} if len(sub) else None
        rows_keys = list(DATASETS[ds]["horizons"]) + ["Avg"]
        label = f"{ds} ($k$={k}, {norm})"
        for i, hk in enumerate(rows_keys):
            cells = []
            for metric in ("mse", "mae"):
                vals = [res[n][metric].get(hk, (None,))[0] if res[n] else None for n, _ in models]
                st = rank_styles(vals)
                cells.append((vals, st))
            row = []
            for mi, (name, _) in enumerate(models):
                for metric_idx, metric in enumerate(("mse", "mae")):
                    vals, st = cells[metric_idx]
                    r = res[name][metric].get(hk) if res[name] else None
                    row.append(cell(r[0], r[1], st[mi]) if r else "--")
                    if r:
                        csv_rows.append(dict(dataset=ds, horizon=hk, model=name, metric=metric,
                                             mean=r[0], std=r[1], n=r[2]))
            first = f"\\multirow{{{len(rows_keys)}}}{{*}}{{\\shortstack{{{ds}\\\\\\scriptsize($k$={k}, {norm})}}}}" if i == 0 else ""
            if hk == "Avg":
                lines.append(r"\cmidrule(lr){2-8}")
            lines.append(f"{first} & {hk} & " + " & ".join(row) + r" \\")
        lines.append(r"\midrule")
        _ = label
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"}", r"\end{table*}"]
    (out / "main_table.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame(csv_rows).to_csv(out / "main_table.csv", index=False)


# ----------------------------------------------------------------------------
def ablation(df, selected, out):
    rows = []
    for ds in PRIORITY:
        if ds not in selected:
            continue
        k, norm = selected[ds]["n_regimes"], selected[ds]["norm"]
        other = "last" if norm == "none" else "none"
        variants = [
            ("DLinear", df[sel_mask(df, ds, "DLinear")]),
            ("NLinear", df[sel_mask(df, ds, "NLinear")]),
            ("k=1", df[sel_mask(df, ds, "FRLinear", "grid", n_regimes=1, norm=norm)]),
            ("k=2", df[sel_mask(df, ds, "FRLinear", "grid", n_regimes=2, norm=norm)]),
            ("k=3", df[sel_mask(df, ds, "FRLinear", "grid", n_regimes=3, norm=norm)]),
            ("Random k=2", df[sel_mask(df, ds, "FRLinear", "random", n_regimes=2, norm=norm)]),
            ("Random k=3", df[sel_mask(df, ds, "FRLinear", "random", n_regimes=3, norm=norm)]),
            ("Fixed MA-25", df[sel_mask(df, ds, "FRLinear", "fixed_decomp", n_regimes=k, norm=norm)]),
            (f"norm={other}", df[sel_mask(df, ds, "FRLinear", "grid", n_regimes=k, norm=other)]),
        ]
        for name, sub in variants:
            if not len(sub):
                rows.append(dict(dataset=ds, variant=name))
                continue
            r = dict(dataset=ds, variant=name, selected_k=k, selected_norm=norm)
            for m in ("mse", "mae"):
                s = stats_by_horizon(sub, m).get("Avg")
                if s:
                    r[f"{m}_mean"], r[f"{m}_std"], r["n"] = s
            rows.append(r)
    ab = pd.DataFrame(rows)
    ab.to_csv(out / "ablation.csv", index=False)
    order = ["DLinear", "NLinear", "k=1", "k=2", "k=3", "Random k=2", "Random k=3", "Fixed MA-25"]
    for m in ("mse", "mae"):
        lines = [r"\begin{table*}[t]", r"\centering",
                 rf"\caption{{Ablation study: {m.upper()} averaged over horizons (mean$\pm$std). "
                 r"$k$ variants and random grouping use the selected normalisation; "
                 r"Fixed MA-25 replaces the adaptive kernel bank at the selected $k$; "
                 r"the last column flips the normalisation at the selected $k$. "
                 r"$^\dagger$ marks the selected $k$.}",
                 rf"\label{{tab:ablation_{m}}}", r"\resizebox{\textwidth}{!}{",
                 r"\begin{tabular}{l|" + "c" * (len(order) + 1) + "}", r"\toprule",
                 "Dataset & " + " & ".join(order) + r" & Norm flipped \\", r"\midrule"]
        for ds in PRIORITY:
            sub = ab[ab.dataset == ds]
            if not len(sub) or f"{m}_mean" not in ab:
                continue
            k = selected[ds]["n_regimes"]
            vals, cells_ = [], []
            names = order + [n for n in sub.variant if n.startswith("norm=")]
            for n in names:
                r = sub[sub.variant == n]
                if len(r) and not pd.isna(r.iloc[0].get(f"{m}_mean")):
                    vals.append(r.iloc[0][f"{m}_mean"])
                else:
                    vals.append(None)
            st = rank_styles(vals)
            for n, v, s in zip(names, vals, st):
                if v is None:
                    cells_.append("--")
                    continue
                r = sub[sub.variant == n].iloc[0]
                c = cell(v, r[f"{m}_std"], s)
                if n == f"k={k}":
                    c += r"$^\dagger$"
                cells_.append(c)
            lines.append(f"{ds} & " + " & ".join(cells_) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
        (out / f"ablation_{m}.tex").write_text("\n".join(lines) + "\n")
    return ab


# ----------------------------------------------------------------------------
def selection_table(df, selected, ab, out):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Regime separability (training split) versus the configuration selected on "
             r"validation and its test gain. $\Delta_{1}$: MSE change of FRLinear w.r.t.\ its $k{=}1$ "
             r"variant; $\Delta_{D}$: w.r.t.\ DLinear (negative is better).}",
             r"\label{tab:selection}", r"\resizebox{\linewidth}{!}{",
             r"\begin{tabular}{lrrrrrcrr}", r"\toprule",
             r"Dataset & $C$ & Season ratio & Trend ratio & Sil.$_{k=2}$ & Sil.$_{k=3}$ & "
             r"Selected & $\Delta_{1}$ (\%) & $\Delta_{D}$ (\%) \\", r"\midrule"]
    rows = []
    for ds in PRIORITY:
        if ds not in selected:
            continue
        path = REPO / "FRLinear" / "regime_cache" / f"{cache_name(DATASETS[ds]['file'])}.json"
        sep = json.load(open(path))["separability"]
        k, norm = selected[ds]["n_regimes"], selected[ds]["norm"]
        a = ab[ab.dataset == ds].set_index("variant")

        def avg(name):
            try:
                return float(a.loc[name, "mse_mean"])
            except (KeyError, TypeError, ValueError):
                return np.nan

        fr, k1, dl = avg(f"k={k}"), avg("k=1"), avg("DLinear")
        d1 = 100 * (fr - k1) / k1 if k1 == k1 else np.nan
        dd = 100 * (fr - dl) / dl if dl == dl else np.nan
        sil = sep.get("silhouette", {})
        r = dict(dataset=ds, C=DATASETS[ds]["enc_in"], season_ratio=sep["season_ratio"],
                 trend_ratio=sep["trend_ratio"], sil_k2=sil.get("2"), sil_k3=sil.get("3"),
                 k=k, norm=norm, delta_vs_k1=d1, delta_vs_dlinear=dd)
        rows.append(r)

        def f(v, d=2):
            return "--" if v is None or v != v else f"{v:.{d}f}"
        lines.append(f"{ds} & {r['C']} & {f(r['season_ratio'], 1)} & {f(r['trend_ratio'])} & "
                     f"{f(r['sil_k2'])} & {f(r['sil_k3'])} & $k$={k}, {norm} & {f(d1)} & {f(dd)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    (out / "selection_table.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(out / "selection_table.csv", index=False)


def kernel_weights(df, selected, out):
    rows = []
    for ds in PRIORITY:
        if ds not in selected:
            continue
        k, norm = selected[ds]["n_regimes"], selected[ds]["norm"]
        sub = df[sel_mask(df, ds, "FRLinear", "grid", n_regimes=k, norm=norm)]
        if not len(sub):
            continue
        bank = [int(x) for x in str(sub.iloc[0]["kernel_bank"]).split(",")]
        for pl, g in list(sub.groupby("pred_len")) + [("all", sub)]:
            W = np.array([w for w in g["kernel_weights"]])      # runs x groups x kernels
            mean = W.mean(axis=0)
            groups = g.iloc[0]["groups"]
            for gi, wv in enumerate(mean):
                r = dict(dataset=ds, pred_len=pl, group=gi, n_channels=len(groups[gi]),
                         effective_kernel=float(np.dot(wv, bank)))
                r.update({f"w_{b}": float(x) for b, x in zip(bank, wv)})
                rows.append(r)
    pd.DataFrame(rows).to_csv(out / "kernel_weights.csv", index=False)


def sanity(df, out):
    rows = []
    for ds in PRIORITY:
        s = df[sel_mask(df, ds, "FRLinear", "sanity")]
        d = df[sel_mask(df, ds, "DLinear")]
        if not len(s) or not len(d):
            continue
        for pl in sorted(s.pred_len.unique()):
            a, b = s[s.pred_len == pl].mse.mean(), d[d.pred_len == pl].mse.mean()
            rows.append(dict(dataset=ds, pred_len=pl, frlinear_k1_fixed=a, dlinear=b,
                             rel_diff_pct=100 * (a - b) / b))
    if rows:
        sc = pd.DataFrame(rows)
        sc.to_csv(out / "sanity_check.csv", index=False)
        print("\nSanity check FRLinear(k=1, MA-25, no norm) vs DLinear (MSE):")
        print(sc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./results_json/")
    p.add_argument("--summary_dir", default="./results_summary/")
    a = p.parse_args()
    out = Path(a.summary_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_runs(a.results_dir)
    df.to_csv(out / "all_runs.csv", index=False)

    keys = ["dataset", "model", "variant", "n_regimes", "norm", "decomp", "regime_mode", "pred_len"]
    summ = (df.fillna({"n_regimes": -1, "norm": "-", "decomp": "-", "regime_mode": "-"})
              .groupby(keys)
              .agg(mse_mean=("mse", "mean"), mse_std=("mse", "std"), mae_mean=("mae", "mean"),
                   mae_std=("mae", "std"), val_mean=("best_val_loss", "mean"), n=("mse", "size"),
                   epochs=("epochs_run", "mean"), train_time_s=("train_time_s", "mean"))
              .reset_index())
    summ.to_csv(out / "summary.csv", index=False)

    sel_path = out / "selected.json"
    if not sel_path.exists():
        print(f"{sel_path} missing: run select_config.py first. Only summary.csv was written.")
        return
    selected = json.load(open(sel_path))
    main_table(df, selected, out)
    ab = ablation(df, selected, out)
    selection_table(df, selected, ab, out)
    kernel_weights(df, selected, out)
    sanity(df, out)
    print(f"\nTables written to {out}/")
    for f in sorted(out.iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    main()
