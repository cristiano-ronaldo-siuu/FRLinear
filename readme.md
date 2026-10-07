# FRLinear: Feature-Regime-Aware Channel Grouping for Lightweight Long-Term Time Series Forecasting

> **Anonymous repository for double-blind review.** All author-identifying information has been removed.

FRLinear is a lightweight linear forecaster that **groups channels by temporal regime** before forecasting:

1. **Offline regime discovery (training split only).** Each channel is described by a trend score (mean |ACF| over `T` lags) and a seasonal score (share of spectral energy at dataset-specific periods, computed on the de-meaned series). The standardised descriptors are clustered with **Ward linkage**, giving nested partitions with `k = 1, 2, 3` groups.
2. **Group-specific decomposition and experts.** Each group has an adaptive multi-kernel moving-average decomposition (shared kernel bank `{5, 9, 13, 21, 31, 41}`, replicate padding, softmax weights per group) and its own DLinear expert.
3. **Residual gate.** A small MLP on the channel-wise mean/std of the raw input rescales each group's forecast within `1 ± δ` (`δ = 0.005`).
4. **Optional last-value normalisation** (as in NLinear).
5. **Model selection on validation data.** `(k, norm)` is chosen per dataset with a one-standard-error rule that prefers fewer groups, so `k = 1` (no grouping) is a built-in fallback. The test set is never used for selection.

The code base is built on the official LTSF-Linear implementation. NLinear, DLinear and FRLinear are all trained here with identical look-back windows, learning rates and batch sizes (official DLinear scripts), early stopping and test protocol (`drop_last=False`).

---

## Repository structure

```
FRLinear/
├── datasets.py                      # dataset registry: splits, periods, ACF lags, lr / batch size per horizon
├── data-analysis/src/dataset_analysis/
│   ├── 05_generate_regime.py        # offline regime discovery (Ward partitions k=1..3, random partitions, separability)
│   └── 06_visualize_regime_separability.py
├── models/
│   ├── FRLinear.py                  # model
│   ├── backbones/dlinear_backbone.py
│   ├── decomposition/adaptive_ma.py
│   └── regime/feature_regime.py
└── regime_cache/                    # pre-computed regime caches for the 9 datasets (JSON)
FRLinear_scripts/
├── run_all.sh                       # full pipeline (regime -> grid -> selection -> ablation -> tables)
├── run_all.py                       # job runner (resumable; finished runs are skipped)
├── select_config.py                 # one-standard-error selection of (k, norm) on validation loss
└── extract_results.py               # tables, significance tests, ablation, kernel weights
results_json/                        # one JSON per run (config, seed, val loss, test MSE/MAE, groups, kernel weights)
results_summary/                     # aggregated tables (CSV / LaTeX) used in the paper
exp/, data_provider/, layers/, models/, utils/, run_longExp.py   # LTSF-Linear training code
```

---

## Requirements

```bash
pip install -r requirements.txt
pip install scipy statsmodels scikit-learn matplotlib
```

Tested with Python 3.10 and PyTorch 2.x on a single NVIDIA RTX A4000 (16 GB).

---

## Datasets

The nine benchmarks are the standard LTSF datasets (ETTh1, ETTh2, ETTm1, ETTm2, ILI, Exchange, Weather, Electricity, Traffic), available from the [Autoformer repository](https://github.com/thuml/Autoformer). Put the CSV files in `./dataset/` **with these names**:

```
ETTh1.csv  ETTh2.csv  ETTm1.csv  ETTm2.csv  national_illness.csv
exchange_rate.csv  weather.csv  electricity.csv  traffic.csv
```

| Dataset | C | Interval | Split | L | Horizons | Seasonal periods P |
|---|---|---|---|---|---|---|
| ETTh1 / ETTh2 | 7 | 1 hour | 12/4/4 months | 336 | 96–720 | 24, 12, 8 |
| ETTm1 / ETTm2 | 7 | 15 min | 12/4/4 months | 336 | 96–720 | 96, 48, 32 |
| ILI | 7 | 1 week | 7:1:2 | 104 | 24–60 | 52, 26 |
| Exchange | 8 | 1 day | 7:1:2 | 336 | 96–720 | 7, 30, 365 |
| Weather | 21 | 10 min | 7:1:2 | 336 | 96–720 | 144, 72, 48 |
| Electricity | 321 | 1 hour | 7:1:2 | 336 | 96–720 | 24, 12, 8 |
| Traffic | 862 | 1 hour | 7:1:2 | 336 | 96–720 | 24, 12, 8 |

---

## Quick start

**Smoke test (~5 min)**: 1 epoch, 1 run, first horizon only.

```bash
bash FRLinear_scripts/run_all.sh --smoke --datasets ETTh1,ILI,Electricity
```

Check that no line contains `FAIL`.

**Full pipeline:**

```bash
nohup bash FRLinear_scripts/run_all.sh > run_all.out 2>&1 &
tail -f logs/run_all_grid_*.log      # progress and ETA
```

The pipeline has five steps:

1. Regime discovery (training split only).
2. Grid: DLinear, NLinear and FRLinear with `k ∈ {1,2,3}` × `norm ∈ {none, last}`.
3. Selection of `(k, norm)` on validation loss.
4. Ablation: random partitions, fixed MA-25 decomposition, the other normalisation.
5. Tables.

The runner is resumable: re-running the same command skips runs whose JSON already exists.

| Option | Default | Meaning |
|---|---|---|
| `--datasets ETTh1,Weather` | all | subset of datasets |
| `--horizons 96,192` | all | subset of horizons |
| `--runs` / `--runs_big` | 5 / 3 | runs for baselines and the selected FRLinear configuration (`big` = Electricity, Traffic) |
| `--grid_runs` / `--grid_runs_big` | 3 / 2 | runs per candidate configuration used for selection |
| `--max_jobs_small / _medium / _big` | 6 / 4 / 2 | parallel processes per dataset size |

Seeds are `2021 + r` for run `r`.

**Single configuration:**

```bash
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path weather.csv --data custom \
  --features M --seq_len 336 --pred_len 96 --enc_in 21 --batch_size 16 --learning_rate 0.0001 \
  --train_epochs 50 --patience 3 --itr 1 --model FRLinear --model_id weather_FR_k3 \
  --n_regimes 3 --regime_mode ward --decomp adaptive --norm none
```

| FRLinear flag | Values |
|---|---|
| `--n_regimes` | 1, 2, 3 |
| `--regime_mode` | `ward`, `random` (with `--rand_seed`) |
| `--decomp` | `adaptive` (kernel bank), `fixed` (single MA-25) |
| `--kernel_bank` | default `5,9,13,21,31,41` |
| `--pad` | `replicate` (default), `zero` |
| `--norm` | `none`, `last` |
| `--gate_delta` | default `0.005` |

**Regime caches** are already provided in `FRLinear/regime_cache/`. To regenerate them:

```bash
python FRLinear/data-analysis/src/dataset_analysis/05_generate_regime.py --dataset all
```

---

## Outputs

| File | Content |
|---|---|
| `results_summary/main_table.*` | NLinear / DLinear / FRLinear (selected configuration), mean ± std |
| `results_summary/selected.json`, `selection_report.json` | selected `(k, norm)` and validation loss of every candidate |
| `results_summary/selection_table.*` | separability statistics vs. selected configuration |
| `results_summary/ablation*.{csv,tex}` | `k = 1/2/3`, random grouping, fixed MA-25, flipped normalisation |
| `results_summary/kernel_weights.csv` | learned kernel-bank weights per group |
| `results_summary/all_runs.csv`, `summary.csv` | all individual runs and per-configuration aggregates |
| `results_json/*.json` | one file per run |

---

## Selected configurations (validation, one-SE rule)

| ETTh1 | ETTh2 | ETTm1 | ETTm2 | ILI | Exchange | Weather | Electricity | Traffic |
|---|---|---|---|---|---|---|---|---|
| k=1, none | k=1, last | k=2, none | k=1, last | k=2, last | k=1, last | k=3, none | k=2, none | k=2, last |

---

## License

This code is derived from LTSF-Linear and is released under the Apache License 2.0 (see `LICENSE`).