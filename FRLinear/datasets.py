"""
Dataset registry shared by the regime generator (05_generate_regime.py) and the
experiment runner (FRLinear_scripts/run_all.py).

Optimisation settings (learning rate / batch size per horizon) follow the official
LTSF-Linear scripts in scripts/EXP-LongForecasting/Linear/*.sh and are applied
identically to FRLinear, DLinear and NLinear.

`periods` are the seasonal periods (in samples) used by the seasonal score;
`trend_lags` is the number of ACF lags averaged by the trend score.
"""

H_LONG = [96, 192, 336, 720]


def _const(value, horizons=H_LONG):
    return {h: value for h in horizons}


DATASETS = {
    # ------------------------------------------------------------------ ETT
    "ETTh1": dict(data="ETTh1", file="ETTh1.csv", enc_in=7, split="ett_h",
                  seq_len=336, horizons=H_LONG, periods=[24, 12, 8], trend_lags=100,
                  lr=_const(0.005), bs=_const(32), size="small"),
    "ETTh2": dict(data="ETTh2", file="ETTh2.csv", enc_in=7, split="ett_h",
                  seq_len=336, horizons=H_LONG, periods=[24, 12, 8], trend_lags=100,
                  lr=_const(0.05), bs=_const(32), size="small"),
    "ETTm1": dict(data="ETTm1", file="ETTm1.csv", enc_in=7, split="ett_m",
                  seq_len=336, horizons=H_LONG, periods=[96, 48, 32], trend_lags=100,
                  lr=_const(0.0001), bs=_const(8), size="small"),
    "ETTm2": dict(data="ETTm2", file="ETTm2.csv", enc_in=7, split="ett_m",
                  seq_len=336, horizons=H_LONG, periods=[96, 48, 32], trend_lags=100,
                  lr={96: 0.001, 192: 0.001, 336: 0.01, 720: 0.01}, bs=_const(32),
                  size="small"),
    # --------------------------------------------------------------- custom
    "ILI": dict(data="custom", file="national_illness.csv", enc_in=7, split="custom",
                seq_len=104, horizons=[24, 36, 48, 60], periods=[52, 26], trend_lags=52,
                lr=_const(0.01, [24, 36, 48, 60]), bs=_const(32, [24, 36, 48, 60]),
                size="small"),
    "Exchange": dict(data="custom", file="exchange_rate.csv", enc_in=8, split="custom",
                     seq_len=336, horizons=H_LONG, periods=[7, 30, 365], trend_lags=100,
                     lr=_const(0.0005), bs={96: 8, 192: 8, 336: 32, 720: 32},
                     size="small"),
    "Weather": dict(data="custom", file="weather.csv", enc_in=21, split="custom",
                    seq_len=336, horizons=H_LONG, periods=[144, 72, 48], trend_lags=100,
                    lr=_const(0.0001), bs=_const(16), size="medium"),
    "Electricity": dict(data="custom", file="electricity.csv", enc_in=321, split="custom",
                        seq_len=336, horizons=H_LONG, periods=[24, 12, 8], trend_lags=100,
                        lr=_const(0.001), bs=_const(16), size="big"),
    "Traffic": dict(data="custom", file="traffic.csv", enc_in=862, split="custom",
                    seq_len=336, horizons=H_LONG, periods=[24, 12, 8], trend_lags=100,
                    lr=_const(0.05), bs=_const(16), size="big"),
}

# Order in which the runner processes datasets (cheapest first).
PRIORITY = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "ILI", "Exchange",
            "Weather", "Electricity", "Traffic"]


def train_end(split, n_rows):
    """Number of leading rows used for training, identical to data_loader.py."""
    if split == "ett_h":
        return 12 * 30 * 24
    if split == "ett_m":
        return 12 * 30 * 24 * 4
    if split == "custom":
        return int(n_rows * 0.7)
    raise ValueError(f"unknown split {split}")


def channel_order(columns, target="OT"):
    """Channel order of the tensor fed to the model (features='M').

    Dataset_ETT_* keeps the file order (date dropped); Dataset_Custom moves the
    target column to the end. For the standard files OT is already last, so both
    rules coincide, but we reproduce the custom rule exactly to be safe.
    """
    cols = [c for c in columns if c != "date"]
    if target in cols:
        cols.remove(target)
        cols.append(target)
    return cols


def cache_name(data_path):
    """Regime cache file stem: file name without extension (e.g. 'weather')."""
    import os
    return os.path.splitext(os.path.basename(data_path))[0]
