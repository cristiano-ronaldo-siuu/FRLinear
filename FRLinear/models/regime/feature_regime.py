import json


class FeatureRegime:
    """Loads a channel partition from a regime cache written by 05_generate_regime.py.

    mode = 'ward'   -> nested Ward partition with n_regimes groups
    mode = 'random' -> random partition with the same group sizes (seed rand_seed)
    n_regimes = 1   -> a single group containing every channel
    """

    def __init__(self, path, n_regimes=3, mode="ward", rand_seed=0, n_channels=None):
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)

        k = str(n_regimes)
        if n_regimes == 1:
            groups = cache["ward"]["1"]
        elif mode == "ward":
            if k not in cache["ward"]:
                raise ValueError(f"{path}: no Ward partition with k={n_regimes}")
            groups = cache["ward"][k]
        elif mode == "random":
            groups = cache["random"][k][str(rand_seed)]
        else:
            raise ValueError(f"unknown regime_mode {mode}")

        flat = sorted(i for g in groups for i in g)
        C = cache["n_channels"] if n_channels is None else n_channels
        if flat != list(range(C)):
            raise ValueError(f"{path}: partition does not cover channels 0..{C - 1} exactly once")

        self.groups = groups
        self.order = cache.get("order")

    def __repr__(self):
        return " | ".join(f"G{i}:{g}" for i, g in enumerate(self.groups))
