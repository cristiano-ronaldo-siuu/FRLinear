from pathlib import Path

import torch
import torch.nn as nn

from FRLinear.datasets import cache_name
from FRLinear.models.backbones.dlinear_backbone import DLinearBackbone
from FRLinear.models.decomposition.adaptive_ma import AdaptiveMADecomposition
from FRLinear.models.regime.feature_regime import FeatureRegime

REGIME_CACHE_DIR = Path(__file__).resolve().parents[1] / "regime_cache"


class Model(nn.Module):
    """FRLinear v2.

    Channels are split into K groups (offline Ward partition, or a random
    partition for ablation). Each group has its own adaptive multi-kernel
    decomposition and DLinear expert; a residual gate rescales each group's
    forecast with global statistics. K = 1 is DLinear + adaptive decomposition.

    Relevant args (run_longExp.py):
      --n_regimes {1,2,3}        number of groups K
      --regime_mode ward|random  partition source
      --rand_seed s              seed of the random partition
      --decomp adaptive|fixed    multi-kernel bank or a single kernel of 25
      --kernel_bank 5,9,...      shared kernel bank (adaptive)
      --pad replicate|zero       moving-average padding
      --norm none|last           NLinear-style last-value normalisation
      --gate_delta 0.005         gate amplitude
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.norm = getattr(configs, "norm", "none")
        self.delta = getattr(configs, "gate_delta", 0.005)

        cache = REGIME_CACHE_DIR / f"{cache_name(configs.data_path)}.json"
        if not cache.exists():
            raise FileNotFoundError(
                f"{cache} not found - run 05_generate_regime.py first")
        regime = FeatureRegime(cache,
                               n_regimes=getattr(configs, "n_regimes", 3),
                               mode=getattr(configs, "regime_mode", "ward"),
                               rand_seed=getattr(configs, "rand_seed", 0),
                               n_channels=self.enc_in)
        self.groups = regime.groups
        self.K = len(self.groups)
        for i, g in enumerate(self.groups):
            self.register_buffer(f"idx_{i}", torch.tensor(g, dtype=torch.long), persistent=False)

        if getattr(configs, "decomp", "adaptive") == "fixed":
            bank = [25]
        else:
            bank = [int(k) for k in str(getattr(configs, "kernel_bank", "5,9,13,21,31,41")).split(",")]
        pad = getattr(configs, "pad", "replicate")

        self.decomps = nn.ModuleList([AdaptiveMADecomposition(bank, pad) for _ in self.groups])
        self.experts = nn.ModuleList([DLinearBackbone(configs, channels=len(g)) for g in self.groups])
        # The gate is initialised on a forked RNG so that the global RNG stream
        # (expert initialisation, data shuffling) is identical to DLinear's:
        # with K=1, decomp=fixed, norm=none FRLinear then reproduces DLinear up
        # to the +-delta gate rescaling (sanity check).
        with torch.random.fork_rng(devices=[]):
            self.regime_gate = nn.Sequential(
                nn.Linear(self.enc_in * 2, 64),
                nn.GELU(),
                nn.Linear(64, self.K),
            )

        if getattr(configs, "debug", False):
            print(f"[FRLinear] K={self.K} groups={self.groups} bank={bank} pad={pad} norm={self.norm}")

    def kernel_weights(self):
        """Learned softmax weights of every group's kernel bank (for analysis)."""
        return [d.weights().detach().cpu().tolist() for d in self.decomps]

    def forward(self, x):
        # x: [B, L, C]
        gate_in = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)
        gate = 1 + self.delta * torch.tanh(self.regime_gate(gate_in))      # [B, K]

        if self.norm == "last":
            last = x[:, -1:, :].detach()
            x = x - last

        out = x.new_zeros(x.size(0), self.pred_len, self.enc_in)
        for k in range(self.K):
            idx = getattr(self, f"idx_{k}")
            trend, seasonal, _ = self.decomps[k](x.index_select(2, idx))
            y = self.experts[k](trend, seasonal) * gate[:, k].view(-1, 1, 1)
            out = out.index_copy(2, idx, y)

        if self.norm == "last":
            out = out + last
        return out
