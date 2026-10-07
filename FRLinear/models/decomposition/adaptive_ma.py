import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    """Centred moving average over the time axis of x: [B, L, C].

    pad='replicate' repeats the first/last value (identical to DLinear's moving_avg);
    pad='zero' reproduces the v1 behaviour (AvgPool1d zero padding).
    """

    def __init__(self, kernel_size, pad="replicate"):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = pad
        if pad == "zero":
            self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        else:
            self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x):
        if self.pad == "zero":
            return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        p = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, p, 1)
        end = x[:, -1:, :].repeat(1, p, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class AdaptiveMADecomposition(nn.Module):
    """Softmax-weighted mixture of moving averages (a single kernel = fixed MA)."""

    def __init__(self, kernel_sizes=(5, 9, 13, 21, 31, 41), pad="replicate"):
        super().__init__()
        for k in kernel_sizes:
            assert k % 2 == 1, "kernel sizes must be odd"
        self.kernel_sizes = list(kernel_sizes)
        self.mas = nn.ModuleList([MovingAverage(k, pad) for k in kernel_sizes])
        if len(kernel_sizes) > 1:
            self.kernel_weights = nn.Parameter(torch.ones(len(kernel_sizes)))
        else:
            self.register_buffer("kernel_weights", torch.ones(1))

    def weights(self):
        return F.softmax(self.kernel_weights, dim=0)

    def forward(self, x):
        w = self.weights()
        trend = 0
        for wi, ma in zip(w, self.mas):
            trend = trend + wi * ma(x)
        return trend, x - trend, None
