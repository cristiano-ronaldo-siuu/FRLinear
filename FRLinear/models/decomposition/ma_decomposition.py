import torch
import torch.nn as nn
from .base_decomposition import BaseDecomposition


class MovingAverage(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size

        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )

    def forward(self, x):
        # x: [B, L, C]

        pad = (self.kernel_size - 1) // 2

        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)

        x_pad = torch.cat([front, x, end], dim=1)

        trend = self.avg(x_pad.permute(0, 2, 1))
        trend = trend.permute(0, 2, 1)

        return trend


class MADecomposition(BaseDecomposition):

    def __init__(self, kernel_size=25):
        super().__init__()

        self.ma = MovingAverage(kernel_size)

    def forward(self, x):
        # x: [B, L, C]

        trend = self.ma(x)

        seasonal = x - trend

        residual = None

        return trend, seasonal, residual