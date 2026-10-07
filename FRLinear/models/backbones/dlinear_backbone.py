import torch
import torch.nn as nn


class DLinearBackbone(nn.Module):
    """Two linear maps (trend / seasonal) over the time axis, shared by the
    channels of one regime (or one per channel when --individual)."""

    def __init__(self, configs, channels):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = channels
        self.individual = configs.individual

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(channels)])
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(channels)])
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        if getattr(configs, "debug", False) and not self.individual:
            print("Trend:", self.Linear_Trend.weight[0, :5])
            print("Seasonal:", self.Linear_Seasonal.weight[0, :5])

    def forward(self, trend, seasonal):
        # [B, L, C] -> [B, C, L]
        trend = trend.permute(0, 2, 1)
        seasonal = seasonal.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.stack(
                [self.Linear_Seasonal[i](seasonal[:, i, :]) for i in range(self.channels)], dim=1)
            trend_output = torch.stack(
                [self.Linear_Trend[i](trend[:, i, :]) for i in range(self.channels)], dim=1)
        else:
            seasonal_output = self.Linear_Seasonal(seasonal)
            trend_output = self.Linear_Trend(trend)

        return (seasonal_output + trend_output).permute(0, 2, 1)
