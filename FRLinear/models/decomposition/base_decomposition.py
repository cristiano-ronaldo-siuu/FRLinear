from abc import ABC, abstractmethod
import torch.nn as nn


class BaseDecomposition(nn.Module, ABC):

    @abstractmethod
    def forward(self, x):
        """
        x: [B,L,C]

        return:
            trend    [B,L,C]
            seasonal [B,L,C]
            residual [B,L,C]
        """
        pass