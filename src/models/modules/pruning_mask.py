"""
pruning_mask.py — Learnable Pruning Mask với Crispness Loss

Ép soft mask m_i = sigmoid(logit_i) về binary {0, 1} bằng BCE loss.
Không dùng STE — gradient tự nhiên qua sigmoid + BCE.

Chỉ bật khi use_pruning_mask=True (ablation_step4_full.yaml).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PruningMask(nn.Module):
    """
    Learnable mask cho structured channel pruning.

    - forward()        : trả về soft mask m ∈ (0,1)
    - crispness_loss() : BCE(m, round(m).detach()) ép m → {0,1}
    - get_binary_mask(): mask nhị phân dùng cho inference

    Args:
        n_elements: số phần tử cần mask (thường = số channels)
    """

    def __init__(self, n_elements: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_elements))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)   # m_i ∈ (0,1)

    def crispness_loss(self) -> torch.Tensor:
        m = self.forward()
        m_hat = m.detach().round()          # target nhị phân, không backprop
        return F.binary_cross_entropy(m, m_hat)

    def get_binary_mask(self) -> torch.Tensor:
        """Mask nhị phân {0,1} cho inference / visualization."""
        return self.forward().detach().round()
