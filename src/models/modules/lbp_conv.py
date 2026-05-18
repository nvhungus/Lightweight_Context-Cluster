"""
lbp_conv.py — Local Binary Pattern Convolution module

Bộ lọc nhị phân tĩnh B ∈ {+1,-1,0}: 8 pattern 3×3, mỗi cái encode
sự khác biệt giữa một neighbor và trung tâm (LBP truyền thống).
Trọng số B dùng register_buffer() — tự chuyển GPU, không học được.

Pipeline:
    X [B, C, H, W]
    → Depthwise conv (8 LBP filters tĩnh) → [B, 8C, H, W]
    → Sigmoid / QReLU  → bit map
    → Conv 1×1 V (learnable) → [B, C, H, W]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .binarize import QReLU


class LBPConv(nn.Module):
    """
    Local Binary Pattern Convolution với bộ lọc cố định.

    8 bộ lọc 3×3 encoding 8 hướng láng giềng:
        center (1,1) = -1,  một neighbor = +1,  còn lại = 0

    Args:
        dim         : số channels đầu vào và đầu ra
        use_sigmoid : True → Sigmoid activation; False → QReLU (ablation)
    """

    # 8 vị trí neighbor theo clockwise từ top-left
    _NEIGHBOR_POS = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]

    def __init__(self, dim: int, use_sigmoid: bool = True):
        super().__init__()
        self.dim = dim

        # Xây 8 base filters [8, 1, 3, 3] rồi tile cho dim channels
        base = self._build_base_filters()          # [8, 1, 3, 3]
        weight = base.repeat(dim, 1, 1, 1)         # [8*dim, 1, 3, 3]
        self.register_buffer("lbp_weight", weight)

        self.act  = nn.Sigmoid() if use_sigmoid else QReLU()
        self.proj = nn.Conv2d(dim * 8, dim, kernel_size=1, bias=False)

    @classmethod
    def _build_base_filters(cls) -> torch.Tensor:
        """8 bộ lọc 3×3 với giá trị {-1, 0, +1}."""
        filters = torch.zeros(8, 1, 3, 3)
        for i, (r, c) in enumerate(cls._NEIGHBOR_POS):
            filters[i, 0, 1, 1] = -1.0  # trung tâm
            filters[i, 0, r, c] =  1.0  # neighbor thứ i
        return filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise: groups=dim → mỗi channel nhận đủ 8 LBP filters
        # lbp_weight: [8*dim, 1, 3, 3], output: [B, 8*dim, H, W]
        out = F.conv2d(x, self.lbp_weight, padding=1, groups=self.dim)
        out = self.act(out)     # bit map ∈ (0,1)
        return self.proj(out)   # [B, dim, H, W]
