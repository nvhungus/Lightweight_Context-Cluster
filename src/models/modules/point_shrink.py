"""
point_shrink.py — Thu gọn điểm: gom 8-neighborhood thành hyper-point

Mỗi pixel + 8 láng giềng của nó → 1 siêu điểm qua Linear projection.
Bước này giảm ngay số lượng điểm (sequence length) xuống 1/9 → giảm
chi phí tính toán cluster O(N) một cách đáng kể.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointShrink(nn.Module):
    """
    Thu gọn 8-neighborhood thành hyper-point bằng Conv 3×3 depthwise + pointwise.

    Input : [B, C, H, W]
    Output: [B, out_dim, H, W]   (shape giữ nguyên, nhưng mỗi điểm đã
                                   tổng hợp thông tin từ 9 pixel xung quanh)

    Cách triển khai:
    - Dùng Conv 3×3 depthwise (groups=C) để gom 8-neighborhood,
      không stride để giữ spatial size.
    - Theo sau là Conv 1×1 pointwise để trộn kênh.
    - Không dùng activation sau pointwise (Linear Bottleneck style).

    Ghi chú: Trong paper gốc CoC, Points Reducer dùng stride để giảm N.
    Ở đây chúng ta không stride mà dùng gom thông tin. Nếu muốn giảm
    spatial size thêm, đặt stride=2 ở conv depthwise.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        stride: int = 1,
        use_bn: bool = True,
    ):
        super().__init__()
        out_dim = out_dim or in_dim

        # Depthwise 3×3: gom 8-neighborhood (1 kernel/channel, không học cách trộn kênh)
        self.dw_conv = nn.Conv2d(
            in_dim, in_dim,
            kernel_size=3, stride=stride, padding=1,
            groups=in_dim, bias=not use_bn,
        )
        # Pointwise 1×1: trộn kênh (Linear, không activation)
        self.pw_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=not use_bn)

        self.bn1 = nn.BatchNorm2d(in_dim)  if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_dim) if use_bn else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        Returns: [B, out_dim, H/stride, W/stride]
        """
        x = self.act(self.bn1(self.dw_conv(x)))
        x = self.bn2(self.pw_conv(x))    # không activation ở cuối (Linear style)
        return x


class PointShrinkV2(nn.Module):
    """
    Phiên bản đúng hơn với Theory.md:
    Concat(center, 8 neighbors) → Linear projection → hyper-point

    Vì không thể thực sự "concat" 9 điểm rồi project trong Conv chuẩn,
    ta dùng kỹ thuật unfold: lấy patch 3×3 tại mỗi vị trí → [B, C*9, H, W]
    rồi dùng Conv 1×1 để project về out_dim.

    Input : [B, C, H, W]
    Output: [B, out_dim, H, W]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        k: int = 3,           # kích thước neighborhood (3→9 điểm)
        use_bn: bool = True,
    ):
        super().__init__()
        out_dim    = out_dim or in_dim
        self.k     = k
        self.pad   = k // 2
        concat_dim = in_dim * k * k   # C * 9

        self.proj = nn.Conv2d(concat_dim, out_dim, kernel_size=1, bias=not use_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        Returns: [B, out_dim, H, W]
        """
        B, C, H, W = x.shape

        # Unfold: lấy patch k×k tại mỗi vị trí → [B, C*k*k, H*W]
        x_unfold = F.unfold(x, kernel_size=self.k, padding=self.pad)  # [B, C*k*k, H*W]
        x_unfold = x_unfold.view(B, C * self.k * self.k, H, W)        # [B, C*k*k, H, W]

        # Linear projection (Conv 1×1)
        out = self.bn(self.proj(x_unfold))
        return out
