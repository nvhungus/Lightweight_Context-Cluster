"""
bottleneck.py — Linear Bottleneck + Channel Split cho HBCC

Triết lý: ép chiều TRƯỚC khi tách nhánh, KHÔNG dùng ReLU ở cuối.
Tránh "Manifold Collapse" (suy biến không gian đặc trưng) theo MobileNetV2.

Cấu trúc:
    X → [Conv 1×1 expand] → GELU → [DW 3×3] → [Conv 1×1 reduce, NO ACT]
                                                      ↓
                                               split → [Local | Global]
"""
import torch
import torch.nn as nn


class LinearBottleneck(nn.Module):
    """
    Bottleneck Module theo MobileNetV2 style nhưng:
    - Không dùng ReLU ở lớp cuối (giữ linearity)
    - Tách đôi kênh đầu ra cho 2 nhánh Local/Global

    Input : [B, in_dim, H, W]
    Output:
        local_feat  [B, out_dim//2, H, W]
        global_feat [B, out_dim//2, H, W]

    Args:
        in_dim:      số kênh đầu vào
        out_dim:     số kênh đầu ra (phải chẵn vì chia đôi)
        expand_ratio: hệ số mở rộng kênh ẩn (như MobileNetV2)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        expand_ratio: int = 4,
        use_bn: bool = True,
    ):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim phải chẵn để split đôi"

        hidden_dim = in_dim * expand_ratio

        # 1. Expand: Conv 1×1
        self.expand = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=not use_bn),
            nn.BatchNorm2d(hidden_dim) if use_bn else nn.Identity(),
            nn.GELU(),
        )

        # 2. Depthwise 3×3: trích xuất spatial info
        self.dw = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                      padding=1, groups=hidden_dim, bias=not use_bn),
            nn.BatchNorm2d(hidden_dim) if use_bn else nn.Identity(),
            nn.GELU(),
        )

        # 3. Reduce: Conv 1×1, KHÔNG activation (Linear Bottleneck)
        self.reduce = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=not use_bn),
            nn.BatchNorm2d(out_dim) if use_bn else nn.Identity(),
            # ← Không có activation ở đây! Đây là điểm mấu chốt
        )

    def forward(self, x: torch.Tensor):
        """
        Returns:
            local_feat:  [B, out_dim//2, H, W]
            global_feat: [B, out_dim//2, H, W]
        """
        x = self.expand(x)
        x = self.dw(x)
        x = self.reduce(x)                         # [B, out_dim, H, W]
        local_feat, global_feat = x.chunk(2, dim=1)  # tách đôi theo channel
        return local_feat, global_feat


class ChannelShuffle(nn.Module):
    """
    Channel Shuffle từ ShuffleNet — dung hợp 2 nhánh với 0 FLOPs.

    Sau khi concat [local | global] theo chiều kênh,
    shuffle để buộc 2 luồng giao thoa với nhau.

    Input : [B, C, H, W]
    Output: [B, C, H, W]  (cùng shape, khác thứ tự kênh)
    """

    def __init__(self, groups: int = 2):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C % self.groups == 0
        g = self.groups
        # Reshape → transpose → flatten
        x = x.view(B, g, C // g, H, W)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, C, H, W)
        return x


class BranchFusion(nn.Module):
    """
    Dung hợp 2 nhánh Local và Global:
    1. Concat theo chiều kênh
    2. Channel Shuffle (0 FLOPs)
    3. Conv 1×1 cuối để trộn (optional — bật khi cần học trọng số fusion)

    Input : local_feat [B, D, H, W], global_feat [B, D, H, W]
    Output: [B, out_dim, H, W]
    """

    def __init__(self, in_dim: int, out_dim: int, use_learned_fusion: bool = False):
        super().__init__()
        self.shuffle = ChannelShuffle(groups=2)
        self.use_learned_fusion = use_learned_fusion
        if use_learned_fusion:
            self.fuse = nn.Conv2d(in_dim * 2, out_dim, kernel_size=1)
        else:
            assert in_dim * 2 == out_dim or in_dim == out_dim, \
                "Nếu không dùng learned fusion, in_dim*2 phải bằng out_dim"

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([local_feat, global_feat], dim=1)  # [B, 2D, H, W]
        x = self.shuffle(x)
        if self.use_learned_fusion:
            x = self.fuse(x)
        return x
