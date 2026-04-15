"""
context_cluster.py — Context Cluster Baseline cho CIFAR-10
Được đơn giản hóa từ repo gốc (ma-xu/Context-Cluster):
  - Bỏ mmseg / mmdet dependencies
  - Adapter cho ảnh 32×32: fold_w=1, fold_h=1 (không chia region)
  - Tích hợp như backbone thay thế trong mạng ResNet-style

Tác giả gốc: Xu Ma et al., ICLR 2023
Mã nguồn gốc: https://github.com/ma-xu/Context-Cluster
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

class GroupNorm1(nn.GroupNorm):
    """GroupNorm với 1 group (tương đương LayerNorm trên channel)."""
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Tính ma trận cosine similarity giữa 2 tập điểm.
    Args:
        x1: [B, M, D]
        x2: [B, N, D]
    Returns:
        sim: [B, M, N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return torch.matmul(x1, x2.transpose(-2, -1))


# ─────────────────────────────────────────────
# Core: Cluster Operation
# ─────────────────────────────────────────────

class Cluster(nn.Module):
    """
    Khối Context Cluster cốt lõi.
    Input : [B, C, H, W]
    Output: [B, out_dim, H, W]

    Quy trình:
    1. Chiếu sang similarity space (f) và value space (v)
    2. Khởi tạo tâm cụm bằng AdaptiveAvgPool
    3. Tính cosine similarity → hard-assign mỗi điểm vào 1 tâm
    4. Aggregate: tổng hợp có trọng số về tâm
    5. Dispatch: truyền đặc trưng tâm về từng điểm
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        proposal_w: int = 2,
        proposal_h: int = 2,
        fold_w: int = 1,
        fold_h: int = 1,
        heads: int = 4,
        head_dim: int = 16,
    ):
        super().__init__()
        self.heads    = heads
        self.head_dim = head_dim
        self.fold_w   = fold_w
        self.fold_h   = fold_h

        self.f    = nn.Conv2d(dim, heads * head_dim, kernel_size=1)   # similarity proj
        self.v    = nn.Conv2d(dim, heads * head_dim, kernel_size=1)   # value proj
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)

        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta  = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W]"""
        value = self.v(x)                                             # [B, heads*d, H, W]
        x     = self.f(x)                                             # [B, heads*d, H, W]

        # Tách multi-head: merge batch và head dims
        x     = rearrange(x,     "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)

        # Region Partition (bỏ qua nếu fold=1 — phù hợp CIFAR-10)
        if self.fold_w > 1 and self.fold_h > 1:
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0
            x     = rearrange(x,     "b c (f1 w) (f2 h) -> (b f1 f2) c w h",
                              f1=self.fold_w, f2=self.fold_h)
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h",
                              f1=self.fold_w, f2=self.fold_h)

        b, c, w, h = x.shape

        # Tâm cụm: [b, c, Cw, Ch]
        centers       = self.centers_proposal(x)
        value_centers = rearrange(self.centers_proposal(value), "b c w h -> b (w h) c")
        _, _, cw, ch  = centers.shape

        # Cosine similarity: [B, M, N]   M=Cw*Ch, N=w*h
        sim = torch.sigmoid(
            self.sim_beta + self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),   # [B, M, D]
                x.reshape(b, c, -1).permute(0, 2, 1),          # [B, N, D]
            )
        )

        # Hard-assign: mỗi điểm thuộc đúng 1 tâm
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.0)
        sim = sim * mask

        # Aggregate: [B, M, D]
        value2 = rearrange(value, "b c w h -> b (w h) c")
        out = (
            (value2.unsqueeze(1) * sim.unsqueeze(-1)).sum(dim=2) + value_centers
        ) / (sim.sum(dim=-1, keepdim=True) + 1.0)

        # Dispatch: [B, N, D] → [B, D, H, W]
        out = (out.unsqueeze(2) * sim.unsqueeze(-1)).sum(dim=1)
        out = rearrange(out, "b (w h) c -> b c w h", w=w)

        # Khôi phục region partition nếu có
        if self.fold_w > 1 and self.fold_h > 1:
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)",
                            f1=self.fold_w, f2=self.fold_h)

        # Khôi phục multi-head
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        return out


# ─────────────────────────────────────────────
# MLP block
# ─────────────────────────────────────────────

class Mlp(nn.Module):
    """Feed-Forward Network sau Cluster block."""

    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, drop: float = 0.0):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → permute để dùng Linear
        x = self.fc1(x.permute(0, 2, 3, 1))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        x = self.drop(x)
        return x


# ─────────────────────────────────────────────
# ClusterBlock (1 block hoàn chỉnh = Cluster + MLP + residual)
# ─────────────────────────────────────────────

class ClusterBlock(nn.Module):
    """
    1 block Context Cluster: Norm → Cluster → residual → Norm → MLP → residual
    Có Layer Scale để ổn định training.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init: float = 1e-5,
        proposal_w: int = 2,
        proposal_h: int = 2,
        fold_w: int = 1,
        fold_h: int = 1,
        heads: int = 4,
        head_dim: int = 16,
    ):
        super().__init__()
        self.norm1 = GroupNorm1(dim)
        self.norm2 = GroupNorm1(dim)
        self.cluster = Cluster(
            dim=dim, out_dim=dim,
            proposal_w=proposal_w, proposal_h=proposal_h,
            fold_w=fold_w, fold_h=fold_h,
            heads=heads, head_dim=head_dim,
        )
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

        self.drop_path = nn.Identity()  # DropPath можно добавить позже

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.ls1 = nn.Parameter(layer_scale_init * torch.ones(dim))
            self.ls2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.ls1.unsqueeze(-1).unsqueeze(-1) * self.cluster(self.norm1(x))
            )
            x = x + self.drop_path(
                self.ls2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
            )
        else:
            x = x + self.drop_path(self.cluster(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ─────────────────────────────────────────────
# Points Reducer (stem + downsampling)
# ─────────────────────────────────────────────

class PointReducer(nn.Module):
    """
    Giảm số điểm bằng Conv có stride (tương đương trong không gian điểm).
    Input : [B, in_chans, H, W]
    Output: [B, embed_dim, H/stride, W/stride]
    """

    def __init__(self, patch_size=2, stride=2, padding=0,
                 in_chans=3, embed_dim=64, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


# ─────────────────────────────────────────────
# Full CoC Network cho CIFAR-10
# ─────────────────────────────────────────────

class CoCCIFAR(nn.Module):
    """
    Context Cluster Network cho CIFAR-10 (32×32).

    Architecture (4 stage):
    Stage 1: 32→16,  embed=64,  N_blocks=2
    Stage 2: 16→8,   embed=128, N_blocks=2
    Stage 3:  8→4,   embed=256, N_blocks=4
    Stage 4:  4→2,   embed=512, N_blocks=2
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dims: list = None,
        depths: list = None,
        heads: int = 4,
        head_dim: int = 16,
        mlp_ratio: float = 4.0,
        proposal_w: int = 2,
        proposal_h: int = 2,
        fold_w: int = 1,
        fold_h: int = 1,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        if embed_dims is None:
            embed_dims = [64, 128, 256, 512]
        if depths is None:
            depths = [2, 2, 4, 2]

        # Stem: 3 → embed_dims[0], stride=2
        self.stem = PointReducer(patch_size=2, stride=2, padding=0,
                                 in_chans=3, embed_dim=embed_dims[0],
                                 norm_layer=GroupNorm1)

        # Stages
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            # Downsampling (trừ stage đầu đã có stem)
            if i > 0:
                self.downsamplers.append(
                    PointReducer(patch_size=2, stride=2, padding=0,
                                 in_chans=embed_dims[i - 1], embed_dim=dim,
                                 norm_layer=GroupNorm1)
                )
            else:
                self.downsamplers.append(nn.Identity())

            blocks = nn.Sequential(*[
                ClusterBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    proposal_w=proposal_w,
                    proposal_h=proposal_h,
                    fold_w=fold_w,
                    fold_h=fold_h,
                    heads=heads,
                    head_dim=head_dim,
                )
                for _ in range(depth)
            ])
            self.stages.append(blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(embed_dims[-1]),
            nn.Linear(embed_dims[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, (down, stage) in enumerate(zip(self.downsamplers, self.stages)):
            if i > 0:
                x = down(x)
            x = stage(x)
        x = self.head(x)
        return x


# ─────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────

def coc_tiny_cifar(num_classes: int = 10, **kwargs) -> CoCCIFAR:
    """CoC-Tiny: ~1M params"""
    return CoCCIFAR(
        num_classes=num_classes,
        embed_dims=[32, 64, 128, 256],
        depths=[2, 2, 2, 2],
        heads=2, head_dim=16,
        **kwargs,
    )


def coc_small_cifar(num_classes: int = 10, **kwargs) -> CoCCIFAR:
    """CoC-Small: ~3-4M params"""
    return CoCCIFAR(
        num_classes=num_classes,
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 4, 2],
        heads=4, head_dim=16,
        **kwargs,
    )
