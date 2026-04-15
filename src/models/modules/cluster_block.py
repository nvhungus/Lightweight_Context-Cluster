"""
cluster_block.py — HBCC Block hoàn chỉnh

Lắp ghép tất cả modules thành 1 block duy nhất theo Theory.md:

    Input X
        ↓
    [LinearBottleneck] → local_feat + global_feat
        ↓                      ↓
    [PointShrink]        [BinarizedCluster]
    [Local Branch]       [Global Branch]
        ↓                      ↓
         ────[BranchFusion]────
                   ↓
               Output X'

Có thể bật/tắt từng kỹ thuật qua flags để phục vụ ablation study.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .binarize import binarize_tensor
from .hamming_sim import weighted_hamming_sim
from .point_shrink import PointShrink
from .bottleneck import LinearBottleneck, BranchFusion


# ─────────────────────────────────────────────
# Global (Binarized) Cluster
# ─────────────────────────────────────────────

class BinarizedCluster(nn.Module):
    """
    Khối gom cụm nhị phân:
    - Nếu use_hamming=True : dùng Hamming Distance (Sign + XNOR-dot)
    - Nếu use_hamming=False: dùng Cosine Similarity (CoC gốc)

    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(
        self,
        dim: int,
        proposal_w: int = 2,
        proposal_h: int = 2,
        heads: int = 4,
        head_dim: int = 16,
        use_hamming: bool = True,
    ):
        super().__init__()
        self.heads      = heads
        self.head_dim   = head_dim
        self.use_hamming = use_hamming

        self.f    = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.v    = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.proj = nn.Conv2d(heads * head_dim, dim, kernel_size=1)

        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta  = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))

    def _compute_similarity(
        self,
        centers: torch.Tensor,   # [B, M, D]
        points: torch.Tensor,    # [B, N, D]
    ) -> torch.Tensor:           # [B, M, N] ∈ (0,1)
        if self.use_hamming:
            # Binarize cả hai (STE trong training)
            centers_b = binarize_tensor(centers, use_ste=self.training)
            points_b  = binarize_tensor(points,  use_ste=self.training)
            return weighted_hamming_sim(points_b, centers_b,
                                        self.sim_alpha, self.sim_beta)
        else:
            # Cosine (CoC gốc)
            c = F.normalize(centers, dim=-1)
            p = F.normalize(points,  dim=-1)
            sim = torch.matmul(c, p.transpose(-2, -1))    # [B, M, N]
            return torch.sigmoid(self.sim_beta + self.sim_alpha * sim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.v(x)
        feat  = self.f(x)

        feat  = rearrange(feat,  "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)

        b, c, w, h = feat.shape

        centers       = self.centers_proposal(feat)
        value_centers = rearrange(self.centers_proposal(value), "b c w h -> b (w h) c")
        _, _, cw, ch  = centers.shape

        # Flatten spatial dims
        pts_flat = feat.reshape(b, c, -1).permute(0, 2, 1)      # [B, N, D]
        ctr_flat = centers.reshape(b, c, -1).permute(0, 2, 1)   # [B, M, D]

        sim = self._compute_similarity(ctr_flat, pts_flat)        # [B, M, N]

        # Hard-assign
        _, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.0)
        sim = sim * mask

        # Aggregate
        value2 = rearrange(value, "b c w h -> b (w h) c")
        out = (
            (value2.unsqueeze(1) * sim.unsqueeze(-1)).sum(dim=2) + value_centers
        ) / (sim.sum(dim=-1, keepdim=True) + 1.0)

        # Dispatch
        out = (out.unsqueeze(2) * sim.unsqueeze(-1)).sum(dim=1)
        out = rearrange(out, "b (w h) c -> b c w h", w=w)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        return self.proj(out)


# ─────────────────────────────────────────────
# Local (LBP-style) Branch
# ─────────────────────────────────────────────

class LocalBranch(nn.Module):
    """
    Nhánh cục bộ: dùng PointShrink để trích đặc trưng cục bộ.
    Thay thế cho LBPConv phức tạp — đơn giản hơn nhưng cùng triết lý.

    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(self, dim: int, use_point_shrink: bool = True):
        super().__init__()
        self.use_point_shrink = use_point_shrink
        if use_point_shrink:
            self.shrink = PointShrink(in_dim=dim, out_dim=dim, stride=1)
        else:
            # Fallback: DW Conv 3×3 đơn giản
            self.shrink = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shrink(x)


# ─────────────────────────────────────────────
# HBCC Block (Hybrid Binarized Context Cluster)
# ─────────────────────────────────────────────

class HBCCBlock(nn.Module):
    """
    Block HBCC hoàn chỉnh.

    Flags ablation:
        use_linear_bottleneck : bật Linear Bottleneck + branch split
        use_point_shrink      : bật Point Shrink trong local branch
        use_hamming           : bật Hamming Distance (thay Cosine)
        use_channel_shuffle   : bật Channel Shuffle khi fusion

    Nếu TẮT hết flags → giống CoC block gốc (baseline)
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        proposal_w: int = 2,
        proposal_h: int = 2,
        heads: int = 4,
        head_dim: int = 16,
        # Ablation flags
        use_linear_bottleneck: bool = False,
        use_point_shrink: bool = False,
        use_hamming: bool = False,
        use_channel_shuffle: bool = False,
        use_layer_scale: bool = True,
        layer_scale_init: float = 1e-5,
    ):
        super().__init__()
        self.use_linear_bottleneck = use_linear_bottleneck
        self.norm1 = nn.GroupNorm(1, dim)
        self.norm2 = nn.GroupNorm(1, dim)

        if use_linear_bottleneck:
            # Nhánh đôi: Local + Global
            branch_dim = dim // 2
            self.bottleneck   = LinearBottleneck(dim, dim, expand_ratio=4)
            self.local_branch = LocalBranch(branch_dim, use_point_shrink=use_point_shrink)
            self.global_branch = BinarizedCluster(
                dim=branch_dim,
                proposal_w=proposal_w, proposal_h=proposal_h,
                heads=max(1, heads // 2), head_dim=head_dim,
                use_hamming=use_hamming,
            )
            self.fusion = BranchFusion(
                in_dim=branch_dim,
                out_dim=dim,
                use_learned_fusion=not use_channel_shuffle,
            )
        else:
            # Không bottleneck: dùng BinarizedCluster trực tiếp (giống CoC gốc)
            self.cluster = BinarizedCluster(
                dim=dim,
                proposal_w=proposal_w, proposal_h=proposal_h,
                heads=heads, head_dim=head_dim,
                use_hamming=use_hamming,
            )

        # MLP (FFN)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.ls1 = nn.Parameter(layer_scale_init * torch.ones(dim))
            self.ls2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def _token_mix(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_linear_bottleneck:
            local_feat, global_feat = self.bottleneck(x)
            local_out  = self.local_branch(local_feat)
            global_out = self.global_branch(global_feat)
            return self.fusion(local_out, global_out)
        else:
            return self.cluster(x)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        x = self.mlp(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layer_scale:
            x = x + self.ls1.view(1, -1, 1, 1) * self._token_mix(self.norm1(x))
            x = x + self.ls2.view(1, -1, 1, 1) * self._mlp(self.norm2(x))
        else:
            x = x + self._token_mix(self.norm1(x))
            x = x + self._mlp(self.norm2(x))
        return x
