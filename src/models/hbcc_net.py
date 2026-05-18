"""
hbcc_net.py — Kiến trúc mạng hoàn chỉnh HBCC cho CIFAR-10

Hỗ trợ 3 mode (chọn qua config['model']):
    'resnet18'    → ResNet-18 adapted CIFAR-10 (baseline)
    'coc_baseline'→ CoC gốc cho CIFAR-10
    'hbcc'        → HBCC full (Lightweight CoC)

Mỗi model có 4 stage, mỗi stage giảm spatial size ×2.
CIFAR-10: 32→16→8→4→2 → GlobalAvgPool → FC
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models

from .modules.cluster_block import HBCCBlock
from .baselines.context_cluster import CoCCIFAR, coc_tiny_cifar, coc_small_cifar


# ─────────────────────────────────────────────
# ResNet-18 adapted cho CIFAR-10
# ─────────────────────────────────────────────

def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 chuẩn nhưng:
    - Conv đầu: 3×3, stride=1, padding=1 (thay vì 7×7, stride=2)
    - Bỏ MaxPool (CIFAR quá nhỏ)
    """
    model = tv_models.resnet18(weights=None)
    # Thay conv1 và bỏ maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # Thay head
    model.fc = nn.Linear(512, num_classes)
    return model


# ─────────────────────────────────────────────
# HBCC Network (Lightweight CoC)
# ─────────────────────────────────────────────

class HBCCNet(nn.Module):
    """
    Lightweight Context Cluster Network.

    Architecture (4 stage giống CoC):
    Stem     : Conv 3×3 stride 1 → 32 channels (CIFAR)
    Stage 1  : 32→32,  HBCCBlocks × N1,  downsampler stride=2 → 16×16
    Stage 2  : 32→64,  HBCCBlocks × N2,  downsampler stride=2 → 8×8
    Stage 3  : 64→128, HBCCBlocks × N3,  downsampler stride=2 → 4×4
    Stage 4  : 128→256, HBCCBlocks × N4, downsampler stride=2 → 2×2
    Head     : GAP → LayerNorm → Linear(256, num_classes)

    Ablation flags truyền vào từ config để bật/tắt từng kỹ thuật.
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
        # Ablation flags
        use_linear_bottleneck: bool = False,
        use_point_shrink: bool = False,
        use_hamming: bool = False,
        use_channel_shuffle: bool = False,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        if embed_dims is None:
            embed_dims = [32, 64, 128, 256]
        if depths is None:
            depths = [2, 2, 4, 2]

        self.ablation_flags = {
            "use_linear_bottleneck": use_linear_bottleneck,
            "use_point_shrink":      use_point_shrink,
            "use_hamming":           use_hamming,
            "use_channel_shuffle":   use_channel_shuffle,
        }

        # Stem: stride=1 cho CIFAR-10
        self.stem = nn.Sequential(
            nn.Conv2d(3, embed_dims[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
        )

        # Stages + Downsamplers
        self.stages       = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            # Downsampler giữa stages
            if i > 0:
                self.downsamplers.append(nn.Sequential(
                    nn.Conv2d(embed_dims[i - 1], dim,
                              kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm2d(dim),
                ))
            else:
                self.downsamplers.append(nn.Identity())

            blocks = nn.Sequential(*[
                HBCCBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    proposal_w=proposal_w,
                    proposal_h=proposal_h,
                    heads=heads,
                    head_dim=head_dim,
                    use_linear_bottleneck=use_linear_bottleneck,
                    use_point_shrink=use_point_shrink,
                    use_hamming=use_hamming,
                    use_channel_shuffle=use_channel_shuffle,
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

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, (down, stage) in enumerate(zip(self.downsamplers, self.stages)):
            if i > 0:
                x = down(x)
            x = stage(x)
        return self.head(x)


# ─────────────────────────────────────────────
# Factory: build_model từ config dict
# ─────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    """
    Tạo model từ config YAML.

    cfg['model'] options:
        'resnet18'     → ResNet-18 CIFAR baseline
        'coc_baseline' → CoC gốc CIFAR (tiny hoặc small)
        'hbcc'         → HBCC Lightweight

    Returns:
        model: nn.Module
    """
    model_name  = cfg.get("model", "resnet18")
    num_classes = cfg.get("num_classes", 10)

    if model_name == "resnet18":
        model = build_resnet18_cifar(num_classes=num_classes)

    elif model_name == "coc_baseline":
        # coc_small_cifar hardcodes heads=4, head_dim=16 — không truyền lại để tránh conflict
        model = coc_small_cifar(
            num_classes=num_classes,
            proposal_w=cfg.get("coc_proposal_w", 2),
            proposal_h=cfg.get("coc_proposal_h", 2),
            fold_w=cfg.get("coc_fold_w", 1),
            fold_h=cfg.get("coc_fold_h", 1),
        )

    elif model_name == "hbcc":
        model = HBCCNet(
            num_classes=num_classes,
            embed_dims=[32, 64, 128, 256],
            depths=[2, 2, 4, 2],
            heads=cfg.get("coc_heads", 4),
            head_dim=cfg.get("coc_head_dim", 16),
            proposal_w=cfg.get("coc_proposal_w", 2),
            proposal_h=cfg.get("coc_proposal_h", 2),
            use_linear_bottleneck=cfg.get("use_linear_bottleneck", False),
            use_point_shrink=cfg.get("use_point_shrink", False),
            use_hamming=cfg.get("use_hamming", False),
            use_channel_shuffle=cfg.get("use_channel_shuffle", False),
        )

    else:
        raise ValueError(f"Model không hỗ trợ: {model_name}. "
                         f"Chọn: resnet18 | coc_baseline | hbcc")

    return model
