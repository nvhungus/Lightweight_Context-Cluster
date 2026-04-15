"""
scheduler.py — Cosine Annealing với Linear Warmup
Phù hợp nhất cho mạng nhị phân theo khuyến nghị trong Theory.md
"""
import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmup(_LRScheduler):
    """
    Cosine Annealing Scheduler với Linear Warmup.

    Giai đoạn 1 (Warmup): lr tăng tuyến tính từ warmup_lr_init → base_lr
    Giai đoạn 2 (Cosine): lr giảm theo cosine từ base_lr → min_lr

    Args:
        optimizer:       PyTorch optimizer
        total_epochs:    tổng số epoch train
        warmup_epochs:   số epoch warmup (mặc định 5)
        min_lr:          lr tối thiểu (mặc định 1e-6)
        warmup_lr_init:  lr khởi đầu của warmup (mặc định 1e-6)
        last_epoch:      epoch bắt đầu (mặc định -1)
    """

    def __init__(
        self,
        optimizer,
        total_epochs: int,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        warmup_lr_init: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.total_epochs    = total_epochs
        self.warmup_epochs   = warmup_epochs
        self.min_lr          = min_lr
        self.warmup_lr_init  = warmup_lr_init
        # base_lr được lấy từ optimizer param_groups khi khởi tạo
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch

        # --- Warmup Phase ---
        if epoch < self.warmup_epochs:
            # tăng tuyến tính
            alpha = epoch / max(self.warmup_epochs, 1)
            return [
                self.warmup_lr_init + alpha * (base_lr - self.warmup_lr_init)
                for base_lr in self.base_lrs
            ]

        # --- Cosine Phase ---
        cosine_epochs = self.total_epochs - self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / max(cosine_epochs, 1)
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cos_val
            for base_lr in self.base_lrs
        ]


def build_scheduler(optimizer, cfg: dict):
    """
    Factory function để tạo scheduler từ config dict.

    Hỗ trợ:
    - cfg['scheduler'] == 'cosine'  → CosineAnnealingWarmup
    - cfg['scheduler'] == 'step'    → StepLR (fallback)
    """
    scheduler_type = cfg.get("scheduler", "cosine")

    if scheduler_type == "cosine":
        return CosineAnnealingWarmup(
            optimizer,
            total_epochs   = cfg["epochs"],
            warmup_epochs  = cfg.get("warmup_epochs", 5),
            min_lr         = cfg.get("min_lr", 1e-6),
            warmup_lr_init = cfg.get("warmup_lr_init", 1e-6),
        )
    elif scheduler_type == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(
            optimizer,
            step_size = cfg.get("lr_step_size", 50),
            gamma     = cfg.get("lr_gamma", 0.1),
        )
    else:
        raise ValueError(f"Scheduler không hỗ trợ: {scheduler_type}")
