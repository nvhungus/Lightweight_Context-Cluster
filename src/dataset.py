"""
dataset.py — Tải và xử lý CIFAR-10
Hỗ trợ cả local (torchvision auto-download) và Kaggle (/kaggle/input/...)
"""
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# ─────────────────────────────────────────────
# 1. Augmentation
# ─────────────────────────────────────────────

def build_transforms(cfg: dict, is_train: bool):
    """
    Xây dựng transform pipeline từ config.

    CIFAR-10 mean/std chuẩn:
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    img_size = cfg.get("img_size", 32)

    if is_train:
        aug_list = []

        if cfg.get("use_random_crop", True):
            aug_list.append(transforms.RandomCrop(img_size, padding=4))

        if cfg.get("use_random_flip", True):
            aug_list.append(transforms.RandomHorizontalFlip())

        if cfg.get("use_autoaugment", False):
            aug_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))

        aug_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

        if cfg.get("use_cutout", False):
            aug_list.append(Cutout(n_holes=1, length=16))

        return transforms.Compose(aug_list)

    else:
        # Validation / Test: chỉ normalize
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


class Cutout:
    """
    Che ngẫu nhiên một vùng vuông trên ảnh (sau khi ToTensor).
    Tham khảo: https://arxiv.org/abs/1708.04552
    """

    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length  = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        mask = torch.ones_like(img)
        for _ in range(self.n_holes):
            cx = torch.randint(w, (1,)).item()
            cy = torch.randint(h, (1,)).item()
            x1 = max(0, cx - self.length // 2)
            x2 = min(w, cx + self.length // 2)
            y1 = max(0, cy - self.length // 2)
            y2 = min(h, cy + self.length // 2)
            mask[:, y1:y2, x1:x2] = 0.0
        return img * mask


# ─────────────────────────────────────────────
# 2. Mixup (áp dụng trên batch)
# ─────────────────────────────────────────────

class MixupCollator:
    """
    Collate function áp dụng Mixup cho mỗi batch.
    Trả về (mixed_images, (labels_a, labels_b, lam)).
    Dùng Mixup loss = lam * CE(pred, a) + (1-lam) * CE(pred, b).
    """

    def __init__(self, alpha: float = 0.2, num_classes: int = 10):
        self.alpha       = alpha
        self.num_classes = num_classes

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        idx = torch.randperm(images.size(0))

        mixed = lam * images + (1 - lam) * images[idx]
        return mixed, labels, labels[idx], lam


# ─────────────────────────────────────────────
# 3. Build DataLoader
# ─────────────────────────────────────────────

def build_dataloaders(cfg: dict):
    """
    Tạo train_loader và val_loader cho CIFAR-10.

    Tự động detect:
    - Nếu data_dir tồn tại và chứa dữ liệu CIFAR-10 → dùng luôn
    - Nếu không → torchvision tự tải về data_dir (cần internet)
    - Trên Kaggle: đặt data_dir = '/kaggle/input/cifar-10-python'

    Args:
        cfg: dict từ file YAML

    Returns:
        train_loader, val_loader
    """
    data_dir    = cfg.get("data_dir", "data/raw")
    batch_size  = cfg.get("batch_size", 128)
    num_workers = cfg.get("num_workers", 4)
    pin_memory  = cfg.get("pin_memory", True)
    use_mixup   = cfg.get("use_mixup", False)

    # Kiểm tra xem có phải đường dẫn Kaggle không
    is_kaggle = "kaggle" in data_dir.lower()

    train_transform = build_transforms(cfg, is_train=True)
    val_transform   = build_transforms(cfg, is_train=False)

    if is_kaggle:
        # Trên Kaggle: CIFAR-10 đã giải nén sẵn
        train_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=True,  download=False, transform=train_transform)
        val_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=False, transform=val_transform)
    else:
        # Local: tự tải nếu chưa có
        os.makedirs(data_dir, exist_ok=True)
        train_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=True,  download=True, transform=train_transform)
        val_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=val_transform)

    # Collate function (có/không có Mixup)
    collate_fn = MixupCollator(num_classes=cfg.get("num_classes", 10)) if use_mixup else None

    train_loader = DataLoader(
        train_set,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        collate_fn  = collate_fn,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    return train_loader, val_loader
