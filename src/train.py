"""
train.py — Vòng lặp training chính

Cách dùng:
    python src/train.py --config configs/ablation_step1_baseline.yaml
    python src/train.py --config configs/ablation_step2_shrink.yaml

Pipeline:
    Load config → Build model → Build data → Build optimizer/scheduler
    → Train loop (với STE cho mạng nhị phân) → Lưu kết quả
"""
import os
import sys
import argparse
import random
import yaml
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# Thêm root vào path để import đúng
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_dataloaders
from src.models import build_model
from src.utils.metrics import AverageMeter, accuracy
from src.utils.logger import TrainLogger
from src.utils.scheduler import build_scheduler
from src.utils.profiler import profile_model
from src.utils.throughput import measure_throughput


# ─────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load base.yaml rồi merge với config cụ thể."""
    base_path = os.path.join(os.path.dirname(config_path), "base.yaml")
    cfg = {}

    if os.path.exists(base_path):
        with open(base_path) as f:
            cfg.update(yaml.safe_load(f))

    with open(config_path) as f:
        override = yaml.safe_load(f)
    cfg.update(override)

    return cfg


# ─────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Build optimizer
# ─────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    opt_name = cfg.get("optimizer", "adamw").lower()
    lr       = cfg.get("lr", 1e-3)
    wd       = cfg.get("weight_decay", 0.05)

    if opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr,
                         momentum=cfg.get("momentum", 0.9),
                         weight_decay=wd, nesterov=True)
    elif opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Optimizer không hỗ trợ: {opt_name}")


# ─────────────────────────────────────────────
# Train 1 epoch
# ─────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: dict,
    scaler=None,
) -> dict:
    model.train()
    loss_meter = AverageMeter("loss")
    acc1_meter = AverageMeter("acc1")

    use_mixup = cfg.get("use_mixup", False)
    log_interval = cfg.get("log_interval", 50)

    for batch_idx, batch in enumerate(loader):
        if use_mixup:
            images, labels_a, labels_b, lam = batch
            images   = images.to(device, non_blocking=True)
            labels_a = labels_a.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)
        else:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                if use_mixup:
                    loss = lam * criterion(outputs, labels_a) + \
                           (1 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if use_mixup:
                loss = lam * criterion(outputs, labels_a) + \
                       (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Metrics (dùng labels gốc nếu có mixup)
        eval_labels = labels if not use_mixup else labels_a
        acc1 = accuracy(outputs.detach(), eval_labels, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1, images.size(0))

        if batch_idx % log_interval == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx}/{len(loader)} "
                  f"Loss: {loss_meter.avg:.4f}  Acc@1: {acc1_meter.avg:.2f}%")

    return {"train_loss": loss_meter.avg, "train_acc1": acc1_meter.avg}


# ─────────────────────────────────────────────
# Validate
# ─────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    loss_meter = AverageMeter("val_loss")
    acc1_meter = AverageMeter("val_acc1")
    acc5_meter = AverageMeter("val_acc5")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss    = criterion(outputs, labels)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1, images.size(0))
        acc5_meter.update(acc5, images.size(0))

    return {
        "val_loss": loss_meter.avg,
        "val_acc1": acc1_meter.avg,
        "val_acc5": acc5_meter.avg,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Lightweight CoC")
    parser.add_argument("--config", type=str, required=True,
                        help="Đường dẫn tới file config YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Đường dẫn checkpoint để tiếp tục train")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    print(f"\n{'='*60}")
    print(f"Config: {args.config}")
    print(f"Model : {cfg.get('model', 'resnet18')}")
    print(f"{'='*60}\n")

    # Seed & device
    set_seed(cfg.get("seed", 42))
    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Data
    train_loader, val_loader = build_dataloaders(cfg)

    # Model
    model = build_model(cfg).to(device)

    # Profile
    print("--- Model Profile ---")
    profile_model(model, input_size=(1, 3, cfg.get("img_size", 32), cfg.get("img_size", 32)))

    # Optimizer & Scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if cfg.get("amp", False) else None

    # Logger
    run_name = f"{cfg.get('model', 'model')}_{cfg.get('dataset', 'cifar10')}"
    logger   = TrainLogger(log_dir=cfg.get("log_dir", "experiments"), run_name=run_name)

    # Resume
    start_epoch = 1
    best_acc    = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc    = ckpt.get("best_acc", 0.0)
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    total_epochs = cfg.get("epochs", 200)
    save_dir     = cfg.get("save_dir", "experiments")
    os.makedirs(save_dir, exist_ok=True)

    # ── Training Loop ──
    for epoch in range(start_epoch, total_epochs + 1):
        lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg, scaler
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        metrics = {**train_metrics, **val_metrics, "lr": lr}
        logger.log(epoch, metrics)
        logger.print_epoch(epoch, total_epochs, metrics)

        # Save best
        if val_metrics["val_acc1"] > best_acc:
            best_acc = val_metrics["val_acc1"]
            ckpt_path = os.path.join(save_dir, f"{run_name}_best.pth")
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc":  best_acc,
                "cfg":       cfg,
            }, ckpt_path)
            print(f"  ★ New best: {best_acc:.2f}% → saved to {ckpt_path}")

    # ── Final Profile & Result ──
    print(f"\n{'='*60}")
    print(f"Training done. Best Val Acc@1: {best_acc:.2f}%")
    print("--- Final Throughput ---")
    tp = measure_throughput(
        model, input_size=(1, 3, cfg.get("img_size", 32), cfg.get("img_size", 32)),
        device=str(device),
    )

    profile_info = profile_model(
        model, input_size=(1, 3, cfg.get("img_size", 32), cfg.get("img_size", 32)),
        verbose=True,
    )

    result = {
        "model":          cfg.get("model"),
        "best_val_acc1":  best_acc,
        "total_params":   profile_info["total_params"],
        "flops":          profile_info["flops"],
        "throughput":     tp["throughput_img_per_sec"],
        "latency_ms":     tp["latency_ms"],
        "ablation_flags": {
            "use_point_shrink":      cfg.get("use_point_shrink", False),
            "use_hamming":           cfg.get("use_hamming", False),
            "use_linear_bottleneck": cfg.get("use_linear_bottleneck", False),
            "use_channel_shuffle":   cfg.get("use_channel_shuffle", False),
        },
    }

    result_path = cfg.get("result_file", f"results/{run_name}.json")
    logger.save_result(result_path, result)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
