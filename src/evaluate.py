"""
evaluate.py — Đánh giá model độc lập từ checkpoint đã lưu

Cách dùng:
    python src/evaluate.py --config configs/ablation_step1_baseline.yaml \
                           --checkpoint experiments/resnet18_cifar10_best.pth
"""
import os
import sys
import argparse
import yaml
import json

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_dataloaders
from src.models import build_model
from src.utils.metrics import AverageMeter, accuracy
from src.utils.profiler import profile_model
from src.utils.throughput import measure_throughput


def load_config(config_path: str) -> dict:
    base_path = os.path.join(os.path.dirname(config_path), "base.yaml")
    cfg = {}
    if os.path.exists(base_path):
        with open(base_path) as f:
            cfg.update(yaml.safe_load(f))
    with open(config_path) as f:
        cfg.update(yaml.safe_load(f))
    return cfg


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    criterion  = nn.CrossEntropyLoss()
    loss_meter = AverageMeter("loss")
    acc1_meter = AverageMeter("acc1")
    acc5_meter = AverageMeter("acc5")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs    = model(images)
        loss       = criterion(outputs, labels)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1, images.size(0))
        acc5_meter.update(acc5, images.size(0))

    return {
        "loss":  loss_meter.avg,
        "acc1":  acc1_meter.avg,
        "acc5":  acc5_meter.avg,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Lightweight CoC")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Đường dẫn file .pth checkpoint")
    parser.add_argument("--device",     type=str, default=None,
                        help="Ghi đè device: cuda | cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device

    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load model
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Checkpoint loaded: {args.checkpoint}")
    print(f"  (Trained to epoch {ckpt.get('epoch', '?')}, "
          f"best acc = {ckpt.get('best_acc', 0.0):.2f}%)")

    # Data
    _, val_loader = build_dataloaders(cfg)

    # Evaluate
    print("\n--- Evaluation ---")
    metrics = evaluate(model, val_loader, device)
    print(f"  Loss  : {metrics['loss']:.4f}")
    print(f"  Acc@1 : {metrics['acc1']:.2f}%")
    print(f"  Acc@5 : {metrics['acc5']:.2f}%")

    # Profile
    img_size = cfg.get("img_size", 32)
    print("\n--- Model Profile ---")
    profile_info = profile_model(model, input_size=(1, 3, img_size, img_size))

    # Throughput
    print("\n--- Throughput ---")
    tp = measure_throughput(model, input_size=(1, 3, img_size, img_size),
                            device=str(device))

    # Lưu kết quả
    result = {
        "checkpoint":   args.checkpoint,
        "model":        cfg.get("model"),
        "acc1":         metrics["acc1"],
        "acc5":         metrics["acc5"],
        "total_params": profile_info["total_params"],
        "flops":        profile_info["flops"],
        "throughput":   tp["throughput_img_per_sec"],
        "latency_ms":   tp["latency_ms"],
    }

    result_path = cfg.get("result_file", "results/eval_result.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nKết quả đã lưu: {result_path}")


if __name__ == "__main__":
    main()
