"""
metrics.py — Tính Top-1, Top-5 Accuracy
"""
import torch


class AverageMeter:
    """Theo dõi giá trị trung bình và tổng qua các batch."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Tính accuracy theo Top-k.

    Args:
        output: logits [B, num_classes]
        target: nhãn đúng [B]
        topk:   tuple các k cần tính, ví dụ (1, 5)

    Returns:
        list[float]: accuracy (%) cho từng k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Lấy top-k predictions
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()                              # [maxk, B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # [maxk, B]

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results
