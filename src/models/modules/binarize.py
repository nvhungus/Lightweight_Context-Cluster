"""
binarize.py — Nhị phân hóa với Straight-Through Estimator (STE)

Vấn đề: hàm Sign(x) có đạo hàm = 0 ở mọi nơi → gradient vanishing.
Giải pháp: STE — forward dùng Sign, backward dùng identity (clamp [-1,1]).
"""
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# 1. STE Function (custom autograd)
# ─────────────────────────────────────────────

class STESignFunction(torch.autograd.Function):
    """
    Forward : r_out = Sign(r_in) ∈ {-1, +1}
    Backward: ∂L/∂r_in = ∂L/∂r_out  nếu |r_in| ≤ 1, else 0
              (Hard-Tanh STE theo BNN / DoReFa-Net)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.sign()    # -1 nếu x < 0, +1 nếu x >= 0 (0 → +1 theo PyTorch)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        # Chỉ truyền gradient qua những phần tử trong [-1, 1]
        grad_input = grad_output * (x.abs() <= 1).float()
        return grad_input


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    """Áp dụng Sign với STE. Kết quả ∈ {-1, +1}."""
    return STESignFunction.apply(x)


# ─────────────────────────────────────────────
# 2. Module có thể dùng trong nn.Sequential
# ─────────────────────────────────────────────

class BinarizeLayer(nn.Module):
    """
    Layer nhị phân hóa với STE.
    Tùy chọn normalize về [-1, 1] trước khi sign để tránh bão hòa.

    Args:
        normalize: nếu True, dùng BatchNorm trước khi sign
        num_features: số channel (cần nếu normalize=True)
    """

    def __init__(self, normalize: bool = True, num_features: int = None):
        super().__init__()
        self.normalize = normalize
        if normalize and num_features is not None:
            self.bn = nn.BatchNorm2d(num_features, affine=True)
        else:
            self.bn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        return ste_sign(x)


# ─────────────────────────────────────────────
# 3. Binarize tensor ngoài module (dùng trong cluster_block)
# ─────────────────────────────────────────────

def binarize_tensor(x: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
    """
    Nhị phân hóa tensor tùy ý.

    Args:
        x:       tensor đầu vào (bất kỳ shape)
        use_ste: nếu True, dùng STE (training); nếu False, dùng sign thuần (inference)

    Returns:
        binary tensor ∈ {-1, +1}
    """
    if use_ste:
        return ste_sign(x)
    return x.sign()
