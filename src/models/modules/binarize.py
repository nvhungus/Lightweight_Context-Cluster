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

# ─────────────────────────────────────────────
# 4. Strict STE Function (Đảm bảo tuyệt đối {-1, 1})
# ─────────────────────────────────────────────

class StrictSTESignFunction(torch.autograd.Function):
    """
    Giống STE thông thường nhưng xử lý triệt để trường hợp x = 0.
    PyTorch x.sign() trả về 0 khi x=0. Hàm này ép x >= 0 thành +1, x < 0 thành -1.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        # Ép nghiêm ngặt về {-1, 1}, không có số 0
        return x.ge(0).float() * 2.0 - 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        grad_input = grad_output * (x.abs() <= 1).float()
        return grad_input

def strict_ste_sign(x: torch.Tensor) -> torch.Tensor:
    return StrictSTESignFunction.apply(x)


# ─────────────────────────────────────────────
# 5. Layer phục vụ Ablation & Trực quan hóa (Notebook 03)
# ─────────────────────────────────────────────

class BinarizeAblationLayer(nn.Module):
    """
    Layer đặc biệt dùng để trực quan hóa sự sụp đổ không gian (Manifold Collapse).
    Cho phép chèn một hàm phi tuyến (vd: ReLU) trước khi binarize để đối chứng.
    """
    def __init__(self, use_activation: bool = False):
        super().__init__()
        self.use_activation = use_activation
        self.act = nn.ReLU() if use_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Nếu use_activation=True: các giá trị âm bị biến thành 0.
        # strict_ste_sign(0) sẽ biến toàn bộ thành +1 -> Mất thông tin!
        x = self.act(x)
        return strict_ste_sign(x)
    
    # src/models/modules/binarize.py

class QReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # Bậc thang: x > 0 thì trả về 1, ngược lại 0
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # STE: Đạo hàm bằng 1 nếu 0 <= x <= 2, ngược lại bằng 0
        mask = (input >= 0) & (input <= 2)
        grad_input[~mask] = 0
        return grad_input

def qrelu(input):
    return QReLUFunction.apply(input)

class QReLU(nn.Module):
    """
    Quantized ReLU Activation với STE.
    Đầu ra nhị phân {0, 1}.
    """
    def forward(self, x):
        return qrelu(x)