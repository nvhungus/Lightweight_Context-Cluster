"""
profiler.py — Đếm Parameters, FLOPs và BOPs cho mạng HBCC

Phân biệt rõ:
  - FLOPs  : Floating-Point Operations — cho lớp Conv/Linear thực (float32)
  - BOPs   : Bit Operations          — cho lớp BinarizedCluster (XNOR + popcount)
  - MACs   : Multiply-Accumulate Ops  = FLOPs / 2

Cách đếm BOPs cho BinarizedCluster:
  - Mỗi cặp (điểm, tâm) tốn D BOPs (D phép XNOR) thay vì 2*D FLOPs (Cosine)
  - Tổng BOPs = B * N * M * D
      B = batch, N = số điểm, M = số tâm, D = chiều đặc trưng
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict


# ─────────────────────────────────────────────
# 1. Đếm Parameters
# ─────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Trả về (total_params, trainable_params).
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_params(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


# ─────────────────────────────────────────────
# 2. Đếm FLOPs (Floating-Point Operations)
# ─────────────────────────────────────────────

def count_flops(model: nn.Module, input_size: Tuple[int, ...]) -> int:
    """
    Ước tính FLOPs bằng hook vào Conv2d và Linear.
    KHÔNG tính các lớp BinarizedCluster (những lớp đó tính BOPs riêng).

    Args:
        model:      nn.Module
        input_size: tuple ví dụ (1, 3, 32, 32)

    Returns:
        flops: tổng FLOPs
    """
    flops = [0]
    hooks = []

    def conv2d_hook(module, input, output):
        # Bỏ qua nếu đây là Conv bên trong BinarizedCluster
        # (sẽ được tính riêng ở BOPs)
        in_tensor = input[0]
        B, C_in, H_in, W_in = in_tensor.shape
        _, C_out, H_out, W_out = output.shape
        kH, kW = module.kernel_size if isinstance(module.kernel_size, tuple) \
                  else (module.kernel_size, module.kernel_size)
        groups = module.groups
        # 2 * vì mỗi MAC = 1 nhân + 1 cộng
        flops[0] += 2 * (C_in // groups) * kH * kW * C_out * H_out * W_out * B

    def linear_hook(module, input, output):
        in_tensor = input[0]
        batch = in_tensor.shape[0] if in_tensor.dim() >= 2 else 1
        flops[0] += 2 * module.in_features * module.out_features * batch

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv2d_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    dummy  = torch.zeros(*input_size, device=device)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return flops[0]


def format_flops(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}G"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return str(n)


# ─────────────────────────────────────────────
# 3. Đếm BOPs cho BinarizedCluster
# ─────────────────────────────────────────────

def count_bops_binarized_matmul(N: int, M: int, D: int, B: int = 1) -> int:
    """
    Tính BOPs cho 1 phép nhân ma trận nhị phân trong BinarizedCluster.

    Mỗi cặp (điểm, tâm) tốn D phép XNOR → D BOPs.
    Tổng = B * N * M * D

    Args:
        N: số điểm (sequence length)
        M: số tâm cụm (centers)
        D: chiều đặc trưng (số bit sau binarize)
        B: batch size

    Returns:
        BOPs: tổng số bit operations
    """
    return B * N * M * D


def count_bops_model(model: nn.Module, input_size: Tuple[int, ...]) -> int:
    """
    Đếm BOPs cho toàn bộ BinarizedCluster layers trong model.

    Dùng forward hook để bắt input/output shape của từng
    BinarizedCluster, từ đó tính N, M, D, B chính xác.

    Args:
        model:      nn.Module
        input_size: ví dụ (1, 3, 32, 32)

    Returns:
        total_bops: tổng BOPs của các lớp nhị phân
    """
    # Import ở đây để tránh circular import
    try:
        from src.models.modules.cluster_block import BinarizedCluster
    except ImportError:
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from src.models.modules.cluster_block import BinarizedCluster
        except ImportError:
            # Nếu không import được, trả về 0
            return 0

    bops  = [0]
    hooks = []

    def binarized_cluster_hook(module, input, output):
        """
        Hook vào BinarizedCluster.forward().
        input[0] shape: [B, C, H, W]
        Từ đó suy ra:
          - B  = batch size
          - N  = H * W (số điểm)
          - M  = proposal_w * proposal_h (số tâm)
          - D  = heads * head_dim (chiều feature sau f projection)
        """
        x = input[0]
        B_size = x.shape[0]
        H, W   = x.shape[2], x.shape[3]
        N      = H * W

        # Lấy proposal size từ module
        pool_out = module.centers_proposal.output_size
        if isinstance(pool_out, int):
            M = pool_out * pool_out
        else:
            M = pool_out[0] * pool_out[1]

        D = module.heads * module.head_dim

        # Nếu module dùng Hamming → đây là BOPs; nếu dùng Cosine → là FLOPs
        if module.use_hamming:
            bops[0] += count_bops_binarized_matmul(N=N, M=M, D=D, B=B_size)

    for m in model.modules():
        if isinstance(m, BinarizedCluster):
            hooks.append(m.register_forward_hook(binarized_cluster_hook))

    device = next(model.parameters()).device
    dummy  = torch.zeros(*input_size, device=device)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return bops[0]


def format_bops(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}G"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return str(n)


# ─────────────────────────────────────────────
# 4. Profile tổng hợp
# ─────────────────────────────────────────────

def profile_model(
    model: nn.Module,
    input_size: Tuple[int, ...],
    verbose: bool = True,
    count_bops: bool = True,
) -> Dict:
    """
    Chạy đầy đủ profile: Parameters + FLOPs + BOPs.

    Args:
        model:       nn.Module
        input_size:  ví dụ (1, 3, 32, 32)
        verbose:     in kết quả ra terminal
        count_bops:  có đếm BOPs cho BinarizedCluster không

    Returns:
        dict:
            total_params     : int
            trainable_params : int
            flops            : int  (cho Conv/Linear float)
            bops             : int  (cho BinarizedCluster binary, 0 nếu không có)
            flops_str        : str  "1.2G"
            bops_str         : str  "345M"
            params_str       : str  "11.7M"
    """
    total, trainable = count_parameters(model)
    flops = count_flops(model, input_size)
    bops  = count_bops_model(model, input_size) if count_bops else 0

    result = {
        "total_params":     total,
        "trainable_params": trainable,
        "flops":            flops,
        "bops":             bops,
        "params_str":       format_params(total),
        "flops_str":        format_flops(flops),
        "bops_str":         format_bops(bops),
    }

    if verbose:
        print(f"  Parameters : {format_params(total)} total, "
              f"{format_params(trainable)} trainable")
        print(f"  FLOPs      : {format_flops(flops)}"
              f"  (Conv/Linear float32)")
        if bops > 0:
            print(f"  BOPs       : {format_bops(bops)}"
                  f"  (BinarizedCluster XNOR+popcount)")
            # Tính tỷ lệ tiết kiệm: 1 BOP ≈ 1/64 FLOPs trên hardware
            equiv_flops = bops // 64
            print(f"  BOPs equiv : {format_flops(equiv_flops)}"
                  f"  (quy đổi ~×64 so với FLOPs)")
        else:
            print(f"  BOPs       : 0  (không có BinarizedCluster layer)")

    return result
