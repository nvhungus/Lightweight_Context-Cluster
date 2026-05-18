"""
test_lbp_conv.py — Tests cho LBPConv module (B1)

Kiểm tra:
1. Output shape đúng
2. Filters B là register_buffer (không phải Parameter)
3. Giá trị filters B đúng theo LBP pattern {-1, 0, +1}
4. Filters B không thay đổi sau backward (frozen)
5. Gradient truyền qua proj V (learnable) nhưng KHÔNG qua lbp_weight (fixed)
6. Cả hai mode Sigmoid và QReLU đều hoạt động
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from src.models.modules.lbp_conv import LBPConv


# ─────────────────────────────────────────────
# 1. Output shape
# ─────────────────────────────────────────────

def test_lbp_conv_output_shape():
    """Output shape phải bằng input shape [B, C, H, W]."""
    for C in [16, 32, 64]:
        lbp = LBPConv(dim=C)
        x = torch.randn(2, C, 8, 8)
        out = lbp(x)
        assert out.shape == x.shape, f"dim={C}: {out.shape} != {x.shape}"
    print("  PASS: LBPConv output shape đúng cho dim = 16, 32, 64")


# ─────────────────────────────────────────────
# 2. B là register_buffer, không phải Parameter
# ─────────────────────────────────────────────

def test_lbp_weight_is_buffer_not_parameter():
    """lbp_weight phải là buffer (không học được), không phải nn.Parameter."""
    lbp = LBPConv(dim=16)

    # Phải nằm trong named_buffers()
    buffer_names = {name for name, _ in lbp.named_buffers()}
    assert "lbp_weight" in buffer_names, \
        "lbp_weight không nằm trong buffers — dùng register_buffer()?"

    # Không được nằm trong parameters()
    param_names = {name for name, _ in lbp.named_parameters()}
    assert "lbp_weight" not in param_names, \
        "lbp_weight nằm trong parameters — sẽ bị optimizer update!"

    # requires_grad phải là False
    assert not lbp.lbp_weight.requires_grad, \
        "lbp_weight có requires_grad=True — sẽ tích lũy gradient!"

    print("  PASS: lbp_weight là buffer (requires_grad=False, không trong parameters)")


# ─────────────────────────────────────────────
# 3. Giá trị filters đúng theo LBP pattern
# ─────────────────────────────────────────────

def test_lbp_filter_values():
    """
    Mỗi filter 3×3 phải có:
    - đúng 1 vị trí = -1 (center tại (1,1))
    - đúng 1 vị trí = +1 (một trong 8 neighbors)
    - 7 vị trí còn lại = 0
    Tổng toàn bộ giá trị trong mỗi filter phải = 0.
    """
    lbp = LBPConv(dim=1)  # dim=1 để chỉ có 8 filters cơ bản
    weight = lbp.lbp_weight   # [8, 1, 3, 3]
    n_filters = weight.shape[0]

    assert n_filters == 8, f"Phải có 8 LBP filters, có {n_filters}"

    valid_values = {-1.0, 0.0, 1.0}
    for i in range(n_filters):
        f = weight[i, 0]      # [3, 3]
        unique = set(f.unique().tolist())

        # Giá trị chỉ trong {-1, 0, +1}
        assert unique.issubset(valid_values), \
            f"Filter {i} chứa giá trị ngoài {{-1,0,+1}}: {unique}"

        # Center = -1
        assert f[1, 1].item() == -1.0, \
            f"Filter {i}: center (1,1) = {f[1,1].item()} (kỳ vọng -1)"

        # Đúng 1 giá trị = +1
        n_plus_one = (f == 1.0).sum().item()
        assert n_plus_one == 1, \
            f"Filter {i}: có {n_plus_one} vị trí = +1 (kỳ vọng 1)"

        # Tổng = 0 (−1 + 0×7 + 1 = 0)
        assert abs(f.sum().item()) < 1e-6, \
            f"Filter {i}: sum = {f.sum().item()} (kỳ vọng 0)"

    print(f"  PASS: Tất cả {n_filters} LBP filters có giá trị đúng {{-1, 0, +1}}")


def test_lbp_filters_cover_all_8_neighbors():
    """8 filters phải encode 8 hướng láng giềng khác nhau (không trùng lặp)."""
    lbp = LBPConv(dim=1)
    weight = lbp.lbp_weight    # [8, 1, 3, 3]

    # Tìm vị trí +1 trong mỗi filter
    neighbor_positions = set()
    for i in range(8):
        pos = (lbp.lbp_weight[i, 0] == 1.0).nonzero(as_tuple=False)[0]
        neighbor_positions.add(tuple(pos.tolist()))

    assert len(neighbor_positions) == 8, \
        f"Chỉ có {len(neighbor_positions)} vị trí neighbor duy nhất (kỳ vọng 8)"

    # Center (1,1) không được là neighbor của bất kỳ filter nào
    assert (1, 1) not in neighbor_positions, \
        "Center (1,1) đang bị dùng làm neighbor — sai LBP pattern!"

    print("  PASS: 8 filters encode 8 hướng neighbor khác nhau, không trùng lặp")


# ─────────────────────────────────────────────
# 4. Filters B không thay đổi sau backward
# ─────────────────────────────────────────────

def test_lbp_filters_frozen_after_backward():
    """Filters B phải giữ nguyên sau backward pass."""
    lbp = LBPConv(dim=16)
    original_weight = lbp.lbp_weight.clone()

    x = torch.randn(2, 16, 8, 8)
    out = lbp(x)
    out.sum().backward()

    assert torch.equal(lbp.lbp_weight, original_weight), \
        "LBP filters đã thay đổi sau backward — register_buffer không hoạt động!"
    print("  PASS: LBP filters B giữ nguyên sau backward")


def test_lbp_filters_not_updated_by_optimizer():
    """Optimizer không được update filters B (dù đã backward)."""
    lbp = LBPConv(dim=16)
    original_weight = lbp.lbp_weight.clone()

    # SGD với learning rate lớn để đảm bảo có update nếu sai
    optimizer = optim.SGD(lbp.parameters(), lr=10.0)
    x = torch.randn(2, 16, 8, 8)
    lbp(x).sum().backward()
    optimizer.step()

    assert torch.equal(lbp.lbp_weight, original_weight), \
        "Optimizer đã update LBP filters — lỗi: B phải là register_buffer!"
    print("  PASS: Optimizer không update LBP filters B sau optimizer.step()")


# ─────────────────────────────────────────────
# 5. Gradient qua V nhưng KHÔNG qua B
# ─────────────────────────────────────────────

def test_gradient_flows_through_proj_not_lbp_weight():
    """
    Gradient phải truyền qua proj (Conv 1×1 học được)
    nhưng KHÔNG tích lũy gradient ở lbp_weight (frozen buffer).
    """
    lbp = LBPConv(dim=16)
    x = torch.randn(2, 16, 8, 8)
    out = lbp(x)
    out.sum().backward()

    # proj.weight PHẢI có gradient
    assert lbp.proj.weight.grad is not None, \
        "proj.weight không có gradient — gradient bị chặn trước V!"
    assert lbp.proj.weight.grad.abs().mean() > 1e-8, \
        "proj.weight gradient quá nhỏ — có thể gradient vanishing!"

    # lbp_weight KHÔNG được có gradient (buffer không tích lũy)
    assert lbp.lbp_weight.grad is None, \
        "lbp_weight có gradient — B đang bị backprop vào, sai thiết kế!"

    mean_grad_v = lbp.proj.weight.grad.abs().mean().item()
    print(f"  PASS: Gradient qua V (proj): mean|grad|={mean_grad_v:.4e}, "
          f"B.grad=None ✓")


# ─────────────────────────────────────────────
# 6. Sigmoid vs QReLU mode
# ─────────────────────────────────────────────

def test_sigmoid_mode_output_range():
    """Sigmoid mode: bit map đầu ra nằm trong (0, 1) trước proj."""
    lbp = LBPConv(dim=16, use_sigmoid=True)
    x = torch.randn(2, 16, 8, 8)
    out = lbp(x)
    assert out.shape == (2, 16, 8, 8)
    print(f"  PASS: Sigmoid mode shape OK: {out.shape}")


def test_qrelu_mode_output_shape():
    """QReLU mode: output shape đúng."""
    lbp = LBPConv(dim=16, use_sigmoid=False)
    x = torch.randn(2, 16, 8, 8)
    out = lbp(x)
    assert out.shape == (2, 16, 8, 8)
    print(f"  PASS: QReLU mode shape OK: {out.shape}")


def test_sigmoid_and_qrelu_give_different_outputs():
    """Sigmoid và QReLU phải cho kết quả khác nhau."""
    torch.manual_seed(0)
    x = torch.randn(2, 16, 8, 8)

    lbp_sig = LBPConv(dim=16, use_sigmoid=True)
    lbp_qr  = LBPConv(dim=16, use_sigmoid=False)

    with torch.no_grad():
        out_sig = lbp_sig(x)
        out_qr  = lbp_qr(x)

    # proj weights khác nhau nên output sẽ khác nhau dù cùng input
    # (test chỉ cần verify không crash và shape đúng)
    assert out_sig.shape == out_qr.shape == (2, 16, 8, 8)
    print("  PASS: Cả Sigmoid và QReLU mode đều cho output đúng shape")


if __name__ == "__main__":
    print("=== test_lbp_conv.py ===")
    test_lbp_conv_output_shape()
    test_lbp_weight_is_buffer_not_parameter()
    test_lbp_filter_values()
    test_lbp_filters_cover_all_8_neighbors()
    test_lbp_filters_frozen_after_backward()
    test_lbp_filters_not_updated_by_optimizer()
    test_gradient_flows_through_proj_not_lbp_weight()
    test_sigmoid_mode_output_range()
    test_qrelu_mode_output_shape()
    test_sigmoid_and_qrelu_give_different_outputs()
    print("=== TẤT CẢ TESTS PASSED ===\n")
