"""
test_bottleneck.py — Tests cho LinearBottleneck, ChannelShuffle, BranchFusion

Kiểm tra:
1. LinearBottleneck luôn split đúng 2 tensor shape cân bằng
2. Output Bottleneck có thể âm — không có activation sau reduce
3. Kiểm tra cấu trúc: lớp cuối của reduce KHÔNG là activation
4. ChannelShuffle không thay đổi shape, chỉ hoán vị channels
5. ChannelShuffle là involution: áp dụng 2 lần = identity (với groups=2)
6. BranchFusion (learned): concat + shuffle + conv → shape đúng
7. BranchFusion (channel shuffle only): concat + shuffle → shape đúng
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.models.modules.bottleneck import LinearBottleneck, ChannelShuffle, BranchFusion


# ─────────────────────────────────────────────
# 1 & 2. LinearBottleneck output shapes và tính Linear
# ─────────────────────────────────────────────

def test_linear_bottleneck_output_shapes():
    """LinearBottleneck luôn split đúng 2 tensor, mỗi tensor có out_dim//2 channels."""
    for in_dim, out_dim in [(16, 16), (32, 32), (32, 64), (64, 128)]:
        bottleneck = LinearBottleneck(in_dim=in_dim, out_dim=out_dim)
        x = torch.randn(2, in_dim, 8, 8)
        local_feat, global_feat = bottleneck(x)

        expected_c = out_dim // 2
        assert local_feat.shape == (2, expected_c, 8, 8), \
            f"local_feat: {local_feat.shape} != (2, {expected_c}, 8, 8)"
        assert global_feat.shape == (2, expected_c, 8, 8), \
            f"global_feat: {global_feat.shape} != (2, {expected_c}, 8, 8)"
        assert local_feat.shape == global_feat.shape, \
            "local_feat và global_feat phải cùng shape!"

    print("  PASS: LinearBottleneck split đúng 2 tensor cân bằng "
          "cho in/out = (16,16), (32,32), (32,64), (64,128)")


def test_linear_bottleneck_spatial_preserved():
    """LinearBottleneck không thay đổi H, W (stride=1)."""
    bottleneck = LinearBottleneck(32, 32)
    for H, W in [(8, 8), (16, 16), (4, 4)]:
        x = torch.randn(2, 32, H, W)
        local_feat, global_feat = bottleneck(x)
        assert local_feat.shape[2:] == (H, W), \
            f"Spatial bị thay đổi: {local_feat.shape[2:]} != ({H},{W})"
    print("  PASS: LinearBottleneck giữ nguyên spatial dims H, W")


def test_linear_bottleneck_output_can_be_negative():
    """
    Output Linear Bottleneck PHẢI có giá trị âm.
    Nếu có ReLU sau reduce thì tất cả output ≥ 0 → test sẽ fail.
    """
    torch.manual_seed(42)
    bottleneck = LinearBottleneck(32, 32)
    x = torch.randn(2, 32, 8, 8)

    with torch.no_grad():
        local_feat, global_feat = bottleneck(x)
        combined = torch.cat([local_feat, global_feat], dim=1)

    has_negative = combined.min().item() < 0
    assert has_negative, \
        "Output bottleneck không có giá trị âm — có thể có ReLU sau reduce! " \
        f"min={combined.min().item():.4f}"
    print(f"  PASS: Output Bottleneck có giá trị âm (min={combined.min().item():.4f}) "
          f"→ không có activation sau reduce")


def test_reduce_layer_no_activation():
    """
    Kiểm tra cấu trúc module: lớp cuối của reduce không phải activation.
    Đây là đặc điểm quan trọng của Linear Bottleneck (theo MobileNetV2).
    """
    activation_types = (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.LeakyReLU,
                        nn.Hardswish, nn.ELU, nn.Sigmoid, nn.Tanh)

    bottleneck = LinearBottleneck(32, 32)
    last_module = list(bottleneck.reduce.children())[-1]

    assert not isinstance(last_module, activation_types), \
        f"Lớp cuối của reduce là activation: {type(last_module).__name__}! " \
        f"Vi phạm nguyên tắc Linear Bottleneck."
    print(f"  PASS: Lớp cuối của reduce là {type(last_module).__name__} "
          f"(không phải activation) ✓")


def test_linear_bottleneck_odd_dim_raises():
    """LinearBottleneck phải raise AssertionError khi out_dim lẻ."""
    try:
        LinearBottleneck(32, 33)   # 33 là số lẻ
        assert False, "Phải raise AssertionError cho out_dim lẻ!"
    except AssertionError:
        pass
    print("  PASS: LinearBottleneck raise lỗi đúng khi out_dim lẻ")


# ─────────────────────────────────────────────
# 3 & 4. ChannelShuffle
# ─────────────────────────────────────────────

def test_channel_shuffle_preserves_shape():
    """ChannelShuffle không thay đổi shape."""
    shuffle = ChannelShuffle(groups=2)
    for shape in [(2, 8, 4, 4), (1, 16, 8, 8), (4, 4, 2, 2)]:
        x = torch.randn(*shape)
        out = shuffle(x)
        assert out.shape == x.shape, \
            f"ChannelShuffle thay đổi shape: {out.shape} != {x.shape}"
    print("  PASS: ChannelShuffle giữ nguyên shape")


def test_channel_shuffle_changes_channel_order():
    """ChannelShuffle phải hoán vị thứ tự channels (không phải identity)."""
    shuffle = ChannelShuffle(groups=2)
    C = 4
    # Tạo input với mỗi channel có giá trị riêng biệt để dễ track
    x = torch.arange(C, dtype=torch.float).view(1, C, 1, 1).expand(1, C, 4, 4).clone()
    out = shuffle(x)

    # Lấy giá trị channel tại pixel (0,0) — phải khác thứ tự ban đầu
    original_order = x[0, :, 0, 0].tolist()    # [0, 1, 2, 3]
    shuffled_order = out[0, :, 0, 0].tolist()   # kỳ vọng [0, 2, 1, 3]

    assert original_order != shuffled_order, \
        f"ChannelShuffle không thay đổi thứ tự channel! " \
        f"original={original_order}, shuffled={shuffled_order}"
    print(f"  PASS: ChannelShuffle hoán vị channel: {original_order} → {shuffled_order}")


def test_channel_shuffle_preserves_all_values():
    """ChannelShuffle không mất hoặc tạo thêm giá trị — chỉ sắp xếp lại."""
    shuffle = ChannelShuffle(groups=2)
    x = torch.randn(2, 8, 4, 4)
    out = shuffle(x)

    # Sort global (flatten trước) để so sánh tập giá trị — không phụ thuộc thứ tự
    x_sorted   = x.flatten().sort()[0]
    out_sorted = out.flatten().sort()[0]
    assert torch.allclose(x_sorted, out_sorted, atol=1e-6), \
        "ChannelShuffle làm mất hoặc thêm giá trị!"
    print("  PASS: ChannelShuffle giữ nguyên tập giá trị (chỉ sắp xếp lại)")


def test_channel_shuffle_double_is_identity():
    """
    Với groups=2 và C=4 (C//groups=2), áp dụng 2 lần trả về tensor gốc.
    Tính chất involution chỉ đúng khi C//groups == groups (vuông: 2×2, 4×4...).
    """
    shuffle = ChannelShuffle(groups=2)
    # C=4 → [0,1,2,3] → shuffle → [0,2,1,3] → shuffle → [0,1,2,3] = identity ✓
    x = torch.randn(2, 4, 4, 4)
    out = shuffle(shuffle(x))
    assert torch.allclose(x, out), \
        "Shuffle×2 không bằng identity với C=4, groups=2!"
    print("  PASS: ChannelShuffle(groups=2, C=4) × 2 = Identity")


# ─────────────────────────────────────────────
# 5 & 6. BranchFusion
# ─────────────────────────────────────────────

def test_branch_fusion_learned_output_shape():
    """BranchFusion với learned fusion cho output đúng shape."""
    in_dim, out_dim = 16, 32
    fusion = BranchFusion(in_dim=in_dim, out_dim=out_dim, use_learned_fusion=True)
    local_feat  = torch.randn(2, in_dim, 8, 8)
    global_feat = torch.randn(2, in_dim, 8, 8)
    out = fusion(local_feat, global_feat)
    assert out.shape == (2, out_dim, 8, 8), \
        f"Learned fusion shape sai: {out.shape}"
    print(f"  PASS: BranchFusion (learned) output shape: {out.shape}")


def test_branch_fusion_shuffle_only_output_shape():
    """BranchFusion với channel shuffle only: 2*in_dim = out_dim."""
    in_dim = 16
    out_dim = in_dim * 2   # 32
    fusion = BranchFusion(in_dim=in_dim, out_dim=out_dim, use_learned_fusion=False)
    local_feat  = torch.randn(2, in_dim, 8, 8)
    global_feat = torch.randn(2, in_dim, 8, 8)
    out = fusion(local_feat, global_feat)
    assert out.shape == (2, out_dim, 8, 8), \
        f"Shuffle-only fusion shape sai: {out.shape}"
    print(f"  PASS: BranchFusion (shuffle-only) output shape: {out.shape}")


def test_branch_fusion_contains_both_inputs():
    """BranchFusion phải giữ thông tin từ cả 2 nhánh (không drop)."""
    in_dim = 16
    fusion = BranchFusion(in_dim=in_dim, out_dim=in_dim * 2, use_learned_fusion=False)
    # Tạo local và global với giá trị rất khác nhau
    local_feat  = torch.ones(1, in_dim, 4, 4)
    global_feat = torch.zeros(1, in_dim, 4, 4)
    out = fusion(local_feat, global_feat)

    # Output phải có cả 1.0 và 0.0 (từ cả 2 nhánh)
    assert (out == 1.0).any(), "BranchFusion đã mất thông tin từ local_feat!"
    assert (out == 0.0).any(), "BranchFusion đã mất thông tin từ global_feat!"
    print("  PASS: BranchFusion giữ thông tin từ cả 2 nhánh")


def test_branch_fusion_backward():
    """Gradient truyền được qua BranchFusion về cả 2 nhánh."""
    in_dim = 16
    fusion = BranchFusion(in_dim=in_dim, out_dim=in_dim * 2, use_learned_fusion=True)
    local_feat  = torch.randn(2, in_dim, 8, 8, requires_grad=True)
    global_feat = torch.randn(2, in_dim, 8, 8, requires_grad=True)

    out = fusion(local_feat, global_feat)
    out.sum().backward()

    assert local_feat.grad is not None and local_feat.grad.abs().sum() > 0, \
        "Gradient không đến local_feat!"
    assert global_feat.grad is not None and global_feat.grad.abs().sum() > 0, \
        "Gradient không đến global_feat!"
    print("  PASS: BranchFusion gradient truyền về cả local và global")


if __name__ == "__main__":
    print("=== test_bottleneck.py ===")
    test_linear_bottleneck_output_shapes()
    test_linear_bottleneck_spatial_preserved()
    test_linear_bottleneck_output_can_be_negative()
    test_reduce_layer_no_activation()
    test_linear_bottleneck_odd_dim_raises()
    test_channel_shuffle_preserves_shape()
    test_channel_shuffle_changes_channel_order()
    test_channel_shuffle_preserves_all_values()
    test_channel_shuffle_double_is_identity()
    test_branch_fusion_learned_output_shape()
    test_branch_fusion_shuffle_only_output_shape()
    test_branch_fusion_contains_both_inputs()
    test_branch_fusion_backward()
    print("=== TẤT CẢ TESTS PASSED ===\n")
