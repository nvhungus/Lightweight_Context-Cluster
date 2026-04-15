"""
test_point_shrink.py — Unit test cho point_shrink.py

Kiểm tra:
1. Output shape đúng
2. Forward pass không lỗi
3. Gradient truyền được qua
4. PointShrinkV2 (unfold version) shape đúng
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.modules.point_shrink import PointShrink, PointShrinkV2


def test_point_shrink_shape_no_stride():
    """stride=1 → spatial size giữ nguyên."""
    module = PointShrink(in_dim=64, out_dim=64, stride=1)
    x = torch.randn(2, 64, 16, 16)
    out = module(x)
    assert out.shape == (2, 64, 16, 16), f"Shape sai: {out.shape}"
    print(f"  PASS: PointShrink (stride=1) shape = {out.shape}")


def test_point_shrink_shape_with_stride():
    """stride=2 → spatial size giảm một nửa."""
    module = PointShrink(in_dim=64, out_dim=128, stride=2)
    x = torch.randn(2, 64, 16, 16)
    out = module(x)
    assert out.shape == (2, 128, 8, 8), f"Shape sai: {out.shape}"
    print(f"  PASS: PointShrink (stride=2) shape = {out.shape}")


def test_point_shrink_gradient():
    """Gradient truyền được qua PointShrink."""
    module = PointShrink(in_dim=32, out_dim=32, stride=1)
    x = torch.randn(1, 32, 8, 8, requires_grad=True)
    out = module(x)
    out.sum().backward()
    assert x.grad is not None, "Gradient là None!"
    assert x.grad.abs().sum() > 0, "Gradient = 0!"
    print(f"  PASS: PointShrink gradient OK, mean |grad| = {x.grad.abs().mean():.5f}")


def test_point_shrink_v2_shape():
    """PointShrinkV2 (unfold) output shape đúng."""
    module = PointShrinkV2(in_dim=32, out_dim=64, k=3)
    x = torch.randn(2, 32, 16, 16)
    out = module(x)
    assert out.shape == (2, 64, 16, 16), f"Shape sai: {out.shape}"
    print(f"  PASS: PointShrinkV2 shape = {out.shape}")


def test_point_shrink_v2_gradient():
    """PointShrinkV2 gradient flow."""
    module = PointShrinkV2(in_dim=16, out_dim=16, k=3)
    x = torch.randn(1, 16, 8, 8, requires_grad=True)
    out = module(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    print(f"  PASS: PointShrinkV2 gradient OK")


def test_point_shrink_reduces_params():
    """PointShrink có ít params hơn Conv 3×3 chuẩn."""
    dw_pw = PointShrink(in_dim=64, out_dim=64)
    std_conv = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
    params_dw = sum(p.numel() for p in dw_pw.parameters())
    params_std = sum(p.numel() for p in std_conv.parameters())
    assert params_dw < params_std, \
        f"PointShrink phải ít params hơn Conv chuẩn: {params_dw} vs {params_std}"
    print(f"  PASS: PointShrink params={params_dw} < Conv3x3 params={params_std}")


if __name__ == "__main__":
    print("=== test_point_shrink.py ===")
    test_point_shrink_shape_no_stride()
    test_point_shrink_shape_with_stride()
    test_point_shrink_gradient()
    test_point_shrink_v2_shape()
    test_point_shrink_v2_gradient()
    test_point_shrink_reduces_params()
    print("=== TẤT CẢ TESTS PASSED ===\n")
