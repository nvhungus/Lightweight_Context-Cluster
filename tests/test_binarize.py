"""
test_binarize.py — Unit test cho binarize.py

Kiểm tra:
1. Output chỉ gồm {-1, +1}
2. Gradient có truyền được qua STE không (không bị vanishing)
3. BinarizeLayer hoạt động với batch input
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.modules.binarize import ste_sign, binarize_tensor, BinarizeLayer


def test_ste_sign_output_binary():
    """Output phải chỉ gồm -1 và +1."""
    x = torch.randn(4, 16, 8, 8)
    out = ste_sign(x)
    unique_vals = out.unique()
    assert set(unique_vals.tolist()).issubset({-1.0, 1.0}), \
        f"Output chứa giá trị ngoài {{-1, +1}}: {unique_vals}"
    print("  PASS: ste_sign output ∈ {-1, +1}")


def test_ste_gradient_flow():
    """Gradient phải truyền được qua (không bằng 0 hết)."""
    x = torch.randn(4, 16, requires_grad=True)
    out = ste_sign(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient là None!"
    assert x.grad.abs().sum() > 0, "Gradient bằng 0 hết — STE không hoạt động!"
    print(f"  PASS: Gradient flow OK. Mean |grad| = {x.grad.abs().mean():.4f}")


def test_ste_gradient_clamp():
    """Gradient chỉ được truyền ở những phần tử |x| <= 1."""
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    out = ste_sign(x)
    out.sum().backward()
    # |x| > 1: grad = 0 ;  |x| <= 1: grad = 1
    expected = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])
    assert torch.allclose(x.grad, expected), \
        f"Gradient STE sai: {x.grad} (kỳ vọng {expected})"
    print("  PASS: STE gradient clamp đúng")


def test_binarize_tensor_no_grad():
    """binarize_tensor với use_ste=False không cần grad."""
    x = torch.randn(2, 8)
    out = binarize_tensor(x, use_ste=False)
    assert out.unique().tolist() == sorted(set(out.unique().tolist()))
    assert set(out.unique().tolist()).issubset({-1.0, 1.0})
    print("  PASS: binarize_tensor (no STE) output đúng")


def test_binarize_layer():
    """BinarizeLayer batch forward + backward."""
    layer = BinarizeLayer(normalize=True, num_features=32)
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    out = layer(x)
    assert out.shape == x.shape, "Shape không khớp"
    assert set(out.unique().tolist()).issubset({-1.0, 1.0})
    out.sum().backward()
    assert x.grad is not None
    print("  PASS: BinarizeLayer forward + backward OK")


if __name__ == "__main__":
    print("=== test_binarize.py ===")
    test_ste_sign_output_binary()
    test_ste_gradient_flow()
    test_ste_gradient_clamp()
    test_binarize_tensor_no_grad()
    test_binarize_layer()
    print("=== TẤT CẢ TESTS PASSED ===\n")
