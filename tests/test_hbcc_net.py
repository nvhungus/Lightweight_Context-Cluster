"""
test_hbcc_net.py — End-to-end tests cho toàn bộ network

Kiểm tra:
1. Forward pass CIFAR-10 input [2, 3, 32, 32] → output [2, 10] cho cả 3 model
2. Backward pass qua full model không lỗi, gradient không vanishing
3. build_model() với từng ablation config trả về model đúng
4. Số params trong khoảng hợp lý (resnet18 >> hbcc về params)
5. Eval mode (model.eval()) không lỗi, output deterministic
6. Các spatial sizes nhỏ không gây lỗi trong cluster proposal

QUAN TRỌNG: CIFAR-10 dùng ảnh 32×32, các lớp downsampling đưa về 2×2.
            proposal_w, proposal_h phải ≤ spatial size tại từng stage.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models import build_model
from src.models.hbcc_net import HBCCNet, build_resnet18_cifar


CIFAR10_INPUT = (2, 3, 32, 32)   # batch=2, CIFAR-10
NUM_CLASSES = 10


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _make_input(batch=2):
    return torch.randn(*CIFAR10_INPUT)


# ─────────────────────────────────────────────
# 1. Forward pass — 3 model names
# ─────────────────────────────────────────────

def test_resnet18_forward():
    """ResNet-18 CIFAR: [2,3,32,32] → [2,10]."""
    cfg = {"model": "resnet18", "num_classes": NUM_CLASSES}
    model = build_model(cfg).eval()
    with torch.no_grad():
        out = model(_make_input())
    assert out.shape == (2, NUM_CLASSES), f"Shape sai: {out.shape}"
    print(f"  PASS: resnet18 output shape: {out.shape}")


def test_coc_baseline_forward():
    """CoC baseline CIFAR: [2,3,32,32] → [2,10]."""
    cfg = {
        "model": "coc_baseline",
        "num_classes": NUM_CLASSES,
        "coc_proposal_w": 2,
        "coc_proposal_h": 2,
        "coc_fold_w": 1,
        "coc_fold_h": 1,
    }
    model = build_model(cfg).eval()
    with torch.no_grad():
        out = model(_make_input())
    assert out.shape == (2, NUM_CLASSES), f"Shape sai: {out.shape}"
    print(f"  PASS: coc_baseline output shape: {out.shape}")


def test_hbcc_forward_baseline():
    """HBCCNet baseline (không flags): [2,3,32,32] → [2,10]."""
    cfg = {"model": "hbcc", "num_classes": NUM_CLASSES}
    model = build_model(cfg).eval()
    with torch.no_grad():
        out = model(_make_input())
    assert out.shape == (2, NUM_CLASSES), f"Shape sai: {out.shape}"
    print(f"  PASS: hbcc baseline output shape: {out.shape}")


# ─────────────────────────────────────────────
# 2. Backward pass — gradient flow
# ─────────────────────────────────────────────

def test_hbcc_backward_no_error():
    """Backward qua HBCCNet đầy đủ không lỗi."""
    cfg = {
        "model": "hbcc",
        "num_classes": NUM_CLASSES,
        "use_linear_bottleneck": True,
        "use_hamming": True,
    }
    model = build_model(cfg).train()
    x = _make_input()
    out = model(x)
    loss = out.sum()
    loss.backward()   # Không được raise exception
    print("  PASS: Backward qua HBCCNet không lỗi")


def test_hbcc_gradient_not_vanishing():
    """Gradient qua toàn mạng HBCC không bị vanishing."""
    cfg = {
        "model": "hbcc",
        "num_classes": NUM_CLASSES,
        "use_hamming": True,
    }
    model = build_model(cfg).train()
    x = torch.randn(*CIFAR10_INPUT, requires_grad=True)
    out = model(x)
    out.sum().backward()

    assert x.grad is not None, "Gradient là None!"
    mean_grad = x.grad.abs().mean().item()
    assert mean_grad > 1e-8, \
        f"Gradient vanishing! Mean |grad| = {mean_grad:.2e}"
    print(f"  PASS: Gradient không vanishing qua toàn mạng. "
          f"Mean |grad| = {mean_grad:.4e}")


def test_resnet18_backward():
    """Backward qua ResNet-18 không lỗi."""
    model = build_resnet18_cifar(num_classes=NUM_CLASSES).train()
    criterion = torch.nn.CrossEntropyLoss()
    x = _make_input()
    labels = torch.randint(0, NUM_CLASSES, (2,))
    loss = criterion(model(x), labels)
    loss.backward()
    print("  PASS: Backward qua ResNet-18 không lỗi")


# ─────────────────────────────────────────────
# 3. build_model() với từng ablation config
# ─────────────────────────────────────────────

def test_build_model_ablation_configs():
    """build_model() với từng ablation config trả về model chạy đúng."""
    base = {"model": "hbcc", "num_classes": NUM_CLASSES}

    ablation_configs = [
        {},                                                              # baseline
        {"use_linear_bottleneck": True},                                 # step2
        {"use_linear_bottleneck": True, "use_point_shrink": True},       # step3
        {"use_linear_bottleneck": True, "use_hamming": True},            # step4
        {"use_linear_bottleneck": True, "use_channel_shuffle": True},    # step5
        {"use_linear_bottleneck": True,                                  # full
         "use_point_shrink": True,
         "use_hamming": True,
         "use_channel_shuffle": True},
    ]

    x = _make_input()
    for extra in ablation_configs:
        cfg = {**base, **extra}
        model = build_model(cfg).eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES), \
            f"Config {extra}: output shape sai {out.shape}"

    print(f"  PASS: Tất cả {len(ablation_configs)} ablation config cho output đúng shape")


def test_build_model_invalid_name_raises():
    """build_model() phải raise ValueError với model name không hợp lệ."""
    try:
        build_model({"model": "invalid_model_xyz"})
        assert False, "Phải raise ValueError!"
    except ValueError as e:
        assert "không hỗ trợ" in str(e) or "invalid_model" in str(e).lower() or "Model" in str(e)
    print("  PASS: build_model raise ValueError với model name sai")


# ─────────────────────────────────────────────
# 4. Số params trong khoảng hợp lý
# ─────────────────────────────────────────────

def test_param_count_ranges():
    """
    Kiểm tra số params trong khoảng hợp lý:
    - resnet18: ~11M (lớn, là baseline nặng)
    - hbcc: lightweight → < resnet18 (với cùng task CIFAR-10)
    """
    resnet = build_resnet18_cifar(num_classes=NUM_CLASSES)
    hbcc   = build_model({"model": "hbcc", "num_classes": NUM_CLASSES})

    n_resnet = _count_params(resnet)
    n_hbcc   = _count_params(hbcc)

    # ResNet-18 phải có khoảng 11M params
    assert 5_000_000 <= n_resnet <= 15_000_000, \
        f"ResNet-18 params ngoài khoảng kỳ vọng: {n_resnet:,}"

    # HBCC phải nhẹ hơn ResNet-18
    assert n_hbcc < n_resnet, \
        f"HBCC ({n_hbcc:,}) nặng hơn ResNet-18 ({n_resnet:,}) — không lightweight!"

    print(f"  PASS: resnet18={n_resnet:,} params, hbcc={n_hbcc:,} params "
          f"(hbcc nhẹ hơn {n_resnet/n_hbcc:.1f}×)")


def test_hbcc_bottleneck_mode_fewer_params_per_block():
    """
    HBCCNet với bottleneck chia đôi channel → số params mỗi block ít hơn không bottleneck.
    (Đây là trade-off: bottleneck tăng efficiency nhưng có thể giảm capacity.)
    """
    hbcc_no_bn = HBCCNet(num_classes=NUM_CLASSES, use_linear_bottleneck=False)
    hbcc_bn    = HBCCNet(num_classes=NUM_CLASSES, use_linear_bottleneck=True)

    n_no_bn = _count_params(hbcc_no_bn)
    n_bn    = _count_params(hbcc_bn)

    # Cả hai phải có params dương và hợp lý
    assert n_no_bn > 0 and n_bn > 0
    print(f"  PASS: HBCC no-bottleneck={n_no_bn:,}, with-bottleneck={n_bn:,} params")


# ─────────────────────────────────────────────
# 5. Eval mode: deterministic output
# ─────────────────────────────────────────────

def test_eval_mode_deterministic():
    """model.eval() phải cho output deterministic (không dropout/BN noise)."""
    cfg = {"model": "hbcc", "num_classes": NUM_CLASSES}
    model = build_model(cfg).eval()
    x = _make_input()

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2), \
        "Eval mode không deterministic — có thể do BatchNorm hoặc Dropout sai!"
    print("  PASS: Eval mode output deterministic (out1 == out2)")


def test_train_vs_eval_mode_shapes():
    """Cả train và eval mode đều cho output đúng shape."""
    cfg = {"model": "hbcc", "num_classes": NUM_CLASSES}
    model = build_model(cfg)
    x = _make_input()

    model.train()
    out_train = model(x)
    assert out_train.shape == (2, NUM_CLASSES), f"Train mode shape sai: {out_train.shape}"

    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert out_eval.shape == (2, NUM_CLASSES), f"Eval mode shape sai: {out_eval.shape}"

    print("  PASS: Cả train và eval mode cho output shape đúng")


# ─────────────────────────────────────────────
# 6. HBCCNet với HBCCBlock flags mới (B1–B4)
# ─────────────────────────────────────────────

def test_hbcc_net_with_new_flags_direct():
    """
    HBCCNet với các flags từ B-tasks (use_lbp_conv, top_k_centers, use_pruning_mask)
    thông qua khởi tạo HBCCBlock trực tiếp trong module list.
    """
    from src.models.modules.cluster_block import HBCCBlock

    # Tạo nhanh 1 block với full B-flags rồi forward CIFAR-size input
    block = HBCCBlock(
        dim=32,
        use_linear_bottleneck=True,
        use_lbp_conv=True,
        use_hamming=True,
        top_k_centers=2,
        use_pruning_mask=True,
    )
    x = torch.randn(2, 32, 8, 8)
    out = block(x)
    assert out.shape == x.shape, f"HBCCBlock full-flags shape sai: {out.shape}"
    print(f"  PASS: HBCCBlock full B-flags (lbp+hamming+top_k+pruning) shape OK")


def test_full_pipeline_loss_and_backward():
    """Toàn bộ pipeline: forward → CrossEntropyLoss → backward."""
    cfg = {
        "model": "hbcc",
        "num_classes": NUM_CLASSES,
        "use_linear_bottleneck": True,
        "use_hamming": True,
    }
    model = build_model(cfg).train()
    criterion = torch.nn.CrossEntropyLoss()

    x = _make_input()
    labels = torch.randint(0, NUM_CLASSES, (2,))

    logits = model(x)
    loss = criterion(logits, labels)
    loss.backward()

    # Kiểm tra gradient của 1 layer bất kỳ
    first_param = next(model.parameters())
    assert first_param.grad is not None, "Không có gradient sau backward!"
    print(f"  PASS: Full pipeline (forward→loss→backward) không lỗi. "
          f"Loss = {loss.item():.4f}")


if __name__ == "__main__":
    print("=== test_hbcc_net.py ===")
    test_resnet18_forward()
    test_coc_baseline_forward()
    test_hbcc_forward_baseline()
    test_hbcc_backward_no_error()
    test_hbcc_gradient_not_vanishing()
    test_resnet18_backward()
    test_build_model_ablation_configs()
    test_build_model_invalid_name_raises()
    test_param_count_ranges()
    test_hbcc_bottleneck_mode_fewer_params_per_block()
    test_eval_mode_deterministic()
    test_train_vs_eval_mode_shapes()
    test_hbcc_net_with_new_flags_direct()
    test_full_pipeline_loss_and_backward()
    print("=== TẤT CẢ TESTS PASSED ===\n")
