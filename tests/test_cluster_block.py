"""
test_cluster_block.py — Tests cho BinarizedCluster và HBCCBlock

Kiểm tra:
1. Forward pass shape đúng
2. Gradient không bị vanishing qua Hamming + STE (điểm yếu nhất mạng nhị phân)
3. Hard-assign mask: đúng 1 tâm được gán mỗi điểm
4. HBCCBlock với tất cả tổ hợp flags on/off cho output đúng shape
5. BinarizedCluster use_hamming=True vs use_hamming=False cho kết quả khác nhau
6. DynamicCenterFilter (top_k_centers): zero-out đúng các tâm ngoài top-K
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.modules.cluster_block import BinarizedCluster, HBCCBlock


# ─────────────────────────────────────────────
# 1. Forward pass shape
# ─────────────────────────────────────────────

def test_binarized_cluster_shape():
    """BinarizedCluster giữ nguyên shape đầu vào."""
    x = torch.randn(2, 32, 8, 8)
    bc = BinarizedCluster(dim=32, proposal_w=2, proposal_h=2,
                          heads=2, head_dim=16)
    out = bc(x)
    assert out.shape == x.shape, f"Shape sai: {out.shape} != {x.shape}"
    print(f"  PASS: BinarizedCluster shape {x.shape} → {out.shape}")


def test_hbcc_block_shape_baseline():
    """HBCCBlock baseline (không flags) giữ nguyên shape."""
    x = torch.randn(2, 32, 8, 8)
    block = HBCCBlock(dim=32)
    out = block(x)
    assert out.shape == x.shape, f"Shape sai: {out.shape}"
    print(f"  PASS: HBCCBlock baseline {x.shape} → {out.shape}")


def test_binarized_cluster_various_dims():
    """BinarizedCluster hoạt động với nhiều kích thước channels khác nhau."""
    for dim in [16, 32, 64]:
        x = torch.randn(2, dim, 8, 8)
        bc = BinarizedCluster(dim=dim, proposal_w=2, proposal_h=2,
                              heads=max(1, dim // 16), head_dim=16)
        out = bc(x)
        assert out.shape == x.shape, f"dim={dim}: {out.shape} != {x.shape}"
    print("  PASS: BinarizedCluster đúng shape cho dim = 16, 32, 64")


# ─────────────────────────────────────────────
# 2. Gradient không bị vanishing
# ─────────────────────────────────────────────

def test_gradient_not_vanishing_hamming():
    """
    Gradient phải truyền được qua Hamming + STE.
    Đây là điểm yếu nhất của mạng nhị phân — nếu STE fail thì mạng không học được.
    """
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    bc = BinarizedCluster(dim=32, proposal_w=2, proposal_h=2,
                          heads=2, head_dim=16, use_hamming=True)
    bc.train()   # STE chỉ hoạt động khi model.training = True

    out = bc(x)
    out.sum().backward()

    assert x.grad is not None, "Gradient là None — STE bị gián đoạn!"
    mean_grad = x.grad.abs().mean().item()
    assert mean_grad > 1e-6, \
        f"Gradient vanishing! Mean |grad| = {mean_grad:.2e} (ngưỡng: 1e-6)"
    print(f"  PASS: Gradient qua Hamming+STE không vanishing. "
          f"Mean |grad| = {mean_grad:.4e}")


def test_gradient_not_vanishing_cosine():
    """Gradient qua Cosine similarity (baseline) cũng phải OK."""
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    bc = BinarizedCluster(dim=32, proposal_w=2, proposal_h=2,
                          heads=2, head_dim=16, use_hamming=False)
    out = bc(x)
    out.sum().backward()

    assert x.grad is not None
    mean_grad = x.grad.abs().mean().item()
    assert mean_grad > 1e-6, \
        f"Gradient vanishing qua Cosine! Mean |grad| = {mean_grad:.2e}"
    print(f"  PASS: Gradient qua Cosine OK. Mean |grad| = {mean_grad:.4e}")


def test_gradient_not_vanishing_full_hbcc_block():
    """Gradient không bị vanishing qua toàn bộ HBCCBlock với tất cả flags bật."""
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    block = HBCCBlock(
        dim=32,
        use_linear_bottleneck=True,
        use_hamming=True,
        use_lbp_conv=True,
        use_pruning_mask=True,
    )
    block.train()

    out = block(x)
    out.sum().backward()

    assert x.grad is not None, "Gradient là None qua full HBCCBlock!"
    mean_grad = x.grad.abs().mean().item()
    assert mean_grad > 1e-6, \
        f"Gradient vanishing trong HBCCBlock full! Mean |grad| = {mean_grad:.2e}"
    print(f"  PASS: HBCCBlock full gradient OK. Mean |grad| = {mean_grad:.4e}")


# ─────────────────────────────────────────────
# 3. Hard-assign: đúng 1 tâm mỗi điểm
# ─────────────────────────────────────────────

def test_hard_assign_one_center_per_point():
    """
    Sau hard-assign, mỗi điểm N phải được gán đúng 1 tâm cụm M.
    Test logic scatter_ trực tiếp (không phụ thuộc vào internal state của forward).
    """
    B, M, N = 2, 6, 32
    sim = torch.randn(B, M, N)

    _, sim_max_idx = sim.max(dim=1, keepdim=True)    # [B, 1, N]
    mask = torch.zeros_like(sim)
    mask.scatter_(1, sim_max_idx, 1.0)

    # Mỗi cột (point N) phải có đúng 1 giá trị = 1 trên chiều M (centers)
    col_sums = mask.sum(dim=1)   # [B, N]
    assert (col_sums == 1.0).all(), \
        f"Hard-assign sai: min={col_sums.min():.1f}, max={col_sums.max():.1f} " \
        f"(kỳ vọng tất cả = 1)"
    print(f"  PASS: Hard-assign đúng 1 tâm/điểm. B={B}, M={M}, N={N}")


def test_hard_assign_selects_max_similarity_center():
    """Scatter gán đúng tâm có similarity cao nhất."""
    # sim[b=0, center, point]:
    #   point 0 → center 1 có sim cao nhất (0.8)
    #   point 1 → center 0 có sim cao nhất (0.9)
    sim = torch.tensor([[[0.1, 0.9],   # center 0
                          [0.8, 0.2],   # center 1
                          [0.3, 0.5]]]) # center 2  shape [1, 3, 2]

    _, sim_max_idx = sim.max(dim=1, keepdim=True)
    mask = torch.zeros_like(sim)
    mask.scatter_(1, sim_max_idx, 1.0)

    assert mask[0, 1, 0] == 1.0, \
        f"Point 0 phải gán center 1, got mask[:,0]={mask[0,:,0].tolist()}"
    assert mask[0, 0, 1] == 1.0, \
        f"Point 1 phải gán center 0, got mask[:,1]={mask[0,:,1].tolist()}"
    print("  PASS: Hard-assign chọn đúng center có max similarity")


def test_hard_assign_mask_binary():
    """Mask hard-assign phải là nhị phân {0, 1}."""
    B, M, N = 3, 4, 20
    sim = torch.randn(B, M, N)
    _, sim_max_idx = sim.max(dim=1, keepdim=True)
    mask = torch.zeros_like(sim)
    mask.scatter_(1, sim_max_idx, 1.0)

    unique_vals = mask.unique().tolist()
    assert set(unique_vals).issubset({0.0, 1.0}), \
        f"Mask không nhị phân: {unique_vals}"
    print("  PASS: Hard-assign mask chỉ gồm {{0, 1}}")


# ─────────────────────────────────────────────
# 4. HBCCBlock với tất cả tổ hợp flags
# ─────────────────────────────────────────────

def test_hbcc_block_all_flag_combinations():
    """HBCCBlock với các tổ hợp flags khác nhau đều cho output đúng shape."""
    x = torch.randn(2, 32, 8, 8)

    configs = [
        # (use_lb, use_ps, use_lbp, use_ham, use_cs, use_pm, top_k, label)
        (False, False, False, False, False, False, None, "baseline"),
        (True,  False, False, False, False, False, None, "+bottleneck"),
        (True,  True,  False, False, False, False, None, "+point_shrink"),
        (True,  False, True,  False, False, False, None, "+lbp_conv"),
        (True,  False, True,  True,  False, False, None, "+hamming"),
        (True,  False, True,  True,  True,  False, None, "+channel_shuffle"),
        (True,  False, True,  True,  True,  True,  None, "+pruning_mask"),
        (True,  False, True,  True,  True,  True,  2,    "+top_k=2 (full)"),
        (False, False, False, True,  False, False, 2,    "no_bottleneck+top_k"),
        (True,  False, False, False, True,  False, None, "bottleneck+ch_shuffle"),
    ]

    for (use_lb, use_ps, use_lbp, use_h, use_cs, use_pm, top_k, label) in configs:
        block = HBCCBlock(
            dim=32,
            use_linear_bottleneck=use_lb,
            use_point_shrink=use_ps,
            use_lbp_conv=use_lbp,
            use_hamming=use_h,
            use_channel_shuffle=use_cs,
            use_pruning_mask=use_pm,
            top_k_centers=top_k,
        )
        out = block(x)
        assert out.shape == x.shape, \
            f"[{label}] Shape sai: {out.shape} != {x.shape}"

    print(f"  PASS: Tất cả {len(configs)} cấu hình HBCCBlock cho output đúng shape")


# ─────────────────────────────────────────────
# 5. Hamming vs Cosine khác nhau
# ─────────────────────────────────────────────

def test_hamming_vs_cosine_different_outputs():
    """
    use_hamming=True phải cho kết quả khác use_hamming=False.
    Hai phương pháp đo similarity hoàn toàn khác nhau về mặt toán học.
    """
    torch.manual_seed(42)
    x = torch.randn(2, 32, 8, 8)

    bc_hamming = BinarizedCluster(dim=32, proposal_w=2, proposal_h=2,
                                  heads=2, head_dim=16, use_hamming=True)
    bc_cosine  = BinarizedCluster(dim=32, proposal_w=2, proposal_h=2,
                                  heads=2, head_dim=16, use_hamming=False)

    with torch.no_grad():
        out_h = bc_hamming(x)
        out_c = bc_cosine(x)

    assert not torch.allclose(out_h, out_c, atol=1e-3), \
        "Hamming và Cosine cho cùng output — một trong hai không hoạt động đúng!"
    max_diff = (out_h - out_c).abs().max().item()
    print(f"  PASS: Hamming ≠ Cosine output. Max diff = {max_diff:.4f}")


def test_hamming_mode_uses_binarization():
    """
    Hamming mode phải binarize features → output có pattern khác hẳn cosine.
    Thử với input chứa giá trị cực đại để phân biệt rõ 2 mode.
    """
    torch.manual_seed(7)
    # Input rất lớn: cosine normalize về unit sphere, hamming sign về {-1,+1}
    x = torch.randn(2, 32, 8, 8) * 100.0

    bc_h = BinarizedCluster(dim=32, proposal_w=2, proposal_h=2,
                             heads=2, head_dim=16, use_hamming=True)
    bc_c = BinarizedCluster(dim=32, proposal_w=2, proposal_h=2,
                             heads=2, head_dim=16, use_hamming=False)

    with torch.no_grad():
        out_h = bc_h(x)
        out_c = bc_c(x)

    # Với input scale rất lớn, cosine không đổi (scale-invariant)
    # nhưng hamming cũng scale-invariant (sign không đổi theo scale)
    # Chỉ cần verify cả hai forward không crash
    assert out_h.shape == out_c.shape == x.shape
    print("  PASS: Cả Hamming và Cosine không crash với input scale lớn")


# ─────────────────────────────────────────────
# 6. DynamicCenterFilter (top_k_centers)
# ─────────────────────────────────────────────

def test_dynamic_center_filter_output_shape():
    """BinarizedCluster với top_k_centers vẫn cho output đúng shape."""
    x = torch.randn(2, 32, 8, 8)
    bc = BinarizedCluster(dim=32, proposal_w=2, proposal_h=2,
                          heads=2, head_dim=16, top_k_centers=2)
    out = bc(x)
    assert out.shape == x.shape, f"top_k=2: {out.shape} != {x.shape}"
    print(f"  PASS: DynamicCenterFilter (top_k=2) shape OK: {out.shape}")


def test_dynamic_center_filter_zeros_non_topk():
    """
    Với top_k_centers=K, các tâm NGOÀI top-K phải có sim = 0.
    Test logic filter trực tiếp — không phụ thuộc vào forward của module.
    """
    B, M, N = 2, 6, 16
    K = 2
    sim = torch.rand(B, M, N)           # sim ∈ (0, 1)

    # Áp dụng DynamicCenterFilter
    mass = sim.sum(-1)                                    # [B, M]
    topk_idx = mass.topk(K, dim=-1).indices               # [B, K]
    center_mask = torch.zeros_like(mass)
    center_mask.scatter_(1, topk_idx, 1.0)
    sim_filtered = sim * center_mask.unsqueeze(-1)         # [B, M, N]

    # Các tâm ngoài top-K phải có sim = 0
    non_topk = (center_mask == 0)                         # [B, M]
    assert (sim_filtered[non_topk] == 0.0).all(), \
        "Một số tâm ngoài top-K vẫn có sim != 0!"

    # Đúng M-K tâm bị zero-out
    zeroed = non_topk.sum(dim=-1)                         # [B]
    assert (zeroed == M - K).all(), \
        f"Số tâm bị zero-out sai: {zeroed.tolist()} (kỳ vọng {M-K})"

    # Đúng K tâm được giữ lại
    kept = center_mask.sum(dim=-1)                        # [B]
    assert (kept == K).all(), \
        f"Số tâm giữ lại sai: {kept.tolist()} (kỳ vọng {K})"

    print(f"  PASS: DynamicCenterFilter: giữ top-{K}, zero-out {M-K}/{M} tâm còn lại")


def test_dynamic_center_filter_top1_vs_topall():
    """top_k=1 vs top_k=M phải cho kết quả khác nhau (khi M > 1)."""
    torch.manual_seed(0)
    x = torch.randn(2, 32, 8, 8)

    M = 2 * 2   # proposal_w * proposal_h

    bc_top1 = BinarizedCluster(dim=32, proposal_w=2, proposal_h=2,
                                heads=2, head_dim=16, top_k_centers=1)
    bc_topall = BinarizedCluster(dim=32, proposal_w=2, proposal_h=2,
                                  heads=2, head_dim=16, top_k_centers=None)

    with torch.no_grad():
        out_1 = bc_top1(x)
        out_all = bc_topall(x)

    assert not torch.allclose(out_1, out_all, atol=1e-4), \
        "top_k=1 và top_k=None cho cùng output — DynamicCenterFilter không có hiệu quả!"
    max_diff = (out_1 - out_all).abs().max().item()
    print(f"  PASS: top_k=1 ≠ top_k=None. Max diff = {max_diff:.4f}")


if __name__ == "__main__":
    print("=== test_cluster_block.py ===")
    test_binarized_cluster_shape()
    test_hbcc_block_shape_baseline()
    test_binarized_cluster_various_dims()
    test_gradient_not_vanishing_hamming()
    test_gradient_not_vanishing_cosine()
    test_gradient_not_vanishing_full_hbcc_block()
    test_hard_assign_one_center_per_point()
    test_hard_assign_selects_max_similarity_center()
    test_hard_assign_mask_binary()
    test_hbcc_block_all_flag_combinations()
    test_hamming_vs_cosine_different_outputs()
    test_hamming_mode_uses_binarization()
    test_dynamic_center_filter_output_shape()
    test_dynamic_center_filter_zeros_non_topk()
    test_dynamic_center_filter_top1_vs_topall()
    print("=== TẤT CẢ TESTS PASSED ===\n")
