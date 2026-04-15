"""
hamming_sim.py — Tính độ tương đồng bằng Khoảng cách Hamming (XNOR-Bitcount)

Thay thế Cosine Similarity bằng phép toán nhị phân siêu nhẹ:
    Sim(x_b, c_b) = D - 2 * bitcount(XNOR(x_b, c_b))

Trên phần cứng thực: dùng cổng XNOR + popcount (0 FLOPs, chỉ BOPs).
Trên PyTorch/GPU: giả lập bằng phép nhân ma trận nhị phân {-1,+1}
    vì x_b · c_b = D - 2 * HD(x_b, c_b)
"""
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Hamming Similarity (soft — training)
# ─────────────────────────────────────────────

def hamming_similarity(
    x: torch.Tensor,
    centers: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Tính độ tương đồng Hamming giữa tập điểm và tập tâm cụm.
    Giả lập bằng dot product của vector nhị phân {-1, +1}.

    x_b · c_b = D - 2 * HD  →  HD nhỏ ⟺ dot product lớn ⟺ giống nhau

    Args:
        x:         [B, N, D]  — điểm (đã binarize hoặc real-valued)
        centers:   [B, M, D]  — tâm cụm (đã binarize hoặc real-valued)
        normalize: nếu True, chia kết quả cho D để về [-1, 1]

    Returns:
        sim: [B, M, N]  — ma trận tương đồng
    """
    # sim[b, m, n] = dot(centers[b,m,:], x[b,n,:])
    sim = torch.matmul(centers, x.transpose(-2, -1))  # [B, M, N]
    if normalize:
        D = x.shape[-1]
        sim = sim / D
    return sim


# ─────────────────────────────────────────────
# 2. Pairwise Hamming Distance (inference / analysis)
# ─────────────────────────────────────────────

def pairwise_hamming_distance(
    x: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    """
    Tính Hamming Distance thực sự (integer count) từ vector nhị phân {-1, +1}.

    HD(x, c) = (D - x·c) / 2

    Args:
        x:       [B, N, D]  binary {-1,+1}
        centers: [B, M, D]  binary {-1,+1}

    Returns:
        hd: [B, M, N]  Hamming Distance (số bit khác nhau)
    """
    D = x.shape[-1]
    dot = torch.matmul(centers, x.transpose(-2, -1))  # [B, M, N]
    hd  = (D - dot) / 2.0
    return hd


# ─────────────────────────────────────────────
# 3. Weighted Hamming Similarity (dùng trong Cluster block)
# ─────────────────────────────────────────────

def weighted_hamming_sim(
    x: torch.Tensor,
    centers: torch.Tensor,
    alpha: torch.Tensor = None,
    beta: torch.Tensor  = None,
) -> torch.Tensor:
    """
    Phiên bản có scale/shift learnable (tương đương sim_alpha, sim_beta trong CoC gốc).
    Sau đó đưa qua Sigmoid để về (0, 1).

    sim_weighted = Sigmoid(beta + alpha * hamming_sim(x, c))

    Args:
        x:       [B, N, D]
        centers: [B, M, D]
        alpha:   scalar tensor (learnable) — scale
        beta:    scalar tensor (learnable) — shift

    Returns:
        sim: [B, M, N] ∈ (0, 1)
    """
    sim = hamming_similarity(x, centers, normalize=True)  # [B, M, N]

    if alpha is not None and beta is not None:
        sim = beta + alpha * sim

    return torch.sigmoid(sim)
