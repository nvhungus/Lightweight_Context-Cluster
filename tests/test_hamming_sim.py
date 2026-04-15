"""
test_hamming_sim.py — Unit test cho hamming_sim.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.modules.hamming_sim import (
    hamming_similarity, pairwise_hamming_distance, weighted_hamming_sim,
)


def test_identical_vectors_max_sim():
    v = torch.ones(1, 1, 16)
    c = torch.ones(1, 1, 16)
    sim = hamming_similarity(v, c, normalize=True)
    assert torch.allclose(sim, torch.ones_like(sim), atol=1e-5)
    print(f"  PASS: Identical vectors sim = {sim.item():.4f} (kỳ vọng 1.0)")


def test_opposite_vectors_min_sim():
    v =  torch.ones(1, 1, 16)
    c = -torch.ones(1, 1, 16)
    sim = hamming_similarity(v, c, normalize=True)
    assert torch.allclose(sim, -torch.ones_like(sim), atol=1e-5)
    print(f"  PASS: Opposite vectors sim = {sim.item():.4f} (kỳ vọng -1.0)")


def test_output_shape():
    B, M, N, D = 2, 5, 10, 32
    x = torch.randn(B, N, D).sign()
    c = torch.randn(B, M, D).sign()
    sim = hamming_similarity(x, c, normalize=True)
    assert sim.shape == (B, M, N)
    print(f"  PASS: Output shape đúng {sim.shape}")


def test_hamming_distance_identical():
    v = torch.ones(1, 1, 16)
    c = torch.ones(1, 1, 16)
    hd = pairwise_hamming_distance(v, c)
    assert torch.allclose(hd, torch.zeros_like(hd), atol=1e-5)
    print(f"  PASS: HD (identical) = {hd.item():.1f}")


def test_hamming_distance_opposite():
    D = 16
    v =  torch.ones(1, 1, D)
    c = -torch.ones(1, 1, D)
    hd = pairwise_hamming_distance(v, c)
    assert torch.allclose(hd, torch.tensor([[[float(D)]]]), atol=1e-5)
    print(f"  PASS: HD (opposite) = {hd.item():.1f} (kỳ vọng {D})")


def test_weighted_hamming_sim_range():
    x = torch.randn(2, 10, 32).sign().float()
    c = torch.randn(2,  5, 32).sign().float()
    sim = weighted_hamming_sim(x, c, torch.ones(1), torch.zeros(1))
    assert sim.shape == (2, 5, 10)
    assert sim.min() > 0.0 and sim.max() < 1.0
    print(f"  PASS: weighted_hamming_sim ∈ (0,1), shape={sim.shape}")


if __name__ == "__main__":
    print("=== test_hamming_sim.py ===")
    test_identical_vectors_max_sim()
    test_opposite_vectors_min_sim()
    test_output_shape()
    test_hamming_distance_identical()
    test_hamming_distance_opposite()
    test_weighted_hamming_sim_range()
    print("=== TẤT CẢ TESTS PASSED ===\n")
