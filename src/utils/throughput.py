"""
throughput.py — Đo tốc độ suy luận (images/second)
"""
import time
import torch
import torch.nn as nn


def measure_throughput(
    model: nn.Module,
    input_size: tuple = (1, 3, 32, 32),
    num_runs: int = 1000,
    warmup_runs: int = 100,
    device: str = "cpu",
) -> dict:
    """
    Đo throughput (ảnh/giây) và latency (ms/ảnh).

    Args:
        model:       mô hình cần đo
        input_size:  kích thước input, ví dụ (1, 3, 32, 32)
        num_runs:    số lần chạy để đo
        warmup_runs: số lần chạy khởi động (không tính thời gian)
        device:      'cpu' hoặc 'cuda'

    Returns:
        dict:
            throughput_img_per_sec: float
            latency_ms:             float (ms/ảnh)
            device:                 str
    """
    model = model.to(device)
    model.eval()
    dummy = torch.zeros(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy)

    # Sync GPU nếu dùng CUDA
    if device == "cuda":
        torch.cuda.synchronize()

    # Đo thời gian
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
    end = time.perf_counter()

    total_time = end - start
    batch_size = input_size[0]
    total_images = num_runs * batch_size

    throughput = total_images / total_time          # ảnh/giây
    latency_ms  = (total_time / num_runs) * 1000    # ms/batch

    result = {
        "throughput_img_per_sec": round(throughput, 2),
        "latency_ms":             round(latency_ms, 4),
        "device":                 device,
        "num_runs":               num_runs,
        "batch_size":             batch_size,
    }

    print(f"  Throughput : {throughput:.1f} images/sec  (device={device})")
    print(f"  Latency    : {latency_ms:.2f} ms/batch")

    return result
