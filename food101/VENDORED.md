# Vendored Food-101 pipeline

This directory is an exact copy of
https://github.com/banlanhat69/Lightweight-Context-Cluster at commit
`bedafe443c7a7bbb569edd34354a831849391661` ("change to 200 epoch"), the code that
produced the Food-101 results (HBCC-2.5M and ResNet-18, 224x224).

It is kept separate from the top-level `lightweight_hbcc` package on purpose: the two
codebases have diverged (weight-decay parameter groups, MLP dropout placement, cluster
assignment options, config inheritance), so the Food-101 numbers are only reproducible
with this exact code.

## Local additions (not in the upstream commit)

| File | Change | Affects training? |
|---|---|---|
| `lightweight_hbcc/metrics.py` | Appended `count_params`, `model_size_mb`, `flop_count` | No |
| `lightweight_hbcc/profiler.py` | New file, copied from the top-level `lightweight_hbcc/profiler.py` | No |
| `tools/benchmark.py` | New file, copied from the top-level `tools/benchmark.py` | No |

Everything else is byte-identical to the upstream commit.

## Usage

Run all commands from this directory (`cd food101`), e.g.

    python tools/run_food101_experiments.py --models hbcc resnet18 --epochs 200 ...
    python tools/benchmark.py --config configs/food101/hbcc_best100.yaml --output results/benchmark_food101
    python tools/benchmark.py --torchvision mobilenet_v2 --num-classes 101 --image-size 224
