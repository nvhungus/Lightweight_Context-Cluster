from __future__ import annotations

import torch

from lightweight_hbcc.config import load_config
from lightweight_hbcc.models import build_model
from lightweight_hbcc.models.layers import make_local_branch, sign_ste


def test_hbcc_smoke_forward() -> None:
    cfg = load_config("configs/smoke.yaml")
    model = build_model(cfg)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_all_reference_models_forward() -> None:
    for path in [
        "configs/baselines/resnet18_cifar.yaml",
        "configs/baselines/mobilenet_v2_cifar.yaml",
        "configs/baselines/shufflenet_v2_x1_0_cifar.yaml",
        "configs/hbcc_latency_tiny.yaml",
        "configs/hbcc_current_reference.yaml",
        "configs/coc_cifar_baseline.yaml",
        "configs/hbcc_accuracy_small.yaml",
        "configs/hbcc_accuracy_medium.yaml",
        "configs/ablations/hbcc_accuracy_small_pruning_mask.yaml",
    ]:
        cfg = load_config(path)
        model = build_model(cfg).eval()
        with torch.no_grad():
            y = model(torch.randn(1, 3, 32, 32))
        assert y.shape == (1, 10), path


ABLATION_CONFIGS = [
    "configs/ablations/hbcc_accuracy_medium_allcluster.yaml",
    "configs/ablations/hbcc_medium_s1_dwconv.yaml",
    "configs/ablations/hbcc_medium_s1_conv3x3.yaml",
    "configs/ablations/hbcc_medium_noshuffle.yaml",
    "configs/ablations/hbcc_medium_nofold.yaml",
    "configs/ablations/hbcc_medium_s4_p2.yaml",
    "configs/ablations/hbcc_medium_r4.yaml",
    "configs/ablations/hbcc_medium_r1.yaml",
    "configs/ablations/hbcc_medium_stem2.yaml",
]


def test_ablation_configs_forward_with_200_classes() -> None:
    for path in ABLATION_CONFIGS:
        cfg = load_config(path)
        cfg["model"]["num_classes"] = 200
        model = build_model(cfg).eval()
        with torch.no_grad():
            y = model(torch.randn(2, 3, 32, 32))
        assert y.shape == (2, 200), path


def test_ablation_configs_change_exactly_one_model_field() -> None:
    base = load_config("configs/hbcc_accuracy_medium.yaml")
    for path in ABLATION_CONFIGS:
        cfg = load_config(path)
        changed = [k for k in base["model"] if cfg["model"].get(k) != base["model"][k]]
        allowed = 4 if path.endswith("allcluster.yaml") else 1  # rho=0 turns off modes/branches/ratios/shuffle together
        assert 1 <= len(changed) <= allowed, (path, changed)
        assert cfg["data"] == base["data"] and cfg["train"] == base["train"], path


def test_conv3x3_local_branch_is_learned_full_conv() -> None:
    branch = make_local_branch("conv3x3", 8, 8)
    conv = branch[0][0]
    assert isinstance(conv, torch.nn.Conv2d)
    assert conv.kernel_size == (3, 3) and conv.groups == 1
    assert all(p.requires_grad for p in branch.parameters())
    assert branch(torch.randn(1, 8, 16, 16)).shape == (1, 8, 16, 16)


def test_sign_ste_backward() -> None:
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
    y = sign_ste(x).sum()
    y.backward()
    assert torch.equal(sign_ste(x).detach(), torch.tensor([-1.0, -1.0, 1.0, 1.0]))
    assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 1.0, 0.0]))
