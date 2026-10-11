import torch

from src.models import build_models
from src.utils import load_config


def test_shapes_and_range():
    G, D = build_models(load_config("configs/base.yaml"))
    x = G(torch.randn(4, 100, 1, 1))
    assert x.shape == (4, 3, 64, 64) and x.abs().max() <= 1
    assert D(x).shape == (4,)


def test_spectral_norm_flag():
    _, D = build_models(load_config("configs/exp_sn.yaml"))
    convs = [m for m in D.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == 5 and all(hasattr(m, "weight_orig") for m in convs)
    _, D0 = build_models(load_config("configs/base.yaml"))
    assert not any(hasattr(m, "weight_orig") for m in D0.modules())


def test_eval_generation_independent_of_batch():
    torch.manual_seed(0)
    G, _ = build_models(load_config("configs/base.yaml"))
    G.train()
    G(torch.randn(64, 100, 1, 1))  # actualiza running stats de BN
    G.eval()
    z = torch.randn(16, 100, 1, 1)
    with torch.no_grad():
        batch = G(z)
        single = torch.cat([G(z[i:i + 1]) for i in range(16)])
    assert torch.allclose(batch, single, atol=1e-5)
