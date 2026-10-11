import torch

from src.train import add_instance_noise, instance_noise_std


def test_noise_std_anneals_linearly_to_zero():
    assert instance_noise_std(1, 100, 0.2) == 0.2
    assert abs(instance_noise_std(51, 100, 0.2) - 0.1) < 1e-9
    assert instance_noise_std(101, 100, 0.2) == 0.0
    assert instance_noise_std(1, 100, 0.0) == 0.0


def test_add_instance_noise():
    x = torch.zeros(64, 3, 64, 64)
    assert add_instance_noise(x, 0.0) is x
    torch.manual_seed(0)
    y = add_instance_noise(x, 0.1)
    assert abs(y.std().item() - 0.1) < 0.005
