import os

import torch

from src.utils import diff_configs, get_fixed_noise, load_config


def test_experiments_change_exactly_one_key():
    base = load_config("configs/base.yaml")
    assert diff_configs(load_config("configs/exp_hinge.yaml"), base) == {"loss": ("bce_nonsat", "hinge")}
    assert diff_configs(load_config("configs/exp_sn.yaml"), base) == {"spectral_norm_d": (False, True)}


def test_inherit_resolves_relative_to_yaml_not_cwd(tmp_path):
    cfg_path = os.path.abspath("configs/exp_hinge.yaml")
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert load_config(cfg_path)["seed"] == 42
    finally:
        os.chdir(old)


def test_fixed_noise_matches_contract(tmp_path):
    torch.manual_seed(1234)
    expected = torch.randn(16, 100, 1, 1)
    z = get_fixed_noise(tmp_path / "fixed_noise.pt")
    assert torch.equal(z, expected)
    assert torch.equal(get_fixed_noise(tmp_path / "fixed_noise.pt"), expected)  # recarga desde disco
