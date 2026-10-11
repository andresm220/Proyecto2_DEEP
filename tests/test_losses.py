import math

import pytest
import torch

from src.losses import get_loss


def test_bce_values():
    d, g = get_loss("bce_nonsat")
    zero = torch.zeros(8)
    assert math.isclose(d(zero, zero).item(), 2 * math.log(2), rel_tol=1e-6)
    assert math.isclose(g(zero).item(), math.log(2), rel_tol=1e-6)


def test_hinge_values_and_margin():
    d, g = get_loss("hinge")
    assert d(torch.full((8,), 2.0), torch.full((8,), -2.0)).item() == 0.0  # fuera del margen -> 0
    assert d(torch.zeros(8), torch.zeros(8)).item() == 2.0
    assert g(torch.full((8,), 3.0)).item() == -3.0


def test_unknown_loss():
    with pytest.raises(ValueError):
        get_loss("wgan")
