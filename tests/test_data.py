import torch

from src.data import TinyHero


def test_size_and_views(dataset):
    assert len(dataset) == 3648
    assert torch.bincount(dataset.views).tolist() == [912] * 4


def test_range_and_shape(dataset):
    x, v = dataset[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
    assert x.min() >= -1 and x.max() <= 1
    assert x.max() == 1.0  # fondo blanco compuesto -> hay píxeles en 1.0


def test_hflip_only_flips():
    ds = TinyHero("data", hflip=True)
    torch.manual_seed(0)
    outs = [ds[5][0] for _ in range(20)]
    orig = ds.images[5].float() / 127.5 - 1
    assert all(torch.equal(o, orig) or torch.equal(o, orig.flip(-1)) for o in outs)
    assert any(torch.equal(o, orig.flip(-1)) for o in outs)


def test_black_background():
    ds = TinyHero("data", hflip=False, background=(0, 0, 0))
    x, _ = ds[0]
    assert x[:, 0, 0].tolist() == [-1.0, -1.0, -1.0]  # esquina = fondo
