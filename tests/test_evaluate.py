import json
import subprocess
import sys

import torch

from src.evaluate import (FeatureExtractor, describe_train_index, diversity, generate, load_generator,
                          nearest_neighbors, novelty, pixel_neighbors, z_from_seed)


def test_z_from_seed_matches_gallery_contract():
    assert torch.equal(z_from_seed(7), torch.randn(1, 100, 1, 1, generator=torch.Generator().manual_seed(7)))


def test_generate_is_deterministic_and_batch_independent(tiny_run):
    G, _ = load_generator(tiny_run / "checkpoints" / "last.pt")
    a, za = generate(G, [3, 1, 4])
    b, _ = generate(G, [3, 1, 4], batch_size=3)
    assert a.shape == (3, 3, 64, 64) and torch.equal(za[1], z_from_seed(1)[0])
    assert torch.equal(a, generate(G, [3, 1, 4])[0])
    assert torch.allclose(a, b, atol=1e-5)


def test_diversity_bounds():
    same = torch.ones(5, 512)
    assert abs(diversity(same)) < 1e-6
    assert abs(diversity(torch.eye(4)) - 1.0) < 1e-6


def test_nearest_neighbor_of_train_sample_is_itself():
    ref = torch.randn(50, 512)
    idx, sim = nearest_neighbors(ref[[7, 20]], ref, k=3)
    assert idx[:, 0].tolist() == [7, 20] and torch.allclose(sim[:, 0], torch.ones(2), atol=1e-5)
    assert novelty(ref[:10], ref) < 1e-5


def test_pixel_neighbors_finds_exact_copy():
    ref = torch.rand(30, 3, 64, 64)
    idx, dist = pixel_neighbors(ref[[12]], ref, k=2)
    assert idx[0, 0] == 12 and dist[0, 0] < 1e-4


def test_describe_train_index(dataset):
    # El índice 0 es el primer archivo de la carpeta 0 (espalda): personaje 0.
    assert describe_train_index(dataset, 0) == "personaje 0 (espalda)"
    assert describe_train_index(dataset, 912 * 2 + 5) == "personaje 5 (frente)"


def test_feature_extractor_shape():
    f = FeatureExtractor()(torch.rand(2, 3, 64, 64) * 2 - 1)
    assert f.shape == (2, 512)


def test_cli_run_and_table(tiny_run):
    root = tiny_run.parent.parent
    subprocess.run([sys.executable, "-m", "src.evaluate", "--run", str(tiny_run), "--n", "20"],
                   check=True, capture_output=True)
    m = json.loads((tiny_run / "metrics.json").read_text())
    assert {"diversity", "novelty", "train_diversity", "n_samples"} <= m.keys() and m["n_samples"] == 20
    assert (tiny_run / "vecinos_muestra.png").exists()
    subprocess.run([sys.executable, "-m", "src.evaluate", "--out-root", str(root), "--table", "base"],
                   check=True, capture_output=True)
    lines = (root / "comparacion.csv").read_text().splitlines()
    assert lines[0].startswith("run,") and lines[1].startswith("base,")


def test_train_set_uses_same_background_as_model():
    # Si el modelo se entrenó con fondo negro, novedad y vecinos deben compararse contra reales con fondo negro.
    from src.evaluate import train_dataset_for
    ds = train_dataset_for({"background": "black"}, "data")
    assert ds.images[0, :, 0, 0].tolist() == [0, 0, 0]
    assert train_dataset_for({}, "data").images[0, :, 0, 0].tolist() == [255, 255, 255]
