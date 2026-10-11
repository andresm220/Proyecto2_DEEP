import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from src.gallery import regenerate


def gallery(*args, check=True):
    return subprocess.run([sys.executable, "-m", "src.gallery", *args], capture_output=True, text=True, check=check)


@pytest.fixture(scope="module")
def built_gallery(tiny_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("galeria")
    w = str(tiny_run / "checkpoints" / "last.pt")
    gallery("--candidates", "--weights", w, "--n", "30", "--out", str(out))
    gallery("--select", "--weights", w, "--out", str(out))
    return out, w


def test_selection_has_ten_distinct_characters(built_gallery):
    out, _ = built_gallery
    meta = json.loads((out / "seeds.json").read_text(encoding="utf-8"))
    chars = meta["characters"]
    assert len(chars) == 10
    assert len({c["seed"] for c in chars}) == 10 and len({c["cluster"] for c in chars}) == 10
    assert "10 de 30" in meta["tasa_seleccion"]
    for c in chars:
        assert Image.open(out / c["file"]).size == (64, 64)
        assert Image.open(out / c["file_256"]).size == (256, 256)
    assert (out / "z_vectors.pt").exists() and (out / "galeria.png").exists()
    assert len(list((out / "candidatos").glob("cluster_*.png"))) == 10


def test_regenerate_matches_exactly(built_gallery):
    out, w = built_gallery
    r = regenerate(w, out / "seeds.json")
    assert r["ok"] and r["weights_ok"] and r["z_ok"] and sum(r["mismatches"].values()) == 0


def test_regenerate_detects_altered_png(built_gallery, tmp_path):
    out, w = built_gallery
    import shutil
    copy = tmp_path / "g"
    shutil.copytree(out, copy)
    meta = json.loads((copy / "seeds.json").read_text(encoding="utf-8"))
    p = copy / meta["characters"][3]["file"]
    a = np.array(Image.open(p))
    a[0, 0] = 255 - a[0, 0]
    Image.fromarray(a).save(p)
    r = regenerate(w, copy / "seeds.json")
    assert not r["ok"] and r["mismatches"][meta["characters"][3]["file"]] == 1
    assert gallery("--regenerate", "--weights", w, "--seeds", str(copy / "seeds.json"), check=False).returncode == 1


def test_regenerate_detects_wrong_weights(built_gallery, tmp_path):
    out, w = built_gallery
    import torch
    ck = torch.load(w, weights_only=False)
    ck["G"] = {k: v + 0.01 if v.is_floating_point() else v for k, v in ck["G"].items()}
    other = tmp_path / "otro.pt"
    torch.save(ck, other)
    r = regenerate(str(other), out / "seeds.json")
    assert not r["ok"] and not r["weights_ok"]


def test_select_keeps_names(built_gallery):
    out, w = built_gallery
    meta = json.loads((out / "seeds.json").read_text(encoding="utf-8"))
    meta["characters"][0]["nombre"] = "Ixbalam"
    meta["characters"][0]["clase"] = "guerrera"
    (out / "seeds.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    gallery("--select", "--weights", w, "--out", str(out))
    meta2 = json.loads((out / "seeds.json").read_text(encoding="utf-8"))
    assert meta2["characters"][0]["nombre"] == "Ixbalam" and meta2["characters"][0]["clase"] == "guerrera"


def test_too_few_candidates(tiny_run, tmp_path):
    r = gallery("--candidates", "--weights", str(tiny_run / "checkpoints" / "last.pt"), "--n", "5",
                "--out", str(tmp_path), check=False)
    assert r.returncode != 0 and "k=10" in r.stderr


def test_gallery_neighbors(built_gallery, tmp_path):
    out, w = built_gallery
    subprocess.run([sys.executable, "-m", "src.evaluate", "--gallery", str(out / "seeds.json"), "--weights", w,
                    "--out-root", str(tmp_path)], check=True, capture_output=True)
    rep = json.loads((out / "vecinos.json").read_text(encoding="utf-8"))
    assert len(rep) == 10
    first = next(iter(rep.values()))
    assert len(first["coseno"]) == 3 and first["coseno"][0]["vecino"].startswith("personaje ")
    assert (out / "vecinos.png").exists() and (out / "vecinos_pixeles.png").exists()


def test_export_g_is_small_and_generates_identical_images(tiny_run, tmp_path):
    import torch
    from src.evaluate import generate, load_generator
    full = tiny_run / "checkpoints" / "last.pt"
    gallery("--export-g", "--weights", str(full), "--out", str(tmp_path))
    small = tmp_path / "G_final.pt"
    assert small.exists() and small.stat().st_size < full.stat().st_size / 3
    ck = torch.load(small, weights_only=False)
    assert set(ck) >= {"G", "config", "epoch"} and "D" not in ck and "optG" not in ck
    a, _ = generate(load_generator(full)[0], [0, 5, 9])
    b, _ = generate(load_generator(small)[0], [0, 5, 9])
    assert torch.equal(a, b)


def test_regenerate_uses_saved_z_and_tolerates_rng_drift(built_gallery, tmp_path):
    # torch.randn con la misma seed puede diferir ~1e-8 entre versiones/CPU: z_vectors.pt es la fuente de verdad.
    import shutil
    import torch
    copy = tmp_path / "g"
    shutil.copytree(out := built_gallery[0], copy)
    z = torch.load(copy / "z_vectors.pt")
    torch.save(z + 1e-8, copy / "z_vectors.pt")
    r = regenerate(built_gallery[1], copy / "seeds.json")
    assert r["z_ok"]
    # y un z claramente distinto se detecta
    torch.save(z + 0.5, copy / "z_vectors.pt")
    assert not regenerate(built_gallery[1], copy / "seeds.json")["z_ok"]


def test_gallery_render_is_float64():
    import torch
    from src.gallery import render
    from src.models import Generator
    G = Generator().eval()
    z = torch.randn(2, 100, 1, 1)
    out = render(G, z)
    assert out.dtype == torch.float64 and out.shape == (2, 3, 64, 64)
    assert next(G.parameters()).dtype == torch.float32  # no modifica el modelo original
